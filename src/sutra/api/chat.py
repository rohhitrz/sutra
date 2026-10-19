"""One chat turn, orchestrated: retrieve → generate → persist.

This is ARCHITECTURE §2's request lifecycle with most of it not yet built. Phase 2 does
retrieve → generate → save; guardrails, the semantic cache, memory and tracing slot in around
it in later phases, which is why the order here follows §2 rather than being convenient.

# CONCEPT: Database connections are not held across the model call. A pooled connection is a
# scarce resource and an LLM call takes seconds, so a handler that keeps a transaction open
# across generation caps concurrency at the pool size and holds a write transaction open for
# the duration. The work is split into short sessions either side instead.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.api.schemas import ChatRequest, ChatResponse, ChatUsage, CitationOut
from sutra.config import Settings, get_settings
from sutra.db.engine import get_sessionmaker
from sutra.db.models import Message as MessageRow
from sutra.db.models import Session as SessionRow
from sutra.generation.answer import generate_answer
from sutra.generation.schema import Answer
from sutra.ingestion.embed import Embedder
from sutra.llm.client import LLMClient
from sutra.prompts import PromptLoader
from sutra.retrieval.search import RetrievedChunk, vector_search

logger = logging.getLogger(__name__)


class SessionNotFound(LookupError):
    """No such session for this user.

    # CONCEPT: One error for "does not exist" and "belongs to somebody else". Distinguishing
    # them tells an attacker which session ids are real, which is a membership oracle — so the
    # ownership check and the existence check deliberately fail the same way.
    """


async def _resolve_session(
    session: AsyncSession, *, session_id: uuid.UUID | None, user_id: str
) -> uuid.UUID:
    """Continue the given conversation, or start one."""
    if session_id is None:
        row = SessionRow(user_id=user_id)
        session.add(row)
        await session.flush()
        return row.id

    owner = (
        await session.execute(select(SessionRow.user_id).where(SessionRow.id == session_id))
    ).scalar_one_or_none()
    if owner is None or owner != user_id:
        raise SessionNotFound(str(session_id))
    return session_id


async def _save_message(
    session: AsyncSession,
    *,
    session_id: uuid.UUID,
    role: str,
    content: str,
    prompt_version: str | None = None,
    token_count: int | None = None,
) -> uuid.UUID:
    row = MessageRow(
        session_id=session_id,
        role=role,
        content=content,
        prompt_version=prompt_version,
        token_count=token_count,
    )
    session.add(row)
    await session.flush()
    return row.id


async def run_chat_turn(
    request: ChatRequest,
    *,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    embedder: Embedder | None = None,
) -> ChatResponse:
    """Answer one turn and record it.

    Raises `SessionNotFound` when a session id is given that this user cannot use.
    """
    settings = settings or get_settings()
    factory = get_sessionmaker()

    # --- phase 1: resolve the session, record the question, retrieve. Short transaction.
    async with factory() as db:
        session_id = await _resolve_session(
            db, session_id=request.session_id, user_id=request.user_id
        )
        await _save_message(db, session_id=session_id, role="user", content=request.query)
        chunks: list[RetrievedChunk] = await vector_search(
            request.query, session=db, settings=settings, embedder=embedder
        )
        await db.commit()

    # --- phase 2: generate. No database connection held while this runs.
    answer: Answer = await generate_answer(
        request.query, chunks, client=client, loader=loader, settings=settings
    )

    # --- phase 3: record the answer. Short transaction.
    async with factory() as db:
        message_id = await _save_message(
            db,
            session_id=session_id,
            role="assistant",
            content=answer.answer,
            prompt_version=answer.prompt_version,
            token_count=answer.usage.total_tokens,
        )
        await db.execute(
            update(SessionRow).where(SessionRow.id == session_id).values(last_active_at=func.now())
        )
        await db.commit()

    logger.info(
        "chat.turn session=%s refused=%s chunks=%d cited=%d dropped=%d cost=%s",
        session_id,
        answer.refused,
        len(chunks),
        len(answer.citations),
        len(answer.dropped_citations),
        answer.usage.cost_usd,
    )

    return ChatResponse(
        answer=answer.answer,
        citations=[
            CitationOut(
                label=c.label,
                chunk_id=c.chunk_id,
                doc_slug=c.doc_slug,
                doc_title=c.doc_title,
                heading=c.heading,
            )
            for c in answer.citations
        ],
        confidence=answer.confidence,
        refused=answer.refused,
        session_id=session_id,
        message_id=message_id,
        prompt_version=answer.prompt_version,
        index_version=settings.index_version,
        model=answer.model,
        usage=ChatUsage(
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            cached_tokens=answer.usage.cached_tokens,
            cost_usd=answer.usage.cost_usd,
        ),
        latency_ms=answer.latency_ms,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        dropped_citations=answer.dropped_citations,
    )
