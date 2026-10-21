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
from collections.abc import AsyncIterator

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.api.schemas import ChatRequest, ChatResponse, ChatUsage, CitationOut
from sutra.api.streaming import sse_frame
from sutra.config import Settings, get_settings
from sutra.db.engine import get_sessionmaker
from sutra.db.models import Message as MessageRow
from sutra.db.models import Session as SessionRow
from sutra.generation.answer import generate_answer
from sutra.generation.schema import Answer
from sutra.generation.stream import AnswerToken, generate_answer_stream
from sutra.ingestion.embed import Embedder
from sutra.llm.client import LLMClient
from sutra.observability.context import request_context
from sutra.observability.tracing import current_trace_id, get_tracer
from sutra.prompts import PromptLoader
from sutra.retrieval.expand import expand_to_parents
from sutra.retrieval.search import RetrievedChunk, search

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
    trace_id: str | None = None,
) -> uuid.UUID:
    row = MessageRow(
        session_id=session_id,
        role=role,
        content=content,
        prompt_version=prompt_version,
        token_count=token_count,
        trace_id=trace_id,
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
    tags: list[str] | None = None,
) -> ChatResponse:
    """Answer one turn and record it.

    Raises `SessionNotFound` when a session id is given that this user cannot use.

    The root span is opened here rather than in the HTTP route, because every client — the
    eval runner today, MCP and voice later — goes through this function and not through HTTP
    (ARCHITECTURE §1). Instrumenting the route would leave every other caller untraced.
    `tags` is how the eval runner groups a run's traces (ARCHITECTURE §11).
    """
    tracer = get_tracer()
    with (
        tracer.request(
            name="chat",
            user_id=request.user_id,
            session_id=str(request.session_id) if request.session_id else None,
            input={"query": request.query},
            tags=tags,
            metadata={"mode": request.mode, "stream": request.stream},
        ) as root_span,
        request_context(
            trace_id=current_trace_id(),
            user_id=request.user_id,
            session_id=str(request.session_id) if request.session_id else None,
            mode=request.mode,
        ),
    ):
        response = await _run_chat_turn_inner(
            request,
            settings=settings,
            client=client,
            loader=loader,
            embedder=embedder,
        )
        root_span.update(
            output={"answer": response.answer, "refused": response.refused},
            metadata={
                "cited": len(response.citations),
                "dropped_citations": response.dropped_citations or None,
                "prompt_version": response.prompt_version,
                "index_version": response.index_version,
            },
        )
    return response


async def resolve_session(request: ChatRequest) -> uuid.UUID:
    """Resolve (or create) the session for this request, raising `SessionNotFound`.

    Exposed so the streaming route can do this *before* the response begins. Once the first
    SSE frame is written the status line is gone, and a bad session id has to be a 404 rather
    than a 200 whose body contains an apology.
    """
    factory = get_sessionmaker()
    async with factory() as db:
        session_id = await _resolve_session(
            db, session_id=request.session_id, user_id=request.user_id
        )
        await db.commit()
    return session_id


async def _retrieve_phase(
    request: ChatRequest,
    *,
    settings: Settings,
    client: LLMClient,
    embedder: Embedder | None,
    trace_id: str | None,
    session_id: uuid.UUID | None = None,
) -> tuple[uuid.UUID, list[RetrievedChunk]]:
    """Resolve the session, record the question, retrieve. One short transaction.

    Shared by the buffered and streaming paths so they cannot drift: the whole value of
    streaming the same prompt through the same schema is lost if the two retrieve differently.
    """
    factory = get_sessionmaker()
    async with factory() as db:
        if session_id is None:
            session_id = await _resolve_session(
                db, session_id=request.session_id, user_id=request.user_id
            )
        await _save_message(
            db,
            session_id=session_id,
            role="user",
            content=request.query,
            trace_id=trace_id,
        )
        chunks: list[RetrievedChunk] = await search(
            request.query, session=db, settings=settings, embedder=embedder, client=client
        )
        # Parent expansion reads the database, so it belongs in this transaction rather than
        # opening a second one after the model call.
        if settings.retrieval_expand_to_parent:
            chunks = await expand_to_parents(chunks, session=db)
        await db.commit()
    return session_id, chunks


async def _persist_phase(
    *, session_id: uuid.UUID, answer: Answer, trace_id: str | None
) -> uuid.UUID:
    """Record the answer and touch the session. One short transaction."""
    factory = get_sessionmaker()
    async with factory() as db:
        message_id = await _save_message(
            db,
            session_id=session_id,
            role="assistant",
            content=answer.answer,
            prompt_version=answer.prompt_version,
            token_count=answer.usage.total_tokens,
            trace_id=trace_id,
        )
        await db.execute(
            update(SessionRow).where(SessionRow.id == session_id).values(last_active_at=func.now())
        )
        await db.commit()
    return message_id


def _citations_payload(answer: Answer) -> list[CitationOut]:
    return [
        CitationOut(
            label=c.label,
            chunk_id=c.chunk_id,
            doc_slug=c.doc_slug,
            doc_title=c.doc_title,
            heading=c.heading,
        )
        for c in answer.citations
    ]


async def _run_chat_turn_inner(
    request: ChatRequest,
    *,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    embedder: Embedder | None = None,
) -> ChatResponse:
    """The turn itself, inside an open root span."""
    settings = settings or get_settings()
    # One client for the whole turn: retrieval's model-backed stages and generation share it.
    # Built here rather than defaulted inside each callee, so one request opens one connection
    # pool instead of one per stage that happens to need a model.
    client = client or LLMClient(settings=settings)
    # Stored on both messages so an answer in the database can be opened as a trace later.
    # The column has existed since the first migration and has been NULL until now.
    trace_id = current_trace_id()

    # --- phase 1: resolve the session, record the question, retrieve. Short transaction.
    session_id, chunks = await _retrieve_phase(
        request,
        settings=settings,
        client=client,
        embedder=embedder,
        trace_id=trace_id,
    )

    # --- phase 2: generate. No database connection held while this runs.
    answer: Answer = await generate_answer(
        request.query, chunks, client=client, loader=loader, settings=settings
    )

    # --- phase 3: record the answer. Short transaction.
    message_id = await _persist_phase(session_id=session_id, answer=answer, trace_id=trace_id)

    logger.info(
        "chat.turn",
        extra={
            "session_id": str(session_id),
            "refused": answer.refused,
            "chunks": len(chunks),
            "cited": len(answer.citations),
            "dropped": len(answer.dropped_citations),
            "cost_usd": answer.usage.cost_usd,
            "latency_ms": answer.latency_ms,
            "prompt_version": answer.prompt_version,
        },
    )

    return ChatResponse(
        answer=answer.answer,
        citations=_citations_payload(answer),
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
        trace_id=trace_id,
    )


async def run_chat_turn_stream(
    request: ChatRequest,
    *,
    session_id: uuid.UUID,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    embedder: Embedder | None = None,
    tags: list[str] | None = None,
) -> AsyncIterator[str]:
    """Answer one turn as SSE frames, recording it exactly as the buffered path does.

    `session_id` is resolved by the caller (`resolve_session`) rather than here, for two
    reasons. A bad session id has to become a 404 before the response starts, and nothing
    inside a generator can set a status once a frame has been written.

    # CONCEPT: The second reason is subtler and cost a bug. An async generator's frame runs
    # in whichever context calls `asend`, so a `contextvar` set before a `yield` and reset
    # after it must have both halves driven by the same task. Pulling the first frame in the
    # route handler and letting Starlette pull the rest put the set and the reset in two
    # different contexts, and `ContextVar.reset` raises `ValueError: Token was created in a
    # different Context`. Resolving the session up front removes the need to pre-pull, so one
    # task drives the whole generator and the trace context is balanced.
    """
    settings = settings or get_settings()
    client = client or LLMClient(settings=settings)
    tracer = get_tracer()

    with (
        tracer.request(
            name="chat",
            user_id=request.user_id,
            session_id=str(request.session_id) if request.session_id else None,
            input={"query": request.query},
            tags=tags,
            metadata={"mode": request.mode, "stream": True},
        ) as root_span,
        request_context(
            trace_id=current_trace_id(),
            user_id=request.user_id,
            session_id=str(request.session_id) if request.session_id else None,
            mode=request.mode,
        ),
    ):
        trace_id = current_trace_id()
        _, chunks = await _retrieve_phase(
            request,
            settings=settings,
            client=client,
            embedder=embedder,
            trace_id=trace_id,
            session_id=session_id,
        )

        # `meta` first, before any token, so a client has the trace id even if the answer
        # fails half way through — it is the identifier they would quote in a bug report.
        yield sse_frame(
            "meta",
            {
                "trace_id": trace_id,
                "session_id": str(session_id),
                "index_version": settings.index_version,
                "prompt_version": settings.prompt_version_answer,
                "model": settings.llm_strong,
                "retrieved_chunk_ids": [c.chunk_id for c in chunks],
            },
        )

        answer: Answer | None = None
        async for event in generate_answer_stream(
            request.query, chunks, client=client, loader=loader, settings=settings
        ):
            if isinstance(event, AnswerToken):
                yield sse_frame("token", {"text": event.text})
            else:
                answer = event.answer

        if answer is None:  # pragma: no cover - the generator always ends with one
            msg = "answer stream ended without a final event"
            raise RuntimeError(msg)

        message_id = await _persist_phase(session_id=session_id, answer=answer, trace_id=trace_id)

        # Citations come after the tokens because they are resolved from the completed
        # object: the model cites passage numbers, and mapping those to real chunks is only
        # possible once the JSON has closed.
        yield sse_frame(
            "citations",
            {
                "citations": [c.model_dump() for c in _citations_payload(answer)],
                "dropped_citations": answer.dropped_citations,
            },
        )
        yield sse_frame(
            "done",
            {
                "message_id": str(message_id),
                "confidence": answer.confidence,
                "refused": answer.refused,
                "model": answer.model,
                "prompt_version": answer.prompt_version,
                "latency_ms": answer.latency_ms,
                "usage": ChatUsage(
                    input_tokens=answer.usage.input_tokens,
                    output_tokens=answer.usage.output_tokens,
                    cached_tokens=answer.usage.cached_tokens,
                    cost_usd=answer.usage.cost_usd,
                ).model_dump(),
            },
        )

        root_span.update(
            output={"answer": answer.answer, "refused": answer.refused},
            metadata={
                "cited": len(answer.citations),
                "dropped_citations": answer.dropped_citations or None,
                "prompt_version": answer.prompt_version,
                "index_version": settings.index_version,
                "streamed": True,
            },
        )
        logger.info(
            "chat.turn",
            extra={
                "session_id": str(session_id),
                "streamed": True,
                "refused": answer.refused,
                "chunks": len(chunks),
                "cited": len(answer.citations),
                "dropped": len(answer.dropped_citations),
                "cost_usd": answer.usage.cost_usd,
                "latency_ms": answer.latency_ms,
                "prompt_version": answer.prompt_version,
            },
        )
