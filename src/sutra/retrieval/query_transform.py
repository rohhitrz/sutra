"""Query transforms — step 2 of the ARCHITECTURE §4 pipeline.

# CONCEPT: Retrieval matches a *query* against passages, but a user writes a *question*, and
# the two are not the same object. A question carries pronouns the corpus never uses, assumes
# context the index does not have, and often asks for several facts at once. Every one of
# those is a retrieval problem before it is a generation problem.
#
# Three transforms, each toggleable so the eval can price them separately:
#
#   rewrite      one standalone query, references resolved. Cheap, almost always safe.
#   multi-query  n queries, one per fact the answer needs, results fused by RRF. This is the
#                one aimed at multi-hop: a single vector cannot sit near two different
#                passages at once, so a question needing two facts is asking the retriever to
#                be in two places. Issuing two queries is the obvious fix and nothing cheaper
#                works.
#   HyDE         embed a *hypothetical answer* rather than the question. Questions and
#                passages are different registers — "who drowned in the flood?" looks unlike
#                "Ravenna Volant drowned in the Vessel" — and a fabricated answer sits in the
#                register of the thing being searched for. The fabrication is never shown to
#                anyone; only its embedding is used.
#
# HyDE's obvious objection is the right one: it invents text about a world it does not know.
# That is tolerable *only* because the output never reaches the answer or the index — it is a
# throwaway probe. The moment a HyDE draft could be cited, this would be inventing lore.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from sutra.config import Settings, get_settings
from sutra.llm.client import LLMClient, client_scope
from sutra.llm.types import Message
from sutra.observability.tracing import stage
from sutra.prompts import PromptLoader

logger = logging.getLogger(__name__)

MAX_TRANSFORM_TOKENS = 1500


class TransformedQueries(BaseModel):
    """What the rewriter returns."""

    queries: list[str] = Field(default_factory=list)


class HypotheticalAnswer(BaseModel):
    """What HyDE returns. Never shown to a user and never indexed — see the module docstring."""

    passage: str


async def transform_query(
    question: str,
    *,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Return the queries to retrieve with. Always includes the original question.

    **The original is never dropped.** A transform is a model call and can fail, drift, or
    paraphrase away the one rare noun that would have matched; keeping the question in the
    pool makes the whole stage strictly additive, so a bad rewrite can only cost ranking
    slots rather than the answer.
    """
    settings = settings or get_settings()
    if not (settings.query_rewrite_enabled or settings.hyde_enabled):
        return [question]

    # `client_scope` rather than `client or LLMClient(...)`: building a provider client here
    # and dropping it leaks its connection pool on every transformed query. `search()` passes
    # one in, so this is the fallback path for a direct caller or a test.
    loader = loader or PromptLoader(settings=settings)
    queries = [question]

    with stage("transform_query", kind="span", input={"question": question}) as span:
        async with client_scope(client, settings) as llm:
            if settings.query_rewrite_enabled:
                queries.extend(await _rewrite(question, llm, loader, settings))
            if settings.hyde_enabled:
                hypothetical = await _hyde(question, llm, loader, settings)
                if hypothetical:
                    queries.append(hypothetical)
        span.update(
            output={"queries": queries},
            metadata={
                "rewrite_enabled": settings.query_rewrite_enabled,
                "hyde_enabled": settings.hyde_enabled,
            },
        )

    # Dedupe case-insensitively while keeping order: a rewrite that returns the question
    # unchanged is the common case for a simple question, and retrieving it twice would
    # double its weight in the fusion for no reason.
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(query.strip())
    return unique


async def _rewrite(
    question: str, client: LLMClient, loader: PromptLoader, settings: Settings
) -> list[str]:
    prompt = loader.get("query_transform", settings.prompt_version_query_transform)
    system, user = prompt.render(
        question=question, max_queries=str(settings.multi_query_count)
    ).split("<<<USER>>>", 1)
    try:
        response = await client.structured(
            _messages(system, user),
            TransformedQueries,
            role="fast",
            max_output_tokens=MAX_TRANSFORM_TOKENS,
            temperature=prompt.meta.get("temperature"),
            stage="rewrite",
            prompt_version=prompt.qualified_version,
        )
    except Exception as exc:
        # Degrade to the original question. A failed rewrite must not fail the request.
        logger.warning("query rewrite failed: %s", exc)
        return []
    parsed = response.parsed
    if not isinstance(parsed, TransformedQueries):
        return []
    return [q.strip() for q in parsed.queries[: settings.multi_query_count] if q.strip()]


async def _hyde(
    question: str, client: LLMClient, loader: PromptLoader, settings: Settings
) -> str | None:
    prompt = loader.get("hyde", settings.prompt_version_hyde)
    system, user = prompt.render(question=question).split("<<<USER>>>", 1)
    try:
        response = await client.structured(
            _messages(system, user),
            HypotheticalAnswer,
            role="fast",
            max_output_tokens=MAX_TRANSFORM_TOKENS,
            temperature=prompt.meta.get("temperature"),
            stage="hyde",
            prompt_version=prompt.qualified_version,
        )
    except Exception as exc:
        logger.warning("hyde failed: %s", exc)
        return None
    parsed = response.parsed
    return parsed.passage.strip() if isinstance(parsed, HypotheticalAnswer) else None


def _messages(system: str, user: str) -> list[Message]:
    return [
        Message(role="system", content=system.strip()),
        Message(role="user", content=user.strip()),
    ]


async def gather_queries(
    question: str,
    *,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """`transform_query`, but never raising. Retrieval must survive a transform outage."""
    try:
        return await transform_query(question, client=client, loader=loader, settings=settings)
    except Exception as exc:
        logger.warning("query transform stage failed entirely: %s", exc)
        return [question]


__all__ = ["gather_queries", "transform_query"]
