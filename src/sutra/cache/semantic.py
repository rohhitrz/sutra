"""Semantic cache over pgvector (ADR-012, ARCHITECTURE §7).

# CONCEPT: An exact-match cache on a question string almost never hits, because people do not
# repeat themselves character for character. A semantic cache keys on the query *embedding*
# and hits when a new question is near enough to one already answered — "what is Kesh Ithar's
# population?" serving "how many people live in Kesh Ithar?".
#
# That is also what makes it dangerous, and this module is mostly about the danger. Two
# questions can be 0.96 cosine apart and have different answers: "what was the tithe in 1143"
# and "what was the tithe in 1144" differ by one token and by the entire fact. A similarity
# threshold is a *guess* about meaning, and every cache hit is that guess being trusted
# without the model in the loop to catch it.
#
# So the key is never similarity alone. A hit requires an exact match on all of:
#
#   user_id         or one person's answer leaks into another's session
#   prompt_version  a rewritten answer prompt produces different answers
#   index_version   re-ingesting changes what the answer was grounded in
#   model           a different model is a different answerer
#
# Those four are a composite index, applied *before* similarity is considered, so the vector
# search runs over the rows that are already known to be comparable. Versioning is what turns
# "probably still right" into "provably the same configuration".
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re

from pydantic import BaseModel
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.config import Settings, get_settings
from sutra.db.models import SemanticCache
from sutra.ingestion.embed import Embedder, get_embedder
from sutra.observability.tracing import get_tracer, stage

logger = logging.getLogger(__name__)


class CacheLookup(BaseModel):
    """The outcome of one cache probe. A miss is as informative as a hit, so both are typed."""

    hit: bool
    answer: dict | None = None
    similarity: float | None = None
    cached_query: str | None = None
    # The nearest entry's similarity even when it missed. Recorded because the threshold is a
    # guess, and the only way to tune it is to see what was almost served.
    nearest_similarity: float | None = None


# # CONCEPT: Similarity alone cannot separate these two questions:
# #
# #     "What was the Waystone Tithe rate in 1143?"   stored
# #     "What was the Waystone Tithe rate in 1144?"   probe     cosine 0.954
# #     "How many people live in Halvern Reach?"      paraphrase of a stored query, 0.955
# #
# # The wrong hit and the right hit are 0.001 apart, measured on this corpus by
# # `make cache`. There is no threshold that admits the paraphrase and rejects the year:
# # raise it and the cache stops working, lower it and it serves 1143's answer for 1144.
# #
# # So the threshold is not asked to carry that weight. A hit additionally requires the two
# # questions to mention the *same figures* — a cheap, exact, deterministic test on precisely
# # the tokens that distinguish otherwise-identical questions. Dates, prices, counts and
# # waystone numbers are what this corpus disambiguates on, and an embedding is worst exactly
# # there, because `1143` and `1144` are near-neighbours as tokens and unrelated as facts.
# #
# # This is the general shape of the fix, not a corpus quirk: wherever a soft signal cannot
# # resolve a distinction, put a hard one beside it rather than tuning the soft one harder.
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def discriminating_figures(text: str) -> frozenset[str]:
    """Figures a question turns on, normalised. Thousands separators are dropped so `2,300`
    and `2300` agree; nothing else is relaxed."""
    return frozenset(
        re.sub(r"(?<=\d),(?=\d)", "", match.group(0).rstrip(".,"))
        for match in _FIGURE.finditer(text)
    ) - {""}


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


async def lookup(
    query: str,
    *,
    user_id: str,
    session: AsyncSession,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> CacheLookup:
    """Probe the cache. Never raises: a cache that can fail a request is worse than no cache.

    The hit/miss event is emitted here rather than at each of the inner function's five
    return points, so no future early return can forget it. A cache whose hit rate you
    cannot see is a cache you cannot reason about: the number is the whole justification for
    the component, and a *wrong* hit is a served-from-the-wrong-question bug that shows up
    nowhere else.
    """
    settings = settings or get_settings()
    # Disabled is not a miss. Emitting one would put a 0% hit rate on a stage that never ran.
    if not settings.semantic_cache_enabled:
        return CacheLookup(hit=False)

    with stage("cache_lookup", kind="span", input={"query": query}) as span:
        result = await _lookup_inner(
            query, user_id=user_id, session=session, settings=settings, embedder=embedder
        )
        span.update(output=result.model_dump(exclude={"answer"}))

    get_tracer().event(
        "cache.hit" if result.hit else "cache.miss",
        metadata={
            "similarity": result.similarity,
            "nearest_similarity": result.nearest_similarity,
            "threshold": settings.cache_similarity_threshold,
            "cached_query": result.cached_query,
        },
    )
    return result


async def _lookup_inner(
    query: str,
    *,
    user_id: str,
    session: AsyncSession,
    settings: Settings,
    embedder: Embedder | None = None,
) -> CacheLookup:
    """The probe itself. See `lookup` for why the event lives in the wrapper."""
    embedder = embedder or get_embedder(settings)
    try:
        vector = await asyncio.to_thread(embedder.embed_query, query)
        distance = SemanticCache.query_embedding.cosine_distance(vector).label("distance")
        stmt = (
            select(SemanticCache.id, SemanticCache.answer, SemanticCache.query, distance)
            # The composite key first. Similarity only ever decides between rows already
            # known to share a configuration.
            .where(
                SemanticCache.user_id == user_id,
                SemanticCache.prompt_version == settings.prompt_version_answer,
                SemanticCache.index_version == settings.index_version,
                SemanticCache.model == settings.llm_strong,
                SemanticCache.query_embedding.is_not(None),
            )
            # An expired row is not a hit. TTL exists because the corpus, the prompts and the
            # model all change outside this table's knowledge.
            .where((SemanticCache.expires_at.is_(None)) | (SemanticCache.expires_at > _now()))
            .order_by(distance)
            .limit(1)
        )
        row = (await session.execute(stmt)).first()
    except Exception as exc:
        logger.warning("semantic cache lookup failed: %s", exc)
        return CacheLookup(hit=False)

    if row is None:
        return CacheLookup(hit=False)

    similarity = 1.0 - float(row.distance)
    if similarity < settings.cache_similarity_threshold:
        return CacheLookup(hit=False, nearest_similarity=similarity)

    if discriminating_figures(query) != discriminating_figures(row.query):
        logger.debug(
            "semantic cache: rejected a %.3f match on differing figures (%r vs %r)",
            similarity,
            query,
            row.query,
        )
        return CacheLookup(hit=False, nearest_similarity=similarity, cached_query=row.query)

    try:
        await session.execute(
            update(SemanticCache)
            .where(SemanticCache.id == row.id)
            .values(hits=SemanticCache.hits + 1)
        )
    except Exception as exc:  # pragma: no cover - accounting must not break a served answer
        logger.warning("semantic cache hit counter failed: %s", exc)

    return CacheLookup(
        hit=True,
        answer=row.answer,
        similarity=similarity,
        cached_query=row.query,
        nearest_similarity=similarity,
    )


async def store(
    query: str,
    answer: dict,
    *,
    user_id: str,
    session: AsyncSession,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> bool:
    """Record an answer against its query embedding. Returns whether anything was written.

    Refusals are stored like any other answer: "the corpus does not record that" is an
    expensive thing to recompute and is exactly as cacheable as a fact, under the same key.
    """
    settings = settings or get_settings()
    if not settings.semantic_cache_enabled:
        return False

    embedder = embedder or get_embedder(settings)
    try:
        vector = await asyncio.to_thread(embedder.embed_query, query)
        await session.execute(
            insert(SemanticCache).values(
                user_id=user_id,
                query=query,
                query_embedding=vector,
                answer=answer,
                prompt_version=settings.prompt_version_answer,
                index_version=settings.index_version,
                model=settings.llm_strong,
                expires_at=_now() + dt.timedelta(hours=settings.cache_ttl_hours),
            )
        )
    except Exception as exc:
        logger.warning("semantic cache store failed: %s", exc)
        return False
    return True


async def invalidate(
    *, session: AsyncSession, index_version: int | None = None, user_id: str | None = None
) -> int:
    """Drop cached answers. Returns rows removed.

    # CONCEPT: Versioning handles the *predictable* invalidation — a new prompt or index stops
    # matching the key by construction. This is for the rest: a corpus edit under the same
    # slot, a model swap behind the same name, anything that changes the answer without
    # changing the key. ARCHITECTURE §7 requires it on re-ingest for exactly that reason.
    """
    stmt = delete(SemanticCache)
    if index_version is not None:
        stmt = stmt.where(SemanticCache.index_version == index_version)
    if user_id is not None:
        stmt = stmt.where(SemanticCache.user_id == user_id)
    result = await session.execute(stmt)
    return int(getattr(result, "rowcount", 0) or 0)
