"""Vector-only retrieval — the deliberately naive Phase 2 baseline.

# CONCEPT: A baseline is a measurement instrument, not a first draft. Its job is to be the
# simplest thing that works, so that every later technique has a number to beat and the cost of
# each one is attributable. Making the baseline clever destroys the only reference point the
# project has.
#
# What this does: embed the query, take the k nearest chunks by cosine distance, return them.
# What it deliberately does NOT do, and where each arrives:
#   - keyword / BM25-ish search over the `tsv` column ..... Phase 3 (hybrid)
#   - rank fusion across retrievers (RRF) ................. Phase 3
#   - cross-encoder reranking ............................. Phase 3
#   - query rewriting and routing ......................... Phase 3
#   - parent-child expansion, contextual prefixes ......... Phase 3
#   - metadata filters, ACL enforcement ................... Phase 3 / later
#
# The known weakness of pure dense retrieval is exact-token recall: a rare proper noun or an
# exact figure ("68,000", "waystone 17") can rank below a fluent paraphrase, because the
# embedding captures topic rather than identity. The corpus is full of such facts on purpose, so
# the Phase 3 hybrid delta should be visible rather than theoretical.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.config import TSV_CONFIG, Settings, get_settings
from sutra.db.models import Chunk as ChunkRow
from sutra.db.models import Document as DocumentRow
from sutra.ingestion.embed import Embedder, get_embedder

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """One retrieved passage, carrying what both the generator and the evals need.

    `doc_slug` and `quote`-able `text` are what the eval metrics match against, and `rank` is
    kept explicitly rather than inferred from list position so a report or a trace can be read
    without assuming the list was never re-sorted.
    """

    chunk_id: str
    # The parent this chunk was cut from, when the index is parent-child. None in a flat index
    # and on parents themselves. `retrieval/expand.py` is what turns it into context.
    parent_id: str | None = None
    doc_slug: str
    doc_title: str
    doc_type: str
    chunk_no: int
    heading: str | None
    text: str
    # The generated prefix this chunk was *indexed* with (step 2), or None. Carried because
    # the reranker has to see what retrieval saw: scoring the bare text would hand the
    # cross-encoder the un-contextualised chunk, which is the handicap step 2 removed from
    # the embedder.
    context_prefix: str | None = None
    token_count: int | None
    # Cosine *similarity* in [-1, 1], not the distance Postgres returns. Higher is nearer.
    score: float
    rank: int


def _doc_type_clause(doc_types: Sequence[str]) -> tuple[Any, ...]:
    """A metadata filter, or nothing at all.

    Returns a tuple so an empty filter contributes no WHERE clause rather than a tautology —
    `general` is the common route and must cost nothing. The filter reads `doc_type` from the
    joined `documents` row, not from chunk metadata, for the same reason the slug does: the
    metadata copy is a denormalised convenience and the join is the source of truth.
    """
    if not doc_types:
        return ()
    return (DocumentRow.doc_type.in_(tuple(doc_types)),)


def _searchable_columns() -> list[Any]:
    """The columns every retriever selects, so their rows build the same `RetrievedChunk`."""
    return [
        ChunkRow.id,
        ChunkRow.parent_id,
        ChunkRow.chunk_no,
        ChunkRow.text,
        ChunkRow.context_prefix,
        ChunkRow.token_count,
        ChunkRow.meta,
        DocumentRow.slug,
        DocumentRow.title,
        DocumentRow.doc_type,
    ]


def _to_chunk(row: Any, *, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(row.id),
        parent_id=str(row.parent_id) if row.parent_id else None,
        # Joined from `documents` rather than read out of chunk metadata: the metadata copy
        # is a denormalised convenience, and the join is the source of truth.
        doc_slug=row.slug,
        doc_title=row.title,
        doc_type=row.doc_type,
        chunk_no=row.chunk_no,
        heading=row.meta.get("heading"),
        text=row.text,
        context_prefix=row.context_prefix,
        token_count=row.token_count,
        score=score,
        rank=rank,
    )


def _or_tsquery(query: str) -> str:
    """Turn a natural question into an OR-ed tsquery: `oren | dask | nineteenth`.

    # CONCEPT: A lexical *retriever* ORs its terms and lets ranking sort them out; a lexical
    # *filter* ANDs them. Postgres ships only the filter: `websearch_to_tsquery` and
    # `plainto_tsquery` both AND, so a chunk must contain every word of the question to be a
    # candidate at all.
    #
    # Measured before changing it, because the default looked reasonable: **21 of the 42
    # golden questions returned zero keyword hits**, median 0. Fused with a dense list, that
    # is not a weak second opinion — it is no second opinion on half the eval, and the hybrid
    # row would have been the vector row with extra steps and a plausible story attached.
    #
    # OR-ing matches far more chunks and pushes all the work onto `ts_rank_cd`, which is what
    # a ranking function is for and what BM25 does too. The cost is that a question's common
    # words now match widely; cover-density ranking handles that by rewarding chunks where
    # several query terms land close together.

    Only `\\w+` runs survive, so no user input can reach the tsquery parser as an operator.
    Postgres drops stop words itself; a question made entirely of them yields an empty query,
    which matches nothing rather than erroring.
    """
    terms = re.findall(r"\w+", query.lower())
    return " | ".join(terms)


async def keyword_search(
    query: str,
    *,
    session: AsyncSession,
    k: int | None = None,
    index_version: int | None = None,
    settings: Settings | None = None,
    doc_types: Sequence[str] = (),
) -> list[RetrievedChunk]:
    """Rank chunks by Postgres full-text search over the `tsv` column.

    # CONCEPT: Lexical retrieval answers a question dense retrieval cannot. An embedding
    # encodes topic, not identity — "waystone 17" and "waystone 19" are neighbours in vector
    # space and different facts in the world. A term index matches the token or it does not,
    # which is exactly the discrimination a rare proper noun or an exact figure needs.
    #
    # `websearch_to_tsquery` rather than `plainto_tsquery`: it accepts quoted phrases and `or`
    # the way a person actually types a query, and it never raises on punctuation — a
    # user-facing parser should not be able to turn a question mark into a 500.
    #
    # ADR-002 is honest that this is **not BM25**. `ts_rank_cd` is cover density ranking: it
    # rewards terms appearing close together and knows nothing about inverse document
    # frequency, so a common word weighs as much as a rare one. BM25's IDF term is the thing
    # we are giving up by keeping one datastore, and on a corpus full of rare proper nouns
    # that is the part that would have helped most. It is still a term match, which is the
    # property dense retrieval lacks entirely.
    """
    settings = settings or get_settings()
    top_k = k if k is not None else settings.retrieval_top_k
    slot = index_version if index_version is not None else settings.index_version

    tsquery = func.to_tsquery(text(f"'{TSV_CONFIG}'"), _or_tsquery(query))
    rank_score = func.ts_rank_cd(ChunkRow.tsv, tsquery).label("rank_score")

    stmt = (
        select(*_searchable_columns(), rank_score)
        .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
        .where(ChunkRow.index_version == slot)
        # Same reason as the vector side: parents are context units, not search units. They
        # *do* carry a tsvector, so without this filter keyword search would return passages
        # the vector half can never return, and fusion would be comparing two different
        # candidate sets.
        .where(ChunkRow.embedding.is_not(None))
        .where(ChunkRow.tsv.op("@@")(tsquery))
        .where(*_doc_type_clause(doc_types))
        .order_by(rank_score.desc())
        .limit(top_k)
    )

    rows = (await session.execute(stmt)).all()
    return [
        _to_chunk(row, score=float(row.rank_score), rank=rank)
        for rank, row in enumerate(rows, start=1)
    ]


async def vector_search(
    query: str,
    *,
    session: AsyncSession,
    k: int | None = None,
    index_version: int | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    doc_types: Sequence[str] = (),
) -> list[RetrievedChunk]:
    """Embed `query` and return its k nearest chunks from one index slot.

    Scoped to a single `index_version` on purpose: an un-scoped search over a database holding
    two index versions would silently blend them, and the result would belong to no measurable
    configuration at all.
    """
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    top_k = k if k is not None else settings.retrieval_top_k
    slot = index_version if index_version is not None else settings.index_version

    # Queries go through `embed_query`, passages through `embed_passages`. For this model under
    # fastembed the two happen to be identical (measured — see `embed.py`), but keeping the seam
    # means swapping in a model with a real query-side instruction is a change in one place.
    #
    # # CONCEPT: A synchronous CPU-bound call inside a coroutine blocks the whole event loop,
    # # not just its own request. ONNX inference is exactly that, so awaiting it in a thread is
    # # what keeps `/v1/chat` concurrent — and what lets the eval runner's `concurrency: 4`
    # # mean anything, since without it four "parallel" rows queue behind one embedder.
    # # It matters more from Phase 3 on: the cross-encoder reranker is far heavier than this.
    query_vector = await asyncio.to_thread(embedder.embed_query, query)

    # # CONCEPT: The distance operator has to match the one the index was built with
    # # (`vector_cosine_ops`, ADR-004). Mismatch them and Postgres silently declines the index
    # # and scans instead — correct answers, no index, no error.
    #
    # Honest caveat at this scale: with 49 chunks the planner declines the HNSW index anyway and
    # filters on `index_version` then sorts (confirmed with EXPLAIN). So this corpus does not
    # exercise the index, and the approximate-recall behaviour of HNSW is untested here — a
    # property of a 10-file corpus, not of the query.
    distance = ChunkRow.embedding.cosine_distance(query_vector).label("distance")

    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.parent_id,
            ChunkRow.chunk_no,
            ChunkRow.text,
            ChunkRow.context_prefix,
            ChunkRow.token_count,
            ChunkRow.meta,
            DocumentRow.slug,
            DocumentRow.title,
            DocumentRow.doc_type,
            distance,
        )
        .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
        .where(ChunkRow.index_version == slot)
        # Parents are stored in the same slot as their children but carry no embedding, because
        # they are context units rather than search units and are routinely larger than the
        # embedder's 512-token limit. Without this filter they come back with a NULL distance
        # and, on a small corpus, pad out the top k with rows that were never searchable.
        .where(ChunkRow.embedding.is_not(None))
        .where(*_doc_type_clause(doc_types))
        .order_by(distance)
        .limit(top_k)
    )

    rows = (await session.execute(stmt)).all()
    if not rows:
        logger.warning("no chunks in index slot %d — has `make ingest` run?", slot)

    return [
        RetrievedChunk(
            chunk_id=str(row.id),
            parent_id=str(row.parent_id) if row.parent_id else None,
            # Joined from `documents` rather than read out of chunk metadata: the metadata copy
            # is a denormalised convenience, and the join is the source of truth.
            doc_slug=row.slug,
            doc_title=row.title,
            doc_type=row.doc_type,
            chunk_no=row.chunk_no,
            heading=row.meta.get("heading"),
            text=row.text,
            context_prefix=row.context_prefix,
            token_count=row.token_count,
            score=1.0 - float(row.distance),
            rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]


async def hybrid_search(
    query: str,
    *,
    session: AsyncSession,
    k: int | None = None,
    index_version: int | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    doc_types: Sequence[str] = (),
) -> list[RetrievedChunk]:
    """Run both retrievers over a wider candidate pool, then fuse by reciprocal rank.

    # CONCEPT: Each half is asked for `retrieval_candidate_k` results, not `k`. Fusing two
    # top-5 lists can only ever reorder 5-10 documents, so a document that sits at rank 7 on
    # one side and rank 2 on the other — precisely the case fusion exists to rescue — is never
    # considered. Retrieve wide, fuse, then cut to k.

    The two queries run **sequentially**, not concurrently. One `AsyncSession` cannot have two
    statements in flight, and giving each half its own session would double the connections
    held per request to save a few milliseconds on a query that is not the bottleneck. The
    embedding, which is the slow part, already runs off the event loop.
    """
    from sutra.retrieval.fusion import reciprocal_rank_fusion

    settings = settings or get_settings()
    top_k = k if k is not None else settings.retrieval_top_k
    candidates = max(settings.retrieval_candidate_k, top_k)

    dense = await vector_search(
        query,
        session=session,
        k=candidates,
        index_version=index_version,
        settings=settings,
        embedder=embedder,
        doc_types=doc_types,
    )
    lexical = await keyword_search(
        query,
        session=session,
        k=candidates,
        index_version=index_version,
        settings=settings,
        doc_types=doc_types,
    )
    logger.debug(
        "hybrid: %d dense, %d lexical candidates -> top %d", len(dense), len(lexical), top_k
    )
    return reciprocal_rank_fusion([dense, lexical], k=settings.rrf_k, limit=top_k)


async def search(
    query: str,
    *,
    session: AsyncSession,
    k: int | None = None,
    index_version: int | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    reranker: object | None = None,
) -> list[RetrievedChunk]:
    """The configured retrieval pipeline: retrieve, optionally rerank, return the top k.

    One entry point, so callers do not branch on mode. `retrieval_mode` and `rerank_enabled`
    exist so `make eval` can ablate each stage: the only way to say what a technique bought is
    to have measured without it, under the same k, on the same index, with everything else
    held still.

    When reranking is on, retrieval fetches `rerank_candidates` rather than `k`. A reranker can
    only reorder what it is given, so its ceiling is the recall of that pool — handing it five
    results would let it sort five and discover nothing.
    """
    settings = settings or get_settings()
    top_k = k if k is not None else settings.retrieval_top_k
    pool = max(settings.rerank_candidates, top_k) if settings.rerank_enabled else top_k

    # # CONCEPT: Multi-query retrieval. A single embedding cannot sit near two unrelated
    # # passages at once, so a question needing two facts is asking the retriever to be in two
    # # places. Issuing one query per fact and fusing the results is the obvious fix, and
    # # nothing cheaper works — which is why this is the stage aimed at multi-hop.
    from sutra.retrieval.query_transform import gather_queries
    from sutra.retrieval.router import route

    decision = await route(query, settings=settings)
    doc_types = decision.doc_types
    queries = await gather_queries(query, settings=settings)
    if len(queries) == 1:
        hits = await _retrieve(
            query,
            session=session,
            k=pool,
            index_version=index_version,
            settings=settings,
            embedder=embedder,
            doc_types=doc_types,
        )
    else:
        from sutra.retrieval.fusion import reciprocal_rank_fusion

        # Sequential, like the hybrid halves and for the same reason: one AsyncSession cannot
        # have two statements in flight, and a second connection per request is a poor trade
        # for a query that is not the bottleneck.
        per_query = [
            await _retrieve(
                q,
                session=session,
                k=pool,
                index_version=index_version,
                settings=settings,
                embedder=embedder,
                doc_types=doc_types,
            )
            for q in queries
        ]
        logger.debug("multi-query: %d queries -> pool of %d", len(queries), pool)
        hits = reciprocal_rank_fusion(per_query, k=settings.rrf_k, limit=pool)
    if not settings.rerank_enabled:
        return hits

    from sutra.retrieval.rerank import Reranker, rerank

    return await rerank(
        query,
        hits,
        top_k=top_k,
        reranker=cast("Reranker | None", reranker),
        settings=settings,
    )


async def _retrieve(
    query: str,
    *,
    session: AsyncSession,
    k: int | None = None,
    index_version: int | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    doc_types: Sequence[str] = (),
) -> list[RetrievedChunk]:
    """Dispatch to the configured retriever, before any reranking."""
    settings = settings or get_settings()
    mode = settings.retrieval_mode
    if mode == "keyword":
        return await keyword_search(
            query,
            session=session,
            k=k,
            index_version=index_version,
            settings=settings,
            doc_types=doc_types,
        )
    if mode == "hybrid":
        return await hybrid_search(
            query,
            session=session,
            k=k,
            index_version=index_version,
            settings=settings,
            embedder=embedder,
            doc_types=doc_types,
        )
    return await vector_search(
        query,
        session=session,
        k=k,
        index_version=index_version,
        settings=settings,
        embedder=embedder,
        doc_types=doc_types,
    )
