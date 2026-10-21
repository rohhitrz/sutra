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

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.config import Settings, get_settings
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
    token_count: int | None
    # Cosine *similarity* in [-1, 1], not the distance Postgres returns. Higher is nearer.
    score: float
    rank: int


async def vector_search(
    query: str,
    *,
    session: AsyncSession,
    k: int | None = None,
    index_version: int | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
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
            token_count=row.token_count,
            score=1.0 - float(row.distance),
            rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]
