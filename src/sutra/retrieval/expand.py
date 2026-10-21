"""Parent expansion — step 6 of the ARCHITECTURE §4 pipeline.

# CONCEPT: Retrieve the child, read the parent. Parent-child chunking splits the unit you
# search from the unit you answer from, and this module is the join between them. Without it
# the index is simply chunked smaller, which costs context and buys nothing — a 200-token
# fragment often contains the fact and not the sentence that says what the fact is about.
#
# Two things make this more than a lookup:
#   - **Dedupe.** Three children of one parent among the top 5 would otherwise send the same
#     800-token passage to the model three times, spending most of the context window on one
#     section. After expansion they collapse to one passage, which is also why the number of
#     passages the generator sees is usually smaller than k.
#   - **Rank is inherited, not recomputed.** A parent takes the best rank among its children,
#     because that is the evidence that brought it back. Re-scoring parents against the query
#     would be a different retriever, and would quietly undo the reason for searching children.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.db.models import Chunk as ChunkRow
from sutra.db.models import Document as DocumentRow
from sutra.observability.tracing import get_tracer, stage
from sutra.retrieval.search import RetrievedChunk

logger = logging.getLogger(__name__)


async def expand_to_parents(
    chunks: Sequence[RetrievedChunk], *, session: AsyncSession
) -> list[RetrievedChunk]:
    """Swap each retrieved child for its parent, deduped, in the children's rank order.

    A chunk with no parent is passed through unchanged. That is not an error case: a flat index
    has no parents at all, and even in a parent-child index a parent is a legitimate thing to
    hold a reference to. The function is therefore safe to call unconditionally, which keeps
    the caller free of a mode flag.
    """
    parent_ids = [c.parent_id for c in chunks if c.parent_id is not None]
    if not parent_ids:
        return list(chunks)

    tracer = get_tracer()

    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.chunk_no,
            ChunkRow.text,
            ChunkRow.token_count,
            ChunkRow.meta,
            DocumentRow.slug,
            DocumentRow.title,
            DocumentRow.doc_type,
        )
        .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
        .where(ChunkRow.id.in_(set(parent_ids)))
    )
    with stage(
        "expand_to_parents",
        kind="span",
        input={"children": len(chunks), "parents_requested": len(set(parent_ids))},
    ) as span:
        parents = {str(row.id): row for row in (await session.execute(stmt)).all()}
        # Fetched against requested: anything less than parity is a dangling parent_id, which
        # the warning below names but the span did not. The collapse ratio itself stays on
        # the `retrieval.expanded` event, which is emitted after the dedupe this span
        # deliberately does not cover.
        span.update(output={"parents_found": len(parents)})

    missing = set(parent_ids) - set(parents)
    if missing:
        # A dangling parent_id means the index is inconsistent. Say so and keep serving the
        # child rather than dropping evidence the retriever did find.
        logger.warning("expand: %d parent id(s) not found in the index", len(missing))

    expanded: list[RetrievedChunk] = []
    seen: set[str] = set()
    for child in chunks:
        row = parents.get(child.parent_id) if child.parent_id else None
        if row is None:
            key = child.chunk_id
            candidate = child
        else:
            key = str(row.id)
            candidate = RetrievedChunk(
                chunk_id=str(row.id),
                parent_id=None,
                doc_slug=row.slug,
                doc_title=row.title,
                doc_type=row.doc_type,
                chunk_no=row.chunk_no,
                heading=row.meta.get("heading"),
                text=row.text,
                token_count=row.token_count,
                # The child's score and rank, kept deliberately: this parent is here because
                # that child matched, and that is what the number should mean.
                score=child.score,
                rank=child.rank,
            )
        if key in seen:
            continue
        seen.add(key)
        expanded.append(candidate)

    # Ranks are renumbered so they stay contiguous after dedupe. Gaps would make MRR read as
    # though a passage had been retrieved further down than it was.
    result = [chunk.model_copy(update={"rank": rank}) for rank, chunk in enumerate(expanded, 1)]

    # The collapse ratio is the parent-child thesis in one number: children of one parent
    # dedupe, so the generator reads fewer, completer passages than the retriever ranked.
    # Phase 3 measured this at 5.0 -> 4.0; a trace says it per request.
    tracer.event(
        "retrieval.expanded",
        metadata={
            "children_in": len(chunks),
            "parents_out": len(result),
            "missing_parents": len(missing),
        },
    )
    return result
