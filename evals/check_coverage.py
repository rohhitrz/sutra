"""Is every cited quote still retrievable after chunking?

# CONCEPT: This is the gate that makes Phase 2's retrieval metrics mean anything. Ground truth
# here is 72 verbatim quotes, each tied to a document — not chunk ids, because chunk ids change
# with every re-ingest. Recall@k is therefore computed by asking "did any retrieved chunk contain
# this quote?".
#
# That measurement has a silent failure mode. If the chunker splits a cited quote across a chunk
# boundary, no chunk contains it, so the quote can never be matched however good retrieval is.
# Recall drops, nothing errors, and the number looks like a *retrieval* result — sending you to
# tune the retriever over a defect in the chunker. This script makes that condition a build
# failure instead of a mystery.
#
# `make validate` (Phase 1) proves each quote exists in the corpus. This proves each quote
# survives chunking. The two together are what let a recall number be read at face value.

Run with `make coverage` after `make ingest`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter

from pydantic import BaseModel

from evals.validate_datasets import CORPUS_DIR, DATASET_DIR, GoldenRow, load_rows, normalise
from sutra.config import Settings, get_settings
from sutra.ingestion.chunk import chunk_document
from sutra.ingestion.embed import get_embedder
from sutra.ingestion.parse import parse_corpus

# A quote longer than this is reported with an ellipsis so the table stays readable.
QUOTE_DISPLAY = 58


class QuoteCheck(BaseModel):
    """One cited quote, and whether chunking left it intact."""

    row_id: str
    doc_slug: str
    quote: str
    in_corpus: bool
    chunks_containing: list[int]

    @property
    def covered(self) -> bool:
        return bool(self.chunks_containing)

    @property
    def diagnosis(self) -> str:
        """Why an uncovered quote is uncovered. The two causes need opposite fixes."""
        if self.covered:
            return "ok"
        if not self.in_corpus:
            # A dataset bug: `make validate` should have caught this first.
            return "NOT IN CORPUS — the dataset cites text the corpus does not contain"
        # A chunker bug: the text is there, but no single chunk holds all of it.
        return "SPLIT — the quote straddles a chunk boundary, so no chunk can match it"


def _display(quote: str) -> str:
    collapsed = normalise(quote)
    if len(collapsed) <= QUOTE_DISPLAY:
        return collapsed
    return collapsed[: QUOTE_DISPLAY - 1] + "…"


def check_coverage(
    chunks_by_slug: dict[str, list[str]], corpus_by_slug: dict[str, str]
) -> list[QuoteCheck]:
    """Match every cited quote against the chunks of its document.

    Normalises whitespace on both sides, exactly as `make validate` does: quotes are copied out
    of hard-wrapped markdown, so a correct quote still carries a newline where the line broke.
    """
    golden, errors = load_rows(DATASET_DIR / "golden_qa.jsonl", GoldenRow)
    if errors:
        msg = f"golden_qa.jsonl does not validate ({len(errors)} errors) — run `make validate`"
        raise SystemExit(msg)

    results: list[QuoteCheck] = []
    for row in golden:
        for source in row.source_chunks:
            needle = normalise(source.quote)
            chunks = chunks_by_slug.get(source.doc_slug, [])
            results.append(
                QuoteCheck(
                    row_id=row.id,
                    doc_slug=source.doc_slug,
                    quote=source.quote,
                    in_corpus=needle in corpus_by_slug.get(source.doc_slug, ""),
                    chunks_containing=[
                        i for i, text in enumerate(chunks) if needle in normalise(text)
                    ],
                )
            )
    return results


def chunks_from_corpus(settings: Settings) -> dict[str, list[str]]:
    """Chunk the corpus in memory, with the real tokenizer.

    Needs no database, so this runs in the test suite and in CI. It uses the embedder's own
    tokenizer rather than an estimate, because a coverage check computed under different chunk
    boundaries than the real ingestion would be worse than no check at all.
    """
    embedder = get_embedder(settings)
    chunks: dict[str, list[str]] = {}
    for doc in parse_corpus(CORPUS_DIR):
        chunks[doc.slug] = [
            c.text
            for c in chunk_document(
                doc.body,
                max_tokens=settings.chunk_tokens,
                overlap_tokens=settings.chunk_overlap,
                count_tokens=embedder.count_tokens,
            )
        ]
    return chunks


async def chunks_from_db(settings: Settings, index_version: int) -> dict[str, dict[str, list[str]]]:
    """Read the chunks actually stored in an index slot, split by role.

    Checks what retrieval will really see, so it catches ingestion faults the in-memory path
    cannot — a dropped column, a half-finished run, the wrong slot.

    Parents and children are returned separately, and this matters. In a parent-child index a
    parent contains all of its children's text, so pooling the two makes coverage pass whenever
    a quote survives in *any* unit — including when the quote was split across children and
    only the parent holds it whole. Recall@k is measured on children, so that pooled pass would
    hide the exact defect this script exists to catch, and the recall drop would be read as a
    retrieval result. Each role is therefore checked on its own terms:

      - **children** are the search units; a split here understates recall
      - **parents** are the context units; a split here means the generator never sees the
        quote whole even when retrieval was perfect

    A flat index has no children, and its rows come back under "search".
    """
    from sqlalchemy import select

    from sutra.db.engine import get_sessionmaker
    from sutra.db.models import Chunk as ChunkRow

    search: dict[str, list[str]] = {}
    context: dict[str, list[str]] = {}
    factory = get_sessionmaker()
    async with factory() as session:
        stmt = (
            select(ChunkRow.meta["doc_slug"].astext, ChunkRow.text, ChunkRow.parent_id)
            .where(ChunkRow.index_version == index_version)
            .order_by(ChunkRow.meta["doc_slug"].astext, ChunkRow.chunk_no)
        )
        rows = (await session.execute(stmt)).all()

    has_children = any(parent_id is not None for _, _, parent_id in rows)
    for slug, text, parent_id in rows:
        is_search_unit = (parent_id is not None) if has_children else True
        bucket = search if is_search_unit else context
        bucket.setdefault(slug, []).append(text)

    roles = {"search": search}
    if has_children:
        roles["context"] = context
    return roles


def corpus_text() -> dict[str, str]:
    return {
        p.stem: normalise(p.read_text())
        for p in sorted(CORPUS_DIR.glob("*.md"))
        if p.name != "README.md"
    }


def report(results: list[QuoteCheck], source: str) -> int:
    uncovered = [r for r in results if not r.covered]
    print(f"\nquote coverage ({source}): {len(results) - len(uncovered)}/{len(results)} intact")

    by_doc = Counter(r.doc_slug for r in results)
    for slug in sorted(by_doc):
        missing = sum(1 for r in uncovered if r.doc_slug == slug)
        flag = "" if not missing else f"   <-- {missing} UNCOVERED"
        print(f"  {slug:24s} {by_doc[slug]:3d} quotes{flag}")

    if not uncovered:
        # A quote found in several chunks is not a problem: overlap is meant to do that, and it
        # gives a fact near a boundary a second chance to appear somewhere whole.
        spread = Counter(len(r.chunks_containing) for r in results)
        print(f"  chunks per quote: {dict(sorted(spread.items()))}")
        print("\nok    every cited quote survives chunking")
        return 0

    print(f"\nFAIL  {len(uncovered)} quote(s) cannot be matched against any chunk:")
    for r in uncovered:
        print(f"  - {r.row_id}  {r.doc_slug}")
        print(f"      {_display(r.quote)!r}")
        print(f"      {r.diagnosis}")
    print(
        "\nEvery recall number computed in this state understates retrieval, because these "
        "quotes are unmatchable no matter what the retriever returns."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="check the chunks stored in Postgres instead of chunking the corpus in memory",
    )
    parser.add_argument("--index-version", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.from_db:
        slot = args.index_version or settings.index_version
        roles = asyncio.run(chunks_from_db(settings, slot))
        if not any(roles.values()):
            print(f"FAIL  index slot {slot} is empty — run `make ingest` first", file=sys.stderr)
            return 1
        source = f"index slot {slot} in Postgres"
    else:
        roles = {"search": chunks_from_corpus(settings)}
        source = "corpus chunked in memory"

    # Every role has to pass. A parent-child index that covers its quotes only in the parents
    # still breaks recall, which is measured on the children.
    text = corpus_text()
    exit_code = 0
    for role, chunks in roles.items():
        total = sum(len(v) for v in chunks.values())
        label = {"search": "search units (what recall is measured on)", "context": "context units"}
        print(f"\n{total} {role} chunks over {len(chunks)} documents — {label[role]}")
        exit_code |= report(check_coverage(chunks, text), f"{source}, {role} units")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
