"""`make ingest` — the ingestion pipeline end to end.

    data/corpus/*.md
      -> parse front-matter          (parse.py)
      -> recursive chunk             (chunk.py)
      -> embed, 384d, local          (embed.py)
      -> tsvector, in Postgres
      -> upsert under an index slot  (ADR-021)

# CONCEPT: Ingestion is idempotent by *identity*, not by timestamp. The slot's fingerprint
# decides: identical inputs are a no-op, different inputs into an occupied slot are an error.
# That is what lets `make ingest` be safe to run repeatedly and still protect an eval baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel
from sqlalchemy import Table, delete, func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.config import Settings, get_settings
from sutra.db.engine import get_sessionmaker
from sutra.db.models import Chunk as ChunkRow
from sutra.db.models import Document as DocumentRow
from sutra.ingestion.chunk import chunk_document
from sutra.ingestion.embed import Embedder, get_embedder
from sutra.ingestion.fingerprint import IndexFingerprint, build_fingerprint
from sutra.ingestion.parse import ParsedDocument, parse_corpus

logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/corpus")
# The chunks Table, not the ORM class. Inserts go through it so the keys in each row dict are the
# real column names — `metadata`, not the mapped attribute `meta` (see `_validate_row_keys`).
CHUNKS_TABLE = cast(Table, ChunkRow.__table__)
# Postgres text-search configuration. 'english' applies stemming and stop-word removal, so
# "waystones" matches "waystone". Unused by Phase 2's vector-only retrieval, but built now
# because Phase 3's hybrid search needs it and backfilling it later is a re-ingest.
TSV_CONFIG = "english"


class IngestionSummary(BaseModel):
    """What one ingestion run did.

    A model rather than a dict because CLAUDE.md puts Pydantic at every boundary, and because a
    dict of `object` makes every field unusable to the type checker — the caller cannot compare
    `summary["chunks"]` to a number without a cast.
    """

    status: Literal["ingested", "unchanged"]
    index_version: int
    fingerprint: str
    documents: int
    chunks: int


class IngestionBlocked(RuntimeError):
    """The slot is occupied by a different index. Refusing rather than overwriting."""


def _validate_row_keys(rows: list[dict[str, object]]) -> None:
    """Reject any key that is not a real column on `chunks`.

    # CONCEPT: A bulk insert built from dicts will happily *ignore* a key it does not recognise.
    # That turned a one-character mistake here ("meta", the mapped attribute, versus "metadata",
    # the column) into chunks written with empty metadata — which made the fingerprint
    # unreadable, which made ingestion neither idempotent nor guarded, and reported success the
    # whole time. Validating the key set converts that silence into an immediate error.
    """
    columns = set(CHUNKS_TABLE.columns.keys())
    for row in rows:
        unknown = set(row) - columns
        if unknown:
            msg = (
                f"chunk row has {len(unknown)} key(s) that are not columns on 'chunks': "
                f"{sorted(unknown)}. Known columns: {sorted(columns)}"
            )
            raise ValueError(msg)


async def _occupied_fingerprint(
    session: AsyncSession, index_version: int
) -> IndexFingerprint | None:
    """The fingerprint already stored in this slot, if anything is there.

    The whole fingerprint is read back, not just its digest, so a mismatch can report which
    parameter changed instead of two opaque hashes.
    """
    stmt = (
        select(ChunkRow.meta["index_fingerprint"])
        .where(ChunkRow.index_version == index_version)
        .limit(1)
    )
    stored = (await session.execute(stmt)).scalar_one_or_none()
    if stored is None:
        return None
    return IndexFingerprint.model_validate(stored)


async def _upsert_documents(
    session: AsyncSession, documents: list[ParsedDocument]
) -> dict[str, object]:
    """Insert or update one row per corpus file, returning slug -> id.

    Documents are keyed by slug and shared across index slots, so this is a true upsert rather
    than the delete-and-insert the chunks get.
    """
    ids: dict[str, object] = {}
    for doc in documents:
        stmt = (
            pg_insert(DocumentRow)
            .values(
                slug=doc.slug,
                title=doc.title,
                doc_type=doc.doc_type,
                meta=doc.meta,
                content_hash=doc.content_hash,
            )
            .on_conflict_do_update(
                index_elements=[DocumentRow.slug],
                set_={
                    "title": doc.title,
                    "doc_type": doc.doc_type,
                    "metadata": doc.meta,
                    "content_hash": doc.content_hash,
                },
            )
            .returning(DocumentRow.id)
        )
        ids[doc.slug] = (await session.execute(stmt)).scalar_one()
    return ids


async def ingest(
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    corpus_dir: Path = CORPUS_DIR,
    force: bool = False,
    index_version: int | None = None,
) -> IngestionSummary:
    """Run the pipeline."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    slot = index_version if index_version is not None else settings.index_version

    documents = parse_corpus(corpus_dir)
    fingerprint = build_fingerprint(settings, documents)
    logger.info("corpus: %d documents, index slot %d", len(documents), slot)

    factory = get_sessionmaker()
    async with factory() as session:
        occupied = await _occupied_fingerprint(session, slot)

        if occupied is not None and occupied.digest == fingerprint.digest and not force:
            # The whole point of the fingerprint: this is the common case when you run
            # `make ingest` twice, and it must be free.
            return IngestionSummary(
                status="unchanged",
                index_version=slot,
                fingerprint=fingerprint.digest,
                documents=len(documents),
                chunks=0,
            )

        if occupied is not None and occupied.digest != fingerprint.digest and not force:
            raise IngestionBlocked(_blocked_message(slot, occupied, fingerprint))

        if occupied is not None:
            await session.execute(delete(ChunkRow).where(ChunkRow.index_version == slot))

        document_ids = await _upsert_documents(session, documents)

        rows: list[dict[str, object]] = []
        for doc in documents:
            chunks = chunk_document(
                doc.body,
                max_tokens=settings.chunk_tokens,
                overlap_tokens=settings.chunk_overlap,
                count_tokens=embedder.count_tokens,
            )
            vectors = embedder.embed_passages([c.text for c in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                rows.append(
                    {
                        "document_id": document_ids[doc.slug],
                        "index_version": slot,
                        "parent_id": None,  # flat chunking; parent-child arrives in Phase 3
                        "chunk_no": chunk.chunk_no,
                        "text": chunk.text,
                        "context_prefix": None,  # contextual retrieval arrives in Phase 3
                        "embedding": vector,
                        "token_count": chunk.token_count,
                        "metadata": {
                            "index_fingerprint": fingerprint.model_dump(),
                            "doc_slug": doc.slug,
                            "doc_type": doc.doc_type,
                            "heading": chunk.heading,
                            **{k: v for k, v in doc.meta.items() if k != "in_world_dates"},
                        },
                    }
                )

        if rows:
            _validate_row_keys(rows)
            await session.execute(insert(CHUNKS_TABLE), rows)
            # # CONCEPT: tsv is *derived* data, so Postgres computes it rather than Python.
            # One set-based UPDATE beats sending a tsvector per row over the wire, and it keeps
            # the text-search configuration ('english' stemming) in the database where the
            # matching query will also resolve it.
            await session.execute(
                update(ChunkRow)
                .where(ChunkRow.index_version == slot)
                .values(tsv=func.to_tsvector(text(f"'{TSV_CONFIG}'"), ChunkRow.text))
            )
        await session.commit()

    return IngestionSummary(
        status="ingested",
        index_version=slot,
        fingerprint=fingerprint.digest,
        documents=len(documents),
        chunks=len(rows),
    )


def _blocked_message(slot: int, occupied: IndexFingerprint, fingerprint: IndexFingerprint) -> str:
    changes = "\n".join(f"  {line}" for line in fingerprint.differences_from(occupied))
    return (
        f"index slot {slot} already holds a different index "
        f"(stored {occupied.digest}, computed {fingerprint.digest}).\n"
        f"What changed:\n{changes}\n"
        "Refusing to overwrite: any eval baseline recorded against this slot would silently "
        "stop comparing.\n"
        "Either bump INDEX_VERSION to keep both, or re-run with --force to replace it."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest data/corpus into Postgres.")
    parser.add_argument("--force", action="store_true", help="replace an occupied index slot")
    parser.add_argument("--index-version", type=int, default=None, help="override the slot")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")

    try:
        summary = asyncio.run(
            ingest(corpus_dir=args.corpus_dir, force=args.force, index_version=args.index_version)
        )
    except IngestionBlocked as exc:
        print(f"\nBLOCKED\n{exc}\n", file=sys.stderr)
        return 1

    if summary.status == "unchanged":
        print(
            f"nothing to do: index slot {summary.index_version} already holds "
            f"fingerprint {summary.fingerprint} ({summary.documents} documents)"
        )
    else:
        print(
            f"ingested {summary.chunks} chunks from {summary.documents} documents "
            f"into index slot {summary.index_version} (fingerprint {summary.fingerprint})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
