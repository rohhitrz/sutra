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
import uuid
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError
from sqlalchemy import Table, delete, func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from sutra.config import CONTEXT_PREFIX_BUDGET_TOKENS, TSV_CONFIG, Settings, get_settings
from sutra.db.engine import get_sessionmaker
from sutra.db.models import Chunk as ChunkRow
from sutra.db.models import Document as DocumentRow
from sutra.ingestion.chunk import chunk_document, chunk_document_parent_child
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


class OccupiedSlot(BaseModel):
    """What is already sitting in an index slot.

    `fingerprint` is None when the stored value cannot be parsed as a current fingerprint — an
    index written before a field was added to the schema, for instance. That is a third state,
    distinct from "slot empty" and "slot matches", and it has to be refused rather than crash:
    the fingerprint is persisted data, so its schema outlives the code that wrote it. ADR-022
    adding `chunker_version` is exactly this case.
    """

    raw: dict[str, Any]
    fingerprint: IndexFingerprint | None


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
    # How many chunks got a generated context prefix this run (step 2). Zero when
    # contextual retrieval is off, and also when every prefix came from the cache.
    context_prefixes: int = 0
    # Prefixes that overran the reserved budget and lost their tail. Reported rather than only
    # logged: a non-zero count means the index was built from trimmed scaffolding, which is a
    # fact about the measurement, not a warning to scroll past.
    context_prefixes_trimmed: int = 0


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


async def _occupied_slot(session: AsyncSession, index_version: int) -> OccupiedSlot | None:
    """Read back whatever fingerprint this slot holds, tolerating an older schema.

    The whole fingerprint is read, not just its digest, so a mismatch can report which parameter
    changed instead of two opaque hashes.

    Occupancy is decided by **whether rows exist**, never by whether a fingerprint can be read
    from them. Selecting the JSON path directly conflates the two: a slot holding chunks whose
    metadata lacks `index_fingerprint` comes back as SQL NULL, reads as an empty slot, and skips
    the delete below — which then either trips `uq_chunk_position` or, worse, leaves the slot
    holding two chunkings blended together. "Rows but no readable fingerprint" is the third
    state this class exists to carry, so it has to survive the query that detects it.
    """
    stmt = select(ChunkRow.meta).where(ChunkRow.index_version == index_version).limit(1)
    # `.first()`, not `scalar_one_or_none()`: a row whose `metadata` is NULL is still a row, and
    # the scalar form would report it as no row at all.
    row = (await session.execute(stmt)).first()
    if row is None:
        return None

    meta = row[0]
    stored = meta.get("index_fingerprint") if isinstance(meta, dict) else None
    if not isinstance(stored, dict):
        return OccupiedSlot(raw={}, fingerprint=None)
    try:
        return OccupiedSlot(raw=stored, fingerprint=IndexFingerprint.model_validate(stored))
    except ValidationError:
        return OccupiedSlot(raw=stored, fingerprint=None)


async def read_slot_fingerprint(index_version: int) -> OccupiedSlot | None:
    """The fingerprint an index slot actually holds, on its own session.

    # CONCEPT: An index's identity is a property of the index, not of whoever is reading it.
    # Anything reporting on a slot — an eval report above all — has to ask the slot, because
    # recomputing the fingerprint from the current config answers a different question:
    # "what would this configuration build?" rather than "what is in there?". Those two
    # coincide only right after `make ingest`, and an eval profile selecting slot 3 while the
    # developer's `.env` still describes slot 1 is the ordinary case, not an edge case.
    """
    factory = get_sessionmaker()
    async with factory() as session:
        return await _occupied_slot(session, index_version)


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
        occupied = await _occupied_slot(session, slot)
        matches = (
            occupied is not None
            and occupied.fingerprint is not None
            and (occupied.fingerprint.digest == fingerprint.digest)
        )

        if matches and not force:
            # The whole point of the fingerprint: this is the common case when you run
            # `make ingest` twice, and it must be free.
            return IngestionSummary(
                status="unchanged",
                index_version=slot,
                fingerprint=fingerprint.digest,
                documents=len(documents),
                chunks=0,
            )

        if occupied is not None and not matches and not force:
            raise IngestionBlocked(_blocked_message(slot, occupied, fingerprint))

        if occupied is not None:
            await session.execute(delete(ChunkRow).where(ChunkRow.index_version == slot))

        document_ids = await _upsert_documents(session, documents)

        build = _parent_child_rows if settings.parent_child_enabled else _flat_rows
        rows = build(documents, document_ids, settings, embedder, fingerprint, slot)
        prefixes_written, prefixes_trimmed = await _contextualise_and_embed(
            session, rows, documents, settings, embedder
        )

        if rows:
            _validate_row_keys(rows)
            await session.execute(insert(CHUNKS_TABLE), rows)
            # # CONCEPT: tsv is *derived* data, so Postgres computes it rather than Python.
            # One set-based UPDATE beats sending a tsvector per row over the wire, and it keeps
            # the text-search configuration ('english' stemming) in the database where the
            # matching query will also resolve it.
            #
            # The prefix is indexed with the text, not instead of it: keyword search should see
            # the contextualised chunk for the same reason the embedding does. `coalesce`
            # rather than a plain concat, because `NULL || text` is NULL in SQL and would
            # silently blank the tsvector of every chunk that has no prefix.
            contextualised = func.concat(
                func.coalesce(ChunkRow.context_prefix + text("' '"), text("''")), ChunkRow.text
            )
            await session.execute(
                update(ChunkRow)
                .where(ChunkRow.index_version == slot)
                .values(tsv=func.to_tsvector(text(f"'{TSV_CONFIG}'"), contextualised))
            )
        await session.commit()

    return IngestionSummary(
        status="ingested",
        index_version=slot,
        fingerprint=fingerprint.digest,
        documents=len(documents),
        chunks=len(rows),
        context_prefixes=prefixes_written,
        context_prefixes_trimmed=prefixes_trimmed,
    )


def _base_metadata(
    doc: ParsedDocument, fingerprint: IndexFingerprint, heading: str | None
) -> dict[str, object]:
    return {
        "index_fingerprint": fingerprint.model_dump(),
        "doc_slug": doc.slug,
        "doc_type": doc.doc_type,
        "heading": heading,
        **{k: v for k, v in doc.meta.items() if k != "in_world_dates"},
    }


def _flat_rows(
    documents: list[ParsedDocument],
    document_ids: dict[str, object],
    settings: Settings,
    embedder: Embedder,
    fingerprint: IndexFingerprint,
    slot: int,
) -> list[dict[str, object]]:
    """One row per chunk, no hierarchy — the Phase 2 shape, and what slot 1 holds.

    This path always chunks recursively: semantic splitting is implemented on the parent-child
    path only. `Settings` refuses `chunk_strategy='semantic'` without `parent_child_enabled`
    rather than letting it land here, because the fingerprint records the *requested* strategy
    and would otherwise stamp "semantic" on a recursive index.
    """
    rows: list[dict[str, object]] = []
    for doc in documents:
        chunks = chunk_document(
            doc.body,
            max_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap,
            count_tokens=embedder.count_tokens,
        )
        for chunk in chunks:
            rows.append(
                {
                    "document_id": document_ids[doc.slug],
                    "index_version": slot,
                    "id": uuid.uuid4(),
                    "parent_id": None,
                    "chunk_no": chunk.chunk_no,
                    "text": chunk.text,
                    "context_prefix": None,
                    # Filled by `_contextualise_and_embed`, after any prefix is generated.
                    "embedding": None,
                    "token_count": chunk.token_count,
                    "metadata": _base_metadata(doc, fingerprint, chunk.heading),
                }
            )
    return rows


def _parent_child_rows(
    documents: list[ParsedDocument],
    document_ids: dict[str, object],
    settings: Settings,
    embedder: Embedder,
    fingerprint: IndexFingerprint,
    slot: int,
) -> list[dict[str, object]]:
    """Parents and children in one slot, children carrying `parent_id`.

    Parent ids are minted here rather than read back from the database after an insert. One
    insert for both kinds is simpler to reason about than two ordered ones, and it removes the
    window in which a failure could leave parents committed without their children.

    **Parents are deliberately not embedded.** They exist to be read, not searched: a parent
    routinely exceeds the embedder's 512-token limit, so embedding one would silently store a
    vector for its first half and nothing else. `search.py` filters on `embedding IS NOT NULL`,
    which is what keeps them out of results.
    """
    rows: list[dict[str, object]] = []
    for doc in documents:
        parents = chunk_document_parent_child(
            doc.body,
            strategy=settings.chunk_strategy,
            parent_tokens=settings.parent_tokens,
            child_tokens=settings.child_tokens,
            child_overlap=settings.child_overlap,
            breakpoint_percentile=settings.semantic_breakpoint_percentile,
            count_tokens=embedder.count_tokens,
            embed=embedder.embed_passages,
        )
        for parent in parents:
            parent_uuid = uuid.uuid4()
            rows.append(
                {
                    "document_id": document_ids[doc.slug],
                    "index_version": slot,
                    "id": parent_uuid,
                    "parent_id": None,
                    "chunk_no": parent.chunk_no,
                    "text": parent.text,
                    "context_prefix": None,
                    "embedding": None,
                    "token_count": parent.token_count,
                    "metadata": {
                        **_base_metadata(doc, fingerprint, parent.heading),
                        # The explicit marker. `embedding IS NULL` happens to identify parents
                        # today, but that is a consequence of how they are used, not a
                        # declaration of what they are — and a future unembedded child would
                        # make it wrong.
                        "is_parent": True,
                        "child_count": len(parent.children),
                    },
                }
            )
            for child in parent.children:
                rows.append(
                    {
                        "document_id": document_ids[doc.slug],
                        "index_version": slot,
                        # Explicit, like the parent's. `insert().executemany()` builds one
                        # statement from the first row's keys, so every row in the batch has
                        # to carry the same ones — a row omitting `id` fails the whole insert.
                        "id": uuid.uuid4(),
                        "parent_id": parent_uuid,
                        "chunk_no": child.chunk_no,
                        "text": child.text,
                        "context_prefix": None,
                        "embedding": None,  # filled by `_contextualise_and_embed`
                        "token_count": child.token_count,
                        "metadata": {
                            **_base_metadata(doc, fingerprint, child.heading),
                            "is_parent": False,
                            "parent_chunk_no": parent.chunk_no,
                        },
                    }
                )
    return rows


async def _cached_prefixes(session: AsyncSession) -> dict[str, str]:
    """Every prefix already generated, keyed by content hash, from any index slot.

    The database is the cache. A prefix depends only on the chunk text, its document and the
    prompt version — all three are in the hash — so one generated for slot 2 is equally valid
    for slot 3, and re-ingesting an unchanged corpus costs nothing. Without this, idempotent
    ingestion would still spend a hundred model calls every run, which would make ADR-021's
    "re-running is free" true about rows and false about money.
    """
    stmt = select(ChunkRow.meta["prefix_hash"].astext, ChunkRow.context_prefix).where(
        ChunkRow.context_prefix.is_not(None)
    )
    return {
        prefix_hash: prefix
        for prefix_hash, prefix in (await session.execute(stmt)).all()
        if prefix_hash and prefix
    }


async def _contextualise_and_embed(
    session: AsyncSession,
    rows: list[dict[str, object]],
    documents: list[ParsedDocument],
    settings: Settings,
    embedder: Embedder,
) -> tuple[int, int]:
    """Fill `context_prefix` and `embedding` on every searchable row.

    Returns (prefixes written, prefixes trimmed to the token budget).

    Parents are skipped: they are context units, never searched, and deliberately carry no
    embedding. Everything else is embedded from the *contextualised* text while the stored
    `text` column keeps the verbatim slice — which is what lets citations quote the source and
    the eval match its 72 verbatim quotes against an index that searched something else.
    """
    searchable = [r for r in rows if not cast(dict, r["metadata"]).get("is_parent", False)]
    if not searchable:
        return 0, 0

    written = 0
    if settings.contextual_prefixes_enabled:
        from sutra.ingestion.contextualize import (
            ChunkToContextualise,
            content_hash,
            generate_prefixes,
        )
        from sutra.llm.client import LLMClient
        from sutra.prompts import PromptLoader

        bodies = {doc.slug: doc.body for doc in documents}
        titles = {doc.slug: doc.title for doc in documents}
        pending: list[ChunkToContextualise] = []
        for row in searchable:
            meta = cast(dict, row["metadata"])
            slug = cast(str, meta["doc_slug"])
            key = content_hash(
                doc_slug=slug,
                text=cast(str, row["text"]),
                prompt_version=settings.prompt_version_context_prefix,
            )
            meta["prefix_hash"] = key
            pending.append(
                ChunkToContextualise(
                    key=key,
                    doc_slug=slug,
                    doc_title=titles.get(slug, slug),
                    heading=meta.get("heading"),
                    text=cast(str, row["text"]),
                )
            )

        prefixes = await generate_prefixes(
            pending,
            documents=bodies,
            cached=await _cached_prefixes(session),
            client=LLMClient(settings=settings),
            loader=PromptLoader(),
            settings=settings,
        )
        for row, chunk in zip(searchable, pending, strict=True):
            prefix = prefixes.get(chunk.key)
            if prefix:
                row["context_prefix"] = prefix
                written += 1

    from sutra.ingestion.contextualize import contextualised_text

    trimmed = _trim_prefixes_to_budget(searchable, embedder)
    texts = [
        contextualised_text(cast("str | None", r["context_prefix"]), cast(str, r["text"]))
        for r in searchable
    ]
    # Blocking ONNX work, kept off the event loop for the same reason `vector_search` does it.
    vectors = await asyncio.to_thread(embedder.embed_passages, texts)
    for row, vector in zip(searchable, vectors, strict=True):
        row["embedding"] = vector
    return written, trimmed


def _trim_prefixes_to_budget(rows: list[dict[str, object]], embedder: Embedder) -> int:
    """Hold every prefix inside `CONTEXT_PREFIX_BUDGET_TOKENS`. Returns how many were trimmed.

    # CONCEPT: Where a soft bound cannot be trusted, put a hard one beside it. `Settings`
    # reserves this many tokens for the prefix so the chunk budget plus the reserve provably
    # fits the embedder's 512-token ceiling — but the prefix is written by a model, so nothing
    # about the generation step keeps that promise. This is what makes the reserve a guarantee:
    # a long prefix loses its tail, loudly, instead of the *chunk* losing its tail silently.
    #
    # The prefix is the right thing to cut. It is retrieval scaffolding, generated and
    # discardable; the chunk is the indexed content, and truncating it would mean searching
    # text the stored row does not contain.
    """
    trimmed = 0
    for row in rows:
        prefix = cast("str | None", row["context_prefix"])
        if not prefix or embedder.count_tokens(prefix) <= CONTEXT_PREFIX_BUDGET_TOKENS:
            continue
        words = prefix.split()
        # Shrink by whole words from the end: a prefix cut mid-token reads as a typo in the
        # index, and binary search over `count_tokens` is not worth it for a rare path.
        while words and embedder.count_tokens(" ".join(words)) > CONTEXT_PREFIX_BUDGET_TOKENS:
            words.pop()
        row["context_prefix"] = " ".join(words)
        trimmed += 1
        logger.warning(
            "context prefix for %s exceeded the %d-token budget and was trimmed to %d words",
            cast(dict, row["metadata"]).get("doc_slug"),
            CONTEXT_PREFIX_BUDGET_TOKENS,
            len(words),
        )
    return trimmed


def _blocked_message(slot: int, occupied: OccupiedSlot, fingerprint: IndexFingerprint) -> str:
    refusal = (
        "Refusing to overwrite: any eval baseline recorded against this slot would silently "
        "stop comparing.\n"
        "Either bump INDEX_VERSION to keep both, or re-run with --force to replace it."
    )
    if occupied.fingerprint is None:
        if not occupied.raw:
            return (
                f"index slot {slot} holds chunks that carry no index fingerprint at all, so "
                f"there is nothing to compare the current one against.\n"
                f"{refusal}"
            )
        missing = sorted(set(IndexFingerprint.model_fields) - set(occupied.raw))
        return (
            f"index slot {slot} holds a fingerprint written by an older schema "
            f"(missing {missing}), so it cannot be compared to the current one.\n"
            f"{refusal}"
        )
    changes = "\n".join(f"  {line}" for line in fingerprint.differences_from(occupied.fingerprint))
    return (
        f"index slot {slot} already holds a different index "
        f"(stored {occupied.fingerprint.digest}, computed {fingerprint.digest}).\n"
        f"What changed:\n{changes}\n"
        f"{refusal}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest data/corpus into Postgres.")
    parser.add_argument("--force", action="store_true", help="replace an occupied index slot")
    parser.add_argument("--index-version", type=int, default=None, help="override the slot")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    # Chunking shape on the command line rather than only in `.env`, so the command that built
    # a slot is a thing you can paste into PROGRESS and re-run exactly.
    parser.add_argument(
        "--parent-child",
        action="store_true",
        help="write parent/child chunks instead of flat ones (Phase 3 step 1)",
    )
    parser.add_argument(
        "--contextual",
        action="store_true",
        help="generate a context prefix per chunk before embedding (Phase 3 step 2)",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic"],
        default=None,
        help="chunking strategy; defaults to the configured one",
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

    overrides: dict[str, object] = {}
    if args.parent_child:
        overrides["parent_child_enabled"] = True
    if args.contextual:
        overrides["contextual_prefixes_enabled"] = True
    if args.strategy is not None:
        overrides["chunk_strategy"] = args.strategy
    if overrides:
        # `model_copy` would skip the validator that keeps the embedded unit under the 512-token
        # ceiling, and that check is the whole reason the validator exists.
        try:
            settings = Settings(**{**settings.model_dump(), **overrides})
        except ValidationError as exc:
            # A rejected flag combination is a usage error, not a crash. Printing the validator's
            # own message keeps the explanation in one place.
            for error in exc.errors():
                print(f"\nBLOCKED\n{error['msg'].removeprefix('Value error, ')}\n", file=sys.stderr)
            return 1

    try:
        summary = asyncio.run(
            ingest(
                corpus_dir=args.corpus_dir,
                force=args.force,
                index_version=args.index_version,
                settings=settings,
            )
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
            + (
                f", {summary.context_prefixes} context prefixes generated"
                if summary.context_prefixes
                else ""
            )
            + (
                f" ({summary.context_prefixes_trimmed} trimmed to the "
                f"{CONTEXT_PREFIX_BUDGET_TOKENS}-token budget)"
                if summary.context_prefixes_trimmed
                else ""
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
