"""Ingestion against a real Postgres (ADR-021's three behaviours).

Marked `integration` and run with `-m 'not integration'` excluded in CI until Phase 9 wires a
service container. These use a throwaway index slot so they never disturb the slot an eval
baseline is pinned to — which is the very property they are testing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select, update

from sutra.config import Settings
from sutra.db.engine import get_engine, get_sessionmaker
from sutra.db.models import Chunk as ChunkRow
from sutra.ingestion.chunk import estimate_tokens
from sutra.ingestion.fingerprint import build_fingerprint
from sutra.ingestion.parse import parse_corpus
from sutra.ingestion.run import IngestionBlocked, IngestionSummary, ingest

pytestmark = pytest.mark.integration

# Far from any real slot, so a failed test can never leave slot 1 in a surprising state.
TEST_SLOT = 9001


class StubEmbedder:
    """Deterministic fake. The point of these tests is the upsert and fingerprint logic, and
    loading a real ONNX model would make them slow without testing anything extra."""

    @property
    def dims(self) -> int:
        return 384

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 383 + [1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 383 + [1.0]


@pytest.fixture(autouse=True)
async def _engine_per_event_loop() -> AsyncIterator[None]:
    """Dispose the cached engine after each test.

    `get_engine` is `lru_cache`d, one engine per process, and its pooled connections belong to
    the event loop that opened them. pytest-asyncio gives each test a fresh loop, so a reused
    engine hands the next test connections tied to a loop that is already closed.
    """
    yield
    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture
async def clean_slot() -> AsyncIterator[None]:
    async def wipe() -> None:
        factory = get_sessionmaker()
        async with factory() as session:
            await session.execute(delete(ChunkRow).where(ChunkRow.index_version == TEST_SLOT))
            await session.commit()

    await wipe()
    yield
    await wipe()


async def _chunk_count(slot: int) -> int:
    factory = get_sessionmaker()
    async with factory() as session:
        stmt = select(func.count()).select_from(ChunkRow).where(ChunkRow.index_version == slot)
        return (await session.execute(stmt)).scalar_one()


async def _ingest(settings: Settings, **kwargs: object) -> IngestionSummary:
    return await ingest(
        settings=settings,
        embedder=StubEmbedder(),
        index_version=TEST_SLOT,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_first_run_ingests_the_whole_corpus(settings: Settings, clean_slot: None):
    summary = await _ingest(settings)
    assert summary.status == "ingested"
    assert summary.documents == 10
    assert summary.chunks > 0
    assert await _chunk_count(TEST_SLOT) == summary.chunks


async def test_re_running_unchanged_is_a_no_op(settings: Settings, clean_slot: None):
    first = await _ingest(settings)
    second = await _ingest(settings)
    assert second.status == "unchanged"
    assert second.chunks == 0
    # The data is still there — "unchanged" means "nothing to do", not "nothing present".
    assert await _chunk_count(TEST_SLOT) == first.chunks


async def test_different_parameters_into_an_occupied_slot_are_refused(
    settings: Settings, clean_slot: None
):
    """The property that protects an eval baseline: re-chunking a pinned slot must fail loudly
    rather than quietly replace the index the baseline was measured on."""
    first = await _ingest(settings)
    changed = settings.model_copy(update={"chunk_tokens": 250})

    with pytest.raises(IngestionBlocked) as excinfo:
        await _ingest(changed)

    message = str(excinfo.value)
    assert "chunk_tokens: 400 -> 250" in message, "the error must name what changed"
    assert "--force" in message
    assert await _chunk_count(TEST_SLOT) == first.chunks, "the slot must be untouched"


async def test_force_replaces_an_occupied_slot(settings: Settings, clean_slot: None):
    first = await _ingest(settings)
    changed = settings.model_copy(update={"chunk_tokens": 250})
    summary = await _ingest(changed, force=True)

    assert summary.status == "ingested"
    assert summary.fingerprint != first.fingerprint
    # Replaced, not appended — otherwise the slot would hold two overlapping indexes.
    assert await _chunk_count(TEST_SLOT) == summary.chunks


async def test_the_stored_fingerprint_round_trips(settings: Settings, clean_slot: None):
    """The regression guard for the silently-dropped `metadata` key: if the fingerprint does not
    survive the insert, idempotency and the mismatch guard both quietly stop working."""
    summary = await _ingest(settings)
    factory = get_sessionmaker()
    async with factory() as session:
        stmt = (
            select(ChunkRow.meta["index_fingerprint"])
            .where(ChunkRow.index_version == TEST_SLOT)
            .limit(1)
        )
        stored = (await session.execute(stmt)).scalar_one()

    assert stored is not None, "metadata was dropped on insert"
    assert stored["chunk_tokens"] == settings.chunk_tokens
    assert stored["embedding_model"] == settings.embedding_model
    assert summary.fingerprint


async def test_a_fingerprint_from_an_older_schema_is_refused_not_crashed(
    settings: Settings, clean_slot: None
):
    """The fingerprint is persisted data, so its schema outlives the code that wrote it.

    ADR-022 added `chunker_version`; every slot written before it holds a dict that no longer
    validates. Reading one must refuse cleanly, because a ValidationError traceback out of
    `make ingest` tells you nothing about what to do next.
    """
    first = await _ingest(settings)

    factory = get_sessionmaker()
    async with factory() as session:
        legacy = {
            k: v
            for k, v in build_fingerprint(settings, parse_corpus(Path("data/corpus")))
            .model_dump()
            .items()
            if k != "chunker_version"
        }
        await session.execute(
            update(ChunkRow)
            .where(ChunkRow.index_version == TEST_SLOT)
            .values(meta={"index_fingerprint": legacy})
        )
        await session.commit()

    with pytest.raises(IngestionBlocked, match=r"older schema.*chunker_version"):
        await _ingest(settings)
    assert await _chunk_count(TEST_SLOT) == first.chunks, "the slot must be untouched"

    # ...and --force is the documented way out.
    forced = await _ingest(settings, force=True)
    assert forced.status == "ingested"


async def test_tsvector_and_embedding_are_populated(settings: Settings, clean_slot: None):
    """Phase 2 retrieval is vector-only, but tsv is built now because Phase 3's hybrid search
    needs it and backfilling it later means a full re-ingest."""
    await _ingest(settings)
    factory = get_sessionmaker()
    async with factory() as session:
        stmt = select(
            func.count(),
            func.count(ChunkRow.embedding),
            func.count(ChunkRow.tsv),
        ).where(ChunkRow.index_version == TEST_SLOT)
        total, embedded, tsv = (await session.execute(stmt)).one()

    assert total == embedded == tsv, "every chunk needs both an embedding and a tsvector"
