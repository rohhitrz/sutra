"""Tests for the vector-only retrieval baseline.

Structural properties (ranking, k, slot isolation) use a stub embedder with deterministic
vectors, because they are true regardless of what the model thinks is similar. One test uses the
real model, for the one property a stub cannot speak to: that nearest actually means nearest.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence

import pytest
from sqlalchemy import delete

from sutra.config import Settings
from sutra.db.engine import get_engine, get_sessionmaker
from sutra.db.models import Chunk as ChunkRow
from sutra.ingestion.chunk import estimate_tokens
from sutra.ingestion.embed import get_embedder
from sutra.ingestion.run import ingest
from sutra.retrieval.search import vector_search

pytestmark = pytest.mark.integration

TEST_SLOT = 9100
OTHER_SLOT = 9101


class StubEmbedder:
    """Deterministic vectors derived from the text, so ordering is stable but arbitrary.

    Good enough for every property that is about the *query*, not the model: that ranks are
    contiguous, that scores descend, that k is honoured, that slots do not leak.
    """

    @property
    def dims(self) -> int:
        return 384

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [(digest[i % len(digest)] - 128) / 128 for i in range(384)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture(autouse=True)
async def _engine_per_event_loop() -> AsyncIterator[None]:
    yield
    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture
async def stub_index(settings: Settings) -> AsyncIterator[Settings]:
    """A test slot populated with stub embeddings, removed afterwards."""

    async def wipe() -> None:
        factory = get_sessionmaker()
        async with factory() as session:
            await session.execute(
                delete(ChunkRow).where(ChunkRow.index_version.in_([TEST_SLOT, OTHER_SLOT]))
            )
            await session.commit()

    await wipe()
    await ingest(settings=settings, embedder=StubEmbedder(), index_version=TEST_SLOT)
    yield settings
    await wipe()


async def test_returns_k_results_ranked_from_one(stub_index: Settings):
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "the tide clock",
            session=session,
            k=5,
            index_version=TEST_SLOT,
            settings=stub_index,
            embedder=StubEmbedder(),
        )

    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


async def test_results_are_ordered_by_descending_similarity(stub_index: Settings):
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "waystones",
            session=session,
            k=8,
            index_version=TEST_SLOT,
            settings=stub_index,
            embedder=StubEmbedder(),
        )

    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    # Cosine similarity, not the distance Postgres returns. A score above 1 means the conversion
    # is wrong, which would silently invert nothing but confuse every report that prints it.
    assert all(-1.0 <= s <= 1.0 for s in scores)


async def test_k_is_honoured(stub_index: Settings):
    factory = get_sessionmaker()
    async with factory() as session:
        for k in (1, 3, 10):
            hits = await vector_search(
                "ledger",
                session=session,
                k=k,
                index_version=TEST_SLOT,
                settings=stub_index,
                embedder=StubEmbedder(),
            )
            assert len(hits) == k


async def test_k_defaults_to_the_configured_top_k(stub_index: Settings):
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "ledger",
            session=session,
            index_version=TEST_SLOT,
            settings=stub_index,
            embedder=StubEmbedder(),
        )
    assert len(hits) == stub_index.retrieval_top_k


async def test_search_is_scoped_to_one_index_slot(stub_index: Settings):
    """An unscoped search over a database holding two index versions would blend them, and the
    result would belong to no measurable configuration at all."""
    changed = stub_index.model_copy(update={"chunk_tokens": 250})
    await ingest(settings=changed, embedder=StubEmbedder(), index_version=OTHER_SLOT)

    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "waystones",
            session=session,
            k=20,
            index_version=TEST_SLOT,
            settings=stub_index,
            embedder=StubEmbedder(),
        )
        other = await vector_search(
            "waystones",
            session=session,
            k=20,
            index_version=OTHER_SLOT,
            settings=changed,
            embedder=StubEmbedder(),
        )

    assert {h.chunk_id for h in hits}.isdisjoint({h.chunk_id for h in other})


async def test_an_empty_slot_returns_nothing_rather_than_failing(settings: Settings):
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "anything",
            session=session,
            index_version=99_999,
            settings=settings,
            embedder=StubEmbedder(),
        )
    assert hits == []


async def test_every_hit_carries_what_citations_and_metrics_need(stub_index: Settings):
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "the fourth dial",
            session=session,
            k=3,
            index_version=TEST_SLOT,
            settings=stub_index,
            embedder=StubEmbedder(),
        )

    for hit in hits:
        # doc_slug is what the eval metrics join on; text is what quotes are matched against.
        assert hit.doc_slug and hit.doc_title and hit.doc_type
        assert hit.text.strip()
        assert hit.chunk_id


async def test_nearest_really_means_nearest(settings: Settings):
    """The one property a stub embedder cannot test.

    Uses the real model: a chunk's own text, used as the query, must retrieve that chunk first.
    This is what catches an embedding stored against the wrong row, or a distance operator that
    disagrees with the vectors.
    """
    embedder = get_embedder(settings)
    factory = get_sessionmaker()
    async with factory() as session:
        await session.execute(delete(ChunkRow).where(ChunkRow.index_version == TEST_SLOT))
        await session.commit()

    await ingest(settings=settings, embedder=embedder, index_version=TEST_SLOT)
    try:
        factory = get_sessionmaker()
        async with factory() as session:
            seed = await vector_search(
                "What is the population of Kesh Ithar?",
                session=session,
                k=1,
                index_version=TEST_SLOT,
                settings=settings,
                embedder=embedder,
            )
            assert seed, "the slot should not be empty"
            hits = await vector_search(
                seed[0].text,
                session=session,
                k=1,
                index_version=TEST_SLOT,
                settings=settings,
                embedder=embedder,
            )

        assert hits[0].chunk_id == seed[0].chunk_id
        assert hits[0].score > 0.99, "a chunk against itself should be essentially identical"
    finally:
        factory = get_sessionmaker()
        async with factory() as session:
            await session.execute(delete(ChunkRow).where(ChunkRow.index_version == TEST_SLOT))
            await session.commit()
