"""Tests for the vector-only retrieval baseline.

Structural properties (ranking, k, slot isolation) use a stub embedder with deterministic
vectors, because they are true regardless of what the model thinks is similar. One test uses the
real model, for the one property a stub cannot speak to: that nearest actually means nearest.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence

import pytest
from sqlalchemy import delete, func, text, update

from sutra.config import TSV_CONFIG, Settings
from sutra.db.engine import get_engine, get_sessionmaker
from sutra.db.models import Chunk as ChunkRow
from sutra.ingestion.chunk import estimate_tokens
from sutra.ingestion.embed import get_embedder
from sutra.ingestion.run import CHUNKS_TABLE, ingest
from sutra.retrieval.expand import expand_to_parents
from sutra.retrieval.search import (
    _or_tsquery,
    hybrid_search,
    keyword_search,
    search,
    vector_search,
)

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


async def test_embedding_runs_off_the_event_loop(stub_index: Settings):
    """ONNX inference is synchronous and CPU-bound. Calling it directly inside a coroutine
    blocks the whole event loop, not just its own request — so `/v1/chat` serialises under
    concurrency and the eval runner's `concurrency: 4` silently means 1.

    Asserting on the thread is the durable form of this test. A timing assertion would be the
    obvious alternative and would be flaky on a loaded machine; the thread identity is exact.
    """
    import threading

    main_thread = threading.main_thread().ident
    seen: list[int | None] = []

    class ThreadRecordingEmbedder(StubEmbedder):
        def embed_query(self, text: str) -> list[float]:
            seen.append(threading.current_thread().ident)
            return super().embed_query(text)

    factory = get_sessionmaker()
    async with factory() as session:
        await vector_search(
            "where does the Line end?",
            session=session,
            settings=stub_index,
            embedder=ThreadRecordingEmbedder(),
            index_version=TEST_SLOT,
        )

    assert seen, "the embedder was never called"
    assert seen[0] != main_thread, "embed_query ran on the event loop thread"


# ---------------------------------------------------------------------------
# parent-child: parents stay out of results, children expand into them
# ---------------------------------------------------------------------------

PARENT_CHILD_SLOT = 9102


@pytest.fixture
async def parent_child_index(settings: Settings) -> AsyncIterator[Settings]:
    """A small two-parent, four-child index, written directly rather than through `ingest`.

    Writing the rows here keeps the test about expansion rather than about chunking, and lets
    the parent text be something a child demonstrably does not contain.
    """
    import uuid as uuid_module

    from sqlalchemy import insert, select

    from sutra.db.models import Document as DocumentRow

    configured = settings.model_copy(update={"index_version": PARENT_CHILD_SLOT})
    factory = get_sessionmaker()

    async def wipe() -> None:
        async with factory() as session:
            await session.execute(
                delete(ChunkRow).where(ChunkRow.index_version == PARENT_CHILD_SLOT)
            )
            await session.commit()

    await wipe()
    async with factory() as session:
        doc_id = (await session.execute(select(DocumentRow.id).limit(1))).scalar_one()
        rows = []
        chunk_no = 0
        for parent_index in range(2):
            parent_uuid = uuid_module.uuid4()
            rows.append(
                {
                    "id": parent_uuid,
                    "document_id": doc_id,
                    "index_version": PARENT_CHILD_SLOT,
                    "parent_id": None,
                    "chunk_no": chunk_no,
                    "text": f"PARENT {parent_index} containing the full surrounding context",
                    "embedding": None,
                    "token_count": 40,
                    "metadata": {"is_parent": True, "doc_slug": "test"},
                }
            )
            chunk_no += 1
            for child_index in range(2):
                rows.append(
                    {
                        "id": uuid_module.uuid4(),
                        "document_id": doc_id,
                        "index_version": PARENT_CHILD_SLOT,
                        "parent_id": parent_uuid,
                        "chunk_no": chunk_no,
                        "text": f"child {parent_index}.{child_index}",
                        "embedding": StubEmbedder()._vector(f"child {parent_index}.{child_index}"),
                        "token_count": 10,
                        "metadata": {"is_parent": False, "doc_slug": "test"},
                    }
                )
                chunk_no += 1
        await session.execute(insert(CHUNKS_TABLE), rows)
        # The tsvector is derived in Postgres during real ingestion; these rows are written
        # directly, so build it here or keyword search has nothing to match against.
        await session.execute(
            update(ChunkRow)
            .where(ChunkRow.index_version == PARENT_CHILD_SLOT)
            .values(tsv=func.to_tsvector(text(f"'{TSV_CONFIG}'"), ChunkRow.text))
        )
        await session.commit()

    yield configured
    await wipe()


async def test_parents_are_never_returned_by_search(parent_child_index: Settings):
    """Parents carry no embedding because they are context units, not search units. Without
    the `embedding IS NOT NULL` filter their NULL distance sorts into the top k on a small
    corpus, and the model is handed rows that were never actually searched."""
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "child",
            session=session,
            k=10,
            index_version=PARENT_CHILD_SLOT,
            settings=parent_child_index,
            embedder=StubEmbedder(),
        )

    assert len(hits) == 4, "expected only the four children"
    assert all(not h.text.startswith("PARENT") for h in hits)
    assert all(h.parent_id is not None for h in hits)


async def test_expansion_swaps_children_for_parents_and_dedupes(parent_child_index: Settings):
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "child",
            session=session,
            k=4,
            index_version=PARENT_CHILD_SLOT,
            settings=parent_child_index,
            embedder=StubEmbedder(),
        )
        expanded = await expand_to_parents(hits, session=session)

    # Four children, two parents: the whole point of deduping.
    assert len(expanded) == 2
    assert all(p.text.startswith("PARENT") for p in expanded)
    # Ranks stay contiguous from 1, or MRR reads a passage as further down than it was.
    assert [p.rank for p in expanded] == [1, 2]
    # A parent inherits the best rank among its children, so the order follows the evidence.
    assert expanded[0].score >= expanded[1].score


async def test_expansion_passes_through_a_flat_index(stub_index: Settings):
    """Safe to call unconditionally: a flat index has no parents, and the caller should not
    need a mode flag to know that."""
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await vector_search(
            "where does the Line end?",
            session=session,
            k=3,
            index_version=TEST_SLOT,
            settings=stub_index,
            embedder=StubEmbedder(),
        )
        expanded = await expand_to_parents(hits, session=session)

    assert [c.chunk_id for c in expanded] == [c.chunk_id for c in hits]


# ---------------------------------------------------------------------------
# keyword search and hybrid fusion (Phase 3 step 3)
# ---------------------------------------------------------------------------


def test_the_tsquery_ors_its_terms():
    """ANDed terms made keyword search return nothing for 21 of the 42 golden questions —
    a second opinion that is silent on half the eval is not a second opinion. A lexical
    retriever ORs and lets ranking sort it out; only a filter ANDs."""
    assert _or_tsquery("Who paid the toll?") == "who | paid | the | toll"
    assert _or_tsquery("waystone 17") == "waystone | 17"


def test_the_tsquery_cannot_carry_operators_through():
    """Only `\\w+` runs survive, so nothing a user types reaches the tsquery parser as syntax."""
    assert _or_tsquery("drop & table; (x | y) !z") == "drop | table | x | y | z"
    assert _or_tsquery("!!!") == ""


async def test_keyword_search_finds_an_exact_token_in_the_real_index(stub_index: Settings):
    """The property dense retrieval does not have. A stub embedder is used everywhere else in
    this file precisely because vectors are arbitrary; here the match is the real thing."""
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await keyword_search(
            "Halvern Reach", session=session, k=5, index_version=TEST_SLOT, settings=stub_index
        )

    assert hits, "an exact phrase from the corpus returned nothing"
    assert any("Halvern" in h.text for h in hits)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


async def test_keyword_search_excludes_parents(parent_child_index: Settings):
    """Parents carry a tsvector but are not search units. Without the filter, keyword search
    would return passages the vector half can never return, and fusion would be comparing two
    different candidate sets rather than two opinions about one."""
    factory = get_sessionmaker()
    async with factory() as session:
        hits = await keyword_search(
            "PARENT child",
            session=session,
            k=10,
            index_version=PARENT_CHILD_SLOT,
            settings=parent_child_index,
        )

    assert hits, "expected the children to match"
    assert all(not h.text.startswith("PARENT") for h in hits)


async def test_hybrid_returns_what_neither_half_had_alone(stub_index: Settings):
    """The fused list must be a genuine merge: drawn from both halves, ranked 1..n, and no
    longer than k."""
    factory = get_sessionmaker()
    async with factory() as session:
        query = "the Tide Clock at Halvern Reach"
        dense = await vector_search(
            query, session=session, k=20, index_version=TEST_SLOT, settings=stub_index
        )
        lexical = await keyword_search(
            query, session=session, k=20, index_version=TEST_SLOT, settings=stub_index
        )
        fused = await hybrid_search(
            query,
            session=session,
            k=5,
            index_version=TEST_SLOT,
            settings=stub_index,
            embedder=StubEmbedder(),
        )

    assert len(fused) <= 5
    assert [c.rank for c in fused] == list(range(1, len(fused) + 1))
    sources = {c.chunk_id for c in dense} | {c.chunk_id for c in lexical}
    assert {c.chunk_id for c in fused} <= sources


async def test_search_dispatches_on_the_configured_mode(stub_index: Settings):
    """One entry point so callers never branch on mode — and so a profile can ablate the
    halves against the whole by changing a yaml field."""
    factory = get_sessionmaker()
    async with factory() as session:
        for mode in ("vector", "keyword", "hybrid"):
            configured = stub_index.model_copy(
                update={"retrieval_mode": mode, "index_version": TEST_SLOT}
            )
            hits = await search(
                "Halvern Reach", session=session, k=3, settings=configured, embedder=StubEmbedder()
            )
            assert len(hits) <= 3, mode
