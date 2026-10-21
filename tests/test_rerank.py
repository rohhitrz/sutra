"""Cross-encoder reranking (Phase 3 step 4).

The model itself is stubbed. What is worth testing is the ordering contract, the shortlist
size, and that the heavy call leaves the event loop — not whether bge-reranker-base has good
judgement, which only the eval can say.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from sutra.retrieval.rerank import Reranker, rerank
from sutra.retrieval.search import RetrievedChunk


class StubReranker:
    """Scores by position of a marker in the text, so the expected order is obvious."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores_by_text = scores or {}
        self.threads: list[int | None] = []

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        self.threads.append(threading.current_thread().ident)
        return [self.scores_by_text.get(p, 0.0) for p in passages]


def _chunk(chunk_id: str, rank: int, text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_slug="d",
        doc_title="D",
        doc_type="wiki",
        chunk_no=rank,
        heading=None,
        text=text or f"passage {chunk_id}",
        token_count=10,
        score=0.5,
        rank=rank,
    )


async def test_reranking_overrides_the_retrieval_order():
    """The whole point: the cross-encoder has read the query against the passage, and the
    retriever has not. Its ordering wins or there was no reason to pay for it."""
    chunks = [_chunk("a", 1), _chunk("b", 2), _chunk("c", 3)]
    stub = StubReranker({"passage a": -9.0, "passage b": -1.0, "passage c": -5.0})

    out = await rerank("q", chunks, top_k=3, reranker=stub)

    assert [c.chunk_id for c in out] == ["b", "c", "a"]
    assert [c.rank for c in out] == [1, 2, 3]
    assert [c.score for c in out] == [-1.0, -5.0, -9.0]


async def test_only_the_top_k_survive():
    chunks = [_chunk(str(i), i) for i in range(10)]
    out = await rerank("q", chunks, top_k=3, reranker=StubReranker())
    assert len(out) == 3


async def test_a_chunk_outside_the_retrieved_top_k_can_be_promoted():
    """The reason the shortlist is 20 and not 5. Handing the reranker the final five would
    let it sort five and discover nothing."""
    chunks = [_chunk(str(i), i) for i in range(20)]
    stub = StubReranker({"passage 17": 5.0})

    out = await rerank("q", chunks, top_k=5, reranker=stub)

    assert out[0].chunk_id == "17"


async def test_scoring_runs_off_the_event_loop():
    """Twenty query-and-passage forward passes is the heaviest thing in the pipeline. Doing
    it inline would block every other request, not just this one."""
    stub = StubReranker()
    await rerank("q", [_chunk("a", 1)], top_k=1, reranker=stub)

    assert stub.threads and stub.threads[0] != threading.main_thread().ident


async def test_ties_break_deterministically():
    """Equal scores must not reorder between runs; a harness has to be reproducible before
    its numbers mean anything."""
    first = await rerank("q", [_chunk("b", 1), _chunk("a", 2)], top_k=2, reranker=StubReranker())
    second = await rerank("q", [_chunk("a", 1), _chunk("b", 2)], top_k=2, reranker=StubReranker())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second] == ["a", "b"]


async def test_an_empty_shortlist_is_not_an_error():
    """Keyword-only retrieval can legitimately return nothing for a query."""
    assert await rerank("q", [], top_k=5, reranker=StubReranker()) == []


def test_the_stub_satisfies_the_protocol():
    """The interface exists so the model can be swapped (ADR-004); a stub that does not
    satisfy it would make these tests prove nothing about the real call site."""
    assert isinstance(StubReranker(), Reranker)
