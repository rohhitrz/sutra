"""Reciprocal Rank Fusion.

Pure arithmetic on ranks, so these assertions are exact rather than approximate. That matters:
a fusion bug does not raise, it just makes retrieval slightly worse, which is indistinguishable
from a technique that did not help.
"""

from __future__ import annotations

from sutra.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from sutra.retrieval.search import RetrievedChunk


def _chunk(chunk_id: str, rank: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_slug="d",
        doc_title="D",
        doc_type="wiki",
        chunk_no=abs(hash(chunk_id)) % 1000,
        heading=None,
        text=f"chunk {chunk_id}",
        token_count=10,
        score=score,
        rank=rank,
    )


def test_agreement_beats_a_single_confident_vote():
    """The property the whole technique exists for. A chunk both retrievers rank third is
    worth more than one that tops a single list and is absent from the other — which is what
    makes fusion robust to either retriever being badly wrong on a given query."""
    dense = [_chunk("1", 1), _chunk("2", 2), _chunk("3", 3)]
    lexical = [_chunk("9", 1), _chunk("8", 2), _chunk("3", 3)]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0].chunk_id == "3"
    assert fused[0].score == 2 / (DEFAULT_RRF_K + 3)


def test_scores_are_ranks_not_the_retrievers_scores():
    """Cosine similarity and ts_rank_cd are not comparable numbers — one is bounded and
    clustered, the other unbounded. Fusion must not let either leak into the result."""
    dense = [_chunk("1", 1, score=0.93)]
    lexical = [_chunk("2", 1, score=0.0001)]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert {c.score for c in fused} == {1 / (DEFAULT_RRF_K + 1)}
    assert [c.rank for c in fused] == [1, 2]


def test_the_same_chunk_from_both_lists_is_one_entry():
    dense = [_chunk("1", 1), _chunk("2", 2)]
    lexical = [_chunk("1", 1), _chunk("3", 2)]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert [c.chunk_id for c in fused] == ["1", "2", "3"]
    assert len({c.chunk_id for c in fused}) == 3


def test_k_is_the_dial_between_one_strong_vote_and_consensus():
    """Low k trusts an individual retriever's ordering; high k trusts agreement.

    `solo` tops one list and is absent from the other. `pair` is fourth in both. The two
    orderings are genuinely different answers to "which is better evidence", and k is the
    only thing that decides — which is why it is config and not a constant.
    """
    first = [_chunk("solo", 1), _chunk("a", 2), _chunk("b", 3), _chunk("pair", 4)]
    second = [_chunk("c", 1), _chunk("d", 2), _chunk("e", 3), _chunk("pair", 4)]

    # Asserted as relative order rather than "who is first", so the result does not depend on
    # how the other four chunks happen to tie against each other.
    low = [c.chunk_id for c in reciprocal_rank_fusion([first, second], k=1)]
    high = [c.chunk_id for c in reciprocal_rank_fusion([first, second], k=60)]

    # k=1: solo scores 1/2, pair scores 1/5 + 1/5 = 0.4. One confident vote wins.
    assert low.index("solo") < low.index("pair")
    # k=60: solo scores 1/61 ≈ 0.0164, pair scores 2/64 ≈ 0.0313. Consensus wins.
    assert high.index("pair") < high.index("solo")


def test_position_in_the_list_is_what_counts_not_the_carried_rank():
    """A caller may hand us a truncated or re-sorted list. The rank that matters is the one
    within the list actually being fused, or a sliced list would score as though its entries
    were still deep in the original."""
    truncated = [_chunk("x", rank=17)]
    fused = reciprocal_rank_fusion([truncated])
    assert fused[0].score == 1 / (DEFAULT_RRF_K + 1)


def test_ties_break_deterministically():
    """Two chunks on identical scores must not swap between runs: an eval harness has to be
    reproducible before any of its numbers can be believed."""
    a = reciprocal_rank_fusion([[_chunk("b", 1)], [_chunk("a", 1)]])
    b = reciprocal_rank_fusion([[_chunk("a", 1)], [_chunk("b", 1)]])
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b] == ["a", "b"]


def test_limit_cuts_after_fusing_not_before():
    dense = [_chunk(str(i), i) for i in range(1, 6)]
    lexical = [_chunk("9", 1)]
    fused = reciprocal_rank_fusion([dense, lexical], limit=2)
    assert len(fused) == 2
    assert [c.rank for c in fused] == [1, 2]


def test_an_empty_list_contributes_nothing():
    """Keyword search legitimately returns nothing for some queries; that must degrade to the
    other retriever's ordering rather than blowing up."""
    dense = [_chunk("1", 1), _chunk("2", 2)]
    assert [c.chunk_id for c in reciprocal_rank_fusion([dense, []])] == ["1", "2"]
    assert reciprocal_rank_fusion([[], []]) == []
