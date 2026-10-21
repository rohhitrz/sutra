"""Reciprocal Rank Fusion (ADR-002, ARCHITECTURE §4 step 4).

# CONCEPT: Reciprocal Rank Fusion — merges ranked lists without score calibration.
#
# The problem it solves is that the two retrievers do not speak the same language. Cosine
# similarity lands in [-1, 1] and clusters tightly: on this corpus the best and fifth-best
# vector hits are routinely 0.72 and 0.68. `ts_rank_cd` is an unbounded cover-density score
# whose magnitude depends on how many query terms matched and how close together they fell —
# 0.09 and 0.0001 are both ordinary values. Adding or averaging those two numbers is
# meaningless, and normalising them per query means a list whose top hit is bad gets the same
# 1.0 as a list whose top hit is perfect.
#
# RRF throws the scores away and keeps only the ranks:
#
#     score(d) = sum over retrievers of  1 / (k + rank_of_d_in_that_retriever)
#
# A document ranked 1st by one retriever and absent from the other scores 1/61. A document
# ranked 3rd by *both* scores 2/63 — more. That is the whole behaviour worth understanding:
# **agreement between retrievers outranks a single confident vote.** It is what makes the
# fusion robust to one retriever being badly wrong on a particular query, which is exactly
# what happens when a question contains a rare proper noun (dense fails) or is a pure
# paraphrase with no shared terms (lexical fails).
#
# The constant k (60 by convention, from Cormack et al. 2009) flattens the curve. At k=0 the
# top rank would be worth 1.0 and the second 0.5, so first place would dominate everything;
# at k=60 ranks 1 and 2 are 0.0164 and 0.0161, nearly equal, and what separates documents is
# how many lists they appear in rather than where they sit in any one. Lower k trusts
# individual rankings more; higher k trusts consensus more.
"""

from __future__ import annotations

from collections.abc import Sequence

from sutra.retrieval.search import RetrievedChunk

# Cormack et al. (2009). Not tuned here: with two retrievers and a 42-row eval there is not
# enough signal to fit it without overfitting, and the published default is a real one.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, by reciprocal rank.

    Pure: no database, no config, no clock. Everything above is arithmetic on ranks, so it can
    be tested exactly rather than approximately — which matters, because a fusion bug shows up
    as "retrieval got slightly worse" and nothing else.

    Each chunk's `score` in the result is its RRF score, and `rank` is its position in the
    fused list. The original per-retriever scores are deliberately dropped: keeping one of
    them would invite reading it as the fused score, and they are not comparable, which is the
    entire reason this function exists.

    A chunk's identity is `chunk_id`, so the same row retrieved by both halves is one entry
    with two contributions, not two entries.
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for position, chunk in enumerate(ranked, start=1):
            # `position`, not `chunk.rank`: the caller may hand us a truncated or re-ordered
            # list, and the rank that matters is the one within the list actually being fused.
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + position)
            best.setdefault(chunk.chunk_id, chunk)

    # Sorted by score, then by chunk_id to break ties deterministically. Without the tiebreak,
    # two chunks with identical RRF scores could swap places between runs and move MRR by a
    # few points for no reason at all — an eval harness has to be reproducible before it can
    # be believed.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[:limit]

    return [
        best[chunk_id].model_copy(update={"score": score, "rank": rank})
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]
