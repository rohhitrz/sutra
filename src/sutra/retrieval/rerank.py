"""Cross-encoder reranking (ADR-004, ARCHITECTURE §4 step 5).

# CONCEPT: A bi-encoder and a cross-encoder answer the same question with different budgets.
# The embedder compresses a chunk into 384 numbers *before it has seen the query*, which is
# what makes search over the whole corpus possible — one vector per chunk, computed once. The
# price is that the comparison happens between two summaries. "What did Dask pay at the
# nineteenth?" and a passage about tolls at the twenty-first look alike from a distance.
#
# A cross-encoder reads the query and the passage *together*, with attention running across
# both, and emits a single relevance score. It is far more accurate and completely unusable
# as a retriever: scoring the whole corpus means one forward pass per chunk per query.
#
# So they compose. Retrieve cheaply and widely with vectors and FTS, fuse, then spend the
# expensive model on the shortlist. That is the entire architecture of a modern retrieval
# stack, and it is why step 3 had to come before step 4 — a reranker can only reorder what
# retrieval already found, so its ceiling is the recall of the pool it is handed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sutra.config import Settings, get_settings
from sutra.observability.tracing import stage
from sutra.retrieval.search import RetrievedChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """What retrieval needs from a cross-encoder.

    Behind an interface for the same reason the embedder is (ADR-004): the model is the part
    most likely to be swapped, and an interface is the difference between changing one class
    and changing every call site.
    """

    def scores(self, query: str, passages: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """`BAAI/bge-reranker-base` over ONNX Runtime.

    Lazy, like the embedder: constructing this is free and the model loads on first use.
    Importing a module should never cost a model load.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("loading reranker %s", self._settings.reranker_model)
            self._model = TextCrossEncoder(
                model_name=self._settings.reranker_model,
                cache_dir=self._settings.model_cache_dir,
            )
        return self._model

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to the query. Higher is better.

        These are raw logits, not probabilities and not comparable between queries. That is
        fine and deliberate: reranking only ever sorts one query's shortlist, and calibrating
        them into something cross-comparable would invite reading a threshold into a number
        that does not have one.
        """
        if not passages:
            return []
        model = self._ensure_model()
        return [float(s) for s in model.rerank(query, list(passages))]  # type: ignore[attr-defined]


def get_reranker(settings: Settings | None = None) -> Reranker:
    return FastEmbedReranker(settings)


async def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    top_k: int,
    reranker: Reranker | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Re-score a shortlist with the cross-encoder and keep the best `top_k`.

    Runs **off the event loop**. A cross-encoder is the heaviest thing in this pipeline —
    twenty forward passes over query-and-passage pairs, against the embedder's one over a
    short string — so blocking the loop here would serialise every concurrent request behind
    it. `vector_search` was moved into a thread for the same reason before this step existed,
    which is why the fix was made then rather than now.

    `score` becomes the cross-encoder's logit and `rank` the new position. The fusion scores
    they replace are not kept: two scores on one object invite reading the wrong one, and the
    whole point of reranking is that the earlier ordering was the less trustworthy of the two.
    """
    if not chunks:
        return []
    settings = settings or get_settings()
    reranker = reranker or get_reranker(settings)

    # The reranker is shown the text as it was *indexed*, prefix included. Scoring the bare
    # chunk would give the cross-encoder less context than the retriever had — measured, and
    # it costs real accuracy on exactly the categories step 2 fixed.
    from sutra.ingestion.contextualize import contextualised_text

    passages = [contextualised_text(c.context_prefix, c.text) for c in chunks]
    with stage(
        "rerank",
        kind="span",
        input={"query": query, "candidates": len(chunks)},
        metadata={"model": settings.reranker_model, "top_k": top_k},
    ) as span:
        scores = await asyncio.to_thread(reranker.scores, query, passages)
        scored = sorted(
            zip(chunks, scores, strict=True),
            # `-score` then `chunk_id` so equal scores order deterministically. An eval harness
            # has to be reproducible before any of its numbers can be believed.
            key=lambda pair: (-pair[1], pair[0].chunk_id),
        )
        kept = [
            chunk.model_copy(update={"score": score, "rank": rank})
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]
        # How far the reranker moved things is the measure of whether it did anything. A
        # rerank that returns the fusion order unchanged cost twenty forward passes for
        # nothing, and that is invisible in a latency number alone.
        before = {c.chunk_id: i for i, c in enumerate(chunks)}
        span.update(
            output={"kept": len(kept), "top_score": kept[0].score if kept else None},
            metadata={
                "max_rank_shift": max(
                    (abs(before[c.chunk_id] - i) for i, c in enumerate(kept)), default=0
                ),
                "promoted_from_outside_top_k": sum(1 for c in kept if before[c.chunk_id] >= top_k),
            },
        )
    return kept
