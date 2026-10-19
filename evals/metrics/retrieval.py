"""Retrieval metrics, hand-written (ADR-008).

# CONCEPT: Ground truth here is 72 verbatim quotes tied to documents, not chunk ids — chunk ids
# change on every re-ingest, so a dataset built on them would expire. A retrieved chunk therefore
# "contains" a piece of evidence when the quote appears in its text, whitespace-normalised.
#
# `make coverage` is the precondition for all of this: if the chunker splits a cited quote across
# a boundary, no chunk can contain it, and every number below understates retrieval for a reason
# that has nothing to do with the retriever.
#
# Three numbers, measuring three different things:
#   recall@k           — of the evidence this question needs, how much did we fetch?
#   MRR                — how near the top was the first useful chunk?
#   context precision  — how much of what we fetched was useful at all?
#
# Recall and precision pull in opposite directions as k grows, which is the point of reporting
# both: raising k always helps recall and always costs precision and tokens.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel


class EvidenceRef(BaseModel):
    """One cited quote from the golden dataset."""

    doc_slug: str
    quote: str


class RetrievedRef(BaseModel):
    """One retrieved chunk, reduced to what the metrics need."""

    doc_slug: str
    text: str


def normalise(text: str) -> str:
    """Collapse whitespace runs. The corpus is hard-wrapped markdown, so a correct quote still
    carries the newline where the source line broke."""
    return re.sub(r"\s+", " ", text).strip()


def contains_evidence(chunk: RetrievedRef, evidence: EvidenceRef) -> bool:
    """Does this chunk hold this piece of evidence?

    The document has to match as well as the text. Two documents can share a sentence — the
    registry and the timeline both record dates — and crediting the wrong document would score a
    retrieval that found the right words in the wrong place as a success.
    """
    return chunk.doc_slug == evidence.doc_slug and normalise(evidence.quote) in normalise(
        chunk.text
    )


class RowRetrievalMetrics(BaseModel):
    """Retrieval metrics for one question."""

    k: int
    evidence_total: int
    evidence_found: int
    # The 1-based rank of the first chunk holding any evidence; None when nothing did.
    first_relevant_rank: int | None
    relevant_chunks: int

    @property
    def recall(self) -> float:
        """Of the evidence this question needs, the fraction retrieved.

        Quote-level rather than chunk-level: a question citing five quotes that lands only one
        has not been served, even though one chunk came back "relevant".
        """
        if self.evidence_total == 0:
            return 1.0
        return self.evidence_found / self.evidence_total

    @property
    def hit(self) -> bool:
        """Did anything relevant come back at all? The floor that recall sits above."""
        return self.first_relevant_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first relevant chunk, 0 when there is none. Averaged, this is MRR.

        Rewards putting the answer first rather than fifth, which recall@k cannot see: a gold
        chunk at rank 1 and at rank 5 score identically on recall@5 and very differently here.
        """
        if self.first_relevant_rank is None:
            return 0.0
        return 1.0 / self.first_relevant_rank

    @property
    def context_precision(self) -> float:
        """The fraction of retrieved chunks that held evidence.

        The cost side of retrieval: every chunk is context the model pays for and has to read
        past. 1 relevant chunk in 5 means 80% of the context was noise.
        """
        if self.k == 0:
            return 0.0
        return self.relevant_chunks / self.k


def score_row(
    evidence: Sequence[EvidenceRef], retrieved: Sequence[RetrievedRef]
) -> RowRetrievalMetrics:
    """Score one question's retrieval against its cited evidence.

    `retrieved` must be in rank order, best first.
    """
    found = sum(1 for ref in evidence if any(contains_evidence(c, ref) for c in retrieved))
    relevant_positions = [
        position
        for position, chunk in enumerate(retrieved, start=1)
        if any(contains_evidence(chunk, ref) for ref in evidence)
    ]
    return RowRetrievalMetrics(
        k=len(retrieved),
        evidence_total=len(evidence),
        evidence_found=found,
        first_relevant_rank=relevant_positions[0] if relevant_positions else None,
        relevant_chunks=len(relevant_positions),
    )


class RetrievalSummary(BaseModel):
    """Aggregated retrieval metrics over a set of rows."""

    rows: int
    k: int
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    context_precision: float


def summarise(rows: Sequence[RowRetrievalMetrics], *, k: int) -> RetrievalSummary:
    """Macro-average across rows.

    Macro, not micro: every question counts once, so a question citing five quotes does not
    outweigh four questions citing one. Micro-averaging would let the handful of multi-hop rows
    dominate the headline number.
    """
    if not rows:
        return RetrievalSummary(
            rows=0, k=k, recall_at_k=0.0, hit_rate_at_k=0.0, mrr=0.0, context_precision=0.0
        )
    count = len(rows)
    return RetrievalSummary(
        rows=count,
        k=k,
        recall_at_k=sum(r.recall for r in rows) / count,
        hit_rate_at_k=sum(1 for r in rows if r.hit) / count,
        mrr=sum(r.reciprocal_rank for r in rows) / count,
        context_precision=sum(r.context_precision for r in rows) / count,
    )
