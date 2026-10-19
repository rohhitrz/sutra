"""Boundary types for generation.

# CONCEPT: A model's output is untrusted, variably shaped data. Constraining it to a schema at
# decode time (rather than asking for JSON and hoping) removes a whole class of parse failures —
# but it does not make the *content* trustworthy. The model can still cite a passage number that
# does not exist, so the schema is the first check and citation resolution is the second.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sutra.llm.types import Usage


class AnswerDraft(BaseModel):
    """Exactly what the model is asked to return.

    Deliberately flat, with only primitive fields and no defaults: provider strict-schema modes
    reject optionals, nested unions and defaults, and a schema the provider silently declines to
    enforce is worse than no schema, because nothing tells you it stopped being enforced.

    Citations are passage *numbers* from the prompt, not chunk ids. Asking a model to echo a UUID
    invites a plausible-looking wrong one; a small integer it can see in front of it is both
    easier to get right and trivially verifiable.
    """

    answer: str
    citations: list[int]
    confidence: float
    refused: bool


class Citation(BaseModel):
    """A resolved citation: a passage number the model used, mapped back to a real chunk."""

    label: int
    chunk_id: str
    doc_slug: str
    doc_title: str
    heading: str | None = None


class Answer(BaseModel):
    """What Sutra returns for one question.

    Carries the provenance an answer needs to be auditable: which prompt version produced it,
    which model, what it cost, and which citations the model invented.
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float
    refused: bool

    prompt_version: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    used_fallback: bool = False

    # # CONCEPT: Citation numbers the model emitted that did not exist in the prompt. Kept rather
    # # than silently discarded: a model citing passage 9 when it was shown five is hallucinating
    # # provenance, which is both an eval signal and a Phase 6 guardrail input. Dropping it
    # # quietly would make the answer look better than it is.
    dropped_citations: list[int] = Field(default_factory=list)

    # True when no passages were retrieved, so no model call was made at all.
    short_circuited: bool = False

    @property
    def cited_slugs(self) -> list[str]:
        """Distinct documents cited, in citation order — what the eval metrics join on."""
        seen: list[str] = []
        for citation in self.citations:
            if citation.doc_slug not in seen:
                seen.append(citation.doc_slug)
        return seen
