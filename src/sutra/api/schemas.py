"""Pydantic models for the HTTP boundary (CLAUDE.md: Pydantic models at every boundary)."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HealthComponent(BaseModel):
    """Status of one dependency."""

    name: str
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """`GET /v1/health`.

    Reports per-dependency status rather than a bare 200. A health check that returns OK while
    the database is unreachable is worse than no health check — it tells a load balancer to keep
    sending traffic to a process that cannot serve it.
    """

    status: Literal["ok", "degraded"]
    version: str
    app_env: str
    components: list[HealthComponent] = Field(default_factory=list)


class CitationOut(BaseModel):
    """One citation, resolved to a real chunk.

    `label` is the passage number the model cited, kept so an answer's "[2]" can be traced to the
    chunk it actually referred to.
    """

    label: int
    chunk_id: str
    doc_slug: str
    doc_title: str
    heading: str | None = None


class ChatRequest(BaseModel):
    """`POST /v1/chat`.

    `mode` and `stream` are in the contract from ARCHITECTURE §13 even though Phase 2 implements
    only `rag` and only non-streaming: a field that silently ignores its value is worse than one
    that refuses it, so unimplemented values are rejected rather than quietly downgraded.
    """

    query: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=128)
    # Omit to start a new conversation; pass one to continue it.
    session_id: uuid.UUID | None = None
    mode: Literal["rag", "agent"] = "rag"
    stream: bool = False


class ChatUsage(BaseModel):
    """What one turn cost."""

    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float | None = None


class ChatResponse(BaseModel):
    """One answered turn, with the provenance needed to audit it.

    `prompt_version` and `index_version` are on the response, not just in the logs, because an
    answer is only reproducible if you know which prompt and which index produced it.
    """

    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float
    refused: bool

    session_id: uuid.UUID
    message_id: uuid.UUID

    prompt_version: str
    index_version: int
    model: str
    usage: ChatUsage
    latency_ms: float

    # Chunk ids that were put in front of the model, cited or not. Keeps a retrieval miss
    # distinguishable from a generation failure after the fact.
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    # Passage numbers the model cited that referred to nothing. Surfaced rather than hidden:
    # it is the clearest single signal that an answer invented its provenance.
    dropped_citations: list[int] = Field(default_factory=list)
    # Populated from Phase 4 onward.
    trace_id: str | None = None


class FeedbackRequest(BaseModel):
    """`POST /v1/feedback`.

    At least one of `trace_id` and `message_id` must be given. A rating that points at
    nothing is unattributable, and storing it would quietly grow a table of numbers about
    no particular answer.
    """

    # # CONCEPT: Thumbs, not stars. A five-point scale invites the middle and produces a
    # # distribution nobody can act on; ±1 with an optional comment is the signal that
    # # actually arrives in production, and it is what the `feedback` table's check
    # # constraint enforces.
    rating: Literal[-1, 1]
    trace_id: str | None = Field(default=None, max_length=128)
    message_id: uuid.UUID | None = None
    comment: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _must_identify_an_answer(self) -> FeedbackRequest:
        if self.trace_id is None and self.message_id is None:
            msg = "give at least one of trace_id or message_id"
            raise ValueError(msg)
        return self


class FeedbackResponse(BaseModel):
    """What was recorded, and whether the trace was scored."""

    id: uuid.UUID
    trace_id: str | None = None
    message_id: uuid.UUID | None = None
    rating: int
    # False when tracing is off or Langfuse could not be reached. The row is still stored —
    # the database is the record of truth for feedback and Langfuse is a view onto it, so a
    # tracing outage must not lose a user's rating.
    scored: bool
