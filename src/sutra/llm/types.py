"""Boundary types for the LLM layer.

CLAUDE.md requires Pydantic models at every boundary, and an LLM response is one of the most
important: it is untrusted, variably shaped data arriving from a network service.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

# The two model roles from ADR-005. Both resolve through Settings, so which concrete model
# serves a role is configuration, never a literal in the call site.
ModelRole = Literal["fast", "strong"]


class Message(BaseModel):
    """One turn sent to the provider."""

    role: Role
    content: str


class Usage(BaseModel):
    """Token counts for one call, plus the cost we computed from them.

    # CONCEPT: Token accounting has to be captured per call, at the one place calls happen.
    # Reconstructing it later from provider invoices tells you what you spent but never which
    # request, prompt version or eval row spent it.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    # Cached input tokens are billed at a much cheaper rate, so they are tracked separately
    # rather than folded into input_tokens.
    cached_tokens: int = 0
    # Reasoning tokens are *part of* output_tokens, not additional to them, so they are not
    # added into cost separately. Recorded because they are invisible in the response text while
    # being most of what a reasoning model spends — and because they are what exhausts
    # max_output_tokens and truncates a structured response.
    reasoning_tokens: int = 0
    # None means the model is absent from the price table — an un-priced call is still a valid
    # answer, just one with unknown cost.
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """The result of one completed call, whatever path produced it."""

    text: str
    model: str
    usage: Usage
    # Populated only by structured(); the validated Pydantic instance.
    parsed: Any = None
    # True when the primary model failed and a fallback produced this answer. Worth surfacing:
    # a silent fallback changes answer quality without changing any code.
    used_fallback: bool = False
    attempts: int = 1
    latency_ms: float = 0.0
    response_id: str | None = None


class StreamDelta(BaseModel):
    """An incremental text chunk."""

    type: Literal["delta"] = "delta"
    text: str


class StreamDone(BaseModel):
    """Terminal event carrying the final usage, which arrives only at the end of a stream."""

    type: Literal["done"] = "done"
    usage: Usage
    model: str
    response_id: str | None = None


StreamEvent = StreamDelta | StreamDone


class GenerationRecord(BaseModel):
    """What the tracing hook receives for one call.

    Shaped to match what Langfuse wants in Phase 4 (model, tokens, cost, prompt version,
    latency), so wiring tracing in later is a new hook implementation rather than a refactor.
    """

    model: str
    role: ModelRole | None = None
    usage: Usage
    latency_ms: float
    attempts: int
    used_fallback: bool
    streamed: bool = False
    prompt_version: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
