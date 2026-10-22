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
    # Populated only for a schema-constrained stream: the provider's validated instance, so a
    # streamed structured answer is parsed exactly like a non-streamed one instead of being
    # re-parsed from the text we happened to accumulate.
    parsed: Any = None
    # The full accumulated output. For a schema-constrained stream this is the JSON document,
    # which is what lets a truncated response still yield a partial answer when `parsed` is
    # absent (a dropped connection, or an output budget that ran out mid-object).
    text: str = ""


StreamEvent = StreamDelta | StreamDone


class GenerationRecord(BaseModel):
    """What the tracing hook receives for one call.

    Shaped to match what Langfuse wants in Phase 4 (model, tokens, cost, prompt version,
    latency), so wiring tracing in later is a new hook implementation rather than a refactor.
    """

    model: str
    role: ModelRole | None = None
    # Which pipeline stage made this call (`answer`, `rewrite`, `route`, `summarize`, ...).
    # The span is already named by it; the record carries it too so the *role split* is
    # auditable from the record alone — "the answer ran strong and the rewrite ran fast" is a
    # claim about routing, and a record naming only the model cannot settle it when both roles
    # resolve to the same model (ADR-017).
    stage: str | None = None
    usage: Usage
    latency_ms: float
    attempts: int
    used_fallback: bool
    streamed: bool = False
    prompt_version: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # What was sent and what came back. Everything else here describes the call; these two are
    # the call. A trace without them can say a generation cost $0.0004 and took 1.2s but not
    # what it was asked or what it answered, which is the question you opened the trace with.
    # Both are gated by `TRACING_CAPTURE_CONTENT` in the tracer, not here: the record is the
    # full truth about a call and the backend decides what may leave the process.
    input: Any = None
    output: Any = None
