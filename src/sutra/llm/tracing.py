"""Tracing hook.

Phase 0 ships only a no-op. Phase 4 replaces it with a Langfuse-backed implementation (ADR-007).
The hook exists now so that every call site is already instrumented and Phase 4 is a swap
rather than a hunt through the codebase for call sites to annotate.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from sutra.llm.types import GenerationRecord

logger = logging.getLogger(__name__)


@runtime_checkable
class Tracer(Protocol):
    """Receives one record per LLM call, successful or not."""

    def on_generation(self, record: GenerationRecord) -> None: ...


class NoOpTracer:
    """Does nothing but log at DEBUG. The default until Phase 4."""

    def on_generation(self, record: GenerationRecord) -> None:
        logger.debug(
            "llm.generation model=%s tokens=%d/%d cost=%s latency=%.0fms attempts=%d fallback=%s",
            record.model,
            record.usage.input_tokens,
            record.usage.output_tokens,
            f"${record.usage.cost_usd:.6f}" if record.usage.cost_usd is not None else "unknown",
            record.latency_ms,
            record.attempts,
            record.used_fallback,
        )
