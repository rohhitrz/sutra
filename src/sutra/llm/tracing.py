"""The tracing hook the LLM client calls.

The protocol lives here, next to its one caller, so that what the client *needs* from a
tracer is stated where the client is. The implementations live in
`sutra.observability.tracing`: `LangfuseTracer` (ADR-007) and a `NoOpTracer` of its own.

`LLMClient` imports `observability.tracing.get_tracer` to default its tracer — a client that
had to be handed one would report no generations whenever a call site forgot, and a trace
silently missing its tokens and cost is the hardest kind of gap to notice. There is no import
cycle: `observability.tracing` depends only on `llm.types`, never on the client.

The `NoOpTracer` here is kept as the explicit "trace nothing" choice for tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from sutra.llm.types import GenerationRecord

logger = logging.getLogger(__name__)


@runtime_checkable
class Tracer(Protocol):
    """Receives one record per LLM call, successful or not."""

    def on_generation(self, record: GenerationRecord) -> None: ...

    def stage(self, name: str, *, kind: Any = ..., input: Any = ..., metadata: Any = ...) -> Any:
        """Open a span. The client uses this to wrap each call in a `generation` observation.

        # CONCEPT: A generation has to be a span *around* the provider call, not a record
        # filed after it. Reporting it afterwards gives you the tokens and the cost but a
        # zero-width span, so the one question a trace exists to answer — which stage spent
        # the time — is the one it cannot. The client therefore opens the span, makes the
        # call, and reports the record from inside it.
        """
        ...


class NoOpTracer:
    """Does nothing but log at DEBUG. Used whenever tracing is off or unconfigured."""

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

    @contextmanager
    def stage(
        self, name: str, *, kind: Any = "span", input: Any = None, metadata: Any = None
    ) -> Iterator[None]:
        yield None
