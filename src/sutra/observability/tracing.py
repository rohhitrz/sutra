"""Langfuse-backed tracing (ADR-007, ADR-023).

One trace per request; one span per pipeline stage (ARCHITECTURE §2, §10). The Langfuse SDK
is OpenTelemetry-based, so nesting comes from OTel's own context: a span opened inside
another span's `with` block becomes its child without being handed a parent. That is the same
contextvar mechanism `sutra.observability.context` uses, which is why neither a parent span
nor a trace id has to be threaded through retrieval's call signatures.

# CONCEPT: Observability must not be able to break the thing it observes. Every call here is
# wrapped so that a Langfuse outage, a bad key or a serialisation failure degrades to a log
# line rather than a failed request. A tracing system that can take production down is a
# liability, and the temptation to let it raise "so you notice" is how that happens.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, Literal, Protocol, runtime_checkable

from sutra.config import Settings, get_settings
from sutra.llm.types import GenerationRecord
from sutra.observability.context import bind_trace_id, get_trace_id

logger = logging.getLogger(__name__)

# Langfuse's observation types. `retriever` and `guardrail` are not decoration: the UI groups
# and filters on them, so a search stage typed as a plain span is harder to find later.
StageKind = Literal["span", "retriever", "generation", "tool", "agent", "chain", "guardrail"]


class Stage(Protocol):
    """A span that is open. `update` attaches whatever is only known at the end."""

    def update(
        self, *, output: Any = None, metadata: dict[str, Any] | None = None, **kwargs: Any
    ) -> None: ...


@runtime_checkable
class SutraTracer(Protocol):
    """What the pipeline needs from a tracing backend.

    Deliberately smaller than the Langfuse SDK's surface: the point of an interface here is
    that `NoOpTracer` is a complete implementation in a dozen lines, so nothing in the
    pipeline has to care whether tracing is switched on.
    """

    def request(
        self,
        *,
        name: str,
        user_id: str | None,
        session_id: str | None,
        input: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def stage(
        self,
        name: str,
        *,
        kind: StageKind = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def event(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] = "DEFAULT",
    ) -> None: ...

    def on_generation(self, record: GenerationRecord) -> None: ...

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
    ) -> bool:
        """Attach a score to a trace. Returns whether it actually reached a backend.

        A bool rather than None because the caller has something to tell the user: feedback
        is stored in Postgres first and scored second, so "recorded but not scored" is a
        real and reportable state rather than a failure.
        """
        ...

    def trace_url(self, trace_id: str) -> str | None: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


class _NullStage:
    """A `Stage` that discards everything."""

    def update(
        self, *, output: Any = None, metadata: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        return None


class NoOpTracer:
    """The implementation used when tracing is off or unconfigured.

    It still assigns a trace id, so `trace_id` stays a working correlation key in the logs
    and on the response when there is no Langfuse to talk to.
    """

    @contextmanager
    def request(
        self,
        *,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[_NullStage]:
        yield _NullStage()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        kind: StageKind = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[_NullStage]:
        yield _NullStage()

    def event(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] = "DEFAULT",
    ) -> None:
        logger.debug("trace.event %s", name, extra={"event_name": name, **(metadata or {})})

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

    def score(
        self, *, trace_id: str, name: str, value: float | str, comment: str | None = None
    ) -> bool:
        logger.debug("trace.score trace=%s %s=%s", trace_id, name, value)
        return False

    def trace_url(self, trace_id: str) -> str | None:
        return None

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class LangfuseTracer:
    """Sends traces to a Langfuse instance."""

    def __init__(self, settings: Settings | None = None) -> None:
        from langfuse import Langfuse

        self._settings = settings or get_settings()
        self._capture_content = self._settings.tracing_capture_content
        self._client = Langfuse(
            public_key=self._settings.langfuse_public_key,
            secret_key=self._settings.langfuse_secret_key,
            host=self._settings.langfuse_host,
            environment=self._settings.app_env,
        )

    # ------------------------------------------------------------------ helpers

    def _payload(self, value: Any) -> Any:
        """Drop span inputs and outputs when content capture is off (see `Settings`)."""
        return value if self._capture_content else None

    # ------------------------------------------------------------------- public

    @contextmanager
    def request(
        self,
        *,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """The root span for one request, and the trace-level attributes that go with it.

        `propagate_attributes` is what puts `user_id`, `session_id` and tags on the *trace*
        rather than on this one span, which is what makes "every trace for this session" and
        "every trace from this eval run" answerable in the UI.
        """
        from langfuse import propagate_attributes

        try:
            with propagate_attributes(
                user_id=user_id, session_id=session_id, tags=tags, trace_name=name
            ):
                with self._client.start_as_current_observation(
                    name=name, as_type="span", input=self._payload(input), metadata=metadata
                ) as span:
                    with bind_trace_id(self._client.get_current_trace_id()):
                        yield span
        except Exception:
            # The request itself must survive a broken tracer. Note this catches failures in
            # *tracing*; an exception from the wrapped body propagates through the inner
            # `with` and is re-raised after the span is closed and marked.
            logger.warning("tracing.request_failed name=%s", name, exc_info=True)
            yield _NullStage()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        kind: StageKind = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """One pipeline stage, as a child of whatever span is currently open."""
        try:
            cm = self._client.start_as_current_observation(
                name=name, as_type=kind, input=self._payload(input), metadata=metadata
            )
        except Exception:
            logger.warning("tracing.stage_failed name=%s", name, exc_info=True)
            yield _NullStage()
            return
        with cm as span:
            yield span

    def event(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] = "DEFAULT",
    ) -> None:
        """A point in time rather than a span: cache hit/miss, a guardrail verdict."""
        try:
            self._client.create_event(name=name, metadata=metadata, level=level)
        except Exception:
            logger.warning("tracing.event_failed name=%s", name, exc_info=True)

    def on_generation(self, record: GenerationRecord) -> None:
        """Attach one LLM call's model, tokens and cost to the generation span around it.

        `LLMClient` opens that span before the call and reports the record inside it, so
        `update_current_generation` lands on the right observation and the span's duration is
        the call's real duration — not a zero-width marker added after the fact.
        """
        try:
            usage: dict[str, int] = {
                "input": record.usage.input_tokens,
                "output": record.usage.output_tokens,
                "total": record.usage.total_tokens,
            }
            # Langfuse's reserved key for the cheap-rate portion of the input. Named so the
            # UI's cache-hit view works instead of showing a flat input count.
            if record.usage.cached_tokens:
                usage["cache_read_input_tokens"] = record.usage.cached_tokens
            if record.usage.reasoning_tokens:
                usage["reasoning"] = record.usage.reasoning_tokens

            metadata: dict[str, Any] = {
                "attempts": record.attempts,
                "used_fallback": record.used_fallback,
                "streamed": record.streamed,
                "role": record.role,
                **record.metadata,
            }
            if record.error:
                metadata["error"] = record.error

            self._client.update_current_generation(
                model=record.model,
                usage_details=usage,
                # `cost_details` is our own priced number from the config table, not
                # Langfuse's model-matching guess. Ours is the one the evals quote, so it is
                # the one the traces should agree with.
                cost_details=(
                    {"total": record.usage.cost_usd} if record.usage.cost_usd is not None else None
                ),
                metadata=metadata,
                version=record.prompt_version,
                level="ERROR" if record.error else None,
                status_message=record.error,
            )
        except Exception:
            logger.warning("tracing.generation_failed model=%s", record.model, exc_info=True)

    def score(
        self, *, trace_id: str, name: str, value: float | str, comment: str | None = None
    ) -> bool:
        try:
            self._client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
            # Feedback arrives on its own request, with nothing after it to trigger a flush,
            # so it is sent now rather than when the process happens to exit.
            self._client.flush()
        except Exception:
            logger.warning("tracing.score_failed name=%s", name, exc_info=True)
            return False
        return True

    def trace_url(self, trace_id: str) -> str | None:
        try:
            return self._client.get_trace_url(trace_id=trace_id)
        except Exception:
            logger.warning("tracing.trace_url_failed", exc_info=True)
            return None

    def auth_check(self) -> bool:
        """Whether the configured keys work. Used by `/v1/health`."""
        try:
            return bool(self._client.auth_check())
        except Exception:
            logger.warning("tracing.auth_check_failed", exc_info=True)
            return False

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.warning("tracing.flush_failed", exc_info=True)

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            logger.warning("tracing.shutdown_failed", exc_info=True)


def build_tracer(settings: Settings | None = None) -> SutraTracer:
    """The configured tracer, or the no-op when tracing is off or unconfigured.

    Missing keys are a no-op rather than an error: `tracing_enabled` defaults to on, and a
    test suite or a CI job without a Langfuse should not have to know that to run.
    """
    settings = settings or get_settings()
    if not settings.tracing_enabled:
        return NoOpTracer()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("tracing.disabled reason=no_langfuse_keys")
        return NoOpTracer()
    try:
        return LangfuseTracer(settings)
    except Exception:
        logger.warning("tracing.init_failed falling back to no-op", exc_info=True)
        return NoOpTracer()


# --------------------------------------------------------------------- ambient tracer
#
# # CONCEPT: One process-wide tracer, set at startup. The alternative is a `tracer=`
# # parameter on every function that might want a span, which is the same mistake as
# # threading the trace id: it puts a cross-cutting concern into the signature of code whose
# # subject is retrieval. The module-level accessor keeps the pipeline's stage functions
# # honest — they ask for the current tracer and get a working no-op if there is none.

_tracer: SutraTracer = NoOpTracer()


def set_tracer(tracer: SutraTracer) -> None:
    global _tracer
    _tracer = tracer


def get_tracer() -> SutraTracer:
    return _tracer


def stage(
    name: str,
    *,
    kind: StageKind = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Open a span on the ambient tracer. The call site retrieval code actually uses."""
    return _tracer.stage(name, kind=kind, input=input, metadata=metadata)


def event(
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] = "DEFAULT",
) -> None:
    """Record a point-in-time event on the ambient tracer."""
    _tracer.event(name, metadata=metadata, level=level)


def current_trace_id() -> str | None:
    """The trace id for the request in flight, if any."""
    return get_trace_id()


def generation_scope(
    *,
    name: str,
    model: str,
    prompt_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """The span an LLM call happens inside.

    Separate from `stage` because `LLMClient` must open the span *before* the call so the
    span's duration is the call's duration, then fill in tokens and cost from inside it.
    """
    span_metadata = {"prompt_version": prompt_version, **(metadata or {})}
    return _tracer.stage(name, kind="generation", metadata=span_metadata)


def nullspan() -> Any:
    """An open-nothing context, for call sites that branch on whether to trace at all."""
    return nullcontext(_NullStage())
