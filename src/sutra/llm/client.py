"""The one LLM client.

Every LLM call in Sutra goes through here (ADR-005, CLAUDE.md). That single choke point is what
makes tracing, cost accounting, retries, timeouts and model fallback apply uniformly instead of
being re-implemented — and forgotten — at each call site.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel

from sutra.config import LONG_CONTEXT_THRESHOLD_TOKENS, Settings, get_settings
from sutra.llm.adapters import LLMAdapter, OpenAIAdapter, RawResult
from sutra.llm.tracing import Tracer
from sutra.llm.types import (
    GenerationRecord,
    LLMResponse,
    Message,
    ModelRole,
    StreamDelta,
    StreamDone,
    StreamEvent,
    Usage,
)
from sutra.observability.tracing import get_tracer

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# # CONCEPT: Retry only what retrying can fix. A 429 or a 5xx is transient and worth another
# # attempt; a 400 (bad schema) or 401 (bad key) will fail identically every time, and retrying
# # it just triples your latency before surfacing the same error.
RETRYABLE: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

DEFAULT_MAX_OUTPUT_TOKENS = 4096


class LLMError(RuntimeError):
    """Raised when every model in the chain has failed."""


class LLMClient:
    """Resolves a model role, calls the provider with retries, falls back, and accounts cost."""

    def __init__(
        self,
        settings: Settings | None = None,
        adapter: LLMAdapter | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Injectable so tests can run the whole retry/fallback/costing path against a fake
        # adapter without a network call or an API key.
        self._adapter = adapter or OpenAIAdapter(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.llm_timeout_seconds,
        )
        # Defaults to the ambient tracer rather than to the no-op. An explicit `tracer=` is
        # still honoured (tests use it), but the default has to be the working one: a client
        # built without one would otherwise report no generations at all, and the only
        # symptom is a trace that is quietly missing its tokens and its cost.
        self._tracer = tracer if tracer is not None else get_tracer()

    async def aclose(self) -> None:
        """Release the adapter's connection pool. Safe on an adapter that does not need it."""
        closer = getattr(self._adapter, "aclose", None)
        if closer is not None:
            await closer()

    # ---------------------------------------------------------------- internals

    def effective_temperature(self, model: str, temperature: float | None) -> float | None:
        """Drop `temperature` for models that reject it.

        The caller's intent stays where it belongs — in the prompt file's front-matter — and this
        is the one place that knows whether the resolved model can be told about it. Without
        this, every prompt declaring a temperature breaks on a reasoning model with a 400 that
        names a parameter the prompt author never passed directly.
        """
        if temperature is None:
            return None
        if self._settings.capabilities_for(model).supports_temperature:
            return temperature
        logger.debug("model %s does not accept temperature; dropping %s", model, temperature)
        return None

    def resolve_model(self, role: ModelRole | None, model: str | None) -> str:
        """An explicit model wins; otherwise the role maps through config. ADR-017 points both
        roles at the same model today, which is why this is config and not a constant."""
        if model is not None:
            return model
        if role == "fast":
            return self._settings.llm_fast
        return self._settings.llm_strong

    def _chain(self, primary: str) -> list[str]:
        """The primary model, then each configured fallback that is not already in the chain."""
        chain = [primary]
        for candidate in self._settings.llm_fallback_models:
            if candidate not in chain:
                chain.append(candidate)
        return chain

    def _price(self, raw: RawResult) -> Usage:
        """Turn raw token counts into a priced Usage."""
        if raw.input_tokens > LONG_CONTEXT_THRESHOLD_TOKENS:
            logger.warning(
                "llm.long_context input_tokens=%d exceeds %d — the price table holds "
                "short-context rates, so this call's cost is understated",
                raw.input_tokens,
                LONG_CONTEXT_THRESHOLD_TOKENS,
            )
        price = self._settings.price_for(raw.model)
        if price is None:
            logger.warning(
                "llm.unpriced_model model=%s — cost will be reported as unknown", raw.model
            )
            cost = None
        else:
            cost = price.cost_usd(raw.input_tokens, raw.output_tokens, raw.cached_tokens)
        return Usage(
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cached_tokens=raw.cached_tokens,
            reasoning_tokens=raw.reasoning_tokens,
            cost_usd=cost,
        )

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with full jitter. Jitter matters: without it, every concurrent
        caller that hit the same rate limit retries in lockstep and trips it again."""
        delay = min(2.0**attempt, 30.0) * random.random()
        await asyncio.sleep(delay)

    def _trace(
        self,
        *,
        model: str,
        role: ModelRole | None,
        usage: Usage,
        started: float,
        attempts: int,
        used_fallback: bool,
        streamed: bool = False,
        error: str | None = None,
        prompt_version: str | None = None,
    ) -> float:
        latency_ms = (time.perf_counter() - started) * 1000
        self._tracer.on_generation(
            GenerationRecord(
                model=model,
                role=role,
                usage=usage,
                latency_ms=latency_ms,
                attempts=attempts,
                used_fallback=used_fallback,
                streamed=streamed,
                error=error,
                prompt_version=prompt_version,
            )
        )
        return latency_ms

    @contextmanager
    def _generation_span(
        self, *, stage: str | None, role: ModelRole | None, model: str | None
    ) -> Iterator[None]:
        """The span one provider call happens inside.

        Opened here rather than filled in afterwards: the span's duration has to be the
        call's duration, or a trace cannot say which stage spent the time. `stage` names it
        (`generate`, `route`, `rewrite`, ...) so sibling generations in one request are
        distinguishable in the UI instead of seven spans all called "generation".
        """
        with self._tracer.stage(
            stage or "llm.call",
            kind="generation",
            metadata={"role": role, "requested_model": model},
        ):
            yield

    async def _call_with_retries(
        self,
        *,
        role: ModelRole | None,
        make_call: Callable[[str], Awaitable[RawResult]],
        started: float,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> tuple[RawResult, int, bool]:
        """Walk the model chain. Within each model, retry transient failures; on exhaustion,
        move to the next model. Returns (result, attempts, used_fallback)."""
        primary = self.resolve_model(role, model)
        # An explicitly named model is a deliberate choice by the caller — an ablation pinning one
        # model, say. Falling back off it would answer from a model the caller did not ask for and
        # report it under their chosen name, so an explicit model gets a chain of one. Retries
        # still apply: a transient 429 is not a reason to abandon the requested model.
        chain = [primary] if model is not None else self._chain(primary)
        attempts = 0
        last_error: Exception | None = None

        for model_index, model in enumerate(chain):
            for attempt in range(self._settings.llm_max_retries):
                attempts += 1
                try:
                    raw = await make_call(model)
                except RETRYABLE as exc:
                    last_error = exc
                    logger.warning(
                        "llm.retryable model=%s attempt=%d/%d error=%s",
                        model,
                        attempt + 1,
                        self._settings.llm_max_retries,
                        type(exc).__name__,
                    )
                    if attempt + 1 < self._settings.llm_max_retries:
                        await self._backoff(attempt)
                    continue
                except Exception as exc:
                    # Not retryable. Do not burn attempts or fall back — a bad request or a bad
                    # key fails the same way on every model in the chain.
                    self._trace(
                        model=model,
                        role=role,
                        usage=Usage(),
                        started=started,
                        attempts=attempts,
                        used_fallback=model_index > 0,
                        error=f"{type(exc).__name__}: {exc}",
                        prompt_version=prompt_version,
                    )
                    raise
                else:
                    return raw, attempts, model_index > 0

            if model_index + 1 < len(chain):
                logger.warning("llm.fallback from=%s to=%s", model, chain[model_index + 1])

        self._trace(
            model=primary,
            role=role,
            usage=Usage(),
            started=started,
            attempts=attempts,
            used_fallback=len(chain) > 1,
            error=f"{type(last_error).__name__}: {last_error}" if last_error else "exhausted",
            prompt_version=prompt_version,
        )
        msg = f"all models failed after {attempts} attempt(s): {chain}"
        raise LLMError(msg) from last_error

    # ------------------------------------------------------------------- public

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        role: ModelRole | None = "strong",
        model: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        stage: str | None = None,
        prompt_version: str | None = None,
    ) -> LLMResponse:
        """One non-streaming call.

        `stage` names the span (ARCHITECTURE §10) and `prompt_version` records which prompt
        file produced it (ADR-006). Both are optional so a call can be made without them,
        and both are passed by every caller in Sutra so no generation is anonymous.
        """
        started = time.perf_counter()

        async def call(resolved: str) -> RawResult:
            return await self._adapter.complete(
                model=resolved,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=self.effective_temperature(resolved, temperature),
                timeout=self._settings.llm_timeout_seconds,
            )

        with self._generation_span(stage=stage, role=role, model=model):
            raw, attempts, used_fallback = await self._call_with_retries(
                role=role,
                make_call=call,
                started=started,
                model=model,
                prompt_version=prompt_version,
            )
            usage = self._price(raw)
            latency_ms = self._trace(
                model=raw.model,
                role=role,
                usage=usage,
                started=started,
                attempts=attempts,
                used_fallback=used_fallback,
                prompt_version=prompt_version,
            )
        return LLMResponse(
            text=raw.text,
            model=raw.model,
            usage=usage,
            used_fallback=used_fallback,
            attempts=attempts,
            latency_ms=latency_ms,
            response_id=raw.response_id,
        )

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[SchemaT],
        *,
        role: ModelRole | None = "strong",
        model: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        stage: str | None = None,
        prompt_version: str | None = None,
    ) -> LLMResponse:
        """One call constrained to `schema`. The validated instance is on `.parsed`."""
        started = time.perf_counter()

        async def call(resolved: str) -> RawResult:
            return await self._adapter.structured(
                model=resolved,
                messages=messages,
                schema=schema,
                max_output_tokens=max_output_tokens,
                temperature=self.effective_temperature(resolved, temperature),
                timeout=self._settings.llm_timeout_seconds,
            )

        with self._generation_span(stage=stage, role=role, model=model):
            raw, attempts, used_fallback = await self._call_with_retries(
                role=role,
                make_call=call,
                started=started,
                model=model,
                prompt_version=prompt_version,
            )
            usage = self._price(raw)
            latency_ms = self._trace(
                model=raw.model,
                role=role,
                usage=usage,
                started=started,
                attempts=attempts,
                used_fallback=used_fallback,
                prompt_version=prompt_version,
            )
        return LLMResponse(
            text=raw.text,
            model=raw.model,
            usage=usage,
            parsed=raw.parsed,
            used_fallback=used_fallback,
            attempts=attempts,
            latency_ms=latency_ms,
            response_id=raw.response_id,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        role: ModelRole | None = "strong",
        model: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        stage: str | None = None,
        prompt_version: str | None = None,
        schema: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream text deltas, then one terminal event carrying usage and cost.

        No retries here on purpose: once the first token has been handed to the caller, a retry
        would duplicate output. `sutra.api.streaming` maps these events onto SSE frames.

        The span stays open for the whole stream, so its duration is the time the caller spent
        receiving tokens rather than the time to the first one.

        With `schema`, the deltas are fragments of a JSON object rather than prose, and
        `StreamDone.parsed` carries the provider's validated instance. See
        `sutra.generation.partial_json` for why that is the shape a reader gets served from.
        """
        started = time.perf_counter()
        resolved = self.resolve_model(role, model)
        final: RawResult | None = None

        with self._generation_span(stage=stage, role=role, model=model):
            async for delta, result in self._adapter.stream(
                model=resolved,
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=self.effective_temperature(resolved, temperature),
                timeout=self._settings.llm_timeout_seconds,
                schema=schema,
            ):
                if delta is not None:
                    yield StreamDelta(text=delta)
                if result is not None:
                    final = result

            if final is None:
                final = RawResult(text="", model=resolved)
            usage = self._price(final)
            self._trace(
                model=final.model,
                role=role,
                usage=usage,
                started=started,
                attempts=1,
                used_fallback=False,
                streamed=True,
                prompt_version=prompt_version,
            )
        yield StreamDone(
            usage=usage,
            model=final.model,
            response_id=final.response_id,
            parsed=final.parsed,
            text=final.text,
        )


@asynccontextmanager
async def client_scope(
    client: LLMClient | None, settings: Settings | None = None
) -> AsyncIterator[LLMClient]:
    """Yield `client` if the caller supplied one, otherwise a temporary one that gets closed.

    # CONCEPT: Ownership, made explicit. The `client or LLMClient(...)` idiom is convenient and
    # leaks: every call that falls through to the default builds a fresh provider SDK — and a
    # fresh HTTP connection pool — that nothing ever closes. One retrieval request was creating
    # up to two of them, per request, held open until the garbage collector noticed.
    #
    # Injecting the client is the real fix, because a pool is meant to be shared across calls,
    # not created per call. This exists for the paths that are still allowed to run without
    # one: it keeps the convenience and makes the clean-up unavoidable.
    """
    if client is not None:
        yield client
        return
    owned = LLMClient(settings=settings)
    try:
        yield owned
    finally:
        await owned.aclose()
