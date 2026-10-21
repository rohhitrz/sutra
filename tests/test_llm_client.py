"""LLMClient: role resolution, retries, fallback, cost accounting, streaming.

The whole client is exercised against a fake adapter — no network, no API key. That is the
point of the adapter protocol in ADR-005: the retry/fallback/costing logic is testable without
a provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import contextmanager

import httpx2
import pytest
from openai import APIConnectionError, BadRequestError
from pydantic import BaseModel

from sutra.config import Settings
from sutra.llm.adapters import RawResult
from sutra.llm.client import LLMClient, LLMError
from sutra.llm.tracing import NoOpTracer
from sutra.llm.types import GenerationRecord, Message, StreamDelta, StreamDone

PROMPT = [Message(role="user", content="Who is the Archivist?")]


def _conn_error() -> APIConnectionError:
    return APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))


def _bad_request() -> BadRequestError:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(400, request=request, json={"error": {"message": "bad schema"}})
    return BadRequestError("bad schema", response=response, body=None)


class FakeAdapter:
    """Scriptable adapter. `failures` maps a model name to the number of times it should fail
    before succeeding; `always_fail` makes a model fail forever."""

    name = "fake"

    def __init__(
        self,
        *,
        failures: dict[str, int] | None = None,
        always_fail: set[str] | None = None,
        raise_non_retryable: bool = False,
        input_tokens: int = 1000,
        output_tokens: int = 500,
        cached_tokens: int = 0,
    ) -> None:
        self.failures = dict(failures or {})
        self.always_fail = set(always_fail or set())
        self.raise_non_retryable = raise_non_retryable
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens
        self.calls: list[str] = []

    def _maybe_fail(self, model: str) -> None:
        self.calls.append(model)
        if self.raise_non_retryable:
            raise _bad_request()
        if model in self.always_fail:
            raise _conn_error()
        remaining = self.failures.get(model, 0)
        if remaining > 0:
            self.failures[model] = remaining - 1
            raise _conn_error()

    def _result(
        self, model: str, *, parsed: BaseModel | None = None, text: str | None = None
    ) -> RawResult:
        return RawResult(
            text=text if text is not None else "The Archivist keeps the Meridian records.",
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            response_id="resp_fake",
            parsed=parsed,
        )

    async def complete(self, *, model: str, messages, max_output_tokens, temperature, timeout):
        self._maybe_fail(model)
        return self._result(model)

    async def structured(
        self, *, model: str, messages, schema, max_output_tokens, temperature, timeout
    ):
        self._maybe_fail(model)
        return self._result(model, parsed=schema(answer="kept the records", confidence=0.9))

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens,
        temperature,
        timeout,
        schema=None,
    ) -> AsyncIterator[tuple[str | None, RawResult | None]]:
        self._maybe_fail(model)
        self.streamed_schema = schema
        if schema is not None:
            # A schema-constrained stream emits JSON fragments, not prose.
            for piece in ('{"answer": "The Arch', 'ivist keeps."', ', "confidence": 0.9}'):
                yield piece, None
            yield None, self._result(model, text='{"answer": "The Archivist keeps."}')
            return
        for piece in ("The ", "Archivist ", "keeps."):
            yield piece, None
        yield None, self._result(model)


class RecordingTracer(NoOpTracer):
    """Captures records, and inherits the no-op `stage` so spans are exercised too."""

    def __init__(self) -> None:
        self.records: list[GenerationRecord] = []
        self.stages: list[tuple[str, object]] = []

    def on_generation(self, record: GenerationRecord) -> None:
        self.records.append(record)

    @contextmanager
    def stage(self, name: str, *, kind=None, input=None, metadata=None):
        self.stages.append((name, kind))
        yield None


class AnswerSchema(BaseModel):
    answer: str
    confidence: float


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch):
    """Backoff is real in production and pointless in tests."""

    async def instant(_delay: float) -> None:
        return None

    monkeypatch.setattr("sutra.llm.client.asyncio.sleep", instant)


# ----------------------------------------------------------------- role resolution


def test_role_resolves_through_config(settings: Settings):
    settings = settings.model_copy(update={"llm_fast": "fast-model", "llm_strong": "strong-model"})
    client = LLMClient(settings=settings, adapter=FakeAdapter())
    assert client.resolve_model("fast", None) == "fast-model"
    assert client.resolve_model("strong", None) == "strong-model"


def test_explicit_model_overrides_role(settings: Settings):
    client = LLMClient(settings=settings, adapter=FakeAdapter())
    assert client.resolve_model("fast", "pinned-model") == "pinned-model"


# `resolve_model` always honoured `model=`, but for a while `complete` and `structured` resolved
# the chain without it and quietly answered from `llm_strong`. Testing the resolver alone did not
# catch that, so these go through the public methods and assert on what the adapter was handed.


async def test_complete_honours_an_explicit_model(settings: Settings):
    adapter = FakeAdapter()
    client = LLMClient(settings=settings, adapter=adapter)
    result = await client.complete(PROMPT, role="strong", model="gpt-6-sol")

    assert adapter.calls == ["gpt-6-sol"]
    assert result.model == "gpt-6-sol"


async def test_structured_honours_an_explicit_model(settings: Settings):
    adapter = FakeAdapter()
    client = LLMClient(settings=settings, adapter=adapter)
    result = await client.structured(PROMPT, AnswerSchema, role="strong", model="gpt-6-sol")

    assert adapter.calls == ["gpt-6-sol"]
    assert result.model == "gpt-6-sol"


async def test_an_explicit_model_does_not_fall_back(settings: Settings):
    """A pinned model is a deliberate choice; answering from a different one would misreport it."""
    settings = settings.model_copy(update={"llm_fallback_models": ["gpt-6-luna"]})
    adapter = FakeAdapter(always_fail={"gpt-6-sol"})
    client = LLMClient(settings=settings, adapter=adapter)

    with pytest.raises(LLMError):
        await client.complete(PROMPT, model="gpt-6-sol")

    assert adapter.calls == ["gpt-6-sol"] * 3


# ------------------------------------------------------------------------ complete


async def test_complete_returns_text_and_priced_usage(settings: Settings):
    client = LLMClient(settings=settings, adapter=FakeAdapter())
    result = await client.complete(PROMPT)

    assert result.text.startswith("The Archivist")
    assert result.model == "gpt-6-luna"
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 500
    # 1000 * 0.10/1M + 500 * 0.50/1M
    assert result.usage.cost_usd == pytest.approx(0.00035)
    assert result.attempts == 1
    assert result.used_fallback is False
    assert result.latency_ms >= 0


async def test_unknown_model_yields_unknown_cost(settings: Settings):
    settings = settings.model_copy(
        update={"llm_strong": "unpriced-model", "llm_fallback_models": []}
    )
    client = LLMClient(settings=settings, adapter=FakeAdapter())
    result = await client.complete(PROMPT)
    # An un-priced call is still a valid answer — cost is unknown, not zero.
    assert result.usage.cost_usd is None
    assert result.usage.input_tokens == 1000


# ------------------------------------------------------------------------- retries


async def test_transient_error_is_retried_then_succeeds(settings: Settings):
    adapter = FakeAdapter(failures={"gpt-6-luna": 2})
    client = LLMClient(settings=settings, adapter=adapter)
    result = await client.complete(PROMPT)

    assert result.attempts == 3
    assert adapter.calls == ["gpt-6-luna"] * 3


async def test_non_retryable_error_is_not_retried(settings: Settings):
    adapter = FakeAdapter(raise_non_retryable=True)
    client = LLMClient(settings=settings, adapter=adapter)

    with pytest.raises(BadRequestError):
        await client.complete(PROMPT)

    # A 400 fails identically on every attempt and every model; retrying only adds latency.
    assert len(adapter.calls) == 1


# ------------------------------------------------------------------------ fallback


async def test_falls_back_to_next_model_when_primary_exhausts(settings: Settings):
    settings = settings.model_copy(
        update={"llm_strong": "gpt-6-luna", "llm_fallback_models": ["gpt-6-sol"]}
    )
    adapter = FakeAdapter(always_fail={"gpt-6-luna"})
    client = LLMClient(settings=settings, adapter=adapter)
    result = await client.complete(PROMPT)

    assert result.used_fallback is True
    assert result.model == "gpt-6-sol"
    # Three attempts on the primary, then one on the fallback.
    assert adapter.calls == ["gpt-6-luna"] * 3 + ["gpt-6-sol"]
    # Cost is priced against the model that actually answered, not the one we asked for.
    assert result.usage.cost_usd == pytest.approx(1000 * 2.0 / 1e6 + 500 * 10.0 / 1e6)


async def test_llm_error_when_every_model_fails(settings: Settings):
    settings = settings.model_copy(
        update={"llm_strong": "gpt-6-luna", "llm_fallback_models": ["gpt-6-sol"]}
    )
    adapter = FakeAdapter(always_fail={"gpt-6-luna", "gpt-6-sol"})
    client = LLMClient(settings=settings, adapter=adapter)

    with pytest.raises(LLMError, match="all models failed"):
        await client.complete(PROMPT)

    assert len(adapter.calls) == 6  # 3 attempts on each of 2 models


async def test_fallback_chain_does_not_duplicate_the_primary(settings: Settings):
    # llm_fallback_models repeats the primary; it must not be tried twice as a "fallback".
    settings = settings.model_copy(
        update={"llm_strong": "gpt-6-luna", "llm_fallback_models": ["gpt-6-luna"]}
    )
    adapter = FakeAdapter(always_fail={"gpt-6-luna"})
    client = LLMClient(settings=settings, adapter=adapter)

    with pytest.raises(LLMError):
        await client.complete(PROMPT)

    assert len(adapter.calls) == 3


# ----------------------------------------------------------------------- structured


async def test_structured_returns_validated_instance(settings: Settings):
    client = LLMClient(settings=settings, adapter=FakeAdapter())
    result = await client.structured(PROMPT, AnswerSchema)

    assert isinstance(result.parsed, AnswerSchema)
    assert result.parsed.confidence == 0.9
    assert result.usage.cost_usd == pytest.approx(0.00035)


# -------------------------------------------------------------------------- stream


async def test_stream_yields_deltas_then_a_terminal_usage_event(settings: Settings):
    client = LLMClient(settings=settings, adapter=FakeAdapter())
    events = [event async for event in client.stream(PROMPT)]

    deltas = [e for e in events if isinstance(e, StreamDelta)]
    done = [e for e in events if isinstance(e, StreamDone)]

    assert "".join(d.text for d in deltas) == "The Archivist keeps."
    # Exactly one terminal event, and it is last — Phase 4 maps this onto an SSE `done` frame.
    assert len(done) == 1
    assert isinstance(events[-1], StreamDone)
    assert done[0].usage.cost_usd == pytest.approx(0.00035)


# -------------------------------------------------------------------------- tracing


async def test_tracer_receives_a_record_per_call(settings: Settings):
    tracer = RecordingTracer()
    client = LLMClient(settings=settings, adapter=FakeAdapter(), tracer=tracer)
    await client.complete(PROMPT)

    assert len(tracer.records) == 1
    record = tracer.records[0]
    assert record.model == "gpt-6-luna"
    assert record.role == "strong"
    assert record.usage.cost_usd == pytest.approx(0.00035)
    assert record.streamed is False


async def test_tracer_records_failures(settings: Settings):
    tracer = RecordingTracer()
    settings = settings.model_copy(update={"llm_fallback_models": []})
    adapter = FakeAdapter(always_fail={"gpt-6-luna"})
    client = LLMClient(settings=settings, adapter=adapter, tracer=tracer)

    with pytest.raises(LLMError):
        await client.complete(PROMPT)

    # A failed call still costs latency and still needs to show up in the trace.
    assert len(tracer.records) == 1
    assert tracer.records[0].error is not None


# ---------------------------------------------------------------------------
# model capabilities
# ---------------------------------------------------------------------------


def test_temperature_is_dropped_for_models_that_reject_it(settings):
    """Regression from the first live call this project ever made:

        400 — Unsupported parameter: 'temperature' is not supported with this model.

    The prompt file legitimately asks for temperature 0, so the intent stays in the prompt and
    the client decides whether the resolved model can be told about it.
    """
    client = LLMClient(settings, adapter=FakeAdapter())
    assert client.effective_temperature("gpt-6-luna", 0.0) is None
    assert client.effective_temperature("gpt-6-luna", None) is None


def test_temperature_passes_through_for_models_that_accept_it(settings):
    """An unknown model is assumed conventional: a provider 400 is more informative than a
    parameter dropped behind your back."""
    client = LLMClient(settings, adapter=FakeAdapter())
    assert client.effective_temperature("some-future-model", 0.3) == 0.3


async def test_a_structured_call_sends_no_temperature_to_an_unsupporting_model(settings):
    """The end the bug actually appeared at: what reaches the adapter."""
    recorder: list[float | None] = []

    class Recording(FakeAdapter):
        async def structured(
            self, *, model, messages, schema, max_output_tokens, temperature, timeout
        ):
            recorder.append(temperature)
            return await super().structured(
                model=model,
                messages=messages,
                schema=schema,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout=timeout,
            )

    client = LLMClient(settings, adapter=Recording())
    await client.structured(
        [Message(role="user", content="q")], AnswerSchema, role="strong", temperature=0.0
    )
    assert recorder == [None]
