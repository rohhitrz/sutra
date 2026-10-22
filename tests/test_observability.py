"""Observability: the trace context, the JSON log format, and the tracer's no-op guarantees.

No Langfuse instance is needed here. That is the point of `build_tracer` degrading to a
no-op: the pipeline's instrumentation is testable without the stack running.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from sutra.config import Settings
from sutra.llm.types import GenerationRecord, Usage
from sutra.observability import context as ctx
from sutra.observability.logs import JsonFormatter, TextFormatter, configure_logging
from sutra.observability.tracing import (
    LangfuseTracer,
    NoOpTracer,
    _GatedStage,
    build_tracer,
    event,
    get_tracer,
    set_tracer,
    stage,
)


def _record(**kwargs: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="sutra.test",
        level=logging.INFO,
        pathname="t.py",
        lineno=1,
        msg=kwargs.pop("msg", "hello"),  # type: ignore[arg-type]
        args=(),
        exc_info=None,
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


# ----------------------------------------------------------------- trace context


def test_request_context_sets_and_clears_the_trace_id():
    assert ctx.get_trace_id() is None
    with ctx.request_context(user_id="u1") as c:
        assert ctx.get_trace_id() == c.trace_id
        assert len(c.trace_id) == 32
    # Reset on exit, not left set: in a long-lived server the next request would otherwise
    # be filed under the previous request's trace.
    assert ctx.get_trace_id() is None


def test_request_context_accepts_a_supplied_trace_id():
    with ctx.request_context(trace_id="abc123", user_id="u1", session_id="s1") as c:
        assert c.trace_id == "abc123"
        fields = ctx.get_request_fields()
        assert fields == {"trace_id": "abc123", "user_id": "u1", "session_id": "s1"}


def test_bind_adds_fields_and_is_a_no_op_outside_a_context():
    ctx.bind(mode="rag")  # must not raise
    assert ctx.get_request_fields() == {}
    with ctx.request_context(trace_id="t"):
        ctx.bind(mode="rag")
        assert ctx.get_request_fields()["mode"] == "rag"


def test_bind_trace_id_restores_the_previous_value():
    """The tracer learns the id only once the root span is open, and must not leave it set.

    Set-and-leave survives a server that runs each request in its own asyncio task, but a
    direct caller in a loop — the eval runner — would file its next row under the previous
    row's trace.
    """
    assert ctx.get_trace_id() is None
    with ctx.bind_trace_id("from-langfuse"):
        assert ctx.get_trace_id() == "from-langfuse"
    assert ctx.get_trace_id() is None

    with ctx.request_context(trace_id="outer"):
        with ctx.bind_trace_id("inner"):
            assert ctx.get_trace_id() == "inner"
        assert ctx.get_trace_id() == "outer"


def test_nested_contexts_restore_the_outer_one():
    with ctx.request_context(trace_id="outer"):
        with ctx.request_context(trace_id="inner"):
            assert ctx.get_trace_id() == "inner"
        assert ctx.get_trace_id() == "outer"


@pytest.mark.asyncio
async def test_concurrent_tasks_do_not_see_each_other_s_trace_id():
    """The property that makes contextvars the right mechanism (see the CONCEPT note)."""
    seen: dict[str, str | None] = {}

    async def one(name: str) -> None:
        with ctx.request_context(trace_id=name):
            await asyncio.sleep(0.01)  # force interleaving
            seen[name] = ctx.get_trace_id()

    await asyncio.gather(one("a"), one("b"), one("c"))
    assert seen == {"a": "a", "b": "b", "c": "c"}


@pytest.mark.asyncio
async def test_a_trace_id_set_before_gather_is_visible_inside_every_branch():
    found: list[str | None] = []

    async def child() -> None:
        found.append(ctx.get_trace_id())

    with ctx.request_context(trace_id="parent"):
        await asyncio.gather(child(), child())
    assert found == ["parent", "parent"]


# -------------------------------------------------------------------- json logs


def test_json_formatter_emits_one_parseable_object():
    payload = json.loads(JsonFormatter().format(_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "sutra.test"
    assert payload["msg"] == "hello"
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_merges_the_trace_context():
    with ctx.request_context(trace_id="t-1", user_id="u-1"):
        payload = json.loads(JsonFormatter().format(_record()))
    assert payload["trace_id"] == "t-1"
    assert payload["user_id"] == "u-1"


def test_json_formatter_promotes_extra_fields():
    payload = json.loads(JsonFormatter().format(_record(chunks=5, refused=False)))
    assert payload["chunks"] == 5
    assert payload["refused"] is False


def test_json_formatter_does_not_leak_logrecord_internals():
    payload = json.loads(JsonFormatter().format(_record()))
    for reserved in ("args", "pathname", "levelno", "msecs", "relativeCreated"):
        assert reserved not in payload


def test_json_formatter_survives_an_unserialisable_value():
    """A log line that raises while formatting loses the information you wanted to keep."""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    payload = json.loads(JsonFormatter().format(_record(thing=Opaque())))
    assert payload["thing"] == "<opaque>"


def test_json_formatter_includes_the_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_text_formatter_appends_the_trace_id_when_there_is_one():
    assert "[trace=" not in TextFormatter().format(_record())
    with ctx.request_context(trace_id="t-9"):
        assert "[trace=t-9]" in TextFormatter().format(_record())


def test_configure_logging_replaces_handlers_rather_than_stacking_them():
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging("INFO", log_format="json")
        configure_logging("INFO", log_format="json")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        configure_logging("DEBUG", log_format="text")
        assert isinstance(root.handlers[0].formatter, TextFormatter)
        assert root.level == logging.DEBUG
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in before:
            root.addHandler(handler)


# ----------------------------------------------------------------- build_tracer


def _settings(**kwargs: object) -> Settings:
    base = {
        "tracing_enabled": True,
        "langfuse_public_key": "pk",
        "langfuse_secret_key": "sk",
    }
    return Settings(**{**base, **kwargs})  # type: ignore[arg-type]


def test_build_tracer_is_a_no_op_when_tracing_is_off():
    assert isinstance(build_tracer(_settings(tracing_enabled=False)), NoOpTracer)


@pytest.mark.parametrize(
    ("public", "secret"),
    [("", "sk"), ("pk", ""), ("", "")],
)
def test_build_tracer_is_a_no_op_without_both_keys(public: str, secret: str):
    """Unconfigured is not an error: `tracing_enabled` defaults on, and CI has no Langfuse."""
    settings = _settings(langfuse_public_key=public, langfuse_secret_key=secret)
    assert isinstance(build_tracer(settings), NoOpTracer)


def test_build_tracer_falls_back_to_the_no_op_when_the_sdk_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
):
    def explode(self: object, settings: object = None) -> None:
        raise RuntimeError("no langfuse here")

    monkeypatch.setattr(LangfuseTracer, "__init__", explode)
    assert isinstance(build_tracer(_settings()), NoOpTracer)


def test_build_tracer_returns_a_langfuse_tracer_when_configured():
    tracer = build_tracer(_settings(langfuse_host="http://localhost:3000"))
    assert isinstance(tracer, LangfuseTracer)
    tracer.shutdown()


# -------------------------------------------------------------- the no-op tracer


def test_the_no_op_tracer_satisfies_every_call_the_pipeline_makes():
    tracer = NoOpTracer()
    with tracer.request(name="chat", user_id="u", session_id="s", input={"q": "x"}) as root:
        root.update(output={"a": 1})
        with tracer.stage("search", kind="retriever") as span:
            span.update(output={"chunks": 3})
        tracer.event("cache.miss", metadata={"reason": "below_threshold"})
    tracer.on_generation(
        GenerationRecord(
            model="m",
            usage=Usage(input_tokens=1, output_tokens=2),
            latency_ms=5.0,
            attempts=1,
            used_fallback=False,
        )
    )
    tracer.score(trace_id="t", name="user-feedback", value=1, comment="fine")
    assert tracer.trace_url("t") is None
    tracer.flush()
    tracer.shutdown()


def test_the_ambient_tracer_is_settable_and_defaults_to_the_no_op():
    original = get_tracer()
    try:
        assert isinstance(original, NoOpTracer)
        recorded: list[str] = []

        class Spy(NoOpTracer):
            def event(self, name: str, *, metadata=None, level="DEFAULT") -> None:
                recorded.append(name)

        set_tracer(Spy())
        with stage("search", kind="retriever"):
            event("cache.hit")
        assert recorded == ["cache.hit"]
    finally:
        set_tracer(original)


# -------------------------------------------------- content capture (known issue 20)


class _SpanSpy:
    """Stands in for a Langfuse observation and records what `update` was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def update(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def test_a_gated_stage_forwards_output_when_capture_is_on():
    spy = _SpanSpy()
    _GatedStage(spy, capture_content=True).update(output={"answer": "a"}, metadata={"n": 1})
    assert spy.calls == [{"output": {"answer": "a"}, "metadata": {"n": 1}}]


def test_a_gated_stage_drops_output_when_capture_is_off():
    """Dropped, not nulled: an explicit None would overwrite whatever the span already had."""
    spy = _SpanSpy()
    _GatedStage(spy, capture_content=False).update(output={"answer": "a"}, metadata={"n": 1})
    assert spy.calls == [{"metadata": {"n": 1}}]


def test_a_gated_stage_passes_metadata_through_either_way():
    """Metadata is structural — counts, versions, flags — and is not content to withhold."""
    for capture in (True, False):
        spy = _SpanSpy()
        _GatedStage(spy, capture_content=capture).update(metadata={"rrf_k": 60})
        assert spy.calls == [{"metadata": {"rrf_k": 60}}]


def test_a_gated_stage_proxies_anything_else_the_sdk_offers():
    class Rich(_SpanSpy):
        id = "obs-1"

    assert _GatedStage(Rich(), capture_content=True).id == "obs-1"


class _GenerationSpy(LangfuseTracer):
    """A `LangfuseTracer` with the SDK replaced, to see what `on_generation` sends."""

    def __init__(self, *, capture_content: bool) -> None:
        self.sent: dict[str, object] = {}
        self._capture_content = capture_content
        parent = self

        class Client:
            def update_current_generation(self, **kwargs: object) -> None:
                parent.sent = kwargs

        self._client = Client()  # type: ignore[assignment]


def _record_with_content() -> GenerationRecord:
    return GenerationRecord(
        model="m",
        usage=Usage(input_tokens=10, output_tokens=4),
        latency_ms=5.0,
        attempts=1,
        used_fallback=False,
        input=[{"role": "user", "content": "who drowned in the flood?"}],
        output='{"answer": "Ravenna Volant."}',
    )


def test_a_generation_span_records_its_prompt_and_its_completion():
    tracer = _GenerationSpy(capture_content=True)
    tracer.on_generation(_record_with_content())
    assert tracer.sent["input"] == [{"role": "user", "content": "who drowned in the flood?"}]
    assert tracer.sent["output"] == '{"answer": "Ravenna Volant."}'


def test_a_generation_span_withholds_both_when_capture_is_off():
    """The tokens, cost and model still go: those are not content."""
    tracer = _GenerationSpy(capture_content=False)
    tracer.on_generation(_record_with_content())
    assert tracer.sent["input"] is None
    assert tracer.sent["output"] is None
    assert tracer.sent["model"] == "m"
    assert tracer.sent["usage_details"] == {"input": 10, "output": 4, "total": 14}


# ----------------------------------- the request contextmanager (review finding)


class _FakeObservation:
    def __enter__(self) -> str:
        return "span"

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        self.exited_with = exc
        return False


class _FakeLangfuse:
    """Just enough client for `LangfuseTracer.request`."""

    def __init__(self, *, explode: bool = False) -> None:
        self.explode = explode
        self.observation = _FakeObservation()

    def start_as_current_observation(self, **kwargs: object) -> _FakeObservation:
        if self.explode:
            raise RuntimeError("langfuse is down")
        return self.observation

    def get_current_trace_id(self) -> str:
        return "t" * 32


def _tracer_with(client: _FakeLangfuse) -> LangfuseTracer:
    """A `LangfuseTracer` without the SDK, so `request` can be exercised offline."""
    tracer = LangfuseTracer.__new__(LangfuseTracer)
    tracer._capture_content = True
    tracer._client = client  # type: ignore[assignment]
    return tracer


class _Boom(LookupError):
    """Stands in for `SessionNotFound`, which is what actually hit this."""


def test_request_lets_the_body_s_exception_through_unchanged():
    """# CONCEPT: a `try` around a `@contextmanager`'s `yield` catches the *body's*
    exception, because `contextlib` throws it in at the yield point. `request` used to do
    that and then yield again, which `contextlib` turns into `RuntimeError: generator didn't
    stop after throw()` — so `SessionNotFound` reached the route as a RuntimeError and a
    foreign session id answered 500 instead of 404. The no-op tracer has a bare yield, which
    is why the suite could not see it.
    """
    tracer = _tracer_with(_FakeLangfuse())
    with pytest.raises(_Boom, match="nope"):
        with tracer.request(name="chat", user_id="u"):
            raise _Boom("nope")


def test_request_closes_and_marks_the_span_before_re_raising():
    """The span must still be closed, and told what went wrong, on the way out."""
    client = _FakeLangfuse()
    tracer = _tracer_with(client)
    with pytest.raises(_Boom):
        with tracer.request(name="chat"):
            raise _Boom("nope")
    assert isinstance(client.observation.exited_with, _Boom)


def test_request_still_degrades_to_a_no_op_when_the_tracer_itself_breaks():
    """The other half of the guarantee, which the fix must not cost."""
    tracer = _tracer_with(_FakeLangfuse(explode=True))
    with tracer.request(name="chat") as span:
        span.update(output={"answer": "a"})  # must not raise


def test_request_clears_the_trace_id_on_the_way_out():
    tracer = _tracer_with(_FakeLangfuse())
    with tracer.request(name="chat"):
        assert ctx.get_trace_id() == "t" * 32
    assert ctx.get_trace_id() is None


# ------------------------------------------------------------- import ordering


@pytest.mark.parametrize(
    "first",
    [
        "sutra.observability.tracing",
        "sutra.cache.semantic",
        "sutra.retrieval.search",
        "sutra.llm.client",
        "sutra.api.app",
    ],
)
def test_no_module_import_order_causes_a_circular_import(first: str):
    """Guards a cycle this suite could not see.

    `observability.tracing` needs `llm.types` for one annotation, and importing it executes
    `sutra/llm/__init__.py`, which eagerly imports `llm.client` — which imports
    `observability.tracing` for `get_tracer`. That is a cycle whenever observability is
    imported first, and `make cache` hit it while pytest did not: pytest happens to import
    the client first, and by then `sutra.llm` is fully initialised.

    A subprocess per case, because the guarantee is about a *cold* interpreter — once
    anything is in `sys.modules` the ordering that breaks is unreachable.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", f"import {first}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"importing {first} first failed:\n{result.stderr}"
