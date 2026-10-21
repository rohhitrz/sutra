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
