"""The ambient request context: who is asking, and which trace is recording it.

# CONCEPT: Cross-cutting context travels in contextvars, not in function signatures. A
# trace id is needed by the logger, by every span and by the response, which is to say by
# code at every depth of the call stack. Threading it as a parameter would add an argument
# to functions whose job has nothing to do with tracing — `keyword_search(query, session,
# k, index_version, trace_id)` — and the first function that forgets to pass it silently
# orphans everything below it.
#
# `contextvars` is the asyncio-correct answer: each task gets its own copy, so two
# concurrent requests cannot see each other's ids, and a value set before `asyncio.gather`
# is visible inside every branch. This is also exactly how OpenTelemetry propagates its own
# span context, which is why the Langfuse SDK's nested spans work without being handed a
# parent.
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Set once per request, read by the log formatter and attached to spans.
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sutra_trace_id", default=None
)
# The default is None, not {}: a mutable default on a ContextVar is shared by every context
# that never set one, so a single in-place mutation would leak across requests.
_request_fields: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "sutra_request_fields", default=None
)


def new_trace_id() -> str:
    """A 32-character hex id.

    Deliberately the same shape as an OpenTelemetry trace id, which is what Langfuse uses.
    When tracing is on, the id comes from the SDK and this is unused; when tracing is off,
    this keeps `trace_id` a real correlation key in the logs rather than a null column.
    """
    return uuid.uuid4().hex


def get_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(trace_id: str | None) -> None:
    _trace_id.set(trace_id)


@contextmanager
def bind_trace_id(trace_id: str | None) -> Iterator[None]:
    """Set the trace id for the duration of a block, then put back what was there.

    Used by the tracer, which learns the id only once the root span is open. Set-and-leave
    would be enough under a server that runs each request in its own asyncio task (the task
    gets a copy of the context), but it leaves the id set in whatever context opened the
    span — so a direct caller in a loop, like the eval runner, would file its next row under
    the previous row's trace. Restoring is one line and removes the need to reason about it.
    """
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        _trace_id.reset(token)


def get_request_fields() -> dict[str, Any]:
    return _request_fields.get() or {}


@dataclass(slots=True)
class RequestContext:
    """What every log line and span for one request should be able to say about it."""

    trace_id: str
    user_id: str | None = None
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"trace_id": self.trace_id}
        if self.user_id is not None:
            fields["user_id"] = self.user_id
        if self.session_id is not None:
            fields["session_id"] = self.session_id
        fields.update(self.extra)
        return fields


@contextmanager
def request_context(
    *,
    trace_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    **extra: Any,
) -> Iterator[RequestContext]:
    """Bind a trace id and request fields for the duration of one request.

    Tokens are reset on exit rather than left set: in a long-lived server the same task can
    serve the next request, and a leaked trace id would quietly file new work under an old
    trace — the kind of bug that makes a tracing system worse than none.
    """
    ctx = RequestContext(
        trace_id=trace_id or new_trace_id(),
        user_id=user_id,
        session_id=session_id,
        extra=extra,
    )
    trace_token = _trace_id.set(ctx.trace_id)
    fields_token = _request_fields.set(ctx.as_fields())
    try:
        yield ctx
    finally:
        _trace_id.reset(trace_token)
        _request_fields.reset(fields_token)


def bind(**fields: Any) -> None:
    """Add fields to the current request context, for things not known at entry.

    A no-op outside a `request_context`, on purpose: a background task that logs is not a bug
    to be punished with an exception.
    """
    current = _request_fields.get()
    if not current:
        return
    # Replaced, never mutated in place, so the dict a parent context sees is unchanged.
    _request_fields.set({**current, **fields})
