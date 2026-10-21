"""Structured JSON logging.

# CONCEPT: A log line is a record, not a sentence. `"chat.turn session=... cost=..."` reads
# well and is useless at scale: finding every turn that cost more than a cent means a regex
# over a format string, and the first person to reword the message breaks it. Emitting one
# JSON object per line makes the same information queryable, and — the part that matters for
# this phase — lets every line carry the `trace_id` so a log search and a Langfuse trace are
# two views of one request.
#
# Hand-rolled rather than a dependency: the whole formatter is forty lines, and a logging
# library is a strange thing to pull in for code whose job is to be read.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from sutra.observability.context import get_request_fields

# Attributes `logging.LogRecord` sets itself. Anything else on a record was put there by a
# caller passing `extra=`, so it belongs in the output.
_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info", "taskName",
        "thread", "threadName",
    }
)  # fmt: skip


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the request's trace context merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # The request context comes first-class: every line inside a traced request is
        # correlatable without the call site remembering to pass anything.
        payload.update(get_request_fields())

        # `logger.info("...", extra={"chunks": 5})` lands as a real field.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # `default=str` rather than letting a stray UUID or datetime raise: a log line that
        # throws while being formatted loses the very information you were trying to keep.
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """The human-readable format, with the trace id appended when there is one."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        trace_id = get_request_fields().get("trace_id")
        return f"{line} [trace={trace_id}]" if trace_id else line


def configure_logging(level: str, *, log_format: str = "json") -> None:
    """Install the formatter on the root handler.

    Replaces existing handlers rather than adding to them, so calling this twice — the app
    factory and a test, say — does not double every line.
    """
    formatter: logging.Formatter = JsonFormatter() if log_format == "json" else TextFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
