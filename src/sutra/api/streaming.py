"""Server-Sent Events for `POST /v1/chat` (ARCHITECTURE §2, §13).

# CONCEPT: SSE over WebSockets for one-way token delivery. It is plain HTTP — one response
# with `content-type: text/event-stream` — so it works through proxies, needs no second
# protocol, and `curl -N` is a complete client. A WebSocket buys bidirectional frames that a
# chat completion does not use.
#
# The wire format is four fields separated by blank lines, and the two rules that matter are
# that every frame ends with `\\n\\n` and that a payload containing a newline must be split
# across several `data:` lines. We send JSON, which `json.dumps` keeps newline-free by
# escaping them, so one `data:` line per frame is always correct here.

Event shape (ARCHITECTURE §5):

    event: meta        — trace_id, session_id, prompt/index version, model. Sent first.
    event: token       — {"text": "..."} one piece of the answer
    event: citations   — resolved citations, once the answer object has closed
    event: done        — usage, cost, latency, confidence, refused, message_id
    event: error       — {"detail": "..."} terminal; nothing follows it
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

# What a proxy has to be told not to do. Without `X-Accel-Buffering`, nginx will happily
# buffer the whole stream and deliver it as one block, which looks exactly like a server that
# does not stream.
SSE_HEADERS = {
    "cache-control": "no-cache",
    "connection": "keep-alive",
    "x-accel-buffering": "no",
}


def sse_frame(event: str, data: Any) -> str:
    """One SSE frame. `data` is JSON-encoded, so it can never contain a bare newline."""
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


async def guarded(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Forward frames, turning a mid-stream exception into a terminal `error` frame.

    Once the first byte is sent the status line is gone, so an exception cannot become a 500 —
    the client would otherwise see a response that simply stops, indistinguishable from a
    network failure, and would have no way to tell a truncated answer from a complete one.
    The exception is logged with the trace id and the client is told in-band.
    """
    try:
        async for frame in stream:
            yield frame
    except Exception:
        logger.exception("chat.stream_failed")
        yield sse_frame(
            "error",
            {"detail": "The answer stream failed before it finished. Please try again."},
        )
