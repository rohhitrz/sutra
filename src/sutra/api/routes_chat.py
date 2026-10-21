"""The chat endpoint.

Thin on purpose: HTTP concerns only. The turn itself lives in `sutra.api.chat`, so the same
orchestration serves the eval runner, and later the MCP server and the voice pipeline, without
going through HTTP (ARCHITECTURE §1).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from sutra.api.chat import (
    SessionNotFound,
    resolve_session,
    run_chat_turn,
    run_chat_turn_stream,
)
from sutra.api.schemas import ChatRequest, ChatResponse
from sutra.api.streaming import SSE_HEADERS, guarded

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=None,
    responses={
        200: {
            "description": (
                "A `ChatResponse`, or an SSE stream of `meta`/`token`/`citations`/`done` "
                "events when `stream` is true."
            ),
            "model": ChatResponse,
        },
        404: {"description": "No such session for this user"},
        501: {"description": "Requested a mode this phase does not implement"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse | StreamingResponse:
    """Answer one question from the corpus, with citations.

    Omit `session_id` to start a conversation; pass the one you were given back to continue it.
    Both the question and the answer are persisted.

    With `stream: true` the answer arrives as Server-Sent Events. It is the *same* prompt and
    schema as the buffered form (`sutra.generation.stream`), so the two cannot differ in
    quality — the tokens are lifted out of the schema-constrained JSON as it arrives.
    """
    if request.mode != "rag":
        # Refuse rather than silently downgrade. A caller asking for agent mode and getting a
        # RAG answer would conclude agent mode works.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"mode={request.mode!r} is not implemented in this phase; use mode='rag'.",
        )

    if request.stream:
        # The session is resolved before the response begins, so a bad id is still a 404.
        # Starting the response and then discovering it would leave the client with a 200
        # whose body contains an apology. It also keeps the whole generator driven by one
        # task — see the CONCEPT note on `run_chat_turn_stream`.
        try:
            session_id = await resolve_session(request)
        except SessionNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"session {exc.args[0]} not found",
            ) from exc

        return StreamingResponse(
            guarded(run_chat_turn_stream(request, session_id=session_id)),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        return await run_chat_turn(request)
    except SessionNotFound as exc:
        # 404 for both "no such session" and "not yours": see SessionNotFound.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"session {exc.args[0]} not found",
        ) from exc
