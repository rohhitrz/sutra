"""The chat endpoint.

Thin on purpose: HTTP concerns only. The turn itself lives in `sutra.api.chat`, so the same
orchestration serves the eval runner, and later the MCP server and the voice pipeline, without
going through HTTP (ARCHITECTURE §1).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from sutra.api.chat import SessionNotFound, run_chat_turn
from sutra.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        404: {"description": "No such session for this user"},
        501: {"description": "Requested a mode or transport this phase does not implement"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer one question from the corpus, with citations.

    Omit `session_id` to start a conversation; pass the one you were given back to continue it.
    Both the question and the answer are persisted.
    """
    # Refuse rather than silently downgrade. A caller asking for `stream: true` and getting a
    # buffered JSON body would conclude streaming works.
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming is not implemented in this phase; call with stream=false.",
        )
    if request.mode != "rag":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"mode={request.mode!r} is not implemented in this phase; use mode='rag'.",
        )

    try:
        return await run_chat_turn(request)
    except SessionNotFound as exc:
        # 404 for both "no such session" and "not yours": see SessionNotFound.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"session {exc.args[0]} not found",
        ) from exc
