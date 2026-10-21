"""The feedback endpoint (ARCHITECTURE §10, §13).

# CONCEPT: Feedback is the only signal in an LLM system that comes from outside the system.
# Every other number here — recall, faithfulness, a judge's score — is Sutra measuring
# itself, with the judge's own biases baked in (ADR-017's same-family caveat). A thumbs-down
# on a specific trace is the one input that can contradict all of them, which is why it is
# stored *and* attached to the trace: stored so it can become an eval row, attached so the
# answer, its retrieved passages and its per-stage timings are one click away from the
# complaint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from sutra.api.schemas import FeedbackRequest, FeedbackResponse
from sutra.db.engine import get_sessionmaker
from sutra.db.models import Feedback as FeedbackRow
from sutra.db.models import Message as MessageRow
from sutra.observability.tracing import get_tracer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"description": "No such message"}},
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record a rating for an answer, and score its trace.

    Give `trace_id` (from a chat response or a `meta` frame), `message_id`, or both.
    """
    factory = get_sessionmaker()
    trace_id = request.trace_id

    async with factory() as db:
        if request.message_id is not None:
            # The message has to exist: the column is a foreign key, so inserting against a
            # missing id would surface as an integrity error rather than a clear 404.
            row = (
                await db.execute(
                    select(MessageRow.id, MessageRow.trace_id).where(
                        MessageRow.id == request.message_id
                    )
                )
            ).first()
            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"message {request.message_id} not found",
                )
            # A message knows its own trace, so a caller holding only a message id still
            # gets the score attached to the right trace.
            trace_id = trace_id or row.trace_id

        feedback = FeedbackRow(
            trace_id=trace_id,
            message_id=request.message_id,
            rating=request.rating,
            comment=request.comment,
        )
        db.add(feedback)
        await db.flush()
        feedback_id = feedback.id
        await db.commit()

    # Stored first, scored second, and deliberately in that order: the database is the record
    # of truth and Langfuse is a view onto it, so a tracing outage must not lose a rating.
    scored = False
    if trace_id:
        # `score` reports whether it reached a backend. It swallows its own failures by
        # design — observability must not break a request — so this bool is how a caller
        # learns that their rating is recorded but not visible in Langfuse.
        scored = get_tracer().score(
            trace_id=trace_id,
            name="user-feedback",
            value=request.rating,
            comment=request.comment,
        )

    logger.info(
        "feedback.recorded",
        extra={
            "feedback_id": str(feedback_id),
            "trace_id": trace_id,
            "rating": request.rating,
            "has_comment": request.comment is not None,
            "scored": scored,
        },
    )

    return FeedbackResponse(
        id=feedback_id,
        trace_id=trace_id,
        message_id=request.message_id,
        rating=request.rating,
        scored=scored,
    )
