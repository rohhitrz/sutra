"""POST /v1/feedback: the one signal that comes from outside the system."""

from __future__ import annotations

import threading
import uuid
from collections.abc import AsyncIterator, Iterator

import httpx2
import pytest
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from sutra.api.app import create_app
from sutra.db.engine import get_engine, get_sessionmaker
from sutra.db.models import Feedback as FeedbackRow
from sutra.db.models import Message as MessageRow
from sutra.db.models import Session as SessionRow
from sutra.observability.tracing import NoOpTracer, get_tracer, set_tracer


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
async def _engine_per_event_loop() -> AsyncIterator[None]:
    yield
    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


class ScoringTracer(NoOpTracer):
    """Records what would have been sent to Langfuse, and on which thread."""

    def __init__(self) -> None:
        self.scores: list[dict[str, object]] = []
        self.threads: list[int] = []

    def score(
        self, *, trace_id: str, name: str, value: float | str, comment: str | None = None
    ) -> bool:
        self.threads.append(threading.get_ident())
        self.scores.append({"trace_id": trace_id, "name": name, "value": value, "comment": comment})
        return True


@pytest.fixture
def tracer() -> Iterator[ScoringTracer]:
    original = get_tracer()
    spy = ScoringTracer()
    set_tracer(spy)
    yield spy
    set_tracer(original)


async def _seed_message(trace_id: str | None) -> tuple[uuid.UUID, uuid.UUID]:
    """A real session and assistant message, so the foreign key has something to point at."""
    factory = get_sessionmaker()
    async with factory() as db:
        session = SessionRow(user_id="rohit")
        db.add(session)
        await db.flush()
        message = MessageRow(
            session_id=session.id,
            role="assistant",
            content="Tulvo Brann keeps it.",
            prompt_version="v1",
            trace_id=trace_id,
        )
        db.add(message)
        await db.flush()
        ids = (session.id, message.id)
        await db.commit()
    return ids


async def _cleanup(session_id: uuid.UUID) -> None:
    factory = get_sessionmaker()
    async with factory() as db:
        await db.execute(delete(SessionRow).where(SessionRow.id == session_id))
        await db.commit()


async def _feedback_rows(trace_id: str) -> list[FeedbackRow]:
    factory = get_sessionmaker()
    async with factory() as db:
        rows = (
            await db.execute(select(FeedbackRow).where(FeedbackRow.trace_id == trace_id))
        ).scalars()
        return list(rows)


# --------------------------------------------------------------- the contract


@pytest.mark.parametrize("rating", [2, 0, -2, "up"])
async def test_only_thumbs_are_accepted(client: httpx2.AsyncClient, rating: object):
    """±1 is what the feedback table's check constraint enforces, so the API enforces it too."""
    response = await client.post("/v1/feedback", json={"rating": rating, "trace_id": "t"})
    assert response.status_code == 422


async def test_feedback_must_identify_an_answer(client: httpx2.AsyncClient):
    """A rating that points at nothing is unattributable, so it is rejected, not stored."""
    response = await client.post("/v1/feedback", json={"rating": 1})
    assert response.status_code == 422
    assert "trace_id" in response.text


async def test_the_endpoint_is_in_the_openapi_contract(client: httpx2.AsyncClient):
    schema = (await client.get("/openapi.json")).json()
    assert "/v1/feedback" in schema["paths"]


# ------------------------------------------------------------------ behaviour


@pytest.mark.integration
async def test_feedback_is_stored_and_the_trace_is_scored(
    client: httpx2.AsyncClient, tracer: ScoringTracer
):
    trace_id = uuid.uuid4().hex
    session_id, message_id = await _seed_message(trace_id)
    try:
        response = await client.post(
            "/v1/feedback",
            json={
                "rating": -1,
                "trace_id": trace_id,
                "message_id": str(message_id),
                "comment": "the figure is not in the passages",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["rating"] == -1
        assert body["scored"] is True

        rows = await _feedback_rows(trace_id)
        assert len(rows) == 1
        assert rows[0].rating == -1
        assert rows[0].comment == "the figure is not in the passages"
        assert rows[0].message_id == message_id

        assert tracer.scores == [
            {
                "trace_id": trace_id,
                "name": "user-feedback",
                "value": -1,
                "comment": "the figure is not in the passages",
            }
        ]
    finally:
        await _cleanup(session_id)


@pytest.mark.integration
async def test_a_message_id_alone_finds_its_own_trace(
    client: httpx2.AsyncClient, tracer: ScoringTracer
):
    """A client holding only a message id still gets the score on the right trace."""
    trace_id = uuid.uuid4().hex
    session_id, message_id = await _seed_message(trace_id)
    try:
        response = await client.post(
            "/v1/feedback", json={"rating": 1, "message_id": str(message_id)}
        )
        assert response.status_code == 201
        assert response.json()["trace_id"] == trace_id
        assert tracer.scores[0]["trace_id"] == trace_id
    finally:
        await _cleanup(session_id)


@pytest.mark.integration
async def test_an_unknown_message_is_a_404_rather_than_an_integrity_error(
    client: httpx2.AsyncClient, tracer: ScoringTracer
):
    response = await client.post(
        "/v1/feedback", json={"rating": 1, "message_id": str(uuid.uuid4())}
    )
    assert response.status_code == 404
    assert tracer.scores == []


@pytest.mark.integration
async def test_scoring_does_not_run_on_the_event_loop(
    client: httpx2.AsyncClient, tracer: ScoringTracer
):
    """`score` is synchronous and flushes — a blocking provider round trip.

    Called directly from the async handler it parks the loop, and every concurrent SSE stream
    with it (CLAUDE.md: no blocking calls inside request handlers). It belongs on a worker
    thread, like the ONNX embedder. Asserting the thread is the only way to see that from a
    test; asserting the latency would be a race.
    """
    trace_id = uuid.uuid4().hex
    session_id, _message_id = await _seed_message(trace_id)
    loop_thread = threading.get_ident()
    try:
        response = await client.post("/v1/feedback", json={"rating": 1, "trace_id": trace_id})
        assert response.status_code == 201
        assert tracer.threads, "score was never called"
        assert tracer.threads[0] != loop_thread
    finally:
        await _cleanup(session_id)


async def test_a_rating_is_still_stored_when_scoring_is_unavailable(client: httpx2.AsyncClient):
    """The database is the record of truth; Langfuse is a view onto it.

    A tracing outage must not lose a user's rating, so the row is written before the score
    is sent and `scored` reports the difference honestly.
    """
    original = get_tracer()
    set_tracer(NoOpTracer())  # tracing off: nothing to score against
    trace_id = uuid.uuid4().hex
    try:
        response = await client.post("/v1/feedback", json={"rating": 1, "trace_id": trace_id})
        assert response.status_code == 201
        assert response.json()["scored"] is False
        assert len(await _feedback_rows(trace_id)) == 1
    finally:
        set_tracer(original)
        factory = get_sessionmaker()
        async with factory() as db:
            await db.execute(delete(FeedbackRow).where(FeedbackRow.trace_id == trace_id))
            await db.commit()


@pytest.mark.integration
async def test_more_than_one_rating_per_trace_is_allowed(
    client: httpx2.AsyncClient, tracer: ScoringTracer
):
    """Someone changing their mind is data, not a conflict to reject."""
    trace_id = uuid.uuid4().hex
    session_id, message_id = await _seed_message(trace_id)
    try:
        for rating in (1, -1):
            response = await client.post(
                "/v1/feedback",
                json={"rating": rating, "trace_id": trace_id, "message_id": str(message_id)},
            )
            assert response.status_code == 201
        rows = await _feedback_rows(trace_id)
        assert sorted(r.rating for r in rows) == [-1, 1]
    finally:
        await _cleanup(session_id)
