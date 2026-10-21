"""The POST /v1/chat contract, and the persistence behind it."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import httpx2
import pytest
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from sutra.api.app import create_app
from sutra.api.chat import SessionNotFound, run_chat_turn
from sutra.api.schemas import ChatRequest
from sutra.api.streaming import sse_frame
from sutra.config import Settings
from sutra.db.engine import get_engine, get_sessionmaker
from sutra.db.models import Message as MessageRow
from sutra.db.models import Session as SessionRow
from sutra.generation.schema import AnswerDraft
from sutra.llm.adapters import RawResult
from sutra.llm.client import LLMClient
from sutra.prompts import PromptLoader
from sutra.retrieval.search import RetrievedChunk


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# HTTP contract — no database, no model
# ---------------------------------------------------------------------------


async def test_streaming_is_advertised_in_the_openapi_contract(client: httpx2.AsyncClient):
    """Phase 2 returned 501 here. Phase 4 implements it, so the 501 is gone from the schema.

    The behaviour is covered by the integration tests below, which need a database.
    """
    schema = (await client.get("/openapi.json")).json()
    responses = schema["paths"]["/v1/chat"]["post"]["responses"]
    assert "501" in responses
    assert "mode" in responses["501"]["description"]
    # The 501 that remains is about agent mode, not about streaming.
    assert "stream" not in responses["501"]["description"].lower()


async def test_agent_mode_is_refused(client: httpx2.AsyncClient):
    response = await client.post("/v1/chat", json={"query": "q", "user_id": "u", "mode": "agent"})
    assert response.status_code == 501
    assert "agent" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "u"},  # no query
        {"query": "q"},  # no user_id
        {"query": "", "user_id": "u"},  # empty query
        {"query": "q", "user_id": "u", "mode": "telepathy"},  # not in the enum
        {"query": "q", "user_id": "u", "session_id": "not-a-uuid"},
    ],
)
async def test_malformed_requests_are_rejected(client: httpx2.AsyncClient, payload: dict):
    assert (await client.post("/v1/chat", json=payload)).status_code == 422


async def test_the_endpoint_is_documented(client: httpx2.AsyncClient):
    """FastAPI's OpenAPI is the API reference (ARCHITECTURE §13), so the contract has to be in
    it rather than only in the code."""
    spec = (await client.get("/openapi.json")).json()
    operation = spec["paths"]["/v1/chat"]["post"]
    assert {"200", "404", "501", "422"} <= set(operation["responses"])


# ---------------------------------------------------------------------------
# the turn itself
# ---------------------------------------------------------------------------


class ScriptedAdapter:
    """Returns one prepared draft, so the turn runs without a network call."""

    name = "scripted"

    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.calls = 0

    async def complete(self, *, model, messages, max_output_tokens, temperature, timeout):
        raise AssertionError("unused")

    def _result(self, model: str) -> RawResult:
        return RawResult(
            text=self.draft.model_dump_json(),
            model=model,
            input_tokens=1200,
            output_tokens=90,
            cached_tokens=0,
            response_id="resp_scripted",
            parsed=self.draft,
        )

    async def structured(self, *, model, messages, schema, max_output_tokens, temperature, timeout):
        self.calls += 1
        return self._result(model)

    async def stream(
        self, *, model, messages, max_output_tokens, temperature, timeout, schema=None
    ):
        """What a schema-constrained provider stream looks like: JSON, in arbitrary pieces.

        The split points are deliberately awkward — mid-key and mid-word — because that is
        what the real thing does and it is what the partial-JSON reader has to survive.
        """
        assert schema is not None, "the answer stream must be schema-constrained"
        document = self.draft.model_dump_json()
        for i in range(0, len(document), 7):
            yield document[i : i + 7], None
        yield None, self._result(model)


def a_chunk(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        doc_slug="06_places_wiki",
        doc_title="Places of the Meridian Line",
        doc_type="wiki",
        chunk_no=n,
        heading="Kesh Ithar — waystone 40",
        text="- **Population:** 68,000",
        token_count=12,
        score=0.8,
        rank=n,
    )


@pytest.fixture(autouse=True)
async def _engine_per_event_loop() -> AsyncIterator[None]:
    yield
    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture
def turn_deps(monkeypatch: pytest.MonkeyPatch, settings: Settings):
    """Stub the two external edges — retrieval and the model — and keep persistence real."""
    chunks = [a_chunk(1), a_chunk(2)]
    monkeypatch.setattr(
        "sutra.api.chat.search",
        lambda *a, **kw: _async_value(chunks),
    )
    draft = AnswerDraft(answer="68,000.", citations=[1], confidence=0.9, refused=False)
    adapter = ScriptedAdapter(draft)
    return chunks, adapter, LLMClient(settings, adapter=adapter), PromptLoader(settings=settings)


async def _async_value(value):
    return value


async def _messages_for(session_id: uuid.UUID) -> list[MessageRow]:
    factory = get_sessionmaker()
    async with factory() as db:
        rows = (
            await db.execute(
                select(MessageRow)
                .where(MessageRow.session_id == session_id)
                .order_by(MessageRow.created_at, MessageRow.role)
            )
        ).scalars()
        return list(rows)


async def _cleanup(session_id: uuid.UUID) -> None:
    factory = get_sessionmaker()
    async with factory() as db:
        await db.execute(delete(SessionRow).where(SessionRow.id == session_id))
        await db.commit()


@pytest.mark.integration
async def test_a_turn_creates_a_session_and_persists_both_messages(settings: Settings, turn_deps):
    chunks, adapter, llm, loader = turn_deps
    response = await run_chat_turn(
        ChatRequest(query="What is the population of Kesh Ithar?", user_id="rohit"),
        settings=settings,
        client=llm,
        loader=loader,
    )
    try:
        assert response.answer == "68,000."
        assert [c.label for c in response.citations] == [1]
        assert response.retrieved_chunk_ids == [c.chunk_id for c in chunks]
        assert response.dropped_citations == []
        assert response.prompt_version == "v1"
        assert response.index_version == settings.index_version
        assert response.usage.cost_usd is not None
        assert adapter.calls == 1

        messages = await _messages_for(response.session_id)
        roles = {m.role for m in messages}
        assert roles == {"user", "assistant"}, "both halves of the turn must be recorded"
        assistant = next(m for m in messages if m.role == "assistant")
        # Recorded per message so an answer can be traced back to the prompt that made it.
        assert assistant.prompt_version == "v1"
        assert assistant.token_count == 1290
        assert assistant.id == response.message_id
    finally:
        await _cleanup(response.session_id)


@pytest.mark.integration
async def test_passing_a_session_id_continues_that_conversation(settings: Settings, turn_deps):
    _, _, llm, loader = turn_deps
    first = await run_chat_turn(
        ChatRequest(query="first question", user_id="rohit"),
        settings=settings,
        client=llm,
        loader=loader,
    )
    try:
        second = await run_chat_turn(
            ChatRequest(query="second question", user_id="rohit", session_id=first.session_id),
            settings=settings,
            client=llm,
            loader=loader,
        )
        assert second.session_id == first.session_id
        assert len(await _messages_for(first.session_id)) == 4
    finally:
        await _cleanup(first.session_id)


@pytest.mark.integration
async def test_another_users_session_is_not_usable(settings: Settings, turn_deps):
    """And it fails the same way an unknown id does.

    Distinguishing "not yours" from "does not exist" would tell a caller which session ids are
    real, which is a membership oracle.
    """
    _, _, llm, loader = turn_deps
    mine = await run_chat_turn(
        ChatRequest(query="q", user_id="rohit"), settings=settings, client=llm, loader=loader
    )
    try:
        with pytest.raises(SessionNotFound):
            await run_chat_turn(
                ChatRequest(query="q", user_id="someone-else", session_id=mine.session_id),
                settings=settings,
                client=llm,
                loader=loader,
            )
        with pytest.raises(SessionNotFound):
            await run_chat_turn(
                ChatRequest(query="q", user_id="rohit", session_id=uuid.uuid4()),
                settings=settings,
                client=llm,
                loader=loader,
            )
    finally:
        await _cleanup(mine.session_id)


@pytest.mark.integration
async def test_an_unknown_session_returns_404_over_http(client: httpx2.AsyncClient):
    response = await client.post(
        "/v1/chat",
        json={"query": "q", "user_id": "rohit", "session_id": str(uuid.uuid4())},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs.

    Hand-written rather than mocked: asserting on the bytes is the only way to know the wire
    format is right, and the format *is* the contract a client depends on.
    """
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name: str | None = None
        payload: str | None = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = line.removeprefix("data: ")
        assert name is not None, f"frame without an event name: {block!r}"
        assert payload is not None, f"frame without data: {block!r}"
        events.append((name, json.loads(payload)))
    return events


@pytest.fixture
def stream_deps(monkeypatch: pytest.MonkeyPatch, settings: Settings):
    """Same stubs as `turn_deps`, wired into the app so an HTTP request can use them."""
    chunks = [a_chunk(1), a_chunk(2)]
    monkeypatch.setattr("sutra.api.chat.search", lambda *a, **kw: _async_value(chunks))
    draft = AnswerDraft(
        answer='Kesh Ithar\'s population is 68,000, per the "places" wiki. [1]',
        citations=[1],
        confidence=0.9,
        refused=False,
    )
    adapter = ScriptedAdapter(draft)
    llm = LLMClient(settings, adapter=adapter)
    monkeypatch.setattr("sutra.api.routes_chat.run_chat_turn_stream", _stream_with(llm, settings))
    return draft, adapter


def _stream_with(llm: LLMClient, settings: Settings):
    from sutra.api.chat import run_chat_turn_stream as real

    def wrapped(request, *, session_id, **kwargs):
        return real(request, session_id=session_id, settings=settings, client=llm, **kwargs)

    return wrapped


@pytest.mark.integration
async def test_a_streamed_turn_emits_meta_tokens_citations_then_done(
    client: httpx2.AsyncClient, stream_deps
):
    draft, _ = stream_deps
    response = await client.post(
        "/v1/chat", json={"query": "population of Kesh Ithar?", "user_id": "rohit", "stream": True}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Without this a proxy may buffer the whole stream and deliver it as one block, which
    # looks exactly like a server that does not stream.
    assert response.headers["x-accel-buffering"] == "no"

    events = _parse_sse(response.text)
    names = [name for name, _ in events]

    # `meta` first: a client has the trace id even if the answer fails half way through.
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert names.count("citations") == 1
    assert names.index("citations") > names.index("token")
    assert "error" not in names

    meta = events[0][1]
    assert meta["trace_id"]
    session_id = uuid.UUID(meta["session_id"])
    try:
        # The tokens reassemble into exactly the answer — the whole point of option A.
        streamed = "".join(data["text"] for name, data in events if name == "token")
        assert streamed == draft.answer

        citations = next(data for name, data in events if name == "citations")
        assert [c["label"] for c in citations["citations"]] == [1]
        assert citations["dropped_citations"] == []

        done = events[-1][1]
        assert done["refused"] is False
        assert done["confidence"] == 0.9
        assert done["usage"]["input_tokens"] == 1200
        assert uuid.UUID(done["message_id"])
    finally:
        await _cleanup(session_id)


@pytest.mark.integration
async def test_a_streamed_turn_persists_the_same_two_messages_as_a_buffered_one(
    client: httpx2.AsyncClient, stream_deps
):
    """Streaming is a transport, not a different kind of turn."""
    draft, _ = stream_deps
    response = await client.post(
        "/v1/chat", json={"query": "population?", "user_id": "rohit", "stream": True}
    )
    events = _parse_sse(response.text)
    session_id = uuid.UUID(events[0][1]["session_id"])
    try:
        messages = await _messages_for(session_id)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].content == draft.answer
        # The trace id is on both rows, so an answer in the database can be opened as a trace.
        assert messages[0].trace_id == events[0][1]["trace_id"]
        assert messages[1].trace_id == events[0][1]["trace_id"]
    finally:
        await _cleanup(session_id)


@pytest.mark.integration
async def test_an_unknown_session_is_a_404_not_an_error_frame(
    client: httpx2.AsyncClient, stream_deps
):
    """A 200 whose body contains an apology is not an error a client can handle."""
    response = await client.post(
        "/v1/chat",
        json={
            "query": "q",
            "user_id": "rohit",
            "stream": True,
            "session_id": str(uuid.uuid4()),
        },
    )
    assert response.status_code == 404
    assert not response.headers["content-type"].startswith("text/event-stream")


@pytest.mark.integration
async def test_a_failure_part_way_through_becomes_a_terminal_error_frame(
    client: httpx2.AsyncClient, monkeypatch: pytest.MonkeyPatch, settings: Settings
):
    """Once the first byte is sent the status is gone, so the client has to be told in-band.

    A response that simply stops is indistinguishable from a dropped connection, and the
    client has no way to tell a truncated answer from a complete one.
    """
    chunks = [a_chunk(1)]
    monkeypatch.setattr("sutra.api.chat.search", lambda *a, **kw: _async_value(chunks))

    async def exploding_stream(*a, **kw):
        yield sse_frame("meta", {"trace_id": "t", "session_id": str(uuid.uuid4())})
        msg = "provider went away"
        raise RuntimeError(msg)

    monkeypatch.setattr("sutra.api.routes_chat.run_chat_turn_stream", exploding_stream)
    response = await client.post(
        "/v1/chat", json={"query": "q", "user_id": "rohit", "stream": True}
    )

    assert response.status_code == 200  # the status was already sent
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["meta", "error"]
    assert "failed" in events[-1][1]["detail"]
