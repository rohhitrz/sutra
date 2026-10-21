"""The POST /v1/chat contract, and the persistence behind it."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx2
import pytest
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from sutra.api.app import create_app
from sutra.api.chat import SessionNotFound, run_chat_turn
from sutra.api.schemas import ChatRequest
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


async def test_streaming_is_refused_rather_than_silently_downgraded(client: httpx2.AsyncClient):
    """A caller asking for stream=true and getting a buffered body would conclude streaming
    works. Phase 2 does not implement it, so it says so."""
    response = await client.post("/v1/chat", json={"query": "q", "user_id": "u", "stream": True})
    assert response.status_code == 501
    assert "stream" in response.json()["detail"].lower()


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

    async def structured(self, *, model, messages, schema, max_output_tokens, temperature, timeout):
        self.calls += 1
        return RawResult(
            text=self.draft.model_dump_json(),
            model=model,
            input_tokens=1200,
            output_tokens=90,
            cached_tokens=0,
            response_id="resp_scripted",
            parsed=self.draft,
        )

    async def stream(self, *, model, messages, max_output_tokens, temperature, timeout):
        raise AssertionError("unused")
        yield None, None


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
