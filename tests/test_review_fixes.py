"""The 2026-10-21 and 2026-10-22 review findings, one test each.

Each test is written against the *defect*, not the fix: it is the assertion that fails on the
code as reviewed. Where a fix could be mistaken for simply tightening a check, a negative
control sits beside it proving the valid case still passes — a guard that rejects everything
is not a guard.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from evals.runners.run_rag import Profile, slot_fingerprint
from pydantic import ValidationError

from sutra.config import CONTEXT_PREFIX_BUDGET_TOKENS, EMBEDDING_MAX_SEQUENCE_TOKENS, Settings
from sutra.ingestion.fingerprint import IndexFingerprint
from sutra.ingestion.run import OccupiedSlot, _trim_prefixes_to_budget
from sutra.llm.adapters import OpenAIAdapter, RawResult
from sutra.llm.client import LLMClient, client_scope
from sutra.llm.types import LLMResponse, Message
from sutra.retrieval.search import RetrievedChunk

# --------------------------------------------------------------------------- finding 1


def _fingerprint(**overrides: object) -> IndexFingerprint:
    base: dict[str, object] = {
        "chunk_strategy": "recursive",
        "chunk_tokens": 400,
        "chunk_overlap": 60,
        "chunker_version": 2,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "corpus_hash": "deadbeef",
    }
    return IndexFingerprint.model_validate({**base, **overrides})


@pytest.mark.asyncio
async def test_report_fingerprint_comes_from_the_slot_not_the_runners_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect: eleven Phase 3 reports over three indexes all carried slot 1's digest.

    The runner's `.env` describes slot 1 while the profile selects slot 3, which is the ordinary
    case. The fingerprint recorded must be the one the slot holds.
    """
    stored = _fingerprint(chunk_strategy="semantic", parent_child=True, child_tokens=200)

    async def fake_read(index_version: int) -> OccupiedSlot:
        assert index_version == 3
        return OccupiedSlot(raw=stored.model_dump(), fingerprint=stored)

    monkeypatch.setattr("evals.runners.run_rag.read_slot_fingerprint", fake_read)
    got = await slot_fingerprint(Profile(name="advanced", index_version=3))

    assert got.digest == stored.digest
    # And specifically *not* what the runner's own settings would have computed.
    assert got.digest != _fingerprint().digest
    assert got.chunk_strategy == "semantic"


@pytest.mark.asyncio
async def test_eval_refuses_a_slot_that_cannot_state_its_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows but no readable fingerprint is the ADR-022 third state. Ingestion refuses it; so
    must measurement, rather than substituting the runner's config as provenance."""

    async def fake_read(index_version: int) -> OccupiedSlot:
        return OccupiedSlot(raw={"chunk_tokens": 400}, fingerprint=None)

    monkeypatch.setattr("evals.runners.run_rag.read_slot_fingerprint", fake_read)
    with pytest.raises(SystemExit, match="no readable index fingerprint"):
        await slot_fingerprint(Profile(name="naive", index_version=1))


@pytest.mark.asyncio
async def test_eval_refuses_an_empty_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_read(index_version: int) -> None:
        return None

    monkeypatch.setattr("evals.runners.run_rag.read_slot_fingerprint", fake_read)
    with pytest.raises(SystemExit, match="is empty"):
        await slot_fingerprint(Profile(name="naive", index_version=7))


# --------------------------------------------------------------------------- finding 2


def test_semantic_strategy_without_parent_child_is_refused(settings: Settings) -> None:
    """The defect: the flat builder ignores `chunk_strategy` while the fingerprint records it,
    so `--strategy semantic` alone built a recursive index stamped "semantic"."""
    with pytest.raises(ValidationError, match="requires parent_child_enabled"):
        Settings(**{**settings.model_dump(), "chunk_strategy": "semantic"})


def test_semantic_strategy_with_parent_child_is_accepted(settings: Settings) -> None:
    """Negative control: the guard must reject only the combination that cannot be honoured."""
    accepted = Settings(
        **{**settings.model_dump(), "chunk_strategy": "semantic", "parent_child_enabled": True}
    )
    assert accepted.chunk_strategy == "semantic"


# --------------------------------------------------------------------------- finding 3


def test_contextual_prefixes_count_against_the_embedder_ceiling(settings: Settings) -> None:
    """The defect: the validator measured the bare chunk, but the embedded unit is prefix +
    chunk. At the default 400 + 60 the chunk fits and the real input does not."""
    base = {**settings.model_dump(), "chunk_tokens": 400, "chunk_overlap": 60}
    # Without prefixes this configuration is fine, and stays fine.
    assert Settings(**base).chunk_tokens == 400
    with pytest.raises(ValidationError, match="context prefix reserve"):
        Settings(**{**base, "contextual_prefixes_enabled": True})


def test_parent_child_plus_prefixes_still_fits(settings: Settings) -> None:
    """Negative control: the measured configuration — slot 3 — must remain legal.

    200 + 40 + 128 = 368, comfortably under 512. If this ever fails the reserve has outgrown
    the child budget and the index would have to be rebuilt, not the check relaxed.
    """
    accepted = Settings(
        **{
            **settings.model_dump(),
            "parent_child_enabled": True,
            "chunk_strategy": "semantic",
            "contextual_prefixes_enabled": True,
            "child_tokens": 200,
            "child_overlap": 40,
        }
    )
    budget = accepted.child_tokens + accepted.child_overlap + CONTEXT_PREFIX_BUDGET_TOKENS
    assert budget <= EMBEDDING_MAX_SEQUENCE_TOKENS


class _WordCounter:
    """One token per word, so the budget arithmetic in the test is readable."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def dims(self) -> int:
        return 384

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384


def test_an_overlong_prefix_is_trimmed_and_the_chunk_is_not() -> None:
    """The reserve is only a guarantee if something enforces it. The prefix is retrieval
    scaffolding and may lose its tail; the chunk is the indexed content and may not."""
    long_prefix = " ".join(f"w{i}" for i in range(CONTEXT_PREFIX_BUDGET_TOKENS + 40))
    rows: list[dict[str, object]] = [
        {"context_prefix": long_prefix, "text": "the body", "metadata": {"doc_slug": "d"}},
        {"context_prefix": "short prefix", "text": "the body", "metadata": {"doc_slug": "d"}},
        {"context_prefix": None, "text": "the body", "metadata": {"doc_slug": "d"}},
    ]

    assert _trim_prefixes_to_budget(rows, _WordCounter()) == 1
    assert len(str(rows[0]["context_prefix"]).split()) == CONTEXT_PREFIX_BUDGET_TOKENS
    assert str(long_prefix).startswith(str(rows[0]["context_prefix"]))
    # Untouched: within budget, and absent.
    assert rows[1]["context_prefix"] == "short prefix"
    assert rows[2]["context_prefix"] is None
    assert all(r["text"] == "the body" for r in rows)


# --------------------------------------------------------------------------- finding 4


class _Adapter:
    name = "fake"
    closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_client_scope_closes_a_client_it_created(settings: Settings) -> None:
    """The defect: `client or LLMClient(...)` built a provider SDK — and an httpx connection
    pool — per call, and nothing closed it."""
    created: list[LLMClient] = []
    adapter = _Adapter()

    class _Tracked(LLMClient):
        def __init__(self, settings: Settings | None = None) -> None:
            super().__init__(settings=settings, adapter=adapter)  # type: ignore[arg-type]
            created.append(self)

    import sutra.llm.client as client_module

    original = client_module.LLMClient
    client_module.LLMClient = _Tracked  # type: ignore[misc]
    try:
        async with client_scope(None, settings) as made:
            assert made in created
            assert adapter.closed is False
    finally:
        client_module.LLMClient = original  # type: ignore[misc]
    assert adapter.closed is True


@pytest.mark.asyncio
async def test_client_scope_does_not_close_an_injected_client(settings: Settings) -> None:
    """Negative control, and the point of the whole fix: a borrowed pool must outlive the call.
    Closing the caller's client would break the next stage of the same request."""
    adapter = _Adapter()
    injected = LLMClient(settings=settings, adapter=adapter)  # type: ignore[arg-type]

    async with client_scope(injected, settings) as got:
        assert got is injected
    assert adapter.closed is False


@pytest.mark.asyncio
async def test_search_passes_one_client_to_both_model_backed_stages(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing and the query transforms must share the caller's client, not build their own."""
    injected = LLMClient(settings=settings, adapter=_Adapter())  # type: ignore[arg-type]
    seen: list[object] = []

    async def fake_route(question: str, **kwargs: object) -> object:
        seen.append(kwargs["client"])

        class _D:
            doc_types = ()

        return _D()

    async def fake_gather(question: str, **kwargs: object) -> list[str]:
        seen.append(kwargs["client"])
        return [question]

    async def fake_retrieve(query: str, **kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr("sutra.retrieval.router.route", fake_route)
    monkeypatch.setattr("sutra.retrieval.query_transform.gather_queries", fake_gather)
    monkeypatch.setattr("sutra.retrieval.search._retrieve", fake_retrieve)

    from sutra.retrieval.search import search

    await search("q", session=None, settings=settings, client=injected)  # type: ignore[arg-type]
    assert seen == [injected, injected]


# --------------------------------------------------------------------------- finding 5


def test_a_misspelled_profile_key_is_refused() -> None:
    """The defect: `extra` defaulted to "ignore", so `rerank_enabed: true` loaded as a profile
    with reranking off — and filed its report under a name claiming otherwise."""
    with pytest.raises(ValidationError, match="rerank_enabed"):
        Profile.model_validate({"name": "hybrid_rerank", "rerank_enabed": True})


@pytest.mark.parametrize(
    "name",
    [
        "naive",
        "semantic",
        "contextual",
        "hybrid",
        "keyword_only",
        "hybrid_rerank",
        "multiquery",
        "hyde",
        "routed",
        "advanced",
    ],
)
def test_every_committed_profile_loads_under_forbid(name: str) -> None:
    """Negative control over the committed profiles: none of the Phase 3 runs was measuring a
    configuration other than the one its file names."""
    assert Profile.load(name).name


# ===========================================================================================
# The 2026-10-22 review. Findings 1 and 2 of that round live with their subjects, in
# `test_observability.py` (the `request` contextmanager) and `test_api_feedback.py` (scoring
# off the event loop); the two below have no closer home.
# ===========================================================================================

# --------------------------------------------------------- 2026-10-22, finding 3


class _Usage:
    def __init__(self, inp: int, out: int, reasoning: int = 0) -> None:
        self.input_tokens = inp
        self.output_tokens = out
        self.input_tokens_details = type("D", (), {"cached_tokens": 0})()
        self.output_tokens_details = type("D", (), {"reasoning_tokens": reasoning})()


class _Response:
    def __init__(self, status: str, *, reason: str | None = None) -> None:
        self.id = "resp_1"
        self.model = "gpt-6-luna"
        self.status = status
        self.usage = _Usage(900, 512, reasoning=500)
        self.output_parsed = None
        self.incomplete_details = type("I", (), {"reason": reason})() if reason else None


class _Event:
    def __init__(self, etype: str, response: object = None, delta: str = "") -> None:
        self.type = etype
        self.response = response
        self.delta = delta


def _adapter_streaming(events: list[_Event]) -> OpenAIAdapter:
    """An `OpenAIAdapter` with the provider's event stream replaced."""
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)

    async def fake_events(kwargs: dict[str, object], schema: object) -> AsyncIterator[object]:
        for event in events:
            yield event

    adapter._raw_stream_events = fake_events  # type: ignore[method-assign,assignment]
    return adapter


async def _drain(adapter: OpenAIAdapter) -> RawResult:
    final: RawResult | None = None
    async for _delta, result in adapter.stream(
        model="gpt-6-luna",
        messages=[Message(role="user", content="q")],
        max_output_tokens=512,
        temperature=None,
        timeout=10.0,
    ):
        if result is not None:
            final = result
    assert final is not None
    return final


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("etype", "status"),
    [("response.incomplete", "incomplete"), ("response.failed", "failed")],
)
async def test_a_stream_that_did_not_complete_is_still_billed(etype: str, status: str) -> None:
    """The defect: the loop matched only `response.completed`, so `final` stayed unset and the
    fallback reported a *zero-usage* result. A truncated streamed answer — the known-issue-10
    case, where reasoning eats `max_output_tokens` — was recorded as costing nothing, which is
    the one call whose cost you most want to see.
    """
    adapter = _adapter_streaming(
        [
            _Event("response.output_text.delta", delta='{"answer": "The Arch'),
            _Event(etype, _Response(status, reason="max_output_tokens")),
        ]
    )
    final = await _drain(adapter)

    assert final.input_tokens == 900
    assert final.output_tokens == 512
    assert final.reasoning_tokens == 500
    assert final.status == status
    assert final.incomplete_reason == "max_output_tokens"
    # The partial text still comes back: ADR-025 tolerates truncation on this path on purpose,
    # since tokens are already with the caller. The bug was the accounting, not the tolerance.
    assert final.text == '{"answer": "The Arch'


@pytest.mark.asyncio
async def test_a_completed_stream_is_unchanged() -> None:
    """Negative control: widening the match must not alter the happy path."""
    adapter = _adapter_streaming(
        [
            _Event("response.output_text.delta", delta="hello"),
            _Event("response.completed", _Response("completed")),
        ]
    )
    final = await _drain(adapter)
    assert final.text == "hello"
    assert final.output_tokens == 512
    assert final.status == "completed"
    assert final.incomplete_reason is None


@pytest.mark.asyncio
async def test_a_stream_with_no_terminal_event_at_all_still_reports_what_arrived() -> None:
    """A dropped connection. Now genuinely the only case the zero-usage fallback covers."""
    adapter = _adapter_streaming([_Event("response.output_text.delta", delta="half")])
    final = await _drain(adapter)
    assert final.text == "half"
    assert final.input_tokens == 0
    assert final.status == "no_terminal_event"


# --------------------------------------------------------- 2026-10-22, finding 4


@pytest.mark.asyncio
async def test_generate_answer_closes_a_client_it_had_to_build(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`client_scope` reached the router and the transforms but not the generation paths, which
    kept `client or LLMClient(...)` and so leaked a pool per call."""
    import sutra.generation.answer as answer_module

    adapter = _Adapter()
    built: list[object] = []

    class _Tracked(LLMClient):
        def __init__(self, settings: Settings | None = None) -> None:
            super().__init__(settings=settings, adapter=adapter)  # type: ignore[arg-type]
            built.append(self)

        async def structured(self, *a: object, **k: object) -> LLMResponse:
            raise RuntimeError("stop here; ownership is what is under test")

    import sutra.llm.client as client_module

    # Both names, deliberately. `client_scope` resolves `LLMClient` in its own module, but
    # `answer.py` imported the name directly, so patching only one lets the *unfixed* code
    # build a real client and reach the provider — a test that fails by making a paid API
    # call is not a test. Patching both means this can only ever fail as an assertion.
    monkeypatch.setattr(client_module, "LLMClient", _Tracked)
    monkeypatch.setattr(answer_module, "LLMClient", _Tracked)

    with pytest.raises(RuntimeError, match="stop here"):
        await answer_module.generate_answer(
            "q",
            [
                RetrievedChunk(
                    chunk_id="c1",
                    doc_slug="d",
                    doc_title="D",
                    doc_type="story",
                    chunk_no=1,
                    heading=None,
                    text="t",
                    token_count=1,
                    score=1.0,
                    rank=1,
                )
            ],
            settings=settings,
        )

    assert built, "no client was built, so this test is not exercising the fallback path"
    assert adapter.closed is True


@pytest.mark.asyncio
async def test_the_no_context_short_circuit_builds_no_client_at_all(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one path that makes no model call used to be the one that built a provider SDK for
    nothing, because the client was created above the `if not chunks` return."""
    built: list[object] = []

    class _Tracked(LLMClient):
        def __init__(self, settings: Settings | None = None) -> None:
            built.append(self)
            super().__init__(settings=settings, adapter=_Adapter())  # type: ignore[arg-type]

    import sutra.generation.answer as answer_module
    import sutra.llm.client as client_module

    monkeypatch.setattr(client_module, "LLMClient", _Tracked)
    monkeypatch.setattr(answer_module, "LLMClient", _Tracked)

    result = await answer_module.generate_answer("q", [], settings=settings)
    assert result.short_circuited is True
    assert built == []
