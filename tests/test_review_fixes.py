"""The 2026-10-21 review findings, one test each.

Each test is written against the *defect*, not the fix: it is the assertion that fails on the
code as reviewed. Where a fix could be mistaken for simply tightening a check, a negative
control sits beside it proving the valid case still passes — a guard that rejects everything
is not a guard.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from evals.runners.run_rag import Profile
from pydantic import ValidationError

from sutra.config import CONTEXT_PREFIX_BUDGET_TOKENS, EMBEDDING_MAX_SEQUENCE_TOKENS, Settings
from sutra.ingestion.run import _trim_prefixes_to_budget
from sutra.llm.client import LLMClient, client_scope

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
