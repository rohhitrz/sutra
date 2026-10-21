"""The 2026-10-21 review findings, one test each.

Each test is written against the *defect*, not the fix: it is the assertion that fails on the
code as reviewed. Where a fix could be mistaken for simply tightening a check, a negative
control sits beside it proving the valid case still passes — a guard that rejects everything
is not a guard.
"""

from __future__ import annotations

import pytest
from evals.runners.run_rag import Profile
from pydantic import ValidationError

from sutra.config import Settings
from sutra.llm.client import LLMClient, client_scope

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
