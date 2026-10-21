"""Query transforms (Phase 3 step 5). The model is stubbed; the contract is what matters."""

from __future__ import annotations

from typing import Any

from sutra.config import Settings
from sutra.prompts import PromptLoader
from sutra.retrieval.query_transform import (
    HypotheticalAnswer,
    TransformedQueries,
    gather_queries,
    transform_query,
)


class FakeClient:
    def __init__(self, queries: list[str] | None = None, *, fail: bool = False) -> None:
        self.queries = queries if queries is not None else ["rewritten query"]
        self.fail = fail
        self.calls = 0

    async def structured(self, messages: Any, schema: type, **kwargs: Any) -> Any:
        self.calls += 1
        if self.fail:
            msg = "provider exploded"
            raise RuntimeError(msg)

        class Response:
            parsed = (
                HypotheticalAnswer(passage="A hypothetical passage.")
                if schema is HypotheticalAnswer
                else TransformedQueries(queries=self.queries)
            )

        return Response()


def _settings(**over: object) -> Settings:
    return Settings(**{"query_rewrite_enabled": True, **over})  # type: ignore[arg-type]


async def test_no_transforms_means_no_model_call():
    """Off must be genuinely off: the default path should not pay for a call it does not use."""
    client = FakeClient()
    out = await transform_query(
        "Who is the Archivist?",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=Settings(),
    )
    assert out == ["Who is the Archivist?"]
    assert client.calls == 0


async def test_the_original_question_is_never_dropped():
    """A transform is a model call and can paraphrase away the one rare noun that would have
    matched. Keeping the question makes the stage strictly additive: a bad rewrite can cost
    ranking slots, never the answer."""
    client = FakeClient(["something else entirely"])
    out = await transform_query(
        "What did Oren Dask pay at the nineteenth?",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_settings(),
    )
    assert out[0] == "What did Oren Dask pay at the nineteenth?"
    assert "something else entirely" in out


async def test_a_rewrite_identical_to_the_question_is_deduped():
    """The common case for a simple question. Retrieving it twice would double its weight in
    the fusion for no reason."""
    client = FakeClient(["  who is the archivist?  "])
    out = await transform_query(
        "Who is the Archivist?",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_settings(),
    )
    assert out == ["Who is the Archivist?"]


async def test_the_query_count_is_capped():
    client = FakeClient([f"query {i}" for i in range(10)])
    out = await transform_query(
        "q",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_settings(multi_query_count=2),
    )
    assert len(out) == 3  # the original plus two


async def test_a_failed_rewrite_degrades_to_the_question():
    client = FakeClient(fail=True)
    out = await transform_query(
        "q",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_settings(),
    )
    assert out == ["q"]


async def test_hyde_adds_a_probe_without_replacing_the_question():
    """The hypothetical is a decoy for the embedder, not a substitute for what was asked."""
    client = FakeClient([])
    out = await transform_query(
        "Who drowned?",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_settings(query_rewrite_enabled=False, hyde_enabled=True),
    )
    assert out == ["Who drowned?", "A hypothetical passage."]


async def test_gather_queries_never_raises():
    """Retrieval must survive a transform outage; this is the seam that guarantees it."""

    class Exploding:
        async def structured(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("down")

    out = await gather_queries(
        "q",
        client=Exploding(),  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_settings(),
    )
    assert out == ["q"]
