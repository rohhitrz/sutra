"""Contextual retrieval prefixes (Phase 3 step 2).

No network: the LLM client is faked. What is worth testing here is the caching key, the
degradation path when a call fails, and the separation between what is indexed and what is
stored — not whether a model writes a good sentence, which only the eval can say.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from sutra.config import Settings
from sutra.ingestion.contextualize import (
    ChunkToContextualise,
    ContextPrefix,
    content_hash,
    contextualised_text,
    generate_prefixes,
)
from sutra.prompts import PromptLoader


class FakeClient:
    """Returns a prefix naming the chunk, and counts calls so caching can be asserted on."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def structured(self, messages: Sequence[Any], schema: type, **kwargs: Any) -> Any:
        self.calls += 1
        if self.fail:
            msg = "provider exploded"
            raise RuntimeError(msg)

        class Response:
            parsed = ContextPrefix(prefix="situated in its document")

        return Response()


def _chunks(n: int = 3) -> list[ChunkToContextualise]:
    return [
        ChunkToContextualise(
            key=content_hash(doc_slug="d", text=f"chunk {i}", prompt_version="v1"),
            doc_slug="d",
            doc_title="A Document",
            heading="A Section",
            text=f"chunk {i}",
        )
        for i in range(n)
    ]


def test_the_cache_key_covers_the_prompt_version():
    """A rewritten prompt writes different prefixes. Reusing a cached one across versions
    would put two prompt versions in one index, so the slot would describe a configuration
    that never ran."""
    a = content_hash(doc_slug="d", text="same text", prompt_version="v1")
    b = content_hash(doc_slug="d", text="same text", prompt_version="v2")
    assert a != b


def test_the_cache_key_covers_the_document():
    """The same sentence in two documents needs two different prefixes — the whole job is to
    say where the passage sits."""
    a = content_hash(doc_slug="one", text="same text", prompt_version="v1")
    b = content_hash(doc_slug="two", text="same text", prompt_version="v1")
    assert a != b


async def test_cached_prefixes_cost_no_calls():
    """Idempotent re-ingestion has to be free in money as well as in rows. Without this,
    every `make ingest` is another hundred model calls."""
    chunks = _chunks()
    cached = {c.key: "from the cache" for c in chunks}
    client = FakeClient()

    resolved = await generate_prefixes(
        chunks,
        documents={"d": "the whole document"},
        cached=cached,
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=Settings(),
    )

    assert client.calls == 0
    assert set(resolved) == {c.key for c in chunks}
    assert all(v == "from the cache" for v in resolved.values())


async def test_only_uncached_chunks_are_generated():
    chunks = _chunks(3)
    client = FakeClient()

    resolved = await generate_prefixes(
        chunks,
        documents={"d": "the whole document"},
        cached={chunks[0].key: "cached one"},
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=Settings(),
    )

    assert client.calls == 2
    assert resolved[chunks[0].key] == "cached one"
    assert resolved[chunks[1].key] == "situated in its document"


async def test_a_failed_prefix_does_not_fail_the_ingest():
    """Degrading to the step 1 behaviour — the chunk indexed bare — beats refusing to build
    the index because one call in a hundred timed out."""
    chunks = _chunks(2)
    client = FakeClient(fail=True)

    resolved = await generate_prefixes(
        chunks,
        documents={"d": "the whole document"},
        cached={},
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=Settings(),
    )

    assert resolved == {}


def test_the_prefix_is_indexed_but_the_stored_text_is_untouched():
    """The separation the whole step depends on: search sees the contextualised form, while
    citations and the eval's 72 verbatim quotes match against the stored slice."""
    assert contextualised_text("Situating sentence.", "the chunk") == (
        "Situating sentence.\n\nthe chunk"
    )
    assert contextualised_text(None, "the chunk") == "the chunk"


def test_the_prompt_refuses_an_empty_prefix():
    """An empty string would be indexed as a leading blank line and quietly do nothing."""
    with pytest.raises(ValueError, match="at least 1 character"):
        ContextPrefix(prefix="")
