"""Streaming generation: the same prompt and schema as the buffered path, delivered in pieces.

The property worth protecting is the equivalence. If these two paths can disagree about an
answer, Phase 3's scoreboard only describes whichever one `make eval` happens to run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from sutra.config import Settings
from sutra.generation.answer import generate_answer
from sutra.generation.schema import AnswerDraft
from sutra.generation.stream import AnswerComplete, AnswerToken, generate_answer_stream
from sutra.llm.adapters import RawResult
from sutra.llm.client import LLMClient
from sutra.llm.types import Message
from sutra.prompts import PromptLoader
from sutra.retrieval.search import RetrievedChunk

DRAFT = AnswerDraft(
    answer='Tulvo Brann keeps it, per the "ledger" rolls. [1]\nHe holds the fourth seat.',
    citations=[1],
    confidence=0.85,
    refused=False,
)


def a_chunk(n: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{n}",
        doc_slug="04_merchant_ledger",
        doc_title="The Merchant Ledger",
        doc_type="ledger",
        chunk_no=n,
        heading="Nine Tables",
        text="Tulvo Brann holds the fourth seat.",
        token_count=10,
        score=0.9,
        rank=n,
    )


class StreamingAdapter:
    """Emits the draft as JSON in fixed-size pieces, as a schema-constrained provider does."""

    name = "streaming-fake"

    def __init__(
        self, draft: AnswerDraft, *, chunk_size: int = 5, ends_cleanly: bool = True
    ) -> None:
        self.draft = draft
        self.chunk_size = chunk_size
        self.ends_cleanly = ends_cleanly
        self.schema_seen: object = None

    async def complete(self, *, model, messages, max_output_tokens, temperature, timeout):
        raise AssertionError("unused")

    async def structured(self, *, model, messages, schema, max_output_tokens, temperature, timeout):
        return RawResult(text=self.draft.model_dump_json(), model=model, parsed=self.draft)

    async def stream(
        self, *, model, messages, max_output_tokens, temperature, timeout, schema=None
    ) -> AsyncIterator[tuple[str | None, RawResult | None]]:
        self.schema_seen = schema
        document = self.draft.model_dump_json()
        if not self.ends_cleanly:
            # A dropped connection: fragments stop and no completion event ever arrives.
            document = document[: len(document) // 2]
            for i in range(0, len(document), self.chunk_size):
                yield document[i : i + self.chunk_size], None
            return
        for i in range(0, len(document), self.chunk_size):
            yield document[i : i + self.chunk_size], None
        yield (
            None,
            RawResult(
                text=document,
                model=model,
                input_tokens=900,
                output_tokens=60,
                parsed=self.draft,
            ),
        )


async def _collect(
    chunks: Sequence[RetrievedChunk], adapter: StreamingAdapter, settings: Settings
) -> tuple[str, AnswerComplete]:
    client = LLMClient(settings, adapter=adapter)
    loader = PromptLoader(settings=settings)
    text = ""
    final: AnswerComplete | None = None
    async for event in generate_answer_stream(
        "Who keeps the ledger?", chunks, client=client, loader=loader, settings=settings
    ):
        if isinstance(event, AnswerToken):
            text += event.text
        else:
            final = event
    assert final is not None, "the stream must always end with a final event"
    return text, final


@pytest.mark.parametrize("chunk_size", [1, 3, 5, 17, 512])
async def test_the_tokens_reassemble_into_the_answer(settings: Settings, chunk_size: int):
    adapter = StreamingAdapter(DRAFT, chunk_size=chunk_size)
    text, final = await _collect([a_chunk()], adapter, settings)
    assert text == DRAFT.answer
    assert final.answer.answer == DRAFT.answer


async def test_the_stream_is_schema_constrained(settings: Settings):
    """Not an implementation detail: it is what keeps citations verifiable while streaming."""
    adapter = StreamingAdapter(DRAFT)
    await _collect([a_chunk()], adapter, settings)
    assert adapter.schema_seen is AnswerDraft


async def test_the_final_answer_matches_the_buffered_path_exactly(settings: Settings):
    """The equivalence this design exists for."""
    chunks = [a_chunk()]
    buffered = await generate_answer(
        "Who keeps the ledger?",
        chunks,
        client=LLMClient(settings, adapter=StreamingAdapter(DRAFT)),
        loader=PromptLoader(settings=settings),
        settings=settings,
    )
    _, streamed = await _collect(chunks, StreamingAdapter(DRAFT), settings)

    assert streamed.answer.answer == buffered.answer
    assert streamed.answer.refused == buffered.refused
    assert streamed.answer.confidence == buffered.confidence
    assert [c.chunk_id for c in streamed.answer.citations] == [
        c.chunk_id for c in buffered.citations
    ]
    assert streamed.answer.dropped_citations == buffered.dropped_citations
    assert streamed.answer.prompt_version == buffered.prompt_version


async def test_citations_are_resolved_against_the_retrieved_chunks(settings: Settings):
    _, final = await _collect([a_chunk(1)], StreamingAdapter(DRAFT), settings)
    assert [c.chunk_id for c in final.answer.citations] == ["chunk-1"]
    assert final.answer.citations[0].doc_slug == "04_merchant_ledger"


async def test_an_invented_citation_is_reported_not_hidden(settings: Settings):
    draft = DRAFT.model_copy(update={"citations": [1, 9]})
    _, final = await _collect([a_chunk(1)], StreamingAdapter(draft), settings)
    assert final.answer.dropped_citations == [9]


async def test_no_passages_short_circuits_without_calling_the_model(settings: Settings):
    adapter = StreamingAdapter(DRAFT)
    text, final = await _collect([], adapter, settings)
    assert text == ""
    assert final.answer.refused is True
    assert final.answer.short_circuited is True
    assert adapter.schema_seen is None  # the model was never called


async def test_a_dropped_stream_still_yields_a_final_event_with_what_arrived(
    settings: Settings,
):
    """A consumer that has emitted tokens must be told how it finished.

    "The connection went away" is not something the caller can infer from a missing event.
    """
    adapter = StreamingAdapter(DRAFT, ends_cleanly=False)
    text, final = await _collect([a_chunk()], adapter, settings)

    assert text  # some of the answer arrived
    assert DRAFT.answer.startswith(text)
    assert final.answer.answer == text.strip()
    # There is an answer, and there is no verified provenance for it. Both are true.
    assert final.answer.citations == []
    assert final.answer.confidence == 0.0


async def test_usage_and_cost_survive_the_stream(settings: Settings):
    _, final = await _collect([a_chunk()], StreamingAdapter(DRAFT), settings)
    assert final.answer.usage.input_tokens == 900
    assert final.answer.usage.output_tokens == 60
    assert final.answer.latency_ms > 0


async def test_the_prompt_sent_is_the_same_one_the_buffered_path_sends(settings: Settings):
    """Guards the equivalence at the input, not just the output."""
    seen: dict[str, Sequence[Message]] = {}

    class Recording(StreamingAdapter):
        async def structured(self, *, messages, **kw):
            seen["buffered"] = messages
            return await super().structured(messages=messages, **kw)

        async def stream(self, *, messages, **kw):
            seen["streamed"] = messages
            async for item in super().stream(messages=messages, **kw):
                yield item

    chunks = [a_chunk()]
    loader = PromptLoader(settings=settings)
    adapter = Recording(DRAFT)
    await generate_answer(
        "Who keeps the ledger?",
        chunks,
        client=LLMClient(settings, adapter=adapter),
        loader=loader,
        settings=settings,
    )
    await _collect(chunks, adapter, settings)

    assert seen["buffered"] == seen["streamed"]
