"""Generating an answer as a stream, without a second answer prompt.

The non-streaming path (`generate_answer`) and this one send **the same prompt, constrained to
the same schema, to the same model**. The only difference is that the provider's output arrives
in fragments and the `answer` field is lifted out of the partial JSON as it does
(`partial_json`), so a reader sees text immediately while citations, confidence and the refusal
flag are resolved from the completed object exactly as before.

That equivalence is the point. A prose-shaped `answer.v2` would have been easier to stream and
would have forked generation into two paths of different quality — leaving Phase 3's scoreboard
describing only the one `make eval` happens to run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel

from sutra.config import Settings, get_settings
from sutra.generation.answer import (
    MAX_ANSWER_TOKENS,
    PROMPT_NAME,
    format_context,
    no_context_answer,
    resolve_citations,
    split_prompt,
)
from sutra.generation.partial_json import StringFieldStreamer, extract_string_field
from sutra.generation.schema import Answer, AnswerDraft
from sutra.llm.client import LLMClient
from sutra.llm.types import StreamDelta, StreamDone, Usage
from sutra.observability.tracing import stage
from sutra.prompts import PromptLoader
from sutra.retrieval.search import RetrievedChunk

logger = logging.getLogger(__name__)


class AnswerToken(BaseModel):
    """A piece of the answer text, ready to forward to a reader."""

    type: str = "token"
    text: str


class AnswerComplete(BaseModel):
    """The finished answer, once the object has closed and citations are resolved."""

    type: str = "complete"
    answer: Answer


AnswerStreamEvent = AnswerToken | AnswerComplete


async def generate_answer_stream(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[AnswerStreamEvent]:
    """Yield answer text as it arrives, then one `AnswerComplete`.

    The final event always arrives, including when nothing was retrieved and when the stream
    ends without the provider's parsed object. A consumer that has emitted tokens needs to be
    told how it finished, and "the connection went away" is not something the caller can infer
    from an absent event.
    """
    settings = settings or get_settings()
    client = client or LLMClient(settings)
    loader = loader or PromptLoader(settings=settings)

    prompt = loader.get(PROMPT_NAME)
    if not chunks:
        # Same short circuit as the non-streaming path: nothing was retrieved, so the honest
        # answer is already known and paying for a call to produce it is waste.
        yield AnswerComplete(answer=no_context_answer(prompt.version, settings.llm_strong))
        return

    messages = split_prompt(prompt, context=format_context(chunks), question=question)
    temperature = prompt.meta.get("temperature")

    streamer = StringFieldStreamer("answer")
    started = time.perf_counter()
    raw_text = ""
    done: StreamDone | None = None

    with stage(
        "generate",
        kind="span",
        input={"question": question, "passages": len(chunks)},
        metadata={"prompt_version": prompt.qualified_version, "streamed": True},
    ) as span:
        async for event in client.stream(
            messages,
            role="strong",
            max_output_tokens=MAX_ANSWER_TOKENS,
            temperature=float(temperature) if isinstance(temperature, int | float) else None,
            stage="answer",
            prompt_version=prompt.qualified_version,
            schema=AnswerDraft,
        ):
            if isinstance(event, StreamDelta):
                raw_text += event.text
                text = streamer.feed(event.text)
                if text:
                    yield AnswerToken(text=text)
            elif isinstance(event, StreamDone):
                done = event

        answer = _finalise(
            done=done,
            raw_text=raw_text,
            chunks=chunks,
            prompt_version=prompt.version,
            settings=settings,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        span.update(
            output={"answer": answer.answer, "refused": answer.refused},
            metadata={
                "confidence": answer.confidence,
                "cited": len(answer.citations),
                "dropped_citations": answer.dropped_citations or None,
            },
        )

    yield AnswerComplete(answer=answer)


def _finalise(
    *,
    done: StreamDone | None,
    raw_text: str,
    chunks: Sequence[RetrievedChunk],
    prompt_version: str,
    settings: Settings,
    elapsed_ms: float,
) -> Answer:
    """Build the final `Answer` from the provider's parsed object, or salvage a partial one.

    The parsed instance is preferred so a streamed answer is validated identically to a
    non-streamed one. When it is absent — a dropped connection, or an output budget that ran
    out mid-object — the text that did arrive is still served, marked `refused=False` with
    zero confidence and no citations: there is an answer, and there is no verified provenance
    for it, and both of those things are true at once.
    """
    draft = done.parsed if done is not None else None

    if isinstance(draft, AnswerDraft):
        citations, dropped = resolve_citations(draft, chunks)
        return Answer(
            answer=draft.answer.strip(),
            citations=citations,
            confidence=min(max(draft.confidence, 0.0), 1.0),
            refused=draft.refused,
            prompt_version=prompt_version,
            model=done.model if done else settings.llm_strong,
            usage=done.usage if done else Usage(),
            latency_ms=elapsed_ms,
            dropped_citations=dropped,
        )

    salvaged = extract_string_field("answer", raw_text).strip()
    logger.warning(
        "generation stream ended without a parsed object",
        extra={"salvaged_chars": len(salvaged), "raw_chars": len(raw_text)},
    )
    return Answer(
        answer=salvaged or "The answer could not be completed. Please try that question again.",
        citations=[],
        confidence=0.0,
        refused=False,
        prompt_version=prompt_version,
        model=done.model if done else settings.llm_strong,
        usage=done.usage if done else Usage(),
        latency_ms=elapsed_ms,
    )


__all__ = ["AnswerComplete", "AnswerStreamEvent", "AnswerToken", "generate_answer_stream"]
