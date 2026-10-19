"""Turn retrieved passages into a cited, structured answer.

# CONCEPT: Generation in RAG is not "ask the model the question". It is: assemble a context the
# model is allowed to use, constrain the output shape, then verify the claims the output makes
# about that context. The third part is the one usually skipped, and it is where provenance is
# either real or decorative.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from string import Template

from sutra.config import Settings, get_settings
from sutra.generation.schema import Answer, AnswerDraft, Citation
from sutra.llm.client import LLMClient
from sutra.llm.types import Message
from sutra.prompts import Prompt, PromptLoader
from sutra.retrieval.search import RetrievedChunk

logger = logging.getLogger(__name__)

PROMPT_NAME = "answer"

# Splits the stable instructions from the per-request part of the prompt file.
#
# # CONCEPT: Prompt caching bills a repeated prefix at a fraction of the input rate, but only
# # while that prefix is byte-identical, so every stable instruction belongs ahead of the first
# # variable byte. Interpolating into the system block would invalidate the cache on every
# # request.
#
# Measured caveat: this block is ~425 tokens and OpenAI caches only prefixes above a 1024-token
# minimum, so it cannot hit the cache by itself today. The ordering is the precondition, not the
# saving; it begins to pay once memory and guardrail instructions (Phases 5 and 6) push the
# stable block past that floor.
USER_DELIMITER = "<<<USER>>>"

# Room for reasoning as well as the answer. `max_output_tokens` budgets both together, so a cap
# sized for the prose you want to read can be consumed entirely by reasoning, leaving a truncated
# body that a schema-constrained call reports as invalid JSON. The judges hit exactly that during
# the first full eval run. Phase 5 replaces this with a real token budget (ARCHITECTURE §5) that
# trims context to fit the model window.
MAX_ANSWER_TOKENS = 2000


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Number the passages so the model has a stable, checkable handle on each one.

    The slug, title and heading go in the header rather than being left implicit: a question
    about dates is answerable only if the model can see which document a passage came from, and
    which section of it.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{index}] {chunk.doc_slug} · {chunk.doc_title}"
        if chunk.heading:
            header += f" · {chunk.heading}"
        blocks.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def split_prompt(prompt: Prompt, **variables: str) -> list[Message]:
    """Build the message list from one versioned prompt file.

    The file holds both halves so a prompt version is a single reviewable artefact; the
    delimiter keeps the stable half first and uninterpolated.

    Variables are whatever the template declares — the answer prompt wants `context` and
    `question`, the correctness judge wants `question`, `answer` and `reference`. Taking them as
    keywords rather than a fixed signature is what lets every versioned prompt in the project
    share this one function. `Template.substitute` raises on a missing key, so a rubric edit that
    introduces a variable nobody passes fails loudly instead of rendering a half-built prompt.
    """
    if USER_DELIMITER not in prompt.body:
        msg = (
            f"prompt {prompt.name}.{prompt.version} is missing the {USER_DELIMITER!r} delimiter "
            f"at {prompt.path}"
        )
        raise ValueError(msg)

    system_part, user_part = prompt.body.split(USER_DELIMITER, 1)
    return [
        Message(role="system", content=system_part.strip()),
        Message(
            role="user",
            content=Template(user_part.strip()).substitute(**variables),
        ),
    ]


def resolve_citations(
    draft: AnswerDraft, chunks: Sequence[RetrievedChunk]
) -> tuple[list[Citation], list[int]]:
    """Map the model's passage numbers back to real chunks, keeping what it got wrong.

    # CONCEPT: Never trust a citation. A schema guarantees the model returned integers; it says
    # nothing about whether those integers refer to anything. Resolving against the passages
    # actually shown is what makes a citation a verifiable pointer instead of a decoration.
    """
    by_label = {index: chunk for index, chunk in enumerate(chunks, start=1)}
    citations: list[Citation] = []
    dropped: list[int] = []
    seen: set[int] = set()

    for label in draft.citations:
        if label in seen:
            continue
        seen.add(label)
        chunk = by_label.get(label)
        if chunk is None:
            dropped.append(label)
            continue
        citations.append(
            Citation(
                label=label,
                chunk_id=chunk.chunk_id,
                doc_slug=chunk.doc_slug,
                doc_title=chunk.doc_title,
                heading=chunk.heading,
            )
        )

    if dropped:
        logger.warning(
            "model cited %d passage(s) that were not in the prompt: %s (shown 1..%d)",
            len(dropped),
            dropped,
            len(chunks),
        )
    return citations, dropped


def _no_context_answer(prompt_version: str, model: str) -> Answer:
    """Refuse without calling the model.

    Nothing was retrieved, so there is no grounded answer available and the correct response is
    already known. Paying for a call whose only honest output is a refusal is waste.
    """
    return Answer(
        answer=(
            "I could not find anything in the corpus to answer that, so I will not guess. "
            "No passages were retrieved for this question."
        ),
        citations=[],
        confidence=0.0,
        refused=True,
        prompt_version=prompt_version,
        model=model,
        short_circuited=True,
    )


async def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    settings: Settings | None = None,
) -> Answer:
    """Answer `question` from `chunks`, with citations resolved against those chunks."""
    settings = settings or get_settings()
    client = client or LLMClient(settings)
    loader = loader or PromptLoader(settings=settings)

    prompt = loader.get(PROMPT_NAME)
    if not chunks:
        return _no_context_answer(prompt.version, settings.llm_strong)

    messages = split_prompt(prompt, context=format_context(chunks), question=question)
    # The prompt file's own front-matter carries the temperature, so changing it is a prompt
    # version bump and shows up in the diff rather than hiding in Python.
    temperature = prompt.meta.get("temperature")

    started = time.perf_counter()
    response = await client.structured(
        messages,
        AnswerDraft,
        role="strong",
        max_output_tokens=MAX_ANSWER_TOKENS,
        temperature=float(temperature) if isinstance(temperature, int | float) else None,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    draft = response.parsed
    if not isinstance(draft, AnswerDraft):
        msg = f"structured call returned {type(draft).__name__}, expected AnswerDraft"
        raise TypeError(msg)

    citations, dropped = resolve_citations(draft, chunks)
    return Answer(
        answer=draft.answer.strip(),
        citations=citations,
        confidence=min(max(draft.confidence, 0.0), 1.0),
        refused=draft.refused,
        prompt_version=prompt.version,
        model=response.model,
        usage=response.usage,
        latency_ms=response.latency_ms or elapsed_ms,
        used_fallback=response.used_fallback,
        dropped_citations=dropped,
    )
