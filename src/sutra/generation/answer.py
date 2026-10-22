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
from sutra.generation.budget import HistoryTurn, fit_context
from sutra.generation.schema import Answer, AnswerDraft, Citation
from sutra.llm.client import LLMClient, client_scope
from sutra.llm.types import Message
from sutra.observability.tracing import stage
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


def split_halves(prompt: Prompt) -> tuple[str, str]:
    """The stable system half and the per-request user half, as raw text.

    Separated out because the token budget has to price the system block *before* anything is
    interpolated into the other half — the two callers would otherwise each re-split the file
    and could disagree about where the boundary is.
    """
    if USER_DELIMITER not in prompt.body:
        msg = (
            f"prompt {prompt.name}.{prompt.version} is missing the {USER_DELIMITER!r} delimiter "
            f"at {prompt.path}"
        )
        raise ValueError(msg)

    system_part, user_part = prompt.body.split(USER_DELIMITER, 1)
    return system_part.strip(), user_part.strip()


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
    system_part, user_part = split_halves(prompt)
    return [
        Message(role="system", content=system_part),
        Message(role="user", content=Template(user_part).substitute(**variables)),
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


def no_context_answer(prompt_version: str, model: str) -> Answer:
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
    memories: Sequence[str] = (),
    summary: str = "",
    history: Sequence[HistoryTurn] = (),
) -> Answer:
    """Answer `question` from `chunks`, with citations resolved against those chunks.

    `memories`, `summary` and `history` are empty until Phase 5's memory steps fill them. They
    are in the signature now because the budget has to price them: adding the parameters later
    would mean every caller that had been fitting without them was fitting against the wrong
    total.
    """
    settings = settings or get_settings()
    loader = loader or PromptLoader(settings=settings)

    prompt = loader.get(PROMPT_NAME)
    if not chunks:
        # Before any client exists, deliberately. The old `client or LLMClient(settings)` ran
        # above this line, so the one path that makes no model call was the one that built a
        # provider SDK and a connection pool for nothing.
        return no_context_answer(prompt.version, settings.llm_strong)

    # Fit before rendering. Trimming a rendered prompt means re-rendering it, and "drop the
    # lowest-ranked passage" is only expressible while the passages are still a list.
    plan = fit_context(
        system=split_halves(prompt)[0],
        question=question,
        chunks=chunks,
        memories=memories,
        summary=summary,
        history=history,
        settings=settings,
    )
    chunks = plan.chunks

    messages = split_prompt(prompt, context=format_context(chunks), question=question)
    # The prompt file's own front-matter carries the temperature, so changing it is a prompt
    # version bump and shows up in the diff rather than hiding in Python.
    temperature = prompt.meta.get("temperature")

    started = time.perf_counter()
    # `client_scope` rather than `client or LLMClient(...)`: the convenience idiom leaks the
    # provider's connection pool on every call that falls through to it, which is why the
    # router and the query transforms were moved off it. This path was missed. An injected
    # client — which `/v1/chat` and the eval runner both pass — is yielded untouched and
    # outlives this call, as it should.
    with stage(
        "generate",
        kind="span",
        input={"question": question, "passages": len(chunks)},
        metadata={"prompt_version": prompt.qualified_version, **plan.as_metadata()},
    ) as span:
        async with client_scope(client, settings) as llm:
            response = await llm.structured(
                messages,
                AnswerDraft,
                role="strong",
                max_output_tokens=settings.answer_reserve_tokens,
                temperature=float(temperature) if isinstance(temperature, int | float) else None,
                stage="answer",
                prompt_version=prompt.qualified_version,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000

        draft = response.parsed
        if not isinstance(draft, AnswerDraft):
            msg = f"structured call returned {type(draft).__name__}, expected AnswerDraft"
            raise TypeError(msg)

        # The estimator checking itself. `plan.parts.input_tokens` is what the budget believed
        # this request would cost; `response.usage.input_tokens` is what it cost. Logging the
        # pair on every call is what keeps CHARS_PER_TOKEN a measurement instead of a belief —
        # and a *negative* drift is the dangerous direction, because it means the budget has
        # been under-counting and the window is nearer than it looks.
        actual_input = response.usage.input_tokens
        if actual_input:
            logger.info(
                "budget.drift",
                extra={
                    "estimated_input_tokens": plan.parts.input_tokens,
                    "actual_input_tokens": actual_input,
                    "drift_tokens": plan.parts.input_tokens - actual_input,
                    "estimator_is_conservative": plan.parts.input_tokens >= actual_input,
                },
            )

        citations, dropped = resolve_citations(draft, chunks)
        span.update(
            output={"answer": draft.answer, "refused": draft.refused},
            metadata={
                "confidence": draft.confidence,
                "cited": len(citations),
                # A citation the model invented is the clearest single signal that an answer
                # made up its provenance, so it goes on the span and not only in the response.
                "dropped_citations": dropped or None,
            },
        )
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
