"""LLM-as-judge metrics (ADR-008).

# CONCEPT: Retrieval metrics are mechanical — a quote is in a chunk or it is not. Generation
# quality is not, so it is graded by a model against a versioned rubric. Three separate judges,
# because the failure modes are independent and a blended score hides them:
#
#   faithfulness — is every claim supported by the passages?      (grounding)
#   relevance    — does it answer the question that was asked?    (responsiveness)
#   correctness  — does it agree with the human reference answer? (accuracy)
#
# Phase 2's very first live call produced an answer that scored perfectly on correctness and
# zero on faithfulness: it stated the right population figure, which appeared in none of the
# retrieved passages. One "quality" number would have called that a success.
#
# ADR-017 caveat, carried into every report: the judge model is the same model that writes the
# answers, so these are same-family self-evaluations. Read them as *relative* — valid for
# comparing configurations against each other, not as ground truth.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from sutra.config import Settings, get_settings
from sutra.generation.answer import split_prompt
from sutra.llm.client import LLMClient
from sutra.prompts import PromptLoader

logger = logging.getLogger(__name__)

JudgeName = Literal["faithfulness", "relevance", "correctness"]
JUDGE_NAMES: tuple[JudgeName, ...] = ("faithfulness", "relevance", "correctness")

# Headroom for reasoning *and* the verdict, not a target for either.
#
# `max_output_tokens` budgets them together, and a judge grading a hard row spends most of it
# reasoning before emitting a single visible character. At 300 the first full run lost three
# verdicts to truncation; at 800 it still lost two. The failure surfaced as
# "Invalid JSON: EOF while parsing a string" — a parse error that reads like a provider fault
# rather than a budget one, which is why `OpenAIAdapter._reject_if_truncated` now names the real
# cause. A verdict is ~40 tokens; the rest of this is reasoning headroom.
MAX_JUDGE_TOKENS = 4000


class JudgeVerdict(BaseModel):
    """What a judge returns.

    `reason` is not decoration: a score with no stated reason cannot be audited, and spot-checking
    a handful of reasons is the only cheap way to notice a judge has misread its own rubric.
    """

    score: float
    reason: str


class JudgeResult(BaseModel):
    """One judge's verdict, with what it cost."""

    name: JudgeName
    score: float
    reason: str
    prompt_version: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    # The cached share of `input_tokens`, billed at the provider's cached rate. Recorded
    # because without it a run's cost is unexplainable: two runs doing identical work can
    # differ by more than half on cost alone, purely on how warm the provider's prefix
    # cache was, and a cost delta that large looks like a regression until you can see why.
    cached_tokens: int = 0
    cost_usd: float = 0.0
    # Set when the judge call failed. The row is then excluded from that judge's average rather
    # than scored 0, because a provider error is not evidence about the answer.
    error: str | None = None


class JudgeScores(BaseModel):
    """All judges for one row."""

    results: list[JudgeResult] = Field(default_factory=list)

    def score(self, name: JudgeName) -> float | None:
        for result in self.results:
            if result.name == name:
                return None if result.error else result.score
        return None

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)


def _prompt_name(name: JudgeName) -> str:
    return f"judge_{name}"


async def run_judge(
    name: JudgeName,
    *,
    question: str,
    answer: str,
    context: str = "",
    reference: str = "",
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    settings: Settings | None = None,
) -> JudgeResult:
    """Grade one answer on one axis."""
    settings = settings or get_settings()
    client = client or LLMClient(settings)
    loader = loader or PromptLoader(settings=settings)
    prompt = loader.get(_prompt_name(name))

    # Each judge takes a different set of variables, so only the ones its template declares are
    # substituted — `Template.substitute` raises on a missing key, which is the behaviour we
    # want if a rubric edit introduces a variable nobody passes.
    variables = {"question": question, "answer": answer}
    if name == "faithfulness":
        variables["context"] = context
    if name == "correctness":
        variables["reference"] = reference

    try:
        messages = split_prompt(prompt, **variables)
        response = await client.structured(
            messages, JudgeVerdict, role="strong", max_output_tokens=MAX_JUDGE_TOKENS
        )
        verdict = response.parsed
        if not isinstance(verdict, JudgeVerdict):
            msg = f"judge {name} returned {type(verdict).__name__}"
            raise TypeError(msg)
        return JudgeResult(
            name=name,
            score=min(max(verdict.score, 0.0), 1.0),
            reason=verdict.reason.strip(),
            prompt_version=prompt.version,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_tokens=response.usage.cached_tokens,
            cost_usd=response.usage.cost_usd or 0.0,
        )
    except Exception as exc:
        # Broad on purpose: one judge failing on one row must not abort a 150-call run. The
        # failure is recorded on the row and excluded from the average rather than scored 0.
        logger.warning("judge %s failed: %s", name, exc)
        return JudgeResult(
            name=name,
            score=0.0,
            reason="",
            prompt_version=prompt.version,
            model=settings.llm_strong,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_judges(
    *,
    question: str,
    answer: str,
    context: str,
    reference: str,
    names: Sequence[JudgeName] = JUDGE_NAMES,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    settings: Settings | None = None,
) -> JudgeScores:
    """Run the judges for one row, concurrently — they are independent of one another."""
    results = await asyncio.gather(
        *(
            run_judge(
                name,
                question=question,
                answer=answer,
                context=context,
                reference=reference,
                client=client,
                loader=loader,
                settings=settings,
            )
            for name in names
        )
    )
    return JudgeScores(results=list(results))


class JudgeSummary(BaseModel):
    """One judge's average over a run."""

    name: JudgeName
    mean_score: float
    scored_rows: int
    failed_rows: int
    prompt_version: str


def summarise_judges(scores: Sequence[JudgeScores]) -> list[JudgeSummary]:
    """Average each judge over the rows it actually scored.

    Failed calls are excluded rather than counted as zero: a timeout is not evidence that an
    answer was bad, and silently scoring it 0 would make an unstable provider look like a quality
    regression. The exclusion is reported so the denominator is never a mystery.
    """
    summaries: list[JudgeSummary] = []
    for name in JUDGE_NAMES:
        results = [r for s in scores for r in s.results if r.name == name]
        if not results:
            continue
        ok = [r for r in results if r.error is None]
        summaries.append(
            JudgeSummary(
                name=name,
                mean_score=sum(r.score for r in ok) / len(ok) if ok else 0.0,
                scored_rows=len(ok),
                failed_rows=len(results) - len(ok),
                prompt_version=results[0].prompt_version,
            )
        )
    return summaries
