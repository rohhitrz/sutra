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
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from sutra.config import Settings, get_settings
from sutra.generation.answer import split_prompt
from sutra.llm.client import LLMClient, client_scope
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
    # Span verification, faithfulness v2 only. `claims_total` is how many claims the judge
    # found; `spans_verified` how many of its supporting quotes really appear in the passages;
    # `fabricated_spans` the ones that do not. A judge inventing its own evidence is the
    # failure v1 could not expose, so it is recorded per row rather than aggregated away.
    claims_total: int = 0
    spans_verified: int = 0
    fabricated_spans: list[str] = Field(default_factory=list)
    # Supported claims the judge gave no span for. A protocol lapse, not invented evidence.
    unquoted_claims: int = 0
    # Set when the judge call failed. The row is then excluded from that judge's average rather
    # than scored 0, because a provider error is not evidence about the answer.
    error: str | None = None


class ClaimCheck(BaseModel):
    """One factual claim the judge extracted, and the span it says supports it."""

    claim: str
    supported: bool
    # Verbatim span copied from the passages. Empty when the judge says it is unsupported.
    quote: str = ""


class FaithfulnessVerdict(BaseModel):
    """What `judge_faithfulness.v2` returns.

    # CONCEPT: Make the judge show its working, then check the working. v1 returned a score and
    # a sentence, and the sentence could say anything — it once asserted "the passage states
    # that Kesh Ithar has 68,000 people" about passages containing neither the figure nor the
    # word "population". A free-text justification is unfalsifiable; a quoted span is not.
    # Requiring the quote costs nothing extra to produce and turns the judge's reasoning into
    # something the harness can verify mechanically.
    """

    claims: list[ClaimCheck] = Field(default_factory=list)
    score: float
    reason: str


def verify_spans(verdict: FaithfulnessVerdict, context: str) -> tuple[int, list[str], int]:
    """Check every claimed supporting quote against the passages.

    Returns (verified, fabricated, unquoted): supported claims whose span really appears, the
    spans that do not appear, and supported claims that came with no span at all.

    **Unquoted is kept separate from fabricated on purpose.** They are different failures: a
    missing quote is the judge not following the protocol, while a quote that does not exist
    is the judge inventing evidence. Folding them together would have reported the first run
    of this check as "2 fabricated spans" when one was an empty string on a refusal row — and
    an over-sensitive detector of dishonesty is its own kind of dishonest.

    A genuinely fabricated span is the strongest signal this harness can produce about a
    judge. It does not mean the answer was unfaithful; it means the judge's *stated reason*
    for its verdict was invented, which makes that verdict worthless in either direction.
    """
    haystack = _normalise(context)
    fabricated: list[str] = []
    verified = unquoted = 0
    for claim in verdict.claims:
        if not claim.supported:
            continue
        quote = _normalise(claim.quote)
        if not quote:
            unquoted += 1
        elif quote in haystack:
            verified += 1
        else:
            fabricated.append(claim.quote.strip())
    return verified, fabricated, unquoted


def _normalise(text: str) -> str:
    """Collapse whitespace and drop markdown emphasis, for comparing a quote to its source.

    Whitespace because the corpus is hard-wrapped markdown and a correct quote still carries
    the newline where the source line broke. Emphasis markers because the first run of this
    check flagged `Population: 2,300 (1143)` as fabricated against a source reading
    `- **Population:** 2,300 (1143)` — the judge had quoted the text and dropped the
    asterisks, which is a correct quotation of the content and a false alarm here.

    Nothing else is relaxed. Every further loosening makes a fabricated quote likelier to pass,
    and this check only earns its place while a failure means something.
    """
    stripped = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", stripped).strip()


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

    # The schema is chosen by the prompt version, not by the judge name: v1 and v2 of
    # faithfulness ask for different things, and running v2's prompt against v1's schema would
    # discard the claims silently and leave the span check with nothing to verify.
    wants_claims = name == "faithfulness" and prompt.version >= "v2"
    schema: type[BaseModel] = FaithfulnessVerdict if wants_claims else JudgeVerdict

    try:
        messages = split_prompt(prompt, **variables)
        # `client_scope` rather than `client or LLMClient(...)`, which leaked a connection
        # pool per judge call on any path that did not inject one. `run_judges` opens one
        # scope around all three, so a row's judges share a pool; this is the fallback for a
        # direct caller, and it yields an injected client untouched.
        async with client_scope(client, settings) as llm:
            response = await llm.structured(
                messages,
                schema,
                # `fast`, because ARCHITECTURE §5 puts grading on the fast role. This changes
                # nothing today — ADR-017 points both roles at one model, and ADR-017 also
                # declined a separate judge role — so it is a statement of intent, recorded on
                # every judge generation, rather than a change in behaviour.
                #
                # It is the intent worth arguing about, though, and the argument is written
                # here rather than lost: a judge weaker than the generator it grades is a poor
                # instrument, and known issue 2 already puts this one at roughly 9/10 against
                # hand labels. So **if `LLM_FAST` is ever pointed at a cheaper model, the eval
                # scoreboard's generation metrics stop being comparable across that change**,
                # and `make judges` has to be re-run against `evals/datasets/judge_labels.jsonl`
                # before any number measured after it is trusted.
                role="fast",
                max_output_tokens=MAX_JUDGE_TOKENS,
                # Named so a row's three judges are distinguishable in the trace instead of
                # arriving as three spans all called `llm.call`.
                stage=f"judge_{name}",
                prompt_version=prompt.qualified_version,
            )
        verdict = response.parsed
        if not isinstance(verdict, schema):
            msg = f"judge {name} returned {type(verdict).__name__}"
            raise TypeError(msg)

        claims_total = spans_verified = unquoted = 0
        fabricated: list[str] = []
        if isinstance(verdict, FaithfulnessVerdict):
            claims_total = len(verdict.claims)
            spans_verified, fabricated, unquoted = verify_spans(verdict, context)
            if fabricated:
                logger.warning(
                    "judge faithfulness %s quoted %d span(s) absent from the passages",
                    prompt.version,
                    len(fabricated),
                )

        return JudgeResult(
            name=name,
            score=min(max(verdict.score, 0.0), 1.0),
            reason=verdict.reason.strip(),
            claims_total=claims_total,
            spans_verified=spans_verified,
            fabricated_spans=fabricated,
            unquoted_claims=unquoted,
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
    settings = settings or get_settings()
    # One client for all three, rather than one each. The judges fan out concurrently, so
    # without this a row with no injected client would open three provider connection pools
    # to make three calls — the same reasoning as `_search_inner`'s shared client.
    async with client_scope(client, settings) as llm:
        results = await asyncio.gather(
            *(
                run_judge(
                    name,
                    question=question,
                    answer=answer,
                    context=context,
                    reference=reference,
                    client=llm,
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
