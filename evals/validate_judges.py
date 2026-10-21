"""Grade the judge (ARCHITECTURE §11: "about 10 rows are hand-labelled to check the judge").

# CONCEPT: An LLM judge is an instrument, and an instrument nobody calibrates is a number
# generator. Every faithfulness figure in the scoreboard rests on the judge being roughly
# right; if it is not, the whole column is decoration and the techniques measured against it
# were chosen on noise.
#
# `evals/datasets/judge_labels.jsonl` holds ten rows, each a (passages, question, answer)
# triple with a score a human stands behind and the failure mode it is testing. The passages
# are verbatim corpus text and every premise behind a label — "this figure is absent from this
# passage" — is checked mechanically by `make validate`, so a label cannot quietly rot.
#
# Two numbers come out, and they answer different questions:
#
#   agreement       how often the judge lands on the labelled score. The headline.
#   fabricated spans  how often a v2 judge quotes a passage that does not exist. This is the
#                   one v1 could not expose at all, and it is worse than a wrong score: a
#                   verdict justified by invented evidence is worthless in either direction.
#
# Run with `make judges`. It is a handful of calls, so it is cheap enough to run on any change
# to a judge prompt — which is the point, since judge prompts are otherwise edited on faith.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from pydantic import BaseModel

from evals.metrics.judges import run_judge
from sutra.config import get_settings
from sutra.llm.client import LLMClient
from sutra.prompts import PromptLoader

LABELS = Path("evals/datasets/judge_labels.jsonl")

# A judge is "in agreement" when it lands on the labelled score. The rubric has three rungs
# (0.0 / 0.5 / 1.0) and the distance between them is meaningful, so near-misses are reported
# separately rather than folded into a tolerance that would hide 0.5-vs-1.0 confusion — which
# is precisely the confusion that matters for an unfaithful answer.
EXACT = 0.001


class LabelRow(BaseModel):
    id: str
    question: str
    context: str
    answer: str
    label_score: float
    failure_mode: str
    label_reason: str


class JudgeOutcome(BaseModel):
    row_id: str
    failure_mode: str
    labelled: float
    scored: float | None
    reason: str
    claims_total: int = 0
    spans_verified: int = 0
    fabricated_spans: list[str] = []
    unquoted_claims: int = 0
    error: str | None = None

    @property
    def agrees(self) -> bool:
        return self.scored is not None and abs(self.scored - self.labelled) < EXACT

    @property
    def direction(self) -> str:
        """Which way a disagreement went. Lenient is the dangerous one: it means the judge
        waved through an answer a human called unfaithful, which is how a rubber stamp looks
        from the outside."""
        if self.scored is None or self.agrees:
            return ""
        return "lenient" if self.scored > self.labelled else "harsh"


def load_labels() -> list[LabelRow]:
    if not LABELS.exists():
        msg = f"no judge labels at {LABELS}"
        raise FileNotFoundError(msg)
    return [LabelRow.model_validate_json(line) for line in LABELS.read_text().splitlines() if line]


async def grade_judge(version: str, rows: list[LabelRow]) -> list[JudgeOutcome]:
    """Run one faithfulness prompt version over every labelled row."""
    settings = get_settings().model_copy(update={"prompt_version_judge_faithfulness": version})
    client = LLMClient(settings=settings)
    loader = PromptLoader(settings=settings)

    async def one(row: LabelRow) -> JudgeOutcome:
        result = await run_judge(
            "faithfulness",
            question=row.question,
            answer=row.answer,
            context=row.context,
            client=client,
            loader=loader,
            settings=settings,
        )
        return JudgeOutcome(
            row_id=row.id,
            failure_mode=row.failure_mode,
            labelled=row.label_score,
            scored=None if result.error else result.score,
            reason=result.reason,
            claims_total=result.claims_total,
            spans_verified=result.spans_verified,
            fabricated_spans=result.fabricated_spans,
            unquoted_claims=result.unquoted_claims,
            error=result.error,
        )

    return list(await asyncio.gather(*(one(row) for row in rows)))


def report(by_version: dict[str, list[JudgeOutcome]]) -> int:
    for version, outcomes in by_version.items():
        agreed = sum(1 for o in outcomes if o.agrees)
        lenient = [o for o in outcomes if o.direction == "lenient"]
        harsh = [o for o in outcomes if o.direction == "harsh"]
        fabricated = [o for o in outcomes if o.fabricated_spans]
        errors = [o for o in outcomes if o.error]
        gap = statistics.mean(
            abs((o.scored or 0.0) - o.labelled) for o in outcomes if o.scored is not None
        )

        print(f"\n## judge_faithfulness.{version}")
        print(f"  agreement         {agreed}/{len(outcomes)}")
        print(f"  mean |error|      {gap:.2f}")
        print(f"  too lenient       {len(lenient)}  {[o.row_id for o in lenient]}")
        print(f"  too harsh         {len(harsh)}  {[o.row_id for o in harsh]}")
        if any(o.claims_total for o in outcomes):
            claims = sum(o.claims_total for o in outcomes)
            verified = sum(o.spans_verified for o in outcomes)
            unquoted = sum(o.unquoted_claims for o in outcomes)
            print(f"  claims extracted  {claims}, supporting spans verified {verified}")
            print(f"  unquoted claims   {unquoted}  (protocol lapse, not invented evidence)")
            print(f"  FABRICATED spans  {sum(len(o.fabricated_spans) for o in fabricated)}")
        if errors:
            print(f"  call failures     {len(errors)}")

        print(f"\n  {'row':7s} {'labelled':>8s} {'scored':>7s}  mode")
        for o in outcomes:
            mark = "ok " if o.agrees else ("<- " + o.direction)
            scored = "err" if o.scored is None else f"{o.scored:.1f}"
            print(f"  {o.row_id:7s} {o.labelled:>8.1f} {scored:>7s}  {o.failure_mode}  {mark}")
            for span in o.fabricated_spans:
                print(f"            FABRICATED QUOTE: {span[:90]!r}")

    # Never a build failure. The judge is a measuring instrument, and the honest response to a
    # miscalibrated one is to read its numbers differently, not to refuse to build the project.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        default="v1,v2",
        help="comma-separated faithfulness prompt versions to compare",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the raw outcomes")
    args = parser.parse_args()

    rows = load_labels()
    print(f"{len(rows)} hand-labelled rows from {LABELS}")

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    by_version = {v: asyncio.run(grade_judge(v, rows)) for v in versions}

    if args.json:
        args.json.write_text(
            json.dumps(
                {v: [o.model_dump() for o in outs] for v, outs in by_version.items()}, indent=2
            )
            + "\n"
        )
    return report(by_version)


if __name__ == "__main__":
    raise SystemExit(main())
