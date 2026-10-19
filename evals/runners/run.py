"""`make eval` — the CLI entry point.

ARCHITECTURE §11 names `run_rag.py`, `run_agent.py` and `run_guardrails.py`; this dispatches to
the right one so the Makefile target does not have to change per phase. Phase 2 has only RAG.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from evals.runners.run_rag import Profile, run, write_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Sutra eval suite.")
    parser.add_argument("--profile", default="naive", help="profile name or path to a .yaml")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also write evals/reports/baseline.json, the reference for later runs",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="skip the quote-coverage precondition (not advisable: recall would understate)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO" if args.verbose else "WARNING",
        format="%(levelname)s %(name)s %(message)s",
    )

    profile = Profile.load(args.profile)
    report = asyncio.run(run(profile, limit=args.limit, skip_coverage=args.skip_coverage))
    md_path, json_path = write_reports(report, as_baseline=args.baseline)

    print(f"\n{report.profile.name}: {len(report.rows)} rows in {report.duration_seconds:.0f}s")
    print(
        f"  recall@{profile.top_k} {report.retrieval.recall_at_k:.1%}   "
        f"mrr {report.retrieval.mrr:.3f}   "
        f"context precision {report.retrieval.context_precision:.1%}"
    )
    for judge in report.judges:
        print(f"  {judge.name:14s} {judge.mean_score:.3f}  ({judge.scored_rows} rows)")
    print(
        f"  refusal recall {report.refusal.refusal_recall:.1%}   "
        f"false refusal {report.refusal.false_refusal_rate:.1%}"
    )
    print(f"  cost ${report.total_cost_usd:.4f}   p50 {report.latency_p50_ms:.0f}ms")
    print(f"\n  {md_path.relative_to(md_path.parents[2])}")
    print(f"  {json_path.relative_to(json_path.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
