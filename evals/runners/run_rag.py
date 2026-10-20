"""Run the RAG eval suite over the golden dataset.

# CONCEPT: An eval run is a measurement, and a measurement nobody can reproduce is an anecdote.
# Every report therefore records the full configuration that produced it — index fingerprint,
# chunking parameters, prompt versions, judge versions, model, k — because "recall went up" is
# only meaningful against a known baseline, and the thing that most often changed is something
# nobody wrote down.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from evals.check_coverage import check_coverage, chunks_from_db, corpus_text
from evals.metrics.grounding import GroundingCheck, check_figures
from evals.metrics.judges import (
    JUDGE_NAMES,
    JudgeName,
    JudgeScores,
    JudgeSummary,
    run_judges,
    summarise_judges,
)
from evals.metrics.retrieval import (
    EvidenceRef,
    RetrievalSummary,
    RetrievedRef,
    RowRetrievalMetrics,
    score_row,
    summarise,
)
from evals.validate_datasets import DATASET_DIR, GoldenRow, load_rows
from sutra.config import Settings, get_settings
from sutra.db.engine import get_sessionmaker
from sutra.generation.answer import format_context, generate_answer
from sutra.ingestion.embed import get_embedder
from sutra.ingestion.fingerprint import build_fingerprint
from sutra.ingestion.parse import parse_corpus
from sutra.llm.client import LLMClient
from sutra.prompts import PromptLoader
from sutra.retrieval.search import vector_search

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "evals" / "profiles"
REPORT_DIR = REPO_ROOT / "evals" / "reports"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


class Profile(BaseModel):
    """One eval configuration (ARCHITECTURE §11)."""

    name: str
    description: str = ""
    top_k: int = 5
    index_version: int = 1
    prompt_version_answer: str = "v1"
    judges: list[JudgeName] = Field(default_factory=lambda: list(JUDGE_NAMES))
    concurrency: int = 4

    @classmethod
    def load(cls, name_or_path: str) -> Profile:
        path = Path(name_or_path)
        if not path.exists():
            path = PROFILE_DIR / f"{name_or_path}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
            msg = f"no eval profile {name_or_path!r}; available: {available}"
            raise FileNotFoundError(msg)
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})

    def apply_to(self, settings: Settings) -> Settings:
        """A profile overrides config, so a run never depends on the developer's `.env`."""
        return settings.model_copy(
            update={
                "retrieval_top_k": self.top_k,
                "index_version": self.index_version,
                "prompt_version_answer": self.prompt_version_answer,
            }
        )


class RowResult(BaseModel):
    """Everything one golden row produced. The JSON report keeps all of these, so a moved
    aggregate can always be traced to the rows that moved it."""

    id: str
    question: str
    category: str
    must_refuse: bool

    retrieval: RowRetrievalMetrics
    retrieved: list[str] = Field(default_factory=list)

    answer: str
    reference: str
    refused: bool
    confidence: float
    citations: list[int] = Field(default_factory=list)
    dropped_citations: list[int] = Field(default_factory=list)
    short_circuited: bool = False

    judges: JudgeScores = Field(default_factory=JudgeScores)
    # The deterministic cross-check on the judge: figures asserted but absent from the context.
    grounding: GroundingCheck = Field(
        default_factory=lambda: GroundingCheck(figures_in_answer=[], unsupported=[])
    )
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    # The answer call's own tokens. Kept separately from the judges' because judge spend is the
    # bulk of a run (three calls per row against one) and conflating them hides that.
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    answer_cached_tokens: int = 0

    @property
    def refusal_correct(self) -> bool:
        return self.refused == self.must_refuse

    @property
    def has_unverifiable_figure(self) -> bool:
        """A figure absent from the context *and* evidence the retriever failed to fetch.

        The pairing is what makes it meaningful. A figure missing from the passages is innocent
        on its own — the corpus ledger's totals are arithmetic, so a correct answer computes
        figures that appear nowhere. But a figure that is neither present nor derivable from
        what was retrieved had to come from somewhere else.
        """
        return bool(self.grounding.unsupported) and self.retrieval.recall < 1.0


class RefusalSummary(BaseModel):
    """Refusal reported as two numbers, never one.

    # CONCEPT: A single "refusal accuracy" blends two opposite failures. A model that refuses
    # everything scores perfectly on the unanswerable rows, and a model that answers everything
    # scores perfectly on the answerable ones. Only the pair says whether refusal is a skill.
    """

    unanswerable_rows: int
    correctly_refused: int
    answerable_rows: int
    falsely_refused: int

    @property
    def refusal_recall(self) -> float:
        if self.unanswerable_rows == 0:
            return 1.0
        return self.correctly_refused / self.unanswerable_rows

    @property
    def false_refusal_rate(self) -> float:
        if self.answerable_rows == 0:
            return 0.0
        return self.falsely_refused / self.answerable_rows


class CategorySummary(BaseModel):
    category: str
    rows: int
    recall_at_k: float
    mrr: float
    refusal_correct: float
    faithfulness: float | None = None
    correctness: float | None = None


class RunReport(BaseModel):
    """A complete eval run: what was measured, under what configuration, and what it cost."""

    profile: Profile
    started_at: dt.datetime
    duration_seconds: float

    # Reproducibility block. Without this a report is a number with no provenance.
    index_fingerprint: str
    chunk_tokens: int
    chunk_overlap: int
    chunker_version: int
    embedding_model: str
    answer_model: str
    judge_model: str
    prompt_versions: dict[str, str]

    rows: list[RowResult]
    retrieval: RetrievalSummary
    judges: list[JudgeSummary]
    refusal: RefusalSummary
    by_category: list[CategorySummary]

    # Mechanical grounding, reported beside the faithfulness judge rather than folded into it,
    # because the two disagreeing is the single most informative thing in a report.
    rows_with_unsupported_figures: int
    # The subset that also failed to retrieve its evidence — the figures that are not merely
    # absent but unaccountable.
    rows_with_unverifiable_figures: int
    figure_support_rate: float

    total_cost_usd: float
    # Whole-run totals: answer calls *and* judge calls, matching what `total_cost_usd` covers.
    # These counted judge calls only for one run, which understated input tokens by about 44%
    # while the cost beside them was complete — the pair looked self-consistent and was not.
    # The split is kept below rather than dropped, because judge spend dominating a run (three
    # calls per row against one) is a real fact about the harness worth seeing.
    total_input_tokens: int
    total_output_tokens: int
    answer_input_tokens: int
    answer_output_tokens: int
    judge_input_tokens: int
    judge_output_tokens: int
    # The cached share of `total_input_tokens`. Cost is not comparable between two runs
    # without it — see `JudgeResult.cached_tokens`.
    total_cached_tokens: int
    latency_p50_ms: float
    latency_p95_ms: float

    # ADR-017: the judge is the same model that writes the answers.
    judge_caveat: str = (
        "The judge model is the same model that produced the answers (ADR-017), so judge scores "
        "are same-family self-evaluation. Read them as relative — valid for comparing "
        "configurations with each other, not as ground truth."
    )


def _evidence_for(row: GoldenRow) -> list[EvidenceRef]:
    return [EvidenceRef(doc_slug=s.doc_slug, quote=s.quote) for s in row.source_chunks]


async def _run_row(
    row: GoldenRow,
    *,
    profile: Profile,
    settings: Settings,
    client: LLMClient,
    loader: PromptLoader,
    embedder: Any,
    semaphore: asyncio.Semaphore,
) -> RowResult:
    async with semaphore:
        factory = get_sessionmaker()
        async with factory() as db:
            chunks = await vector_search(
                row.question,
                session=db,
                k=profile.top_k,
                index_version=profile.index_version,
                settings=settings,
                embedder=embedder,
            )

        answer = await generate_answer(
            row.question, chunks, client=client, loader=loader, settings=settings
        )
        context = format_context(chunks)

        judges = await run_judges(
            question=row.question,
            answer=answer.answer,
            context=context,
            reference=row.answer,
            names=profile.judges,
            client=client,
            loader=loader,
            settings=settings,
        )

        return RowResult(
            id=row.id,
            question=row.question,
            category=row.category,
            must_refuse=row.must_refuse,
            retrieval=score_row(
                _evidence_for(row),
                [RetrievedRef(doc_slug=c.doc_slug, text=c.text) for c in chunks],
            ),
            retrieved=[f"{c.doc_slug}#{c.chunk_no}" for c in chunks],
            answer=answer.answer,
            reference=row.answer,
            refused=answer.refused,
            confidence=answer.confidence,
            citations=[c.label for c in answer.citations],
            dropped_citations=answer.dropped_citations,
            short_circuited=answer.short_circuited,
            judges=judges,
            grounding=check_figures(answer.answer, context),
            answer_input_tokens=answer.usage.input_tokens,
            answer_output_tokens=answer.usage.output_tokens,
            answer_cached_tokens=answer.usage.cached_tokens,
            latency_ms=answer.latency_ms,
            cost_usd=(answer.usage.cost_usd or 0.0) + judges.cost_usd,
        )


def _by_category(rows: list[RowResult]) -> list[CategorySummary]:
    out: list[CategorySummary] = []
    for category in sorted({r.category for r in rows}):
        group = [r for r in rows if r.category == category]
        faith = [s for s in (r.judges.score("faithfulness") for r in group) if s is not None]
        correct = [s for s in (r.judges.score("correctness") for r in group) if s is not None]
        out.append(
            CategorySummary(
                category=category,
                rows=len(group),
                recall_at_k=sum(r.retrieval.recall for r in group) / len(group),
                mrr=sum(r.retrieval.reciprocal_rank for r in group) / len(group),
                refusal_correct=sum(1 for r in group if r.refusal_correct) / len(group),
                faithfulness=sum(faith) / len(faith) if faith else None,
                correctness=sum(correct) / len(correct) if correct else None,
            )
        )
    return out


async def assert_coverage(profile: Profile) -> None:
    """Refuse to measure retrieval over an index whose evidence is unreachable.

    If the chunker split a cited quote, recall understates by construction and the report would
    attribute a chunker defect to the retriever. Better to stop than to publish that.
    """
    settings = get_settings()
    chunks = await chunks_from_db(settings, profile.index_version)
    if not chunks:
        msg = f"index slot {profile.index_version} is empty — run `make ingest` first"
        raise SystemExit(msg)
    uncovered = [c for c in check_coverage(chunks, corpus_text()) if not c.covered]
    if uncovered:
        detail = ", ".join(f"{c.row_id}/{c.doc_slug}" for c in uncovered[:5])
        msg = (
            f"{len(uncovered)} cited quote(s) are unreachable in index slot "
            f"{profile.index_version} ({detail}). Every recall number would understate. "
            "Run `make coverage` for the diagnosis."
        )
        raise SystemExit(msg)


async def run(
    profile: Profile, *, limit: int | None = None, skip_coverage: bool = False
) -> RunReport:
    """Evaluate `profile` over the golden dataset."""
    settings = profile.apply_to(get_settings())
    if not skip_coverage:
        await assert_coverage(profile)

    golden, errors = load_rows(DATASET_DIR / "golden_qa.jsonl", GoldenRow)
    if errors:
        msg = "golden_qa.jsonl does not validate — run `make validate`\n" + "\n".join(errors)
        raise SystemExit(msg)
    if limit is not None:
        golden = golden[:limit]

    client = LLMClient(settings)
    loader = PromptLoader(settings=settings)
    embedder = get_embedder(settings)
    semaphore = asyncio.Semaphore(profile.concurrency)

    started = dt.datetime.now(dt.UTC)
    logger.info("eval start profile=%s rows=%d k=%d", profile.name, len(golden), profile.top_k)
    rows = await asyncio.gather(
        *(
            _run_row(
                row,
                profile=profile,
                settings=settings,
                client=client,
                loader=loader,
                embedder=embedder,
                semaphore=semaphore,
            )
            for row in golden
        )
    )
    duration = (dt.datetime.now(dt.UTC) - started).total_seconds()

    fingerprint = build_fingerprint(settings, parse_corpus(CORPUS_DIR))
    latencies = sorted(r.latency_ms for r in rows) or [0.0]
    unanswerable = [r for r in rows if r.must_refuse]
    answerable = [r for r in rows if not r.must_refuse]

    return RunReport(
        profile=profile,
        started_at=started,
        duration_seconds=duration,
        index_fingerprint=fingerprint.digest,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
        chunker_version=fingerprint.chunker_version,
        embedding_model=settings.embedding_model,
        answer_model=settings.llm_strong,
        judge_model=settings.llm_strong,
        prompt_versions={
            "answer": settings.prompt_version_answer,
            **{
                f"judge_{n}": getattr(settings, f"prompt_version_judge_{n}") for n in profile.judges
            },
        },
        rows=list(rows),
        retrieval=summarise([r.retrieval for r in rows], k=profile.top_k),
        judges=summarise_judges([r.judges for r in rows]),
        refusal=RefusalSummary(
            unanswerable_rows=len(unanswerable),
            correctly_refused=sum(1 for r in unanswerable if r.refused),
            answerable_rows=len(answerable),
            falsely_refused=sum(1 for r in answerable if r.refused),
        ),
        by_category=_by_category(list(rows)),
        rows_with_unsupported_figures=sum(1 for r in rows if not r.grounding.grounded),
        rows_with_unverifiable_figures=sum(1 for r in rows if r.has_unverifiable_figure),
        figure_support_rate=(
            sum(r.grounding.support_rate for r in rows) / len(rows) if rows else 1.0
        ),
        total_cost_usd=sum(r.cost_usd for r in rows),
        total_input_tokens=sum(r.answer_input_tokens for r in rows)
        + sum(j.input_tokens for r in rows for j in r.judges.results),
        total_output_tokens=sum(r.answer_output_tokens for r in rows)
        + sum(j.output_tokens for r in rows for j in r.judges.results),
        answer_input_tokens=sum(r.answer_input_tokens for r in rows),
        answer_output_tokens=sum(r.answer_output_tokens for r in rows),
        judge_input_tokens=sum(j.input_tokens for r in rows for j in r.judges.results),
        judge_output_tokens=sum(j.output_tokens for r in rows for j in r.judges.results),
        total_cached_tokens=sum(r.answer_cached_tokens for r in rows)
        + sum(j.cached_tokens for r in rows for j in r.judges.results),
        latency_p50_ms=statistics.median(latencies),
        latency_p95_ms=latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
    )


def write_reports(report: RunReport, *, as_baseline: bool = False) -> tuple[Path, Path]:
    """Write the markdown and JSON reports, and optionally update `baseline.json`."""
    from evals.runners.report import render_markdown

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y-%m-%d")
    stem = f"{stamp}_{report.profile.name}"

    json_path = REPORT_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    md_path = REPORT_DIR / f"{stem}.md"
    md_path.write_text(render_markdown(report))

    if as_baseline:
        # Both forms: the JSON is what a later run is compared against mechanically, the
        # markdown is what makes the committed reference reviewable in a diff.
        (REPORT_DIR / "baseline.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2) + "\n"
        )
        (REPORT_DIR / "baseline.md").write_text(md_path.read_text())
    return md_path, json_path
