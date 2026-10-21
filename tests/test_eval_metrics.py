"""Tests for the eval metrics.

These are the numbers every retrieval decision from Phase 3 on will be argued from, so they are
tested on hand-built cases where the right answer is obvious by inspection.
"""

from __future__ import annotations

import sys

import pytest
from evals.metrics.grounding import check_figures
from evals.metrics.judges import JudgeResult, JudgeScores, summarise_judges
from evals.metrics.retrieval import (
    EvidenceRef,
    RetrievedRef,
    score_row,
    summarise,
)
from evals.runners.run_rag import RefusalSummary


def ev(slug: str, quote: str) -> EvidenceRef:
    return EvidenceRef(doc_slug=slug, quote=quote)


def got(slug: str, text: str) -> RetrievedRef:
    return RetrievedRef(doc_slug=slug, text=text)


# ---------------------------------------------------------------------------
# per-row retrieval
# ---------------------------------------------------------------------------


def test_all_evidence_found_in_the_first_chunk():
    row = score_row(
        [ev("06_places_wiki", "Population: 68,000")],
        [got("06_places_wiki", "## Kesh Ithar\nPopulation: 68,000\n"), got("08_faq", "unrelated")],
    )
    assert row.recall == 1.0
    assert row.hit is True
    assert row.first_relevant_rank == 1
    assert row.reciprocal_rank == 1.0
    assert row.context_precision == 0.5, "one of two retrieved chunks was useful"


def test_recall_is_quote_level_not_chunk_level():
    """A question citing four quotes that lands one has not been served, even though a chunk
    came back 'relevant'. Chunk-level recall would call this 100%."""
    evidence = [ev("x", "alpha one"), ev("x", "beta two"), ev("x", "gamma three"), ev("x", "delta")]
    row = score_row(evidence, [got("x", "alpha one")])
    assert row.recall == 0.25
    assert row.hit is True


def test_nothing_relevant_scores_zero_without_crashing():
    row = score_row([ev("x", "alpha")], [got("x", "nothing here"), got("y", "nor here")])
    assert row.recall == 0.0
    assert row.hit is False
    assert row.first_relevant_rank is None
    assert row.reciprocal_rank == 0.0
    assert row.context_precision == 0.0


def test_reciprocal_rank_rewards_ranking_the_evidence_first():
    """What recall@k cannot see: evidence at rank 1 and at rank 5 score the same on recall@5."""
    evidence = [ev("x", "alpha")]
    early = score_row(evidence, [got("x", "alpha"), *[got("y", "no") for _ in range(4)]])
    late = score_row(evidence, [*[got("y", "no") for _ in range(4)], got("x", "alpha")])

    assert early.recall == late.recall == 1.0
    assert early.reciprocal_rank == 1.0
    assert late.reciprocal_rank == 0.2


def test_the_document_must_match_too():
    """Two documents can share a sentence — the registry and the timeline both record dates.
    Crediting the wrong document would score finding the right words in the wrong place."""
    row = score_row(
        [ev("05_timeline", "Hollow 1143")],
        [got("04_character_registry", "Hollow 1143")],
    )
    assert row.recall == 0.0
    assert row.hit is False


def test_quotes_match_across_the_line_wrapping_of_the_source():
    """The corpus is hard-wrapped markdown, so a correct quote carries the newline where the
    source line broke."""
    row = score_row(
        [ev("x", "no family name is\nrecorded")],
        [got("x", "The registry states no family name   is recorded for this person.")],
    )
    assert row.recall == 1.0


def test_a_row_citing_nothing_counts_as_satisfied():
    """Guards the denominator: 0/0 must not be 0% recall, which would drag every average down."""
    assert score_row([], [got("x", "anything")]).recall == 1.0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_summary_macro_averages_so_every_question_counts_once():
    """Micro-averaging would let the few multi-hop rows, which cite up to five quotes each,
    dominate the headline number."""
    many = score_row([ev("x", f"q{i}") for i in range(5)], [got("x", "q0")])  # recall 0.2
    one = score_row([ev("x", "q")], [got("x", "q")])  # recall 1.0
    summary = summarise([many, one], k=5)

    assert summary.rows == 2
    assert summary.recall_at_k == pytest.approx(0.6), "macro: (0.2 + 1.0) / 2"
    assert summary.hit_rate_at_k == 1.0
    assert summary.mrr == pytest.approx(1.0)


def test_an_empty_run_summarises_to_zero_rather_than_dividing_by_zero():
    summary = summarise([], k=5)
    assert summary.rows == 0 and summary.recall_at_k == 0.0


# ---------------------------------------------------------------------------
# refusal, as two numbers
# ---------------------------------------------------------------------------


def test_a_model_that_refuses_everything_is_caught_by_the_second_number():
    always = RefusalSummary(
        unanswerable_rows=6, correctly_refused=6, answerable_rows=36, falsely_refused=36
    )
    assert always.refusal_recall == 1.0, "looks perfect on the unanswerable rows alone"
    assert always.false_refusal_rate == 1.0, "and is useless, which only this number shows"


def test_a_model_that_answers_everything_is_caught_by_the_first():
    never = RefusalSummary(
        unanswerable_rows=6, correctly_refused=0, answerable_rows=36, falsely_refused=0
    )
    assert never.false_refusal_rate == 0.0
    assert never.refusal_recall == 0.0


def test_refusal_rates_handle_an_empty_split():
    empty = RefusalSummary(
        unanswerable_rows=0, correctly_refused=0, answerable_rows=0, falsely_refused=0
    )
    assert empty.refusal_recall == 1.0 and empty.false_refusal_rate == 0.0


# ---------------------------------------------------------------------------
# judges
# ---------------------------------------------------------------------------


def _result(name: str, score: float, error: str | None = None) -> JudgeResult:
    return JudgeResult(
        name=name,  # type: ignore[arg-type]
        score=score,
        reason="because",
        prompt_version="v1",
        model="gpt-6-luna",
        error=error,
    )


def test_a_failed_judge_call_is_excluded_not_scored_zero():
    """A timeout is not evidence that an answer was bad. Scoring it 0 would make an unstable
    provider look like a quality regression."""
    scores = [
        JudgeScores(results=[_result("faithfulness", 1.0)]),
        JudgeScores(results=[_result("faithfulness", 0.0, error="TimeoutError: …")]),
    ]
    (summary,) = summarise_judges(scores)

    assert summary.mean_score == 1.0, "averaged over the one row actually scored"
    assert summary.scored_rows == 1
    assert summary.failed_rows == 1, "and the exclusion is reported, not silent"


def test_a_failed_judge_reads_as_no_score_on_the_row():
    scores = JudgeScores(results=[_result("faithfulness", 0.0, error="boom")])
    assert scores.score("faithfulness") is None
    assert scores.score("relevance") is None, "a judge that did not run has no score"


# ---------------------------------------------------------------------------
# mechanical grounding — the cross-check on the judge
# ---------------------------------------------------------------------------


def test_the_case_the_faithfulness_judge_got_wrong():
    """The first full eval run of this project produced exactly this, and the judge scored it
    faithfulness 1.0 claiming the passage stated the figure. It did not."""
    context = "The standoff at the twenty-first stone, Frostmark of 1143. Dask paid the toll."
    answer = "Kesh Ithar has a population of 68,000 people (recorded in Frostmark of 1143)."
    check = check_figures(answer, context)

    assert check.unsupported == ["68,000"]
    assert "1143" in check.figures_in_answer, "the date was real and must not be flagged"
    assert check.grounded is False
    assert check.support_rate == 0.5


def test_thousands_separators_do_not_decide_the_outcome():
    assert check_figures("It is 68,000.", "Population: 68000").grounded
    assert check_figures("It is 68000.", "Population: 68,000").grounded


def test_an_answer_with_no_figures_is_grounded_by_default():
    check = check_figures("The corpus does not record a surname for Hesh.", "anything")
    assert check.figures_in_answer == []
    assert check.grounded is True
    assert check.support_rate == 1.0


def test_small_integers_are_ignored_as_structural():
    """Passage labels and list positions are not factual claims; flagging them produced noise
    without catching anything a reader would call a fabricated figure."""
    assert check_figures("See passage 2 and point 3.", "no numbers here at all").grounded


def test_a_figure_present_in_the_context_is_supported():
    assert check_figures("Forty-one thousand, or 41,000.", "held 41,000 people").grounded


def test_every_distinct_unsupported_figure_is_listed_once():
    check = check_figures("It was 999 in 999, and also 1234.", "nothing")
    assert check.unsupported == ["999", "1234"]


def test_a_figure_is_not_supported_by_being_a_substring_of_a_longer_one():
    """The corpus is set in 1143 and 1144, so a substring test clears almost any three-digit
    fabrication: `143` appears inside every date in the book. That defeats the whole point of
    having a deterministic check beside a judge known to rubber-stamp."""
    assert check_figures("The toll was 143 marks.", "Frostmark of 1143.").unsupported == ["143"]
    assert check_figures("It cost 144.", "the spring of 1144").unsupported == ["144"]
    # The containing figure itself is still supported.
    assert check_figures("Recorded in 1143.", "Frostmark of 1143.").grounded

    # And the reverse direction: a long figure is not supported by a short one inside it.
    assert check_figures("A population of 68,000.", "68 wagons").unsupported == ["68,000"]


# ---------------------------------------------------------------------------
# the CLI guard on the committed baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--baseline", "--limit", "5"],
        ["--baseline", "--skip-coverage"],
        ["--baseline", "--limit", "5", "--skip-coverage"],
    ],
)
def test_baseline_refuses_flags_that_make_the_run_partial(argv: list[str], monkeypatch):
    """`evals/reports/baseline.json` is the reference every later run is compared against.
    A five-row run or one with the coverage precondition skipped would overwrite it with
    numbers that are not comparable, and the file carries nothing that would say so."""
    from evals.runners import run as run_cli

    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        run_cli.main()

    assert exit_info.value.code == 2  # argparse's usage error


def test_baseline_alone_is_allowed_through_the_guard(monkeypatch):
    """The negative control: the guard must not block a complete run, which is the one way the
    baseline is ever meant to be written."""
    from evals.runners import run as run_cli

    monkeypatch.setattr(sys, "argv", ["run.py", "--baseline"])
    # Stop immediately after the guard — this test is about argument validation, not a real run.
    sentinel = RuntimeError("reached profile loading")

    def _boom(_name: str) -> None:
        raise sentinel

    monkeypatch.setattr(run_cli.Profile, "load", staticmethod(_boom))
    with pytest.raises(RuntimeError, match="reached profile loading"):
        run_cli.main()


# ---------------------------------------------------------------------------
# context recall — the metric that survives a change of chunking scheme
# ---------------------------------------------------------------------------


def test_context_recall_defaults_to_recall_when_there_is_no_expansion():
    """A flat index hands the retrieved chunks straight to the model, so the two numbers are
    the same thing. This is what makes the Phase 2 baseline comparable without re-running it."""
    evidence = [EvidenceRef(doc_slug="d", quote="the ninth stone")]
    retrieved = [RetrievedRef(doc_slug="d", text="crossed at the ninth stone and went south")]
    scored = score_row(evidence, retrieved)

    assert scored.recall == 1.0
    assert scored.context_recall == scored.recall
    assert scored.context_units == 1


def test_context_recall_credits_evidence_the_parent_restored():
    """The case parent-child exists for: the child that was retrieved does not contain the
    quote, but the parent it expanded to does. Recall says the retriever missed; context
    recall says the model got it anyway, and the answer depends on the second."""
    evidence = [EvidenceRef(doc_slug="d", quote="forty-one thousand people")]
    retrieved = [RetrievedRef(doc_slug="d", text="Halvern Reach held")]
    context = [RetrievedRef(doc_slug="d", text="Halvern Reach held forty-one thousand people")]
    scored = score_row(evidence, retrieved, context)

    assert scored.recall == 0.0
    assert scored.context_recall == 1.0


def test_context_recall_does_not_invent_evidence():
    """Expansion can only add the text around a child. Evidence in neither must stay missing."""
    evidence = [EvidenceRef(doc_slug="d", quote="a fact in no passage")]
    retrieved = [RetrievedRef(doc_slug="d", text="something else")]
    context = [RetrievedRef(doc_slug="d", text="something else, at greater length")]
    scored = score_row(evidence, retrieved, context)

    assert scored.recall == 0.0
    assert scored.context_recall == 0.0


def test_summary_reports_mean_context_units():
    """Below k after dedupe: several children of one parent collapse to one passage."""
    evidence = [EvidenceRef(doc_slug="d", quote="x")]
    retrieved = [RetrievedRef(doc_slug="d", text="x"), RetrievedRef(doc_slug="d", text="y")]
    context = [RetrievedRef(doc_slug="d", text="x and y together")]
    summary = summarise([score_row(evidence, retrieved, context)], k=2)

    assert summary.mean_context_units == 1.0
    assert summary.context_recall == 1.0
