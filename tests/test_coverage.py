"""Tests for the quote-coverage guard.

Includes negative controls, following the Phase 1 validator: a check that cannot be made to fail
is not evidence of anything. Both failure causes are exercised, because they need opposite fixes —
one is a dataset bug, the other a chunker bug.
"""

from __future__ import annotations

import pytest
from evals.check_coverage import check_coverage, chunks_from_corpus, corpus_text, report

from sutra.config import Settings

# From gq-001, cited against 08_faq. Used to build a deliberately split chunk boundary.
KNOWN_QUOTE = "Forty. One at Halvern Reach, forty at Kesh Ithar."


@pytest.fixture(scope="module")
def real_chunks(settings_module: Settings) -> dict[str, list[str]]:
    return chunks_from_corpus(settings_module)


@pytest.fixture(scope="module")
def settings_module() -> Settings:
    """Module-scoped settings, because `chunks_from_corpus` loads the embedding model."""
    return Settings(app_env="ci", openai_api_key="test-key")


def test_every_cited_quote_survives_chunking(real_chunks: dict[str, list[str]]):
    """The gate on Phase 2's retrieval metrics.

    A quote split across a chunk boundary is unmatchable no matter how good retrieval is, so
    recall would silently understate and look like a retriever problem.
    """
    results = check_coverage(real_chunks, corpus_text())
    uncovered = [r for r in results if not r.covered]
    assert len(results) == 72, "the golden set cites 72 quotes"
    assert not uncovered, [f"{r.row_id} {r.doc_slug}: {r.diagnosis}" for r in uncovered]


def test_overlap_can_place_a_quote_in_more_than_one_chunk(real_chunks: dict[str, list[str]]):
    """Not a fault: overlap exists so a fact near a boundary appears somewhere whole."""
    results = check_coverage(real_chunks, corpus_text())
    assert any(len(r.chunks_containing) > 1 for r in results)


# ---------------------------------------------------------------------------
# negative controls — the check must be able to fail
# ---------------------------------------------------------------------------


def test_a_quote_split_across_chunks_is_caught_and_named_a_chunker_fault():
    corpus = corpus_text()
    # The quote is present in the corpus, but no single chunk holds all of it.
    split = {"08_faq": ["Forty. One at Halvern", "Reach, forty at Kesh Ithar."]}
    results = check_coverage(split, corpus)

    failed = [r for r in results if r.doc_slug == "08_faq" and not r.covered]
    assert failed, "a split quote must be reported"
    assert failed[0].in_corpus is True
    assert "SPLIT" in failed[0].diagnosis


def test_a_quote_absent_from_the_corpus_is_caught_and_named_a_dataset_fault():
    results = check_coverage({"08_faq": ["unrelated text"]}, {"08_faq": "unrelated text"})
    failed = [r for r in results if r.doc_slug == "08_faq" and not r.covered]
    assert failed
    assert failed[0].in_corpus is False
    assert "NOT IN CORPUS" in failed[0].diagnosis


def test_report_exits_non_zero_when_a_quote_is_uncovered(capsys: pytest.CaptureFixture[str]):
    """`make coverage` has to fail the build, not just print something."""
    results = check_coverage({"08_faq": ["nothing relevant here"]}, corpus_text())
    assert report(results, "test") == 1
    assert "FAIL" in capsys.readouterr().out


def test_report_exits_zero_when_everything_is_covered(
    real_chunks: dict[str, list[str]], capsys: pytest.CaptureFixture[str]
):
    assert report(check_coverage(real_chunks, corpus_text()), "test") == 0
    assert "ok" in capsys.readouterr().out
