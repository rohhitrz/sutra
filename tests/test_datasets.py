"""The Phase 1 datasets must stay consistent with the corpus, forever.

# CONCEPT: the validator is itself tested, including a negative control. A checker that cannot
# fail is not a checker, and "the validator passed" is only evidence if we have watched it fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.render_datasets import main as render_main
from evals.validate_datasets import (
    ATTACK_BUDGET,
    GOLDEN_BUDGET,
    AttackRow,
    GoldenRow,
    check_quotes,
    load_rows,
    main,
    normalise,
)

DATASET_DIR = Path(__file__).resolve().parent.parent / "evals" / "datasets"


def test_validator_passes_on_the_real_datasets() -> None:
    assert main() == 0


def test_normalise_collapses_hard_wrapping() -> None:
    assert normalise("the only place in the Vessel's whole\nlength where") == (
        "the only place in the Vessel's whole length where"
    )


def test_check_quotes_catches_a_quote_that_is_not_in_the_corpus() -> None:
    """The negative control: a plausible but invented quote must be rejected."""
    row = GoldenRow(
        id="gq-999",
        question="Who built the waystones?",
        answer="The Concord did.",
        category="single-hop",
        source_chunks=[  # type: ignore[list-item]
            {"doc_slug": "06_places_wiki", "quote": "The waystones were set by the Concord"}
        ],
        must_refuse=False,
    )
    errors = check_quotes([row], {"06_places_wiki": normalise("**Who set them is not known.**")})
    assert len(errors) == 1
    assert "quote not found" in errors[0]


def test_check_quotes_catches_an_unknown_doc_slug() -> None:
    row = GoldenRow(
        id="gq-999",
        question="Who built the waystones?",
        answer="Nobody knows.",
        category="single-hop",
        source_chunks=[{"doc_slug": "11_does_not_exist", "quote": "anything at all here"}],  # type: ignore[list-item]
        must_refuse=False,
    )
    assert "no corpus file named" in check_quotes([row], {})[0]


@pytest.mark.parametrize(
    ("filename", "model", "budget"),
    [("golden_qa.jsonl", GoldenRow, GOLDEN_BUDGET), ("attacks.jsonl", AttackRow, ATTACK_BUDGET)],
)
def test_every_row_parses_and_the_budget_is_met(
    filename: str, model: type[GoldenRow] | type[AttackRow], budget: dict[str, int]
) -> None:
    rows, errors = load_rows(DATASET_DIR / filename, model)  # type: ignore[type-var]
    assert errors == []
    assert len(rows) == sum(budget.values())


def test_the_datasets_are_one_json_object_per_line() -> None:
    for filename in ("golden_qa.jsonl", "attacks.jsonl"):
        for line in (DATASET_DIR / filename).read_text().splitlines():
            assert isinstance(json.loads(line), dict)


def test_readable_mirrors_are_in_sync_with_the_jsonl_sources() -> None:
    """A generated view that has drifted is worse than no view: it reads as ground truth."""
    assert render_main(check=True) == 0


def test_a_stale_mirror_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control for the sync check."""
    import evals.render_datasets as renderer

    monkeypatch.setattr(renderer, "READABLE_DIR", tmp_path)
    (tmp_path / "golden_qa.json").write_text("{}")
    (tmp_path / "attacks.json").write_text("{}")
    assert renderer.main(check=True) == 1
