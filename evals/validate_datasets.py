"""Validate the Phase 1 golden datasets against the frozen corpus.

# CONCEPT: Dataset validation — an eval suite is only as trustworthy as its ground truth.
# A golden answer that cites a quote which is not in the corpus is not ground truth, it is a
# second model output, and every metric computed against it is noise. This script is the gate
# that lets the corpus be frozen (ADR-014) with confidence.

Run with `make validate`. Exits non-zero on the first category of failure found, after printing
every failure in that category, so one run tells you everything you have to fix.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from evals.render_datasets import DATASETS, READABLE_DIR, render

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
DATASET_DIR = REPO_ROOT / "evals" / "datasets"

GoldenCategory = Literal[
    "single-hop", "multi-hop", "temporal", "numeric", "routing", "unanswerable"
]
AttackCategory = Literal[
    "direct-injection", "jailbreak", "pii-extraction", "off-topic", "toxic", "benign-lookalike"
]

# The budgets the trap map (evals/datasets/README.md) commits to. Kept here so that drifting
# from the map is a test failure rather than something nobody notices for three phases.
GOLDEN_BUDGET: dict[str, int] = {
    "single-hop": 12,
    "multi-hop": 8,
    "temporal": 7,
    "numeric": 6,
    "routing": 3,
    "unanswerable": 6,
}
ATTACK_BUDGET: dict[str, int] = {
    "direct-injection": 7,
    "jailbreak": 7,
    "pii-extraction": 6,
    "off-topic": 5,
    "toxic": 5,
    "benign-lookalike": 10,
}


class SourceChunk(BaseModel):
    doc_slug: str
    quote: str = Field(min_length=12)


class GoldenRow(BaseModel):
    id: str = Field(pattern=r"^gq-\d{3}$")
    question: str = Field(min_length=8)
    answer: str = Field(min_length=2)
    category: GoldenCategory
    source_chunks: list[SourceChunk]
    must_refuse: bool


class AttackRow(BaseModel):
    id: str = Field(pattern=r"^atk-\d{3}$")
    input: str = Field(min_length=8)
    category: AttackCategory
    expected: Literal["block", "allow"]


def normalise(text: str) -> str:
    """Collapse all runs of whitespace to a single space.

    Quotes are copied out of hard-wrapped markdown, so a quote that is correct will still contain
    a newline where the corpus line broke. Normalising both sides makes the check about the words
    and not about the wrapping.
    """
    return re.sub(r"\s+", " ", text).strip()


def load_rows[T: BaseModel](path: Path, model: type[T]) -> tuple[list[T], list[str]]:
    rows: list[T] = []
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"{path.name}:{lineno}  {exc}")
    return rows, errors


def check_ids(rows: list[GoldenRow] | list[AttackRow], prefix: str) -> list[str]:
    ids = [r.id for r in rows]
    errors = [f"duplicate id {i}" for i, n in Counter(ids).items() if n > 1]
    expected = [f"{prefix}-{n:03d}" for n in range(1, len(ids) + 1)]
    if ids != expected:
        missing = sorted(set(expected) - set(ids))
        extra = sorted(set(ids) - set(expected))
        errors.append(
            f"ids are not {prefix}-001..{prefix}-{len(ids):03d}: "
            f"missing={missing} unexpected={extra}"
        )
    return errors


def check_budget(
    rows: list[GoldenRow] | list[AttackRow], budget: dict[str, int], name: str
) -> list[str]:
    counts = Counter(r.category for r in rows)
    return [
        f"{name}: category {cat!r} has {counts.get(cat, 0)} rows, the trap map commits to {want}"
        for cat, want in budget.items()
        if counts.get(cat, 0) != want
    ]


def check_quotes(rows: list[GoldenRow], corpus: dict[str, str]) -> list[str]:
    """# CONCEPT: the whole point of this script — every cited quote must really exist."""
    errors: list[str] = []
    for row in rows:
        for chunk in row.source_chunks:
            if chunk.doc_slug not in corpus:
                errors.append(f"{row.id}: no corpus file named {chunk.doc_slug!r}")
                continue
            if normalise(chunk.quote) not in corpus[chunk.doc_slug]:
                errors.append(f"{row.id}: quote not found in {chunk.doc_slug}: {chunk.quote!r}")
    return errors


def check_golden_shape(rows: list[GoldenRow]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if row.must_refuse != (row.category == "unanswerable"):
            errors.append(f"{row.id}: must_refuse={row.must_refuse} but category={row.category!r}")
        if not row.must_refuse and not row.source_chunks:
            errors.append(f"{row.id}: an answerable row must cite at least one source chunk")
    return errors


def check_attack_shape(rows: list[AttackRow]) -> list[str]:
    return [
        f"{row.id}: category {row.category!r} must expect "
        f"{'allow' if row.category == 'benign-lookalike' else 'block'}, found {row.expected!r}"
        for row in rows
        if (row.expected == "allow") != (row.category == "benign-lookalike")
    ]


def check_readable_mirrors() -> list[str]:
    """The indented copies under datasets/readable/ are generated, so drift is a build failure."""
    errors: list[str] = []
    for name in DATASETS:
        target = READABLE_DIR / f"{name}.json"
        if not target.exists():
            errors.append(f"{target.name} is missing — run `make datasets`")
        elif target.read_text() != render(name):
            errors.append(f"{target.name} is stale — run `make datasets`")
    return errors


def report(stage: str, errors: list[str]) -> bool:
    if errors:
        print(f"\nFAIL  {stage} — {len(errors)} problem(s):")
        for err in errors:
            print(f"  - {err}")
        return False
    print(f"ok    {stage}")
    return True


def main() -> int:
    # README.md is the freeze notice, not a corpus document. The numbered files are the corpus.
    corpus = {
        p.stem: normalise(p.read_text())
        for p in sorted(CORPUS_DIR.glob("*.md"))
        if p.name != "README.md"
    }
    print(f"corpus: {len(corpus)} files in {CORPUS_DIR.relative_to(REPO_ROOT)}")
    if len(corpus) != 10:
        print(f"FAIL  the corpus is frozen at 10 files, found {len(corpus)}")
        return 1

    golden, golden_errors = load_rows(DATASET_DIR / "golden_qa.jsonl", GoldenRow)
    attacks, attack_errors = load_rows(DATASET_DIR / "attacks.jsonl", AttackRow)

    stages = [
        ("golden_qa.jsonl parses and validates", golden_errors),
        ("attacks.jsonl parses and validates", attack_errors),
        ("golden ids are contiguous", check_ids(golden, "gq")),
        ("attack ids are contiguous", check_ids(attacks, "atk")),
        (
            "golden category budget matches the trap map",
            check_budget(golden, GOLDEN_BUDGET, "golden_qa"),
        ),
        (
            "attack category budget matches the trap map",
            check_budget(attacks, ATTACK_BUDGET, "attacks"),
        ),
        ("must_refuse agrees with category", check_golden_shape(golden)),
        ("attack expectations agree with category", check_attack_shape(attacks)),
        ("every golden quote exists verbatim in its corpus file", check_quotes(golden, corpus)),
        ("the readable mirrors match the .jsonl sources", check_readable_mirrors()),
    ]

    ok = True
    for stage, errors in stages:
        ok &= report(stage, errors)

    quoted = sum(len(r.source_chunks) for r in golden)
    print(f"\n{len(golden)} golden rows ({quoted} source chunks), {len(attacks)} attack rows")
    print("PASS — the datasets are consistent with the corpus." if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
