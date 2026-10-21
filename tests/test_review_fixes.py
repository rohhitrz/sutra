"""The 2026-10-21 review findings, one test each.

Each test is written against the *defect*, not the fix: it is the assertion that fails on the
code as reviewed. Where a fix could be mistaken for simply tightening a check, a negative
control sits beside it proving the valid case still passes — a guard that rejects everything
is not a guard.
"""

from __future__ import annotations

import pytest
from evals.runners.run_rag import Profile
from pydantic import ValidationError

# --------------------------------------------------------------------------- finding 5


def test_a_misspelled_profile_key_is_refused() -> None:
    """The defect: `extra` defaulted to "ignore", so `rerank_enabed: true` loaded as a profile
    with reranking off — and filed its report under a name claiming otherwise."""
    with pytest.raises(ValidationError, match="rerank_enabed"):
        Profile.model_validate({"name": "hybrid_rerank", "rerank_enabed": True})


@pytest.mark.parametrize(
    "name",
    [
        "naive",
        "semantic",
        "contextual",
        "hybrid",
        "keyword_only",
        "hybrid_rerank",
        "multiquery",
        "hyde",
        "routed",
        "advanced",
    ],
)
def test_every_committed_profile_loads_under_forbid(name: str) -> None:
    """Negative control over the committed profiles: none of the Phase 3 runs was measuring a
    configuration other than the one its file names."""
    assert Profile.load(name).name
