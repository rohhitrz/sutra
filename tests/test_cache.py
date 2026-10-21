"""The semantic cache's guard against wrong hits (Phase 3 step 7).

Unit-level, no database: the part worth pinning is the deterministic figure guard, because it
is what stops a confident, fluent, wrong answer being served with nothing downstream able to
detect it. The end-to-end behaviour is covered by `make cache`.
"""

from __future__ import annotations

from sutra.cache.semantic import discriminating_figures


def test_two_questions_differing_only_by_a_year_do_not_agree():
    """The case no threshold can catch. Measured on this corpus, the wrong hit sits at cosine
    0.954 and a correct paraphrase at 0.955 — one thousandth apart, so there is no setting
    that admits the paraphrase and rejects the year."""
    a = discriminating_figures("What was the Waystone Tithe rate in 1143?")
    b = discriminating_figures("What was the Waystone Tithe rate in 1144?")
    assert a != b


def test_a_paraphrase_with_no_figures_still_agrees():
    """The guard must not block the thing the cache is for."""
    a = discriminating_figures("What is the population of Halvern Reach?")
    b = discriminating_figures("How many people live in Halvern Reach?")
    assert a == b == frozenset()


def test_thousands_separators_do_not_split_a_match():
    assert discriminating_figures("is it 2,300?") == discriminating_figures("is it 2300?")


def test_trailing_punctuation_is_not_part_of_the_figure():
    """Same greedy-regex trap as the grounding check: `1143,` would otherwise never match."""
    assert discriminating_figures("in 1143, the flood") == frozenset({"1143"})


def test_waystone_numbers_discriminate():
    """Questions about stone 17 and stone 19 are near-identical as text and different facts."""
    assert discriminating_figures("what is at waystone 17?") != discriminating_figures(
        "what is at waystone 19?"
    )
