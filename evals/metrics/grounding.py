"""A mechanical grounding check, to cross-examine the faithfulness judge.

# CONCEPT: An LLM judge grades with the same faculties that produced the answer, and ADR-017
# points both roles at one model, so the judge can share the answer's mistakes. On the first
# full eval run of this project it did exactly that:
#
#   question  : What is the population of Kesh Ithar?
#   answer    : "Kesh Ithar has a population of 68,000 people (recorded in Frostmark of 1143)."
#   judge     : faithfulness 1.0 — "The passage states that Kesh Ithar has 68,000 people."
#   the truth : "68,000" appears in none of the five retrieved passages, nor does the word
#               "population". "Frostmark of 1143" *is* there — so the model took the one real
#               detail from the context, attached it to an invented figure, and the judge
#               anchored on the verifiable half to approve the unverifiable one.
#
# This module is the deterministic floor under that judgement. It cannot assess reasoning, but a
# figure either appears in the context or it does not, and no amount of fluency changes that.
# Cheap, reproducible, and immune to the failure above.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

# A run of digits, optionally with thousands separators or a decimal part. Deliberately narrow:
# the aim is high precision on a specific, checkable claim type, not broad coverage.
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Small integers are excluded because they are usually structural rather than factual — passage
# labels ("[2]"), list positions, "one of three". Including them produced false alarms without
# catching anything a reader would call a fabricated figure.
MIN_INTERESTING = 10


class GroundingCheck(BaseModel):
    """Which figures in an answer are absent from the context it was given."""

    figures_in_answer: list[str]
    unsupported: list[str]

    @property
    def grounded(self) -> bool:
        return not self.unsupported

    @property
    def support_rate(self) -> float:
        """Fraction of the answer's figures that appear in the context. 1.0 when it cites none."""
        if not self.figures_in_answer:
            return 1.0
        supported = len(self.figures_in_answer) - len(self.unsupported)
        return supported / len(self.figures_in_answer)


def _canonical(text: str) -> str:
    """Drop thousands separators so "68,000" and "68000" compare equal."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def _figures_in(text: str) -> set[str]:
    """Every figure the text *states*, as canonical digit strings.

    Support has to be tested against this set and never as a substring of the raw context. A
    substring test counts a figure as supported whenever its digits appear anywhere inside a
    longer number, and on this corpus that is not a corner case: the two book years are 1143
    and 1144, so a fabricated `143` or `144` is "supported" by every passage that mentions a
    date. That silently disarms the one check that does not share the judge's blind spots.
    """
    return {match.group(0) for match in _FIGURE.finditer(_canonical(text))}


def check_figures(answer: str, context: str) -> GroundingCheck:
    """Every digit figure in `answer` that does not appear in `context`.

    Limitations, stated because a check whose blind spots are unknown is worse than none:
      - **A derived figure is not a fabricated one.** This is the big one. The corpus contains a
        merchant's ledger whose totals are units x per-unit on purpose, so a correct answer
        routinely computes a figure that appears nowhere in the passages. `2,750` summed from
        `946 + 1,034 + 770` is right, and flagged here. Absence from the context is therefore a
        *question*, not a verdict, and the runner reads it together with recall: unsupported
        figure plus missing evidence is suspicious, unsupported figure with every cited quote
        retrieved is usually arithmetic.
      - **Digits only.** The corpus writes many numbers as words ("forty waystones"), and those
        pass unexamined. A fabricated "forty" would not be caught here.
      - **Presence, not meaning.** A figure lifted from the wrong row of a ledger is "supported"
        by this check. Only the judge can see that, which is why both are reported.
      - Figures below `MIN_INTERESTING` are skipped as structural.
      - **Whole figures only.** `143` is not supported by `1143`. The cost of that strictness is
        that a figure written differently in the context ("forty-one thousand" against `41,000`)
        reads as unsupported — a false alarm, which the runner surfaces for a human, rather than
        a false clearance, which nobody ever sees.
    """
    context_figures = _figures_in(context)
    figures: list[str] = []
    unsupported: list[str] = []

    for match in _FIGURE.finditer(answer):
        raw = match.group(0).rstrip(".,")
        canonical = _canonical(raw)
        if not canonical.isdigit() and "." not in canonical:
            continue
        try:
            if float(canonical) < MIN_INTERESTING:
                continue
        except ValueError:
            continue
        if raw in figures:
            continue
        figures.append(raw)
        if canonical not in context_figures:
            unsupported.append(raw)

    return GroundingCheck(figures_in_answer=figures, unsupported=unsupported)
