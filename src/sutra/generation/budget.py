"""Fitting one turn's context into the model's window, and deciding what to drop first.

# CONCEPT: A context window is a hard wall, not a soft preference. Cross it and the provider
# rejects the whole request — so something has to be dropped, and *what* gets dropped is a
# design decision rather than an accident of assembly order. The naive version ("send
# everything and hope") works until the first long conversation, at which point it fails on
# the turn that needed memory most.

Sutra's budget is ARCHITECTURE §5's:

    system + memory + history + chunks + answer reserve  <=  window x safety

and its trim order is §5's too: **fewer chunks -> shorter history -> shorter summary**. That
order encodes what is cheapest to lose. The k-th retrieved passage is the least relevant thing
in the prompt by construction (retrieval already ranked it last), an old turn is usually
already represented in the rolling summary, and the summary is the only record of everything
that came before it — so it is trimmed last and never dropped outright.

Two things are deliberately *not* trimmable. The system block is what makes the answer cited
and honest, and the question is what is being asked; a budget that can drop either has
misunderstood which part of the prompt is the product.

# CONCEPT: Counting tokens needs the provider's tokenizer, which we do not have locally, so
# this module **estimates** and errs high. Erring high costs a little unused window; erring low
# costs a 400 on the request that overflows — which is why the divisor below sits under the
# ~4.0 chars/token English prose averages rather than on it.
#
# An estimator nobody checks is a guess with a docstring. `generate_answer` logs this estimate
# beside the provider's own reported `input_tokens` on every call (`budget.drift`), so the
# divisor is a measurement that can be re-tuned from real runs rather than a constant somebody
# once believed. It is **not yet calibrated against a full eval run** — see PROGRESS.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from sutra.config import Settings, context_window_tokens, get_settings
from sutra.retrieval.search import RetrievedChunk

logger = logging.getLogger(__name__)

# Characters per token, deliberately *below* the ~4.0 that English prose averages, so the
# estimate lands above the truth. Measured on this corpus in Phase 5; see the README.
CHARS_PER_TOKEN = 3.6

# Per-message framing (role markers and delimiters) that the provider adds and the text does
# not show. Small, fixed, and cheaper to over-reserve than to discover at the wall.
TOKENS_PER_MESSAGE = 4

# The `[3] ledger-of-tides · The Ledger of Tides · Second Crossing` header `format_context`
# writes above each passage. A reserve rather than a measurement, so this module does not have
# to import the renderer (and so answer.py can change the header without changing the budget).
TOKENS_PER_CHUNK_HEADER = 24


class HistoryTurn(BaseModel):
    """One earlier message in this conversation, as the prompt will carry it.

    Defined here rather than in `sutra.memory` because the budget is what consumes it, and the
    budget exists a step before the memory module does.
    """

    role: str
    content: str


def estimate_tokens(text: str) -> int:
    """Estimate the tokens `text` will cost, rounding up.

    Not exact, and not trying to be: see the module docstring on why this errs high.
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_turn_tokens(turn: HistoryTurn) -> int:
    """A history turn costs its content plus the framing the provider wraps it in."""
    return estimate_tokens(turn.content) + TOKENS_PER_MESSAGE


def estimate_chunk_tokens(chunk: RetrievedChunk) -> int:
    """A passage costs its text plus the numbered header it is shown under."""
    return estimate_tokens(chunk.text) + TOKENS_PER_CHUNK_HEADER


class BudgetParts(BaseModel):
    """What each part of the assembled prompt costs, after trimming.

    Recorded per turn so "long chats stay within budget" is a logged number rather than a
    claim, and so the first turn that trims anything is visible in the trace instead of being
    inferred from a shorter answer.
    """

    system: int = 0
    memories: int = 0
    summary: int = 0
    history: int = 0
    chunks: int = 0
    question: int = 0
    answer_reserve: int = 0

    @property
    def input_tokens(self) -> int:
        """Everything that goes *into* the request, excluding the reserve for the answer."""
        return (
            self.system + self.memories + self.summary + self.history + self.chunks + self.question
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.answer_reserve


class BudgetPlan(BaseModel):
    """The trimmed context, plus an honest account of what it took to fit.

    `over_budget` is the case where even the floors do not fit. It is reported rather than
    raised: a conversation that has grown too long should degrade, not 500, and the provider
    remains the backstop for the case this estimator gets wrong.
    """

    memories: list[str] = Field(default_factory=list)
    summary: str = ""
    history: list[HistoryTurn] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)

    parts: BudgetParts = Field(default_factory=BudgetParts)
    limit: int = 0

    chunks_dropped: int = 0
    history_dropped: int = 0
    summary_truncated: bool = False
    over_budget: bool = False

    @property
    def trimmed(self) -> bool:
        """Whether anything was given up to make this turn fit."""
        return bool(self.chunks_dropped or self.history_dropped or self.summary_truncated)

    def as_metadata(self) -> dict[str, int | bool]:
        """The flat shape a span or a log line wants."""
        return {
            "budget_limit": self.limit,
            "budget_input_tokens": self.parts.input_tokens,
            "budget_total_tokens": self.parts.total,
            "budget_system": self.parts.system,
            "budget_memories": self.parts.memories,
            "budget_summary": self.parts.summary,
            "budget_history": self.parts.history,
            "budget_chunks": self.parts.chunks,
            "chunks_dropped": self.chunks_dropped,
            "history_dropped": self.history_dropped,
            "summary_truncated": self.summary_truncated,
            "over_budget": self.over_budget,
        }


def _truncate_to_tokens(text: str, allowance: int) -> str:
    """Cut `text` to roughly `allowance` tokens, on a word boundary where one is near.

    Truncating the *tail* rather than the head: a rolling summary is written oldest-first, so
    the end is the most recent and most relevant part — which is why what survives here is the
    end, not the beginning.
    """
    if allowance <= 0:
        return ""
    keep_chars = int(allowance * CHARS_PER_TOKEN)
    if len(text) <= keep_chars:
        return text
    tail = text[-keep_chars:]
    space = tail.find(" ")
    if 0 <= space <= 40:
        tail = tail[space + 1 :]
    return "..." + tail


def fit_context(
    *,
    system: str,
    question: str,
    chunks: Sequence[RetrievedChunk],
    memories: Sequence[str] = (),
    summary: str = "",
    history: Sequence[HistoryTurn] = (),
    settings: Settings | None = None,
    model: str | None = None,
) -> BudgetPlan:
    """Trim `chunks`, `history` and `summary` until the turn fits, and say what it cost.

    The parts are passed as the materials they are, not as a rendered prompt, because trimming
    a rendered string means re-rendering it — and because dropping "the last passage" is only
    expressible while the passages are still a list.
    """
    settings = settings or get_settings()
    model = model or settings.llm_strong

    window = context_window_tokens(model)
    limit = int(window * settings.context_budget_safety)
    reserve = settings.answer_reserve_tokens

    kept_chunks = list(chunks)
    kept_history = list(history)
    kept_summary = summary
    kept_memories = list(memories)

    fixed = (
        estimate_tokens(system)
        + TOKENS_PER_MESSAGE
        + estimate_tokens(question)
        + TOKENS_PER_MESSAGE
        + sum(estimate_tokens(m) for m in kept_memories)
        + reserve
    )

    chunk_costs = [estimate_chunk_tokens(c) for c in kept_chunks]
    history_costs = [estimate_turn_tokens(t) for t in kept_history]
    summary_cost = estimate_tokens(kept_summary)

    def used() -> int:
        return fixed + sum(chunk_costs) + sum(history_costs) + summary_cost

    chunks_dropped = 0
    history_dropped = 0
    summary_truncated = False

    # --- trim 1: fewer chunks. The last one is the lowest-ranked, so it goes first.
    while used() > limit and len(chunk_costs) > settings.context_min_chunks:
        chunk_costs.pop()
        kept_chunks.pop()
        chunks_dropped += 1

    # --- trim 2: shorter history. The oldest turn goes first; the rolling summary is what is
    # supposed to be carrying it by now.
    while used() > limit and history_costs:
        history_costs.pop(0)
        kept_history.pop(0)
        history_dropped += 1

    # --- trim 3: shorter summary. Last, because it is the only record of everything that the
    # window has already forgotten, and truncated rather than dropped for the same reason.
    if used() > limit and kept_summary:
        allowance = summary_cost - (used() - limit)
        kept_summary = _truncate_to_tokens(kept_summary, allowance)
        summary_cost = estimate_tokens(kept_summary)
        summary_truncated = True

    parts = BudgetParts(
        system=estimate_tokens(system) + TOKENS_PER_MESSAGE,
        memories=sum(estimate_tokens(m) for m in kept_memories),
        summary=summary_cost,
        history=sum(history_costs),
        chunks=sum(chunk_costs),
        question=estimate_tokens(question) + TOKENS_PER_MESSAGE,
        answer_reserve=reserve,
    )
    plan = BudgetPlan(
        memories=kept_memories,
        summary=kept_summary,
        history=kept_history,
        chunks=kept_chunks,
        parts=parts,
        limit=limit,
        chunks_dropped=chunks_dropped,
        history_dropped=history_dropped,
        summary_truncated=summary_truncated,
        over_budget=used() > limit,
    )

    if plan.over_budget:
        # Not an exception: see BudgetPlan. But it is the one state here worth an ERROR, because
        # it means the floors themselves do not fit and the provider is about to be the one to
        # say so.
        logger.error(
            "context does not fit even at the floor: %d tokens against a %d limit",
            parts.total,
            limit,
            extra=plan.as_metadata(),
        )
    elif plan.trimmed:
        logger.info("context.trimmed", extra=plan.as_metadata())

    return plan
