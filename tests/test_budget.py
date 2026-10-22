"""Tests for the context token budget (ARCHITECTURE §5, Phase 5).

The behaviour worth pinning here is the *order* things are given up in, and the two things
that are never given up at all. A budget that fits by dropping the question would also pass a
test that only checked the total.

Every trim test drives a deliberately tiny window rather than building a megabyte of text: the
trim order is the subject, and the limit is the cheapest thing to vary.
"""

from __future__ import annotations

from sutra import config
from sutra.config import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    LONG_CONTEXT_THRESHOLD_TOKENS,
    Settings,
    context_window_tokens,
)
from sutra.generation.budget import (
    CHARS_PER_TOKEN,
    HistoryTurn,
    estimate_tokens,
    fit_context,
)
from sutra.retrieval.search import RetrievedChunk

TINY_MODEL = "test-tiny-model"


def chunk(n: int, *, chars: int = 400) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{n}",
        doc_slug="06_places_wiki",
        doc_title="Places of the Meridian Line",
        doc_type="wiki",
        chunk_no=n,
        heading="Kesh Ithar",
        text="x" * chars,
        token_count=chars // 4,
        score=0.9 - n / 100,
        rank=n,
    )


def turn(n: int, *, chars: int = 400) -> HistoryTurn:
    return HistoryTurn(role="user" if n % 2 else "assistant", content=str(n) * chars)


def settings_for(window: int, monkeypatch, *, min_chunks: int = 1) -> Settings:
    """Settings pointed at a model whose window is `window` tokens.

    `context_budget_safety` is 1.0 so each test's arithmetic is its own, rather than every
    assertion also having to encode the safety margin.
    """
    monkeypatch.setitem(config.MODEL_CONTEXT_WINDOWS, TINY_MODEL, window)
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_strong=TINY_MODEL,
        answer_reserve_tokens=10,
        context_budget_safety=1.0,
        context_min_chunks=min_chunks,
    )


def roomy() -> Settings:
    return Settings(_env_file=None, context_budget_safety=1.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------


def test_the_estimate_errs_high_against_the_prose_average():
    """Erring low costs a rejected request; erring high costs unused window. The divisor has
    to stay below the ~4.0 chars/token English prose averages for that to hold."""
    assert CHARS_PER_TOKEN < 4.0
    assert estimate_tokens("a" * 400) > 400 / 4.0


def test_empty_text_costs_nothing():
    assert estimate_tokens("") == 0


def test_an_unknown_model_gets_the_conservative_floor_not_a_guess():
    """A model nobody has entered is the case where a permissive default would be dangerous:
    an unknown price produces a wrong number, an unknown window produces a rejected request."""
    assert context_window_tokens("gpt-6-luna") == LONG_CONTEXT_THRESHOLD_TOKENS
    assert context_window_tokens("some-model-shipped-tomorrow") == DEFAULT_CONTEXT_WINDOW_TOKENS
    assert DEFAULT_CONTEXT_WINDOW_TOKENS < LONG_CONTEXT_THRESHOLD_TOKENS


# ---------------------------------------------------------------------------
# the trim order: chunks -> history -> summary
# ---------------------------------------------------------------------------


def test_nothing_is_trimmed_when_everything_fits():
    plan = fit_context(
        system="system block",
        question="how many waystones?",
        chunks=[chunk(1), chunk(2)],
        history=[turn(1)],
        summary="a summary",
        settings=roomy(),
    )
    assert plan.trimmed is False
    assert len(plan.chunks) == 2
    assert len(plan.history) == 1
    assert plan.summary == "a summary"
    assert plan.over_budget is False


def test_chunks_are_given_up_before_history(monkeypatch):
    """The k-th passage is the least relevant thing in the prompt by construction — retrieval
    already ranked it last — so it is the cheapest thing to lose."""
    settings = settings_for(150, monkeypatch)
    history = [turn(1, chars=100)]

    plan = fit_context(
        system="s",
        question="q",
        chunks=[chunk(n, chars=200) for n in range(1, 5)],
        history=history,
        settings=settings,
    )
    assert plan.chunks_dropped == 3
    assert plan.history == history
    assert plan.history_dropped == 0


def test_the_lowest_ranked_passage_is_the_one_that_goes(monkeypatch):
    settings = settings_for(250, monkeypatch)
    chunks = [chunk(n, chars=200) for n in range(1, 5)]

    plan = fit_context(system="s", question="q", chunks=chunks, settings=settings)

    assert plan.chunks_dropped > 0
    # What survives is the head of the ranked list, never a hole in the middle of it.
    assert plan.chunks == chunks[: len(plan.chunks)]


def test_history_is_only_trimmed_once_the_chunks_are_at_the_floor(monkeypatch):
    settings = settings_for(200, monkeypatch)

    plan = fit_context(
        system="s",
        question="q",
        chunks=[chunk(n, chars=200) for n in range(1, 5)],
        history=[turn(n, chars=200) for n in range(1, 5)],
        settings=settings,
    )
    assert len(plan.chunks) == settings.context_min_chunks
    assert plan.chunks_dropped == 3
    assert plan.history_dropped > 0


def test_the_oldest_turn_is_the_one_that_goes(monkeypatch):
    settings = settings_for(150, monkeypatch)
    history = [turn(n, chars=120) for n in range(1, 5)]

    plan = fit_context(
        system="s",
        question="q",
        chunks=[chunk(1, chars=100)],
        history=history,
        settings=settings,
    )
    assert plan.history_dropped > 0
    # What survives is the tail of the conversation, not its head.
    assert plan.history == history[plan.history_dropped :]


def test_the_summary_is_truncated_last_and_never_dropped_outright(monkeypatch):
    settings = settings_for(150, monkeypatch)

    plan = fit_context(
        system="s",
        question="q",
        chunks=[chunk(1, chars=100)],
        history=[turn(1, chars=400)],
        summary="the oldest part. " * 30 + "THE MOST RECENT PART.",
        settings=settings,
    )
    assert plan.history == []
    assert plan.summary_truncated is True
    assert plan.summary != ""
    # Cutting the tail would throw away the most recent turns, which is backwards: a rolling
    # summary is written oldest-first.
    assert plan.summary.endswith("THE MOST RECENT PART.")


# ---------------------------------------------------------------------------
# what is never trimmed
# ---------------------------------------------------------------------------


def test_the_chunk_floor_holds_even_when_it_does_not_fit(monkeypatch):
    """Zero passages is not a cheaper answer, it is a refusal. The budget reports that it
    could not fit rather than trimming the turn into one."""
    settings = settings_for(20, monkeypatch)

    plan = fit_context(system="s", question="q", chunks=[chunk(1, chars=4000)], settings=settings)
    assert len(plan.chunks) == 1
    assert plan.over_budget is True


def test_the_system_block_and_the_question_are_always_priced_in():
    plan = fit_context(
        system="a long system block " * 20,
        question="how many waystones stand on the Meridian Line?",
        chunks=[chunk(1)],
        settings=roomy(),
    )
    assert plan.parts.system > 0
    assert plan.parts.question > 0


def test_the_answer_reserve_is_part_of_the_total():
    """The provider bills reasoning and visible output against one cap, so a context assembled
    to fill the whole window leaves the answer no room to be written in."""
    settings = Settings(_env_file=None, answer_reserve_tokens=4000)  # type: ignore[call-arg]
    plan = fit_context(system="s", question="q", chunks=[chunk(1)], settings=settings)
    assert plan.parts.answer_reserve == 4000
    assert plan.parts.total == plan.parts.input_tokens + 4000


# ---------------------------------------------------------------------------
# the accounting
# ---------------------------------------------------------------------------


def test_the_plan_reports_tokens_per_component_for_the_log_line():
    """'Long chats stay within budget' has to be a number somebody can read off a turn."""
    plan = fit_context(
        system="s",
        question="q",
        chunks=[chunk(1)],
        memories=["The reader prefers short answers."],
        summary="earlier: the ledger was discussed",
        history=[turn(1, chars=50)],
        settings=roomy(),
    )
    meta = plan.as_metadata()
    for key in ("budget_limit", "budget_input_tokens", "budget_chunks", "budget_memories"):
        assert key in meta
    assert meta["budget_memories"] > 0
    assert plan.parts.input_tokens == (
        plan.parts.system
        + plan.parts.memories
        + plan.parts.summary
        + plan.parts.history
        + plan.parts.chunks
        + plan.parts.question
    )
