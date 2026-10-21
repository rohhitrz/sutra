"""Query routing (Phase 3 step 6).

The asymmetry under every test here: a correct narrow route saves a few retrieval slots, a
wrong one hides the answer with no second pass. So the tests care much more about the router
declining to narrow than about it narrowing well.
"""

from __future__ import annotations

from typing import Any

from sutra.config import Settings
from sutra.prompts import PromptLoader
from sutra.retrieval.router import FILTERS, RouteDecision, route, route_by_rule


class FakeClient:
    def __init__(self, intent: str = "general", *, fail: bool = False) -> None:
        self.intent = intent
        self.fail = fail
        self.calls = 0

    async def structured(self, messages: Any, schema: type, **kwargs: Any) -> Any:
        self.calls += 1
        if self.fail:
            msg = "router model down"
            raise RuntimeError(msg)

        class Response:
            parsed = schema(intent=self.intent)

        return Response()


def _on(**over: object) -> Settings:
    return Settings(**{"router_enabled": True, **over})  # type: ignore[arg-type]


def test_rules_fire_on_unambiguous_wording():
    assert route_by_rule("How much did the salt cost?") == "numeric"
    assert route_by_rule("In what year did the flood happen?") == "timeline"
    assert route_by_rule("Who is Sera Volant?") == "character"


def test_rules_decline_anything_subtler():
    """A rule that guesses is worse than no rule, because the model it pre-empts is better
    at the hard cases and the cost of a wrong narrow route is the whole answer."""
    assert route_by_rule("Describe the Ashen Flats") is None
    assert route_by_rule("What happened to the caravan?") is None


async def test_general_applies_no_filter():
    """The common and safe route. It must cost nothing — not a filter matching everything,
    but no WHERE clause at all."""
    decision = await route(
        "Describe the Ashen Flats",
        client=FakeClient("general"),  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_on(),
    )
    assert decision.intent == "general"
    assert decision.doc_types == ()
    assert decision.filters_anything is False


async def test_a_rule_match_skips_the_model():
    """Most routing on this corpus is visible in the question's own words, and a rule is
    free, instant and auditable where a classifier is none of those."""
    client = FakeClient("timeline")
    decision = await route(
        "How much did the salt cost?",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_on(),
    )
    assert decision.intent == "numeric"
    assert decision.source == "rule"
    assert client.calls == 0


async def test_a_failed_router_falls_back_to_no_filter():
    """A router that can fail a request is worse than no router: it buys a little precision
    and would cost the answer."""
    decision = await route(
        "Describe the Ashen Flats",
        client=FakeClient(fail=True),  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=_on(),
    )
    assert decision == RouteDecision(intent="general", source="fallback")


async def test_disabled_router_makes_no_call():
    client = FakeClient()
    decision = await route(
        "How much did the salt cost?",
        client=client,  # type: ignore[arg-type]
        loader=PromptLoader(),
        settings=Settings(),
    )
    assert decision.doc_types == ()
    assert client.calls == 0


def test_every_filter_is_wider_than_the_obvious_answer():
    """Encodes the design rule rather than trusting it to survive an edit. A `numeric` filter
    of ledger-only would be exactly the confident-and-wrong case this module exists to avoid:
    populations and tolls are reference facts, and dated figures live in the timeline."""
    for intent, types in FILTERS.items():
        assert len(types) >= 3, f"{intent} filter is too narrow to be safe"
    assert "wiki" in FILTERS["numeric"]
    assert "narrative" in FILTERS["timeline"]
