"""Query routing and metadata filters — step 1 of the ARCHITECTURE §4 pipeline.

# CONCEPT: Routing narrows the search space before searching it. If a question is plainly
# about a price, the ledger is where the answer lives and the other nine documents are noise
# competing for the same five slots. Restricting by document type raises precision for free.
#
# The asymmetry that governs the whole design: a *correct* narrow route saves a few slots, and
# an *incorrect* one hides the answer completely, with no second pass to recover it. The
# expected value of a confident guess is therefore bad, which is why the rules below fire only
# on unambiguous signals, the prompt is told to prefer `general`, and `general` applies no
# filter at all.
#
# Rules run before the model. Most of what a router does on this corpus is recognisable from
# the question's own words — "how much", "what year" — and a rule is free, instant and
# auditable where a classifier is none of those. The model handles only what the rules leave.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from sutra.config import Settings, get_settings
from sutra.llm.client import LLMClient, client_scope
from sutra.llm.types import Message
from sutra.observability.tracing import stage
from sutra.prompts import PromptLoader

logger = logging.getLogger(__name__)

Intent = Literal["timeline", "character", "numeric", "general", "out_of_scope"]

MAX_ROUTER_TOKENS = 1000

# Which document types each intent is allowed to search. `general` is deliberately absent:
# no entry means no filter, which is the safe default and the common case.
#
# Each list is wider than the obvious answer. `numeric` keeps the wiki because populations and
# tolls are reference facts, and the timeline because dated figures live there — a filter that
# kept only the ledger would be the confident-and-wrong case this module exists to avoid.
FILTERS: dict[str, tuple[str, ...]] = {
    "timeline": ("timeline", "narrative", "wiki"),
    "character": ("registry", "narrative", "wiki"),
    "numeric": ("ledger", "wiki", "timeline", "faq"),
}

# Rules fire only on unambiguous openings. Anything subtler goes to the model.
_RULES: tuple[tuple[re.Pattern[str], Intent], ...] = (
    (
        re.compile(
            r"\b(how much|how many|what (?:did|does) .{0,40}\b(?:cost|pay|owe)|total value|"
            r"what is the total)\b",
            re.I,
        ),
        "numeric",
    ),
    (
        re.compile(
            r"\b(in what year|what year|when did|when was|how long after|"
            r"what changed between)\b",
            re.I,
        ),
        "timeline",
    ),
    (re.compile(r"\b(who is|who was|whose|who holds|who kept)\b", re.I), "character"),
)


class RouteDecision(BaseModel):
    """What the router concluded, and how."""

    intent: Intent
    doc_types: tuple[str, ...] = ()
    source: Literal["rule", "model", "disabled", "fallback"] = "disabled"

    @property
    def filters_anything(self) -> bool:
        return bool(self.doc_types)


class _RouterVerdict(BaseModel):
    intent: Intent


def route_by_rule(question: str) -> Intent | None:
    """The intent, when the question's own words settle it. None when they do not."""
    for pattern, intent in _RULES:
        if pattern.search(question):
            return intent
    return None


async def route(
    question: str,
    *,
    client: LLMClient | None = None,
    loader: PromptLoader | None = None,
    settings: Settings | None = None,
) -> RouteDecision:
    """Classify a question and return the document types retrieval may search.

    Never raises. A router that can fail a request is worse than no router, because the thing
    it buys is a few points of precision and the thing it would cost is the answer.
    """
    settings = settings or get_settings()
    if not settings.router_enabled:
        return RouteDecision(intent="general", source="disabled")

    with stage("route", kind="span", input={"question": question}) as span:
        decision = await _route_inner(question, client, loader, settings)
        span.update(
            output={
                "intent": decision.intent,
                "doc_types": list(decision.doc_types) or None,
                "source": decision.source,
            }
        )
    return decision


async def _route_inner(
    question: str,
    client: LLMClient | None,
    loader: PromptLoader | None,
    settings: Settings,
) -> RouteDecision:
    """The routing itself. Split out so the span above reads as one statement."""
    intent = route_by_rule(question)
    source: Literal["rule", "model", "fallback"] = "rule"

    if intent is None:
        try:
            # `client_scope` rather than `client or LLMClient(...)`: a router that builds its
            # own provider client must also close it, or every unrouted question leaks a
            # connection pool. `search()` normally passes one in, so this is the fallback path.
            async with client_scope(client, settings) as llm:
                loader = loader or PromptLoader(settings=settings)
                prompt = loader.get("router", settings.prompt_version_router)
                system, user = prompt.render(question=question).split("<<<USER>>>", 1)
                response = await llm.structured(
                    [
                        Message(role="system", content=system.strip()),
                        Message(role="user", content=user.strip()),
                    ],
                    _RouterVerdict,
                    role="fast",
                    max_output_tokens=MAX_ROUTER_TOKENS,
                    temperature=prompt.meta.get("temperature"),
                    stage="route",
                    prompt_version=prompt.qualified_version,
                )
                parsed = response.parsed
                intent = parsed.intent if isinstance(parsed, _RouterVerdict) else "general"
                source = "model"
        except Exception as exc:
            logger.warning("router failed, falling back to general: %s", exc)
            return RouteDecision(intent="general", source="fallback")

    decision = RouteDecision(intent=intent, doc_types=FILTERS.get(intent, ()), source=source)
    logger.debug("route: %s -> %s (%s)", intent, decision.doc_types or "no filter", source)
    return decision
