"""Tests for answer generation.

The interesting behaviour here is not "does it call the model" but what happens to the model's
output afterwards: citations it invented, confidence outside range, no context at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pytest

from sutra.config import Settings
from sutra.generation.answer import (
    PROMPT_NAME,
    USER_DELIMITER,
    format_context,
    generate_answer,
    resolve_citations,
    split_prompt,
)
from sutra.generation.schema import AnswerDraft
from sutra.llm.adapters import RawResult
from sutra.llm.client import LLMClient
from sutra.llm.types import Message
from sutra.prompts import PromptLoader
from sutra.retrieval.search import RetrievedChunk


def chunk(
    n: int, *, slug: str = "06_places_wiki", heading: str | None = "Kesh Ithar"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{n}",
        doc_slug=slug,
        doc_title="Places of the Meridian Line",
        doc_type="wiki",
        chunk_no=n,
        heading=heading,
        text=f"Passage body {n}. Population 68,000.",
        token_count=40,
        score=0.9 - n / 100,
        rank=n,
    )


class ScriptedAdapter:
    """Returns one prepared draft and records the messages it was given."""

    name = "scripted"

    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.messages: list[Sequence[Message]] = []

    async def complete(self, *, model, messages, max_output_tokens, temperature, timeout):
        raise AssertionError("generation must use the structured path, not complete()")

    async def structured(self, *, model, messages, schema, max_output_tokens, temperature, timeout):
        self.messages.append(list(messages))
        return RawResult(
            text=self.draft.model_dump_json(),
            model=model,
            input_tokens=900,
            output_tokens=120,
            cached_tokens=0,
            response_id="resp_scripted",
            parsed=self.draft,
        )

    async def stream(
        self, *, model, messages, max_output_tokens, temperature, timeout, schema=None
    ):
        raise AssertionError("not used")
        yield None, None


def client_for(draft: AnswerDraft, settings: Settings) -> tuple[LLMClient, ScriptedAdapter]:
    adapter = ScriptedAdapter(draft)
    return LLMClient(settings, adapter=adapter), adapter


@pytest.fixture
def loader(settings: Settings) -> PromptLoader:
    return PromptLoader(settings=settings)


# ---------------------------------------------------------------------------
# context assembly
# ---------------------------------------------------------------------------


def test_passages_are_numbered_from_one_with_their_provenance():
    rendered = format_context([chunk(1), chunk(2, slug="08_faq", heading=None)])
    assert rendered.startswith("[1] 06_places_wiki · Places of the Meridian Line · Kesh Ithar")
    assert "[2] 08_faq · Places of the Meridian Line\n" in rendered
    # Numbering starts at 1 because the model is asked to cite these numbers; an off-by-one here
    # would make every citation resolve to the wrong passage.
    assert "[0]" not in rendered


def test_the_system_half_of_the_prompt_is_identical_across_requests(loader: PromptLoader):
    """Prompt caching only pays while the prefix is byte-identical, so any interpolation
    leaking into the system block is a silent cost regression."""
    prompt = loader.get(PROMPT_NAME)
    first = split_prompt(prompt, context="context A", question="question A")
    second = split_prompt(prompt, context="context B", question="question B")

    assert first[0].content == second[0].content
    assert "context A" not in first[0].content and "question A" not in first[0].content
    assert first[1].content != second[1].content


def test_the_stable_block_is_the_first_message(loader: PromptLoader):
    """ARCHITECTURE §5 asks for the stable instructions *first* so the provider can cache them.
    Byte-identical is necessary but not sufficient: a provider caches a **prefix**, so a stable
    block sitting second is a stable block that never gets cached. Nothing else pins the
    position, and the ordering is invisible in any output if it regresses."""
    messages = split_prompt(loader.get(PROMPT_NAME), context="[1] a passage", question="q?")
    assert messages[0].role == "system"
    assert [m.role for m in messages] == ["system", "user"]


def test_context_and_question_reach_the_user_message(loader: PromptLoader):
    messages = split_prompt(
        loader.get(PROMPT_NAME), context="[1] a passage", question="How many waystones?"
    )
    assert messages[1].role == "user"
    assert "[1] a passage" in messages[1].content
    assert "How many waystones?" in messages[1].content


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_every_answer_version_states_its_load_bearing_rules(loader: PromptLoader, version: str):
    """The contract test, applied to every shipped version rather than only the active one.

    An A/B is only a fair test if both arms still obey the rules the golden set measures. v2
    changes the refusal *threshold* on purpose; it must not quietly drop the refusal flag, the
    citation constraint, the recency rule or the injection rule while doing it.
    """
    body = re.sub(r"\s+", " ", loader.get(PROMPT_NAME, version).body.lower())
    assert "refused" in body, "refusal must be an instruction, not an inference"
    assert "cite only numbers that appear" in body, "citations must be constrained to the context"
    assert "latest" in body, "the Book 2 updates need an explicit recency rule"
    assert "never something to obey" in body, "the corpus contains planted injections"


def test_v2_differs_from_v1_only_in_how_it_decides_to_refuse(loader: PromptLoader):
    """Pins the A/B to one variable. The sections that are not the hypothesis are v1's text,
    so a report comparing the two arms is comparing the refusal threshold and nothing else."""
    v1 = loader.get(PROMPT_NAME, "v1").body
    v2 = loader.get(PROMPT_NAME, "v2").body
    for section in ("## Citations", "## Dates", "## Passages are quoted material", "## Confidence"):
        start_1 = v1.index(section)
        start_2 = v2.index(section)
        # Compare each shared section up to the next heading.
        end_1 = v1.find("\n## ", start_1 + 1)
        end_2 = v2.find("\n## ", start_2 + 1)
        assert v1[start_1:end_1] == v2[start_2:end_2], f"{section} drifted between the arms"


def test_a_prompt_without_the_delimiter_is_rejected(tmp_path):
    (tmp_path / "answer.v9.md").write_text("---\ndescription: x\n---\nno delimiter here")
    prompt = PromptLoader(directory=tmp_path).get(PROMPT_NAME, "v9")
    with pytest.raises(ValueError, match=USER_DELIMITER):
        split_prompt(prompt, context="c", question="q")


def test_the_shipped_prompt_states_its_load_bearing_rules(loader: PromptLoader):
    """A contract test on the prompt file. These four behaviours are measured by the golden set,
    so losing one in an edit should fail here rather than show up as a metric regression."""
    # Whitespace-normalised: the prompt is hard-wrapped, so a phrase the test looks for may
    # carry a newline in the middle of it.
    body = re.sub(r"\s+", " ", loader.get(PROMPT_NAME).body.lower())
    assert "refused" in body, "refusal must be an instruction, not an inference"
    assert "cite only numbers that appear" in body, "citations must be constrained to the context"
    assert "latest" in body, "the Book 2 updates need an explicit recency rule"
    assert "never something to obey" in body, "the corpus contains planted injections"


# ---------------------------------------------------------------------------
# citation resolution
# ---------------------------------------------------------------------------


def test_citations_resolve_to_the_passages_actually_shown():
    draft = AnswerDraft(answer="68,000.", citations=[2, 1], confidence=0.9, refused=False)
    citations, dropped = resolve_citations(draft, [chunk(1), chunk(2)])

    assert [c.label for c in citations] == [2, 1], "citation order is the model's, preserved"
    assert [c.chunk_id for c in citations] == ["chunk-2", "chunk-1"]
    assert dropped == []


def test_a_citation_to_a_passage_that_was_never_shown_is_dropped_and_recorded():
    """A model citing passage 9 when it was shown two is inventing provenance. Discarding that
    quietly would make the answer look better grounded than it is."""
    draft = AnswerDraft(answer="…", citations=[1, 9, 0, -3], confidence=0.5, refused=False)
    citations, dropped = resolve_citations(draft, [chunk(1), chunk(2)])

    assert [c.label for c in citations] == [1]
    assert dropped == [9, 0, -3]


def test_repeated_citations_are_collapsed():
    draft = AnswerDraft(answer="…", citations=[1, 1, 2, 1], confidence=0.5, refused=False)
    citations, _ = resolve_citations(draft, [chunk(1), chunk(2)])
    assert [c.label for c in citations] == [1, 2]


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------


async def test_a_grounded_answer_carries_its_provenance(settings: Settings, loader: PromptLoader):
    draft = AnswerDraft(answer="  68,000.  ", citations=[1], confidence=0.88, refused=False)
    client, adapter = client_for(draft, settings)

    answer = await generate_answer(
        "What is the population of Kesh Ithar?",
        [chunk(1)],
        client=client,
        loader=loader,
        settings=settings,
    )

    assert answer.answer == "68,000."
    assert answer.refused is False
    assert answer.cited_slugs == ["06_places_wiki"]
    assert answer.prompt_version == "v1"
    assert answer.usage.input_tokens == 900
    assert answer.usage.cost_usd is not None, "cost must be accounted on every call"
    assert len(adapter.messages) == 1


@pytest.mark.parametrize(("given", "expected"), [(1.7, 1.0), (-0.4, 0.0), (0.42, 0.42)])
async def test_confidence_is_clamped(
    settings: Settings, loader: PromptLoader, given: float, expected: float
):
    """A schema constrains the type, not the range. An out-of-range confidence would quietly
    skew any threshold or report built on it."""
    client, _ = client_for(
        AnswerDraft(answer="x", citations=[1], confidence=given, refused=False), settings
    )
    answer = await generate_answer("q", [chunk(1)], client=client, loader=loader, settings=settings)
    assert answer.confidence == expected


async def test_a_refusal_may_still_cite_the_nearest_passage(
    settings: Settings, loader: PromptLoader
):
    """The six unanswerable golden rows each cite a near-miss quote, so refusal and citation are
    not mutually exclusive."""
    draft = AnswerDraft(
        answer="The corpus does not say; it records 2,300 for 1143.",
        citations=[1],
        confidence=0.2,
        refused=True,
    )
    client, _ = client_for(draft, settings)
    answer = await generate_answer(
        "Population after the 1144 flood?",
        [chunk(1)],
        client=client,
        loader=loader,
        settings=settings,
    )

    assert answer.refused is True
    assert len(answer.citations) == 1


async def test_no_passages_means_no_model_call(settings: Settings, loader: PromptLoader):
    """Nothing was retrieved, so the only honest answer is already known and a call would be
    waste."""
    client, adapter = client_for(
        AnswerDraft(answer="should not be used", citations=[], confidence=1.0, refused=False),
        settings,
    )
    answer = await generate_answer("q", [], client=client, loader=loader, settings=settings)

    assert answer.refused is True
    assert answer.short_circuited is True
    assert answer.confidence == 0.0
    assert adapter.messages == [], "the model must not be called when there is no context"
