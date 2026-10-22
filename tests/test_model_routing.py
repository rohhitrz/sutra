"""Which model role each stage asks for, and whether the fallback chain is real.

# CONCEPT: Model routing is the cheapest quality/cost dial in a RAG system — a rewrite, a
# classification and a summary do not need the model that writes the answer. ARCHITECTURE §5
# sets the split: a `fast` model for routing, rewriting and summaries, a `strong` model for
# answers.

ADR-017 points both roles at one model today, so none of this changes a single byte of output.
That is exactly why it needs a test. A seam that costs nothing to get wrong is a seam that
*will* be wrong by the time someone sets `LLM_FAST` to something cheap — and the symptom then
is not an error, it is an answer written by the wrong model. These tests assert the *intent*
recorded on each generation, which is the only thing observable while both roles agree.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import contextmanager

import pytest
from evals.metrics.judges import run_judge
from pydantic import BaseModel

from sutra.config import Settings
from sutra.generation.answer import generate_answer
from sutra.generation.stream import generate_answer_stream
from sutra.llm.adapters import RawResult
from sutra.llm.client import LLMClient
from sutra.llm.tracing import NoOpTracer
from sutra.llm.types import GenerationRecord, Message
from sutra.prompts import PromptLoader
from sutra.retrieval.query_transform import transform_query
from sutra.retrieval.router import route
from sutra.retrieval.search import RetrievedChunk


def _placeholder(schema: type[BaseModel]) -> BaseModel:
    """A minimal valid instance of `schema`, whatever its fields are.

    The subject here is which role a stage asks for, not what the model says, so the payload is
    generated from the schema rather than hand-written per stage. That also means a new stage
    can be added to these tests without a new fake.
    """
    values: dict[str, object] = {}
    for name, field in schema.model_fields.items():
        annotation = field.annotation
        if field.default is not None and not field.is_required():
            continue
        if annotation is str:
            values[name] = "placeholder"
        elif annotation is float:
            values[name] = 0.5
        elif annotation is bool:
            values[name] = False
        elif annotation is int:
            values[name] = 1
        elif annotation == list[str]:
            values[name] = ["placeholder"]
        elif annotation == list[int]:
            values[name] = [1]
        else:
            # A Literal or anything else: take the first value the schema will accept.
            args = getattr(annotation, "__args__", ())
            values[name] = args[0] if args else "general"
    return schema.model_validate(values)


class RoleRecordingAdapter:
    """Answers any schema, and leaves the record-keeping to the tracer."""

    name = "role-recording"

    async def complete(self, *, model, messages, max_output_tokens, temperature, timeout):
        return RawResult(text="placeholder", model=model, input_tokens=10, output_tokens=5)

    async def structured(self, *, model, messages, schema, max_output_tokens, temperature, timeout):
        return RawResult(
            text="{}",
            model=model,
            input_tokens=10,
            output_tokens=5,
            parsed=_placeholder(schema),
        )

    async def stream(
        self,
        *,
        model,
        messages: Sequence[Message],
        max_output_tokens,
        temperature,
        timeout,
        schema=None,
    ) -> AsyncIterator[tuple[str | None, RawResult | None]]:
        yield '{"answer": "placeholder", "citations": [1], ', None
        yield '"confidence": 0.5, "refused": false}', None
        yield (
            None,
            RawResult(
                text='{"answer": "placeholder", "citations": [1], "confidence": 0.5,'
                ' "refused": false}',
                model=model,
                input_tokens=10,
                output_tokens=5,
                parsed=_placeholder(schema) if schema else None,
                status="completed",
            ),
        )


class RoleTracer(NoOpTracer):
    """Keeps every record, so a test can ask what role a named stage used."""

    def __init__(self) -> None:
        self.records: list[GenerationRecord] = []

    def on_generation(self, record: GenerationRecord) -> None:
        self.records.append(record)

    @contextmanager
    def stage(self, name: str, *, kind=None, input=None, metadata=None):
        yield None

    def role_for(self, stage: str) -> str | None:
        for record in self.records:
            if record.stage == stage:
                return record.role
        raise AssertionError(f"no generation recorded for stage {stage!r}: {self.stages()}")

    def stages(self) -> list[str | None]:
        return [r.stage for r in self.records]


@pytest.fixture
def tracer() -> RoleTracer:
    return RoleTracer()


@pytest.fixture
def client(settings: Settings, tracer: RoleTracer) -> LLMClient:
    return LLMClient(settings=settings, adapter=RoleRecordingAdapter(), tracer=tracer)


def chunk(n: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{n}",
        doc_slug="06_places_wiki",
        doc_title="Places of the Meridian Line",
        doc_type="wiki",
        chunk_no=n,
        heading="Kesh Ithar",
        text="Kesh Ithar stands at the third waystone.",
        token_count=20,
        score=0.9,
        rank=n,
    )


# ---------------------------------------------------------------------------
# every generation says which stage made it
# ---------------------------------------------------------------------------


async def test_every_generation_names_its_stage(settings: Settings, client, tracer):
    """A record that names only the model cannot settle a claim about routing while both roles
    resolve to the same model — which is the entire situation under ADR-017."""
    await generate_answer(
        "Where is Kesh Ithar?",
        [chunk()],
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    )
    assert tracer.stages() == ["answer"]
    assert all(r.stage for r in tracer.records), "an anonymous generation is an unauditable one"


# ---------------------------------------------------------------------------
# §5's split, stage by stage
# ---------------------------------------------------------------------------


async def test_the_answer_runs_on_the_strong_role(settings: Settings, client, tracer):
    await generate_answer(
        "Where is Kesh Ithar?",
        [chunk()],
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    )
    assert tracer.role_for("answer") == "strong"


async def test_the_streamed_answer_runs_on_the_same_role_as_the_buffered_one(
    settings: Settings, client, tracer
):
    """ADR-025 makes streaming a transport choice. A different role would make it a different
    answer path, which is the thing that ADR exists to prevent."""
    async for _event in generate_answer_stream(
        "Where is Kesh Ithar?",
        [chunk()],
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    ):
        pass
    assert tracer.role_for("answer") == "strong"


async def test_query_rewriting_runs_on_the_fast_role(settings: Settings, client, tracer):
    settings = settings.model_copy(update={"query_rewrite_enabled": True, "hyde_enabled": False})
    await transform_query(
        "What did she do next?",
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    )
    assert tracer.role_for("rewrite") == "fast"


async def test_hyde_runs_on_the_fast_role(settings: Settings, client, tracer):
    settings = settings.model_copy(update={"query_rewrite_enabled": False, "hyde_enabled": True})
    await transform_query(
        "How many waystones?",
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    )
    assert tracer.role_for("hyde") == "fast"


async def test_routing_runs_on_the_fast_role(settings: Settings, client, tracer):
    settings = settings.model_copy(update={"router_enabled": True})
    await route(
        "Tell me about the Archivist",
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    )
    assert tracer.role_for("route") == "fast"


async def test_grading_runs_on_the_fast_role(settings: Settings, client, tracer):
    """§5 puts grading on the fast role, and the judges now say so.

    Worth stating the consequence in the test, because it is not visible today: while both
    roles resolve to one model this is free, but pointing `LLM_FAST` at a cheaper model would
    silently weaken the instrument the scoreboard is measured with. `make judges` against the
    hand-labelled rows is the check that has to be re-run if that ever happens.
    """
    await run_judge(
        "faithfulness",
        question="Where is Kesh Ithar?",
        answer="At the third waystone.",
        context="[1] Kesh Ithar stands at the third waystone.",
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    )
    assert tracer.role_for("judge_faithfulness") == "fast"


async def test_the_answer_is_the_only_stage_left_on_the_strong_role(
    settings: Settings, client, tracer
):
    """A guard on §5 as a whole rather than on one stage. If a new stage is added on `strong`
    without a reason, this is what notices — the split is the decision, and the decision is
    that answering is the one thing worth the expensive model."""
    await generate_answer(
        "Where is Kesh Ithar?",
        [chunk()],
        client=client,
        settings=settings,
        loader=PromptLoader(settings=settings),
    )
    settings_fast = settings.model_copy(update={"router_enabled": True})
    await route(
        "Tell me about the Archivist",
        client=client,
        settings=settings_fast,
        loader=PromptLoader(settings=settings),
    )
    strong_stages = {r.stage for r in tracer.records if r.role == "strong"}
    assert strong_stages == {"answer"}


# ---------------------------------------------------------------------------
# the fallback chain
# ---------------------------------------------------------------------------


def test_the_default_fallback_chain_is_empty_rather_than_self_referential():
    """`LLM_FALLBACK_MODELS=gpt-6-luna` was the default for three phases. It is the primary
    model under ADR-017, so the chain de-duplicated it away and no fallback ever existed —
    while the config read as though one did. Empty is the honest statement of the same fact."""
    assert Settings(_env_file=None).llm_fallback_models == []  # type: ignore[call-arg]


def test_a_chain_of_one_is_what_an_empty_fallback_list_produces(settings: Settings):
    settings = settings.model_copy(update={"llm_strong": "gpt-6-luna", "llm_fallback_models": []})
    client = LLMClient(settings=settings, adapter=RoleRecordingAdapter())
    assert client._chain("gpt-6-luna") == ["gpt-6-luna"]


def test_a_different_model_makes_the_chain_real(settings: Settings):
    settings = settings.model_copy(update={"llm_fallback_models": ["gpt-6-sol"]})
    client = LLMClient(settings=settings, adapter=RoleRecordingAdapter())
    assert client._chain("gpt-6-luna") == ["gpt-6-luna", "gpt-6-sol"]
