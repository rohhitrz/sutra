"""Prompt loading: front-matter, version resolution, caching, failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from sutra.config import Settings
from sutra.prompts import PromptLoader, PromptNotFoundError

ANSWER_V1 = """---
description: Answer a lore question from retrieved context.
temperature: 0.2
---
You are answering questions about $universe.
Cite your sources.
"""

ANSWER_V2 = """---
description: Second iteration.
---
Answer questions about $universe, and refuse when the context is insufficient.
"""

NO_FRONT_MATTER = "Just a body with a $variable.\n"


@pytest.fixture
def prompt_dir(tmp_path: Path) -> Path:
    (tmp_path / "answer.v1.md").write_text(ANSWER_V1)
    (tmp_path / "answer.v2.md").write_text(ANSWER_V2)
    (tmp_path / "bare.v1.md").write_text(NO_FRONT_MATTER)
    # Must be ignored: not the <name>.v<N>.md pattern.
    (tmp_path / "notes.md").write_text("scratch")
    return tmp_path


def _loader(prompt_dir: Path, settings: Settings) -> PromptLoader:
    return PromptLoader(directory=prompt_dir, settings=settings)


def test_front_matter_is_parsed_and_separated_from_body(prompt_dir: Path, settings: Settings):
    prompt = _loader(prompt_dir, settings).get("answer", "v1")

    assert prompt.meta["description"].startswith("Answer a lore question")
    assert prompt.meta["temperature"] == 0.2
    assert prompt.body.startswith("You are answering")
    assert "---" not in prompt.body


def test_body_without_front_matter_is_loaded_whole(prompt_dir: Path, settings: Settings):
    prompt = _loader(prompt_dir, settings).get("bare", "v1")
    assert prompt.meta == {}
    assert prompt.body == "Just a body with a $variable."


def test_render_substitutes_variables(prompt_dir: Path, settings: Settings):
    prompt = _loader(prompt_dir, settings).get("answer", "v1")
    rendered = prompt.render(universe="The Meridian Chronicles")
    assert "The Meridian Chronicles" in rendered
    assert "$universe" not in rendered


def test_render_raises_on_a_missing_variable(prompt_dir: Path, settings: Settings):
    prompt = _loader(prompt_dir, settings).get("answer", "v1")
    # Failing loud beats shipping a prompt with a literal $universe in it.
    with pytest.raises(KeyError):
        prompt.render()


def test_render_leaves_literal_braces_alone(tmp_path: Path, settings: Settings):
    # The reason for $-substitution over str.format: prompts are full of JSON examples.
    (tmp_path / "schema.v1.md").write_text('Return {"answer": "...", "citations": []} for $q.')
    prompt = PromptLoader(directory=tmp_path, settings=settings).get("schema", "v1")
    rendered = prompt.render(q="who?")
    assert '{"answer": "...", "citations": []}' in rendered


def test_list_versions_is_sorted_numerically(prompt_dir: Path, settings: Settings):
    # String sorting would put v10 before v2; version ordering has to be numeric.
    (prompt_dir / "answer.v10.md").write_text("---\n---\nv10 body $universe")
    assert _loader(prompt_dir, settings).list_versions("answer") == ["v1", "v2", "v10"]


def test_configured_version_wins(prompt_dir: Path, settings: Settings):
    settings = settings.model_copy(update={"prompt_version_answer": "v1"})
    loader = _loader(prompt_dir, settings)
    assert loader.active_version("answer") == "v1"
    assert loader.get("answer").version == "v1"


def test_unconfigured_version_falls_back_to_the_highest(prompt_dir: Path, settings: Settings):
    # `bare` has no PROMPT_VERSION_BARE setting, so the loader picks the newest and logs it.
    assert _loader(prompt_dir, settings).active_version("bare") == "v1"


def test_resolved_version_is_recorded_on_the_prompt(prompt_dir: Path, settings: Settings):
    # ADR-006: whatever version actually ran must be recordable on messages and traces.
    settings = settings.model_copy(update={"prompt_version_answer": "v2"})
    assert _loader(prompt_dir, settings).get("answer").version == "v2"


def test_missing_version_raises_and_names_the_alternatives(prompt_dir: Path, settings: Settings):
    with pytest.raises(PromptNotFoundError, match=r"v1.*v2|Available"):
        _loader(prompt_dir, settings).get("answer", "v99")


def test_unknown_prompt_name_raises(prompt_dir: Path, settings: Settings):
    with pytest.raises(PromptNotFoundError):
        _loader(prompt_dir, settings).get("nonexistent")


def test_files_not_matching_the_naming_pattern_are_ignored(prompt_dir: Path, settings: Settings):
    assert _loader(prompt_dir, settings).list_versions("notes") == []


def test_prompts_are_cached(prompt_dir: Path, settings: Settings):
    loader = _loader(prompt_dir, settings)
    first = loader.get("answer", "v1")
    assert loader.get("answer", "v1") is first
    loader.clear_cache()
    assert loader.get("answer", "v1") is not first


def test_non_mapping_front_matter_is_rejected(tmp_path: Path, settings: Settings):
    (tmp_path / "bad.v1.md").write_text("---\n- just\n- a list\n---\nbody")
    with pytest.raises(ValueError, match="must be a mapping"):
        PromptLoader(directory=tmp_path, settings=settings).get("bad", "v1")
