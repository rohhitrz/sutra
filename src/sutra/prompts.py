"""Versioned prompt loading (ADR-006).

Prompts live in `prompts/<name>.v<N>.md` as files, never inline in Python. Two reasons:
a prompt change is then a reviewable diff, and the exact version used can be recorded on every
message and trace — which is what makes prompt A/B testing through evals possible at all.

File format: optional YAML front-matter between `---` fences, then the template body.

    ---
    description: Answer a lore question from retrieved context.
    temperature: 0.2
    ---
    You are answering questions about $universe.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from string import Template
from typing import Any

import yaml
from pydantic import BaseModel, Field

from sutra.config import Settings, get_settings

logger = logging.getLogger(__name__)

# `<name>.v<N>.md` — the version is part of the filename so `ls prompts/` shows the history.
_FILENAME = re.compile(r"^(?P<name>[a-z0-9_\-]+)\.v(?P<version>\d+)\.md$")
_FRONT_MATTER = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)$", re.DOTALL)


class PromptNotFoundError(FileNotFoundError):
    """No prompt file matched the requested name and version."""


class Prompt(BaseModel):
    """One loaded prompt file."""

    name: str
    # Always the *resolved* version, never None — this is the value recorded on messages and
    # traces, so it has to reflect what actually ran.
    version: str
    body: str
    meta: dict[str, Any] = Field(default_factory=dict)
    path: Path

    model_config = {"arbitrary_types_allowed": True}

    @property
    def qualified_version(self) -> str:
        """`answer.v1` — name and version together.

        What traces record. `version` alone is what goes on a stored message, where the
        column already sits next to the thing it versions; on a span it would be one `v1`
        among seven sibling generations, saying nothing about which prompt ran.
        """
        return f"{self.name}.{self.version}"

    def render(self, /, **variables: object) -> str:
        """Substitute `$variable` placeholders.

        # CONCEPT: `$`-substitution rather than str.format. Prompts are full of literal braces —
        # JSON examples, few-shot output schemas, code — and `.format()` would choke on every
        # one of them. `Template.substitute` also raises on a missing variable instead of
        # silently emitting a half-built prompt.
        """
        return Template(self.body).substitute(**variables)


class PromptLoader:
    """Finds, parses and caches prompt files."""

    def __init__(
        self, directory: Path | str | None = None, settings: Settings | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._dir = Path(directory) if directory is not None else Path("prompts")
        self._cache: dict[tuple[str, str], Prompt] = {}

    @property
    def directory(self) -> Path:
        return self._dir

    def list_versions(self, name: str) -> list[str]:
        """Every version of `name` on disk, ascending by number."""
        found: list[tuple[int, str]] = []
        if not self._dir.is_dir():
            return []
        for path in self._dir.iterdir():
            match = _FILENAME.match(path.name)
            if match and match.group("name") == name:
                found.append((int(match.group("version")), f"v{match.group('version')}"))
        return [version for _, version in sorted(found)]

    def active_version(self, name: str) -> str:
        """Which version to use when the caller does not pin one.

        Config wins (`PROMPT_VERSION_<NAME>` in .env). Failing that, the highest version on
        disk, logged — so a missing config entry is visible rather than silent.
        """
        configured = getattr(self._settings, f"prompt_version_{name}", None)
        if isinstance(configured, str) and configured:
            return configured
        versions = self.list_versions(name)
        if not versions:
            msg = f"no prompt files found for {name!r} in {self._dir}"
            raise PromptNotFoundError(msg)
        logger.info(
            "prompts.unconfigured_version name=%s resolved=%s — set PROMPT_VERSION_%s to pin it",
            name,
            versions[-1],
            name.upper(),
        )
        return versions[-1]

    def get(self, name: str, version: str | None = None) -> Prompt:
        """Load `name` at `version`, or at the active version when none is given."""
        resolved = version or self.active_version(name)
        key = (name, resolved)
        if key in self._cache:
            return self._cache[key]

        path = self._dir / f"{name}.{resolved}.md"
        if not path.is_file():
            available = self.list_versions(name)
            msg = (
                f"prompt {name!r} version {resolved!r} not found at {path}. "
                f"Available: {available or 'none'}"
            )
            raise PromptNotFoundError(msg)

        prompt = self._parse(name=name, version=resolved, path=path, raw=path.read_text("utf-8"))
        self._cache[key] = prompt
        return prompt

    @staticmethod
    def _parse(*, name: str, version: str, path: Path, raw: str) -> Prompt:
        meta: dict[str, Any] = {}
        body = raw
        match = _FRONT_MATTER.match(raw)
        if match:
            parsed = yaml.safe_load(match.group("yaml")) or {}
            if not isinstance(parsed, dict):
                msg = f"front-matter in {path} must be a mapping, got {type(parsed).__name__}"
                raise ValueError(msg)
            meta = parsed
            body = match.group("body")
        return Prompt(name=name, version=version, body=body.strip(), meta=meta, path=path)

    def clear_cache(self) -> None:
        self._cache.clear()
