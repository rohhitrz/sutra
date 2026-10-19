"""Read a corpus file into a typed document.

# CONCEPT: Ingestion starts with parsing, not chunking. Metadata extracted here (doc_type, book,
# characters) becomes the filter surface for metadata-filtered retrieval later, and the content
# hash is what makes re-ingestion idempotent. Get this wrong and every later stage inherits it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# `---\n ... \n---\n` at the very top of the file, then the body.
_FRONT_MATTER = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)$", re.DOTALL)

# `data/corpus/README.md` is the freeze notice (ADR-014), not a corpus document. The dataset
# validator skips it the same way; the two must agree or the quote checks drift.
EXCLUDED_STEMS = frozenset({"README"})


class ParsedDocument(BaseModel):
    """One corpus file, parsed but not yet chunked."""

    # The filename stem. This is the identifier the golden datasets cite as `doc_slug`, so it is
    # ground truth shared with `evals/`, not an internal detail we are free to reshape.
    slug: str
    title: str
    doc_type: str
    body: str
    meta: dict[str, Any] = Field(default_factory=dict)
    # sha256 of the file as it sits on disk, front-matter included. Any edit changes it, which
    # changes the corpus hash, which changes the index fingerprint (ADR-021).
    content_hash: str
    path: Path

    model_config = {"arbitrary_types_allowed": True}


class CorpusError(ValueError):
    """A corpus file is malformed. Fatal: a silently skipped document is a silent recall hole."""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_document(path: Path) -> ParsedDocument:
    """Parse one `.md` corpus file. Raises `CorpusError` rather than guessing."""
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if match is None:
        msg = f"{path.name}: no YAML front-matter found (file must open with a '---' fence)"
        raise CorpusError(msg)

    try:
        front: Any = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as exc:
        msg = f"{path.name}: front-matter is not valid YAML: {exc}"
        raise CorpusError(msg) from exc

    if not isinstance(front, dict):
        msg = f"{path.name}: front-matter must be a mapping, got {type(front).__name__}"
        raise CorpusError(msg)

    for required in ("title", "doc_type"):
        if not front.get(required):
            msg = f"{path.name}: front-matter is missing required key {required!r}"
            raise CorpusError(msg)

    # title and doc_type are promoted to real columns (§12); everything else stays in the JSONB
    # metadata, where the GIN index makes it filterable without a migration per new key.
    meta = {k: v for k, v in front.items() if k not in {"title", "doc_type"}}

    return ParsedDocument(
        slug=path.stem,
        title=str(front["title"]),
        doc_type=str(front["doc_type"]),
        body=match.group("body"),
        meta=meta,
        content_hash=content_hash(raw),
        path=path,
    )


def parse_corpus(directory: Path) -> list[ParsedDocument]:
    """Parse every corpus document, in a stable order.

    Sorted by slug so the corpus hash is deterministic — filesystem order is not.
    """
    paths = sorted(p for p in directory.glob("*.md") if p.stem not in EXCLUDED_STEMS)
    if not paths:
        msg = f"no corpus documents found in {directory}"
        raise CorpusError(msg)
    return [parse_document(p) for p in paths]


def corpus_hash(documents: list[ParsedDocument]) -> str:
    """One hash over the whole corpus, used in the index fingerprint (ADR-021).

    Built from (slug, content_hash) pairs rather than file bytes so it is insensitive to
    read order but sensitive to any edit, addition or removal.
    """
    joined = "\n".join(
        f"{d.slug}:{d.content_hash}" for d in sorted(documents, key=lambda d: d.slug)
    )
    return content_hash(joined)
