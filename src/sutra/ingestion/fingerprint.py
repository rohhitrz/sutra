"""The index fingerprint (ADR-021).

# CONCEPT: An index is only comparable to itself. An eval baseline is a claim about a specific
# corpus chunked a specific way and embedded by a specific model; change any of those and the
# numbers stop meaning what the report says they mean. The fingerprint makes that identity
# explicit, so re-chunking into a slot an eval baseline is pinned to fails loudly instead of
# quietly invalidating every number recorded against it.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

from sutra.config import Settings
from sutra.ingestion.chunk import CHUNKER_VERSION, SEMANTIC_CHUNKER_VERSION
from sutra.ingestion.parse import ParsedDocument, corpus_hash


class IndexFingerprint(BaseModel):
    """Everything that, if changed, makes an index incomparable to its predecessor."""

    chunk_strategy: str
    chunk_tokens: int
    chunk_overlap: int
    # The chunker's own version (ADR-022). Parameters alone cannot tell you whether the code that
    # applied them has changed, and a chunker change is exactly as index-invalidating as a
    # parameter change.
    chunker_version: int
    embedding_model: str
    corpus_hash: str

    # Parent-child / semantic parameters (Phase 3 step 1). All default to None, and a flat
    # recursive index leaves them None.
    #
    # Defaulting rather than requiring them is what keeps **slot 1 idempotent**. The stored
    # fingerprint is a dict, and `digest` is recomputed from the parsed model on both sides of
    # the comparison — so a slot written before these fields existed parses with them as None
    # and produces exactly the digest a flat ingest computes today. Making them required would
    # have turned every pre-existing slot into an "older schema" refusal, and the committed
    # baseline's index would have had to be rebuilt to say nothing new.
    parent_child: bool | None = None
    parent_tokens: int | None = None
    child_tokens: int | None = None
    child_overlap: int | None = None
    semantic_breakpoint_percentile: int | None = None
    # Versioned separately from `chunker_version` so a change to the semantic path cannot
    # invalidate a flat index whose output has not moved.
    semantic_chunker_version: int | None = None
    # Contextual retrieval (step 2). The prompt *version*, not a boolean: a rewritten prompt
    # produces different prefixes, so it produces a different index — and two slots built with
    # different prompt versions must not compare equal.
    context_prefix_prompt: str | None = None

    @property
    def digest(self) -> str:
        """Short, stable hash. `sort_keys` matters: field order must not change the digest."""
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def differences_from(self, other: IndexFingerprint) -> list[str]:
        """Field-by-field diff, so a mismatch error can say *what* changed.

        "fingerprint mismatch" sends you reading code; "chunk_tokens: 400 -> 250" tells you
        exactly which knob you turned.
        """
        mine, theirs = self.model_dump(), other.model_dump()
        return [f"{k}: {theirs[k]!r} -> {mine[k]!r}" for k in mine if mine[k] != theirs[k]]


def build_fingerprint(settings: Settings, documents: list[ParsedDocument]) -> IndexFingerprint:
    parent_child = settings.parent_child_enabled
    return IndexFingerprint(
        chunk_strategy=settings.chunk_strategy,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
        chunker_version=CHUNKER_VERSION,
        embedding_model=settings.embedding_model,
        corpus_hash=corpus_hash(documents),
        # Left None when flat, so a flat fingerprint is byte-identical to one computed before
        # these fields existed. Recording `parent_tokens` on an index that has no parents would
        # make two identical indexes look different.
        parent_child=parent_child or None,
        parent_tokens=settings.parent_tokens if parent_child else None,
        child_tokens=settings.child_tokens if parent_child else None,
        child_overlap=settings.child_overlap if parent_child else None,
        semantic_breakpoint_percentile=(
            settings.semantic_breakpoint_percentile
            if parent_child and settings.chunk_strategy == "semantic"
            else None
        ),
        semantic_chunker_version=SEMANTIC_CHUNKER_VERSION if parent_child else None,
        context_prefix_prompt=(
            settings.prompt_version_context_prefix if settings.contextual_prefixes_enabled else None
        ),
    )
