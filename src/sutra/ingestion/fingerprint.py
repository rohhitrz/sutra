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
from sutra.ingestion.chunk import CHUNKER_VERSION
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
    return IndexFingerprint(
        chunk_strategy=settings.chunk_strategy,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
        chunker_version=CHUNKER_VERSION,
        embedding_model=settings.embedding_model,
        corpus_hash=corpus_hash(documents),
    )
