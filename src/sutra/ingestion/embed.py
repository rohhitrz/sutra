"""Local embeddings via fastembed / ONNX Runtime (ADR-004, ADR-020).

# CONCEPT: The embedder sits behind a Protocol so the model is a swappable detail. Retrieval
# quality depends heavily on the embedder, so it is the component most likely to be replaced —
# and an interface is the difference between "change one class" and "change every call site".
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from sutra.config import EMBEDDING_MAX_SEQUENCE_TOKENS, Settings, get_settings

logger = logging.getLogger(__name__)

# Re-exported so retrieval and ingestion can reason about the ceiling without importing config
# constants directly. `Settings` enforces that chunk_tokens + chunk_overlap stays under it.
MAX_SEQUENCE_TOKENS = EMBEDDING_MAX_SEQUENCE_TOKENS


@runtime_checkable
class Embedder(Protocol):
    """What the rest of Sutra needs from an embedding model."""

    @property
    def dims(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """`BAAI/bge-small-en-v1.5` over ONNX Runtime.

    Loading is lazy: constructing this class is free, and the ~65 MB model is pulled in on first
    use. Importing a module should never cost a model load.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("loading embedding model %s", self._settings.embedding_model)
            self._model = TextEmbedding(
                model_name=self._settings.embedding_model,
                cache_dir=self._settings.model_cache_dir,
            )
        return self._model

    @property
    def dims(self) -> int:
        return self._settings.embedding_dims

    def count_tokens(self, text: str) -> int:
        """Token count from the model's *own* tokenizer.

        Chunk sizing must use this and not a word-count heuristic: the 512-token ceiling is
        defined by this tokenizer, so any other counter is guessing at the one number that
        decides whether a chunk gets silently truncated.
        """
        model = self._ensure_model()
        tokenizer = model.model.tokenizer  # type: ignore[attr-defined]
        # The tokenizer truncates at 512 by default, which would make every long string report
        # exactly 512 and hide oversized chunks. Encoding without truncation gives the true length.
        return len(tokenizer.encode(text, add_special_tokens=True).ids)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus text. Batched by fastembed internally."""
        if not texts:
            return []
        model = self._ensure_model()
        vectors: Iterable[object] = model.embed(list(texts))  # type: ignore[attr-defined]
        return [[float(x) for x in vector] for vector in vectors]  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        # CONCEPT: Asymmetric embedding. Some retrieval models are trained with an instruction
        # prefix on the *query* side only, so a query embedded as though it were a passage lands
        # slightly off-manifold and costs recall with nothing to show why. Queries and passages
        # therefore go through separate methods, and this is the seam where that difference
        # belongs.
        #
        # Measured, not assumed: for `bge-small-en-v1.5` under fastembed, `query_embed` is
        # currently identical to `embed` — `cos(query_embed(x), embed(x)) == 1.000000`, so no
        # prefix is applied. Prepending bge's documented prefix by hand was tried against the 42
        # golden rows and moved almost nothing: Recall@5 74% -> 76%, MRR 0.519 -> 0.522, helping
        # 12 rows and hurting 7. Too close to justify diverging from the library's default, so
        # Phase 2's baseline sends the query through unchanged. Revisit with the eval harness
        # rather than on intuition.
        """
        model = self._ensure_model()
        vectors: Iterable[object] = model.query_embed([text])  # type: ignore[attr-defined]
        return [float(x) for x in next(iter(vectors))]  # type: ignore[call-overload]


def get_embedder(settings: Settings | None = None) -> Embedder:
    return FastEmbedEmbedder(settings)
