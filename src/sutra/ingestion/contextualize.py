"""Contextual retrieval — a generated prefix per chunk (Phase 3 step 2).

# CONCEPT: A chunk is embedded alone but was written to be read in place. "He paid the toll at
# the nineteenth" is unambiguous in its chapter and nearly unsearchable out of it: no name, no
# year, no road. The embedding of that sentence sits nowhere near the embedding of "what did
# Oren Dask pay in Emberwane 1143?", and no amount of retriever tuning fixes a vector that
# never carried the subject.
#
# Contextual retrieval repairs the chunk before embedding it: a fast model reads the whole
# document, writes one or two sentences saying what this passage is and where it sits, and that
# prefix is prepended for the embedding and the tsvector. The *stored* chunk text is unchanged,
# so citations and the eval's quote matching still work against verbatim source.
#
# Two properties make this affordable, and both matter more than they look:
#
#   - **Caching by content hash.** The prefix depends only on the chunk text and its document,
#     so re-ingesting an unchanged corpus must cost nothing. Without this, every `make ingest`
#     is another hundred model calls and the idempotence ADR-021 promises becomes a lie about
#     money.
#   - **Document-ordered calls.** Every child of one document shares the same multi-thousand
#     token document prefix. Issued together, the provider's prefix cache serves almost all of
#     it. This is the first place in Sutra where prompt caching actually pays — the stable
#     block is finally over the 1024-token minimum that made it inert in Phase 2.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from sutra.config import Settings
from sutra.llm.client import LLMClient
from sutra.llm.types import Message
from sutra.prompts import PromptLoader

logger = logging.getLogger(__name__)

# The prefix is a sentence or two; this is a ceiling against a runaway generation, not a target.
MAX_PREFIX_TOKENS = 1200

# How much of the document the writer sees. The corpus files are 100-200 lines, so this takes
# all of every one of them — it exists so that a larger corpus degrades by truncation rather
# than by a surprise bill.
MAX_OVERVIEW_CHARS = 24_000


class ContextPrefix(BaseModel):
    """What the model is asked to return. Structured rather than free text so a refusal or a
    preamble fails validation here instead of being silently embedded into the index."""

    prefix: str = Field(min_length=1, max_length=600)


class ChunkToContextualise(BaseModel):
    """One chunk awaiting a prefix, with the document fields the prompt needs."""

    key: str
    doc_slug: str
    doc_title: str
    heading: str | None
    text: str


def content_hash(*, doc_slug: str, text: str, prompt_version: str) -> str:
    """The cache key for one generated prefix.

    Includes the prompt version: a rewritten prompt produces different prefixes, so reusing a
    cached one would put two prompt versions in a single index and make the slot describe a
    configuration that never ran.
    """
    payload = f"{prompt_version}\x00{doc_slug}\x00{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


async def _one_prefix(
    chunk: ChunkToContextualise,
    *,
    overview: str,
    client: LLMClient,
    loader: PromptLoader,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    prompt = loader.get("context_prefix", settings.prompt_version_context_prefix)
    system, user = prompt.render(
        doc_title=chunk.doc_title,
        doc_slug=chunk.doc_slug,
        heading=chunk.heading or "(none)",
        doc_overview=overview,
        chunk_text=chunk.text,
    ).split("<<<USER>>>", 1)

    async with semaphore:
        try:
            response = await client.structured(
                [
                    Message(role="system", content=system.strip()),
                    Message(role="user", content=user.strip()),
                ],
                ContextPrefix,
                role="fast",
                max_output_tokens=MAX_PREFIX_TOKENS,
                temperature=prompt.meta.get("temperature"),
                # Named like every other stage. Without this the ingest-time prefix calls
                # arrive as spans called `llm.call` — one per chunk — which is the one place
                # in the pipeline where that is also the highest-volume generation.
                stage="context_prefix",
                prompt_version=prompt.qualified_version,
            )
        except Exception as exc:
            # One failed prefix is not a failed ingest. The chunk is indexed bare, exactly as
            # it would have been in step 1, and the count of failures is reported — degrading
            # to the previous behaviour beats refusing to build the index.
            logger.warning("context prefix failed for %s: %s", chunk.key, exc)
            return chunk.key, None

    parsed = response.parsed
    if not isinstance(parsed, ContextPrefix):
        logger.warning("context prefix for %s did not parse", chunk.key)
        return chunk.key, None
    return chunk.key, parsed.prefix.strip()


async def generate_prefixes(
    chunks: Sequence[ChunkToContextualise],
    *,
    documents: dict[str, str],
    cached: dict[str, str],
    client: LLMClient,
    loader: PromptLoader,
    settings: Settings,
) -> dict[str, str]:
    """Return `key -> prefix` for every chunk that got one.

    `cached` maps content hash to a previously generated prefix and is consulted first;
    `documents` maps doc_slug to the full document body the writer reads.

    Chunks are grouped by document and issued in document order, so the provider's prefix cache
    sees the same multi-thousand-token document block on consecutive calls.
    """
    pending = [c for c in chunks if c.key not in cached]
    resolved = {c.key: cached[c.key] for c in chunks if c.key in cached}
    if not pending:
        logger.info("context prefixes: all %d served from cache", len(resolved))
        return resolved

    logger.info(
        "context prefixes: %d cached, generating %d with %s",
        len(resolved),
        len(pending),
        settings.llm_fast,
    )
    semaphore = asyncio.Semaphore(settings.contextualize_concurrency)
    by_document: dict[str, list[ChunkToContextualise]] = {}
    for chunk in pending:
        by_document.setdefault(chunk.doc_slug, []).append(chunk)

    failures = 0
    for slug, group in by_document.items():
        overview = documents.get(slug, "")[:MAX_OVERVIEW_CHARS]
        results = await asyncio.gather(
            *(
                _one_prefix(
                    chunk,
                    overview=overview,
                    client=client,
                    loader=loader,
                    settings=settings,
                    semaphore=semaphore,
                )
                for chunk in group
            )
        )
        for key, prefix in results:
            if prefix is None:
                failures += 1
            else:
                resolved[key] = prefix

    if failures:
        logger.warning("context prefixes: %d chunk(s) will be indexed without one", failures)
    return resolved


def contextualised_text(prefix: str | None, text: str) -> str:
    """What actually gets embedded and turned into a tsvector.

    The stored `text` column keeps the verbatim slice; this joined form exists only for the
    index. Keeping them separate is what lets citations quote the source and lets the eval
    match its 72 verbatim quotes, while search still sees the contextualised version.
    """
    return f"{prefix}\n\n{text}" if prefix else text
