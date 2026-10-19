"""Recursive character chunking, hand-rolled (ADR-003).

# CONCEPT: Recursive chunking splits on a *hierarchy* of separators, trying the most semantic
# boundary first and only falling back to cruder ones when a piece is still too large. Splitting
# blindly every N characters is cheaper and much worse: it cuts sentences, tables and ledger rows
# in half, and a fact cut in half is a fact no retriever can return whole.
#
# Three invariants this module guarantees, all load-bearing elsewhere:
#   1. Every chunk's `text` is a verbatim contiguous slice of the document body. Nothing is
#      rewritten or tidied. Phase 2's eval ground truth is 72 verbatim quotes matched against
#      chunk text, so a chunker that normalised whitespace would destroy the measurement.
#   2. No content is lost: in order, the chunks reconstruct the body up to whitespace at their
#      edges (each chunk is stripped, so the separators *between* chunks are not preserved).
#   3. Every chunk is <= `max_tokens + overlap_tokens`, since overlap is prepended after the
#      budget check. `Settings` enforces that this sum stays under the embedder's 512-token
#      ceiling, because truncation there is silent rather than an error.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from pydantic import BaseModel

# # CONCEPT: The index fingerprint has to cover the *code* that built the index, not only the
# # parameters passed to it (ADR-022). Without this, fixing a chunker bug changes every chunk
# # while leaving the fingerprint identical — so `make ingest` reports "nothing to do" and the
# # stale index stays in place, still carrying an eval baseline that no longer describes it.
#
# Bump this whenever a change can alter chunker output. Not for comments, renames or type hints.
#   1 — Phase 2: recursive split on h2/h3/paragraph/line/sentence/word, merge-up, tail overlap,
#       h2+h3 heading capture.
CHUNKER_VERSION = 1

TokenCounter = Callable[[str], int]

# An h2 or h3 at the start of a line. Both levels count because the corpus is not uniform: most
# documents section on h2 (one place, one person, one year), but the traveller notes use h3 for
# each box and the FAQ has no sections below its title at all. Matching only h2 would leave the
# documents that most need a label without one. h1 is excluded — it is the document title, which
# is already a column on `documents`.
_HEADING = re.compile(r"^#{2,3} +(.+?)[ \t]*$", re.MULTILINE)

# Tried in order, most semantic first. Tuned to this corpus: markdown sections are the natural
# unit (one place, one person, one ledger entry), and bullet blocks must stay whole because a
# single `- **Population:** 68,000` line is itself a cited fact.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n## ",  # section — a place, a character, a year
    "\n### ",  # subsection
    "\n\n",  # paragraph or bullet block
    "\n",  # line — individual bullets
    ". ",  # sentence
    " ",  # word
)


class Chunk(BaseModel):
    """One retrievable passage, with its position in the source document."""

    chunk_no: int
    text: str
    token_count: int
    # The `## ` heading this chunk sits under, if any. Kept in metadata rather than prepended to
    # `text`, because `text` must stay a verbatim slice of the document (invariant 1). Phase 3's
    # contextual-prefix experiment is what measures whether prepending it actually helps.
    heading: str | None = None
    start_char: int
    end_char: int


def estimate_tokens(text: str) -> int:
    """Crude fallback counter, for tests and callers with no model loaded.

    Takes the larger of ~1.3 tokens per word and ~1 token per 4 characters. The second term is
    not redundant: on word-like input the first dominates, but a long unbroken run of characters
    is one "word" and would otherwise be counted as a single token — so a 5,000-character blob
    would look like it fit in any budget and sail through to the embedder to be truncated.

    Real ingestion passes the embedder's own tokenizer. The 512-token ceiling is defined by that
    tokenizer and by nothing else, so any other counter is a guess at the one number that
    decides whether a chunk is silently cut in half.
    """
    by_words = len(text.split()) * 1.3
    by_chars = len(text) / 4
    return max(1, int(max(by_words, by_chars)))


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Split on `separator`, re-attaching it to the start of each following piece.

    Keeping the separator is what preserves invariant 1: `"".join(result) == text`.
    """
    parts = text.split(separator)
    if len(parts) == 1:
        return parts
    rejoined = [parts[0], *(separator + part for part in parts[1:])]
    return [p for p in rejoined if p]


def _merge_adjacent(pieces: Sequence[str], max_tokens: int, count: TokenCounter) -> list[str]:
    """Greedily glue neighbouring pieces back together while they still fit.

    Without this, splitting on `\\n` would emit one chunk per bullet — forty useless two-line
    chunks instead of one coherent section. Recursive chunking is as much about re-merging as
    about splitting.
    """
    merged: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if current and count(candidate) > max_tokens:
            merged.append(current)
            current = piece
        else:
            current = candidate
    if current:
        merged.append(current)
    return merged


def _hard_split(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Last resort for a single token-dense run with no separator left to try.

    Splits on characters, proportionally. Reached only by pathological input (an enormous
    unbroken word); the corpus never triggers it, but an oversized chunk would be invisible
    corruption, so the path exists rather than letting one through.
    """
    total = count(text)
    if total <= max_tokens:
        return [text]
    pieces = -(-total // max_tokens)  # ceiling division
    width = -(-len(text) // pieces)
    return [text[i : i + width] for i in range(0, len(text), width)]


def _recurse(
    text: str, separators: Sequence[str], max_tokens: int, count: TokenCounter
) -> list[str]:
    if count(text) <= max_tokens:
        return [text]
    for position, separator in enumerate(separators):
        pieces = _split_keeping_separator(text, separator)
        if len(pieces) == 1:
            continue
        # Each piece may still be too big, so recurse with the *remaining* (cruder) separators,
        # then merge the results back up to the budget.
        fitted: list[str] = []
        for piece in pieces:
            fitted.extend(_recurse(piece, separators[position + 1 :], max_tokens, count))
        return _merge_adjacent(fitted, max_tokens, count)
    return _hard_split(text, max_tokens, count)


def _heading_at(body: str, start: int, end: int) -> str | None:
    """The `## ` section heading that best describes the slice `[start, end)`.

    Two cases, because a document's first chunk has no heading *before* it:
      - the chunk continues a section → the last heading at or before `start`
      - the chunk opens the document (an `# h1` title, then the first section) → the first
        heading *inside* the slice

    Resolving only the first case returns None for the opening chunk of every document, which
    is exactly the kind of gap that looks like missing data rather than a bug.
    """
    before: str | None = None
    inside: str | None = None
    for match in _HEADING.finditer(body):
        if match.start() <= start:
            before = match.group(1).strip() or None
        elif match.start() < end and inside is None:
            inside = match.group(1).strip() or None
    return before or inside


def _overlap_prefix(previous: str, overlap_tokens: int, count: TokenCounter) -> str:
    """The trailing slice of the previous chunk, measured in tokens, cut on a word boundary."""
    if overlap_tokens <= 0 or not previous:
        return ""
    words = previous.split(" ")
    taken: list[str] = []
    for word in reversed(words):
        taken.insert(0, word)
        if count(" ".join(taken)) >= overlap_tokens:
            break
    return " ".join(taken)


def chunk_document(
    body: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
    count_tokens: TokenCounter | None = None,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Chunk one document body.

    Overlap is added *after* splitting, by prepending the tail of the previous chunk. Because
    chunks are contiguous and ordered, the overlapped text is still an exact slice of the
    document — which keeps invariant 1 true and gives a fact near a boundary a second chance to
    appear somewhere whole.
    """
    count = count_tokens or estimate_tokens
    stripped = body.strip("\n")
    if not stripped.strip():
        return []

    base_offset = body.index(stripped) if stripped else 0
    pieces = _recurse(stripped, separators, max_tokens, count)

    chunks: list[Chunk] = []
    cursor = 0
    for number, piece in enumerate(pieces):
        start = cursor
        cursor += len(piece)
        prefix = _overlap_prefix(pieces[number - 1], overlap_tokens, count) if number else ""
        text = (prefix + piece).strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_no=len(chunks),
                text=text,
                token_count=count(text),
                heading=_heading_at(stripped, start, cursor),
                start_char=base_offset + start - len(prefix),
                end_char=base_offset + cursor,
            )
        )
    return chunks
