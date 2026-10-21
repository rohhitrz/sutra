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
#   3. Every chunk is <= `max_tokens + overlap_tokens`, plus at most one word. Overlap is
#      prepended after the budget check, and `_overlap_prefix` takes whole words until it
#      *reaches* the overlap budget, so it can cross it by part of a word — a 200/40 child can
#      measure 241 tokens. The slack is stated rather than engineered away because removing it
#      would change the output of every chunk written so far, including index slot 1, which the
#      committed eval baseline is pinned to. `Settings` enforces that the sum stays under the
#      embedder's 512-token ceiling with room to spare, because truncation there is silent
#      rather than an error.
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

# The semantic + parent-child path has its own version, kept separate so that slot 1 — the flat
# recursive index the committed baseline is pinned to — keeps the fingerprint it was written
# with. One shared counter would have made every semantic-chunker change invalidate an index
# whose output had not moved by a character.
#   1 — Phase 3 step 1: sentence embeddings, percentile breakpoints, recursive child split.
SEMANTIC_CHUNKER_VERSION = 1

TokenCounter = Callable[[str], int]
# Embeds a batch of texts. Passed in rather than imported so this module keeps knowing nothing
# about ONNX, fastembed or `Settings` — the same reason `count_tokens` is a parameter.
TextEmbedder = Callable[[Sequence[str]], list[list[float]]]

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


# ---------------------------------------------------------------------------
# Semantic chunking + parent-child (Phase 3 step 1)
# ---------------------------------------------------------------------------

# Sentence end: terminal punctuation, then whitespace. Kept deliberately simple — the corpus is
# plain English prose and markdown. An abbreviation ("No. 14") splits here wrongly, which costs
# a boundary in the wrong place and nothing else: the pieces are re-merged by similarity
# immediately afterwards, so a spurious split usually disappears.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ParentChunk(BaseModel):
    """A context-sized passage and the search-sized children cut from it.

    # CONCEPT: Parent-child chunking. The child is what gets embedded and searched; the parent
    # is what the model reads. Small children keep a vector about one thing, which is what makes
    # retrieval precise. Large parents keep the sentence that explains what that thing refers
    # to, which is what makes the answer correct. Retrieval-time expansion (`retrieval/expand`)
    # is the step that connects them, and without it this is just small chunks.
    """

    chunk_no: int
    text: str
    token_count: int
    heading: str | None = None
    start_char: int
    end_char: int
    children: list[Chunk] = []


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping every character.

    `"".join(result) == text` must hold: these pieces are reassembled into chunks, and chunk
    text has to stay a verbatim slice of the document (invariant 1).
    """
    if not text:
        return []
    pieces: list[str] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        pieces.append(text[cursor : match.end()])
        cursor = match.end()
    if cursor < len(text):
        pieces.append(text[cursor:])
    return pieces


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear-interpolated percentile. Written out rather than pulled from numpy: this is four
    lines, and the dependency would be carried for them alone."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def semantic_breakpoints(
    sentences: Sequence[str], embed: TextEmbedder, *, percentile: int
) -> set[int]:
    """Indices where a new chunk should start, found by similarity between neighbours.

    # CONCEPT: Embed every sentence, measure cosine similarity between each adjacent pair, and
    # cut where that similarity falls furthest. A topic shift shows up as a trough: the sentence
    # that introduces the ledger does not resemble the sentence that closed the travel notes.
    #
    # The threshold is a percentile of *this document's own* distances, never an absolute
    # cosine. Absolute thresholds do not transfer between documents — a ledger's rows all
    # resemble each other and a cutoff tuned on prose would split it at every row or not at all.
    """
    if len(sentences) < 2:
        return set()

    vectors = embed(list(sentences))
    distances = [1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = _percentile(distances, percentile)
    # `>` not `>=`: when every distance is identical (a uniform ledger), the percentile equals
    # all of them, and `>=` would break between every single sentence.
    return {i + 1 for i, distance in enumerate(distances) if distance > threshold}


def _group_by_breakpoints(
    sentences: Sequence[str], breakpoints: set[int], max_tokens: int, count: TokenCounter
) -> list[str]:
    """Join sentences into groups, cutting only at breakpoints — or at the budget.

    A breakpoint is a *permitted* place to cut, not a required one. Splitting at every one of
    them produces far more, far smaller pieces than `max_tokens` asks for: on this corpus the
    90th percentile fires roughly every four sentences, which would make a "parent" barely
    larger than a child and leave the whole parent-child split buying nothing. So groups are
    merged greedily up to the budget, and the breakpoints decide only *where* the cuts land.

    The budget still wins when a run of closely-related sentences overflows it, because an
    oversized parent is a worse failure than a cut in a slightly wrong place.
    """
    groups: list[str] = []
    current = ""
    for index, sentence in enumerate(sentences):
        if not current:
            current = sentence
            continue
        candidate = current + sentence
        at_breakpoint = index in breakpoints
        over_budget = count(candidate) > max_tokens
        # Cut at a breakpoint only once the group is substantial enough to be worth closing;
        # otherwise a document whose every sentence is a breakpoint yields one chunk per
        # sentence, which is the failure this function exists to avoid.
        if over_budget or (at_breakpoint and count(current) >= max_tokens // 2):
            groups.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _parent_pieces(
    body: str,
    *,
    strategy: str,
    parent_tokens: int,
    breakpoint_percentile: int,
    count: TokenCounter,
    embed: TextEmbedder | None,
    separators: Sequence[str],
) -> list[str]:
    """The parent-sized pieces of a document, by the configured strategy.

    Both strategies are capped at `parent_tokens`. The semantic one finds its boundaries by
    meaning and then obeys the budget; the recursive one obeys the budget and finds boundaries
    by typography. That is the whole difference, and it is what the eval measures.
    """
    if strategy == "semantic":
        if embed is None:
            msg = "chunk_strategy='semantic' needs an embedder; none was passed"
            raise ValueError(msg)
        sentences = split_sentences(body)
        breakpoints = semantic_breakpoints(sentences, embed, percentile=breakpoint_percentile)
        pieces = _group_by_breakpoints(sentences, breakpoints, parent_tokens, count)
        # A single sentence can still exceed the budget; fall back to the recursive splitter
        # for those rather than emitting an oversized parent.
        fitted: list[str] = []
        for piece in pieces:
            fitted.extend(_recurse(piece, separators, parent_tokens, count))
        return fitted
    return _recurse(body, separators, parent_tokens, count)


def chunk_document_parent_child(
    body: str,
    *,
    strategy: str = "semantic",
    parent_tokens: int,
    child_tokens: int,
    child_overlap: int = 0,
    breakpoint_percentile: int = 90,
    count_tokens: TokenCounter | None = None,
    embed: TextEmbedder | None = None,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[ParentChunk]:
    """Chunk one document body into parents, each holding its own children.

    `chunk_no` is assigned across the whole document in reading order — parent, then its
    children, then the next parent — so the citation handle `doc_slug#chunk_n` stays unique and
    roughly ordered, which is what makes a report readable.

    The three invariants at the top of this module still hold, for parents and children alike:
    every `text` is a verbatim contiguous slice of the body, nothing is dropped, and no child
    exceeds `child_tokens + child_overlap`. Parents are exempt from the embedder's ceiling
    because they are never embedded.
    """
    count = count_tokens or estimate_tokens
    stripped = body.strip("\n")
    if not stripped.strip():
        return []

    base_offset = body.index(stripped)
    pieces = _parent_pieces(
        stripped,
        strategy=strategy,
        parent_tokens=parent_tokens,
        breakpoint_percentile=breakpoint_percentile,
        count=count,
        embed=embed,
        separators=separators,
    )

    parents: list[ParentChunk] = []
    chunk_no = 0
    cursor = 0
    for piece in pieces:
        start = cursor
        cursor += len(piece)
        parent_text = piece.strip()
        if not parent_text:
            continue

        # Children are cut from the parent's text, not from the document, so a child can never
        # straddle a parent boundary — which would make expansion ambiguous.
        parent_no = chunk_no
        chunk_no += 1
        children: list[Chunk] = []
        for child in chunk_document(
            piece,
            max_tokens=child_tokens,
            overlap_tokens=child_overlap,
            count_tokens=count,
            separators=separators,
        ):
            children.append(
                child.model_copy(
                    update={
                        "chunk_no": chunk_no,
                        # Child offsets are relative to the parent piece; lift them into
                        # document coordinates so every chunk's offsets mean the same thing.
                        "start_char": base_offset + start + child.start_char,
                        "end_char": base_offset + start + child.end_char,
                        "heading": _heading_at(stripped, start, cursor),
                    }
                )
            )
            chunk_no += 1

        parents.append(
            ParentChunk(
                chunk_no=parent_no,
                text=parent_text,
                token_count=count(parent_text),
                heading=_heading_at(stripped, start, cursor),
                start_char=base_offset + start,
                end_char=base_offset + cursor,
                children=children,
            )
        )
    return parents
