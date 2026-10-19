"""Tests for the ingestion pipeline.

The chunker tests assert *invariants* rather than exact outputs. Pinning "document 6 yields 6
chunks" would break on every parameter change and tell us nothing; "no chunk exceeds the budget"
and "every chunk is a verbatim slice" are the properties the eval ground truth actually depends on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sutra.config import Settings
from sutra.ingestion.chunk import (
    DEFAULT_SEPARATORS,
    chunk_document,
    estimate_tokens,
)
from sutra.ingestion.fingerprint import IndexFingerprint, build_fingerprint
from sutra.ingestion.parse import (
    CorpusError,
    ParsedDocument,
    corpus_hash,
    parse_corpus,
    parse_document,
)
from sutra.ingestion.run import _validate_row_keys

CORPUS = Path("data/corpus")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parses_every_corpus_document_and_skips_the_freeze_notice():
    docs = parse_corpus(CORPUS)
    slugs = {d.slug for d in docs}
    assert len(docs) == 10
    # README.md is the freeze notice (ADR-014), not a document. The dataset validator skips it
    # the same way; if these two ever disagree, quote coverage drifts silently.
    assert "README" not in slugs
    assert all(d.title and d.doc_type for d in docs)


def test_promotes_title_and_doc_type_and_keeps_the_rest_as_metadata():
    doc = next(d for d in parse_corpus(CORPUS) if d.slug == "06_places_wiki")
    assert doc.doc_type == "wiki"
    assert "title" not in doc.meta and "doc_type" not in doc.meta
    assert "characters" in doc.meta


def test_front_matter_is_stripped_from_the_body():
    doc = next(d for d in parse_corpus(CORPUS) if d.slug == "08_faq")
    assert "doc_type:" not in doc.body


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("no front matter at all", "no YAML front-matter"),
        ("---\ndoc_type: faq\n---\nbody", "missing required key 'title'"),
        ("---\ntitle: x\n---\nbody", "missing required key 'doc_type'"),
        ("---\n- just\n- a list\n---\nbody", "must be a mapping"),
        ("---\ntitle: [unclosed\n---\nbody", "not valid YAML"),
    ],
)
def test_malformed_front_matter_is_fatal(tmp_path: Path, content: str, expected: str):
    """A skipped document is a silent recall hole, so parsing fails loudly instead."""
    path = tmp_path / "bad.md"
    path.write_text(content)
    with pytest.raises(CorpusError, match=expected):
        parse_document(path)


def test_empty_corpus_directory_is_fatal(tmp_path: Path):
    with pytest.raises(CorpusError, match="no corpus documents"):
        parse_corpus(tmp_path)


def test_corpus_hash_is_order_independent_but_edit_sensitive():
    docs = parse_corpus(CORPUS)
    assert corpus_hash(docs) == corpus_hash(list(reversed(docs)))

    edited = [
        d.model_copy(update={"content_hash": "changed"}) if d.slug == "08_faq" else d for d in docs
    ]
    assert corpus_hash(edited) != corpus_hash(docs)


# ---------------------------------------------------------------------------
# chunk — the three invariants from the module docstring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_docs() -> list[ParsedDocument]:
    return parse_corpus(CORPUS)


def test_invariant_every_chunk_is_a_verbatim_slice(corpus_docs: list[ParsedDocument]):
    """Ground truth is 72 verbatim quotes matched against chunk text. A chunker that rewrote or
    tidied text would silently destroy the measurement rather than fail."""
    for doc in corpus_docs:
        for chunk in chunk_document(doc.body, max_tokens=400, overlap_tokens=60):
            assert chunk.text in doc.body, f"{doc.slug} chunk {chunk.chunk_no} is not a slice"


def test_invariant_no_content_is_lost(corpus_docs: list[ParsedDocument]):
    for doc in corpus_docs:
        chunks = chunk_document(doc.body, max_tokens=400, overlap_tokens=0)
        rejoined = " ".join(c.text for c in chunks)
        assert norm(rejoined) == norm(doc.body), f"{doc.slug} lost content"


def test_invariant_chunks_respect_the_token_budget(corpus_docs: list[ParsedDocument]):
    """Overlap is prepended after the budget check, so the bound is the sum of the two."""
    for doc in corpus_docs:
        for chunk in chunk_document(doc.body, max_tokens=400, overlap_tokens=60):
            assert chunk.token_count <= 400 + 60


def test_a_tight_budget_still_respects_the_budget():
    """The recursive fallback has to keep working when the budget is smaller than a paragraph."""
    body = "## A\n\n" + ("Lorem ipsum dolor sit amet consectetur. " * 60)
    for chunk in chunk_document(body, max_tokens=30, overlap_tokens=0):
        assert chunk.token_count <= 30


def test_an_unbreakable_run_still_gets_split():
    """No separator left to try: the hard-split path, which exists so an oversized chunk can
    never reach the embedder and get silently truncated."""
    chunks = chunk_document("x" * 5000, max_tokens=20, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.token_count <= 20 * 2 for c in chunks)


def test_small_pieces_are_merged_rather_than_emitted_one_per_line():
    """Splitting a bullet list on newlines and *not* re-merging would emit one useless chunk per
    bullet. Recursive chunking is as much about merging back up as about splitting."""
    body = "## Places\n\n" + "\n".join(f"- item number {i}" for i in range(40))
    chunks = chunk_document(body, max_tokens=400, overlap_tokens=0)
    assert len(chunks) == 1


def test_overlap_repeats_the_tail_of_the_previous_chunk():
    body = "\n\n".join(f"Paragraph {i} " + "filler words here " * 30 for i in range(4))
    plain = chunk_document(body, max_tokens=60, overlap_tokens=0)
    overlapped = chunk_document(body, max_tokens=60, overlap_tokens=20)
    assert len(plain) > 1
    assert overlapped[1].token_count > plain[1].token_count
    assert overlapped[0].text.split()[-1] in overlapped[1].text


def test_heading_is_captured_in_metadata_not_prepended_to_text():
    body = "## Kesh Ithar — waystone 40\n\n- **Population:** 68,000\n"
    (chunk,) = chunk_document(body, max_tokens=400, overlap_tokens=0)
    assert chunk.heading == "Kesh Ithar — waystone 40"
    # text stays verbatim; Phase 3 is what measures whether prepending the heading helps
    assert chunk.text.startswith("## Kesh Ithar")


def test_heading_is_found_when_the_document_opens_with_an_h1_title():
    """Most corpus documents start `# Title` then `## Section`, so the opening chunk has no
    heading before it — only one inside it. Resolving just the "before" case left the first
    chunk of all ten documents with a null heading, which reads as missing data, not a bug."""
    body = "# The Long Patience, Part One\n\n## Chapter 1 — The Tide Clock\n\nHalvern Reach begins."
    (chunk,) = chunk_document(body, max_tokens=400, overlap_tokens=0)
    assert chunk.heading == "Chapter 1 — The Tide Clock"


def test_h3_counts_as_a_heading_too():
    """The corpus is not uniform: the traveller notes section on h3, not h2. Matching only h2
    left every chunk of that document unlabelled."""
    body = "### Box at waystone 34 — unsigned\n\nLost: a brass alidade, cracked."
    (chunk,) = chunk_document(body, max_tokens=400, overlap_tokens=0)
    assert chunk.heading == "Box at waystone 34 — unsigned"


def test_empty_and_whitespace_bodies_yield_no_chunks():
    assert chunk_document("", max_tokens=400) == []
    assert chunk_document("\n\n   \n", max_tokens=400) == []


def test_chunk_numbers_are_contiguous_from_zero(corpus_docs: list[ParsedDocument]):
    for doc in corpus_docs:
        chunks = chunk_document(doc.body, max_tokens=400, overlap_tokens=60)
        assert [c.chunk_no for c in chunks] == list(range(len(chunks)))


def test_separators_run_from_most_to_least_semantic():
    assert DEFAULT_SEPARATORS[0] == "\n## "
    assert DEFAULT_SEPARATORS[-1] == " "


def test_estimate_tokens_is_only_a_fallback():
    assert estimate_tokens("") == 1
    assert estimate_tokens("one two three") == 3


# ---------------------------------------------------------------------------
# fingerprint (ADR-021)
# ---------------------------------------------------------------------------


def _fingerprint(**overrides: object) -> IndexFingerprint:
    base: dict[str, object] = {
        "chunk_strategy": "recursive",
        "chunk_tokens": 400,
        "chunk_overlap": 60,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "corpus_hash": "abc123",
    }
    return IndexFingerprint.model_validate(base | overrides)


def test_digest_is_stable_and_field_order_independent():
    assert _fingerprint().digest == _fingerprint().digest
    reordered = IndexFingerprint.model_validate(
        {
            "corpus_hash": "abc123",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "chunk_overlap": 60,
            "chunk_tokens": 400,
            "chunk_strategy": "recursive",
        }
    )
    assert reordered.digest == _fingerprint().digest


@pytest.mark.parametrize(
    "field", ["chunk_strategy", "chunk_tokens", "chunk_overlap", "embedding_model", "corpus_hash"]
)
def test_every_field_changes_the_digest(field: str):
    """If a field could change without changing the digest, an eval baseline could silently stop
    describing the index it was measured on."""
    changed = {"chunk_tokens": 250, "chunk_overlap": 20}.get(field, "different")
    assert _fingerprint(**{field: changed}).digest != _fingerprint().digest


def test_differences_names_what_changed():
    diff = _fingerprint(chunk_tokens=250).differences_from(_fingerprint())
    assert diff == ["chunk_tokens: 400 -> 250"]


def test_build_fingerprint_reads_the_chunking_settings(settings: Settings):
    fp = build_fingerprint(settings, parse_corpus(CORPUS))
    assert fp.chunk_tokens == settings.chunk_tokens
    assert fp.chunk_overlap == settings.chunk_overlap
    assert fp.embedding_model == settings.embedding_model


# ---------------------------------------------------------------------------
# the regression test for a bug that reported success while losing data
# ---------------------------------------------------------------------------


def test_row_keys_must_be_real_columns():
    """A bulk insert built from dicts silently ignores a key it does not recognise.

    That turned `meta` (the mapped attribute) where `metadata` (the column) was needed into
    chunks written with empty metadata: the fingerprint became unreadable, ingestion was
    neither idempotent nor guarded, and every run still printed success.
    """
    good = {"document_id": None, "index_version": 1, "chunk_no": 0, "text": "x", "metadata": {}}
    _validate_row_keys([good])  # does not raise

    with pytest.raises(ValueError, match=r"not columns on 'chunks'.*meta"):
        _validate_row_keys([{**good, "meta": {}}])
