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
    CHUNKER_VERSION,
    DEFAULT_SEPARATORS,
    chunk_document,
    chunk_document_parent_child,
    estimate_tokens,
    semantic_breakpoints,
    split_sentences,
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
        "chunker_version": 1,
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
            "chunker_version": 1,
            "chunk_overlap": 60,
            "chunk_tokens": 400,
            "chunk_strategy": "recursive",
        }
    )
    assert reordered.digest == _fingerprint().digest


@pytest.mark.parametrize(
    "field",
    [
        "chunk_strategy",
        "chunk_tokens",
        "chunk_overlap",
        "chunker_version",
        "embedding_model",
        "corpus_hash",
    ],
)
def test_every_field_changes_the_digest(field: str):
    """If a field could change without changing the digest, an eval baseline could silently stop
    describing the index it was measured on."""
    changed = {"chunk_tokens": 250, "chunk_overlap": 20, "chunker_version": 99}.get(
        field, "different"
    )
    assert _fingerprint(**{field: changed}).digest != _fingerprint().digest


def test_differences_names_what_changed():
    diff = _fingerprint(chunk_tokens=250).differences_from(_fingerprint())
    assert diff == ["chunk_tokens: 400 -> 250"]


def test_build_fingerprint_reads_the_chunking_settings(settings: Settings):
    fp = build_fingerprint(settings, parse_corpus(CORPUS))
    assert fp.chunk_tokens == settings.chunk_tokens
    assert fp.chunk_overlap == settings.chunk_overlap
    assert fp.embedding_model == settings.embedding_model
    # Not a setting: it comes from the code, which is the point of ADR-022.
    assert fp.chunker_version == CHUNKER_VERSION


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


# ---------------------------------------------------------------------------
# the token counter that feeds the oversize guard
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_count_tokens_does_not_saturate_at_the_sequence_limit():
    """The counter must keep counting past 512, or the guard it feeds is decorative.

    fastembed configures its tokenizer to truncate at the model's sequence limit. Counting on
    that tokenizer makes every oversized text report exactly `MAX_SEQUENCE_TOKENS`, so a chunk
    that would be silently cut off looks like one sitting exactly on the line. Phase 2 never
    noticed because `chunk_tokens + chunk_overlap` is 460 by config, but Phase 3's semantic
    chunking makes chunk size data-dependent, and then this number has to be true.
    """
    from sutra.ingestion.embed import MAX_SEQUENCE_TOKENS, get_embedder

    embedder = get_embedder()

    assert embedder.count_tokens("The Archivist walked to the waystone.") < 20

    long_text = "waystone " * 4444
    counted = embedder.count_tokens(long_text)
    assert counted > MAX_SEQUENCE_TOKENS, (
        f"counted {counted} tokens for a text of roughly 8,900 — the counting tokenizer is "
        "still truncating"
    )


@pytest.mark.model
def test_counting_does_not_disturb_the_embedding_tokenizer():
    """Counting gets a copy. Disabling truncation on the tokenizer fastembed is *using* would
    push over-length sequences into ONNX, trading a silent truncation for a crash."""
    from sutra.ingestion.embed import FastEmbedEmbedder

    # The concrete class, not the Protocol: this asserts on the library state behind it.
    embedder = FastEmbedEmbedder()
    embedder.count_tokens("waystone " * 4444)

    # Reaching into the embedder is the point here: the assertion is about library state.
    model = embedder._ensure_model()
    assert model.model.tokenizer.truncation is not None  # type: ignore[attr-defined]
    assert len(embedder.embed_passages(["waystone " * 4444])[0]) == embedder.dims


# ---------------------------------------------------------------------------
# semantic chunking + parent-child (Phase 3 step 1)
# ---------------------------------------------------------------------------


def _fake_embed(texts):
    """Vectors in two clusters, keyed by a marker in the text.

    Deterministic and dependency-free: the tests below are about *where the boundaries land*
    given a similarity signal, not about what a real model thinks is similar.
    """
    return [[1.0, 0.0] if "ALPHA" in t else [0.0, 1.0] for t in texts]


def test_split_sentences_loses_nothing():
    """These pieces are reassembled into chunk text, which must stay a verbatim slice."""
    text = "One thing happened. Then another!  And a third?\nA fourth with no end"
    pieces = split_sentences(text)
    assert "".join(pieces) == text
    assert len(pieces) == 4


def test_split_sentences_handles_empty_and_single():
    assert split_sentences("") == []
    assert split_sentences("No terminal punctuation") == ["No terminal punctuation"]


def test_semantic_breakpoints_land_on_the_topic_shift():
    sentences = ["ALPHA one. ", "ALPHA two. ", "BETA one. ", "BETA two."]
    breaks = semantic_breakpoints(sentences, _fake_embed, percentile=50)
    # The only real shift is between index 1 and 2, so the new chunk starts at 2.
    assert breaks == {2}


def test_semantic_breakpoints_split_nothing_when_everything_is_similar():
    """A uniform document — the merchant ledger is close to this — must not be cut between
    every sentence. `>` rather than `>=` against the percentile is what prevents it."""
    sentences = ["ALPHA one. ", "ALPHA two. ", "ALPHA three. ", "ALPHA four."]
    assert semantic_breakpoints(sentences, _fake_embed, percentile=90) == set()


def test_breakpoints_are_permitted_cuts_not_required_ones():
    """Regression test for a real mistake made building this.

    Cutting at every breakpoint produced 67 parents averaging ~290 tokens against a budget of
    800 — a "parent" barely bigger than a child, which makes the whole parent-child split buy
    nothing. Groups must merge up toward the budget and cut only *at* breakpoints.
    """
    body = " ".join(f"ALPHA sentence number {i} with some words in it." for i in range(40))
    parents = chunk_document_parent_child(
        body,
        strategy="semantic",
        parent_tokens=400,
        child_tokens=100,
        embed=_fake_embed,
        count_tokens=estimate_tokens,
    )
    assert len(parents) <= 3, "parents are being cut far below the budget"
    assert max(p.token_count for p in parents) > 100, "no parent is meaningfully parent-sized"


def test_parent_child_invariants_on_the_real_corpus():
    """The three module invariants, for parents and children alike, over all ten documents."""
    settings = Settings()
    for path in sorted(Path("data/corpus").glob("*.md")):
        if path.name == "README.md":
            continue
        doc = parse_document(path)
        parents = chunk_document_parent_child(
            doc.body,
            strategy="recursive",  # no embedder needed, and the invariants are strategy-free
            parent_tokens=settings.parent_tokens,
            child_tokens=settings.child_tokens,
            child_overlap=settings.child_overlap,
            count_tokens=estimate_tokens,
        )
        assert parents, f"{doc.slug} produced no parents"

        numbers = [p.chunk_no for p in parents] + [c.chunk_no for p in parents for c in p.children]
        assert len(numbers) == len(set(numbers)), f"{doc.slug} reused a chunk_no"

        for parent in parents:
            assert parent.text in doc.body, f"{doc.slug} parent text is not a verbatim slice"
            assert parent.token_count <= settings.parent_tokens
            for child in parent.children:
                assert child.text in doc.body, f"{doc.slug} child text is not a verbatim slice"
                # The documented bound: budget plus overlap, plus at most one word of slack.
                assert child.token_count <= settings.child_tokens + settings.child_overlap + 20


def test_children_are_cut_from_their_own_parent():
    """A child straddling a parent boundary would make expansion ambiguous — the retrieved
    evidence would sit partly outside the passage the model is handed."""
    body = "\n\n".join(f"## Section {i}\n\n" + ("word " * 300) for i in range(4))
    parents = chunk_document_parent_child(
        body,
        strategy="recursive",
        parent_tokens=400,
        child_tokens=100,
        count_tokens=estimate_tokens,
    )
    for parent in parents:
        for child in parent.children:
            assert child.text.strip() in parent.text or parent.text in child.text


def test_semantic_strategy_requires_an_embedder():
    """Failing loudly beats silently falling back to recursive and writing an index whose
    fingerprint claims it is semantic."""
    with pytest.raises(ValueError, match="needs an embedder"):
        chunk_document_parent_child(
            "Some text. More text.",
            strategy="semantic",
            parent_tokens=400,
            child_tokens=100,
            count_tokens=estimate_tokens,
        )
