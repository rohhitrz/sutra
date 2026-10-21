"""The partial-JSON extractor that makes streaming a structured answer possible.

The fragments a provider sends do not respect JSON boundaries, so the interesting bugs are
all about state carried across a split: a key cut in half, an escape whose second character
arrives in the next delta, a `\\uXXXX` spread over five.
"""

from __future__ import annotations

import json

import pytest

from sutra.generation.partial_json import StringFieldStreamer, extract_string_field

DRAFT = {
    "answer": "Tulvo Brann keeps the ledger. [2]",
    "citations": [2],
    "confidence": 0.9,
    "refused": False,
}


def _stream(fragments: list[str], field: str = "answer") -> str:
    streamer = StringFieldStreamer(field)
    return "".join(streamer.feed(f) for f in fragments)


def test_extracts_the_field_from_a_whole_document():
    assert _stream([json.dumps(DRAFT)]) == DRAFT["answer"]


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 23, 64])
def test_every_fixed_chunk_size_yields_the_same_text(size: int):
    """Character-by-character included: the hardest case for any carried state."""
    text = json.dumps(DRAFT)
    fragments = [text[i : i + size] for i in range(0, len(text), size)]
    assert _stream(fragments) == DRAFT["answer"]


@pytest.mark.parametrize("split", range(1, len(json.dumps(DRAFT))))
def test_every_single_split_point_yields_the_same_text(split: int):
    """Exhaustive over two-fragment splits, so no boundary is left untested."""
    text = json.dumps(DRAFT)
    assert _stream([text[:split], text[split:]]) == DRAFT["answer"]


def test_the_key_may_be_split_across_fragments():
    assert _stream(['{"ans', 'wer": "hi", "refused": false}']) == "hi"


def test_a_field_whose_name_is_a_prefix_of_another_is_not_confused():
    doc = json.dumps({"answer_draft": "wrong", "answer": "right"})
    # The needle is the quoted key, so `"answer"` cannot match inside `"answer_draft"`.
    assert _stream([doc]) == "right"


def test_other_fields_before_the_target_are_skipped():
    doc = json.dumps({"refused": False, "citations": [1, 2], "answer": "after"})
    assert _stream([doc]) == "after"


# ------------------------------------------------------------------ escaping


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"answer": "a\\"quoted\\" word"}', 'a"quoted" word'),
        ('{"answer": "line\\nbreak"}', "line\nbreak"),
        ('{"answer": "a\\\\backslash"}', "a\\backslash"),
        ('{"answer": "tab\\there"}', "tab\there"),
        ('{"answer": "slash\\/here"}', "slash/here"),
        ('{"answer": "\\u00e9t\\u00e9"}', "été"),
        ('{"answer": "em \\u2014 dash"}', "em — dash"),
    ],
)
def test_escapes_are_decoded(raw: str, expected: str):
    assert _stream([raw]) == expected


def test_an_escape_split_across_fragments_still_decodes():
    assert _stream(['{"answer": "a\\', 'nb"}']) == "a\nb"


def test_a_unicode_escape_split_across_five_fragments_still_decodes():
    assert _stream(['{"answer": "\\', "u", "00", "e", '9"}']) == "é"


def test_an_escaped_quote_does_not_end_the_value():
    assert _stream(['{"answer": "say \\"hi\\" now", "refused": false}']) == 'say "hi" now'


def test_a_quote_in_a_later_field_is_not_mistaken_for_the_end():
    doc = json.dumps({"answer": "done", "other": "more text"})
    assert _stream([doc]) == "done"


# ------------------------------------------------- truncation and malformed input


def test_a_truncated_value_yields_what_arrived():
    """A model that stops mid-string produced a short answer, not a failed request."""
    assert _stream(['{"answer": "half an ans']) == "half an ans"


def test_a_missing_field_yields_nothing_and_does_not_raise():
    assert _stream([json.dumps({"refused": True})]) == ""


def test_empty_and_absent_input_are_safe():
    assert _stream([]) == ""
    assert _stream(["", "", ""]) == ""
    assert _stream(['{"answer": ""}']) == ""


def test_a_non_string_value_is_ignored_rather_than_guessed_at():
    assert _stream([json.dumps({"answer": None})]) == ""
    assert _stream([json.dumps({"answer": 42})]) == ""


def test_done_is_set_only_once_the_closing_quote_arrives():
    streamer = StringFieldStreamer("answer")
    streamer.feed('{"answer": "partial')
    assert not streamer.done
    streamer.feed('"}')
    assert streamer.done


def test_nothing_is_emitted_after_the_field_closes():
    streamer = StringFieldStreamer("answer")
    assert streamer.feed('{"answer": "a"') == "a"
    assert streamer.feed(', "answer": "b"}') == ""


def test_the_one_shot_helper_matches_the_streaming_form():
    text = json.dumps(DRAFT)
    assert extract_string_field("answer", text) == DRAFT["answer"]
    assert extract_string_field("answer", '{"answer": "cut') == "cut"


@pytest.mark.parametrize(
    "answer",
    [
        "plain text",
        'with "quotes" inside',
        "with\nnewlines\tand\ttabs",
        "with a backslash \\ in it",
        "unicode: été, 日本語, — em dash",
        "braces { } and brackets [ ] and a colon:",
        'a fake field "citations": [9] inside the prose',
        "",
    ],
)
def test_round_trips_whatever_json_dumps_produces(answer: str):
    """The real contract: whatever the encoder emits, we decode back, at any chunk size."""
    text = json.dumps({"answer": answer, "citations": [1], "refused": False})
    for size in (1, 3, 8):
        fragments = [text[i : i + size] for i in range(0, len(text), size)]
        assert _stream(fragments) == answer
