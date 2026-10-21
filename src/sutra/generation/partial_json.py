"""Pull one string field out of a JSON object that is still arriving.

# CONCEPT: Streaming a *structured* output means parsing incomplete JSON. The answer prompt
# is schema-constrained (`AnswerDraft`: answer, citations, confidence, refused), which is what
# keeps citations verifiable and refusals explicit — but it also means the provider's deltas
# are fragments of a JSON object, not prose:
#
#     {"ans   ·   wer":"The Arch   ·   ivist keeps   ·   the led   ·   ger.","citations":[2]...
#
# A token stream for a reader has to be the value of `answer`, unescaped, with the braces and
# the other fields left out. The alternative was a second prose-shaped prompt, which would
# have given streaming and non-streaming answers different quality and left Phase 3's
# scoreboard describing only one of them.
#
# This is a deliberately small scanner, not a JSON parser. It tracks one string value and
# knows nothing about nesting, because the field it is asked for is a top-level string in a
# flat schema — a constraint the schema already enforces (see `AnswerDraft`). The full object
# is still parsed properly at the end, by the provider, from the complete text.
"""

from __future__ import annotations

# JSON's two-character escapes, per RFC 8259. `\uXXXX` is handled separately because it needs
# four more characters before it means anything.
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class StringFieldStreamer:
    """Feed it JSON fragments; it yields the decoded text of one top-level string field.

    Call `feed()` with each delta and emit whatever it returns. The emitted pieces
    concatenate to exactly the field's value, so a caller can forward them as tokens.

    Deliberately total: malformed or truncated input yields less text, never an exception. A
    model that stops mid-string has produced a short answer, and a stream that raises in that
    case turns a degraded answer into a failed request.
    """

    def __init__(self, field: str) -> None:
        self._needle = f'"{field}"'
        self._buffer = ""
        # Where we are: before the key, between the key and its opening quote, inside the
        # value, or finished with it.
        self._state: str = "seeking"
        self._pending_escape = False
        self._unicode: str | None = None

    @property
    def done(self) -> bool:
        """True once the field's closing quote has been seen."""
        return self._state == "done"

    def feed(self, fragment: str) -> str:
        """Consume a fragment and return the newly decoded characters of the field."""
        if self._state == "done" or not fragment:
            return ""

        if self._state == "seeking":
            self._buffer += fragment
            index = self._buffer.find(self._needle)
            if index == -1:
                # Keep only enough tail to match a key split across two deltas.
                self._buffer = self._buffer[-len(self._needle) :]
                return ""
            # Past the key; the colon and the opening quote may still be coming.
            rest = self._buffer[index + len(self._needle) :]
            self._buffer = ""
            self._state = "pre_value"
            return self._consume_pre_value(rest)

        if self._state == "pre_value":
            return self._consume_pre_value(fragment)

        return self._consume_value(fragment)

    def _consume_pre_value(self, fragment: str) -> str:
        """Skip the `:` and any whitespace, then hand off at the opening quote."""
        for position, char in enumerate(fragment):
            if char == '"':
                self._state = "in_value"
                return self._consume_value(fragment[position + 1 :])
            if char not in ": \t\r\n":
                # Not a string value after all (`null`, a number, a nested object). Nothing
                # to stream, and nothing to raise about: the final parse is the authority.
                self._state = "done"
                return ""
        return ""

    def _consume_value(self, fragment: str) -> str:
        out: list[str] = []
        for char in fragment:
            if self._unicode is not None:
                self._unicode += char
                if len(self._unicode) == 4:
                    try:
                        out.append(chr(int(self._unicode, 16)))
                    except ValueError:
                        pass  # Not valid hex; drop it rather than fail the stream.
                    self._unicode = None
                continue

            if self._pending_escape:
                self._pending_escape = False
                if char == "u":
                    self._unicode = ""
                else:
                    out.append(_ESCAPES.get(char, char))
                continue

            if char == "\\":
                self._pending_escape = True
                continue

            if char == '"':
                self._state = "done"
                break

            out.append(char)

        return "".join(out)


def extract_string_field(field: str, text: str) -> str:
    """One-shot form of the streamer, for a complete or truncated document.

    Used when a stream ends without the provider's parsed instance — a dropped connection, or
    a response cut short by the output budget — so a partial answer can still be served
    rather than discarded.
    """
    return StringFieldStreamer(field).feed(text)
