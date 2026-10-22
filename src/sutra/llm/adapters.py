"""Provider adapters.

ADR-005 makes `sutra.llm` the only place that talks to a provider; this module is the only place
that imports a provider SDK. Everything above it works against the `LLMAdapter` protocol, so
adding a second provider (Phase 5) means adding a class here and nothing else.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from sutra.llm.types import Message

SchemaT = TypeVar("SchemaT", bound=BaseModel)


# Every event that ends a response stream. `response.completed` is the happy one;
# `response.incomplete` is a truncation (the max_output_tokens case) and `response.failed` is
# a provider-side error. All three carry usage, and all three are billed.
TERMINAL_STREAM_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed"})


def _incomplete_reason(response: object) -> str | None:
    """The provider's reason for stopping early, when it stopped early."""
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    return str(reason) if reason else None


class RawResult(BaseModel):
    """What an adapter returns: provider-shaped data normalised, but not yet priced.

    Costing deliberately happens in the client, not the adapter — the price table is shared and
    one adapter must not be able to disagree with another about what a call cost.
    """

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    # Reasoning tokens are *included* in output_tokens, not additional to them. Tracked
    # separately because they are invisible in the response text while being most of what a
    # reasoning model charges for, and because they are what exhausts max_output_tokens.
    reasoning_tokens: int = 0
    response_id: str | None = None
    parsed: BaseModel | None = None
    # The provider's terminal status, when it gave one. `complete` and `structured` raise on
    # `incomplete` (see `_reject_if_truncated`); the streaming path cannot, because tokens are
    # already out the door, so it records it instead and the caller decides. `None` means the
    # non-streaming paths, which only ever return a status they did not object to.
    status: str | None = None
    incomplete_reason: str | None = None


class LLMAdapter(Protocol):
    """The surface every provider must implement."""

    name: str

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int,
        temperature: float | None,
        timeout: float,
    ) -> RawResult: ...

    def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int,
        temperature: float | None,
        timeout: float,
        schema: type[SchemaT] | None = ...,
    ) -> AsyncIterator[tuple[str | None, RawResult | None]]: ...

    async def structured(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        schema: type[SchemaT],
        max_output_tokens: int,
        temperature: float | None,
        timeout: float,
    ) -> RawResult: ...

    # Deliberately *not* on the protocol. An adapter that owns a connection pool offers
    # `aclose()` and `LLMClient.aclose` finds it by duck-typing; requiring it here would force
    # every test fake to implement a no-op for a resource it does not hold.


def _split_instructions(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, str]]]:
    """The Responses API takes system content as a separate `instructions` argument rather than
    as a message. Collapse any system turns into it and pass the rest through."""
    system_parts = [m.content for m in messages if m.role == "system"]
    rest = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    instructions = "\n\n".join(system_parts) if system_parts else None
    return instructions, rest


class TruncatedResponseError(RuntimeError):
    """The provider stopped before finishing because the output budget ran out."""


class OpenAIAdapter:
    """OpenAI adapter, built on the Responses API (ADR-017).

    Note `max_retries=0`: the client owns retry policy. Leaving the SDK's own retries on would
    multiply them — three client attempts over two SDK retries is nine requests, not three.
    """

    name = "openai"

    def __init__(self, api_key: str, timeout: float) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool.

        An `AsyncOpenAI` owns an httpx pool with live sockets. Dropping the reference without
        closing leaks those until the garbage collector gets to them, which under load means
        file descriptors accumulating faster than they are reclaimed.
        """
        await self._client.close()

    @staticmethod
    def _usage_from(response: object) -> tuple[int, int, int, int]:
        """Pull (input, output, cached, reasoning) token counts out of a response, tolerating
        their absence — usage is occasionally missing on error paths."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0, 0
        in_details = getattr(usage, "input_tokens_details", None)
        out_details = getattr(usage, "output_tokens_details", None)
        return (
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            getattr(in_details, "cached_tokens", 0) or 0,
            getattr(out_details, "reasoning_tokens", 0) or 0,
        )

    @staticmethod
    def _reject_if_truncated(response: object, max_output_tokens: int) -> None:
        """Fail with the real cause when the model ran out of room.

        # CONCEPT: `max_output_tokens` budgets reasoning *and* visible output together, so a cap
        # sized for the answer you want to read can be entirely consumed by reasoning. The
        # response then comes back `incomplete` with a half-written body, and a schema-parsing
        # call reports it as "Invalid JSON: EOF while parsing a string" — which sends you
        # debugging the provider, or the schema, instead of raising the budget. Naming the cause
        # here is the difference between a five-minute fix and an afternoon.
        """
        if getattr(response, "status", None) != "incomplete":
            return
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) or "unknown"
        usage = getattr(response, "usage", None)
        out_details = getattr(usage, "output_tokens_details", None)
        reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
        msg = (
            f"the model stopped early (reason={reason}) with max_output_tokens="
            f"{max_output_tokens}, of which {reasoning} went on reasoning. The body is "
            "truncated; raise max_output_tokens rather than treating this as a bad response."
        )
        raise TruncatedResponseError(msg)

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int,
        temperature: float | None,
        timeout: float,
    ) -> RawResult:
        instructions, input_messages = _split_instructions(messages)
        kwargs: dict[str, object] = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
        }
        if instructions is not None:
            kwargs["instructions"] = instructions
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await self._client.responses.create(**kwargs)  # type: ignore[arg-type]
        self._reject_if_truncated(response, max_output_tokens)
        inp, out, cached, reasoning = self._usage_from(response)
        return RawResult(
            text=response.output_text or "",
            model=getattr(response, "model", model),
            input_tokens=inp,
            output_tokens=out,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            response_id=getattr(response, "id", None),
        )

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_output_tokens: int,
        temperature: float | None,
        timeout: float,
        schema: type[SchemaT] | None = None,
    ) -> AsyncIterator[tuple[str | None, RawResult | None]]:
        """Yields `(delta, None)` for each text chunk, then exactly one `(None, result)` with
        the final usage. Usage is only known once the stream completes, which is why the
        terminal event is separate.

        With `schema`, the response is still constrained at decode time and **the deltas are
        fragments of the JSON object**, not prose. Callers that want a field out of it as it
        arrives have to parse the partial JSON — see `sutra.generation.partial_json`. The
        alternative was a second, prose-shaped answer prompt, which would have meant two
        generation paths of differing quality and a scoreboard describing only one of them.
        """
        instructions, input_messages = _split_instructions(messages)
        kwargs: dict[str, object] = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
        }
        if instructions is not None:
            kwargs["instructions"] = instructions
        if temperature is not None:
            kwargs["temperature"] = temperature

        chunks: list[str] = []
        final: RawResult | None = None

        async for event in self._raw_stream_events(kwargs, schema):
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    chunks.append(delta)
                    yield delta, None
            elif etype in TERMINAL_STREAM_EVENTS:
                # All three terminal events carry the same `response`, and the two that are
                # not `completed` are the ones that cost money and used to be dropped:
                # matching only `response.completed` left `final` unset, so the fallback
                # below reported a *zero-usage* result and a truncated streamed answer was
                # recorded as costing nothing. That is the failure known issue 10 describes —
                # `max_output_tokens` budgets reasoning and visible output together, so a cap
                # sized for the answer gets eaten by reasoning and the response comes back
                # `incomplete` — and it is exactly the call whose cost you want to see.
                #
                # Unlike `complete`/`structured`, this does not raise on a truncated
                # response. It cannot: tokens have already been handed to the caller, which
                # is why there are no retries here either, and ADR-025 deliberately lets a
                # truncated stream still yield the partial answer it managed to produce. The
                # bug was the accounting, not the tolerance — so the status is recorded and
                # the tokens are billed.
                response = getattr(event, "response", None)
                inp, out, cached, reasoning = self._usage_from(response)
                final = RawResult(
                    text="".join(chunks),
                    model=getattr(response, "model", model) if response else model,
                    input_tokens=inp,
                    output_tokens=out,
                    cached_tokens=cached,
                    reasoning_tokens=reasoning,
                    response_id=getattr(response, "id", None) if response else None,
                    # The provider's own validated instance, so a streamed structured answer
                    # is parsed exactly like a non-streamed one rather than re-parsed from
                    # the text we happened to accumulate.
                    parsed=getattr(response, "output_parsed", None) if response else None,
                    status=getattr(response, "status", None) if response else None,
                    incomplete_reason=_incomplete_reason(response),
                )

        if final is None:
            # The stream ended without *any* terminal event (e.g. a dropped connection), so
            # there is no usage to report. Now genuinely that case, rather than also catching
            # every truncated and failed response.
            final = RawResult(text="".join(chunks), model=model, status="no_terminal_event")
        yield None, final

    async def _raw_stream_events(
        self, kwargs: dict[str, object], schema: type[SchemaT] | None
    ) -> AsyncIterator[object]:
        """The provider's event stream, schema-constrained or not.

        Two call shapes because the SDK offers two: `responses.create(stream=True)` for free
        text, and `responses.stream(text_format=...)`, a context manager that also attaches
        the parsed instance. Isolated here so `stream` above has one loop rather than two.
        """
        if schema is None:
            raw = await self._client.responses.create(stream=True, **kwargs)  # type: ignore[arg-type]
            async for event in raw:  # type: ignore[union-attr]
                yield event
            return

        async with self._client.responses.stream(text_format=schema, **kwargs) as raw:  # type: ignore[arg-type]
            async for event in raw:
                yield event

    async def structured(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        schema: type[SchemaT],
        max_output_tokens: int,
        temperature: float | None,
        timeout: float,
    ) -> RawResult:
        """Uses the provider's native schema-constrained parsing, so the model is constrained
        at decode time rather than asked politely for JSON and validated afterwards."""
        instructions, input_messages = _split_instructions(messages)
        kwargs: dict[str, object] = {
            "model": model,
            "input": input_messages,
            "text_format": schema,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
        }
        if instructions is not None:
            kwargs["instructions"] = instructions
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await self._client.responses.parse(**kwargs)  # type: ignore[arg-type]
        self._reject_if_truncated(response, max_output_tokens)
        inp, out, cached, reasoning = self._usage_from(response)
        return RawResult(
            text=response.output_text or "",
            model=getattr(response, "model", model),
            input_tokens=inp,
            output_tokens=out,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            response_id=getattr(response, "id", None),
            parsed=getattr(response, "output_parsed", None),
        )
