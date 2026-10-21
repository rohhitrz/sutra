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
    ) -> AsyncIterator[tuple[str | None, RawResult | None]]:
        """Yields `(delta, None)` for each text chunk, then exactly one `(None, result)` with
        the final usage. Usage is only known once the stream completes, which is why the
        terminal event is separate."""
        instructions, input_messages = _split_instructions(messages)
        kwargs: dict[str, object] = {
            "model": model,
            "input": input_messages,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
            "stream": True,
        }
        if instructions is not None:
            kwargs["instructions"] = instructions
        if temperature is not None:
            kwargs["temperature"] = temperature

        chunks: list[str] = []
        final: RawResult | None = None
        stream = await self._client.responses.create(**kwargs)  # type: ignore[arg-type]
        async for event in stream:  # type: ignore[union-attr]
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    chunks.append(delta)
                    yield delta, None
            elif etype == "response.completed":
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
                )

        if final is None:
            # The stream ended without a completion event (e.g. a dropped connection). Report
            # what we received rather than silently returning nothing.
            final = RawResult(text="".join(chunks), model=model)
        yield None, final

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
