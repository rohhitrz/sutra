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


def _split_instructions(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, str]]]:
    """The Responses API takes system content as a separate `instructions` argument rather than
    as a message. Collapse any system turns into it and pass the rest through."""
    system_parts = [m.content for m in messages if m.role == "system"]
    rest = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    instructions = "\n\n".join(system_parts) if system_parts else None
    return instructions, rest


class OpenAIAdapter:
    """OpenAI adapter, built on the Responses API (ADR-017).

    Note `max_retries=0`: the client owns retry policy. Leaving the SDK's own retries on would
    multiply them — three client attempts over two SDK retries is nine requests, not three.
    """

    name = "openai"

    def __init__(self, api_key: str, timeout: float) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    @staticmethod
    def _usage_from(response: object) -> tuple[int, int, int]:
        """Pull (input, output, cached) token counts out of a response, tolerating their
        absence — usage is occasionally missing on error paths."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        return (
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            cached,
        )

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
        inp, out, cached = self._usage_from(response)
        return RawResult(
            text=response.output_text or "",
            model=getattr(response, "model", model),
            input_tokens=inp,
            output_tokens=out,
            cached_tokens=cached,
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
                inp, out, cached = self._usage_from(response)
                final = RawResult(
                    text="".join(chunks),
                    model=getattr(response, "model", model) if response else model,
                    input_tokens=inp,
                    output_tokens=out,
                    cached_tokens=cached,
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
        inp, out, cached = self._usage_from(response)
        return RawResult(
            text=response.output_text or "",
            model=getattr(response, "model", model),
            input_tokens=inp,
            output_tokens=out,
            cached_tokens=cached,
            response_id=getattr(response, "id", None),
            parsed=getattr(response, "output_parsed", None),
        )
