"""Ask one question end to end: retrieve, generate, print the provenance.

Not part of the serving path. It exists because `make eval` reports aggregates, and when a number
moves you need to see one answer in full — the passages that were retrieved, the citations the
model made, what it cost.

    uv run python -m sutra.generation.demo "What is the population of Kesh Ithar?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sutra.config import get_settings
from sutra.db.engine import get_sessionmaker
from sutra.generation.answer import generate_answer
from sutra.generation.schema import Answer
from sutra.retrieval.search import RetrievedChunk, vector_search


async def ask(question: str, *, k: int | None = None) -> tuple[list[RetrievedChunk], Answer]:
    settings = get_settings()
    factory = get_sessionmaker()
    async with factory() as session:
        chunks = await vector_search(question, session=session, k=k, settings=settings)
    return chunks, await generate_answer(question, chunks, settings=settings)


def _print(question: str, chunks: list[RetrievedChunk], answer: Answer) -> None:
    cited = {c.label for c in answer.citations}
    print(f"\nQ  {question}\n")
    print("retrieved")
    for chunk in chunks:
        mark = "*" if chunk.rank in cited else " "
        heading = chunk.heading or "—"
        print(f"  {mark} [{chunk.rank}] {chunk.score:.3f}  {chunk.doc_slug:22s} {heading[:44]}")
    print("\nanswer")
    for line in answer.answer.splitlines() or [""]:
        print(f"  {line}")
    print(
        f"\n  refused={answer.refused}  confidence={answer.confidence:.2f}  "
        f"cited={sorted(cited)}  dropped={answer.dropped_citations}"
    )
    cost = answer.usage.cost_usd
    cost_text = f"${cost:.6f}" if cost is not None else "unpriced"
    print(
        f"  model={answer.model}  prompt={answer.prompt_version}  "
        f"tokens={answer.usage.input_tokens}in/{answer.usage.output_tokens}out  "
        f"cost={cost_text}"
    )
    print(f"  latency={answer.latency_ms:.0f}ms  fallback={answer.used_fallback}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the corpus one question.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=None, help="how many passages to retrieve")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level="DEBUG" if args.verbose else "WARNING", format="%(levelname)s %(name)s %(message)s"
    )
    chunks, answer = asyncio.run(ask(args.question, k=args.k))
    _print(args.question, chunks, answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
