"""Does the semantic cache ever serve the wrong answer? (`make cache`)

# CONCEPT: A cache that is merely fast is not interesting; a cache that is fast and wrong is
# a correctness bug with a performance benefit attached. The useful question is never "what
# is the hit rate" but "what did it hit that it should have missed".
#
# Three failure modes, each with its own rows below:
#
#   paraphrase miss   the cache is too strict and pays for an answer it already had. Cheap.
#   near-miss HIT     two questions a token apart with different answers — 1143 against 1144,
#                     Halvern Reach against Kesh Ithar. The cache serves a confident, fluent,
#                     wrong answer and nothing downstream can detect it. Expensive.
#   cross-user HIT    one person's answer served to another. A privacy failure, not a
#                     quality one, and no threshold tuning fixes it — only the key does.
#
# The asymmetry is the whole point. A miss costs one model call; a wrong hit costs the
# answer's correctness and leaves no trace. So this script fails the build on a wrong hit and
# merely reports a missed paraphrase.
#
# No LLM is involved: answers are stubbed, and only the embedding and the key are exercised.
# That makes the check deterministic and free, so it can run on every change to the threshold.
"""

from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel

from sutra.cache.semantic import invalidate, lookup, store
from sutra.config import Settings, get_settings
from sutra.db.engine import get_sessionmaker

CACHE_TEST_USER = "cache-check-user"
OTHER_USER = "cache-check-other-user"


class Probe(BaseModel):
    """One question put to the cache, and whether a hit would be correct."""

    label: str
    stored: str
    probe: str
    should_hit: bool
    why: str


PROBES = [
    Probe(
        label="identical",
        stored="What is the population of Kesh Ithar?",
        probe="What is the population of Kesh Ithar?",
        should_hit=True,
        why="the same question twice is the case the cache exists for",
    ),
    Probe(
        label="paraphrase",
        stored="What is the population of Halvern Reach?",
        probe="How many people live in Halvern Reach?",
        should_hit=True,
        why="same fact, different words — an exact-match cache would miss this",
    ),
    Probe(
        label="near-miss: different year",
        stored="What was the Waystone Tithe rate in 1143?",
        probe="What was the Waystone Tithe rate in 1144?",
        should_hit=False,
        why="one token apart and a different fact; the corpus changes the rate between books",
    ),
    Probe(
        label="near-miss: different place",
        stored="What is the population of Halvern Reach?",
        probe="What is the population of Kesh Ithar?",
        should_hit=False,
        why="same shape, different subject — 41,000 against 68,000",
    ),
    Probe(
        label="near-miss: different person",
        stored="Who is Sera Volant?",
        probe="Who is Oren Dask?",
        should_hit=False,
        why="identical phrasing, different entity",
    ),
    Probe(
        label="near-miss: negated",
        stored="Is the Thennic Spur open?",
        probe="Is the Thennic Spur closed?",
        should_hit=False,
        why="embeddings treat negation poorly, which is exactly why this is tested",
    ),
]


class ProbeResult(BaseModel):
    label: str
    should_hit: bool
    did_hit: bool
    similarity: float | None
    why: str

    @property
    def ok(self) -> bool:
        return self.did_hit == self.should_hit

    @property
    def severity(self) -> str:
        if self.ok:
            return "ok"
        return "WRONG HIT" if self.did_hit else "missed"


async def run_probes(settings: Settings) -> tuple[list[ProbeResult], bool, bool]:
    """Returns (results, cross_user_leaked, stale_version_hit)."""
    factory = get_sessionmaker()
    results: list[ProbeResult] = []

    async with factory() as session:
        await invalidate(session=session, user_id=CACHE_TEST_USER)
        await invalidate(session=session, user_id=OTHER_USER)
        await session.commit()

        # Each probe runs against a cache holding *only* its own stored question. Seeding
        # them all at once was the first design and it was wrong: one probe's near-miss
        # question is another probe's stored question, so the cache hit it correctly and the
        # report called that a wrong hit. An isolated cache is the only way the result is
        # about the threshold rather than about the fixture.
        for probe in PROBES:
            await invalidate(session=session, user_id=CACHE_TEST_USER)
            await store(
                probe.stored,
                {"answer": f"stored answer for: {probe.stored}", "refused": False},
                user_id=CACHE_TEST_USER,
                session=session,
                settings=settings,
            )
            await session.commit()

            found = await lookup(
                probe.probe, user_id=CACHE_TEST_USER, session=session, settings=settings
            )
            results.append(
                ProbeResult(
                    label=probe.label,
                    should_hit=probe.should_hit,
                    did_hit=found.hit,
                    similarity=found.nearest_similarity,
                    why=probe.why,
                )
            )

        # Re-seed one entry for the scope and versioning checks below.
        await invalidate(session=session, user_id=CACHE_TEST_USER)
        await store(
            PROBES[0].stored,
            {"answer": "stored", "refused": False},
            user_id=CACHE_TEST_USER,
            session=session,
            settings=settings,
        )
        await session.commit()

        # Scope: another user must see nothing, however similar the question.
        leaked = (
            await lookup(PROBES[0].stored, user_id=OTHER_USER, session=session, settings=settings)
        ).hit

        # Versioning: a changed prompt version must not match, however similar the question.
        stale = (
            await lookup(
                PROBES[0].stored,
                user_id=CACHE_TEST_USER,
                session=session,
                settings=settings.model_copy(update={"prompt_version_answer": "v99"}),
            )
        ).hit

        await invalidate(session=session, user_id=CACHE_TEST_USER)
        await invalidate(session=session, user_id=OTHER_USER)
        await session.commit()

    return results, leaked, stale


def main() -> int:
    settings = get_settings().model_copy(update={"semantic_cache_enabled": True})
    results, leaked, stale = asyncio.run(run_probes(settings))

    print(f"semantic cache at threshold {settings.cache_similarity_threshold}\n")
    print(f"  {'probe':28s} {'want':>5s} {'got':>5s} {'sim':>6s}  ")
    for r in results:
        sim = "-" if r.similarity is None else f"{r.similarity:.3f}"
        want = "hit" if r.should_hit else "miss"
        got = "hit" if r.did_hit else "miss"
        flag = "" if r.ok else f"   <- {r.severity}"
        print(f"  {r.label:28s} {want:>5s} {got:>5s} {sim:>6s}{flag}")
        if not r.ok:
            print(f"      {r.why}")

    print(f"\n  cross-user leak             {'LEAKED' if leaked else 'no'}")
    print(f"  stale prompt version hit    {'HIT' if stale else 'no'}")

    wrong_hits = [r for r in results if not r.ok and r.did_hit]
    missed = [r for r in results if not r.ok and not r.did_hit]

    if missed:
        print(
            f"\nnote  {len(missed)} paraphrase(s) missed. The cache is strict; this costs a "
            "model call and nothing else."
        )
    if wrong_hits or leaked or stale:
        print(
            f"\nFAIL  {len(wrong_hits)} wrong hit(s)"
            f"{', cross-user leak' if leaked else ''}"
            f"{', stale version served' if stale else ''}.\n"
            "A wrong hit serves a confident, fluent, wrong answer that nothing downstream "
            "can detect. Lower the threshold or fix the key.",
            file=sys.stderr,
        )
        return 1

    print("\nok    no wrong hits, no cross-user leak, no stale version served")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
