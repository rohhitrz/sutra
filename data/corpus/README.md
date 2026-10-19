# The Meridian Chronicles — frozen corpus

**Frozen 2026-10-19, end of Phase 1 (ADR-014).**

These ten files are the **only source of truth for story facts** in Sutra. Nothing in the code,
the tests, the prompts or the eval data may assert a fact about this world that is not written
here.

## Do not edit these files without a decision

Every eval baseline from Phase 2 onward is measured against this exact text, and
`evals/datasets/golden_qa.jsonl` cites **72 verbatim quotes** from it. An edit does two things at
once: it invalidates the scoreboard, and it may silently orphan a quote.

If a change is agreed:

1. make it;
2. run `make validate` and repair every quote it reports;
3. re-run `make eval` and record the new numbers in `PROGRESS.md` as a new baseline, saying what
   changed and why the old numbers no longer compare.

## What is in here, and why

`docs/world_bible.md` is the design record — the world, the cast, and §8, which lists every trap
planted here and what it is for. `evals/datasets/README.md` maps each trap to the eval rows that
test it. Read those two before reading the files themselves; they are much faster than inferring
the design from the prose.

Summary of what is planted: five facts updated in Book 2, seven multi-hop facts, four numeric
traps, three routing cases, six questions the corpus deliberately does **not** answer, and — all
confined to `10_traveler_notes.md` — three indirect prompt injections, four fake PII items and one
confidently wrong traveller claim.

## A note on the poisoned file

`10_traveler_notes.md` contains working prompt-injection text and fake contact details. Both are
**test fixtures**. The PII is drawn from ranges reserved for fiction (`.example` domains,
`+1-555-01xx` numbers) and cannot belong to a real person. The injections are there to be
retrieved and ignored; if a guardrail change makes them stop being retrieved at all, the
guardrail is being measured on the wrong thing.
