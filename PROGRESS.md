# Sutra — Progress

> Handoff file between chats. Read at the start of every chat. Update at the end of every chat.

## Current phase
**Phase 2 — Ingestion + naive RAG + eval harness** · Status: ⬜ not started

## Phase status

| # | Phase | Status | Chat date |
|---|---|---|---|
| 0 | Foundation | ✅ done | 2026-10-19 |
| 1 | Corpus + golden datasets | ✅ done | 2026-10-19 |
| 2 | Ingestion + naive RAG + eval harness | ⬜ not started | |
| 3 | Advanced retrieval + routing + semantic cache | ⬜ | |
| 4 | Observability | ⬜ | |
| 5 | Generation, context management, memory | ⬜ | |
| 6 | Guardrails | ⬜ | |
| 7 | Agentic RAG + MCP server | ⬜ | |
| 8 | Voice agent | ⬜ | |
| 9 | CI evals, hardening, CONCEPTS.md, ablation report | ⬜ | |

Legend: ⬜ not started · 🟨 in progress · ✅ done

## Eval scoreboard
(Filled from Phase 2 onward. One row per eval run that matters.)

| Date | Config / change | Recall@5 | MRR | Faithfulness | Answer correctness | p50 latency | Cost/query |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Caveat to carry forward (ADR-017):** `LLM_STRONG` is also the judge model, so judge scores are
same-family self-evaluation. Read faithfulness and correctness as **relative** — valid for
comparing profiles against each other, not as ground truth.

## Decisions taken in chats

- **ADR-017** — LLM provider is **OpenAI**, `gpt-6-luna` for *both* `LLM_FAST` and `LLM_STRONG`.
  Verified rates: $0.10 / $0.50 per 1M input / output, $0.01 cached. Chosen on cost, because eval
  volume (~250 calls per `make eval`, re-run after every retrieval change) dominates spend, not
  chat traffic. Closes the Phase 0 open question. Declined a separate `LLM_JUDGE` role.
- **ADR-018** — recorded the concrete runtime dependency set, since ADR-015 covered only the dev
  toolchain. Dev-only `httpx2` (not `httpx`) because the `openai` SDK is built on httpx2.
- **ADR-019** — `make eval` is exempt from the ">200 LLM calls needs explicit OK" rule, which one
  eval run exceeds by construction.

## What was built (Phase 1)

- `docs/world_bible.md` — the design record for "The Meridian Chronicles"
- `data/corpus/` — the **10 frozen corpus files** (100–200 lines each, full YAML front-matter),
  plus `README.md`, the freeze notice
- `evals/datasets/golden_qa.jsonl` — 42 rows, 72 verbatim source quotes
- `evals/datasets/attacks.jsonl` — 40 rows (30 attacks + 10 benign look-alikes)
- `evals/datasets/README.md` — the trap map: every trap, the row that tests it, and what a wrong
  answer looks like
- `evals/validate_datasets.py` + `make validate` — seven checks, including the quote check
- `evals/render_datasets.py` + `make datasets` — indented, readable mirrors of both datasets under
  `evals/datasets/readable/`, generated from the `.jsonl` sources, with drift checked by
  `make validate`
- `tests/test_datasets.py` — 9 tests, including negative controls proving the validator fails on an
  invented quote and on a stale mirror
- `evals` added to the ruff and pyright paths

## What was built (Phase 0)

- `pyproject.toml` (Python 3.12 pinned), ruff + pyright config, `Makefile` (12 targets),
  `.env.example`, `README.md`, `.gitignore`
- `docker-compose.yml` — Postgres 16 + pgvector in the default profile; the Langfuse v4 stack
  (its own Postgres, ClickHouse, Redis, MinIO) behind an `observability` profile
- `src/sutra/config.py` — `Settings` + the checked-in `MODEL_PRICES` table
- `src/sutra/db/` — async engine, `models.py` (all 9 tables from ARCHITECTURE §12),
  Alembic wired to `Settings`, migration `0001_initial_schema`
- `src/sutra/llm/` — `types.py`, `tracing.py` (no-op until Phase 4), `adapters.py`
  (OpenAI Responses API), `client.py` (`complete` / `stream` / `structured`, retries, fallback
  chain, token + cost accounting)
- `src/sutra/prompts.py` — versioned loader with YAML front-matter
- `src/sutra/api/` — app factory, `/v1/health`, Pydantic schemas
- `tests/` — 41 tests across config, LLM client, prompt loader, health endpoint
- Stub READMEs for all 13 module folders (db, llm, api now written)

## Verified in this chat

- `make up` → Postgres healthy in ~6s; pgvector 0.8.6 available
- `make migrate` → applies cleanly; **round-trips** (9 tables → `downgrade base` → 0 → 9)
- `alembic revision --autogenerate` → **no drift** between `models.py` and the migration
- All 3 HNSW and 2 GIN indexes confirmed present via `pg_indexes`
- `/v1/health` → `200 {"status":"ok", ...}` against the real server, Postgres reachable
- `make lint` clean (ruff + ruff format + pyright, 0 errors) · `make test` → 41 passed
- Repo live, public, 12 topics, all commits authored by `rohhitrz` with no AI attribution

## Known issues / tech debt

0. **The corpus is frozen.** Editing `data/corpus/` invalidates every eval baseline from Phase 2
   on and can orphan one of the 72 cited quotes. If it must change: `make validate`, repair the
   quotes, re-run `make eval`, and record a new baseline saying why the old numbers no longer
   compare.

1. **No real LLM call has ever been made.** There is no `.env` yet and the client tests use a fake
   adapter. The OpenAI adapter's request shape is written against the installed SDK (3.35.1,
   Responses API) but is **unverified against the live API**. First real call happens in Phase 2 —
   expect to debug the adapter there, not the retrieval logic.
2. **The Langfuse stack was never started.** Only `docker compose config` was validated; the
   `observability` profile has not been booted, so image tags, env vars and the auto-provisioned
   `LANGFUSE_INIT_*` keys are unproven. Budget time for this in Phase 4.
3. **Fallback chain and model routing are structural no-ops.** Both roles point at one model
   (ADR-017), so `LLM_FALLBACK_MODELS` has nothing to fall back to. The code paths are tested, but
   only against a fake adapter with two synthetic model names.
4. `prompts/` is empty — `answer.v1.md` is Phase 2's job. The loader is tested against fixtures.
5. Two justified `ASYNC109` lint exceptions (`src/sutra/llm/adapters.py`, `tests/**`): the
   `timeout` parameter is delegated to the provider SDK, and the fake adapter must mirror the
   protocol. Reasons are recorded inline in `pyproject.toml`.
6. `requires-python = ">=3.12,<3.13"` while the host runs Python 3.13; uv fetches 3.12. Worth
   remembering if a dependency later needs 3.13.

## Repo
- GitHub: https://github.com/rohhitrz/sutra (public)

## Next step

**Phase 1 is done and the corpus is frozen** (ADR-014, notice in `data/corpus/README.md`).
Start **Phase 2 — Ingestion + naive RAG + eval harness** with the opening prompt in
`docs/BUILD_PLAN.md`.

Two things to do before the first retrieval code:

1. copy `.env.example` to `.env` and set a real `OPENAI_API_KEY` — **no live LLM call has been
   made in this project yet**, so budget time to debug the OpenAI adapter in Phase 2, not the
   retrieval logic (known issue 1);
2. `make ingest` has no implementation yet. It must skip `data/corpus/README.md`, which is the
   freeze notice and not a corpus document — the validator skips it the same way.

The eval harness has ground truth waiting for it: 42 golden rows over six categories and 40 attack
rows, with the trap-to-row map in `evals/datasets/README.md`.

---

## Session log
<!-- newest first: date — phase — 2–4 line summary -->

- **2026-10-19 — Phase 1 — done. Corpus frozen.**
  Wrote both golden datasets against the reserved ids in the trap map, and the validator that
  makes the freeze meaningful: all 72 cited quotes were found verbatim on the first run. Added
  `make validate`, seven tests including a negative control for the validator itself, and put
  `evals` under ruff and pyright. 48 tests pass, lint clean. No eval numbers yet — Phase 2 builds
  the harness that produces them.

- **2026-10-19 — Phase 1 — corpus written (steps 1–2).**
  World bible and trap map approved and committed, then all 10 corpus files written per SPEC §4,
  each 100–200 lines with the full front-matter. Checked mechanically: every ledger row's total is
  units × per-unit, all four deliberate holes are still holes, and all 8 planted injection/PII/wrong-claim
  strings are present and confined to file 10. Corpus not yet frozen — the datasets come first.

- **2026-10-19 — Phase 1 — step 1 written, awaiting review.**
  Designed "The Meridian Chronicles": 40 numbered waystones between Halvern Reach and Kesh Ithar,
  Book 1 in 1143 and Book 2 in 1144. Wrote `docs/world_bible.md` (world, cast, factions, timeline,
  the ten-file map) and `evals/datasets/README.md` (the trap map, with reserved `gq-`/`atk-` ids).
  Planted trap counts exceed SPEC §4: 5 Book 2 updates, 7 multi-hop facts, 3 injections, 4 fake PII
  items, 6 unanswerables. No corpus files written — the freeze waits on review.

- **2026-10-19 — Phase 0 — Foundation. Done.**
  Resolved the LLM provider open question (ADR-017: OpenAI `gpt-6-luna` for both roles, chosen on
  eval cost). Built the full foundation: uv project, compose stack, all 9 §12 tables with a
  round-trip-verified migration, the single LLM client with retries/fallback/costing, the versioned
  prompt loader, and `/v1/health`. 41 tests, lint clean, repo public and pushed.
  Carrying forward: no live LLM call made yet, and the Langfuse stack is unbooted.
