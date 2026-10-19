# Sutra — Progress

> Handoff file between chats. Read at the start of every chat. Update at the end of every chat.

## Current phase
**Phase 3 — Advanced retrieval + routing + semantic cache** · Status: ⬜ not started

## Phase status

| # | Phase | Status | Chat date |
|---|---|---|---|
| 0 | Foundation | ✅ done | 2026-10-19 |
| 1 | Corpus + golden datasets | ✅ done | 2026-10-19 |
| 2 | Ingestion + naive RAG + eval harness | ✅ done | 2026-10-20 |
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
| 2026-10-20 | **baseline** — `naive`, vector-only, k=5, answer.v1 | 61.9% | 0.492 | 0.964 | 0.774 | 2373 ms | $0.00050 |

Also from the baseline run (`evals/reports/baseline.md`): context precision 19.5%, hit rate@5
73.8%, refusal recall 100% (6/6), false refusal 11.1% (4/36), $0.0212 for the whole 42-row run.

Weakest categories by recall@5: multi-hop 37.5%, routing 44.4%. Strongest: numeric 86.1%.
Weakest by correctness: routing 0.33, multi-hop 0.62.

**Two caveats that govern how this scoreboard may be read.**

1. **Judge scores carry ±0.05 run-to-run noise.** Three runs on identical configuration gave
   faithfulness 0.940 / 0.964 / 0.988 and correctness 0.726 / 0.774 / 0.774. Retrieval metrics
   are exact and repeated to the digit every time. A Phase 3 change that moves a judge score by
   0.02 has moved nothing measurable; a change that moves recall by 0.02 has.
2. **The faithfulness judge is demonstrably unreliable in the direction that matters** — see
   known issue 2. Treat 0.964 as an upper bound, not a measurement.

**Caveat to carry forward (ADR-017):** `LLM_STRONG` is also the judge model, so judge scores are
same-family self-evaluation. Read faithfulness and correctness as **relative** — valid for
comparing profiles against each other, not as ground truth.

## Decisions taken in chats

- **ADR-020** — `fastembed` (ONNX Runtime) is the local embedding runtime, not
  `sentence-transformers`. ADR-004's *models* are unchanged; only the engine differs, so vectors
  and eval baselines are unaffected. `sentence-transformers` requires PyTorch, a full training
  framework, where Sutra needs one CPU forward pass over a frozen 67 MB model. Measured `.venv`
  growth: **141 MB** via ONNX against roughly 500 MB via PyTorch. Verified before accepting that
  `bge-small-en-v1.5` is supported at dim 384 with L2-normalised output (which is what makes
  `vector_cosine_ops` correct) and that `bge-reranker-base` is a supported cross-encoder, so
  Phase 3 needs no second runtime. Amends ADR-018 and the CLAUDE.md stack line.
- **ADR-021** — `index_version` is an integer **slot** chosen by config or an eval profile, with
  the chunking fingerprint stored in `chunks.metadata`. Resolves a three-way contradiction:
  ARCHITECTURE §3 described a hash, the §12 schema declares an Integer column, `config.py` held a
  plain int. Same fingerprint in the slot is a no-op, a different one is refused with the changed
  parameters named, a different slot is kept alongside.
- **ADR-022** — amends ADR-021 on two points found by building it. The fingerprint lives in
  `chunks.metadata` **only** (a `documents` row is shared by every slot, so a per-slot fingerprint
  there would be ambiguous), and it includes **`chunker_version`**, because parameters alone
  cannot say whether the code applying them changed. Not hypothetical: fixing the heading scan
  mid-phase changed the output for all ten documents while leaving the fingerprint identical, so
  `make ingest` reported "nothing to do" over stale chunks.
- **ADR-017** — LLM provider is **OpenAI**, `gpt-6-luna` for *both* `LLM_FAST` and `LLM_STRONG`.
  Verified rates: $0.10 / $0.50 per 1M input / output, $0.01 cached. Chosen on cost, because eval
  volume (~250 calls per `make eval`, re-run after every retrieval change) dominates spend, not
  chat traffic. Closes the Phase 0 open question. Declined a separate `LLM_JUDGE` role.
- **ADR-018** — recorded the concrete runtime dependency set, since ADR-015 covered only the dev
  toolchain. Dev-only `httpx2` (not `httpx`) because the `openai` SDK is built on httpx2.
- **ADR-019** — `make eval` is exempt from the ">200 LLM calls needs explicit OK" rule, which one
  eval run exceeds by construction.

## What was built (Phase 2)

- `src/sutra/ingestion/` — `parse.py` (front-matter), `chunk.py` (hand-rolled recursive chunker
  with three guaranteed invariants), `embed.py` (bge-small over ONNX behind an `Embedder`
  Protocol), `fingerprint.py` (ADR-021/022), `run.py` (`make ingest`), `README.md`
- `src/sutra/retrieval/search.py` — vector-only top-k within one index slot, plus `README.md`
  naming every absent technique and the phase it arrives in
- `prompts/answer.v1.md` and the three judge prompts, all split by a `<<<USER>>>` delimiter so
  the stable half stays uninterpolated
- `src/sutra/generation/` — `schema.py` (`AnswerDraft`, `Citation`, `Answer`), `answer.py`
  (context assembly, structured call, citation resolution), `demo.py` (one question end to end),
  `README.md`
- `src/sutra/api/` — `chat.py` (the turn: resolve session → save question → retrieve → generate
  → save answer, with **no DB connection held across the model call**), `routes_chat.py`,
  chat schemas, `README.md`
- `evals/metrics/` — `retrieval.py` (recall@k, hit rate, MRR, context precision),
  `judges.py` (three versioned LLM judges), `grounding.py` (the mechanical figure check)
- `evals/runners/` — `run_rag.py`, `report.py`, `run.py` (`make eval`); `evals/profiles/naive.yaml`
- `evals/check_coverage.py` + `make coverage` — asserts every cited quote survives chunking
- `evals/reports/baseline.json` and `baseline.md`, **committed** as the reference for later runs
- `evals/README.md` — the metric table and what each one does not tell you
- Config: chunking parameters, `retrieval_top_k`, judge prompt versions, and a
  `MODEL_CAPABILITIES` table beside `MODEL_PRICES`
- Tests: **156 passing** (up from 48), including both negative controls on the coverage guard

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

- `make ingest` → 49 chunks from 10 documents, 384-dim embeddings and tsvectors on all of them;
  tsv stemming confirmed (`to_tsquery('waystones')` matches 34 chunks)
- **ADR-021/022 behaviours all exercised live:** re-run is a no-op by fingerprint; changed
  parameters into an occupied slot are refused *naming what changed*; a second slot coexists;
  `--force` replaces; a fingerprint from an older schema is refused rather than crashing
- `make coverage` → **72/72 cited quotes land whole inside a chunk** (11 in two chunks, which is
  the overlap working). Both negative controls fail as they should
- Retrieval: stored vectors verified against fresh embeddings (`cos = 1.000000`), and a chunk's
  own text retrieves that chunk at similarity 1.0
- **First live LLM calls ever made in this project.** The OpenAI adapter works end to end:
  structured output parsed, citations resolved, tokens and cost accounted
- `POST /v1/chat` against a real uvicorn server: session created and continued, both messages
  persisted with `prompt_version`, a foreign user's `session_id` returns 404
- `make eval` → the full 42-row baseline, all three judges scoring every row, $0.0212
- `make test` → 156 passed · `make lint` clean (ruff + ruff format + pyright, 0 errors)
- All commits authored by `rohhitrz`, no AI attribution, pushed to `origin main`

## Known issues / tech debt

0. **The corpus is frozen.** Editing `data/corpus/` invalidates every eval baseline from Phase 2
   on and can orphan one of the 72 cited quotes. If it must change: `make validate`,
   `make coverage`, repair the quotes, re-run `make eval`, and record a new baseline saying why
   the old numbers no longer compare.

1. **`make ingest` does not notice a chunker code change on its own.** `chunker_version`
   (ADR-022) is bumped by hand. Forget it and ingestion reports "nothing to do" over stale
   chunks. Deriving it from the source would fire on every comment edit, so the manual signal is
   deliberate — but it is a thing you can forget. **Bump it, or use `--force`, after any change
   to chunking behaviour.**

2. **The faithfulness judge cannot be trusted at face value.** ADR-017 points the judge and the
   answer model at one model, and the consequence is not theoretical: on the baseline it scored
   **faithfulness 1.00** on an answer whose central figure (`68,000`) appeared in none of its
   five passages, reasoning that "the passage states that Kesh Ithar has 68,000 people". It had
   anchored on a date that *was* present and approved the rest. The mechanical figure check
   (`evals/metrics/grounding.py`) is the deterministic floor under it. **Phase 3 should write
   `judge_faithfulness.v2` — requiring the judge to quote the supporting span — and validate the
   judges against the ~10 hand-labelled rows ARCHITECTURE §11 asks for.** Until then read
   faithfulness as an upper bound.

3. **Judge scores carry ±0.05 run-to-run noise** (measured: three identical runs gave
   faithfulness 0.940/0.964/0.988, correctness 0.726/0.774/0.774). Retrieval metrics are exact.
   Do not read a small judge delta as a result.

4. **`POST /v1/chat` has no authentication.** `SUTRA_API_KEY` exists in config and nothing checks
   it. Fine for a local dev server; belongs with the Phase 6 guardrail work. Also no rate
   limiting, and no streaming (the response is buffered JSON; `LLMClient.stream()` already
   exists and ARCHITECTURE §5 specifies the SSE event shape).

5. **No token budget.** `MAX_ANSWER_TOKENS` is a flat cap. ARCHITECTURE §5's real budget — system
   + memory + history + chunks + answer reserve within the model window, with a defined trim
   order — is Phase 5. At k=5 and 400-token chunks nothing comes near the window, so this is a
   gap by schedule rather than oversight.

6. **The HNSW index is not exercised.** With 49 chunks the planner filters on `index_version` and
   sorts instead (confirmed with `EXPLAIN`). The index is correct and will matter at volume, but
   nothing here tests HNSW's approximate-recall behaviour.

7. **Prompt caching does not pay yet.** The stable system block is ~425 tokens and OpenAI caches
   only prefixes above a 1024-token minimum. The ordering is the precondition, not a saving; it
   starts paying once memory and guardrail instructions grow the stable block (Phases 5–6).

8. **The Langfuse stack has never been started.** Only `docker compose config` was validated, so
   image tags, env vars and the auto-provisioned `LANGFUSE_INIT_*` keys are unproven. Budget time
   in Phase 4.

9. **Fallback chain and model routing are still structural no-ops.** Both roles point at one
   model (ADR-017), so `LLM_FALLBACK_MODELS` has nothing to fall back to.

10. **Two provider quirks are now encoded, and worth remembering.** `gpt-6-luna` rejects
    `temperature` outright (hence `MODEL_CAPABILITIES`), and `max_output_tokens` budgets
    *reasoning and visible output together* — a cap sized for the visible answer gets eaten by
    reasoning and the truncated body arrives as "Invalid JSON". `OpenAIAdapter` now detects that
    and names the real cause; `Usage.reasoning_tokens` records where the spend went.

11. Two justified `ASYNC109` lint exceptions (`src/sutra/llm/adapters.py`, `tests/**`), reasons
    recorded inline in `pyproject.toml`.

12. `requires-python = ">=3.12,<3.13"` while the host runs Python 3.13; uv fetches 3.12.

13. `docs/BUILD_PLAN.md` has uncommitted whitespace-only changes (blank lines around headings)
    from an editor's markdown formatter, not from any phase work. Left alone deliberately.

## Repo
- GitHub: https://github.com/rohhitrz/sutra (public)

## Next step

**Phase 2 is done.** `make ingest && make eval` produces the committed baseline at
`evals/reports/baseline.json`, and its numbers are in the scoreboard above.

Start **Phase 3 — Advanced retrieval + routing + semantic cache** with the opening prompt in
`docs/BUILD_PLAN.md`. The baseline says where the room is:

- **multi-hop recall@5 is 37.5% and routing is 44.4%** — the two weakest categories, and between
  them the clearest case for hybrid search and query decomposition
- **context precision is 19.5%**: roughly four of every five retrieved chunks carry no cited
  evidence, which is what reranking is for
- **correctness on routing rows is 0.33** against 0.88 on single-hop
- dense-only retrieval is weak at exact-token recall, and the corpus is full of rare proper nouns
  and exact figures on purpose, so the hybrid delta should be visible rather than theoretical

Two things to do before the Phase 3 retrieval work:

1. **Decide whether `judge_faithfulness.v2` comes first** (known issue 2). Phase 3 is measured by
   judges that are currently provable rubber-stampers, so improving them may be a prerequisite
   rather than a follow-up.
2. Remember that a new chunking scheme needs a **new index slot** (`INDEX_VERSION=2`), not
   `--force` on slot 1 — slot 1 is what the committed baseline was measured against.

---

## Session log
<!-- newest first: date — phase — 2–4 line summary -->

- **2026-10-20 — Phase 2 — done. Baseline recorded.**
  Built ingestion, the vector-only retrieval baseline, structured generation with citations,
  `POST /v1/chat`, and the eval harness; made the project's first live LLM calls and debugged two
  provider quirks (`temperature` unsupported, and `max_output_tokens` being shared with
  reasoning). Baseline: recall@5 61.9%, MRR 0.492, correctness 0.774, refusal recall 100%,
  $0.0212 a run. The finding that matters: the faithfulness judge scored 1.00 on an answer whose
  central figure was in none of its passages, so a deterministic figure check now sits beside it
  and judge scores are an upper bound until `judge_faithfulness.v2`. ADR-020/021/022 recorded.
  156 tests, lint clean.

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
