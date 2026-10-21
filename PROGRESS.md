# Sutra — Progress

> Handoff file between chats. Read at the start of every chat. Update at the end of every chat.

## Current phase
**Phase 3 — Advanced retrieval + routing + semantic cache** · Status: 🟨 in progress
(**steps 1–3 of 7 done**; step 4 — cross-encoder rerank — next)

## Phase status

| # | Phase | Status | Chat date |
|---|---|---|---|
| 0 | Foundation | ✅ done | 2026-10-19 |
| 1 | Corpus + golden datasets | ✅ done | 2026-10-19 |
| 2 | Ingestion + naive RAG + eval harness | ✅ done | 2026-10-20 |
| 3 | Advanced retrieval + routing + semantic cache | 🟨 | 2026-10-20 |
| 4 | Observability | ⬜ | |
| 5 | Generation, context management, memory | ⬜ | |
| 6 | Guardrails | ⬜ | |
| 7 | Agentic RAG + MCP server | ⬜ | |
| 8 | Voice agent | ⬜ | |
| 9 | CI evals, hardening, CONCEPTS.md, ablation report | ⬜ | |

Legend: ⬜ not started · 🟨 in progress · ✅ done

## Eval scoreboard
(Filled from Phase 2 onward. One row per eval run that matters.)

| Date | Config / change | Recall@5 | **Context recall** | MRR | Faithfulness | Answer correctness | p50 latency | Cost/query |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | ~~`naive` baseline, as first recorded~~ — **superseded** | 61.9% | — | 0.492 | 0.964 | 0.774 | 2373 ms | $0.00050 |
| 2026-10-21 | **baseline** — `naive`, vector-only, k=5, answer.v1, slot 1 | 61.9% | 61.9% | 0.492 | 0.976 | 0.774 | 2734 ms | $0.00056 |
| 2026-10-21 | **step 1** — `semantic`, parent-child + expansion, slot 2 | 63.9% | 76.2% | 0.437 | 0.988 | 0.774 | 2747 ms | $0.00056 |
| 2026-10-21 | **step 2** — `contextual`, + generated prefixes, slot 3 | 69.4% | 79.8% | 0.506 | 0.952 | 0.810 | 2529 ms | $0.00086 |
| 2026-10-21 | ablation — `keyword_only`, FTS alone, slot 3 | 61.9% | 68.3% | 0.471 | 1.000 | 0.774 | 2860 ms | $0.00092 |
| 2026-10-21 | **step 3** — `hybrid`, vector + FTS fused by RRF, slot 3 | **73.0%** | **81.0%** | **0.528** | 0.964 | 0.810 | 2572 ms | $0.00085 |

All three rows above were produced by the **same harness code on the same day**, which earlier
rows were not. That re-run was forced by a bug found in step 2 (the figure check was flagging
comma-terminated figures — see below), and it is the reason the baseline's figure count and
judge scores differ from the 2026-10-20 recording.

**Context recall is the column to read across chunking schemes**, and it is new in step 1.
Recall@5 is measured on the retrieved units, so it moves with their size — slot 2's children
are 200 tokens against slot 1's 400, so the same retriever fetches less text at the same k
whatever its quality. Context recall is measured on the passages actually sent to the
generator. On the flat baseline the two are identical by construction, which is why that row
can carry the number without being re-run.

Also from the current baseline run (`evals/reports/baseline.md`): context precision 19.5%,
hit rate@5 73.8%, refusal recall 100% (6/6), false refusal 11.1% (4/36), 4 rows asserting an
unsupported figure, $0.0236 for the whole 42-row run at 52% prefix-cache hits.

**Why the baseline was re-recorded.** Three defects in the eval harness were fixed (see the
review-fix section below), two of which changed what the report says: token totals counted judge
calls only, and the figure check cleared a fabrication whenever its digits appeared inside a
longer number. The system under measurement did not change. The proof is that **recall@5, MRR and
context precision are identical to the digit across both rows** — those metrics are exact and
depend only on the index, which was verified byte-identical (same 49 chunks, same text, same
token counts).

The judge and latency columns moved for ordinary run-to-run reasons, not because of the fixes.

### Step 1 — semantic chunking + parent-child (slot 2, `semantic.yaml`)

**Verdict: kept, and carried into step 2.** Context recall 61.9% → **76.2% (+14.3pp)**, and it
did that while sending the generator *fewer* passages — 4.0 against 5.0, because children of one
parent dedupe. That is the parent-child thesis working exactly as advertised: search a precise
unit, read a complete one.

Context recall by category, the comparable view:

| Category | naive | step 1 | |
|---|---|---|---|
| single-hop | 58.3% | 83.3% | +25.0 |
| multi-hop | 37.5% | 60.4% | +22.9 |
| unanswerable | 83.3% | 100.0% | +16.7 |
| temporal | 64.3% | 78.6% | +14.3 |
| routing | 44.4% | 44.4% | — |
| **numeric** | **86.1%** | **72.2%** | **−13.9** |

Multi-hop was the weakest category in the baseline and gained the most, which is the result
that justifies the step.

**The one real regression is numeric, and it is not an artefact** — it is −13.9pp on the
comparable metric too, and MRR fell 0.492 → 0.437 overall. More, shorter units make a denser
field of near-identical candidates, and the merchant ledger is the worst case because its rows
genuinely do all resemble each other. Dense retrieval cannot distinguish `946` from `1,034` at
all. **This is precisely what step 3's keyword half is for, so numeric recall is the number to
watch there.** If hybrid does not recover it, the parent-child child size is the thing to revisit.

Judges moved within the noise band and should not be read. Cost rose ~12% cache-neutral, from
longer context passages.

### Step 2 — contextual retrieval prefixes (slot 3, `contextual.yaml`)

**Verdict: kept, and the best configuration so far.** A fast model writes 1–2 sentences per
child saying what the passage is and where it sits; that prefix is prepended before embedding
and before the tsvector, while the stored chunk text stays verbatim.

It is the strongest single result in the project so far, and it **repaired step 1's
regressions rather than trading against them**:

| | naive | step 1 | step 2 |
|---|---|---|---|
| Context recall | 61.9% | 76.2% | **79.8%** |
| Recall@5 | 61.9% | 63.9% | **69.4%** |
| MRR | 0.492 | 0.437 | **0.506** |
| Hit rate@5 | 73.8% | 73.8% | **81.0%** |
| Context precision | 19.5% | 17.6% | 19.5% |
| Passages to model | 5.0 | 4.0 | **3.6** |
| False refusals | 4 | 4 | **3** |

**MRR recovered past the baseline** (0.437 → 0.506), and **numeric context recall came all the
way back**: 86.1% → 72.2% → 86.1%. Step 1's notes predicted step 3's keyword half would be
needed for that; the prefix did it instead, by giving a bare ledger row back the words naming
what it is a row of. Routing — the second-weakest baseline category — is up 22.2pp on context
recall.

Context recall by category, step 2 against the baseline: single-hop +33.3, routing +22.2,
unanswerable +16.7, temporal +14.3, multi-hop +10.4, numeric unchanged.

**The one trade-off: multi-hop fell back from step 1's 60.4% to 47.9%** (still +10.4 over
baseline). The obvious hypothesis — prefixes pull every chunk toward its document's general
topic, concentrating results in fewer documents — is only half supported: mean distinct
documents per row does drop (3.40 → 2.88), but on multi-hop rows specifically it barely moves
(3.50 → 3.38). So the cause is **not established**. Worth revisiting at step 4 (rerank) and
step 5 (multi-query), both of which bear directly on it.

**Cost: +6% cache-neutral** ($0.0348 → $0.0368 at list rate), plus a one-off ingestion cost of
104 fast-model calls. Re-ingestion is free: prefixes are cached by a hash of prompt version,
document and chunk text.

### Step 3 — keyword FTS + RRF hybrid (slot 3, `hybrid.yaml`)

**Verdict: kept. The best configuration so far, and best on every headline metric** — context
recall 81.0%, recall@5 73.0%, MRR 0.528, hit rate 83.3%, context precision 20.5% (the only
profile to beat the baseline on precision). Cost is flat against step 2 (+3% cache-neutral),
latency unchanged.

**The ablation is what makes the row mean something.** `keyword_only` is a worse retriever
overall (68.3% context recall against step 2's 79.8%) — nobody should ship it. But it is not
uniformly worse, and that is the entire basis of the fusion:

| Context recall | naive | step 2 | keyword only | **step 3** |
|---|---|---|---|---|
| routing | 44.4% | 66.7% | **100.0%** | **100.0%** |
| temporal | 64.3% | 78.6% | **92.9%** | **92.9%** |
| numeric | 86.1% | **86.1%** | 72.2% | **86.1%** |
| single-hop | 58.3% | **91.7%** | 66.7% | **91.7%** |
| unanswerable | 83.3% | **100.0%** | 50.0% | 83.3% |
| multi-hop | 37.5% | 47.9% | 47.9% | 41.7% |

Hybrid takes the lexical half's wins on routing and temporal **and** keeps the dense half's
wins on numeric and single-hop. That is RRF doing exactly what it is for, and it is only
visible because the ablation was run.

**Two things it did not fix, stated plainly.**

1. **Multi-hop is now the clear outstanding problem, and it has got worse with every step
   since step 1**: 37.5% → 60.4% (step 1) → 47.9% (step 2) → 41.7% (step 3). Step 1's larger
   deduped parents remain the best thing this project has done for multi-hop, and the two
   techniques since have each eroded it while winning elsewhere. No explanation has survived
   checking. **This is the thing step 5's multi-query decomposition has to address**, and it
   should be judged on this number specifically.
2. **Unanswerable fell 100% → 83.3%.** OR-ed keyword terms always match *something*, so rows
   that should retrieve nothing now retrieve plausible-looking passages. Refusal recall stayed
   at 100% and false refusals actually improved (4 → 3), so the model is still refusing
   correctly — but the guardrail is the model's judgement rather than an empty result set,
   which is a weaker place for it to live. Worth remembering in Phase 6.

Weakest categories by recall@5 in the baseline: multi-hop 37.5%, routing 44.4%. Strongest:
numeric 86.1%.
Weakest by correctness: routing 0.33, multi-hop 0.62.

**Three caveats that govern how this scoreboard may be read.**

1. **Judge scores carry run-to-run noise, and the band is wider than ±0.05.** Five runs on
   identical configuration have now given faithfulness 0.940 / 0.952 / 0.964 / 0.988 / 1.000 and
   correctness 0.726 / 0.762 / 0.774 / 0.774 / 0.821. Retrieval metrics are exact and repeat to
   the digit every time. A Phase 3 change that moves a judge score by 0.05 has moved nothing
   measurable; a change that moves recall by 0.02 has.
2. **The faithfulness judge is demonstrably unreliable in the direction that matters** — see
   known issue 2. The current run makes this as stark as it gets: the judge returned
   **faithfulness 1.000 on all 42 rows** while the deterministic figure check flagged
   **8 of those same rows** as asserting a figure that appears nowhere in their context. Treat
   1.000 as an upper bound, and specifically as an upper bound that is provably wrong here.

3. **Cost per run swings ~50% on prefix-cache warmth alone.** Three runs over the same index
   doing the same work, with input-token counts within 0.1% of each other, cost $0.0212,
   $0.0348 and $0.0232 — the difference being that the provider served 0% to 51.2% of input
   tokens from its cache depending on how recently the last run finished. The report now records
   `total_cached_tokens` so a cost delta can be checked before it is read as a result. **Do not
   put a cost delta in a Phase 3 scoreboard row without the cached share beside it.**

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

## Review fixes (2026-10-20, before Phase 3 step 1)

An external code review raised seven findings. All seven reproduced; one detail in the report
was wrong (the token counter saturates at 512, not at 3). Fixed in six commits, each with a
test, lint and types clean. Three of the fixes were verified as negative controls — the test
was confirmed to fail with the fix reverted, which for the slot bug meant reproducing the exact
`uq_chunk_position` violation the review predicted.

| Finding | Fix |
|---|---|
| `complete()`/`structured()` ignored `model=` and answered from `LLM_STRONG` | model threaded through `_call_with_retries`; an explicit model now gets a chain of one |
| eval token totals summed judge calls while cost covered answer calls too | totals cover both; answer/judge split kept beside them |
| `count_tokens` saturated at 512, so the truncation guard could never fire | counting runs on a truncation-free copy of the tokenizer |
| figure check credited `143` to `1143`, disarming the cross-check on the judge | support tested against the set of figures the context states |
| a slot holding chunks without `index_fingerprint` read as empty, skipping the delete | occupancy decided by row existence, not fingerprint readability |
| blocking ONNX `embed_query` on the event loop in `/v1/chat` | awaited in a thread; test asserts on thread identity, not timing |
| `--limit` with `--baseline` overwrote the committed reference | the CLI refuses the combination |

Two things worth carrying forward that the review did not ask for:

- **The chunking fix is a no-op on the current corpus, verified rather than assumed.** Slot 1
  re-chunks to the same 49 chunks with identical text and token counts, so the committed
  baseline's index still stands and no re-ingest was needed. At `chunk_tokens=400` any piece
  reporting 512 was already over the split threshold. It starts mattering in Phase 3, where
  semantic chunking makes chunk size data-dependent.
- **Correcting the token totals exposed a second problem** — cost swinging 50% on prefix-cache
  warmth with nothing in the report able to explain it. `total_cached_tokens` was added, and
  the reported cost now reconciles to the cent against the cached rate.

The blocking-embedder fix matters more from Phase 3 on than it did here: the cross-encoder
reranker is far heavier than the embedder and lands on the same event loop. `embed_passages` in
ingestion is still synchronous and deliberately so — it runs in a CLI script, not a request.

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

   **Still true, but the earlier statement of it was overstated.** PROGRESS previously said
   the judge returned 1.000 while the figure check flagged 8 of 42 rows. The 8 was inflated by
   a bug in the check itself (it was flagging comma-terminated figures, fixed 2026-10-21); the
   true count on the baseline is **4 of 42**, against faithfulness 0.976. A judge scoring 0.976
   on a set containing four answers that assert a figure from nowhere is still a judge not
   doing its job — but it is a weaker statement than the one recorded before, and the
   difference was a defect in our own instrument, not in the model.

   The figure check has now been wrong in both directions within two days: first clearing
   fabrications whose digits sat inside a longer number, then flagging real figures followed by
   a comma. Both are fixed and both have tests. Treat its absolute counts with the same
   suspicion as the judge's until `judge_faithfulness.v2` and the hand-labelled rows exist.

3. **Judge scores carry run-to-run noise of at least ±0.05** (measured over five identical runs:
   faithfulness 0.940–1.000, correctness 0.726–0.821). Retrieval metrics are exact. Do not read
   a judge delta smaller than the band as a result.

3b. **Cost per run is not a stable measurement.** It swings ~50% on how warm the provider's
   prefix cache is (0% to 51.2% of input tokens served cached across three runs doing identical
   work). `total_cached_tokens` is now in the report; quote it alongside any cost number.

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

**Phase 2 is done and the review fixes are in.** `make ingest && make eval` produces the
committed baseline at `evals/reports/baseline.json`, and its numbers are in the scoreboard
above. 168 tests, lint and types clean, everything pushed.

**Steps 1–3 are done and kept** (see the scoreboard). `hybrid.yaml` is the current best
profile and the base for step 4 under the carry-forward rule.

**The immediate next step is Phase 3 step 4: cross-encoder reranking** —
`bge-reranker-base` (ADR-004) over the top ~20 fused candidates, keeping the top 5
(ARCHITECTURE §4 step 5). What it needs:

- `retrieval/rerank.py`, with the model behind an interface like `Embedder` is, and **run off
  the event loop** — a cross-encoder is far heavier than the embedder, which is why
  `vector_search`'s `asyncio.to_thread` fix was made before this step rather than after.
- The candidates to rerank are the *fused* list, so `hybrid_search` needs to return its wider
  pool rather than cutting to k before the reranker sees it.
- Expect real latency: 20 cross-encoder passes per query on CPU is the most expensive thing
  in the pipeline so far. Record p50 honestly against hybrid's 2572 ms.

**Do `judge_faithfulness.v2` first.** This was deferred on the explicit grounds that steps 1–3
were judged on exact retrieval metrics. Step 4 is the first step whose case may rest on answer
quality rather than recall, so the judge has to be worth reading before it is used to decide
anything. That means: the v2 prompt requiring a quoted supporting span, plus the ~10
hand-labelled rows ARCHITECTURE §11 asks for, validated against both judge versions.

**And judge step 4 on multi-hop specifically** — see the step 3 notes. It is the one category
that has degraded across the phase.

Settled, and not to be re-litigated:

- **No migration is needed.** `chunks.parent_id` and `chunks.context_prefix` both exist from
  the Phase 0 schema. Step 1 used the first; step 2 uses the second.
- **Carry-forward rule.** Each step is evaluated against the current best profile. A step that
  does not win gets its scoreboard row but is *not* carried into the next step's base.
- **`judge_faithfulness.v2` is deferred** until before step 4 (rerank), where the case for a
  change starts to rest on answer quality rather than on exact retrieval metrics.
- **Compare chunking schemes on `context_recall`, rankers on `recall@k` and MRR.** Step 1 added
  the metric for this reason; the baseline's value is exact, not estimated, because a flat
  index with no expansion makes the two identical by construction.
- **`make coverage` checks search units and context units separately** on a parent-child index,
  and both must pass. Slot 2: 72/72 on each.

The baseline says where the room is:

- **multi-hop recall@5 is 37.5% and routing is 44.4%** — the two weakest categories, and between
  them the clearest case for hybrid search and query decomposition
- **context precision is 19.5%**: roughly four of every five retrieved chunks carry no cited
  evidence, which is what reranking is for
- **correctness on routing rows is 0.33** against 0.88 on single-hop
- dense-only retrieval is weak at exact-token recall, and the corpus is full of rare proper nouns
  and exact figures on purpose, so the hybrid delta should be visible rather than theoretical

And one thing that has not changed: a new chunking scheme needs a **new index slot**
(`INDEX_VERSION=2`), not `--force` on slot 1 — slot 1 is what the committed baseline was
measured against.

---

## Session log
<!-- newest first: date — phase — 2–4 line summary -->

- **2026-10-21 — Phase 3 — step 3 done: keyword FTS + RRF hybrid.**
  `keyword_search` over the existing tsvector, `fusion.py` (RRF, k=60, pure and exactly
  testable), and a `search()` entry point dispatching on `retrieval_mode` so the halves can be
  ablated against the whole. Best configuration so far on every headline metric: context
  recall 81.0%, recall@5 73.0%, MRR 0.528, hit rate 83.3%, context precision 20.5% — and cost
  and latency flat. The `keyword_only` ablation is what makes it meaningful: a worse retriever
  overall (68.3%) that is *better* on routing (100%) and temporal (92.9%), and hybrid keeps
  both of those while retaining the dense half's numeric and single-hop. Caught before
  trusting it: `websearch_to_tsquery` ANDs its terms, so 21 of 42 questions returned zero
  keyword hits and the fusion would have been inert on half the eval — OR-ing the terms took
  that to 0. Two open problems recorded rather than smoothed over: multi-hop has now degraded
  at every step since step 1 (60.4% → 47.9% → 41.7%) and is step 5's job, and unanswerable
  fell 100% → 83.3% because OR-ed terms always match something. 205 tests, lint clean.

- **2026-10-21 — Phase 3 — step 2 done: contextual retrieval prefixes.**
  A fast model writes 1–2 sentences per chunk situating it in its document, prepended before
  embedding and tsvector while the stored text stays verbatim; cached by content hash so
  re-ingestion stays free. **The best result in the project so far**, and it repaired step 1's
  regressions instead of trading against them: context recall 76.2% → 79.8%, recall@5 63.9% →
  69.4%, MRR 0.437 → **0.506** (past the baseline), hit rate 73.8% → 81.0%, on 3.6 passages per
  row. Numeric context recall came all the way back to 86.1% — step 1 predicted step 3's
  keyword half would be needed for that, and the prefix did it first. Multi-hop fell back from
  step 1 (60.4% → 47.9%) and the obvious explanation does not hold up, so it is recorded as
  unexplained. Two bugs found and fixed on the way: the figure check was flagging every
  comma-terminated figure (which had inflated the baseline's count from 4 to 8, and the
  known-issue-2 claim with it), and one row's answer call died of reasoning eating the whole
  token budget, taking the other 41 rows with it. All three scoreboard rows were then re-run
  on identical harness code. 191 tests, lint clean.

- **2026-10-21 — Phase 3 — step 1 done: semantic chunking + parent-child.**
  Built semantic chunking (sentence embeddings, percentile breakpoints), parent-child chunks
  into index slot 2, and `retrieval/expand.py`. Slot 1 verified untouched — `make ingest` still
  reports "nothing to do", which the new fingerprint fields were designed to preserve. The
  headline: **context recall 61.9% → 76.2%** while sending the generator *fewer* passages
  (4.0 vs 5.0), with multi-hop — the baseline's weakest category — up 22.9pp. Had to add
  `context_recall` first, because recall@k is not comparable across chunk sizes and would have
  measured the chunking rather than the retrieval. One real regression: numeric recall
  −13.9pp, dense retrieval being unable to tell `946` from `1,034`; step 3's keyword half is
  the test of whether that recovers. A mistake worth remembering: cutting at every semantic
  breakpoint gave 67 parents averaging 290 tokens against an 800 budget — breakpoints have to
  be permitted cuts, not required ones. 183 tests, lint clean.

- **2026-10-20 — Phase 3 — review fixes, baseline re-recorded. No Phase 3 code yet.**
  An external review raised seven findings; all seven reproduced and all seven are fixed, in six
  commits with tests, three of them verified as negative controls. Two were corrupting the
  measurements themselves (eval token totals counted judge calls only while cost counted both;
  the figure check cleared any fabrication whose digits sat inside a longer number, which on a
  corpus set in 1143/1144 is most of them), so the baseline was re-recorded. Retrieval metrics
  came back identical to the digit — 61.9% / 0.492 / 19.5% — which is the proof that the fixes
  changed the measuring and not the system. Correcting the token totals then exposed a problem
  nobody had flagged: run cost swings ~50% on prefix-cache warmth alone, with nothing in the
  report able to explain it, so `total_cached_tokens` was added and cost now reconciles to the
  cent. The baseline's faithfulness came back **1.000 across all 42 rows** while the repaired
  figure check flags 8 of them — the sharpest evidence yet that the judge is a rubber stamp.
  168 tests, lint clean.

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
