# Sutra — Build Plan

> **One phase = one new chat.** Paste the phase's opening prompt. Claude Code proposes a plan, I approve, it builds,
> then updates PROGRESS.md. Big phases may take 2 chats; continue from PROGRESS.md.
>
> **Every phase ends with:** tests green, `make lint` clean, module README(s) written, PROGRESS.md updated, changes committed and pushed.

---

## Phase 0 — Foundation
**Read:** SPEC §1–2, §5 · ARCHITECTURE §1, §12, §15 · DECISIONS ADR-001, 002, 005, 007, 015

**Build**
- `uv` project, `pyproject.toml`, ruff + pyright config, `Makefile`, `.env.example`
- `docker-compose.yml`: Postgres + pgvector, Langfuse stack (the api service comes later)
- `sutra/config.py` (pydantic-settings), `db/` (async SQLAlchemy engine + Alembic, initial migration with **all §12 tables**)
- `llm/client.py` skeleton: one provider adapter, `complete()` + `stream()` + `structured()`, retries, timeout, fallback chain, token + cost accounting (price table in config). Tracing hook is a no-op until Phase 4.
- `prompts/` loader (front-matter + version), FastAPI app with `/v1/health`
- Repo folders from the CLAUDE.md layout, each with a stub README
- Git + public GitHub repo setup per CLAUDE.md "Git & GitHub" (`.gitignore`, `gh repo create`, description, topics, first push)

**Done when:** `make up` starts the stack, `/v1/health` is green, the migration applies, a test calls the LLM client once (mockable), and the repo is live on GitHub with the Phase 0 commits pushed.

**Opening prompt**
```
We're starting Sutra, Phase 0 — Foundation.
Read CLAUDE.md, PROGRESS.md, and the Phase 0 section of docs/BUILD_PLAN.md, plus the doc sections it lists.
First resolve the open question in DECISIONS (LLM provider): give me 2–3 options with cost/quality tradeoffs and wait.
Then propose a plan for the phase and wait for my OK.
```

---

## Phase 1 — Corpus + golden datasets
**Read:** SPEC §3–4 · DECISIONS ADR-014

**Build**
- The universe "The Meridian Chronicles": a short **world bible** first (characters, places, factions, timeline, which facts get updated in Book 2, the planted traps), for my review
- 10 files in `data/corpus/` per SPEC §4, with YAML front-matter
- `evals/datasets/golden_qa.jsonl` (~40 rows): `{id, question, answer, category, source_chunks: [doc_slug + quote], must_refuse}`
- `evals/datasets/attacks.jsonl` (~30 + ~10 benign): `{id, input, category, expected: block|allow}`
- `evals/datasets/README.md`: the trap map (which question tests which trap)
- A validator script: every golden answer's supporting quote really exists in the corpus

**Done when:** the validator passes, and I've reviewed the world bible and the trap map. Then the corpus is **frozen**.

**Opening prompt**
```
Sutra, Phase 1 — Corpus + golden datasets.
Read CLAUDE.md, PROGRESS.md, Phase 1 in docs/BUILD_PLAN.md and the listed sections.
Step 1: write only the world bible (docs/world_bible.md) and the trap map, then stop for my review.
Do not write corpus files until I approve.
```

---

## Phase 2 — Ingestion + naive RAG + eval harness
**Read:** ARCHITECTURE §3, §5 (basics), §11, §12 · DECISIONS ADR-003, 004, 006, 008

**Build**
- `ingestion/`: front-matter parse, **recursive** chunker, embeddings (bge-small), tsvector, index_version, idempotent upsert, `make ingest`
- `retrieval/search.py`: vector-only top-k (the **naive** baseline)
- `generation/`: `answer.v1` prompt, structured `Answer`, citations
- `POST /v1/chat` (non-streaming JSON for now), sessions + messages saved
- `evals/`: retrieval metrics (Recall@k, MRR, context precision), LLM judges (faithfulness, relevance, correctness), refusal accuracy, runner with profiles, markdown + JSON report, `make eval`
- Profile `naive.yaml` → **baseline report saved as `evals/reports/baseline.json`**

**Done when:** `make ingest && make eval` produces a baseline report, and the numbers are in the PROGRESS scoreboard.

**Opening prompt**
```
Sutra, Phase 2 — Ingestion, naive RAG, eval harness.
Read CLAUDE.md, PROGRESS.md, Phase 2 in docs/BUILD_PLAN.md and the listed sections.
Keep retrieval deliberately naive (vector-only) — it's the baseline. Propose a plan and wait.
```

---

## Phase 3 — Advanced retrieval + routing + semantic cache
**Read:** ARCHITECTURE §3, §4, §7 · DECISIONS ADR-002, 004, 012

**Build** (one step at a time; **run `make eval` after each step** and add a scoreboard row)
1. Semantic chunking + parent-child chunks
2. Contextual retrieval prefixes
3. Keyword (FTS) search + RRF hybrid
4. Cross-encoder rerank
5. Query transforms: rewrite, multi-query, HyDE (each toggleable)
6. Metadata filters + query router
7. Semantic cache (+ cache-hit eval: repeated and paraphrased queries; check that hits never return the wrong answer)

**Done when:** each technique has its own scoreboard row, and the best profile is saved as `advanced.yaml`.

**Opening prompt**
```
Sutra, Phase 3 — Advanced retrieval, routing, semantic cache.
Read CLAUDE.md, PROGRESS.md, Phase 3 in docs/BUILD_PLAN.md and the listed sections.
Implement the steps in order, one at a time, running make eval after each and recording results.
Propose the plan for steps 1–3 first and wait.
```

---

## Phase 4 — Observability
**Read:** ARCHITECTURE §2, §10 · DECISIONS ADR-005, 007

**Build**
- Langfuse wired into `llm/client` (generations with tokens, cost, prompt version) and a span per pipeline stage
- Events: cache hit/miss, retrieval counts; `trace_id` returned in responses and logs
- Structured JSON logging
- SSE streaming for `/v1/chat` (`token`, `citations`, `meta`, `done`, `error`)
- `POST /v1/feedback` → stored + sent as a Langfuse score
- Eval runs tagged in Langfuse (so traces can be grouped per run)

**Done when:** one chat request shows a full trace tree in the Langfuse UI with per-stage latency and cost, streaming works from curl, and feedback appears on the trace.

**Opening prompt**
```
Sutra, Phase 4 — Observability + streaming + feedback.
Read CLAUDE.md, PROGRESS.md, Phase 4 in docs/BUILD_PLAN.md and the listed sections. Propose a plan and wait.
```

---

## Phase 5 — Generation, context management, memory
**Read:** ARCHITECTURE §5, §6 · DECISIONS ADR-005, 006, 013

**Build**
- Token budgeting with trim order; model routing (fast vs strong) + fallback; prompt caching layout
- `answer.v2` prompt → A/B via evals (v1 vs v2 rows on the scoreboard)
- Short-term memory: window + rolling summary/compaction
- Long-term memory: extract → dedupe → store → retrieve; memory guardrail (no PII/secrets)
- Eval: a scripted multi-turn conversation set (~10 dialogues), checking follow-up resolution ("what did *she* do next?") and memory recall across sessions

**Done when:** long chats stay within budget (tokens per turn plotted or logged), the multi-turn eval passes, and the v1 vs v2 prompt result is recorded.

**Opening prompt**
```
Sutra, Phase 5 — Generation, context management, memory.
Read CLAUDE.md, PROGRESS.md, Phase 5 in docs/BUILD_PLAN.md and the listed sections. Propose a plan and wait.
```

---

## Phase 6 — Guardrails
**Read:** ARCHITECTURE §9 · SPEC §4 (attacks) · DECISIONS open question on injection classifier

**Build**
- Guardrail framework: `Guardrail.check(ctx) -> Verdict`, pipeline composition, every verdict traced
- Input: length, rate limit, injection/jailbreak, PII detect + redact, topic scope
- Retrieval: indirect injection scan (the poisoned doc), ACL filter
- Output: schema repair, groundedness check, PII leak, low-confidence refusal
- `evals/runners/run_guardrails.py`: block rate, false-positive rate, per category; latency overhead per guardrail

**Done when:** attack eval ≥ 90% blocked and ≤ 10% false positives, the poisoned doc never steers an answer, and golden-set quality has not regressed.

**Opening prompt**
```
Sutra, Phase 6 — Guardrails.
Read CLAUDE.md, PROGRESS.md, Phase 6 in docs/BUILD_PLAN.md and the listed sections.
Resolve the injection-classifier open question first (options + tradeoffs), then propose a plan and wait.
```

---

## Phase 7 — Agentic RAG + MCP server
**Read:** ARCHITECTURE §8, §9 (tool row) · DECISIONS ADR-009, 010

**Build**
- Tools: `search_lore`, `get_timeline`, `lookup_character`, `calculator` (with Pydantic arg schemas)
- Hand-rolled agent loop: max steps, cost cap, tool guardrails, corrective-RAG grader + retry
- `mode: agent` in `/v1/chat` (streams tool-call events too)
- MCP server (stdio) exposing the same tools; instructions to connect from Claude Desktop
- Eval: `agentic.yaml` profile — compare against `advanced.yaml`, especially on multi-hop and numeric questions; record avg steps and cost

**Done when:** the agent beats advanced RAG on multi-hop correctness (or the doc explains why not), and the MCP tools work from an MCP client.

**Opening prompt**
```
Sutra, Phase 7 — Agentic RAG + MCP server.
Read CLAUDE.md, PROGRESS.md, Phase 7 in docs/BUILD_PLAN.md and the listed sections. Propose a plan and wait.
```

---

## Phase 8 — Voice agent (likely 2–3 chats: 8a, 8b, 8c)
**Read:** ARCHITECTURE §14 · DECISIONS ADR-011

**Build**
- **8a DIY:** push-to-talk (CLI or tiny HTML page) → STT API → core (agent mode) → TTS → playback; per-hop latency logged
- **8b Pipecat:** real-time WebRTC pipeline, VAD, turn detection, barge-in, streaming STT/TTS, custom processor calling Sutra core; traced in Langfuse
- **8c Gemini Live comparison:** speech-to-speech with Sutra tools via function calling; note what guardrails/tracing are lost
- `voice/README.md`: latency table (STT, TTFT, time-to-first-audio, total) for 8a / 8b / 8c and the cascaded vs speech-to-speech tradeoff

**Done when:** I can talk to Sutra in real time with interruptions, and the latency comparison table is filled.

**Opening prompt**
```
Sutra, Phase 8a — DIY voice pipeline.
Read CLAUDE.md, PROGRESS.md, Phase 8 in docs/BUILD_PLAN.md and the listed sections.
First resolve the STT/TTS provider open question: search current pricing, give 2–3 cheap options, and wait.
Then propose the 8a plan and wait.
```

---

## Phase 9 — CI evals, hardening, CONCEPTS.md, ablation report
**Read:** ARCHITECTURE §11, §15 · SPEC §7

**Build**
- Dockerfile for the api + full `docker compose up` of the whole stack
- GitHub Actions: lint + tests on push; eval smoke subset on PRs vs `baseline.json`, failing on regression thresholds (ask me to add the API-key secrets in the repo settings)
- `evals/reports/ABLATION.md`: naive → each retrieval step → advanced → agentic, plus guardrail and voice tables
- `docs/CONCEPTS.md`: every concept → one-paragraph explanation → file(s) implementing it → how to see it working
- Root `README.md`: what Sutra is, architecture diagram, quickstart, results headline

**Done when:** a fresh clone runs with `cp .env.example .env && make up && make ingest && make eval`, and SPEC §7 targets are met or the gaps are explained in ABLATION.md.

**Opening prompt**
```
Sutra, Phase 9 — CI evals, hardening, CONCEPTS.md, ablation report.
Read CLAUDE.md, PROGRESS.md, Phase 9 in docs/BUILD_PLAN.md and the listed sections. Propose a plan and wait.
```
