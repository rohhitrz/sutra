# Sutra — Decisions (ADR log)

> One entry per decision. Never edit an accepted ADR. To change one, add a new ADR that supersedes it.
> Format: Context → Decision → Why → Tradeoffs.

---

### ADR-001 · Python + FastAPI
- **Decision:** Python 3.12, FastAPI, async.
- **Why:** the AI/eval/voice ecosystem (RAGAS, Pipecat, sentence-transformers, the MCP SDK) is most mature in Python.
- **Tradeoffs:** my TS skills are not exercised here. The NotebookLM-style app already covers TS-side RAG.

### ADR-002 · Postgres + pgvector as the only datastore
- **Decision:** one Postgres holds vectors, full-text search, sessions, memory, cache, feedback, and ACL.
- **Why:** fewer moving parts. Hybrid search can be done in SQL. Very common in production.
- **Tradeoffs:** Postgres FTS is not true BM25 (`ts_rank_cd` is a different scoring). Accepted, and documented in retrieval/README.
- **Supersedes:** the earlier "Qdrant or pgvector" idea.

### ADR-003 · No LangChain / LlamaIndex for core logic
- **Decision:** hand-roll chunking, retrieval, fusion, the agent loop, and memory.
- **Why:** the goal is learning. Frameworks hide exactly the concepts we want to see.
- **Tradeoffs:** more code. Mitigated by keeping modules small.

### ADR-004 · Local embeddings and reranker
- **Decision:** `BAAI/bge-small-en-v1.5` (384d) for embeddings, `BAAI/bge-reranker-base` for reranking.
- **Why:** free, offline, fast enough on CPU for a ~10-file corpus, and reproducible evals.
- **Tradeoffs:** lower quality than top hosted embedders. The embedder is behind an interface, so it can be swapped later.

### ADR-005 · One thin LLM client
- **Decision:** `sutra.llm.client` is the only place that calls providers. It handles tracing, cost, retries, timeouts, fallback, and structured output.
- **Why:** one choke point for observability, cost control, and provider swaps.
- **Provider:** chosen in Phase 0 (open question). Two model roles from config: `LLM_FAST`, `LLM_STRONG`.

### ADR-006 · Prompts as versioned files
- **Decision:** `prompts/<name>.v<N>.md`. The active version is set in config and recorded on every message and trace.
- **Why:** enables A/B testing through evals and reproducibility.

### ADR-007 · Langfuse for observability
- **Decision:** self-hosted Langfuse in Docker Compose.
- **Why:** open source, LLM-native (traces, costs, prompt versions, scores), and OpenTelemetry-compatible.
- **Tradeoffs:** heavier compose stack (Langfuse brings its own services).

### ADR-008 · Evals: custom metrics + LLM judges (RAGAS where it fits)
- **Decision:** retrieval metrics are hand-written. Generation metrics use versioned judge prompts; RAGAS may be used for cross-checking.
- **Why:** understand every metric, not just call a library.

### ADR-009 · Hand-rolled agent loop
- **Decision:** our own tool-calling loop with max steps, cost cap, and corrective-RAG grading. No LangGraph.
- **Why:** learn the loop mechanics. LangGraph is already used in the Job Application Copilot project.

### ADR-010 · MCP server via official Python SDK, stdio transport
- **Decision:** expose the 4 agent tools as an MCP server.
- **Why:** MCP is the current industry standard for tool interoperability. Testable from Claude Desktop.

### ADR-011 · Voice: DIY first, then Pipecat; Gemini Live only as a comparison
- **Decision:** 8a DIY push-to-talk, 8b Pipecat real-time, 8c Gemini Live experiment.
- **Why:** DIY teaches the mechanics. Pipecat is pure Python, open source, and provider-agnostic, with no extra server (unlike LiveKit). Speech-to-speech skips our text-layer guardrails and tracing, so it is a comparison point, not the main path.
- **Providers:** decided in Phase 8 after checking current pricing (candidates: Groq Whisper or Deepgram for STT; Kokoro local or a cheap cloud TTS).

### ADR-012 · Semantic cache in Postgres
- **Decision:** pgvector table; threshold 0.95; keyed also by prompt_version, index_version, model; per-user scope.
- **Why:** shows cache correctness issues (stale answers, cross-user leaks) and how versioning fixes them.

### ADR-013 · Memory: window + rolling summary + long-term facts
- **Decision:** as in ARCHITECTURE §6.
- **Why:** this is the context management concept. It is measurable (tokens saved, answer quality over long chats).

### ADR-014 · Fictional corpus written by the LLM in Phase 1, then frozen
- **Decision:** 10 markdown files with planted traps. Frozen after Phase 1; changes need my OK, because they invalidate the eval baselines.

### ADR-015 · Tooling: uv, ruff, pyright, pytest, Alembic, Makefile
- **Decision:** as listed.
- **Why:** modern, fast, and standard in 2026 Python projects.

### ADR-016 · Public GitHub repo, Conventional Commits, push per step
- **Decision:** public repo `sutra` created with `gh` in Phase 0. Conventional Commits per logical step, pushed to `main`. No AI co-author trailers or attribution.
- **Why:** the repo is a portfolio piece; a clean, readable history shows how the system was built step by step.

### ADR-017 · LLM provider: OpenAI, `gpt-6-luna` for both model roles
- **Decision:** OpenAI is the provider. `LLM_FAST` and `LLM_STRONG` both resolve to `gpt-6-luna`.
  They stay two separate config values, so a later phase can split them without a code change.
  Adds the `openai` SDK as a dependency.
- **Why:** verified against OpenAI's own pricing page at $0.10 / $0.50 per 1M input / output tokens
  ($0.01 cached), which puts the whole project's eval budget in the $5–10 range. Cost matters here
  more than raw capability: `make eval` is ~250 calls and Phase 3 re-runs it after each of seven
  retrieval steps, so the model choice is dominated by eval volume, not by chat traffic.
- **Tradeoffs:** `gpt-6-luna` is a small model, and three places will feel it —
  (1) Phase 7's hand-rolled agent loop depends on reliable tool-calling,
  (2) Phase 6's guardrail classification depends on schema adherence, and
  (3) with no separate `LLM_JUDGE` role, eval judges run on `LLM_STRONG`, so the model judges its
  own answers. Same-family self-evaluation is known to inflate faithfulness scores, which means the
  absolute judge numbers on the scoreboard should be read as **relative** — good for comparing
  profiles against each other, not as ground truth. Also: with both roles on one model, the fallback
  chain and Phase 5's model routing are structurally present but functionally no-ops until a second
  model is configured.
- **Supersedes:** the "Provider: chosen in Phase 0 (open question)" line in ADR-005.

### ADR-018 · Concrete runtime dependency set
- **Decision:** beyond the tooling in ADR-015, the runtime deps are `fastapi`, `uvicorn`,
  `pydantic`, `pydantic-settings`, `sqlalchemy[asyncio]`, `asyncpg`, `alembic`, `pgvector`,
  `openai` and `pyyaml`. Dev-only: `httpx2` (the HTTP client the `openai` SDK is built on,
  reused for the ASGI test client so tests never mix two httpx major versions).
- **Why:** each one is implied by an existing ADR rather than a new choice — `asyncpg` and
  `sqlalchemy[asyncio]` by ADR-001's async requirement, `pgvector` by ADR-002, `alembic` by
  ADR-015, `pyyaml` by ADR-006's prompt front-matter and ADR-008's eval profile files.
- **Tradeoffs:** none significant. Recorded explicitly because CLAUDE.md requires sign-off on any
  dependency not already listed in DECISIONS, and ADR-015 covered only the dev toolchain.

### ADR-019 · Eval runs are exempt from the bulk-API-usage rule
- **Decision:** `make eval` may exceed the ">200 LLM calls in one run needs explicit OK" rule in
  CLAUDE.md without asking each time. Anything outside the eval runner still needs sign-off.
- **Why:** one eval run is ~250 calls by construction (~40 golden questions × one answer plus
  roughly three judges, plus query transforms). Phase 3 runs it after every retrieval change. Asking
  each time would make the measure-everything workflow unusable, and at ADR-017's prices a run costs
  well under a cent.
- **Tradeoffs:** removes a cost guardrail on the single highest-volume code path. Mitigated by the
  per-call cost accounting in `sutra.llm.client`, which makes actual spend per run visible, and by
  keeping the exemption scoped to the eval runner only.

### ADR-020 · `fastembed` (ONNX) is the local embedding runtime, not `sentence-transformers`
- **Context:** ADR-004 fixed the *models* (`bge-small-en-v1.5`, `bge-reranker-base`) but never the
  library that executes them. CLAUDE.md's stack line said "via `sentence-transformers`", and
  ADR-018's dependency set omitted it entirely, so the runtime was never actually decided.
- **Decision:** run both models through `fastembed` (ONNX Runtime). `sentence-transformers` is not
  installed. The models themselves are unchanged, so ADR-004 stands in full.
- **Why:** `sentence-transformers` requires PyTorch, which is a complete training framework —
  autograd, GPU kernels, thousands of compiled operators. Sutra needs exactly one capability from
  it: a single forward pass over a frozen 67 MB model, on CPU, once per chunk. It never trains
  anything. Measured on this machine, the two routes cost:

  | route | engine | `.venv` growth |
  |---|---|---|
  | `sentence-transformers` | PyTorch + transformers + scipy + scikit-learn | ~500 MB (est.) |
  | `fastembed` | ONNX Runtime | **141 MB (measured)** |

  Identical weights, identical 384 dimensions, identical vectors, so eval baselines are unaffected
  by the choice.
- **Verified before accepting:** `BAAI/bge-small-en-v1.5` is a supported fastembed model at
  `dim=384`, returns **L2-normalised** vectors (which is what makes `vector_cosine_ops` on the HNSW
  index correct), and `BAAI/bge-reranker-base` is one of 8 supported cross-encoders — so Phase 3's
  reranker needs no second runtime. Cold load 9.7 s including download; 3 texts in 8 ms after that.
- **Tradeoffs:** fastembed exposes a smaller surface than `sentence-transformers` — no fine-tuning,
  no pooling-strategy knobs, and a fixed list of 41 supported models (extensible via
  `add_custom_model`). None of that is wanted here; a model swap stays a one-line change because
  the embedder sits behind an interface, per ADR-004. ONNX Runtime also pulls `numpy`, `pillow`,
  `protobuf`, `tokenizers`, `huggingface-hub`, `loguru`, `mmh3`, `py-rust-stemmers` and `requests`
  as transitive dependencies.
- **Operational note:** model weights are **not** installed by the package manager. They download
  on first use into `Settings.model_cache_dir` (`./.models_cache`, 64 MB, gitignored), so they live
  in the project folder rather than the home directory.
- **Amends:** ADR-018's dependency set (adds `fastembed`). **Amends** the CLAUDE.md stack line,
  which has been updated to match.

### ADR-021 · `index_version` is an integer slot, with the real fingerprint in `metadata`
- **Context:** three places disagreed. ARCHITECTURE §3 described `index_version =
  hash(strategy + params + corpus)`; the §12 schema (already migrated in Phase 0) declares it
  `Integer`; `config.py` had `index_version: int = 1`. A hash does not fit an integer column.
- **Decision:** `index_version` is a small integer **slot** chosen by config or an eval profile
  (`1` = Phase 2 naive recursive chunking, `2` = Phase 3's scheme, and so on). The derived
  fingerprint — `{chunk_strategy, chunk_tokens, chunk_overlap, embedding_model, corpus_hash}` — is
  stored in `documents.metadata` and `chunks.metadata`, and ingestion compares it:
  - same slot, same fingerprint → **no-op** (this is what makes `make ingest` idempotent)
  - same slot, **different** fingerprint → **hard error** naming what changed, telling you to bump
    the slot or pass `--force`
  - different slot → written alongside, both kept, which is how ablations compare chunking schemes
- **Why:** this delivers everything §3's hash was for — idempotency, coexisting index versions,
  reproducible eval baselines — while *adding* a safety property a bare hash lacks: silently
  re-chunking into a slot an eval baseline is pinned to becomes impossible rather than automatic.
  It needs no schema change, and reports and the PROGRESS scoreboard carry a readable `1` instead
  of a hex digest.
- **Tradeoffs:** the slot number is chosen by a human, so bumping it is a thing you can forget. The
  fingerprint mismatch error is the mitigation and turns the forgotten bump into a loud failure at
  ingest time rather than a corrupted baseline. Reading ARCHITECTURE §3 literally would instead
  mean a `String(64)` column and a rewritten Phase 0 migration, for no behavioural gain.
- **Clarifies:** ARCHITECTURE §3's `index_version` wording, which was looser than the §12 schema
  it has to live in.

### ADR-022 · Amends ADR-021: fingerprint storage, and the code that produced the index
- **Context:** implementing ADR-021 showed two faults in it, both found by building the thing
  rather than by reasoning about it.
- **Decision 1 — the fingerprint lives in `chunks.metadata` only.** ADR-021 said
  "`documents.metadata` and `chunks.metadata`". That is wrong: a `documents` row is shared by
  every index slot, so one `06_places_wiki` row serves slot 1 and slot 2 at once and a per-slot
  fingerprint stored on it would be ambiguous at best. Chunks are per-slot, so the fingerprint
  belongs there and nowhere else. `documents.metadata` keeps the document's own front-matter.
- **Decision 2 — the fingerprint includes `chunker_version`.** ADR-021's field list covered the
  chunking *parameters* but not the *code* that applies them, which leaves a silent trap:

  > While building Phase 2, a bug fix in the heading scan changed the chunker's output for all ten
  > documents. The parameters had not changed, so the fingerprint had not changed, so `make ingest`
  > reported "nothing to do" and left the stale chunks in place. The fix had to be applied with
  > `--force`, and nothing in the system would have said so.

  `CHUNKER_VERSION`, a constant in `sutra.ingestion.chunk`, is bumped whenever a change can alter
  chunker output. It then flows into the fingerprint, so re-ingesting after a chunker change is
  refused with `chunker_version: 1 -> 2` instead of being mistaken for a no-op.
- **Why:** the fingerprint's job is to answer "is the index in this slot the one my eval baseline
  was measured against?". A fingerprint that ignores the code producing the index cannot answer
  that, which was the whole point of ADR-021.
- **Tradeoffs:** `CHUNKER_VERSION` is bumped by hand, so it can be forgotten — the same class of
  problem as the slot number, and mitigated the same way, by failing loudly at ingest time once it
  *is* bumped. Deriving it automatically (hashing the chunker's source, say) would catch a
  forgotten bump but also fire on a comment edit or a rename, making every refactor look like a
  re-index. A deliberate human signal is the better trade here. Bumping the constant invalidates
  every existing slot by design, which is the correct outcome and costs one `--force` re-ingest.
- **Amends:** ADR-021 (storage location, and the fingerprint field list). ADR-021 otherwise stands:
  `index_version` remains an integer slot, with the same no-op / refuse / coexist behaviour.

---

## Open questions (resolve in the phase noted)
- [x] Phase 0: which LLM provider(s) for `LLM_FAST` / `LLM_STRONG`? → **ADR-017** (OpenAI, `gpt-6-luna` for both)
- [ ] Phase 6: which classifier for injection detection — LLM-based only, or also a small local model?
- [ ] Phase 8: STT/TTS providers (after a pricing check)
