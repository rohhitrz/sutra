# Sutra — Rules for Claude Code

Sutra is a **learning-grade, production-style AI system** that answers questions about a fictional story corpus.
Every module exists to teach one industry concept: advanced RAG, context management, memory, guardrails,
observability, evaluation, agentic RAG, MCP, and voice. The code must be readable as a textbook.

---

## Read first (every new chat)

1. `PROGRESS.md` — where we are, what's done, open issues
2. `docs/BUILD_PLAN.md` — **only the current phase's section**
3. `docs/ARCHITECTURE.md` and `docs/DECISIONS.md` — **only the sections the phase lists**

Do not read the whole `docs/` folder or future phases unless asked. Do not work ahead of the current phase.

**Always propose a short plan for the phase and wait for my OK before writing code.**

---

## Stack (fixed — changes need a new ADR in `docs/DECISIONS.md`)

- **Language:** Python 3.12, `uv` for deps and venv, `ruff` (lint and format), `pyright` (type checking)
- **API:** FastAPI (async) with SSE streaming, Pydantic v2, `pydantic-settings` for config
- **DB:** Postgres 16 + `pgvector`. One DB for vectors, full-text search, sessions, memory, cache, feedback.
- **Embeddings:** local `BAAI/bge-small-en-v1.5` (384 dims) via `sentence-transformers`
- **Reranker:** local cross-encoder `BAAI/bge-reranker-base`
- **LLMs:** go through our own thin client `src/sutra/llm/`. Model names come from `.env` and are never hardcoded.
- **Observability:** Langfuse (OpenTelemetry-compatible)
- **Evals:** custom retrieval metrics + RAGAS / custom LLM-judges
- **Agent:** hand-rolled agent loop (no LangGraph). **MCP:** official `mcp` Python SDK.
- **Voice:** Phase 8a DIY pipeline, Phase 8b Pipecat
- **Infra:** Docker Compose, GitHub Actions (evals in CI), `Makefile` for commands

**Not allowed for core logic:** LangChain or LlamaIndex. We hand-roll RAG so the concepts stay visible.

---

## Repo layout

```
sutra/
├── CLAUDE.md  PROGRESS.md  Makefile  docker-compose.yml  pyproject.toml  .env.example
├── docs/                 SPEC, ARCHITECTURE, DECISIONS, BUILD_PLAN, CONCEPTS (Phase 9)
├── data/corpus/          the fictional story files (source of truth)
├── prompts/              versioned prompt files, e.g. answer.v1.md, answer.v2.md
├── src/sutra/
│   ├── config.py  db/  llm/
│   ├── ingestion/  retrieval/  generation/  memory/  cache/
│   ├── guardrails/  agent/  observability/
│   ├── api/  mcp_server/  voice/
├── evals/                datasets/, metrics/, runners/, reports/
└── tests/
```

Each module folder under `src/sutra/` has a `README.md` with these sections:
**Concept → Why it matters in production → How Sutra implements it → Tradeoffs → How to see it working.**

---

## Coding standards

- Type hints everywhere. Pydantic models at every boundary (API, LLM outputs, eval rows).
- Mark key teaching points with short comments: `# CONCEPT: Reciprocal Rank Fusion — merges ranked lists without score calibration.`
- **Every LLM call goes through `sutra.llm.client`.** It handles tracing, token and cost tracking, retries, timeouts, and model fallback. No direct SDK calls anywhere else.
- **Prompts live in `prompts/` as versioned files.** Never inline prompt text in Python. The active version is set in config. The version used is recorded on every message and trace.
- Config only through `sutra.config.Settings`. Secrets come only from `.env`, which is never committed. Keep `.env.example` updated.
- Async I/O for DB and HTTP. No blocking calls inside request handlers.
- Small, pure functions. Prefer clarity over cleverness. This is a learning codebase.
- Tests: `pytest`. Every phase adds tests for what it built. Never mark a phase done with failing tests.
- **Any retrieval or generation change → run `make eval` and record the numbers in `PROGRESS.md`.**

---

## Hard rules

- **Never commit secrets:** `.env`, API keys, model caches, and eval outputs containing keys stay out of git (`.gitignore`).
- **Ask me before:**
  - adding a dependency not listed in DECISIONS
  - changing the DB schema after Phase 2
  - deleting or renaming files
  - any bulk paid API usage (>200 LLM calls in one run)
  - changing any decision in DECISIONS.md — propose a new ADR instead
- The corpus in `data/corpus/` is the only source of truth for story facts. Never invent lore in code, tests, or eval data.
- If something in the docs is ambiguous or contradictory, **stop and ask**. Do not guess.

---

## Git & GitHub

**Setup (once, in Phase 0):**
- `git init` with `main` as the default branch, and a proper `.gitignore` for Python, `.env`, `.venv`, model caches, and `__pycache__`.
- Use my **existing** global git identity. Never set or change `user.name` / `user.email`. If none is configured, stop and ask me.
- Check `gh auth status`. If not logged in, ask me to run `gh auth login`.
- Create a **public** GitHub repo named `sutra` with `gh repo create`:
  - **Description:** "Production-style learning codebase: advanced RAG, guardrails, observability & evals, agentic RAG, MCP and voice agents — every concept hand-built and measured."
  - **Topics:** `rag`, `llm`, `ai-engineering`, `guardrails`, `observability`, `evaluation`, `agents`, `mcp`, `voice-agent`, `fastapi`, `pgvector`, `python`
  - Push `main` and set the upstream.
  - Confirm the repo name with me once before creating it.

**Ongoing (every phase):**
- Commit after each **logical step** (not one giant commit per phase), then push to `origin main`.
- Use **Conventional Commits** with a scope:
  - Examples: `feat(retrieval): add RRF hybrid fusion`, `test(evals): add recall@k metric tests`, `docs(progress): complete phase 2`
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`
- Short imperative subject (≤ 72 chars). Add a body when the *why* isn't obvious, including eval deltas when relevant.
- **No co-author trailers and no Claude/AI attribution** in commit messages, PRs, or the repo. Commits are authored by me only.
- Only commit when tests and lint pass.
- Never force-push, rewrite pushed history, or delete branches.
- The phase-completion commit includes the updated `PROGRESS.md`.

---

## Commands

```
make up        # docker compose up (postgres, etc.)
make down
make dev       # run API locally with reload
make ingest    # ingest data/corpus into Postgres
make eval      # run eval suite, write report to evals/reports/
make test      # pytest
make lint      # ruff + pyright
```

---

## Updating PROGRESS.md (mandatory)

**When a phase is completed**, before saying it's done:
1. Mark it ✅ in the phase table, with the date.
2. Set "Current phase" to the next phase.
3. Add the final eval numbers to the scoreboard.
4. Commit with `docs(progress): complete phase N — <short title>` and push.

**At the end of every chat** (even mid-phase), update `PROGRESS.md` with these items:
- phase status
- what was built (files)
- eval numbers, if any
- decisions taken (and any ADRs added)
- known issues
- the exact next step

Keep it concise. The next chat starts from it.
