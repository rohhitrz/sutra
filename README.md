# Sutra

A question-answering system over a fictional story universe — built as a **learning-grade,
production-style AI system**. Every module exists to teach one industry concept: advanced RAG,
context management, memory, guardrails, observability, evaluation, agentic RAG, MCP and voice.

The point is not the answers. The point is that every layer is **hand-built and measured**, so each
concept can be explained, compared and ablated. See [`docs/SPEC.md`](docs/SPEC.md) for what it does
and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how.

## Status

Built in phases. See [`PROGRESS.md`](PROGRESS.md) for where things stand and
[`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) for what each phase delivers.

## Stack

Python 3.12 · FastAPI (async, SSE) · Postgres 16 + pgvector · local `bge-small-en-v1.5` embeddings
and `bge-reranker-base` · Langfuse · Alembic · uv / ruff / pyright / pytest.

No LangChain or LlamaIndex: chunking, retrieval, fusion, the agent loop and memory are hand-rolled
so the concepts stay visible ([ADR-003](docs/DECISIONS.md)).

## Quickstart

```bash
cp .env.example .env        # then fill in OPENAI_API_KEY
make install                # create the venv, install deps
make up                     # start Postgres + pgvector
make migrate                # apply the schema
make dev                    # API on http://localhost:8000
curl localhost:8000/v1/health
```

## Commands

| Command | What it does |
|---|---|
| `make up` / `make down` | start / stop the Docker stack |
| `make dev` | run the API locally with reload |
| `make migrate` | apply Alembic migrations |
| `make ingest` | ingest `data/corpus/` into Postgres |
| `make eval` | run the eval suite, write a report to `evals/reports/` |
| `make test` | pytest |
| `make lint` | ruff + pyright |

Run `make help` for the full list.

## Documentation

Each module under `src/sutra/` has a README explaining its concept, why it matters in production,
how Sutra implements it, the tradeoffs, and how to see it working.

- [`docs/SPEC.md`](docs/SPEC.md) — what Sutra is, scope, success criteria
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design, data model, request lifecycle
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — the ADR log
- [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) — the phase-by-phase plan
