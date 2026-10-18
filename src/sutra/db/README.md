# Database

## Concept
One Postgres instance holds vectors, full-text search, sessions, memory, cache and feedback.

## Why it matters in production
Most teams reach for a dedicated vector DB and end up operating two stores that must be kept in sync. One Postgres means one backup story, one transaction boundary, and hybrid search expressible in a single SQL query.

## How Sutra implements it
Async SQLAlchemy engine + session factory, Alembic migrations. The `vector` extension, an HNSW index on embeddings, and GIN indexes on `tsv` and `metadata` are created in the initial migration (ARCHITECTURE §12).

## Tradeoffs
Postgres full-text search is not true BM25 — `ts_rank_cd` scores differently. Accepted; documented in `retrieval/README.md`.

## How to see it working
`make up`, then `uv run alembic current` shows the applied revision. `/v1/health` reports DB connectivity.
