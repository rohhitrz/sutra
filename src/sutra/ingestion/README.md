# Ingestion

> **Status:** stub — filled in Phase 2.

## Concept
Turning corpus files into retrievable, versioned chunks.

## Why it matters in production
Chunking strategy drives retrieval quality more than the embedding model does. Versioned indexes let you re-ingest without invalidating live traffic or eval baselines.

## How Sutra implements it
Front-matter parsing, recursive chunking (semantic and parent-child in Phase 3), bge-small embeddings, tsvector generation, and idempotent upsert keyed by content hash and `index_version`.

## Tradeoffs
Smaller chunks retrieve precisely but lose context; larger chunks do the opposite. Phase 3 measures the tradeoff instead of guessing.

## How to see it working
`make ingest`, then query `chunks` grouped by `index_version`.
