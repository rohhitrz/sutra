# Retrieval

> **Status:** stub — filled in Phase 2.

## Concept
Finding the passages that should be in the model's context.

## Why it matters in production
This is where most RAG quality is won or lost. Hybrid search plus reranking reliably beats vector-only search, and the only way to know by how much is to measure it.

## How Sutra implements it
Phase 2 ships vector-only top-k as a deliberate baseline. Phase 3 adds FTS, RRF fusion, a cross-encoder reranker, query transforms, metadata filters and a router — each toggleable and each measured.

## Tradeoffs
Every stage adds latency and cost. The scoreboard in PROGRESS.md records what each one actually bought.

## How to see it working
`make eval` after each change; compare the new scoreboard row against `evals/reports/baseline.json`.
