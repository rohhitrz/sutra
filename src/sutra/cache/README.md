# Semantic cache

> **Status:** stub — filled in Phase 3.

## Concept
Serving a cached answer when a new question is semantically close to an old one.

## Why it matters in production
Cuts cost and latency on repeated questions. It is also a classic source of production bugs: stale answers after a re-index, and cross-user data leaks.

## How Sutra implements it
A pgvector table with a 0.95 similarity threshold, keyed additionally by `prompt_version`, `index_version` and model, and scoped per user (ADR-012).

## Tradeoffs
Too low a threshold returns confidently wrong answers. The cache-hit eval checks that hits never serve the wrong answer.

## How to see it working
Ask the same question twice, then a paraphrase; the cache-hit event appears in the trace and the eval reports hit rate alongside correctness.
