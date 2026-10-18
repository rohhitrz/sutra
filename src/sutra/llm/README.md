# LLM client

## Concept
A single thin client is the only place that talks to an LLM provider.

## Why it matters in production
One choke point for tracing, token and cost accounting, retries, timeouts and model fallback. Swapping providers touches one file instead of the whole codebase.

## How Sutra implements it
`complete()`, `stream()` and `structured()` behind a provider-adapter protocol. Cost is computed per call from a price table in config. The tracing hook is a no-op until Phase 4.

## Tradeoffs
A thin abstraction can hide provider-specific features. Mitigated by keeping the adapter surface deliberately small.

## How to see it working
Unit tests call the client with a fake adapter and assert the token and cost accounting. From Phase 4, every call appears in Langfuse.
