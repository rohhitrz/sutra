# Memory & context management

> **Status:** stub — filled in Phase 5.

## Concept
Short-term conversation memory plus durable long-term user facts.

## Why it matters in production
Context windows are finite and expensive. Naively replaying a whole conversation blows the budget and buries the relevant turns.

## How Sutra implements it
A sliding window plus a rolling summary for short-term memory; extract to dedupe to store to retrieve for long-term facts, with a guardrail that keeps PII and secrets out (ARCHITECTURE §6).

## Tradeoffs
Summarisation loses detail, and extracted facts can go stale or wrong. Measured by a multi-turn eval set rather than assumed.

## How to see it working
Run the multi-turn eval and watch tokens-per-turn stay flat as the conversation grows.
