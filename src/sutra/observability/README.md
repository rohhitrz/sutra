# Observability

> **Status:** stub — filled in Phase 4.

## Concept
Traces, costs, latencies and user feedback for every request.

## Why it matters in production
You cannot improve or debug an LLM system you cannot see. Per-stage latency and cost are what turn a vague 'it feels slow' into a fix.

## How Sutra implements it
Langfuse with a span per pipeline stage, token and cost on every generation, prompt version recorded, guardrail and cache events, and `trace_id` returned to the caller (ADR-007).

## Tradeoffs
Tracing adds overhead and a self-hosted Langfuse makes the compose stack heavier.

## How to see it working
Send one chat request and open its full trace tree in the Langfuse UI.
