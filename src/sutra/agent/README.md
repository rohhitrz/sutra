# Agent (agentic RAG)

> **Status:** stub — filled in Phase 7.

## Concept
A hand-rolled tool-calling loop with corrective retrieval.

## Why it matters in production
Single-shot RAG fails on multi-hop questions. An agent that can grade its own retrieval and re-query handles them — at the cost of unbounded loops and spend if it is not capped.

## How Sutra implements it
Our own loop with max steps, a cost cap and retrieval grading. Tools: `search_lore`, `get_timeline`, `lookup_character`, `calculator`. No LangGraph (ADR-009).

## Tradeoffs
More latency and cost per question, and harder to debug than a fixed pipeline. The trace tree is what makes it debuggable.

## How to see it working
Ask a multi-hop question and watch the step-by-step tool calls in the Langfuse trace.
