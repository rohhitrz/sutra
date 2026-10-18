# Guardrails

> **Status:** stub — filled in Phase 6.

## Concept
Input, retrieval, output and tool-level checks.

## Why it matters in production
A system that will answer anything is not shippable. Guardrails are also where false positives hurt most — blocking legitimate questions is its own failure.

## How Sutra implements it
Input: injection, jailbreak, PII, topic and length checks plus rate limiting. Retrieval: indirect injection and ACL. Output: groundedness, PII, schema and low-confidence refusal. Tool: allowlist, max steps and cost cap.

## Tradeoffs
Every check adds latency and can misfire. Both block rate and false-positive rate are measured against `evals/datasets/attacks.jsonl`.

## How to see it working
Run the attacks dataset through `make eval` and read the block and false-positive rates.
