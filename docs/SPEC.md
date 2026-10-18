# Sutra — Spec

## 1. What it is
Sutra is a question-answering system over a fictional story universe. It answers with citations, remembers the
conversation, refuses unsafe or out-of-scope requests, and can be used by **text (API)**, **agent**, **MCP client**,
or **voice**.

The real goal is **learning**. Each layer implements one production concept, so it can be explained, measured,
and compared in an ablation.

## 2. Who it's for
- **Primary:** me, Rohit, learning production AI engineering by building it.
- **Secondary:** recruiters and interviewers. The repo, the ablation report, and CONCEPTS.md should show
  production-level understanding.

## 3. Why a fictional corpus
The LLM cannot know these facts from pretraining, so every correct answer must come from retrieval.
This makes the evals honest.

## 4. The corpus (`data/corpus/`)
A fictional universe, working title **"The Meridian Chronicles"**. The details are invented in Phase 1.
There are **10 markdown files, each 100–200 lines**, with YAML front-matter metadata
(`title, doc_type, book, chapters, characters, in_world_dates`).

| File | Type | Purpose for testing |
|---|---|---|
| 01_book1_part1.md | narrative | basic retrieval |
| 02_book1_part2.md | narrative | multi-hop with 01 |
| 03_book2.md | narrative | **contradicts/updates** facts from Book 1 (recency) |
| 04_character_registry.md | table-like | structured lookup, the `lookup_character` tool |
| 05_timeline.md | chronological list | **query routing**, temporal questions |
| 06_places_wiki.md | wiki | entity-heavy, metadata filtering |
| 07_factions.md | wiki | multi-hop relationships |
| 08_faq.md | Q&A | short-chunk behavior |
| 09_merchant_ledger.md | numbers | calculator tool, numeric questions |
| 10_traveler_notes.md | notes | **poisoned**: hidden prompt injection + fake PII (emails/phones) to test guardrails |

### Planted traps
- At least 5 multi-hop facts (each needs 2+ files)
- At least 3 facts updated in Book 2
- At least 1 indirect prompt injection
- At least 3 fake PII items
- Several plausible questions the corpus does **not** answer

### Datasets (`evals/datasets/`)
- `golden_qa.jsonl`: about 40 questions. Categories: single-hop, multi-hop, temporal/updated, numeric, routing, unanswerable.
- `attacks.jsonl`: about 30 inputs. Categories: direct injection, jailbreak, PII extraction, off-topic, toxic. Also about 10 **benign look-alikes** to measure false positives.

## 5. Scope (in)

| Layer | Features |
|---|---|
| Ingestion | front-matter parsing, recursive + semantic chunking, parent-child chunks, contextual retrieval prefixes, embeddings, FTS index, versioned indexes |
| Retrieval | vector + keyword hybrid, RRF fusion, cross-encoder rerank, query rewrite / multi-query / HyDE, metadata filters, **query routing**, citations |
| Cache | **semantic cache** (embedding similarity, invalidated by prompt or index version) |
| Generation | structured output (answer, citations, confidence), token budgeting, prompt versioning, prompt caching, model routing + fallback, SSE streaming |
| Memory | short-term (sliding window + rolling summary/compaction), long-term user memories (extract → store → retrieve) |
| Guardrails | input (injection, jailbreak, PII, topic, length, rate limit), retrieval (indirect injection, ACL), output (groundedness, PII, schema, low-confidence refusal), tool (allowlist, max steps, cost cap) |
| Agent | hand-rolled tool loop, corrective RAG (grade, then re-query), multi-hop; tools: search_lore, get_timeline, lookup_character, calculator |
| MCP | Sutra's tools exposed as an MCP server (stdio) |
| Voice | 8a DIY push-to-talk pipeline; 8b Pipecat real-time (VAD, barge-in); 8c Gemini Live comparison experiment |
| Observability | Langfuse traces with a span per pipeline stage; tokens, cost, latency, prompt version; guardrail and cache events; user feedback linked to the trace |
| Evals | retrieval metrics, LLM-judge generation metrics, guardrail block/false-positive rates, ablation table, evals in CI |
| Infra | Docker Compose, Makefile, GitHub Actions |

## 6. Non-goals
- Polished frontend. Swagger, a CLI, and a minimal HTML test page are enough. Voice uses Pipecat's small web client.
- Auth beyond a simple API key + `user_id`.
- Multi-agent systems, GraphRAG, embedding fine-tuning. These are possible v2 ideas.
- Real deployment to the cloud. Local Docker is the target.

## 7. Success criteria (targets, verified by evals)
- Naive RAG baseline recorded in Phase 2. Every later retrieval change shows its delta.
- Advanced retrieval beats the baseline on Recall@5 and on multi-hop/temporal correctness.
- Faithfulness ≥ 0.85 on answerable questions. Correct refusal on ≥ 80% of unanswerable questions.
- Attack set: block rate ≥ 90%, false-positive rate ≤ 10% on benign look-alikes.
- Every request is traced with per-stage latency and cost.
- Voice: time-to-first-audio is measured and reported for DIY, Pipecat, and Gemini Live.
- CI fails if faithfulness or Recall@5 drops more than a set threshold vs the stored baseline.
