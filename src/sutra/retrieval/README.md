# Retrieval

## Concept
Finding the passages that should be in the model's context. Phase 2 ships one stage: embed the
query, take the k nearest chunks by cosine distance within one index slot.

## Why it matters in production
This is where most RAG quality is won or lost. Hybrid search plus reranking reliably beats
vector-only search — but "reliably beats" is a claim, and a project that cannot put a number on it
is choosing techniques by reputation. That is what the baseline is for: it is a measurement
instrument, not a first draft, and making it clever would destroy the only reference point the
project has.

## How Sutra implements it
`search.py` holds `vector_search`, and nothing else. It is scoped to a single `index_version`,
because an unscoped search over a database holding two index versions blends them and the result
belongs to no measurable configuration at all.

Deliberately **not** implemented here, and where each arrives:

| Absent from Phase 2 | Arrives |
|---|---|
| keyword search over the `tsv` column | Phase 3 (hybrid) |
| rank fusion across retrievers (RRF) | Phase 3 |
| cross-encoder reranking (`bge-reranker-base`) | Phase 3 |
| query rewriting and routing | Phase 3 |
| parent-child expansion, contextual prefixes | Phase 3 |
| metadata filters, ACL enforcement | Phase 3 / later |

## Tradeoffs
- **Dense-only retrieval is weak at exact-token recall.** A rare proper noun or an exact figure
  ("68,000", "waystone 17") can rank below a fluent paraphrase, because the embedding captures
  topic rather than identity. The corpus is full of such facts on purpose, so the Phase 3 hybrid
  delta should be visible rather than theoretical.
- **Chunk-size dilution.** At 400 tokens a chunk often spans several wiki sections, so a question
  about one place competes against embeddings averaging several places. Phase 3's parent-child
  split is the intended answer, and the baseline is what makes its value measurable.
- **The HNSW index is not exercised at this scale.** With 49 chunks the planner filters on
  `index_version` and sorts rather than using the vector index (confirmed with `EXPLAIN`). The
  index is correct and will matter at volume, but nothing in this project tests the
  approximate-recall behaviour of HNSW — a property of a 10-file corpus, not of the query.
- Every later stage adds latency and cost. The scoreboard in `PROGRESS.md` records what each one
  actually bought.

## Where the baseline stands
Measured over the 42 golden rows against the 49 chunks in index slot 1, scoring a row by the best
rank of any chunk containing one of its cited quotes:

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | median rank | MRR |
|---|---|---|---|---|---|
| 36% | 57% | 74% | 88% | 2 | 0.519 |

Weakest category is `routing` (median rank 5), strongest is `numeric` (median 1.5). These come
from an ad-hoc measurement during Phase 2 step 4 — the figures of record are whatever
`make eval` writes to `evals/reports/baseline.json`.

## How to see it working
```bash
make ingest && make coverage     # the index, and proof the quotes survive it
make eval                        # the numbers of record
```
`make eval` after each retrieval change; compare the new scoreboard row against
`evals/reports/baseline.json`.
