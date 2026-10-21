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
`search.py` holds `vector_search`. It is scoped to a single `index_version`, because an
unscoped search over a database holding two index versions blends them and the result belongs
to no measurable configuration at all. It also filters on `embedding IS NOT NULL`, which is
what keeps parent rows out of results.

`expand.py` (Phase 3 step 1) is pipeline step 6: swap each retrieved child for its parent,
dedupe, and inherit the child's rank. Parent-child chunking without this is just a smaller
chunk size, and strictly worse.

Deliberately **not** implemented here, and where each arrives:

| Absent | Arrives |
|---|---|
| keyword search over the `tsv` column | Phase 3 step 3 (hybrid) |
| rank fusion across retrievers (RRF) | Phase 3 step 3 |
| cross-encoder reranking (`bge-reranker-base`) | Phase 3 step 4 |
| query rewriting and routing | Phase 3 steps 5–6 |
| metadata filters, ACL enforcement | Phase 3 step 6 / later |

### Parent-child, and why the headline metric had to change
Searching 200-token children and answering from their ~800-token parents separates the unit
that embeds well from the unit that reads well. It also breaks the obvious way of measuring
retrieval. `recall@k` is computed over the retrieved units, so halving the chunk size means the
same retriever returns half as much text under the same `k` — the number falls for a reason
that has nothing to do with retrieval quality.

`context_recall` is the fix: the same quote-level recall, measured over the passages actually
sent to the generator after expansion. It means the same thing on a flat index and a
parent-child one, and on a flat index it is identical to `recall@k` by construction. **Compare
chunking schemes on `context_recall`; compare rankers on `recall@k` and MRR.**

## Tradeoffs
- **Dense-only retrieval is weak at exact-token recall.** A rare proper noun or an exact figure
  ("68,000", "waystone 17") can rank below a fluent paraphrase, because the embedding captures
  topic rather than identity. The corpus is full of such facts on purpose, so the Phase 3 hybrid
  delta should be visible rather than theoretical.
- **Chunk-size dilution.** At 400 tokens a chunk often spans several wiki sections, so a question
  about one place competes against embeddings averaging several places. Parent-child (step 1) is
  the intended answer, and it worked: context recall 61.9% → 76.2%.
- **Smaller children cost ranking precision — until the chunks carry their context.** Step 1
  moved MRR 0.492 → 0.437 and numeric context recall 86.1% → 72.2%: more, shorter units make a
  denser field of near-identical candidates, worst on the merchant ledger whose rows genuinely
  do all resemble each other. Step 2's generated prefixes repaired both (MRR 0.506, numeric
  back to 86.1%) by giving a bare ledger row back the words naming what it is a row of. The
  lesson is that small chunks and contextual prefixes are closer to one technique than two.
- **Dense retrieval still cannot tell `946` from `1,034`.** Exact-token recall on rare proper
  nouns and figures is what the keyword half of step 3 is for — a real gap, but a smaller one
  than it looked before step 2 closed the numeric regression.
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

The parent-child index is a second slot, built and measured alongside the baseline rather than
replacing it:
```bash
python -m sutra.ingestion.run --index-version 2 --parent-child --strategy semantic
python -m evals.check_coverage --from-db --index-version 2   # both roles must pass
python -m evals.runners.run --profile semantic
```
`make coverage` checks search units and context units **separately** on a parent-child index. A
parent contains all of its children's text, so a pooled check would pass whenever a quote
survived anywhere — including when it was split across children and only the parent held it
whole, which is precisely the defect that silently depresses recall.
`make eval` after each retrieval change; compare the new scoreboard row against
`evals/reports/baseline.json`.
