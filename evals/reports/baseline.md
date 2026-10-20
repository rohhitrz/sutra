# Eval report — `naive`

> Vector-only retrieval, answer.v1, no reranking or query transforms.

2026-10-20 16:52 UTC · 42 rows · 80s · $0.0232

## Configuration

| | |
|---|---|
| index fingerprint | `5b93e6401fd6b687` |
| index slot | 1 |
| chunking | 400 tokens, 60 overlap, chunker v1 |
| embedding model | `BAAI/bge-small-en-v1.5` |
| retrieval | vector-only, k=5 |
| answer model | `gpt-6-luna` |
| prompt versions | answer=v1, judge_faithfulness=v1, judge_relevance=v1, judge_correctness=v1 |

## Retrieval

| Recall@5 | Hit rate@5 | MRR | Context precision |
|---|---|---|---|
| 61.9% | 73.8% | 0.492 | 19.5% |

Recall is quote-level: of the quotes a question cites, the fraction that appear in some retrieved chunk. Hit rate is the weaker question of whether *anything* relevant came back. Context precision is the share of retrieved chunks that held evidence — the cost side, since every chunk is context the model pays to read.

## Generation

| Judge | Mean | Rows scored | Failed | Prompt |
|---|---|---|---|---|
| faithfulness | 1.000 | 42 | 0 | v1 |
| relevance | 0.976 | 42 | 0 | v1 |
| correctness | 0.821 | 42 | 0 | v1 |

_The judge model is the same model that produced the answers (ADR-017), so judge scores are same-family self-evaluation. Read them as relative — valid for comparing configurations with each other, not as ground truth._

## Grounding (mechanical)

A deterministic cross-check on the faithfulness judge: a digit figure in an answer either appears in the passages it was given or it does not. It cannot be talked round, which the judge can — but absence alone is a question, not a verdict, because this corpus has a ledger whose totals are arithmetic and a correct answer computes figures that appear nowhere in the text.

| | Rows |
|---|---|
| asserting a figure absent from the context | 8/42 |
| ...**and** missing some of their cited evidence | 4/42 |
| figure support rate | 87.6% |

### Unaccountable figures

A figure neither present in the retrieved passages nor derivable from them had to come from somewhere else. **Compare the last column.**

| Row | Category | Recall@5 | Figures | Judge faithfulness |
|---|---|---|---|---|
| `gq-002` | single-hop | 0.0% | 68,000 | 1.00 |
| `gq-018` | multi-hop | 0.0% | 1131 | 1.00 |
| `gq-028` | numeric | 66.7% | 40, 14 | 1.00 |
| `gq-029` | numeric | 50.0% | 224, 22.4, 14, 16 | 1.00 |

### Figures absent but accountable (probably arithmetic)

Every cited quote for these rows was retrieved, so the figure is most likely computed from the passages rather than invented. Listed for completeness, not as a finding.

| Row | Figures |
|---|---|
| `gq-030` | 2,750 |
| `gq-031` | 2,564 |
| `gq-033` | 135 |
| `gq-038` | 1012 |

## Refusal

| | Rows | Rate |
|---|---|---|
| correctly refused (unanswerable) | 6/6 | 100.0% |
| falsely refused (answerable) | 4/36 | 11.1% |

Two numbers, never one: a model that refuses everything scores perfectly on the first row and terribly on the second, and a model that answers everything does the reverse.

## By category

| Category | Rows | Recall@5 | MRR | Refusal correct | Faithfulness | Correctness |
|---|---|---|---|---|---|---|
| multi-hop | 8 | 37.5% | 0.385 | 75.0% | 1.00 | 0.69 |
| numeric | 6 | 86.1% | 0.672 | 83.3% | 1.00 | 0.83 |
| routing | 3 | 44.4% | 0.400 | 66.7% | 1.00 | 0.50 |
| single-hop | 12 | 58.3% | 0.458 | 100.0% | 1.00 | 0.92 |
| temporal | 7 | 64.3% | 0.521 | 100.0% | 1.00 | 0.86 |
| unanswerable | 6 | 83.3% | 0.533 | 100.0% | 1.00 | 0.92 |

## Cost and latency

| | |
|---|---|
| total | $0.0232 |
| per row | $0.00055 |
| answer latency p50 | 3157 ms |
| answer latency p95 | 5235 ms |

| tokens | input | output |
|---|---|---|
| answer calls | 106,262 | 7,033 |
| judge calls | 133,014 | 13,669 |
| **total** | **239,276** | **20,702** |

122,588 input tokens (51.2%) were served from the provider's prefix cache at the cached rate. Two runs doing identical work can differ substantially on cost from this alone, so read a cost delta against this number before reading it as a regression.

## Lowest-faithfulness rows

Where to look first. Low faithfulness with low recall is a retrieval problem; low faithfulness with high recall is the model ignoring what it was given.

| Row | Category | Recall@5 | Faith | Refused | Dropped cites |
|---|---|---|---|---|---|
| `gq-001` | single-hop | 0.0% | 1.00 | False | — |
| `gq-002` | single-hop | 0.0% | 1.00 | False | — |
| `gq-008` | single-hop | 0.0% | 1.00 | False | — |
| `gq-009` | single-hop | 0.0% | 1.00 | False | — |
| `gq-010` | single-hop | 0.0% | 1.00 | False | — |
| `gq-014` | multi-hop | 0.0% | 1.00 | False | — |
| `gq-016` | multi-hop | 0.0% | 1.00 | False | — |
| `gq-018` | multi-hop | 0.0% | 1.00 | True | — |
| `gq-024` | temporal | 0.0% | 1.00 | False | — |
| `gq-035` | routing | 0.0% | 1.00 | False | — |

