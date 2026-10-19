# Evals

## Concept
Measure retrieval and generation separately, against ground truth, so that every later change can
be attributed rather than argued about.

## Why it matters in production
A RAG system has no single quality number, and inventing one hides the failures that matter. Phase
2's first live call made the case concretely: the model answered "68,000" — exactly right, and
present in none of the five passages it was given. Correctness said 1.0. Faithfulness should have
said 0.0. A blended score would have called it a success and moved on.

So: four independent families of metric, and a report that puts them beside each other.

## How Sutra implements it
| Metric | Kind | What it answers |
|---|---|---|
| Recall@k | mechanical | of the evidence a question cites, how much was fetched? |
| Hit rate@k | mechanical | did *anything* relevant come back? |
| MRR | mechanical | how near the top was the first useful chunk? |
| Context precision | mechanical | how much of the fetched context was useful at all? |
| Faithfulness | LLM judge | is every claim supported by the passages? |
| Relevance | LLM judge | does it answer the question asked? |
| Correctness | LLM judge | does it agree with the human reference? |
| Refusal recall / false refusal | mechanical | is declining a skill, or a reflex? |
| Figure grounding | mechanical | does every digit figure appear in the context? |

- **Ground truth is 72 verbatim quotes tied to documents**, never chunk ids — ids change on every
  re-ingest, so a dataset keyed on them would expire. `make coverage` is the precondition: if the
  chunker split a cited quote, recall understates for a reason that has nothing to do with the
  retriever, and `make eval` refuses to run.
- **Recall is quote-level and macro-averaged.** A question citing five quotes that lands one has
  not been served; and every question counts once, so the handful of multi-hop rows cannot
  dominate the headline.
- **Refusal is two numbers, never one.** A model that refuses everything scores perfectly on the
  unanswerable rows; a model that answers everything scores perfectly on the rest.
- **Failed judge calls are excluded, not scored zero**, and the exclusion is reported. A timeout
  is not evidence that an answer was bad.
- **Every report carries the configuration that produced it** — index fingerprint, chunking
  parameters, chunker version, prompt versions, judge versions, model, k. A number without that
  block cannot be compared to the next one.

## Tradeoffs
- **The judge is the same model that writes the answers (ADR-017), and it shows.** On the
  baseline it scored faithfulness 1.00 on an answer whose central figure appeared nowhere in its
  context, reasoning that "the passage states that Kesh Ithar has 68,000 people". It had anchored
  on the one real detail in the answer — a date that *was* in the passages — and approved the
  rest. Read judge scores as **relative**, and prefer the mechanical check where they disagree.
- **Judge scores carry run-to-run noise of roughly ±0.05** on identical configuration, because
  the judge is a sampled model. Retrieval metrics are exact and repeat to the digit. A Phase 3
  change that moves faithfulness by 0.02 has moved nothing measurable.
- **The figure-grounding check cannot tell derivation from fabrication by itself.** The corpus
  ledger's totals are arithmetic on purpose, so a correct answer computes figures that appear
  nowhere. The report therefore pairs it with recall: unsupported figure *plus* missing evidence
  is suspicious; unsupported figure with every quote retrieved is usually a sum.
- One judge call per axis, with no claim decomposition. RAGAS-style per-claim faithfulness would
  be more precise and several times the cost.
- The judges have never been validated against human labels. ARCHITECTURE §11 wants about ten
  hand-labelled rows for that, and the evidence above says it is overdue.

## How to see it working
```bash
make validate     # the datasets are consistent with the frozen corpus
make ingest       # build the index
make coverage     # every cited quote survived chunking
make eval         # the numbers, into evals/reports/
```
`--limit N` evaluates the first N rows (one row is four LLM calls, so this is the cheap way to
check a change to the harness itself). `--baseline` also writes `evals/reports/baseline.json`,
the reference every later run is read against. `--profile hybrid` will select a Phase 3 profile
from `evals/profiles/`.
