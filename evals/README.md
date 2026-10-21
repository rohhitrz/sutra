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

## Grading the judge (`make judges`)

An LLM judge is an instrument, and an instrument nobody calibrates is a number generator.
`evals/datasets/judge_labels.jsonl` holds ten rows (ARCHITECTURE §11), each a
passages/question/answer triple with a score a human stands behind and the failure mode it
tests. The passages are verbatim corpus text, and every premise behind a label — "this figure
is absent from this passage" — is checked mechanically, so a label cannot quietly rot.

| Row | Failure mode | Label |
|---|---|---|
| jl-01 | faithful, stated directly | 1.0 |
| jl-02 | fabricated figure anchored on a real detail | 0.0 |
| jl-03 | correct arithmetic over quoted ledger rows | 1.0 |
| jl-04 | true in the world, absent from these passages | 0.0 |
| jl-05 | correct refusal | 1.0 |
| jl-06 | refusal when the passages do say | 0.0 |
| jl-07 | near-miss date | 0.0 |
| jl-08 | causal link the passages do not make | 0.5 |
| jl-09 | rounded figure | 0.5 |
| jl-10 | one supported claim plus one fabrication | 0.0 |

Two numbers come out, answering different questions. **Agreement** is how often the judge lands
on the labelled score — 9/10 for both v1 and v2, stable over three repeats, with the only
persistent miss being `jl-03` where both are too *harsh* about ledger arithmetic. **Fabricated
spans** is how often a v2 judge quotes a passage that does not exist; this is the one v1 cannot
expose, and it is worse than a wrong score, because a verdict justified by invented evidence is
worthless in either direction.

`jl-02` reproduces the anecdote this project built its distrust of the judge on. Both versions
score it correctly. One bad roll on a 42-row run is consistent with a judge that is right nine
times in ten, which is what the evidence actually shows.
