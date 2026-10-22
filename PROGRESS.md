# Sutra — Progress

> Handoff file between chats. Read at the start of every chat. Update at the end of every chat.

## Current phase
**Phase 5 — Generation, context management, memory** · Status: 🟨 in progress — **steps 1-3 of 7 done**
(Token budget, model routing/fallback auditability, and the answer.v1-vs-v2 A/B are built,
measured and pushed. v2 was **not adopted** — the result is inside the noise band. Steps 4-7
are short-term memory, long-term memory, the multi-turn eval, and the phase wrap-up.)

## Phase status

| # | Phase | Status | Chat date |
|---|---|---|---|
| 0 | Foundation | ✅ done | 2026-10-19 |
| 1 | Corpus + golden datasets | ✅ done | 2026-10-19 |
| 2 | Ingestion + naive RAG + eval harness | ✅ done | 2026-10-20 |
| 3 | Advanced retrieval + routing + semantic cache | ✅ done | 2026-10-21 |
| 4 | Observability | ✅ done | 2026-10-21 |
| 5 | Generation, context management, memory | 🟨 steps 1-3/7 | 2026-10-22 |
| 6 | Guardrails | ⬜ | |
| 7 | Agentic RAG + MCP server | ⬜ | |
| 8 | Voice agent | ⬜ | |
| 9 | CI evals, hardening, CONCEPTS.md, ablation report | ⬜ | |

Legend: ⬜ not started · 🟨 in progress · ✅ done

## Eval scoreboard
(Filled from Phase 2 onward. One row per eval run that matters.)

| Date | Config / change | Recall@5 | **Context recall** | MRR | Faithfulness | Answer correctness | p50 latency | Cost/query |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | ~~`naive` baseline, as first recorded~~ — **superseded** | 61.9% | — | 0.492 | 0.964 | 0.774 | 2373 ms | $0.00050 |
| 2026-10-21 | **baseline** — `naive`, vector-only, k=5, answer.v1, slot 1 | 61.9% | 61.9% | 0.492 | 0.976 | 0.774 | 2734 ms | $0.00056 |
| 2026-10-21 | **step 1** — `semantic`, parent-child + expansion, slot 2 | 63.9% | 76.2% | 0.437 | 0.988 | 0.774 | 2747 ms | $0.00056 |
| 2026-10-21 | **step 2** — `contextual`, + generated prefixes, slot 3 | 69.4% | 79.8% | 0.506 | 0.952 | 0.810 | 2529 ms | $0.00086 |
| 2026-10-21 | ablation — `keyword_only`, FTS alone, slot 3 | 61.9% | 68.3% | 0.471 | 1.000 | 0.774 | 2860 ms | $0.00092 |
| 2026-10-21 | **step 3** — `hybrid`, vector + FTS fused by RRF, slot 3 | **73.0%** | **81.0%** | **0.528** | 0.964 | 0.810 | 2572 ms | $0.00085 |
| 2026-10-21 | step 3 re-graded by **judge v2** — same retrieval, new judge | 73.0% | 81.0% | 0.528 | 0.893 | 0.750 | 2612 ms | $0.00096 |
| 2026-10-21 | step 4 — rerank on **bare** chunk text (wrong, kept as evidence) | 69.4% | 76.2% | 0.510 | 0.976 | 0.833 | 3742 ms | $0.00129 |
| 2026-10-21 | **step 4** — `hybrid_rerank`, cross-encoder on indexed text | 77.8% | 82.1% | 0.563 | 0.988 | 0.833 | 3502 ms | $0.00116 |
| 2026-10-21 | step 5a — `multiquery`, rewrite + decomposition | 77.8% | 82.1% | 0.563 | 0.988 | 0.845 | 3319 ms | $0.00083 |
| 2026-10-21 | **step 5b** — `hyde`, hypothetical-answer probe | 79.0% | 84.9% | 0.578 | 0.964 | 0.821 | 3719 ms | $0.00101 |
| 2026-10-21 | step 6 — `routed`, query router + doc-type filters | 72.2% | 76.6% | 0.567 | 0.988 | 0.821 | 3523 ms | $0.00112 |
| **2026-10-21** | **`advanced` — the Phase 3 profile** | **77.8%** | **83.7%** | **0.571** | 0.976 | 0.857 | 3550 ms | $0.00091 |
| 2026-10-21 | `advanced` re-run **after Phase 4 tracing** — no behaviour change intended | 78.2% | 83.3% | 0.565 | 0.976 | 0.821 | 3213 ms | $0.00121 |
| 2026-10-21 | `advanced` **repeat**, rewrite still on — the comparison point for the row below | 78.2% | 83.3% | 0.570 | 0.988 | 0.821 | 2793 ms | $0.00119 |
| **2026-10-21** | **`hyde_only` — `advanced` with query rewriting off. Now what `advanced` is.** | **79.0%** | **84.1%** | **0.574** | 0.964 | 0.821 | 2706 ms | $0.00093 |
| 2026-10-22 | `advanced` **repeat** — the v1 arm of the answer.v2 A/B, and the budget's first run | 78.2% | 83.3% | 0.565 | 0.976¹ | 0.833 | 2917 ms | $0.00116 |
| 2026-10-22 | `answer_v2` — **the v2 arm. Not adopted: inside the noise band.** | 79.0% | 84.9% | 0.571 | 0.988 | 0.845 | 2607 ms | $0.00120 |

¹ Over **41 rows, not 42**: one `judge_faithfulness` call was lost to JSON
truncation (new known issue 22). Not comparable to the row below it on equal terms.

**The faithfulness column changes meaning at the last row.** Rows above it were graded by
`judge_faithfulness.v1`; from step 4 onward the default is v2, which must quote the span
supporting each claim. The two are not comparable — v2 reads 0.071 lower on identical
retrieval — so the bridge row exists to make the size of that shift visible rather than
letting it look like a regression. Retrieval metrics are unaffected and identical, as they
must be: the judge does not touch retrieval.

All three rows above were produced by the **same harness code on the same day**, which earlier
rows were not. That re-run was forced by a bug found in step 2 (the figure check was flagging
comma-terminated figures — see below), and it is the reason the baseline's figure count and
judge scores differ from the 2026-10-20 recording.

**Context recall is the column to read across chunking schemes**, and it is new in step 1.
Recall@5 is measured on the retrieved units, so it moves with their size — slot 2's children
are 200 tokens against slot 1's 400, so the same retriever fetches less text at the same k
whatever its quality. Context recall is measured on the passages actually sent to the
generator. On the flat baseline the two are identical by construction, which is why that row
can carry the number without being re-run.

Also from the current baseline run (`evals/reports/baseline.md`): context precision 19.5%,
hit rate@5 73.8%, refusal recall 100% (6/6), false refusal 11.1% (4/36), 4 rows asserting an
unsupported figure, $0.0236 for the whole 42-row run at 52% prefix-cache hits.

**Why the baseline was re-recorded.** Three defects in the eval harness were fixed (see the
review-fix section below), two of which changed what the report says: token totals counted judge
calls only, and the figure check cleared a fabrication whenever its digits appeared inside a
longer number. The system under measurement did not change. The proof is that **recall@5, MRR and
context precision are identical to the digit across both rows** — those metrics are exact and
depend only on the index, which was verified byte-identical (same 49 chunks, same text, same
token counts).

The judge and latency columns moved for ordinary run-to-run reasons, not because of the fixes.

### `answer.v2` versus `answer.v1` — a negative result, and v1 stays active

**Verdict: no measurable difference. v2 is not adopted.** `prompt_version_answer` stays `v1`.
`answer.v2` and `evals/profiles/answer_v2.yaml` are kept as the record of the experiment, the
way `hyde_only.yaml` is.

The hypothesis was specific and is written in `prompts/answer.v2.md`: **v1 over-refuses.** It
came from `2026-10-21_advanced.run2`, where multi-hop scored correctness 0.50 against
faithfulness 1.00 and 3 of 36 answerable rows were falsely refused. A refusal asserts nothing
about the corpus, so it cannot be unfaithful — perfect faithfulness beside half correctness is
what refusing looks like from outside. v2 rebalanced the refusal threshold toward synthesis and
changed nothing else; a test compares the untouched sections of the two files so the A/B is one
variable.

Headline: **correctness 0.833 → 0.845, +0.012 against a noise band of at least ±0.05**
(known issue 3). That is not a result. Per category, on a 42-row set where one row is worth
0.167 in a 6-row category:

| category | rows | correctness v1 → v2 | refusal_correct v1 → v2 |
|---|---|---|---|
| single-hop | 12 | 0.917 → 0.917 | 1.000 → 1.000 |
| multi-hop | 8 | **0.562 → 0.562** | 0.750 → 0.875 |
| temporal | 7 | 0.786 → 0.857 | 1.000 → 1.000 |
| numeric | 6 | 1.000 → 1.000 | 1.000 → 1.000 |
| unanswerable | 6 | 0.917 → 1.000 | **1.000 → 0.833** |
| routing | 3 | 0.833 → 0.667 | 1.000 → 1.000 |

**Three things this run established, none of them the thing it set out to test.**

1. **The predicted gain did not happen.** Multi-hop correctness is identical in both arms. And
   the 0.50 the hypothesis was built on was itself partly noise: the *same* `advanced` profile
   scored 0.562 on multi-hop today against 0.500 on 2026-10-21. **The baseline moved on a
   repeat by more than the effect being looked for**, which is known issue 3 doing exactly what
   it warns about, on the first occasion it mattered.

2. **The risk named in advance is the one that materialised, and it netted zero.** Falsely
   refused went 2 → 1 of 36; correctly refused went 6 → 5 of 6. One row each way. v2 does
   refuse less, and it traded one false refusal for one missed refusal. On this corpus that is
   a wash, and the trade is worse than it looks: a missed refusal is a confident wrong answer,
   a false refusal is a visible non-answer.

3. **The cost and cache comparison between the arms is confounded and must not be read.**
   v1 reported 0 cached input tokens, v2 reported 24,121. The tempting conclusion — v2's larger
   prompt crossed OpenAI's 1024-token caching floor — is wrong. v1 ran *first*, against a cold
   provider cache; v2 ran ~200s later, and the three judge prompts are byte-identical between
   the arms, so v2 harvested a prefix that v1 paid to warm. Known issue 3b predicted this
   precisely, and it would have produced a wrong conclusion here if read straight.

**Also found, and worth its own attention: a judge call was lost to truncation.** v1's arm
reports faithfulness over **41 rows, not 42** — one `judge_faithfulness` call returned invalid
JSON, truncated mid-string at column 3526. That is known issue 10's `max_output_tokens`
truncation arriving on the *judge* path rather than the answer path, where it had already been
diagnosed. v1's 0.976 and v2's 0.988 are therefore over different row counts and are not
strictly comparable. `MAX_JUDGE_TOKENS` needs raising — see the new known issue.

### The token budget's estimator, measured (step 1 follow-up)

`CHARS_PER_TOKEN = 3.6` is **conservative, and by much less than first claimed.** Measured on
one real answer call per prompt version, estimate against the provider's reported
`input_tokens`:

| prompt | system block | estimated input | actual input | drift | implied chars/token |
|---|---|---|---|---|---|
| answer.v1 | 2018 chars | 3497 | 3363 | **+134** | 3.72 |
| answer.v2 | 3349 chars | 3867 | 3666 | **+201** | 3.77 |

So the real ratio on this prompt-and-passage mix is ~3.75, against an assumed 3.6 — roughly
**4% headroom, not the "third" an earlier reading of known issue 7 suggested.** Known issue 7's
"~425 tokens" for the stable block is stale: the real figure is ~540 for v1 and ~890 for v2,
and **neither crosses the 1024-token caching floor.** Prompt caching therefore still does not
pay for the answer call, and v2 being bigger did not change that.

4% is thinner than intended but still the safe side, and `context_budget_safety = 0.9` sits in
front of it. Left at 3.6 rather than tuned on two data points; the `budget.drift` log line is
in place to accumulate more. **Note the runner suppresses it:** `evals.runners.run` logs at
WARNING unless `-v` is passed, which is why the eval runs produced no drift lines and this
measurement needed a separate script.

### Settling query rewriting (known issue 19, closed)

Phase 3 dropped the rewrite stage and `advanced.yaml` kept it on anyway, because the flag that
drives multi-query is `query_rewrite_enabled` and the profile's own header claimed it was off.
Phase 3 had measured rewrite *without* HyDE (step 5a, byte-identical to step 4) and rewrite
*with* HyDE (step 5b, kept) and never isolated HyDE alone — so the stage ran for all of Phase 4
on a measurement that could not speak to it.

`hyde_only.yaml` is `advanced.yaml` with that one flag flipped and nothing else changed. Read
against a **repeat** of `advanced`, not the single recorded row, because HyDE is a model call in
front of the retriever and the retrieval metrics are stochastic for this family (known issue 3):

| | advanced run 1 | advanced run 2 | hyde_only |
|---|---|---|---|
| recall@5 | 78.2% | 78.2% | **79.0%** |
| context recall | 83.3% | 83.3% | **84.1%** |
| MRR | 0.565 | 0.570 | 0.574 |
| hit rate@5 | 88.1% | 90.5% | 92.9% |
| faithfulness | 0.976 | 0.988 | 0.964 |
| relevance | 1.000 | 1.000 | 0.988 |
| answer correctness | 0.821 | 0.821 | 0.821 |
| p50 latency | 3213 ms | 2793 ms | 2706 ms |
| wall clock | 265 s | 259 s | **213 s** |

**Neutral, so it is off.** Nothing moved beyond what two identical runs of `advanced` already
move between themselves — hit rate@5 shifted 2.4pp across that identical pair, which is the
whole of hyde_only's apparent gain on it. Answer correctness is identical to the digit across
all three. The faithfulness and relevance dips are 0.024 and 0.012, inside the ±0.05 band.

**The saving is not the headline cost delta.** The run totals differ by $0.0109, but
`hyde_only` met 113,739 cached input tokens against the repeat's 3,235 (known issue 3b), so
most of that is prefix-cache warmth rather than work avoided. The honest number is the stage's
own spend, read per-stage out of the traces:

| stage | calls | input tok | output tok | cost | model time |
|---|---|---|---|---|---|
| `rewrite` (removed) | 42 | 15,867 | 2,823 | **$0.0030** | **114 s** |
| `answer` | 42 | 125,862 | 6,275 | — | 143 s |
| `judge_faithfulness` | 42 | 140,544 | 25,583 | — | 461 s |

So: **42 LLM calls, $0.0030 and 114s of model time per run**, for a stage that produced no extra
query at all in 33 of 42 rows — an average of 0.21 additional queries per question, bought with
a model call on every one. Wall clock fell 46s, which is more than 114s/4 concurrency predicts,
because a row that did get extra queries also paid for extra retrievals.

`advanced.yaml` now carries `query_rewrite_enabled: false`, which makes it settings-identical to
`hyde_only.yaml`. Both are kept: the ablation profile is the file that records which flag moved,
and `evals/reports/2026-10-21_hyde_only.*` is the report for the configuration `advanced` now
has. The dated `2026-10-21_advanced.*` on disk is the **repeat with rewrite still on**, which no
longer matches the profile of that name — the two runs are also kept beside it as
`2026-10-21_advanced.run1.*` and `.run2.*` (reports are gitignored, so this is local only).

### The Phase 4 confirmation run — tracing cost nothing measurable

Phase 4 changed no retrieval or generation behaviour; it added spans around existing calls and
`stage`/`prompt_version` arguments that are used for tracing and never sent to the provider.
The run exists to prove that, and it does — **every delta is inside a band this file already
documents**:

| | Phase 3 `advanced` | after tracing | reading |
|---|---|---|---|
| Recall@5 | 77.8% | 78.2% | +0.4pp — stochastic, see below |
| Context recall | 83.7% | 83.3% | −0.4pp — stochastic |
| MRR | 0.571 | 0.565 | −0.006 — stochastic |
| Faithfulness | 0.976 | 0.976 | identical |
| Correctness | 0.857 | 0.821 | −0.036, inside the ±0.05 judge band (issue 3) |
| p50 latency | 3550 ms | 3213 ms | **−337 ms** — tracing overhead is below the noise |
| Cost/query | $0.00091 | $0.00121 | +33%, and **0.8% prefix-cached against this run** |

**Retrieval metrics are not exact for this profile, and that correction matters.** This file has
said that "retrieval metrics are exact" — true when it was written, because it was comparing
`naive` runs, where the query reaching the retriever is the question itself. `advanced` has HyDE
and query rewriting in front of retrieval, and both are *model calls*: the query is different on
every run, so recall@k and MRR are stochastic here in a way they were not at the baseline. The
sub-percentage-point movement above is that, not a regression — and it means any future
`advanced`-to-`advanced` comparison needs more than one run before a small delta means anything.

The cost rise is known issue 3b doing exactly what it says: **0.8% of input tokens were served
from the provider's prefix cache in this run**, against the 52% that produced some earlier
figures. Quoting the cost without that number beside it would read as a 33% regression caused by
tracing, which it is not.

### Step 1 — semantic chunking + parent-child (slot 2, `semantic.yaml`)

**Verdict: kept, and carried into step 2.** Context recall 61.9% → **76.2% (+14.3pp)**, and it
did that while sending the generator *fewer* passages — 4.0 against 5.0, because children of one
parent dedupe. That is the parent-child thesis working exactly as advertised: search a precise
unit, read a complete one.

Context recall by category, the comparable view:

| Category | naive | step 1 | |
|---|---|---|---|
| single-hop | 58.3% | 83.3% | +25.0 |
| multi-hop | 37.5% | 60.4% | +22.9 |
| unanswerable | 83.3% | 100.0% | +16.7 |
| temporal | 64.3% | 78.6% | +14.3 |
| routing | 44.4% | 44.4% | — |
| **numeric** | **86.1%** | **72.2%** | **−13.9** |

Multi-hop was the weakest category in the baseline and gained the most, which is the result
that justifies the step.

**The one real regression is numeric, and it is not an artefact** — it is −13.9pp on the
comparable metric too, and MRR fell 0.492 → 0.437 overall. More, shorter units make a denser
field of near-identical candidates, and the merchant ledger is the worst case because its rows
genuinely do all resemble each other. Dense retrieval cannot distinguish `946` from `1,034` at
all. **This is precisely what step 3's keyword half is for, so numeric recall is the number to
watch there.** If hybrid does not recover it, the parent-child child size is the thing to revisit.

Judges moved within the noise band and should not be read. Cost rose ~12% cache-neutral, from
longer context passages.

### Step 2 — contextual retrieval prefixes (slot 3, `contextual.yaml`)

**Verdict: kept, and the best configuration so far.** A fast model writes 1–2 sentences per
child saying what the passage is and where it sits; that prefix is prepended before embedding
and before the tsvector, while the stored chunk text stays verbatim.

It is the strongest single result in the project so far, and it **repaired step 1's
regressions rather than trading against them**:

| | naive | step 1 | step 2 |
|---|---|---|---|
| Context recall | 61.9% | 76.2% | **79.8%** |
| Recall@5 | 61.9% | 63.9% | **69.4%** |
| MRR | 0.492 | 0.437 | **0.506** |
| Hit rate@5 | 73.8% | 73.8% | **81.0%** |
| Context precision | 19.5% | 17.6% | 19.5% |
| Passages to model | 5.0 | 4.0 | **3.6** |
| False refusals | 4 | 4 | **3** |

**MRR recovered past the baseline** (0.437 → 0.506), and **numeric context recall came all the
way back**: 86.1% → 72.2% → 86.1%. Step 1's notes predicted step 3's keyword half would be
needed for that; the prefix did it instead, by giving a bare ledger row back the words naming
what it is a row of. Routing — the second-weakest baseline category — is up 22.2pp on context
recall.

Context recall by category, step 2 against the baseline: single-hop +33.3, routing +22.2,
unanswerable +16.7, temporal +14.3, multi-hop +10.4, numeric unchanged.

**The one trade-off: multi-hop fell back from step 1's 60.4% to 47.9%** (still +10.4 over
baseline). The obvious hypothesis — prefixes pull every chunk toward its document's general
topic, concentrating results in fewer documents — is only half supported: mean distinct
documents per row does drop (3.40 → 2.88), but on multi-hop rows specifically it barely moves
(3.50 → 3.38). So the cause is **not established**. Worth revisiting at step 4 (rerank) and
step 5 (multi-query), both of which bear directly on it.

**Cost: +6% cache-neutral** ($0.0348 → $0.0368 at list rate), plus a one-off ingestion cost of
104 fast-model calls. Re-ingestion is free: prefixes are cached by a hash of prompt version,
document and chunk text.

### Step 3 — keyword FTS + RRF hybrid (slot 3, `hybrid.yaml`)

**Verdict: kept. The best configuration so far, and best on every headline metric** — context
recall 81.0%, recall@5 73.0%, MRR 0.528, hit rate 83.3%, context precision 20.5% (the only
profile to beat the baseline on precision). Cost is flat against step 2 (+3% cache-neutral),
latency unchanged.

**The ablation is what makes the row mean something.** `keyword_only` is a worse retriever
overall (68.3% context recall against step 2's 79.8%) — nobody should ship it. But it is not
uniformly worse, and that is the entire basis of the fusion:

| Context recall | naive | step 2 | keyword only | **step 3** |
|---|---|---|---|---|
| routing | 44.4% | 66.7% | **100.0%** | **100.0%** |
| temporal | 64.3% | 78.6% | **92.9%** | **92.9%** |
| numeric | 86.1% | **86.1%** | 72.2% | **86.1%** |
| single-hop | 58.3% | **91.7%** | 66.7% | **91.7%** |
| unanswerable | 83.3% | **100.0%** | 50.0% | 83.3% |
| multi-hop | 37.5% | 47.9% | 47.9% | 41.7% |

Hybrid takes the lexical half's wins on routing and temporal **and** keeps the dense half's
wins on numeric and single-hop. That is RRF doing exactly what it is for, and it is only
visible because the ablation was run.

**Two things it did not fix, stated plainly.**

1. **Multi-hop is now the clear outstanding problem, and it has got worse with every step
   since step 1**: 37.5% → 60.4% (step 1) → 47.9% (step 2) → 41.7% (step 3). Step 1's larger
   deduped parents remain the best thing this project has done for multi-hop, and the two
   techniques since have each eroded it while winning elsewhere. No explanation has survived
   checking. **This is the thing step 5's multi-query decomposition has to address**, and it
   should be judged on this number specifically.
2. **Unanswerable fell 100% → 83.3%.** OR-ed keyword terms always match *something*, so rows
   that should retrieve nothing now retrieve plausible-looking passages. Refusal recall stayed
   at 100% and false refusals actually improved (4 → 3), so the model is still refusing
   correctly — but the guardrail is the model's judgement rather than an empty result set,
   which is a weaker place for it to live. Worth remembering in Phase 6.

### `judge_faithfulness.v2` and the hand-labelled rows (done before step 4)

Deferred twice on the grounds that steps 1–3 were decided by exact retrieval metrics. Done now
because step 4 is the first step whose case may rest on answer quality.

**v2 does not score better than v1. It was adopted anyway, and the distinction matters.**
Agreement with the labels is 9/10 for both, stable over three repeats. What v2 adds is that
its verdicts can be checked: it decomposes the answer into claims and must copy a verbatim
supporting span for each one, and `verify_spans` tests those quotes against the passages. A v1
score was unfalsifiable prose; a v2 score arrives with evidence the harness audits.

On the full 42-row hybrid run v2 extracted **131 claims, verified 114 spans, 0 unquoted, and
caught 2 quotes that do not appear in the passages** — invisible to v1 by construction. Both
turned out to be *stitched*, not invented: real text joined non-contiguously, and in `gq-013`
joined **across two different documents**. That is a weaker failure than fabricating a fact
and a real one, because a stitched span is exactly how a judge would justify a cross-document
inference that neither passage makes. Described accurately rather than as "the judge invented
evidence", which it did not.

Two costs, both stated rather than buried:

- **+36% at list price** on a full run ($0.0378 → $0.0513), from the claim decomposition.
- **Faithfulness reads 0.071 lower** on identical retrieval (0.964 → 0.893). v2 is stricter in
  practice despite identical label agreement, so the scoreboard column is discontinuous and
  the bridge row above marks where.

Building the validation set also found a bug in the validator: the span check initially
reported two fabricated quotes that were a judge dropping markdown asterisks and an empty
quote on a refusal row. Both were defects in the checker, now fixed and tested, and the
distinction between "unquoted" and "fabricated" is kept in the data — an over-sensitive
detector of dishonesty is its own kind of dishonest.

### Step 4 — cross-encoder reranking (`hybrid_rerank.yaml`)

**Verdict: kept, but it is the thinnest win of the phase.** Context recall 81.0% → 82.1%
(+1.2pp) for **+34% latency** (p50 2612 → 3502 ms). Ranking metrics moved much more than the
headline: recall@5 73.0% → 77.8%, MRR 0.528 → 0.563, hit rate 83.3% → 88.1%.

**That gap is the finding.** The reranker is clearly doing its job on ranking, and most of
that gain does not reach the generator — because parent expansion already collapses several
children into the same parents, so reordering children often changes nothing about what is
read. **Once you expand to parents, improving child ranking has diminishing returns**, and
measuring only recall@5 here would have overstated the technique by 4x.

**The first implementation lost on every metric, and the reason is worth keeping.** It scored
`chunk.text` — the bare verbatim slice — giving the cross-encoder the un-contextualised chunk
while retrieval had matched against the contextualised one. That is precisely the handicap
step 2 removed from the embedder, reintroduced one stage later:

| | bare text | indexed text |
|---|---|---|
| recall@5 | 69.4% | 77.8% |
| context recall | 76.2% | 82.1% |
| MRR | 0.510 | 0.563 |
| refusal recall | 83.3% | 100% |

Both rows are on the scoreboard. The losing one is kept because "we added a reranker and
retrieval got worse" is the most common way a correct technique gets abandoned.

By category: unanswerable 83.3% → 100%, single-hop 91.7% → 100%, multi-hop 41.7% → 47.9%
(the first gain on that category since step 1). **Temporal regressed hard, 92.9% → 64.3%**,
and is unexplained — it is the third unexplained category movement in this phase and the
second involving temporal or multi-hop.

Cost is flat at list price ($0.0513 → $0.0490); the spend here is latency, not money, because
the reranker is local.

### Steps 5–7, and what Phase 3 ended up at

**Step 5 — query transforms. HyDE won; multi-query did not, on its own.**

HyDE — embed a hypothetical answer beside the question — was the single largest retrieval gain
after step 1: context recall 82.1% → 84.9%, hit rate 88.1% → 92.9%, at flat cost. Questions
and passages are written in different registers, and a fabricated answer sits in the register
of the thing being searched for. It was also the first real gain on **multi-hop** since step 1
(47.9% → 56.2%). The draft is embedded and discarded — never shown, never indexed, never
citable — which is the only basis on which asking a model to invent corpus-shaped text belongs
in this project.

Multi-query on its own produced **byte-identical retrieval on all 42 rows**. Traced rather
than assumed: the extra queries do shift the candidate pool (18–20 of 20 overlap), and the
cross-encoder then re-sorts that pool to the same five. On 104 searchable chunks a
20-candidate pool already holds what is relevant. **Combined with HyDE it does appear to
help** (multi-hop 47.9% → 56.2% in `advanced`), but the three HyDE-containing runs returned
context recall of 82.1 / 83.7 / 84.9 — the generated probe varies between runs — so rewrite's
individual contribution sits inside that band and **is not separable at this sample size.**
It is left on in `advanced` because that is the configuration actually measured best, not
because its value is established.

**Step 6 — query router and metadata filters. Lost, exactly as designed against.**
Context recall 84.9% → 76.6%, multi-hop −12.4pp, temporal −21.4pp. The asymmetry the module
was built around is the explanation: a correct narrow route saves a few slots, a wrong one
hides the answer with no second pass. Rules fire only on unambiguous wording, the classifier
is told to prefer `general`, the filters are deliberately wider than the obvious answer, and
`general` applies no WHERE clause — and it still lost, because on ten documents multi-hop
evidence routinely spans document types. Committed and switched off: it is a real technique
and the eval is what says not to enable it here.

**Step 7 — semantic cache. The threshold in ADR-012 is unsafe, and no threshold is safe.**
`make cache` probes for the failure that matters — not hit rate but wrong hits — and found
one immediately:

| | cosine | |
|---|---|---|
| `...tithe rate in 1143?` vs `...1144?` | **0.954** | wrong answer, above threshold |
| `population of Halvern Reach?` paraphrased | **0.955** | right answer |

One thousandth apart. Raise the threshold and the cache stops working; lower it and it serves
1143's answer for 1144 — and the corpus changes that rate between books, so the wrong answer
is plausible. **The fix is not a better threshold.** A hit now also requires both questions to
mention the same figures: an exact test on precisely the tokens that distinguish
otherwise-identical questions, and precisely where an embedding is weakest, since `1143` and
`1144` are neighbours as tokens and unrelated as facts. All six probes resolve correctly, with
no cross-user leak and no stale version served. The general rule, worth carrying to Phases 5–7:
*where a soft signal cannot resolve a distinction, put a hard one beside it rather than tuning
the soft one harder.*

### Phase 3 overall: `naive` → `advanced`

| | naive | advanced | |
|---|---|---|---|
| **Context recall** | 61.9% | **83.7%** | **+21.8** |
| Recall@5 | 61.9% | 77.8% | +15.9 |
| Hit rate@5 | 73.8% | 90.5% | +16.7 |
| MRR | 0.492 | 0.571 | +0.079 |
| Context precision | 19.5% | 21.0% | +1.5 |
| Passages to the model | 5.0 | 3.9 | −1.1 |
| False refusals | 4 | 2 | −2 |

By category: routing +55.6, single-hop +41.7, multi-hop +18.8, unanswerable +16.7, numeric
and temporal unchanged. Latency p50 2734 → 3550 ms (+30%), cost +45% at list price.

**Two categories did not move across the whole phase.** Numeric ended exactly where it
started (86.1%) having dipped to 50% at step 1 and been recovered twice. Temporal also ended
at 64.3%, having reached 92.9% at step 3 before step 4 cost it 28.6pp — that regression is
still unexplained and is the clearest open retrieval question going into Phase 4.

Weakest categories by recall@5 in the baseline: multi-hop 37.5%, routing 44.4%. Strongest:
numeric 86.1%.
Weakest by correctness: routing 0.33, multi-hop 0.62.

**Three caveats that govern how this scoreboard may be read.**

1. **Judge scores carry run-to-run noise, and the band is wider than ±0.05.** Five runs on
   identical configuration have now given faithfulness 0.940 / 0.952 / 0.964 / 0.988 / 1.000 and
   correctness 0.726 / 0.762 / 0.774 / 0.774 / 0.821. A Phase 3 change that moves a judge score
   by 0.05 has moved nothing measurable; a change that moves recall by 0.02 has.

   Those five runs were on a configuration with no model call in front of retrieval, where
   recall and MRR do repeat to the digit. **That is a property of those profiles, not of the
   metrics** — `advanced` puts HyDE and a rewrite before the retriever, and they move by
   fractions of a point between identical runs. See known issue 3.
2. **The faithfulness judge is better than this file has been claiming** — see known issue 2,
   now rewritten against evidence rather than one anecdote. Measured against ten hand-labelled
   rows, it agrees **9/10**, including the exact failure shape this file was built around.
   Faithfulness is still same-family self-evaluation (ADR-017) and still worth less than an
   exact metric, but it is not the rubber stamp the earlier entries claimed.

3. **Cost per run swings ~50% on prefix-cache warmth alone.** Three runs over the same index
   doing the same work, with input-token counts within 0.1% of each other, cost $0.0212,
   $0.0348 and $0.0232 — the difference being that the provider served 0% to 51.2% of input
   tokens from its cache depending on how recently the last run finished. The report now records
   `total_cached_tokens` so a cost delta can be checked before it is read as a result. **Do not
   put a cost delta in a Phase 3 scoreboard row without the cached share beside it.**

**Caveat to carry forward (ADR-017):** `LLM_STRONG` is also the judge model, so judge scores are
same-family self-evaluation. Read faithfulness and correctness as **relative** — valid for
comparing profiles against each other, not as ground truth.

## Decisions taken in chats

- **ADR-024** — tracing context travels in contextvars, through one process-wide tracer. A
  trace id is needed by the logger, by every span and by the response, so threading it would
  put a tracing parameter on functions whose subject is retrieval, and the first call site to
  forget it would orphan everything below. This is also how OpenTelemetry — and so the Langfuse
  SDK — already propagates span context, which is why nested spans need no parent passed.
  Two costs were paid in this chat and are written into the ADR: a contextvar set inside a
  `@contextmanager` propagates to the caller and must be restored, and an async generator's
  frame runs in whichever task calls `asend`.
- **ADR-025** — a streamed answer is the *same* structured answer, with the `answer` field
  lifted out of partial JSON, rather than a second prose-shaped prompt. One prompt and one
  schema means streaming is a transport choice and the eval scoreboard describes both paths.
  The alternative was easier to stream and would have forked generation into two paths whose
  quality could drift apart with no test noticing.
- **The semantic cache stays off, and is now instrumented.** Phase 4's build list asked for
  cache hit/miss *events*, not for the cache to be wired into `/v1/chat`. Turning it on is a
  correctness question that deserves its own step rather than arriving as a side effect of the
  observability phase. The events are in place for when it does.
- **Tracing is on by default** (`TRACING_ENABLED=true`) and inert without keys. A tracing
  system nobody switched on traces nothing, and the first thing you reach for it is the
  request that already went wrong; `build_tracer` returns the no-op when either key is blank,
  so CI and anyone without a Langfuse is unaffected by the default rather than broken by it.
- **Langfuse scores are a view; Postgres is the record.** Feedback is stored before it is
  scored, and the response's `scored` says honestly whether the trace was annotated, so a
  tracing outage cannot lose a user's rating.

- **ADR-020** — `fastembed` (ONNX Runtime) is the local embedding runtime, not
  `sentence-transformers`. ADR-004's *models* are unchanged; only the engine differs, so vectors
  and eval baselines are unaffected. `sentence-transformers` requires PyTorch, a full training
  framework, where Sutra needs one CPU forward pass over a frozen 67 MB model. Measured `.venv`
  growth: **141 MB** via ONNX against roughly 500 MB via PyTorch. Verified before accepting that
  `bge-small-en-v1.5` is supported at dim 384 with L2-normalised output (which is what makes
  `vector_cosine_ops` correct) and that `bge-reranker-base` is a supported cross-encoder, so
  Phase 3 needs no second runtime. Amends ADR-018 and the CLAUDE.md stack line.
- **ADR-021** — `index_version` is an integer **slot** chosen by config or an eval profile, with
  the chunking fingerprint stored in `chunks.metadata`. Resolves a three-way contradiction:
  ARCHITECTURE §3 described a hash, the §12 schema declares an Integer column, `config.py` held a
  plain int. Same fingerprint in the slot is a no-op, a different one is refused with the changed
  parameters named, a different slot is kept alongside.
- **ADR-022** — amends ADR-021 on two points found by building it. The fingerprint lives in
  `chunks.metadata` **only** (a `documents` row is shared by every slot, so a per-slot fingerprint
  there would be ambiguous), and it includes **`chunker_version`**, because parameters alone
  cannot say whether the code applying them changed. Not hypothetical: fixing the heading scan
  mid-phase changed the output for all ten documents while leaving the fingerprint identical, so
  `make ingest` reported "nothing to do" over stale chunks.
- **ADR-017** — LLM provider is **OpenAI**, `gpt-6-luna` for *both* `LLM_FAST` and `LLM_STRONG`.
  Verified rates: $0.10 / $0.50 per 1M input / output, $0.01 cached. Chosen on cost, because eval
  volume (~250 calls per `make eval`, re-run after every retrieval change) dominates spend, not
  chat traffic. Closes the Phase 0 open question. Declined a separate `LLM_JUDGE` role.
- **ADR-018** — recorded the concrete runtime dependency set, since ADR-015 covered only the dev
  toolchain. Dev-only `httpx2` (not `httpx`) because the `openai` SDK is built on httpx2.
- **ADR-019** — `make eval` is exempt from the ">200 LLM calls needs explicit OK" rule, which one
  eval run exceeds by construction.

## Review fixes (2026-10-22, before Phase 5)

A second external review raised four findings. **All four reproduced exactly as reported**,
and each is fixed with a test confirmed to fail against the pre-fix code. Three of the four
were invisible to the existing suite for structural reasons worth remembering.

**1. The root span swallowed the request's exception** (`observability/tracing.py`).
`LangfuseTracer.request` had its `yield` inside `try/except Exception`. In a
`@contextmanager` the body's exception is thrown *into* the generator at the yield point, so
that except caught the body's exception, not the tracer's — and yielding a second time from
the handler makes `contextlib` raise `RuntimeError: generator didn't stop after throw()`,
destroying the original.

Reproduced end to end against the live stack: `POST /v1/chat` with another user's session id
answered **500** with `generator didn't stop after throw()` in the log, where `routes_chat`
has an `except SessionNotFound` that should make it a 404. After the fix, 404. This was not
limited to that one path — *every* exception raised anywhere under a traced request was being
converted.

Two reasons it survived: the comment directly under the `except` asserted the opposite
("an exception from the wrapped body propagates through the inner `with` and is re-raised"),
and **`NoOpTracer.request` has a bare `yield`**, so the suite's own 404 tests passed while the
real tracer was broken. Setup is now separated from the yield — only the three context enters
are guarded, through `ExitStack` so one `try` can cover them without nesting the yield inside
it. Unwinding through the stack still hands the exception to the observation's `__exit__`, so
the span is closed and marked as before, and the degrade-to-no-op guarantee is kept and tested.

**2. The feedback score blocked the event loop** (`api/routes_feedback.py`). `tracer.score()`
is synchronous and flushes inline — the flush is deliberate (feedback arrives on its own
request with nothing after it to trigger a send) but it is a blocking provider round trip
inside an async handler. **Measured: 6–9ms against a local Langfuse**, unbounded against a
remote one, and for its duration every concurrent SSE stream on the loop is parked. Now
`asyncio.to_thread`, the same treatment `vector_search` gives the ONNX embedder and for the
same reason — CLAUDE.md's "no blocking calls inside request handlers". The test asserts the
*thread*, because asserting the latency would be a race.

**3. A streamed response that did not complete was billed at zero** (`llm/adapters.py`). The
stream loop matched only `response.completed`, so `response.incomplete` and `response.failed`
fell through to the "no terminal event" fallback — a `RawResult` with zero tokens. Confirmed
by reverting just that one line: `assert 0 == 900` on the input tokens.

**This is exactly the known-issue-10 failure, priced at nothing.** `max_output_tokens` budgets
reasoning and visible output together, so a cap sized for the answer gets eaten by reasoning
and the response returns `incomplete` — and that is the call whose cost you most want to see.
All three terminal events carry the same `response`, so one branch handles them. It
deliberately does **not** raise, unlike `complete`/`structured`: tokens are already with the
caller, which is why there are no retries on this path either, and ADR-025 lets a truncated
stream still yield the partial answer it managed. The defect was the accounting, not the
tolerance. `RawResult` gains `status` and `incomplete_reason`, the client logs a warning
naming `max_output_tokens` as the remedy, and the status goes on the generation span's
metadata — so a truncation that was invisible in the trace, the logs *and* the cost is now in
all three. The zero-usage fallback is now only a genuinely terminated stream, reported as
`no_terminal_event`.

**4. The generation paths and the judges never got `client_scope`** (`generation/answer.py`,
`generation/stream.py`, `evals/metrics/judges.py`). `client_scope` exists because
`client or LLMClient(...)` leaks the provider's connection pool on every call that falls
through to it. The router and query transforms were moved onto it in Phase 3; these three were
missed. **Severity is lower than the reviewer's framing implies** — `/v1/chat` and the eval
runner both inject a client, so the hot paths were never leaking. The live leak is
`generation/demo.py`, one pool per `ask()`, plus the certainty that the next caller added
would leak too.

Three things fell out of fixing it properly:

- **The no-context short circuit now builds no client at all.** It sat *below*
  `client or LLMClient(settings)`, so the one path that makes no model call was the one
  building a provider SDK and a pool for nothing.
- On the streaming path the scope wraps the whole stream, not one call, since the client must
  outlive the last token.
- **`run_judges` opens one scope around all three judges** instead of each opening its own.
  They fan out concurrently, so a row with no injected client was opening three pools to make
  three calls — the same reasoning as `_search_inner`'s shared client.

**A note on the tests themselves.** The finding-4 tests initially patched `LLMClient` only in
`sutra.llm.client`. `client_scope` resolves the name there, but `answer.py` imports it
directly — so against the *unfixed* code the patch missed and the test built a real client and
called the OpenAI API. It failed with a 401 rather than an assertion. Both names are now
patched: a test that fails by making a paid API call is not a test, and it would have been a
live one with a valid key in `.env`.

## Review fixes (2026-10-20, before Phase 3 step 1)

An external code review raised seven findings. All seven reproduced; one detail in the report
was wrong (the token counter saturates at 512, not at 3). Fixed in six commits, each with a
test, lint and types clean. Three of the fixes were verified as negative controls — the test
was confirmed to fail with the fix reverted, which for the slot bug meant reproducing the exact
`uq_chunk_position` violation the review predicted.

| Finding | Fix |
|---|---|
| `complete()`/`structured()` ignored `model=` and answered from `LLM_STRONG` | model threaded through `_call_with_retries`; an explicit model now gets a chain of one |
| eval token totals summed judge calls while cost covered answer calls too | totals cover both; answer/judge split kept beside them |
| `count_tokens` saturated at 512, so the truncation guard could never fire | counting runs on a truncation-free copy of the tokenizer |
| figure check credited `143` to `1143`, disarming the cross-check on the judge | support tested against the set of figures the context states |
| a slot holding chunks without `index_fingerprint` read as empty, skipping the delete | occupancy decided by row existence, not fingerprint readability |
| blocking ONNX `embed_query` on the event loop in `/v1/chat` | awaited in a thread; test asserts on thread identity, not timing |
| `--limit` with `--baseline` overwrote the committed reference | the CLI refuses the combination |

Two things worth carrying forward that the review did not ask for:

- **The chunking fix is a no-op on the current corpus, verified rather than assumed.** Slot 1
  re-chunks to the same 49 chunks with identical text and token counts, so the committed
  baseline's index still stands and no re-ingest was needed. At `chunk_tokens=400` any piece
  reporting 512 was already over the split threshold. It starts mattering in Phase 3, where
  semantic chunking makes chunk size data-dependent.
- **Correcting the token totals exposed a second problem** — cost swinging 50% on prefix-cache
  warmth with nothing in the report able to explain it. `total_cached_tokens` was added, and
  the reported cost now reconciles to the cent against the cached rate.

The blocking-embedder fix matters more from Phase 3 on than it did here: the cross-encoder
reranker is far heavier than the embedder and lands on the same event loop. `embed_passages` in
ingestion is still synchronous and deliberately so — it runs in a CLI script, not a request.

## What was built (Phase 5, steps 1-3)

**Step 1 — the token budget** (`f45f743`)

- `src/sutra/generation/budget.py` — `fit_context`, `BudgetPlan`, `BudgetParts`, `HistoryTurn`,
  `estimate_tokens`. ARCHITECTURE §5's budget and §5's trim order. The system block and the
  question are not trimmable; the chunk trim floors at `context_min_chunks`.
- `src/sutra/config.py` — `MODEL_CONTEXT_WINDOWS`, `DEFAULT_CONTEXT_WINDOW_TOKENS`,
  `context_window_tokens()`, and `answer_reserve_tokens` / `context_budget_safety` /
  `context_min_chunks`. The window table errs *small* where `MODEL_PRICES` and
  `MODEL_CAPABILITIES` err permissive: an unknown price gives a wrong number, an unknown
  window gives a rejected request.
- `answer.py` / `stream.py` — both fit identically (ADR-025), `MAX_ANSWER_TOKENS` deleted,
  `split_halves` extracted so the budget can price the system block before interpolation.
  Citations resolve against the *trimmed* chunk list.
- `answer.py` logs `budget.drift` per call: estimate beside the provider's `input_tokens`.
- Tests: 13 new.

**Step 2 — routing and fallback made auditable** (`78e03b3`)

- `GenerationRecord.stage` — the span was already named by stage, the record was not, so
  "the answer ran strong and the rewrite ran fast" was unfalsifiable while both roles resolve
  to one model. Threaded through all six `_trace` call sites, success and failure.
- `llm_fallback_models` now defaults to `[]`. See known issue 9 for what that fixed.
- `contextualize.py` — the ingest-time prefix call was the last generation still arriving as
  `llm.call`, and the highest-volume one at one call per chunk. Now `stage="context_prefix"`
  with its prompt version.
- `tests/test_model_routing.py` — §5's split pinned stage by stage, plus the chain's shape.
- `tests/test_generation.py` — the stable block is message zero (a prefix cache needs it first,
  and nothing asserted the position).

**Step 3 — `answer.v2`, and grading moved to fast** (`d2d9a04`, numbers in the commit that follows it)

- `prompts/answer.v2.md` + `evals/profiles/answer_v2.yaml` — one hypothesis, one variable, the
  risk named in the profile header before the run. **Not adopted**; see the A/B section above.
- `evals/metrics/judges.py` — `role="fast"`, as §5 puts grading there, with the consequence
  written at the call site. Confirmed 9/10 against the hand labels after the move.
- `tests/test_generation.py` — the contract test now runs over *every* shipped answer version,
  and a test compares v1's and v2's untouched sections so the A/B stays one variable.

## What was built (after Phase 4, before Phase 5)

Two known issues the first real look at the traces turned up (19 and 20), closed together.

- `src/sutra/observability/tracing.py` — `_GatedStage`, which wraps every span handed to a
  call site so `span.update(output=...)` obeys `TRACING_CAPTURE_CONTENT`. The flag previously
  gated inputs only, so the `generate` span shipped the full answer text with capture off.
  `on_generation` now sends `input` and `output` through the same gate.
- `src/sutra/llm/types.py` — `GenerationRecord.input` / `.output`.
- `src/sutra/llm/client.py` — `_as_prompt` (messages as dicts, the shape Langfuse renders as a
  chat log) and content threaded through `complete`, `structured`, `stream` and both failure
  paths in `_call_with_retries`.
- `src/sutra/retrieval/search.py` — `fusion` gains its input; `multi_query_fusion` gains the
  n-query equivalent of `fusion`'s breakdown, including `contributed`; `embed_query` gains an
  output.
- `src/sutra/retrieval/expand.py` — `expand_to_parents` gains `parents_requested` /
  `parents_found`, which is where a dangling parent id becomes visible on the span rather than
  only in a log warning.
- `evals/runners/run_rag.py` — the `eval.row` root span gains an output: the answer, the
  refusal flag, recall, context recall and the judge scores, so scanning a run for the rows
  that went wrong does not mean opening 42 traces.
- `evals/profiles/hyde_only.yaml` — the HyDE-without-rewrite ablation Phase 3 never ran.
- `evals/profiles/advanced.yaml` — `query_rewrite_enabled: false`, and a header that now
  states what was measured instead of the claim that was wrong.
- Tests: 11 new (468 total). The gate in both directions, metadata passing through either way,
  the proxy, and content on all three client paths plus the retryable and non-retryable
  failures. The retrieval-side span additions are verified from `events_full` rather than by
  test — the suite has no span assertions for retrieval and adding DB-mocking scaffolding for
  observational metadata was not worth it.

## What was built (Phase 4)

- `src/sutra/observability/` — the module that was a stub since Phase 0:
  - `context.py` — trace id and request fields in contextvars, with `bind_trace_id` for the
    tracer, which learns the id only once the root span is open.
  - `logs.py` — a hand-rolled JSON formatter (no new dependency) plus `TextFormatter` for
    reading a local run by eye. `LOG_FORMAT` in config, default `json`.
  - `tracing.py` — `LangfuseTracer`, a complete `NoOpTracer`, `build_tracer()`, and the
    process-wide tracer accessors the pipeline calls.
- `src/sutra/generation/partial_json.py` — the incremental reader that makes a streamed
  structured answer possible (ADR-025).
- `src/sutra/generation/stream.py` — `generate_answer_stream`, the same prompt and schema as
  the buffered path.
- `src/sutra/api/streaming.py` — SSE frames, headers, and `guarded`.
- `src/sutra/api/routes_feedback.py` + `FeedbackRequest`/`FeedbackResponse`.
- Spans across the pipeline: `retrieve` (parent) over `route`, `transform_query`,
  `hybrid_search` → `vector_search`/`keyword_search`/`fusion`, `multi_query_fusion` and
  `rerank`; `expand_to_parents` and `generate` beside it; a `GENERATION` per provider call,
  named by stage and carrying its prompt version.
- Events: `cache.hit`/`cache.miss`, `retrieval.candidates`, `retrieval.expanded`.
- `LLMClient.complete/structured/stream` take `stage` and `prompt_version`;
  `GenerationRecord.prompt_version` is populated for the first time.
- `adapters.stream(schema=...)` — schema-constrained streaming via `responses.stream`.
- `messages.trace_id` is written; `ChatResponse.trace_id` is returned.
- `/v1/health` reports Langfuse but never fails on it.
- Eval runs group in Langfuse by `session_id` (the run id) with `eval` / `profile:` / `slot:`
  tags; the judges' generations are named.
- `docker-compose.yml` fixed so the observability profile actually starts.
- ADR-024 (contextvars + ambient tracer) and ADR-025 (streaming the structured answer).

## What was built (Phase 2)

- `src/sutra/ingestion/` — `parse.py` (front-matter), `chunk.py` (hand-rolled recursive chunker
  with three guaranteed invariants), `embed.py` (bge-small over ONNX behind an `Embedder`
  Protocol), `fingerprint.py` (ADR-021/022), `run.py` (`make ingest`), `README.md`
- `src/sutra/retrieval/search.py` — vector-only top-k within one index slot, plus `README.md`
  naming every absent technique and the phase it arrives in
- `prompts/answer.v1.md` and the three judge prompts, all split by a `<<<USER>>>` delimiter so
  the stable half stays uninterpolated
- `src/sutra/generation/` — `schema.py` (`AnswerDraft`, `Citation`, `Answer`), `answer.py`
  (context assembly, structured call, citation resolution), `demo.py` (one question end to end),
  `README.md`
- `src/sutra/api/` — `chat.py` (the turn: resolve session → save question → retrieve → generate
  → save answer, with **no DB connection held across the model call**), `routes_chat.py`,
  chat schemas, `README.md`
- `evals/metrics/` — `retrieval.py` (recall@k, hit rate, MRR, context precision),
  `judges.py` (three versioned LLM judges), `grounding.py` (the mechanical figure check)
- `evals/runners/` — `run_rag.py`, `report.py`, `run.py` (`make eval`); `evals/profiles/naive.yaml`
- `evals/check_coverage.py` + `make coverage` — asserts every cited quote survives chunking
- `evals/reports/baseline.json` and `baseline.md`, **committed** as the reference for later runs
- `evals/README.md` — the metric table and what each one does not tell you
- Config: chunking parameters, `retrieval_top_k`, judge prompt versions, and a
  `MODEL_CAPABILITIES` table beside `MODEL_PRICES`
- Tests: **156 passing** (up from 48), including both negative controls on the coverage guard

## What was built (Phase 1)

- `docs/world_bible.md` — the design record for "The Meridian Chronicles"
- `data/corpus/` — the **10 frozen corpus files** (100–200 lines each, full YAML front-matter),
  plus `README.md`, the freeze notice
- `evals/datasets/golden_qa.jsonl` — 42 rows, 72 verbatim source quotes
- `evals/datasets/attacks.jsonl` — 40 rows (30 attacks + 10 benign look-alikes)
- `evals/datasets/README.md` — the trap map: every trap, the row that tests it, and what a wrong
  answer looks like
- `evals/validate_datasets.py` + `make validate` — seven checks, including the quote check
- `evals/render_datasets.py` + `make datasets` — indented, readable mirrors of both datasets under
  `evals/datasets/readable/`, generated from the `.jsonl` sources, with drift checked by
  `make validate`
- `tests/test_datasets.py` — 9 tests, including negative controls proving the validator fails on an
  invented quote and on a stale mirror
- `evals` added to the ruff and pyright paths

## What was built (Phase 0)

- `pyproject.toml` (Python 3.12 pinned), ruff + pyright config, `Makefile` (12 targets),
  `.env.example`, `README.md`, `.gitignore`
- `docker-compose.yml` — Postgres 16 + pgvector in the default profile; the Langfuse v4 stack
  (its own Postgres, ClickHouse, Redis, MinIO) behind an `observability` profile
- `src/sutra/config.py` — `Settings` + the checked-in `MODEL_PRICES` table
- `src/sutra/db/` — async engine, `models.py` (all 9 tables from ARCHITECTURE §12),
  Alembic wired to `Settings`, migration `0001_initial_schema`
- `src/sutra/llm/` — `types.py`, `tracing.py` (no-op until Phase 4), `adapters.py`
  (OpenAI Responses API), `client.py` (`complete` / `stream` / `structured`, retries, fallback
  chain, token + cost accounting)
- `src/sutra/prompts.py` — versioned loader with YAML front-matter
- `src/sutra/api/` — app factory, `/v1/health`, Pydantic schemas
- `tests/` — 41 tests across config, LLM client, prompt loader, health endpoint
- Stub READMEs for all 13 module folders (db, llm, api now written)

## Verified in this chat

Against the live stack, not by inspection. **The first three items are from the chat that
closed known issues 19 and 20; the rest are Phase 4's.**

- **All six generation spans carry their prompt and their completion**, read out of
  `events_full` (the v4 list API omits both — known issue 15). `rewrite`, `hyde`, `answer`,
  `judge_faithfulness`, `judge_relevance`, `judge_correctness`, each non-zero on both sides.
  HyDE's invented passage is now readable on HyDE's own span.
- **`TRACING_CAPTURE_CONTENT=false` now closes both sides.** Re-ran with it off: every span
  reports 0 bytes of input and 0 of output while `usage_details` still arrives. Before the
  change the `generate` span's answer text went out regardless of the flag.
- **The rewrite stage's price, per stage rather than per run.** 42 calls, 15,867 input and
  2,823 output tokens, $0.0030, 114s of model time — queried from `events_full` grouped by
  `session_id` and `name`, which is how the cache-warmth confound in the run totals was caught.

- **The Langfuse stack starts** (4.67.0, all seven services healthy) and the auto-provisioned
  `LANGFUSE_INIT_*` keys authenticate. Two defects fixed to get there — see known issue 8.
- **A trace tree with real per-stage latency.** An `advanced` turn produces 23 observations:
  `chat` → `retrieve` → {`route` 3.8s, `transform_query` 4.3s (`rewrite` + `hyde`),
  `hybrid_search` 0.26s → {`embed_query`, `vector_search`, `keyword_search`, `fusion`},
  `rerank` 2.4s} and `generate` 5.3s → `answer`. Nesting comes from OTel's context; no parent
  is passed anywhere.
- **Generations carry what they should**: model, `{input, output, total, cache_read_input_tokens,
  reasoning}`, our own priced cost (not Langfuse's model-matching guess), and a qualified prompt
  version — `answer.v1`, `hyde.v1`, `query_transform.v1`, `judge_faithfulness.v2`.
- **Streaming from `curl -N`**: `meta` → 27 `token` frames → `citations` → `done`. The tokens
  are prose, not JSON fragments, and reassemble into exactly the answer.
- **Feedback end to end**: a −1 with a comment on a real turn, the row in Postgres and the score
  on that trace id in ClickHouse.
- **`/v1/health`** reports `tracing: langfuse at http://localhost:3000` and stays `ok`.
- **Eval runs group**: a 3-row run produced one `session_id`, 3 traces, 75 observations, tags
  `['eval','profile:advanced','slot:3']`.
- **`make eval` on `advanced`, the full 42 rows** → recall@5 78.2%, context recall 83.3%,
  MRR 0.565, faithfulness 0.976, correctness 0.821, p50 3213 ms, $0.0508 for the run at 0.8%
  prefix-cached. Every delta against Phase 3's row is inside a documented band — see the
  confirmation-run section above.
- `make coverage` → 72/72 cited quotes survive chunking · `make validate` → datasets consistent
  · `make cache` → all six probes correct, no cross-user leak, no stale version
- `make test` → **457 passed** · `make lint` clean (ruff + ruff format + pyright, 0 errors)
- `make judges` → **8/10 this run** (jl-03 and jl-08, both scored *too harsh*, never lenient).
  PROGRESS has recorded 9/10 with jl-03 as the single persistent disagreement; jl-08 is new
  here. Within the band known issue 3 describes — ten rows means one row is ten points — and
  the direction is the property that matters.

## Known issues / tech debt

0. **The corpus is frozen.** Editing `data/corpus/` invalidates every eval baseline from Phase 2
   on and can orphan one of the 72 cited quotes. If it must change: `make validate`,
   `make coverage`, repair the quotes, re-run `make eval`, and record a new baseline saying why
   the old numbers no longer compare.

1. **`make ingest` does not notice a chunker code change on its own.** `chunker_version`
   (ADR-022) is bumped by hand. Forget it and ingestion reports "nothing to do" over stale
   chunks. Deriving it from the source would fire on every comment edit, so the manual signal is
   deliberate — but it is a thing you can forget. **Bump it, or use `--force`, after any change
   to chunking behaviour.**

2. **The faithfulness judge is roughly 90% reliable, and the earlier claim here was wrong.**
   `make judges` grades it against `evals/datasets/judge_labels.jsonl` — ten hand-labelled rows
   (ARCHITECTURE §11), each a passages/question/answer triple with a score a human stands
   behind, verbatim corpus passages, and every premise behind a label ("this figure is absent
   from this passage") checked mechanically.

   **Result: v1 and v2 both agree 9/10**, stable across three repeat runs. The single
   persistent disagreement is `jl-03`, where both judges score correct ledger arithmetic 0.5
   and the label says 1.0 — they are too *harsh*, not too lenient. Neither version was lenient
   on any row.

   **This overturns what Phase 2 recorded.** The founding anecdote — faithfulness 1.00 on an
   answer whose central figure (`68,000`) was in none of its passages — is reproduced
   deliberately as `jl-02`, and both judge versions score it **0.0**, correctly. One bad roll
   on a 42-row run is entirely consistent with a judge that is right about nine times in ten;
   it is not evidence of a systematic rubber stamp, and this file treated it as though it
   were for two phases.

   What remains true: ten rows is a small set (one row is ten points), the labels are the
   author's, and ADR-017's same-family caveat is unaffected. Read faithfulness as a reasonable
   relative signal, not as ground truth and not as a known fiction.

3. **Judge scores carry run-to-run noise of at least ±0.05** (measured over five identical runs:
   faithfulness 0.940–1.000, correctness 0.726–0.821). Do not read a judge delta smaller than
   the band as a result.

   **Retrieval metrics are exact only when nothing model-backed sits in front of retrieval.**
   That held for every `naive`-family comparison, where the query reaching the retriever is the
   question itself. It does *not* hold for `advanced`: HyDE and query rewriting are model calls,
   so the query differs run to run and recall@k and MRR move with it (measured: ±0.4pp and
   ±0.006 on two runs of identical code). A small `advanced`-to-`advanced` delta needs repeats
   before it means anything.

3b. **Cost per run is not a stable measurement.** It swings ~50% on how warm the provider's
   prefix cache is (0% to 51.2% of input tokens served cached across three runs doing identical
   work). `total_cached_tokens` is now in the report; quote it alongside any cost number.

4. **`POST /v1/chat` has no authentication.** `SUTRA_API_KEY` exists in config and nothing checks
   it. Fine for a local dev server; belongs with the Phase 6 guardrail work. Also no rate
   limiting. (Streaming was the other half of this item and is done — Phase 4, ADR-025.)

5. ~~**No token budget.**~~ **Closed in Phase 5 step 1.** `generation/budget.py` implements
   ARCHITECTURE §5's budget — system + memory + history + chunks + answer reserve against
   window x safety — and §5's trim order: fewer chunks, then shorter history, then a truncated
   summary. The system block and the question are never trimmed; the chunk trim stops at
   `context_min_chunks` because zero passages is a refusal, not a cheaper answer.

   **Two caveats that remain.** It *estimates* rather than tokenizes (measured at ~4% headroom
   — see the estimator section above), and at k=5 nothing in traffic comes near the window, so
   the trims are exercised by tests at an artificial limit rather than by real turns. Memory
   (steps 4-5) is what makes history the first unbounded component and the trim order
   load-bearing.

6. **The HNSW index is not exercised.** With 49 chunks the planner filters on `index_version` and
   sorts instead (confirmed with `EXPLAIN`). The index is correct and will matter at volume, but
   nothing here tests HNSW's approximate-recall behaviour.

7. **Prompt caching does not pay yet, and the block is bigger than this said.** The stable
   system block measures **~540 tokens** for `answer.v1` (not the ~425 recorded here for two
   phases) and ~890 for `answer.v2`, against OpenAI's 1024-token minimum. **Neither crosses
   it**, so v2 being 1300 characters longer did not buy caching — measured, not assumed, in
   the estimator section above.

   The ordering remains the precondition rather than the saving, and a test now pins the part
   that was unasserted: the stable block is message *zero*, because a provider caches a prefix
   and a stable block sitting second never caches at all. Memory and guardrail instructions
   (steps 4-5, Phase 6) are still what would push it over.

8. ~~**The Langfuse stack has never been started.**~~ **Closed in Phase 4**, and it was right to
   budget time for it: two defects were waiting behind `docker compose config`, which
   type-checks the file and starts nothing. `minio/minio:latest` no longer pulls anonymously
   (now `cgr.dev/chainguard/minio`, as upstream Langfuse does), and
   `LANGFUSE_INIT_USER_EMAIL=dev@localhost` failed Langfuse's own validation for having no
   TLD, crash-looping the web container on every request. Both fixed and the stack verified
   end to end on Langfuse 4.67.0.

9. **Fallback chain and model routing are still structural no-ops — but no longer silent
   ones.** Both roles point at one model (ADR-017), so the role split changes no output and
   `LLM_FALLBACK_MODELS` has nothing to fall back to.

   **What Phase 5 step 2 found:** the default was `["gpt-6-luna"]`, which *is* the primary
   model, so `LLMClient._chain` de-duplicated it away and **no fallback has existed in any
   phase** — while the config and `.env.example` both read as though one did. The default is
   now empty, which states the same fact honestly, and a dropped duplicate is logged. The
   chain's real behaviour was already covered by tests, including that a fallback answer is
   priced at the answering model's rate.

   Routing is now *auditable*: `GenerationRecord` carries `stage`, and
   `tests/test_model_routing.py` pins §5's split stage by stage. Answering is the only stage
   on `strong`; grading moved to `fast` as §5 specifies, with the consequence recorded at the
   call site — **pointing `LLM_FAST` at a cheaper model silently weakens the instrument the
   scoreboard is measured with**, and `make judges` is the check to re-run if that happens.

10. **Two provider quirks are now encoded, and worth remembering.** `gpt-6-luna` rejects
    `temperature` outright (hence `MODEL_CAPABILITIES`), and `max_output_tokens` budgets
    *reasoning and visible output together* — a cap sized for the visible answer gets eaten by
    reasoning and the truncated body arrives as "Invalid JSON". `OpenAIAdapter` now detects that
    and names the real cause; `Usage.reasoning_tokens` records where the spend went.

    **The streaming path used to be exempt from all of that, and billed it at zero** — fixed
    in the 2026-10-22 review (finding 3). It still does not *raise* on a truncation, which is
    deliberate (ADR-025), but it now reports the tokens, the `status` and the
    `incomplete_reason` and logs the remedy.

11. Two justified `ASYNC109` lint exceptions (`src/sutra/llm/adapters.py`, `tests/**`), reasons
    recorded inline in `pyproject.toml`.

12. `requires-python = ">=3.12,<3.13"` while the host runs Python 3.13; uv fetches 3.12.

13. `docs/BUILD_PLAN.md` has uncommitted whitespace-only changes (blank lines around headings)
    from an editor's markdown formatter, not from any phase work. Left alone deliberately.

14. **An eval run with `--limit` overwrites the dated report of the same profile.** The runner
    guards `--baseline` against `--limit` but not the dated file, so a 2-row smoke run silently
    replaced `evals/reports/2026-10-21_advanced.*`. The numbers survived in this scoreboard and
    the full run regenerated it, but the guard is one-sided and should be extended — a limited
    run is not a result and should not be able to look like one.

15. **Langfuse v4 runs in `events_only` mode.** The legacy `GET /api/public/traces/{id}` is gone
    (404 with a pointer to the v4 docs), the `observations` ClickHouse table stays empty, and
    everything lands in `events_full`. Read traces through
    `/api/public/v2/observations?traceId=...` and `/api/public/v3/scores` — and note both list
    endpoints omit fields (model, usage, cost, input/output on observations; trace id and
    comment on scores), so verifying those needs ClickHouse or the UI, not the list API.

16. **The tracer is process-global mutable state** (ADR-024), set by `set_tracer()` at startup
    and by the eval runner. Accepted deliberately over threading `tracer=` through retrieval,
    but it means a caller that builds an `LLMClient` before `set_tracer` gets the no-op. The
    client defaulting to `get_tracer()` makes that the only remaining way to lose generations.

    **A second cost of this shape, learned the hard way:** `NoOpTracer` and `LangfuseTracer`
    can diverge in behaviour and the suite will not notice, because every test runs on the
    no-op. That is how the `request` contextmanager shipped broken for a whole phase
    (2026-10-22 review, finding 1) — the no-op has a bare `yield` and the real one did not.
    Any new guarantee about a tracer needs a test against `LangfuseTracer` itself, with the
    SDK faked, not only against the no-op.

19. ~~**`advanced.yaml` runs query rewriting, which Phase 3 measured as contributing nothing
    — and this file said it was off.**~~ **Closed.** Measured and dropped: `advanced.yaml` now
    has `query_rewrite_enabled: false`. The ablation Phase 3 never ran is `hyde_only.yaml`, and
    the comparison is a repeat of `advanced` rather than the recorded row because the metrics
    are stochastic here. Neutral on every quality metric (correctness identical to the digit);
    the stage's own spend was 42 LLM calls, $0.0030 and 114s of model time per run, producing
    no extra query at all in 33 of 42 rows. Full numbers in the scoreboard section above.

    One thing the closing measurement exposed: the headline cost delta between the two runs
    ($0.0109) is mostly prefix-cache warmth, not work avoided — 113,739 cached input tokens
    against 3,235. Known issue 3b said cost is not a stable measurement and this is the first
    time it would have produced a wrong conclusion if read straight.

20. ~~**No generation span records its prompt or its completion.**~~ **Closed.**
    `GenerationRecord` carries `input` and `output`, `LLMClient` fills them on all three paths
    (`complete`, `structured`, `stream`) and on the failure paths too — a non-retryable 400 is
    almost always about the prompt, which was the one thing not recorded. All six generations
    (`rewrite`, `hyde`, `answer`, the three judges) verified populated on both sides in
    `events_full`; HyDE's fabricated passage is now visible on its own span rather than only
    via the next stage's input.

    The smaller holes went with it: `fusion` has its input, so its
    `from_both`/`dense_only`/`lexical_only` breakdown reads as a ratio;
    `multi_query_fusion` has the equivalent over n queries, including `contributed` — how many
    of the extra queries put anything into the surviving pool, which is the per-request form of
    the measurement known issue 19 settled in aggregate; and `embed_query`,
    `expand_to_parents` and `eval.row` have outputs.

    **`TRACING_CAPTURE_CONTENT` did not mean what known issue 17 claimed, and now does.** It
    gated span *inputs* only. `generate` put the full answer text on its output through
    `span.update`, which the flag never touched, so "capture off" still shipped answers. Spans
    handed to call sites are now wrapped in a gate, so the next span someone adds cannot
    forget it. Verified: with the flag off every span reports 0 bytes on both sides while
    model, tokens and cost still arrive.

    Deliberate consequence: with the flag off, *structural* outputs go too (`{"fused": 20}`,
    `{"dims": 384}`). Telling structural from sensitive would be a judgement call at every
    call site, and `input` has always behaved this way — `input={"dense": 20, "lexical": 20}`
    was already dropped. Consistency over cleverness.

21. **`sutra.observability.tracing.generation_scope` is dead code.** Noticed while wiring known
    issue 20. `LLMClient` opens its generation spans through `self._tracer.stage` directly and
    nothing calls `generation_scope`; it also has no `input` parameter, so it was never the
    seam the content fix could have used. Left in place rather than deleted — removing it needs
    the usual ask — but it should go.

22. **`MAX_JUDGE_TOKENS` is too small, and a judge that truncates scores nothing.** On the
    2026-10-22 `advanced` run, one of 42 `judge_faithfulness` calls returned invalid JSON —
    truncated mid-string at column 3526 — so that row has no faithfulness score and the run's
    mean is over 41 rows. This is known issue 10's reasoning-plus-output budget arriving on the
    judge path, where `judge_faithfulness.v2` is the most exposed prompt in the system: it must
    quote a supporting span per claim, so its output grows with the number of claims in the
    answer. The failure is handled (the row is counted as failed, not scored 0, which would
    have been worse) but it is silent in the headline number and it biases *which* rows get
    graded — the longest, most claim-dense answers are the ones that fall out.

    **Raise it, and re-run before trusting a faithfulness delta of this size.** Phase 3 hit the
    same wall on the answer path and the fix was the same: size the cap for the worst case, not
    the median.

23. **An eval run's `budget.drift` lines are invisible by default.** `evals.runners.run`
    configures logging at WARNING unless `-v` is passed, so the estimator-versus-actual line
    added in Phase 5 step 1 produced nothing across two full runs. Measuring the estimator
    needed a separate one-call script. Either the runner should log INFO by default or the
    drift should go somewhere the report can see it; today it goes nowhere unless you ask.

18. **The `meta` SSE frame departs from ARCHITECTURE §5 in two documented ways.** §5 describes
    it as "trace_id, cost, cache_hit". Cost is on `done` instead, because `meta` is sent before
    the answer call and the cost is not known until the provider finishes — a field that is
    always null at the moment it is sent is worse than one read from the terminal frame. And
    `cache_hit` is structurally `false`, since the semantic cache is not wired into the turn;
    it is sent anyway so a client written against §5 needs no change when the cache is turned
    on. Neither is a spec change, so no ADR — but if §5 is ever revised, revise it to say this.

17. **Span inputs and outputs carry the question, the retrieved passages, and now every
    prompt and completion.** Harmless on a fictional corpus and `TRACING_CAPTURE_CONTENT=false`
    turns it off — which, since known issue 20, is actually true of the output side as well.
    The default is on and would be the wrong default for real data, and the content now
    captured is considerably more than when this item was written.

## Repo
- GitHub: https://github.com/rohhitrz/sutra (public)

## Next step

**Phase 5 steps 1-3 are done, measured and pushed.** Continue with **step 4 — short-term
memory**. The remaining steps, unchanged from the approved plan:

- **Step 4 — short-term memory.** Window + rolling summary. `memory/window.py`,
  `memory/summarize.py`, a `summarize.v1` prompt on the **fast** role (§5), compaction resuming
  from `session_summaries.covers_until_message_id`. Wire it into `chat.py`'s turn assembly and
  into `fit_context`'s `summary=` / `history=` parameters, which already exist and are empty.
- **Step 5 — long-term memory.** Extract → dedupe by cosine threshold → store → retrieve top 3
  from `user_memories` (table exists, HNSW index exists, unused). Plus §6's pre-store PII and
  secret filter. Facts about the **user**, not the corpus — this is not a second retrieval path.
- **Step 6 — the multi-turn eval.** ~10 dialogues in `evals/datasets/multi_turn.jsonl`, written
  only from `data/corpus/` facts and validated like the other datasets. Two metrics: follow-up
  resolution and cross-session memory recall. **This is where the contextualize decision comes
  due** — a history-aware query stage, gated on non-empty history, so Phase 3's measured
  single-turn result (rewriting off) is untouched. New stage, so it wants an ADR.
- **Step 7 — wrap-up.** `memory/` and `generation/` READMEs, `make eval`, scoreboard, PROGRESS.

Carry these in:

- **Known issue 22 is a precondition for step 6's numbers, not a nicety.** `MAX_JUDGE_TOKENS`
  truncated one `judge_faithfulness` call in 42 on this run, and it is biased toward the
  longest, most claim-dense answers — which is exactly what a multi-turn answer is. Raise it
  before measuring anything in step 6, or the rows that fall out will be the rows that matter.
- **Known issue 3 is not theoretical any more.** The `advanced` baseline's multi-hop
  correctness moved 0.500 → 0.562 between two runs of identical code — more than the effect
  step 3 was looking for. **Any step 4-6 generation claim needs repeats, or it needs to be
  larger than ±0.05.** This is the single most important thing learned this chat.
- **The budget is built but barely exercised.** Step 4 is what makes history unbounded and the
  trim order real. The estimator has ~4% headroom (measured), so watch `budget.drift` once
  summaries and memories are in the prompt — a summary is prose and should tokenize nearer 4.0
  chars/token, but a memory list is short fragments and may tokenize denser.
- **Prompt caching still does not pay** and the stable block is ~540 tokens, not the ~425 this
  file claimed (known issue 7, corrected). Memory instructions in step 4 are the next
  realistic chance of crossing 1024 — the traces already record `cache_read_input_tokens`, so
  it will be visible rather than inferred. **Do not read a cached-token delta between two runs
  as a property of the change** (known issue 3b, which nearly produced a wrong conclusion in
  step 3).
- **`.env` still serves the naive baseline** (`INDEX_VERSION=1`, everything off), so `make dev`
  answers from slot 1 with vector-only retrieval. **Still undecided, and now overdue:** whether
  `.env.example` should default to the advanced configuration. Step 7 is the last sensible
  place for it.
- **The semantic cache is still off.** Unchanged, deliberate, separate.
- `make cache`, `make judges`, `make coverage` and `make validate` all pass as of this chat and
  remain preconditions, not extras.

Leftovers:

- `docs/BUILD_PLAN.md` still carries whitespace-only editor changes, uncommitted (known issue
  13). Deliberately not swept into any Phase 5 commit.
- `answer.v2` and `answer_v2.yaml` are kept as the record of a negative result, like
  `hyde_only.yaml`. `prompt_version_answer` stays `v1`.
- `generation_scope` is still dead code (known issue 21); the `--limit` report-overwrite guard
  is still one-sided (known issue 14) and bit this chat — it is why the drift measurement
  needed a standalone script instead of a 2-row run.

## Session log
<!-- newest first: date — phase — 2–4 line summary -->

- **2026-10-22 — Phase 5 steps 1-3.** Built the §5 token budget with its trim order (closing
  known issue 5), made the fast/strong split and the fallback chain auditable via
  `GenerationRecord.stage` — finding that `LLM_FALLBACK_MODELS` had never provided a fallback
  in any phase — and ran the `answer.v1` vs `answer.v2` A/B. **v2 not adopted:** +0.012
  correctness against a ±0.05 noise band, and the predicted multi-hop gain did not appear.
  Three incidental findings mattered more than the A/B: the baseline moved 0.500 → 0.562 on
  multi-hop between identical runs, a judge call was lost to truncation (new known issue 22),
  and the stable prompt block is ~540 tokens rather than the ~425 recorded since Phase 2.

- **2026-10-22 — between 4 and 5 — four review findings, all real.**
  The serious one was a 500: `LangfuseTracer.request` had its `yield` inside
  `try/except Exception`, so **the body's** exception was caught and the second yield turned it
  into `RuntimeError: generator didn't stop after throw()`. A foreign session id answered 500
  instead of 404, reproduced live and fixed with `ExitStack`. It survived a phase because the
  comment under the except claimed the opposite *and* `NoOpTracer.request` has a bare yield, so
  every test passed — the lesson is in known issue 16. Also: a truncated streamed answer was
  billed at **zero** (only `response.completed` was matched, so `incomplete`/`failed` fell
  through to a zero-usage fallback — the known-issue-10 case, invisible in cost, logs and
  trace); `tracer.score()` blocked the event loop for 6–9ms inside an async handler; and
  `client_scope` had never reached the generation paths or the judges. Each fix has a test
  confirmed to fail against the pre-fix code. One of those tests, written carelessly, made a
  real 401'd OpenAI call before being tightened. 479 tests, lint clean.

- **2026-10-21 — between 4 and 5 — known issues 19 and 20 closed.**
  Query rewriting is off: `hyde_only.yaml` against a repeat of `advanced` came back neutral on
  every quality metric (correctness identical to the digit), so the stage's 42 LLM calls,
  $0.0030 and 114s of model time per run are gone — a stage that had been adding no query at
  all in 33 of 42 rows, kept on for two phases by a Phase 3 measurement that never separated
  it from HyDE. **The run totals would have given the wrong reason:** they differ by $0.0109,
  but `hyde_only` met 113,739 cached input tokens against 3,235, so most of that is prefix
  cache and not work avoided — known issue 3b's first real bite. All six generation spans now
  carry their prompt and completion, which makes HyDE's fabrications visible on HyDE's own
  span instead of only via the next stage's input. Wiring that found that
  `TRACING_CAPTURE_CONTENT` never gated the output side at all, so "capture off" had been
  shipping answer text; the gate now lives on the span object rather than at each call site.
  468 tests, lint clean.

- **2026-10-21 — Phase 4 — DONE. Tracing, streaming, feedback.**
  The Langfuse stack had never been started (known issue 8) and two defects were waiting
  behind `docker compose config`: `minio/minio` no longer pulls anonymously, and
  `dev@localhost` fails Langfuse's own email validation, crash-looping the web container.
  **The first full trace earned the phase on its own** — of 10.8s in `retrieve`, `route` is
  3.8s, the query transforms 4.3s and the cross-encoder 2.4s, while the actual hybrid search
  is 0.2s. Streaming keeps one prompt and one schema by lifting the `answer` field out of
  partial JSON (ADR-025), so the scoreboard still describes both paths. Three bugs found by
  doing rather than by reading: `LLMClient` defaulted to the no-op tracer, so every trace was
  silently missing its generations, tokens and cost; `LangfuseTracer.request` leaked the trace
  id past the span's end, which would have made the eval runner file each row under the
  previous row's trace; and `make cache` hit a circular import the whole test suite could not
  see, because pytest happens to import the modules in the one order that works. 457 tests,
  lint clean.

- **2026-10-21 — Phase 3 — DONE. Steps 5–7, and `advanced.yaml` saved.**
  HyDE was the biggest retrieval win after step 1 (context recall 82.1% → 84.9%) and the
  first real gain on multi-hop since step 1. Multi-query alone produced byte-identical
  retrieval on all 42 rows — traced to the cross-encoder re-sorting a pool that a second
  query barely changes on a 104-chunk corpus. The router lost 8.3pp exactly as its own
  docstring predicted, and is committed switched off. The semantic cache found that ADR-012's
  0.95 threshold serves 1143's answer for 1144 at cosine 0.954, one thousandth below a
  correct paraphrase at 0.955 — so a deterministic figure-match guard sits beside the
  threshold rather than the threshold being tuned. **Phase 3 overall: context recall 61.9% →
  83.7%, hit rate 73.8% → 90.5%, MRR 0.492 → 0.571, on 3.9 passages instead of 5.0, for +30%
  latency.** Numeric and temporal both ended exactly where they started, and temporal's
  28.6pp loss at step 4 is still unexplained. 239 tests, lint clean.

- **2026-10-21 — Phase 3 — judge v2 and the hand-labelled rows. Known issue 2 overturned.**
  Wrote `judge_faithfulness.v2` (decompose into claims, quote a verbatim span for each),
  `verify_spans` to check those quotes mechanically, ten hand-labelled rows with every label
  premise verified against the corpus, and `make judges`. **The headline is a correction:**
  the judge agrees with the labels 9/10 — and both versions score the founding anecdote
  (faithfulness 1.00 on an absent `68,000`) correctly at 0.0 when it is reproduced as a
  labelled row. PROGRESS had called the judge a provable rubber stamp for two phases on the
  strength of one bad roll. v2 is adopted anyway, for auditability not accuracy: on the full
  run it verified 114 of 131 claim spans and caught 2 stitched quotes, one joining text across
  two documents, which v1 could not have surfaced. Costs +36% and reads 0.071 lower, so the
  scoreboard carries a bridge row. The span checker itself was wrong first — flagging dropped
  markdown and an empty refusal quote as fabrications — which is now fixed and tested.
  211 tests, lint clean.

- **2026-10-21 — Phase 3 — step 3 done: keyword FTS + RRF hybrid.**
  `keyword_search` over the existing tsvector, `fusion.py` (RRF, k=60, pure and exactly
  testable), and a `search()` entry point dispatching on `retrieval_mode` so the halves can be
  ablated against the whole. Best configuration so far on every headline metric: context
  recall 81.0%, recall@5 73.0%, MRR 0.528, hit rate 83.3%, context precision 20.5% — and cost
  and latency flat. The `keyword_only` ablation is what makes it meaningful: a worse retriever
  overall (68.3%) that is *better* on routing (100%) and temporal (92.9%), and hybrid keeps
  both of those while retaining the dense half's numeric and single-hop. Caught before
  trusting it: `websearch_to_tsquery` ANDs its terms, so 21 of 42 questions returned zero
  keyword hits and the fusion would have been inert on half the eval — OR-ing the terms took
  that to 0. Two open problems recorded rather than smoothed over: multi-hop has now degraded
  at every step since step 1 (60.4% → 47.9% → 41.7%) and is step 5's job, and unanswerable
  fell 100% → 83.3% because OR-ed terms always match something. 205 tests, lint clean.

- **2026-10-21 — Phase 3 — step 2 done: contextual retrieval prefixes.**
  A fast model writes 1–2 sentences per chunk situating it in its document, prepended before
  embedding and tsvector while the stored text stays verbatim; cached by content hash so
  re-ingestion stays free. **The best result in the project so far**, and it repaired step 1's
  regressions instead of trading against them: context recall 76.2% → 79.8%, recall@5 63.9% →
  69.4%, MRR 0.437 → **0.506** (past the baseline), hit rate 73.8% → 81.0%, on 3.6 passages per
  row. Numeric context recall came all the way back to 86.1% — step 1 predicted step 3's
  keyword half would be needed for that, and the prefix did it first. Multi-hop fell back from
  step 1 (60.4% → 47.9%) and the obvious explanation does not hold up, so it is recorded as
  unexplained. Two bugs found and fixed on the way: the figure check was flagging every
  comma-terminated figure (which had inflated the baseline's count from 4 to 8, and the
  known-issue-2 claim with it), and one row's answer call died of reasoning eating the whole
  token budget, taking the other 41 rows with it. All three scoreboard rows were then re-run
  on identical harness code. 191 tests, lint clean.

- **2026-10-21 — Phase 3 — step 1 done: semantic chunking + parent-child.**
  Built semantic chunking (sentence embeddings, percentile breakpoints), parent-child chunks
  into index slot 2, and `retrieval/expand.py`. Slot 1 verified untouched — `make ingest` still
  reports "nothing to do", which the new fingerprint fields were designed to preserve. The
  headline: **context recall 61.9% → 76.2%** while sending the generator *fewer* passages
  (4.0 vs 5.0), with multi-hop — the baseline's weakest category — up 22.9pp. Had to add
  `context_recall` first, because recall@k is not comparable across chunk sizes and would have
  measured the chunking rather than the retrieval. One real regression: numeric recall
  −13.9pp, dense retrieval being unable to tell `946` from `1,034`; step 3's keyword half is
  the test of whether that recovers. A mistake worth remembering: cutting at every semantic
  breakpoint gave 67 parents averaging 290 tokens against an 800 budget — breakpoints have to
  be permitted cuts, not required ones. 183 tests, lint clean.

- **2026-10-20 — Phase 3 — review fixes, baseline re-recorded. No Phase 3 code yet.**
  An external review raised seven findings; all seven reproduced and all seven are fixed, in six
  commits with tests, three of them verified as negative controls. Two were corrupting the
  measurements themselves (eval token totals counted judge calls only while cost counted both;
  the figure check cleared any fabrication whose digits sat inside a longer number, which on a
  corpus set in 1143/1144 is most of them), so the baseline was re-recorded. Retrieval metrics
  came back identical to the digit — 61.9% / 0.492 / 19.5% — which is the proof that the fixes
  changed the measuring and not the system. Correcting the token totals then exposed a problem
  nobody had flagged: run cost swings ~50% on prefix-cache warmth alone, with nothing in the
  report able to explain it, so `total_cached_tokens` was added and cost now reconciles to the
  cent. The baseline's faithfulness came back **1.000 across all 42 rows** while the repaired
  figure check flags 8 of them — the sharpest evidence yet that the judge is a rubber stamp.
  168 tests, lint clean.

- **2026-10-20 — Phase 2 — done. Baseline recorded.**
  Built ingestion, the vector-only retrieval baseline, structured generation with citations,
  `POST /v1/chat`, and the eval harness; made the project's first live LLM calls and debugged two
  provider quirks (`temperature` unsupported, and `max_output_tokens` being shared with
  reasoning). Baseline: recall@5 61.9%, MRR 0.492, correctness 0.774, refusal recall 100%,
  $0.0212 a run. The finding that matters: the faithfulness judge scored 1.00 on an answer whose
  central figure was in none of its passages, so a deterministic figure check now sits beside it
  and judge scores are an upper bound until `judge_faithfulness.v2`. ADR-020/021/022 recorded.
  156 tests, lint clean.

- **2026-10-19 — Phase 1 — done. Corpus frozen.**
  Wrote both golden datasets against the reserved ids in the trap map, and the validator that
  makes the freeze meaningful: all 72 cited quotes were found verbatim on the first run. Added
  `make validate`, seven tests including a negative control for the validator itself, and put
  `evals` under ruff and pyright. 48 tests pass, lint clean. No eval numbers yet — Phase 2 builds
  the harness that produces them.

- **2026-10-19 — Phase 1 — corpus written (steps 1–2).**
  World bible and trap map approved and committed, then all 10 corpus files written per SPEC §4,
  each 100–200 lines with the full front-matter. Checked mechanically: every ledger row's total is
  units × per-unit, all four deliberate holes are still holes, and all 8 planted injection/PII/wrong-claim
  strings are present and confined to file 10. Corpus not yet frozen — the datasets come first.

- **2026-10-19 — Phase 1 — step 1 written, awaiting review.**
  Designed "The Meridian Chronicles": 40 numbered waystones between Halvern Reach and Kesh Ithar,
  Book 1 in 1143 and Book 2 in 1144. Wrote `docs/world_bible.md` (world, cast, factions, timeline,
  the ten-file map) and `evals/datasets/README.md` (the trap map, with reserved `gq-`/`atk-` ids).
  Planted trap counts exceed SPEC §4: 5 Book 2 updates, 7 multi-hop facts, 3 injections, 4 fake PII
  items, 6 unanswerables. No corpus files written — the freeze waits on review.

- **2026-10-19 — Phase 0 — Foundation. Done.**
  Resolved the LLM provider open question (ADR-017: OpenAI `gpt-6-luna` for both roles, chosen on
  eval cost). Built the full foundation: uv project, compose stack, all 9 §12 tables with a
  round-trip-verified migration, the single LLM client with retries/fallback/costing, the versioned
  prompt loader, and `/v1/health`. 41 tests, lint clean, repo public and pushed.
  Carrying forward: no live LLM call made yet, and the Langfuse stack is unbooted.
