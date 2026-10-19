---
description: Does the answer agree with the reference answer?
notes: v1 — Phase 2. Reference answers are human-written and authoritative.
model_hint: strong
---
You are grading one answer against a **reference answer** that is authoritative. Where they
disagree, the reference is right.

Grade only **correctness**: whether the answer says the same thing as the reference. Wording,
length, ordering and style do not matter. Extra true detail does not matter. What matters is
whether a reader would come away believing the same facts.

Score:

- **1.0** — the same facts. Every figure, name and date that both give agrees.
- **0.5** — partly right: correct on the main point but wrong or missing on a secondary one, or
  right but so hedged that the reader would not know what to conclude.
- **0.0** — contradicts the reference, or fails to state what the reference states.

When the reference says the corpus does not record something, an answer that declines to answer
is **1.0** and an answer that supplies a figure is **0.0**, however plausible that figure is.

When the reference gives a figure or date, an answer giving a different one is **0.0**. Near is
not correct.

In `reason`, name the specific agreement or disagreement. One sentence.

<<<USER>>>
Question: $question

Reference answer: $reference

Answer to grade: $answer
