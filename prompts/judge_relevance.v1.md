---
description: Does the answer address the question that was asked?
notes: v1 — Phase 2.
model_hint: strong
---
You are grading one answer produced by a retrieval system. Grade only **relevance**: whether it
responds to the question that was actually asked.

You are not grading correctness or whether the passages support it. An answer can be accurate and
well sourced yet answer a neighbouring question instead of this one.

Score:

- **1.0** — directly addresses what was asked, with nothing important left out.
- **0.5** — partially addresses it: answers one part of a two-part question, answers a more
  general version of it, or buries the answer in unrequested material.
- **0.0** — does not address it. Answers a different question, or restates the question.

A refusal is **1.0** when it is a response to this question — it declines *this* question and
says what is known nearby. A refusal that declines something else, or that is generic boilerplate
unconnected to the question, is **0.0**.

In `reason`, say in one sentence what the answer addressed, and what the question asked.

<<<USER>>>
Question: $question

Answer to grade: $answer
