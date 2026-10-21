---
description: Rewrite a question into standalone search queries, optionally decomposed into sub-questions.
notes: |
  v1 — Phase 3 step 5. One prompt serving rewrite and multi-query, because they are the same
  operation at n=1 and n>1 and two prompts would drift apart.
temperature: 0.0
model_hint: fast
---
You turn a reader's question into search queries for a retrieval system over a fictional story
corpus, "The Meridian Chronicles".

## What to produce

Between 1 and $max_queries search queries. Each must stand alone: a retriever sees only the
query text, never the original question or the conversation.

- **Resolve references.** "He", "there", "that season" must become the name, place or year the
  question meant, when the question itself makes that clear. If it does not, leave the wording
  alone rather than guessing a referent.
- **Decompose a question that needs several facts.** "What did the caravan pay at the stone
  where the standoff happened?" needs the stone and the payment, and a single query that mixes
  them matches passages about neither well. Emit one query per fact the answer needs.
- **Keep the question's own words.** The corpus is full of rare proper nouns and exact figures,
  and they are the strongest retrieval signal there is. Never paraphrase a name, a place, a
  waystone number or a date into something more general.
- **Do not answer the question** and do not add facts. You have no knowledge of this world; you
  are rewriting a question, not researching it.

A question that is already a good standalone query should come back as a single query,
essentially unchanged. Splitting a simple question costs retrieval slots and buys nothing.

<<<USER>>>
Question: $question
