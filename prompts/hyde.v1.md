---
description: Write a hypothetical passage answering the question, for embedding only (HyDE).
notes: |
  v1 — Phase 3 step 5. The output is embedded and thrown away. It is never shown to a user,
  never indexed, and never citable — which is the only reason a prompt that asks a model to
  invent corpus-shaped text is acceptable in a project whose corpus is the sole source of
  truth for story facts.
temperature: 0.0
model_hint: fast
---
You write a short passage that *would* answer a question, if it existed in a reference work.

# CONCEPT: Questions and passages are written in different registers. "Who drowned in the
# flood?" shares almost no wording with "The cartographer Ravenna Volant drowned in it", so
# their embeddings sit further apart than the facts warrant. A hypothetical answer is written
# in the register of the thing being searched for, so its embedding lands nearer the real
# passage than the question's does.

The corpus is a fictional world called the Meridian Chronicles: a chain of forty numbered
waystones, the cities at either end, the caravans that cross it, and the factions that tax it.
You do not know this world, and you are not being asked to. **You are writing a decoy, not an
answer.** Its only job is to look like the kind of paragraph the real answer lives in.

Write two or three sentences. Use the vocabulary of a reference entry — place names, dates,
roles, figures — and reuse every specific term the question gives you, since those are the
words most likely to appear in the passage being searched for. Invent the rest freely; nobody
will read it.

Do not hedge, do not say the answer is unknown, and do not explain what you are doing. A
hedge is useless as a decoy because the real passage does not hedge.

<<<USER>>>
Question: $question
