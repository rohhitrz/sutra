# Generation

> **Status:** stub — filled in Phase 2.

## Concept
Producing a grounded, cited, structured answer.

## Why it matters in production
Free-text answers cannot be validated or safely rendered. A typed answer with citations can be checked for groundedness and refused when confidence is low.

## How Sutra implements it
Versioned prompts from `prompts/`, a Pydantic `Answer` model with citations and confidence, token budgeting and a documented trim order (Phase 5).

## Tradeoffs
Structured output constrains the model and can slightly reduce fluency. Worth it for verifiability.

## How to see it working
`POST /v1/chat` returns citations alongside the answer; the faithfulness judge in `make eval` scores groundedness.
