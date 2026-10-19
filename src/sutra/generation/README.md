# Generation

## Concept
Turn retrieved passages into an answer that states where each claim came from, or that declines
to answer when the passages do not support one.

## Why it matters in production
A RAG answer is a claim about a corpus, and an uncited claim is unauditable. Three things have to
hold, and only the first is usually implemented:

1. The output has the shape the application expects.
2. The output's *citations point at something real*. A schema guarantantees the model returned
   integers; it says nothing about whether those integers refer to passages that were shown.
3. Refusal is a first-class outcome rather than an accident of a cautious model.

## How Sutra implements it
- **One versioned prompt file**, `prompts/answer.v1.md`, holding both halves of the request split
  by a `<<<USER>>>` delimiter. Stable instructions sit before the first variable byte, which is a
  cost decision: prompt caching bills a repeated prefix at a fraction of the input rate, but only
  while that prefix is byte-identical. Interpolating anything into the system half would
  invalidate the cache on every request. A test asserts the system half is identical across two
  different questions.
- **`AnswerDraft` is what the model returns** — flat, primitive fields, no defaults, because
  provider strict-schema modes reject optionals and nested unions, and a schema the provider
  silently declines to enforce is worse than none.
- **Citations are passage numbers, not chunk ids.** Asking a model to echo a UUID invites a
  plausible-looking wrong one; a small integer it can see is both easier to get right and
  trivially checkable. `resolve_citations` maps each number back to the chunk actually shown.
- **Invented citations are kept, not discarded.** A model citing passage 9 when it was shown five
  is hallucinating provenance. `Answer.dropped_citations` records it: an eval signal now and a
  Phase 6 guardrail input later. Dropping it quietly would make the answer look better grounded
  than it is.
- **No passages means no model call.** Nothing was retrieved, so the only honest answer is
  already known and the call would be waste. Flagged as `short_circuited`.
- **Temperature comes from the prompt file's front-matter**, so changing it is a prompt version
  bump visible in the diff rather than a constant buried in Python.

The prompt instructs that passages are quoted material and never instructions. The corpus has
injection strings planted in `10_traveler_notes.md` on purpose, and they will be retrieved during
evals — prompt hygiene is not a guardrail, and Phase 6 adds the actual detection and blocking,
but a baseline that can be hijacked by its own corpus would not be measuring retrieval.

## Tradeoffs
- **No token budget yet.** `MAX_ANSWER_TOKENS` is a flat cap; ARCHITECTURE §5's real budget —
  system + memory + history + chunks + answer reserve within the model window, with a defined
  trim order — is Phase 5. With k=5 and 400-token chunks nothing comes close to the window, so
  this is a gap by schedule rather than by oversight.
- **Confidence is self-reported**, so it measures the model's disposition as much as the
  evidence. It is recorded rather than acted on; nothing branches on it in Phase 2.
- **Non-streaming.** `POST /v1/chat` returns JSON; SSE arrives with the streaming work in a later
  phase, and the LLM client's `stream()` already exists for it.
- A citation asserts the passage contains what the answer says. Nothing verifies that claim at
  this stage — the faithfulness judge in the eval harness is what measures it.

## How to see it working
```bash
make eval        # faithfulness, relevance and correctness over the 42 golden rows
```
Or one question end to end:
```bash
uv run python -m sutra.generation.demo "What is the population of Kesh Ithar?"
```
