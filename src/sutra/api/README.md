# API

## Concept
The HTTP surface. `POST /v1/chat` answers one question from the corpus with citations;
`GET /v1/health` reports dependency status.

## Why it matters in production
Writing guardrails, memory and tracing once and reusing them across HTTP, voice and MCP is what
keeps behaviour consistent across entry points. So the route stays thin and the turn itself lives
in `chat.py`, where the eval runner calls it directly without going through HTTP.

## How Sutra implements it
- **`routes_chat.py`** is HTTP only: validate, dispatch, map errors to status codes.
- **`chat.py`** is ARCHITECTURE §2's lifecycle, with most of it not yet built: resolve the
  session → save the question → retrieve → generate → save the answer. Guardrails, the semantic
  cache, memory and tracing slot in around that order in later phases, which is why the order
  follows §2 rather than being convenient.
- **No database connection is held across the model call.** A pooled connection is scarce and an
  LLM call takes seconds, so a handler that keeps a transaction open across generation caps
  concurrency at the pool size and holds a write transaction open for the duration. The turn uses
  two short sessions, one either side of the call.
- **Unimplemented contract values are refused, not downgraded.** `mode` and `stream` exist in the
  §13 contract; Phase 2 implements `rag` and non-streaming only, and asking for either of the
  others returns 501. A caller who asked for `stream: true` and got a buffered body would
  conclude streaming works.
- **One error for "no such session" and "not your session".** Distinguishing them tells a caller
  which session ids are real, which is a membership oracle. Both return 404.
- The response carries `prompt_version`, `index_version`, `retrieved_chunk_ids` and
  `dropped_citations`, because an answer is only reproducible if you know which prompt and which
  index produced it, and only auditable if a retrieval miss stays distinguishable from a
  generation failure after the fact.

## Tradeoffs
- **No authentication.** `SUTRA_API_KEY` exists in config and nothing checks it. Fine for a local
  dev server, not fine otherwise; auth belongs with the Phase 6 guardrail work and is recorded in
  the PROGRESS known issues rather than quietly assumed.
- **No streaming.** The response is buffered JSON. `LLMClient.stream()` already exists and the
  SSE event shape is specified in ARCHITECTURE §5; wiring it is later work.
- **No rate limiting and no request size cap beyond `query` max_length.**
- SSE is simpler than WebSockets but one-directional — fine here, since voice has its own
  transport.

## How to see it working
```bash
make up && make migrate && make ingest
make dev
curl -s localhost:8000/v1/health | jq
curl -s -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"query":"How many waystones are on the Line?","user_id":"you"}' | jq
```
Pass the `session_id` you get back to continue the conversation. Swagger is at
`localhost:8000/docs`.
