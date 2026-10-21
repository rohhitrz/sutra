# API

## Concept
The HTTP surface. `POST /v1/chat` answers one question from the corpus with citations,
buffered or as an SSE stream; `POST /v1/feedback` records a rating against a trace;
`GET /v1/health` reports dependency status.

## Why it matters in production
Writing guardrails, memory and tracing once and reusing them across HTTP, voice and MCP is what
keeps behaviour consistent across entry points. So the route stays thin and the turn itself lives
in `chat.py`, where the eval runner calls it directly without going through HTTP.

## How Sutra implements it
- **`routes_chat.py`** is HTTP only: validate, dispatch, map errors to status codes.
- **`chat.py`** is ARCHITECTURE §2's lifecycle, with some of it not yet built: resolve the
  session → save the question → retrieve → generate → save the answer, inside a root trace
  span. Guardrails, the semantic cache and memory slot in around that order in later phases,
  which is why the order follows §2 rather than being convenient.
- **The root span is opened in `chat.py`, not in the route.** The eval runner — and later MCP
  and voice — call the turn directly, so instrumenting the route would leave every non-HTTP
  caller untraced.
- **No database connection is held across the model call.** A pooled connection is scarce and an
  LLM call takes seconds, so a handler that keeps a transaction open across generation caps
  concurrency at the pool size and holds a write transaction open for the duration. The turn uses
  two short sessions, one either side of the call.
- **Streaming sends the same prompt through the same schema as the buffered path.** The
  answer is schema-constrained (`AnswerDraft`), so the provider's deltas are fragments of JSON;
  the `answer` field is lifted out of the partial document as it arrives
  (`generation/partial_json.py`). A prose-shaped second prompt would have streamed more easily
  and forked generation into two paths of different quality, leaving the eval scoreboard
  describing only one of them.
- **Frames are `meta` → `token`* → `citations` → `done`, or a terminal `error`**, which is
  ARCHITECTURE §5's set. Two deliberate departures from §5's description of `meta` as
  "trace_id, cost, cache_hit": cost is on `done`, because it is unknown until the provider
  finishes and `meta` is sent before the answer call; and `cache_hit` is structurally `false`
  until the semantic cache is wired into the turn, sent anyway so the frame's shape is stable
  for clients.
- **Ordering.** `meta` comes
  first so a client has the `trace_id` even if the answer fails half way through; `citations`
  come last because resolving passage numbers to real chunks is only possible once the JSON
  object has closed. Past the first byte the status line is gone, so a mid-stream failure
  becomes an in-band `error` frame rather than a response that simply stops.
- **Unimplemented contract values are refused, not downgraded.** `mode: agent` returns 501. A
  caller who asked for it and got a RAG answer would conclude agent mode works.
- **Feedback is stored before it is scored.** Postgres is the record of truth and Langfuse is a
  view onto it, so a tracing outage must not lose a rating; the response's `scored` says
  honestly whether the trace was annotated.
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
- **Streaming costs a 404-shaped compromise.** The session is resolved in its own small
  transaction before the response begins, because once a frame is written no status can be set.
  That is one extra `SELECT` per streamed turn, bought to keep a bad session id a 404.
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

# the same question, streamed — note -N, or curl buffers it for you
curl -N -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"query":"How many waystones are on the Line?","user_id":"you","stream":true}'

# then rate it, using the trace_id from the response or the meta frame
curl -s -X POST localhost:8000/v1/feedback -H 'content-type: application/json' \
  -d '{"rating":-1,"trace_id":"<trace_id>","comment":"missed the ledger"}' | jq
```
Pass the `session_id` you get back to continue the conversation. Swagger is at
`localhost:8000/docs`. With `make up PROFILE=observability` running, every turn above appears
as a trace tree at `localhost:3000`, and the feedback as a score on it.
