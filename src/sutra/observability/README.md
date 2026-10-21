# Observability

## Concept
Traces, costs, latencies and user feedback for every request.

## Why it matters in production
You cannot improve or debug an LLM system you cannot see. One Sutra chat turn is a router call, a rewrite, a HyDE call, two retrievals per query, a fusion, twenty cross-encoder passes, a parent expansion and an answer call. Something in there spends three and a half seconds. Per-stage latency and cost are what turn "it feels slow" into a fix, and an aggregate number over the whole request cannot name the stage.

## How Sutra implements it
Langfuse, self-hosted in the `observability` compose profile (ADR-007). One trace per request, one span per stage from ARCHITECTURE §2, and a `generation` span wrapped *around* each provider call so its duration is the call's duration rather than a zero-width marker filed afterwards.

- `context.py` — the trace id and request fields, in contextvars. Cross-cutting state does not belong in function signatures (ADR-024).
- `logs.py` — a hand-rolled JSON formatter. Every line inside a request carries its `trace_id`, so a log search and a Langfuse trace are two views of one request.
- `tracing.py` — `LangfuseTracer`, a complete `NoOpTracer`, and `build_tracer()`, which returns the no-op whenever tracing is off or the keys are blank.

Every LLM call still goes through `sutra.llm.client` (ADR-005), which is what makes model, prompt version, tokens and our own priced cost appear on every generation without a single call site remembering to report them.

## Tradeoffs
Tracing costs latency and the self-hosted stack is heavy — Langfuse brings its own Postgres, ClickHouse, Redis and MinIO, which is why it sits behind a compose profile. The ambient tracer is process-global state, accepted deliberately in ADR-024 over threading a `tracer=` parameter through retrieval. Span inputs and outputs carry the question and the retrieved passages, so `TRACING_CAPTURE_CONTENT` exists for corpora where that is not acceptable.

Observability must not break what it observes: every call into the SDK is wrapped so a Langfuse outage degrades to a log line rather than a failed request.

## How to see it working
```
make up PROFILE=observability     # wait for http://localhost:3000 to answer
make dev
curl -N -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"query":"Who keeps the merchant ledger?","user_id":"u1","stream":true}'
```
The response's `meta` frame carries the `trace_id`. Open `http://localhost:3000` (sign in with `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`) and that trace shows the full tree with per-stage latency and per-generation cost. `POST /v1/feedback` with the same `trace_id` attaches a score to it.

Note that Langfuse v4 runs in **events_only mode**: the legacy `/api/public/traces/{id}` endpoint is gone, and traces are read through `/api/public/v2/observations?traceId=...` and `/api/public/v3/scores`.

Without the stack running, `build_tracer()` returns the no-op and everything still works — which is how the test suite exercises the instrumentation with no Langfuse at all.
