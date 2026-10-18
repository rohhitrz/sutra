# API

## Concept
The HTTP surface: the same core behind every client.

## Why it matters in production
Writing guardrails, memory and tracing once and reusing them across HTTP, voice and MCP is what keeps behaviour consistent across entry points.

## How Sutra implements it
FastAPI, async, Pydantic models at the boundary. `/v1/health` in Phase 0; `/v1/chat` with SSE streaming and `/v1/feedback` follow in Phases 2 and 4 (ARCHITECTURE §13).

## Tradeoffs
SSE is simpler than WebSockets but one-directional — fine here, since voice has its own transport.

## How to see it working
`make dev`, then `curl localhost:8000/v1/health` or open the Swagger page.
