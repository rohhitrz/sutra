# MCP server

> **Status:** stub — filled in Phase 7.

## Concept
Exposing Sutra's tools over the Model Context Protocol.

## Why it matters in production
MCP is the current standard for letting other agents call your tools. It turns an internal tool surface into something any MCP client can use.

## How Sutra implements it
The four agent tools exposed over stdio using the official `mcp` Python SDK (ADR-010).

## Tradeoffs
Stdio transport is easy to test but single-client. Good enough for a local learning setup.

## How to see it working
Point Claude Desktop at the server and call `search_lore` from a chat.
