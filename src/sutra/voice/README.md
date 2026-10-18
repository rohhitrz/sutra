# Voice

> **Status:** stub — filled in Phase 8.

## Concept
Speech in, speech out, over the same core.

## Why it matters in production
Voice exposes latency honestly — a pipeline that feels fine in text feels broken at 3 seconds to first audio. It also shows what speech-to-speech models give up.

## How Sutra implements it
8a: a DIY push-to-talk pipeline (STT to core to TTS). 8b: Pipecat for real-time with VAD and barge-in. 8c: Gemini Live as a comparison point only (ADR-011).

## Tradeoffs
Speech-to-speech skips our text-layer guardrails and tracing entirely, which is exactly why it stays a comparison rather than the main path.

## How to see it working
Talk to it, and compare time-to-first-audio between 8a and 8b.
