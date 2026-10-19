# Ingestion

## Concept
Turning corpus files into retrievable, versioned chunks: parse front-matter → chunk recursively →
embed locally → build a tsvector → upsert under an index slot.

## Why it matters in production
Chunking strategy drives retrieval quality more than the embedding model does, and ingestion is
where silent data loss is cheapest to cause and most expensive to find. Three failure modes this
module is built to make impossible:

- **Silent truncation.** bge-small cuts input at 512 tokens with no error. An oversized chunk is
  not a crash; it is an embedding of the chunk's first half, and the rest is invisible to search
  forever. `Settings` refuses to start if `chunk_tokens + chunk_overlap` exceeds the ceiling.
- **A chunker that rewrites text.** Eval ground truth here is 72 verbatim quotes matched against
  chunk text. A chunker that normalised whitespace would not fail — it would quietly make the
  measurement meaningless.
- **Overwriting a measured index.** An eval baseline is a claim about one corpus, chunked one way,
  embedded by one model. Re-chunking into the slot a baseline is pinned to must fail loudly rather
  than silently invalidate every number recorded against it.

## How Sutra implements it
- `parse.py` — YAML front-matter; `title` and `doc_type` are promoted to columns, the rest stays in
  JSONB under a GIN index so new metadata keys need no migration. `data/corpus/README.md` is
  skipped: it is the freeze notice (ADR-014), not a document. Malformed front-matter is fatal,
  because a skipped document is a silent recall hole.
- `chunk.py` — hand-rolled recursive chunking (ADR-003) over a separator hierarchy
  `## → ### → paragraph → line → sentence → word`, most semantic first, re-merging small pieces
  back up to the budget. Guarantees three invariants: every chunk is a verbatim contiguous slice
  of its document, no content is lost, and no chunk exceeds `chunk_tokens + chunk_overlap`.
  Section headings go in metadata, not into `text`, so `text` stays verbatim.
- `embed.py` — `bge-small-en-v1.5` over ONNX Runtime (ADR-020) behind an `Embedder` Protocol.
  Passages use `embed`, queries use `query_embed`: bge is trained with an instruction prefix on
  the query side only, and embedding a query as if it were a passage costs recall for no visible
  reason. Token counting uses the model's own tokenizer, because the 512 ceiling is defined by
  that tokenizer and any other counter is a guess.
- `fingerprint.py`, `run.py` — the index slot (ADR-021). The fingerprint is
  `{chunk_strategy, chunk_tokens, chunk_overlap, embedding_model, corpus_hash}`, stored in
  `chunks.metadata`. Same fingerprint in the slot → no-op. Different fingerprint → refused, with
  the differing parameters named. Different slot → both kept, which is how Phase 3 compares
  chunking schemes. `tsv` is computed by Postgres in one set-based `UPDATE`, since it is derived
  data and the `english` configuration has to resolve where the matching query will resolve it.

Phase 2 writes **flat** chunks: `parent_id` and `context_prefix` stay NULL. Parent-child retrieval
and contextual prefixes are Phase 3, and the point of leaving them empty is that the baseline
measures what they are worth.

## Tradeoffs
- Smaller chunks retrieve precisely but lose context; larger chunks do the opposite. 400 tokens
  with 60 overlap is a starting point, not a finding — Phase 3 measures it rather than arguing it.
- The slot number is chosen by a human, so bumping it is a thing you can forget. The fingerprint
  mismatch error is the mitigation, turning a forgotten bump into a loud failure at ingest time.
- **The fingerprint covers parameters, not code.** Fixing a bug in the chunker changes its output
  without changing the fingerprint, so `make ingest` will report "nothing to do" over stale
  chunks. Use `--force` after any change to chunking behaviour. See the known issues in
  `PROGRESS.md`.
- Heading metadata reads h2 and h3 but not h1. Two corpus documents section on h3, and one
  (`08_faq`) has no headings below its title at all, so six chunks legitimately carry no heading.

## How to see it working
```bash
make ingest                      # first run: 49 chunks from 10 documents
make ingest                      # second run: "nothing to do", by fingerprint
CHUNK_TOKENS=250 make ingest     # refused, naming chunk_tokens: 400 -> 250
uv run python -m sutra.ingestion.run --index-version 2   # a second slot, side by side
```
Then, in Postgres:
```sql
SELECT index_version, count(*), max(token_count) FROM chunks GROUP BY 1;
SELECT metadata->'index_fingerprint' FROM chunks LIMIT 1;
SELECT count(*) FROM chunks WHERE tsv @@ to_tsquery('english', 'waystones');  -- stemming
```
