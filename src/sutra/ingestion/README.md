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
  Passages use `embed`, queries use `query_embed`, keeping the asymmetric-embedding seam even
  though the two are currently identical for this model under fastembed (measured: no query
  prefix is applied, and adding bge's documented one by hand moved Recall@5 from 74% to 76%,
  which is too little to justify diverging from the default). Token counting uses the model's
  own tokenizer, because the 512 ceiling is defined by that tokenizer and any other counter is
  a guess.
- `fingerprint.py`, `run.py` — the index slot (ADR-021). The fingerprint is
  `{chunk_strategy, chunk_tokens, chunk_overlap, embedding_model, corpus_hash}`, stored in
  `chunks.metadata`. Same fingerprint in the slot → no-op. Different fingerprint → refused, with
  the differing parameters named. Different slot → both kept, which is how Phase 3 compares
  chunking schemes. The fingerprint also carries `chunker_version` (ADR-022), so a change to the
  chunker's *code* invalidates the slot too — without it, a chunker bug fix re-chunks every
  document while the fingerprint stays identical, and `make ingest` calls that a no-op. A stored
  fingerprint from an older schema is refused rather than parsed, since it cannot be compared. `tsv` is computed by Postgres in one set-based `UPDATE`, since it is derived
  data and the `english` configuration has to resolve where the matching query will resolve it.

Phase 2 writes **flat** chunks: `parent_id` and `context_prefix` stay NULL. Parent-child retrieval
and contextual prefixes are Phase 3, and the point of leaving them empty is that the baseline
measures what they are worth.

## Tradeoffs
- Smaller chunks retrieve precisely but lose context; larger chunks do the opposite. 400 tokens
  with 60 overlap is a starting point, not a finding — Phase 3 measures it rather than arguing it.
- The slot number is chosen by a human, so bumping it is a thing you can forget. The fingerprint
  mismatch error is the mitigation, turning a forgotten bump into a loud failure at ingest time.
- `CHUNKER_VERSION` (ADR-022) is bumped by hand, so it can be forgotten. Deriving it from the
  chunker's source would catch that but would also fire on a comment edit, making every refactor
  look like a re-index — a deliberate human signal is the better trade. Bumping it invalidates
  every existing slot by design, at the cost of one `--force` re-ingest.
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

## Contextual retrieval (Phase 3 step 2)

`contextualize.py` asks a fast model to write one or two sentences situating each chunk in its
document, and that prefix is prepended before the embedding and the tsvector are built.

The problem it solves is a chunk that cannot be found when read alone. "He paid the toll at the
nineteenth" is unambiguous in its chapter and nearly unsearchable out of it — no name, no year,
no road — so its vector sits nowhere near the question that wants it. No amount of retriever
tuning fixes a vector that never carried the subject.

**What is indexed and what is stored are deliberately different.** `chunks.text` keeps the
verbatim source slice; only the embedding and the tsvector see `prefix + text`. That is what
lets citations quote the source and lets the eval match its 72 verbatim quotes against an index
that searched something else.

**The prefix is never shown to the answering model as citable text.** It is model-written, and
the corpus is the only source of truth for story facts — an invented sentence would otherwise
become a retrievable claim about a world where it is not true. The prompt forbids adding any
fact not already present, and all 104 generated prefixes were audited with the eval's figure
check: none asserts a figure absent from its source document.

**Caching is not an optimisation here, it is what makes ingestion idempotent.** The key is a
hash of (prompt version, document slug, chunk text), read back out of any index slot, so
re-running `make ingest` on an unchanged corpus costs nothing. Without it ADR-021's promise
that re-running is free would be true about rows and false about money. The prompt version is
in the key because a rewritten prompt writes different prefixes — reusing one across versions
would put two configurations in a single slot.

Calls are issued grouped by document, so the provider's prefix cache sees the same
multi-thousand-token document block on consecutive requests. This is the first place in Sutra
where prompt caching actually pays: the stable block is finally over the 1024-token minimum
that made it inert in Phase 2 (known issue 7).

```bash
python -m sutra.ingestion.run --index-version 3 --parent-child --strategy semantic --contextual
```
