"""Central configuration.

Everything configurable in Sutra comes from here, and nothing else reads the environment
directly. Secrets come only from `.env` (never committed); non-secrets like model prices live
in this file so cost accounting stays reviewable in git.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ModelPrice(BaseModel):
    """Price per 1M tokens for one model.

    # CONCEPT: Cost accounting needs a price table the code can read. Hardcoding prices at the
    # call site makes per-request cost untrackable; putting them in the environment makes them
    # invisible to review. A checked-in table gives reproducible cost numbers in every eval run.
    """

    input: float = Field(description="USD per 1M input tokens")
    cached_input: float = Field(description="USD per 1M cached input tokens")
    output: float = Field(description="USD per 1M output tokens")

    def cost_usd(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """Cost of one call. Cached tokens are billed at the cheaper cached rate, not the
        standard input rate, so they are subtracted out of `input_tokens` first."""
        uncached_input = max(input_tokens - cached_tokens, 0)
        return (
            uncached_input * self.input / 1_000_000
            + cached_tokens * self.cached_input / 1_000_000
            + output_tokens * self.output / 1_000_000
        )


# Verified against OpenAI's published pricing (see ADR-017). These are the short-context rates,
# which apply at or below 272K input tokens; above that threshold OpenAI bills a higher
# long-context rate for the whole request. Sutra's corpus is ~10 small files, so no request
# should come close — `LLMClient` logs a warning if one ever does.
MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-6-luna": ModelPrice(input=0.10, cached_input=0.01, output=0.50),
    # Not used by default, but kept here so switching a role in .env needs no code change.
    "gpt-6-sol": ModelPrice(input=2.00, cached_input=0.20, output=10.00),
    "gpt-6.1-sol": ModelPrice(input=2.00, cached_input=0.10, output=10.00),
    "gpt-6-astra": ModelPrice(input=10.00, cached_input=1.00, output=50.00),
}

LONG_CONTEXT_THRESHOLD_TOKENS = 272_000


class ModelCapabilities(BaseModel):
    """What a model will accept, as opposed to what it costs.

    # CONCEPT: Providers do not expose a capability endpoint, so an unsupported parameter is
    # discovered as a 400 at request time. A checked-in capability table turns that into a
    # reviewable fact: the *intent* ("answer deterministically") stays in the prompt file, and
    # the table decides whether this particular model can be told so. Stripping the parameter at
    # the call site instead would lose the intent the moment the model changed.
    """

    # Reasoning-style models reject `temperature` rather than ignoring it:
    #   400 — "Unsupported parameter: 'temperature' is not supported with this model."
    supports_temperature: bool = True


# Permissive by default: an unknown model is assumed to behave conventionally, and the 400 it
# raises is more informative than a silently dropped parameter.
DEFAULT_CAPABILITIES = ModelCapabilities()

MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "gpt-6-luna": ModelCapabilities(supports_temperature=False),
    "gpt-6-sol": ModelCapabilities(supports_temperature=False),
    "gpt-6.1-sol": ModelCapabilities(supports_temperature=False),
    "gpt-6-astra": ModelCapabilities(supports_temperature=False),
}

# bge-small's hard input ceiling (ADR-004). Above it the tokenizer truncates with no error and no
# warning, so an oversized chunk is not a crash — it is an embedding of the chunk's first half.
# Lives here rather than in the ingestion package so `Settings` can enforce it without importing
# the embedder.
EMBEDDING_MAX_SEQUENCE_TOKENS = 512

# Tokens reserved for a contextual prefix (Phase 3 step 2), which is prepended before embedding.
# The embedded unit is prefix + chunk, so the chunk budget alone is not what has to fit under the
# ceiling above. A *reserve* rather than a measurement: the prefix is model-written and so has no
# natural bound, and `_contextualise_and_embed` trims any prefix that exceeds this so the
# arithmetic here is a guarantee instead of an assumption. 128 against a measured maximum of 99
# over the frozen corpus, which is the headroom a differently-shaped corpus gets before trimming.
CONTEXT_PREFIX_BUDGET_TOKENS = 128

# # CONCEPT: The text-search configuration decides the stemmer and the stop-word list, and it
# # has to be the *same one* when the tsvector is built and when the query is parsed. Mismatch
# # them and the index stems "waystones" to "waystone" while the query does not — no error, no
# # warning, just a term index that quietly stops matching. One constant, imported by both
# # ingestion and retrieval, is what makes that mismatch impossible rather than unlikely.
TSV_CONFIG = "english"


class Settings(BaseSettings):
    """Typed application settings, loaded from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ----
    app_env: Literal["local", "ci", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sutra_api_key: str = "change-me"

    # ---- Postgres (ADR-002) ----
    database_url: str = "postgresql+asyncpg://sutra:sutra@localhost:5432/sutra"

    # ---- LLM (ADR-005, ADR-017) ----
    openai_api_key: str = ""
    # Two roles, deliberately separate config values even though ADR-017 points both at the same
    # model today. Phase 5 splits them (fast for rewrites/extraction, strong for answers) by
    # editing .env, not code.
    llm_fast: str = "gpt-6-luna"
    llm_strong: str = "gpt-6-luna"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    # # CONCEPT: pydantic-settings treats a `list[str]` field as "complex" and runs json.loads
    # # on the raw environment value *before* any validator sees it. So a perfectly reasonable
    # # `LLM_FALLBACK_MODELS=gpt-6-luna` in .env raised a JSONDecodeError, and the `_split_csv`
    # # validator below — written precisely to accept that form — never ran at all for env or
    # # .env sources. `NoDecode` turns the JSON step off and hands the raw string to the
    # # validator, which is where the parsing was always meant to happen.
    llm_fallback_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["gpt-6-luna"]
    )

    # ---- Prompts (ADR-006) ----
    prompt_version_answer: str = "v1"
    # The judges are versioned prompts like any other, and the version used is recorded in every
    # eval report — a judge rubric edit changes the scores, so a report that did not name its
    # judge version would not be comparable to the next one.
    # v2 from Phase 3. Adopted for *auditability*, not accuracy: it agrees with the
    # hand-labelled rows exactly as often as v1 (9/10 both), but it must quote the span
    # supporting each claim, and `verify_spans` checks those quotes against the passages. A
    # v1 score was unfalsifiable; a v2 score comes with evidence the harness can check.
    prompt_version_judge_faithfulness: str = "v2"
    prompt_version_judge_relevance: str = "v1"
    prompt_version_judge_correctness: str = "v1"
    prompt_version_context_prefix: str = "v1"
    prompt_version_query_transform: str = "v1"
    prompt_version_hyde: str = "v1"
    prompt_version_router: str = "v1"

    # ---- Retrieval / ingestion ----
    # # CONCEPT: Index versioning — re-ingesting under a new version lets old chunks keep serving
    # # traffic until the new index is ready, and keeps eval baselines pinned to a known index.
    # An integer *slot*, not a hash (ADR-021): 1 = Phase 2 naive recursive chunking. The derived
    # fingerprint of the chunking parameters lives in `chunks.metadata`, and ingestion refuses to
    # write mismatched parameters into an occupied slot rather than silently re-chunking it.
    index_version: int = 1
    # Chunking parameters. These three are part of the index fingerprint (ADR-021), so changing
    # any of them and re-ingesting into an occupied slot is a hard error rather than a silent
    # re-chunk. Phase 3 adds the "semantic" strategy and the parent-child split.
    # # CONCEPT: Recursive chunking splits on *typography* — headings, blank lines, sentences.
    # # Semantic chunking splits on *meaning*: embed each sentence, and cut where consecutive
    # # sentences stop resembling each other. It finds topic shifts that have no markdown
    # # boundary, which is most of them in prose, at the cost of an embedding pass per sentence
    # # at ingest time and a boundary set that is no longer a pure function of the text.
    chunk_strategy: Literal["recursive", "semantic"] = "recursive"
    # # CONCEPT: Chunk size is bounded above by the embedder, not by taste. bge-small truncates at
    # # 512 tokens *silently* — a 900-token chunk embeds only its first 512 and the rest is
    # # invisible to search forever. 400 leaves headroom for Phase 3's contextual prefix.
    chunk_tokens: int = 400
    chunk_overlap: int = 60

    # ---- Parent-child chunking (Phase 3 step 1) ----
    # # CONCEPT: Parent-child separates the unit you *search* from the unit you *read*. A small
    # # child embeds precisely — one fact, one vector, little dilution — while the parent gives
    # # the model enough surrounding text to interpret it. Searching on 800-token sections
    # # blurs every vector toward the section's average topic; answering from 200-token
    # # fragments strands facts without their referents. Doing both is the point.
    # Off by default: slot 1 and the committed baseline are flat, and must stay that way.
    parent_child_enabled: bool = False
    # Parents are never embedded, so they are not bound by the embedder's 512-token ceiling.
    parent_tokens: int = 800
    child_tokens: int = 200
    child_overlap: int = 40
    # # CONCEPT: The breakpoint threshold is a *percentile* of this document's own consecutive
    # # similarity drops, not an absolute cosine value. Absolute thresholds do not transfer
    # # between documents: a tightly-written ledger's sentences all resemble each other, and a
    # # cutoff tuned on narrative prose would either split it everywhere or nowhere.
    semantic_breakpoint_percentile: int = 90

    # # CONCEPT: Retrieval-time expansion — search the children, then hand the model their
    # # parents. Without this, parent-child is just small chunks and strictly worse.
    retrieval_expand_to_parent: bool = False

    # ---- Hybrid retrieval (Phase 3 step 3) ----
    # # CONCEPT: Dense retrieval matches topic, lexical retrieval matches tokens. Neither is a
    # # superset of the other, which is why fusing them beats either — and why the mode is a
    # # dial rather than a constant, so `make eval` can measure each half alone.
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    # How many candidates each retriever contributes before fusion. Wider than k on purpose:
    # fusing two top-5 lists can only reorder 5-10 documents, so the rank-7-and-rank-2 case
    # that fusion exists to rescue would never be seen.
    retrieval_candidate_k: int = 20
    # The RRF constant (Cormack et al. 2009). Higher trusts agreement between retrievers more;
    # lower trusts an individual retriever's top rank more.
    rrf_k: int = 60

    # ---- Cross-encoder reranking (Phase 3 step 4) ----
    # # CONCEPT: A bi-encoder embeds the chunk before it has seen the query, so retrieval is
    # # a comparison of two summaries. A cross-encoder reads query and passage together and
    # # is far more accurate — and unusable as a retriever, at one forward pass per chunk per
    # # query. So retrieve widely and cheaply, then spend the expensive model on a shortlist.
    rerank_enabled: bool = False
    # The shortlist handed to the cross-encoder. A reranker can only reorder what it is given,
    # so this, not `retrieval_top_k`, is what bounds how much it can discover.
    rerank_candidates: int = 20

    # ---- Query transforms (Phase 3 step 5) ----
    # # CONCEPT: Retrieval matches a query against passages, but a user writes a question.
    # # Pronouns the corpus never uses, context the index does not have, several facts asked
    # # at once — each is a retrieval problem before it is a generation problem.
    # Rewrite into standalone queries, decomposing when the answer needs several facts. The
    # original question always stays in the pool, so a bad rewrite can only cost slots.
    query_rewrite_enabled: bool = False
    # Upper bound on queries from the rewriter. Each one is a full retrieval, so this is a
    # latency and cost dial as much as a recall one.
    multi_query_count: int = 3
    # HyDE: embed a hypothetical answer instead of only the question. The draft is never
    # shown and never indexed — see `retrieval/query_transform.py`.
    hyde_enabled: bool = False

    # ---- Routing and metadata filters (Phase 3 step 6) ----
    # # CONCEPT: Narrow the search space before searching it. The asymmetry is what matters:
    # # a correct narrow route saves a few slots, a wrong one hides the answer with no second
    # # pass to recover it. So rules fire only on unambiguous wording and the classifier is
    # # told to prefer "general", which applies no filter at all.
    router_enabled: bool = False

    # ---- Contextual retrieval (Phase 3 step 2) ----
    # # CONCEPT: A fast model writes 1-2 sentences situating each chunk in its document,
    # # prepended before embedding so a chunk carries the subject its own sentences omit.
    # # Off by default: it costs one model call per chunk on a cold cache, and slots 1-2
    # # are measured without it.
    contextual_prefixes_enabled: bool = False
    # In-flight prefix generations. Modest on purpose — ingestion is not latency-bound,
    # and calls issued in document order let the provider's prefix cache do its work.
    contextualize_concurrency: int = 4
    # # CONCEPT: k is a recall/precision dial *and* a cost dial — every retrieved chunk is
    # # context the model pays for. 5 matches the Recall@5 column in the PROGRESS scoreboard.
    retrieval_top_k: int = 5
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dims: int = 384
    reranker_model: str = "BAAI/bge-reranker-base"
    model_cache_dir: str = "./.models_cache"

    # ---- Semantic cache (ADR-012) ----
    # # CONCEPT: Every cache hit trusts a similarity score as a judgement about meaning, with
    # # no model in the loop to catch it. 0.95 is deliberately high: two questions differing
    # # only in a year are far closer than that, and serving the wrong year is worse than
    # # paying for the answer twice.
    semantic_cache_enabled: bool = False
    cache_similarity_threshold: float = 0.95
    cache_ttl_hours: int = 168

    # ---- Observability (ADR-007); inert until Phase 4 ----
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    tracing_enabled: bool = False

    @model_validator(mode="after")
    def _chunks_must_fit_the_embedder(self) -> Settings:
        """Fail fast when the chunk budget can exceed the embedder's silent-truncation ceiling.

        Overlap is added on top of `chunk_tokens`, so the real worst case is the sum. Getting this
        wrong costs nothing visible at ingest time and shows up only as unexplained recall loss,
        which is the worst kind of bug to go looking for.
        """
        # The embedded unit is the child when parent-child is on, and the flat chunk otherwise.
        # Parents are deliberately exempt: they are never embedded, which is why they are
        # allowed to be larger than the model's sequence limit.
        if self.parent_child_enabled:
            name_a, size_a = "child_tokens", self.child_tokens
            name_b, size_b = "child_overlap", self.child_overlap
        else:
            name_a, size_a = "chunk_tokens", self.chunk_tokens
            name_b, size_b = "chunk_overlap", self.chunk_overlap

        # The prefix is part of the embedded unit, so it is part of the budget. Checking the
        # bare chunk was the whole defect: at the default 400 + 60 the chunk fits with 52
        # tokens to spare, and a ~90-token prefix then pushes the real input past 512 — where
        # the tokenizer truncates the *end* of the chunk, silently, with the fingerprint and
        # every recall number recorded as if nothing had happened.
        reserved = CONTEXT_PREFIX_BUDGET_TOKENS if self.contextual_prefixes_enabled else 0
        worst_case = size_a + size_b + reserved
        if worst_case > EMBEDDING_MAX_SEQUENCE_TOKENS:
            prefix_term = (
                f" + context prefix reserve ({CONTEXT_PREFIX_BUDGET_TOKENS})" if reserved else ""
            )
            msg = (
                f"{name_a} ({size_a}) + {name_b} ({size_b}){prefix_term} "
                f"= {worst_case} exceeds the embedder's {EMBEDDING_MAX_SEQUENCE_TOKENS}-token "
                "limit; chunks would be silently truncated"
            )
            raise ValueError(msg)
        if self.parent_child_enabled and self.child_tokens >= self.parent_tokens:
            msg = (
                f"child_tokens ({self.child_tokens}) must be smaller than parent_tokens "
                f"({self.parent_tokens}); equal sizes make the parent-child split a no-op "
                "that costs a second row per chunk and buys nothing"
            )
            raise ValueError(msg)
        if self.chunk_strategy == "semantic" and not self.parent_child_enabled:
            # # CONCEPT: A configuration that cannot be honoured must be refused, not
            # # approximated. Semantic splitting lives on the parent-child path only, so the
            # # flat builder ignores `chunk_strategy` entirely — and the index fingerprint
            # # records it regardless. Accepting this pair would stamp "semantic" on a
            # # recursive index, which is the one thing the fingerprint exists to prevent:
            # # two different chunkings comparing equal, or one lying about what it is.
            msg = (
                "chunk_strategy='semantic' requires parent_child_enabled; semantic splitting "
                "is implemented on the parent-child path only, so a flat index would be "
                "chunked recursively while its fingerprint claimed 'semantic'"
            )
            raise ValueError(msg)
        return self

    @field_validator("llm_fallback_models", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Accept a comma-separated string from .env as well as a real list.

        Reachable only because the field is annotated `NoDecode`; without that, pydantic-settings
        JSON-decodes the value first and this never runs.
        """
        if not isinstance(v, str):
            return v
        text = v.strip()
        if text.startswith("["):
            # Still accept the JSON form, since that is what pydantic-settings would have
            # expected and someone may have written it that way.
            try:
                return json.loads(text)
            except ValueError:
                pass
        return [item.strip() for item in text.split(",") if item.strip()]

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        """Fail fast on a sync driver. CLAUDE.md forbids blocking I/O in request handlers, and a
        `postgresql://` URL silently gives you psycopg2 — blocking the event loop under load."""
        if not v.startswith("postgresql+asyncpg://"):
            msg = f"DATABASE_URL must use the asyncpg driver (postgresql+asyncpg://), got: {v!r}"
            raise ValueError(msg)
        return v

    def capabilities_for(self, model: str) -> ModelCapabilities:
        """What this model accepts. Unknown models get the permissive default."""
        return MODEL_CAPABILITIES.get(model, DEFAULT_CAPABILITIES)

    def price_for(self, model: str) -> ModelPrice | None:
        """Price table lookup. Returns None for an unknown model so the caller can decide
        whether a missing price is fatal (it should not be — an un-priced call is still a
        valid answer, just one with unknown cost)."""
        return MODEL_PRICES.get(model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. FastAPI dependencies and the LLM client all go through this, so the
    `.env` file is parsed once per process."""
    return Settings()
