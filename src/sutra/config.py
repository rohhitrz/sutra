"""Central configuration.

Everything configurable in Sutra comes from here, and nothing else reads the environment
directly. Secrets come only from `.env` (never committed); non-secrets like model prices live
in this file so cost accounting stays reviewable in git.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    llm_fallback_models: list[str] = Field(default_factory=lambda: ["gpt-6-luna"])

    # ---- Prompts (ADR-006) ----
    prompt_version_answer: str = "v1"

    # ---- Retrieval / ingestion ----
    # # CONCEPT: Index versioning — re-ingesting under a new version lets old chunks keep serving
    # # traffic until the new index is ready, and keeps eval baselines pinned to a known index.
    index_version: int = 1
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dims: int = 384
    reranker_model: str = "BAAI/bge-reranker-base"
    model_cache_dir: str = "./.models_cache"

    # ---- Semantic cache (ADR-012) ----
    cache_similarity_threshold: float = 0.95
    cache_ttl_hours: int = 168

    # ---- Observability (ADR-007); inert until Phase 4 ----
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    tracing_enabled: bool = False

    @field_validator("llm_fallback_models", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Accept a comma-separated string from .env as well as a real list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        """Fail fast on a sync driver. CLAUDE.md forbids blocking I/O in request handlers, and a
        `postgresql://` URL silently gives you psycopg2 — blocking the event loop under load."""
        if not v.startswith("postgresql+asyncpg://"):
            msg = f"DATABASE_URL must use the asyncpg driver (postgresql+asyncpg://), got: {v!r}"
            raise ValueError(msg)
        return v

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
