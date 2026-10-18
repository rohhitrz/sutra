"""Shared fixtures."""

from __future__ import annotations

import pytest

from sutra.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings built explicitly rather than from .env, so tests never depend on a developer's
    local environment or a real API key."""
    return Settings(
        app_env="ci",
        openai_api_key="test-key",
        llm_fast="gpt-6-luna",
        llm_strong="gpt-6-luna",
        llm_max_retries=3,
        llm_fallback_models=["gpt-6-luna"],
        database_url="postgresql+asyncpg://sutra:sutra@localhost:5432/sutra",
    )
