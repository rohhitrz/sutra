"""Config, validators and the price table."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sutra.config import MODEL_PRICES, Settings, get_settings


def test_both_model_roles_default_to_luna(settings: Settings):
    # ADR-017: both roles point at gpt-6-luna, but remain separate settings so a later phase
    # can split them without a code change.
    assert settings.llm_fast == "gpt-6-luna"
    assert settings.llm_strong == "gpt-6-luna"


def test_sync_database_driver_is_rejected():
    # A plain postgresql:// URL selects a blocking driver, which would block the event loop.
    with pytest.raises(ValidationError, match="asyncpg"):
        Settings(database_url="postgresql://sutra:sutra@localhost:5432/sutra")


def test_async_database_driver_is_accepted():
    s = Settings(database_url="postgresql+asyncpg://u:p@host:5432/db")
    assert s.database_url.startswith("postgresql+asyncpg://")


def test_fallback_models_accepts_csv_string():
    # Untyped input, so model_validate rather than the typed constructor.
    s = Settings.model_validate({"llm_fallback_models": "gpt-6-luna, gpt-6-sol ,"})
    assert s.llm_fallback_models == ["gpt-6-luna", "gpt-6-sol"]


def test_fallback_models_accepts_list():
    s = Settings(llm_fallback_models=["a", "b"])
    assert s.llm_fallback_models == ["a", "b"]


def test_luna_price_matches_published_rates():
    # Verified against OpenAI's pricing page (ADR-017). If this test fails, the published
    # prices changed and every cost number on the scoreboard needs re-reading.
    luna = MODEL_PRICES["gpt-6-luna"]
    assert (luna.input, luna.cached_input, luna.output) == (0.10, 0.01, 0.50)


def test_cost_is_computed_per_million_tokens():
    luna = MODEL_PRICES["gpt-6-luna"]
    # 1M input + 1M output = $0.10 + $0.50
    assert luna.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.60)


def test_cached_tokens_are_billed_at_the_cached_rate():
    luna = MODEL_PRICES["gpt-6-luna"]
    # 1M input of which 1M cached, no output: 1M * $0.01, not 1M * $0.10.
    assert luna.cost_usd(1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.01)
    # Cached tokens are subtracted from the standard-rate input, never double-counted.
    mixed = luna.cost_usd(1_000_000, 0, cached_tokens=400_000)
    assert mixed == pytest.approx(600_000 * 0.10 / 1e6 + 400_000 * 0.01 / 1e6)


def test_cached_tokens_exceeding_input_do_not_go_negative():
    # Defensive: a provider reporting cached > input must not produce a negative bill.
    luna = MODEL_PRICES["gpt-6-luna"]
    assert luna.cost_usd(100, 0, cached_tokens=500) >= 0


def test_unknown_model_has_no_price(settings: Settings):
    assert settings.price_for("no-such-model") is None
    assert settings.price_for("gpt-6-luna") is not None


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
