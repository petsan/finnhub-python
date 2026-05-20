"""Tests for finn_predictor.config."""

from __future__ import annotations

import pytest

from finn_predictor.config import DEFAULT_DB_URL, Settings, load_settings


def test_load_settings_reads_api_key_and_defaults() -> None:
    s = load_settings({"FINNHUB_API_KEY": "sk-test"})
    assert s.finnhub_api_key == "sk-test"
    assert s.database_url == DEFAULT_DB_URL
    assert s.request_timeout == 15.0
    assert s.rate_limit_per_minute == 55


def test_load_settings_overrides() -> None:
    s = load_settings(
        {
            "FINNHUB_API_KEY": "sk-test",
            "FINN_PREDICTOR_DB_URL": "sqlite:///./other.db",
            "FINN_PREDICTOR_TIMEOUT": "5",
            "FINN_PREDICTOR_RATE_LIMIT": "30",
        }
    )
    assert s.database_url == "sqlite:///./other.db"
    assert s.request_timeout == 5.0
    assert s.rate_limit_per_minute == 30


def test_load_settings_missing_key_raises() -> None:
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        load_settings({})


def test_load_settings_blank_key_raises() -> None:
    with pytest.raises(RuntimeError):
        load_settings({"FINNHUB_API_KEY": "   "})


def test_settings_is_frozen() -> None:
    s = Settings(finnhub_api_key="k")
    with pytest.raises(Exception):
        s.finnhub_api_key = "other"  # type: ignore[misc]


def test_load_settings_optional_key_returns_empty_string() -> None:
    """UI bootstrap path: no key set, but we still get a usable Settings."""
    s = load_settings({}, require_api_key=False)
    assert s.finnhub_api_key == ""
    assert s.database_url == DEFAULT_DB_URL


def test_load_settings_optional_still_returns_explicit_key() -> None:
    s = load_settings({"FINNHUB_API_KEY": "abc"}, require_api_key=False)
    assert s.finnhub_api_key == "abc"
