"""Tests for the curated ticker → sector mapping + universe helpers."""

from __future__ import annotations

import pytest

from finn_predictor.storage.repo import DEFAULT_SECTORS
from finn_predictor.storage.sector_membership import (
    SECTOR_MEMBERSHIP,
    merge_sector_universes,
    sector_for_ticker,
    sector_universe_from_tickers,
)


def test_sector_for_ticker_recognises_mega_caps() -> None:
    """The map covers the obvious mega-caps a user is likely to type."""
    cases = {
        "AAPL": "TECH",
        "MSFT": "TECH",
        "JPM": "FIN",
        "XOM": "ENERGY",
        "NVDA": "TECH",
        "GOOGL": "COMM",
        "META": "COMM",
        "AMZN": "DISCRETIONARY",
        "LLY": "HEALTH",
        "NEE": "UTILITIES",
        "PLD": "REAL_ESTATE",
        "LIN": "MATERIAL",
        "CAT": "INDUSTRIAL",
        "PG": "STAPLES",
    }
    for symbol, expected in cases.items():
        assert sector_for_ticker(symbol) == expected, f"{symbol} → {expected}"


def test_sector_for_ticker_case_and_whitespace_insensitive() -> None:
    assert sector_for_ticker("  aapl  ") == "TECH"
    assert sector_for_ticker("aAPl") == "TECH"
    assert sector_for_ticker("MSFT") == sector_for_ticker(" msft ")


def test_sector_for_ticker_returns_none_for_unmapped() -> None:
    assert sector_for_ticker("UNKNOWN_TICKER_42") is None
    assert sector_for_ticker("") is None
    assert sector_for_ticker(None) is None


def test_membership_codes_are_subset_of_default_sectors() -> None:
    """Every mapping target maps to a real sector in DEFAULT_SECTORS."""
    canonical_codes = {code for code, _, _ in DEFAULT_SECTORS}
    used_codes = set(SECTOR_MEMBERSHIP.values())
    unknown = used_codes - canonical_codes
    assert not unknown, f"membership references unknown codes: {unknown}"


def test_each_sector_has_at_least_one_member() -> None:
    """Every DEFAULT_SECTORS code is represented — otherwise that sector
    can never produce a synthesized prediction from user tickers."""
    used_codes = set(SECTOR_MEMBERSHIP.values())
    canonical_codes = {code for code, _, _ in DEFAULT_SECTORS}
    missing = canonical_codes - used_codes
    assert not missing, f"sectors with zero curated members: {missing}"


def test_sector_universe_from_tickers_groups_correctly() -> None:
    out = sector_universe_from_tickers(["AAPL", "MSFT", "JPM", "XOM", "NVDA"])
    assert set(out.keys()) == {"TECH", "FIN", "ENERGY"}
    assert set(out["TECH"]) == {"AAPL", "MSFT", "NVDA"}
    assert out["FIN"] == ["JPM"]
    assert out["ENERGY"] == ["XOM"]


def test_sector_universe_from_tickers_preserves_input_order() -> None:
    out = sector_universe_from_tickers(["NVDA", "AAPL", "MSFT"])
    assert out["TECH"] == ["NVDA", "AAPL", "MSFT"]


def test_sector_universe_from_tickers_dedupes() -> None:
    out = sector_universe_from_tickers(["AAPL", " aapl ", "AAPL"])
    assert out == {"TECH": ["AAPL"]}


def test_sector_universe_from_tickers_drops_unmapped() -> None:
    out = sector_universe_from_tickers(["AAPL", "UNKNOWN_42", "JPM"])
    assert set(out.keys()) == {"TECH", "FIN"}
    # The unmapped ticker is silently absent from the result; callers
    # who want to surface it (e.g. UI "Other" section) keep their own
    # copy of the input.
    assert "UNKNOWN_42" not in {t for ts in out.values() for t in ts}


def test_sector_universe_from_tickers_handles_empty_input() -> None:
    assert sector_universe_from_tickers([]) == {}
    assert sector_universe_from_tickers(["", "   ", None]) == {}


def test_merge_sector_universes_unions_unique_tickers() -> None:
    cached = {"TECH": ["AAPL", "MSFT"], "FIN": ["JPM"]}
    derived = {"TECH": ["MSFT", "NVDA"], "ENERGY": ["XOM"]}
    out = merge_sector_universes(cached, derived)
    assert set(out["TECH"]) == {"AAPL", "MSFT", "NVDA"}
    assert out["FIN"] == ["JPM"]
    assert out["ENERGY"] == ["XOM"]


def test_merge_sector_universes_primary_order_wins() -> None:
    """``primary`` (cached) tickers appear first in the merged list.

    This matters because :func:`predict_sector` iterates the symbol
    list; the order influences which ticker the cap lookup considers
    first when multiple are scored on the same article. Keeping the
    paid-plan-derived order first is the right default.
    """
    cached = {"TECH": ["MSFT", "AAPL"]}
    derived = {"TECH": ["AAPL", "NVDA", "MSFT"]}
    out = merge_sector_universes(cached, derived)
    assert out["TECH"] == ["MSFT", "AAPL", "NVDA"]


def test_merge_sector_universes_handles_none_inputs() -> None:
    """Either side can be None — used by run_daily_ingest when one
    source isn't populated yet."""
    assert merge_sector_universes(None, None) == {}
    derived = {"TECH": ["AAPL"]}
    assert merge_sector_universes(None, derived) == {"TECH": ["AAPL"]}
    assert merge_sector_universes(derived, None) == {"TECH": ["AAPL"]}
