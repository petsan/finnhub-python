"""Tests for finn_predictor.storage.symbol_names."""

from __future__ import annotations

from finn_predictor.storage.models import Sector
from finn_predictor.storage.repo import ensure_default_sectors
from finn_predictor.storage.symbol_names import (
    WELL_KNOWN_NAMES,
    expand_symbol,
    expand_symbol_short,
)


def test_well_known_includes_sector_etfs() -> None:
    for sym in ["XLK", "XLE", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]:
        assert sym in WELL_KNOWN_NAMES, sym
        # Sector entries follow "<name> (<symbol>)" format.
        assert WELL_KNOWN_NAMES[sym].endswith(f"({sym})")


def test_well_known_includes_gspc() -> None:
    assert WELL_KNOWN_NAMES["^GSPC"] == "S&P 500 Index"


def test_expand_symbol_known_company(session) -> None:
    assert expand_symbol(session, "AAPL") == "Apple Inc."
    assert expand_symbol(session, "MSFT") == "Microsoft Corp."


def test_expand_symbol_known_index_and_etf(session) -> None:
    assert expand_symbol(session, "^GSPC") == "S&P 500 Index"
    assert expand_symbol(session, "XLK") == "Information Technology (XLK)"


def test_expand_symbol_falls_back_to_sector_table(session) -> None:
    """Sector ETFs are pre-baked in WELL_KNOWN_NAMES, but the DB Sector
    table is the fallback for any future sector additions."""
    session.add(Sector(code="XYZ", name="Made-up Sector", etf_symbol="XLZ"))
    session.commit()
    assert expand_symbol(session, "XLZ") == "Made-up Sector (XLZ)"


def test_expand_symbol_unknown_returns_symbol_unchanged(session) -> None:
    assert expand_symbol(session, "ZZZUNKNOWN") == "ZZZUNKNOWN"


def test_expand_symbol_empty_returns_empty(session) -> None:
    assert expand_symbol(session, "") == ""
    assert expand_symbol(session, None) == ""  # type: ignore[arg-type]


def test_expand_symbol_short_strips_parenthetical(session) -> None:
    assert expand_symbol_short(session, "XLK") == "Information Technology"
    assert expand_symbol_short(session, "AAPL") == "Apple Inc."
    assert expand_symbol_short(session, "ZZZ") == "ZZZ"


def test_expand_symbol_works_with_seeded_sectors(session) -> None:
    """Sanity: with the default sectors seeded, every code in the table
    resolves to a non-trivial string."""
    ensure_default_sectors(session)
    from finn_predictor.storage.repo import all_sectors

    for s in all_sectors(session):
        out = expand_symbol(session, s.etf_symbol)
        assert out and out != s.etf_symbol
