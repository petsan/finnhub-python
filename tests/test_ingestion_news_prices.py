"""Tests for ingestion.news and ingestion.prices using a mocked gateway."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from finn_predictor.ingestion.news import ingest_company_news, ingest_general_news
from finn_predictor.ingestion.prices import ingest_market_caps, ingest_price_history
from finn_predictor.storage.models import HistoricalMarketCap
from finn_predictor.storage.repo import (
    articles_in_window,
    latest_market_caps,
    price_bars,
)


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


def _gateway_with_news(payloads):
    gw = MagicMock()
    gw.general_news.return_value = payloads
    gw.company_news.return_value = payloads
    return gw


def test_ingest_general_news_persists_articles(session) -> None:
    payloads = [
        {
            "id": 101,
            "category": "general",
            "datetime": int(D.timestamp()),
            "headline": "Stocks rally",
            "summary": "S&P 500 closes higher",
            "source": "Reuters",
            "url": "https://example.com/1",
        },
        {
            "id": 102,
            "datetime": int(D.timestamp()) + 60,
            "headline": "Tech leads",
        },
    ]
    gw = _gateway_with_news(payloads)
    assert ingest_general_news(session, gw) == 2

    rows = articles_in_window(session, D - timedelta(hours=2), D + timedelta(hours=2))
    assert {a.finnhub_id for a in rows} == {101, 102}
    assert all(a.category == "general" for a in rows)
    assert all(a.symbol is None for a in rows)


def test_ingest_general_news_dedupes_on_finnhub_id(session) -> None:
    payload = [{"id": 1, "datetime": int(D.timestamp()), "headline": "Once"}]
    gw = _gateway_with_news(payload)
    assert ingest_general_news(session, gw) == 1
    assert ingest_general_news(session, gw) == 0


def test_ingest_general_news_skips_zero_id_entries(session) -> None:
    payload = [
        {"id": 0, "datetime": int(D.timestamp()), "headline": "junk"},
        {"id": 5, "datetime": int(D.timestamp()), "headline": "real"},
    ]
    gw = _gateway_with_news(payload)
    assert ingest_general_news(session, gw) == 1
    rows = articles_in_window(session, D - timedelta(hours=2), D + timedelta(hours=2))
    assert [a.finnhub_id for a in rows] == [5]


def test_ingest_company_news_tags_symbol(session) -> None:
    payload = [{"id": 11, "datetime": int(D.timestamp()), "headline": "AAPL beats"}]
    gw = _gateway_with_news(payload)
    n = ingest_company_news(
        session,
        gw,
        symbol="AAPL",
        start=D - timedelta(days=1),
        end=D + timedelta(days=1),
    )
    assert n == 1
    rows = articles_in_window(
        session, D - timedelta(days=2), D + timedelta(days=2), symbol="AAPL"
    )
    assert [a.symbol for a in rows] == ["AAPL"]
    assert rows[0].category == "company"


def test_ingest_price_history_writes_bars(session) -> None:
    t0 = int(D.replace(hour=0, minute=0).timestamp())
    payload = {
        "s": "ok",
        "t": [t0, t0 + 86400],
        "o": [100.0, 101.0],
        "h": [102.0, 103.0],
        "l": [99.0, 100.5],
        "c": [101.0, 102.5],
        "v": [1_000_000.0, 1_100_000.0],
    }
    gw = MagicMock()
    gw.stock_candles.return_value = payload

    n = ingest_price_history(
        session,
        gw,
        symbol="^GSPC",
        start=D - timedelta(days=1),
        end=D + timedelta(days=2),
    )
    assert n == 2
    bars = price_bars(session, "^GSPC", D - timedelta(days=2), D + timedelta(days=3))
    assert [b.close for b in bars] == [101.0, 102.5]


def test_ingest_price_history_handles_no_data(session) -> None:
    gw = MagicMock()
    gw.stock_candles.return_value = {"s": "no_data"}
    assert ingest_price_history(
        session, gw, symbol="^GSPC", start=D, end=D + timedelta(days=1)
    ) == 0


def test_ingest_price_history_raises_on_bad_status(session) -> None:
    gw = MagicMock()
    gw.stock_candles.return_value = {"s": "wat"}
    with pytest.raises(RuntimeError):
        ingest_price_history(session, gw, symbol="X", start=D, end=D + timedelta(days=1))


def test_ingest_price_history_handles_naive_start_end(session) -> None:
    """Naive datetimes should be treated as UTC, not crash."""
    naive_start = datetime(2026, 5, 1)
    naive_end = datetime(2026, 5, 2)
    gw = MagicMock()
    gw.stock_candles.return_value = {"s": "no_data"}
    assert (
        ingest_price_history(
            session, gw, symbol="^GSPC", start=naive_start, end=naive_end
        )
        == 0
    )
    # Confirm the call carried sane epoch integers.
    args = gw.stock_candles.call_args
    assert isinstance(args.args[2], int) and args.args[2] > 0
    assert isinstance(args.args[3], int) and args.args[3] > args.args[2]


def test_ingest_market_caps_persists_rows(session) -> None:
    """Happy path: parses atDate + marketCapitalization, upserts via repo."""
    payload = {
        "symbol": "AAPL",
        "data": [
            {"atDate": "2026-05-18", "marketCapitalization": 2_900_000.0},
            {"atDate": "2026-05-19", "marketCapitalization": 2_910_000.0},
        ],
    }
    gw = MagicMock()
    gw.historical_market_cap.return_value = payload

    n = ingest_market_caps(
        session,
        gw,
        symbol="AAPL",
        start=D - timedelta(days=2),
        end=D,
    )
    assert n == 2

    # Gateway was called with ISO date strings, not epoch ints.
    args = gw.historical_market_cap.call_args.args
    assert args[0] == "AAPL"
    assert args[1] == (D - timedelta(days=2)).date().isoformat()
    assert args[2] == D.date().isoformat()

    caps = latest_market_caps(session, ["AAPL"])
    assert caps == {"AAPL": pytest.approx(2_910_000.0)}


def test_ingest_market_caps_handles_empty_payload(session) -> None:
    """Missing / empty ``data`` is a successful no-op (free-tier-friendly)."""
    gw = MagicMock()
    gw.historical_market_cap.return_value = {"symbol": "XYZ"}
    assert (
        ingest_market_caps(
            session, gw, symbol="XYZ", start=D - timedelta(days=2), end=D
        )
        == 0
    )

    gw.historical_market_cap.return_value = {"symbol": "XYZ", "data": []}
    assert (
        ingest_market_caps(
            session, gw, symbol="XYZ", start=D - timedelta(days=2), end=D
        )
        == 0
    )


def test_ingest_market_caps_skips_malformed_rows(session) -> None:
    """Bad dates, missing caps, and non-positive caps drop silently."""
    payload = {
        "symbol": "WAT",
        "data": [
            {"atDate": "not-a-date", "marketCapitalization": 100.0},
            {"atDate": "2026-05-19"},  # missing cap
            {"atDate": "2026-05-19", "marketCapitalization": "abc"},  # not numeric
            {"atDate": "2026-05-19", "marketCapitalization": 0},  # non-positive
            {"atDate": "2026-05-19", "marketCapitalization": 123.0},  # the one good row
        ],
    }
    gw = MagicMock()
    gw.historical_market_cap.return_value = payload

    n = ingest_market_caps(
        session, gw, symbol="WAT", start=D - timedelta(days=2), end=D
    )
    assert n == 1
    caps = latest_market_caps(session, ["WAT"])
    assert caps == {"WAT": pytest.approx(123.0)}


def test_ingest_market_caps_is_idempotent(session) -> None:
    """Running the same window twice inserts nothing the second time."""
    payload = {
        "symbol": "MSFT",
        "data": [{"atDate": "2026-05-19", "marketCapitalization": 3_000_000.0}],
    }
    gw = MagicMock()
    gw.historical_market_cap.return_value = payload
    assert ingest_market_caps(session, gw, symbol="MSFT", start=D, end=D) == 1
    assert ingest_market_caps(session, gw, symbol="MSFT", start=D, end=D) == 0


def test_ingest_price_history_raises_on_mismatched_arrays(session) -> None:
    gw = MagicMock()
    gw.stock_candles.return_value = {
        "s": "ok",
        "t": [1, 2],
        "o": [1.0, 2.0],
        "h": [1.0, 2.0],
        "l": [1.0, 2.0],
        "c": [1.0],  # short!
        "v": [0.0, 0.0],
    }
    with pytest.raises(RuntimeError, match="Mismatched"):
        ingest_price_history(session, gw, symbol="X", start=D, end=D + timedelta(days=1))
