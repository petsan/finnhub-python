"""Historical backfill tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from finn_predictor.ingestion.backfill import (
    BackfillResult,
    backfill_company_news,
    backfill_many,
)
from finn_predictor.ingestion.client import IngestionError
from finn_predictor.storage.repo import articles_in_window


START = datetime(2025, 5, 19, tzinfo=timezone.utc)
END = datetime(2026, 5, 19, tzinfo=timezone.utc)


def _payload(article_id: int, ts: datetime) -> dict:
    return {
        "id": article_id,
        "datetime": int(ts.timestamp()),
        "headline": f"News {article_id}",
        "source": "Reuters",
        "url": f"https://example.com/{article_id}",
    }


def _gateway(payloads_by_chunk: list[list[dict]] | None = None) -> MagicMock:
    """Return a mocked gateway whose company_news() side-effects the chunks list."""
    gw = MagicMock()
    if payloads_by_chunk is None:
        gw.company_news.return_value = []
    else:
        gw.company_news.side_effect = payloads_by_chunk
    return gw


# ---------------- backfill_company_news ----------------


def test_backfill_rejects_empty_symbol(session) -> None:
    with pytest.raises(ValueError):
        backfill_company_news(
            session, _gateway(), symbol="", start=START, end=END
        )


def test_backfill_rejects_inverted_range(session) -> None:
    with pytest.raises(ValueError):
        backfill_company_news(
            session, _gateway(), symbol="AAPL", start=END, end=START
        )


def test_backfill_rejects_zero_chunk_days(session) -> None:
    with pytest.raises(ValueError):
        backfill_company_news(
            session,
            _gateway(),
            symbol="AAPL",
            start=START,
            end=END,
            chunk_days=0,
        )


def test_backfill_pages_in_chunks(session) -> None:
    """A 90-day range with chunk_days=30 → exactly 3 gateway calls."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=90)
    gw = _gateway([
        [_payload(101, start + timedelta(days=2))],
        [_payload(102, start + timedelta(days=35))],
        [_payload(103, start + timedelta(days=70))],
    ])

    result = backfill_company_news(
        session, gw, symbol="AAPL", start=start, end=end, chunk_days=30
    )
    assert result.symbol == "AAPL"
    assert result.chunks_attempted == 3
    assert result.chunks_failed == 0
    assert result.inserted == 3
    assert gw.company_news.call_count == 3


def test_backfill_deduplicates_on_finnhub_id(session) -> None:
    """Same article in two overlapping chunks → only inserted once."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=60)
    same_article = _payload(7, start + timedelta(days=15))
    gw = _gateway([[same_article], [same_article]])  # 2 chunks both return id=7

    result = backfill_company_news(
        session, gw, symbol="AAPL", start=start, end=end, chunk_days=30
    )
    assert result.inserted == 1  # second chunk's duplicate is dropped


def test_backfill_continues_past_failed_chunk(session) -> None:
    """A 403 on month 2 must not prevent month 1 + month 3 from ingesting."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=90)
    gw = MagicMock()
    gw.company_news.side_effect = [
        [_payload(1, start + timedelta(days=5))],
        IngestionError("FinnhubAPI 403: blocked"),
        [_payload(3, start + timedelta(days=70))],
    ]
    result = backfill_company_news(
        session, gw, symbol="AAPL", start=start, end=end, chunk_days=30
    )
    assert result.chunks_attempted == 3
    assert result.chunks_failed == 1
    assert result.inserted == 2  # month 1 + month 3
    assert "FinnhubAPI 403" in str(result.failures[0]["error"])
    assert result.failures[0]["op"] == "backfill_company_news:AAPL"


def test_backfill_short_range_one_chunk(session) -> None:
    """Range smaller than chunk_days → exactly one call."""
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=5)
    gw = _gateway([[_payload(1, start + timedelta(days=2))]])
    result = backfill_company_news(
        session, gw, symbol="AAPL", start=start, end=end, chunk_days=30
    )
    assert result.chunks_attempted == 1
    assert result.inserted == 1


def test_backfill_naive_datetimes_are_treated_as_utc(session) -> None:
    naive_start = datetime(2026, 5, 1)
    naive_end = datetime(2026, 5, 5)
    gw = _gateway([[]])
    result = backfill_company_news(
        session, gw, symbol="AAPL", start=naive_start, end=naive_end
    )
    assert result.chunks_attempted == 1
    # Confirm naive datetimes didn't trip the inverted-range check.
    assert result.chunks_failed == 0


def test_backfill_persists_articles_with_company_symbol(session) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=10)
    gw = _gateway([
        [
            _payload(50, start + timedelta(days=2)),
            _payload(51, start + timedelta(days=4)),
        ]
    ])
    backfill_company_news(
        session, gw, symbol="AAPL", start=start, end=end, chunk_days=30
    )
    rows = articles_in_window(
        session,
        start - timedelta(days=1),
        end + timedelta(days=1),
        symbol="AAPL",
        category="company",
    )
    assert {a.finnhub_id for a in rows} == {50, 51}


def test_backfill_result_succeeded_chunks_property() -> None:
    r = BackfillResult(
        symbol="AAPL", inserted=5, chunks_attempted=4, chunks_failed=1,
        failures=[],
    )
    assert r.succeeded_chunks == 3


# ---------------- backfill_many ----------------


def test_backfill_many_keys_by_symbol(session) -> None:
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=5)
    gw = MagicMock()
    gw.company_news.return_value = []

    out = backfill_many(
        session, gw, symbols=["AAPL", "MSFT"], start=start, end=end
    )
    assert set(out.keys()) == {"AAPL", "MSFT"}
    assert all(isinstance(r, BackfillResult) for r in out.values())


def test_backfill_many_skips_blank_entries(session) -> None:
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=5)
    gw = MagicMock()
    gw.company_news.return_value = []

    out = backfill_many(
        session, gw, symbols=["AAPL", "", "  ", "MSFT"], start=start, end=end
    )
    assert list(out.keys()) == ["AAPL", "MSFT"]
