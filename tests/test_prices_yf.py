"""Tests for the yfinance-backed price ingestion."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from finn_predictor.ingestion.prices_yf import (
    PriceBackfillResult,
    backfill_prices_yf,
)
from finn_predictor.storage.repo import latest_price_bar, price_bars


START = datetime(2026, 5, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 10, tzinfo=timezone.utc)


def _multi_index_frame(per_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mimic yf.download(group_by='ticker') output with multi-index columns."""
    frames = []
    for sym, df in per_symbol.items():
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([[sym], df.columns])
        frames.append(df)
    return pd.concat(frames, axis=1)


def _ohlc_frame(index, closes):
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000 for _ in closes],
        },
        index=pd.DatetimeIndex(index),
    )


# ---------------- happy paths ----------------


def test_backfill_writes_bars_for_multiple_symbols(session) -> None:
    days = pd.date_range("2026-05-01", "2026-05-05", freq="D")
    aapl = _ohlc_frame(days, [100, 101, 102, 103, 104])
    msft = _ohlc_frame(days, [200, 201, 202, 203, 204])

    def fake(symbols, start, end):
        return _multi_index_frame({"AAPL": aapl, "MSFT": msft})

    out = backfill_prices_yf(
        session,
        symbols=["AAPL", "MSFT"],
        start=START,
        end=END,
        history_fn=fake,
    )

    assert isinstance(out, PriceBackfillResult)
    assert out.symbols_requested == 2
    assert out.symbols_with_data == 2
    assert out.bars_inserted == 10
    assert out.failures == []

    aapl_bars = price_bars(session, "AAPL", START, END)
    assert [b.close for b in aapl_bars] == [100, 101, 102, 103, 104]


def test_backfill_dedupes_on_rerun(session) -> None:
    days = pd.date_range("2026-05-01", "2026-05-03", freq="D")
    aapl = _ohlc_frame(days, [100, 101, 102])
    fake = lambda *_: _multi_index_frame({"AAPL": aapl})

    a = backfill_prices_yf(
        session, symbols=["AAPL"], start=START, end=END, history_fn=fake
    )
    b = backfill_prices_yf(
        session, symbols=["AAPL"], start=START, end=END, history_fn=fake
    )
    assert a.bars_inserted == 3
    assert b.bars_inserted == 0  # all rows already present


def test_backfill_skips_nan_rows(session) -> None:
    """yfinance occasionally returns gap rows. Those must be ignored."""
    days = pd.date_range("2026-05-01", "2026-05-04", freq="D")
    df = _ohlc_frame(days, [100.0, float("nan"), 102.0, float("nan")])
    fake = lambda *_: _multi_index_frame({"AAPL": df})

    out = backfill_prices_yf(
        session, symbols=["AAPL"], start=START, end=END, history_fn=fake
    )
    assert out.bars_inserted == 2  # only 2 clean rows
    last = latest_price_bar(session, "AAPL")
    assert last is not None and last.close == 102.0


# ---------------- failure paths ----------------


def test_backfill_handles_missing_symbol_in_response(session) -> None:
    """The wrapper records a per-symbol failure rather than crashing."""
    days = pd.date_range("2026-05-01", "2026-05-03", freq="D")
    aapl = _ohlc_frame(days, [100, 101, 102])
    fake = lambda *_: _multi_index_frame({"AAPL": aapl})

    out = backfill_prices_yf(
        session,
        symbols=["AAPL", "FAKETICKER"],
        start=START,
        end=END,
        history_fn=fake,
    )
    assert out.symbols_requested == 2
    assert out.symbols_with_data == 1
    assert len(out.failures) == 1
    assert out.failures[0]["symbol"] == "FAKETICKER"


def test_backfill_handles_complete_call_failure(session) -> None:
    def boom(symbols, start, end):
        raise RuntimeError("yfinance offline")

    out = backfill_prices_yf(
        session, symbols=["AAPL"], start=START, end=END, history_fn=boom
    )
    assert out.bars_inserted == 0
    assert len(out.failures) == 1
    assert "offline" in out.failures[0]["error"]


def test_backfill_empty_symbol_list(session) -> None:
    out = backfill_prices_yf(
        session, symbols=[], start=START, end=END, history_fn=lambda *a: None
    )
    assert out == PriceBackfillResult(
        symbols_requested=0, symbols_with_data=0, bars_inserted=0, failures=[]
    )


def test_backfill_strips_and_uppercases_symbols(session) -> None:
    days = pd.date_range("2026-05-01", "2026-05-02", freq="D")
    aapl = _ohlc_frame(days, [100, 101])

    seen = {}

    def fake(symbols, start, end):
        seen["symbols"] = list(symbols)
        return _multi_index_frame({"AAPL": aapl})

    backfill_prices_yf(
        session,
        symbols=[" aapl ", "  "],
        start=START,
        end=END,
        history_fn=fake,
    )
    assert seen["symbols"] == ["AAPL"]


def test_backfill_single_symbol_flat_dataframe(session) -> None:
    """yfinance returns a flat DataFrame (no multi-index) for one ticker."""
    days = pd.date_range("2026-05-01", "2026-05-03", freq="D")
    flat = _ohlc_frame(days, [50, 51, 52])
    fake = lambda *_: flat

    out = backfill_prices_yf(
        session, symbols=["AAPL"], start=START, end=END, history_fn=fake
    )
    assert out.bars_inserted == 3
