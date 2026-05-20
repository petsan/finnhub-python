"""Daily OHLC price ingestion via yfinance.

Why a separate price source? Finnhub's ``/stock/candle`` is gated on
free-tier keys (we get 403). Without prices we cannot build the
``prediction_outcomes`` table, which means no backtest, no hit-rate,
no PnL curve.

yfinance is a wrapper over Yahoo Finance's public price endpoints — no
API key, no plan tier, no rate-limit headaches at our usage volume.
We use it purely as a price feed; news still comes from Finnhub.

Bars are written to the existing ``price_bars`` table via
:func:`storage.repo.upsert_price_bars`, so every backtest helper that
already reads ``PriceBar`` works unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

from sqlalchemy.orm import Session

from finn_predictor.storage.models import PriceBar
from finn_predictor.storage.repo import upsert_price_bars


logger = logging.getLogger(__name__)


# Callable signature for the injectable yfinance fetcher. The default
# implementation calls ``yfinance.download``; tests pass a stub.
HistoryFn = Callable[..., object]


@dataclass(frozen=True)
class PriceBackfillResult:
    """Counts + failures for a single yfinance price backfill run."""

    symbols_requested: int
    symbols_with_data: int
    bars_inserted: int
    failures: list[dict[str, str]]


def _default_history_fn(
    symbols: list[str], start: datetime, end: datetime
) -> object:
    """Lazy import so plain ``finn_predictor.storage`` users don't pull yfinance."""
    import yfinance as yf

    # progress=False keeps yfinance quiet under Streamlit; auto_adjust=True
    # gives us split/dividend-adjusted closes, which is what we want when
    # comparing prices across the prediction window.
    return yf.download(
        tickers=symbols,
        start=start.date().isoformat(),
        end=(end + timedelta(days=1)).date().isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


def _bars_from_dataframe(symbol: str, df) -> list[PriceBar]:
    """Turn a single-ticker yfinance OHLC DataFrame into PriceBar rows.

    yfinance returns columns named Open/High/Low/Close/Volume and a
    DatetimeIndex of trading days (one row per session).
    """
    bars: list[PriceBar] = []
    for ts, row in df.iterrows():
        # Skip rows with any NaN — yfinance occasionally returns gaps.
        try:
            o = float(row["Open"])
            h = float(row["High"])
            lo = float(row["Low"])
            c = float(row["Close"])
            v = float(row.get("Volume", 0.0) or 0.0)
        except (TypeError, ValueError, KeyError):
            continue
        if any(x != x for x in (o, h, lo, c)):  # NaN check
            continue
        # ts can be tz-naive (older yfinance) or tz-aware. Coerce to UTC.
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bars.append(
            PriceBar(
                symbol=symbol,
                trade_date=ts,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=v,
            )
        )
    return bars


def backfill_prices_yf(
    session: Session,
    *,
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    history_fn: Optional[HistoryFn] = None,
) -> PriceBackfillResult:
    """Pull daily OHLC for ``symbols`` over ``[start, end]`` and persist.

    ``history_fn`` is injected only for tests — production callers leave
    it None and pick up the real yfinance.

    All exceptions are caught per-symbol so a single bad ticker doesn't
    take down the whole backfill. Failures land in
    :attr:`PriceBackfillResult.failures` (without leaking any of the
    symbol list back into log strings).
    """
    syms = [s.strip().upper() for s in symbols if s and s.strip()]
    if not syms:
        return PriceBackfillResult(
            symbols_requested=0, symbols_with_data=0, bars_inserted=0,
            failures=[],
        )

    history_fn = history_fn or _default_history_fn
    failures: list[dict[str, str]] = []
    total_inserted = 0
    symbols_with_data = 0

    try:
        data = history_fn(syms, start, end)
    except Exception as exc:
        # Whole call failed — record one umbrella failure and bail.
        return PriceBackfillResult(
            symbols_requested=len(syms),
            symbols_with_data=0,
            bars_inserted=0,
            failures=[{"symbol": "*", "error": str(exc)}],
        )

    # yfinance returns a multi-index DataFrame when ``group_by="ticker"``
    # is set and multiple tickers are passed. For a single ticker it
    # returns a flat DataFrame.
    for sym in syms:
        try:
            if hasattr(data, "columns") and getattr(data.columns, "nlevels", 1) > 1:
                # multi-index: levels are (ticker, OHLCV)
                if sym in set(data.columns.get_level_values(0)):
                    df_sym = data[sym].dropna(how="all")
                else:
                    raise KeyError(f"no data for {sym}")
            else:
                df_sym = data.dropna(how="all")
            bars = _bars_from_dataframe(sym, df_sym)
            if not bars:
                continue
            inserted = upsert_price_bars(session, bars)
            total_inserted += inserted
            symbols_with_data += 1
        except Exception as exc:
            failures.append({"symbol": sym, "error": str(exc)})
            logger.warning("yfinance backfill failed for %s: %s", sym, exc)

    return PriceBackfillResult(
        symbols_requested=len(syms),
        symbols_with_data=symbols_with_data,
        bars_inserted=total_inserted,
        failures=failures,
    )
