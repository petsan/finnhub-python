"""Price ingestion using Finnhub's ``/stock/candle`` endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import FinnhubGateway
from finn_predictor.storage.models import PriceBar
from finn_predictor.storage.repo import upsert_price_bars


def _to_epoch(d: datetime) -> int:
    """Convert (naive-or-aware) datetime to a UTC unix-second epoch."""
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp())


def ingest_price_history(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: str = "D",
) -> int:
    """Pull daily candles for ``symbol`` and persist them.

    Finnhub responds with parallel arrays (``o`` / ``h`` / ``l`` / ``c`` /
    ``v`` / ``t``) plus a status field ``s``. ``no_data`` is treated as a
    successful no-op (just nothing to write).
    """
    payload = gateway.stock_candles(symbol, resolution, _to_epoch(start), _to_epoch(end))

    status = payload.get("s")
    if status == "no_data":
        return 0
    if status != "ok":
        raise RuntimeError(
            f"Unexpected stock_candles status {status!r} for {symbol}"
        )

    ts = payload.get("t") or []
    opens = payload.get("o") or []
    highs = payload.get("h") or []
    lows = payload.get("l") or []
    closes = payload.get("c") or []
    volumes = payload.get("v") or [0.0] * len(ts)

    if not (len(ts) == len(opens) == len(highs) == len(lows) == len(closes) == len(volumes)):
        raise RuntimeError(f"Mismatched candle arrays for {symbol}")

    bars = [
        PriceBar(
            symbol=symbol,
            trade_date=datetime.fromtimestamp(int(t), tz=timezone.utc),
            open=float(o),
            high=float(h),
            low=float(l),
            close=float(c),
            volume=float(v),
        )
        for t, o, h, l, c, v in zip(ts, opens, highs, lows, closes, volumes)
    ]
    return upsert_price_bars(session, bars)
