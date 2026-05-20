"""Price ingestion using Finnhub's ``/stock/candle`` endpoint."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import FinnhubGateway
from finn_predictor.storage.models import HistoricalMarketCap, PriceBar
from finn_predictor.storage.repo import upsert_market_caps, upsert_price_bars


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


def _to_iso_date(d: datetime | date) -> str:
    """Render a date (or datetime) as the ``YYYY-MM-DD`` Finnhub expects."""
    if isinstance(d, datetime):
        d = d.astimezone(timezone.utc).date() if d.tzinfo else d.date()
    return d.isoformat()


def ingest_market_caps(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbol: str,
    start: datetime | date,
    end: datetime | date,
) -> int:
    """Pull historical market-cap snapshots for ``symbol`` and persist them.

    Finnhub's ``/stock/historical-market-cap`` returns
    ``{"symbol": "...", "data": [{"atDate": "YYYY-MM-DD",
    "marketCapitalization": float (millions USD)}, ...]}``. Empty / missing
    ``data`` is treated as a successful no-op so callers can fan this
    out across a long ticker list without each gap raising.

    Per-symbol idempotency is provided by the unique ``(symbol, as_of_date)``
    index — running the same window twice inserts nothing the second time.
    """
    payload = gateway.historical_market_cap(symbol, _to_iso_date(start), _to_iso_date(end))

    rows = payload.get("data") or []
    if not rows:
        return 0

    caps: list[HistoricalMarketCap] = []
    for row in rows:
        raw_date = row.get("atDate")
        raw_cap = row.get("marketCapitalization")
        if raw_date is None or raw_cap is None:
            continue
        try:
            cap_val = float(raw_cap)
        except (TypeError, ValueError):
            continue
        if cap_val <= 0:
            continue
        # atDate is a date string like "2024-01-31"; pin to UTC midnight.
        try:
            parsed = datetime.fromisoformat(str(raw_date)).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        caps.append(
            HistoricalMarketCap(
                symbol=symbol,
                as_of_date=parsed,
                market_cap=cap_val,
            )
        )

    if not caps:
        return 0
    return upsert_market_caps(session, caps)
