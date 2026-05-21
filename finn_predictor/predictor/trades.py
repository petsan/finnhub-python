"""Hypothetical-trade tracking + accuracy aggregations.

Each :class:`storage.models.Prediction` is treated as a paper trade:

* ``UP``   → buy 1 unit at the prediction-day close, sell at the next
             session close. PnL is the percentage move.
* ``DOWN`` → sell short 1 unit, cover at the next session close. PnL is
             the negative of the percentage move.
* ``FLAT`` → no trade; included in the timeline only so the hit-rate
             denominator reflects every prediction the model emitted.

The data already lives in the DB: :class:`PredictionOutcome` carries
``realised_return`` (next-session move) and ``hit`` (the backtester's
sign-of-direction call). This module just joins predictions with their
outcomes, signs the return by the predicted direction, and computes
rolling / cumulative aggregates the UI can chart.

Everything here is a pure function over the session — no writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import (
    Prediction,
    PredictionOutcome,
    Sector,
)


# Threshold below which we treat |realised_return| as "noise" for the
# FLAT-hit rule. Matches backtest._label_matches's "<0.25%" rule.
FLAT_NOISE_THRESHOLD = 0.0025


@dataclass(frozen=True)
class TradeRecord:
    """One paper trade derived from a Prediction + its PredictionOutcome.

    Open trades (no outcome row yet) are surfaced with ``closed=False``
    and ``pnl_pct=None`` so the UI can show "pending" rows.
    """

    prediction_id: int
    target_symbol: str
    direction: str             # LONG / SHORT / FLAT
    label: str                 # UP / DOWN / FLAT (mirrors prediction.label)
    target_kind: str           # MARKET / SECTOR / STOCK
    entry_date: datetime
    confidence: float
    realised_return: Optional[float]
    pnl_pct: Optional[float]   # signed by direction; None when open
    hit: Optional[bool]
    closed: bool

    @property
    def is_winner(self) -> bool:
        return bool(self.hit)


def _direction_from_label(label: str) -> str:
    if label == "UP":
        return "LONG"
    if label == "DOWN":
        return "SHORT"
    return "FLAT"


def _signed_pnl(label: str, realised_return: Optional[float]) -> Optional[float]:
    if realised_return is None:
        return None
    if label == "UP":
        return realised_return
    if label == "DOWN":
        return -realised_return
    # FLAT — there's no trade, but for cumulative-PnL math we treat it
    # as zero so the curve doesn't drop NaNs into the running sum.
    return 0.0


def _classify_target(target_symbol: str, sector_etfs: set[str]) -> str:
    if target_symbol == "^GSPC":
        return "MARKET"
    if target_symbol in sector_etfs:
        return "SECTOR"
    return "STOCK"


def _as_utc(d: datetime) -> datetime:
    """SQLite drops tz on roundtrip — coerce naive datetimes back to UTC."""
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def hypothetical_trades(
    session: Session,
    *,
    since: Optional[datetime] = None,
    target_symbol: Optional[str] = None,
) -> list[TradeRecord]:
    """Materialise paper trades for every prediction (optionally filtered).

    Ordered ascending by ``prediction_date`` so cumulative aggregates
    sum in chronological order.
    """
    stmt = select(Prediction).order_by(Prediction.prediction_date)
    if since is not None:
        stmt = stmt.where(Prediction.prediction_date >= since)
    if target_symbol is not None:
        stmt = stmt.where(Prediction.target_symbol == target_symbol)
    preds: list[Prediction] = list(session.scalars(stmt))
    if not preds:
        return []

    # One round-trip to get all outcomes; build a lookup table.
    pred_ids = [p.id for p in preds]
    outcomes = session.execute(
        select(
            PredictionOutcome.prediction_id,
            PredictionOutcome.realised_return,
            PredictionOutcome.hit,
        ).where(PredictionOutcome.prediction_id.in_(pred_ids))
    ).all()
    outcome_map = {pid: (ret, bool(hit)) for pid, ret, hit in outcomes}

    sector_etfs = {
        s for (s,) in session.execute(select(Sector.etf_symbol)).all()
    }

    out: list[TradeRecord] = []
    for p in preds:
        ret, hit = outcome_map.get(p.id, (None, None))
        closed = ret is not None
        out.append(
            TradeRecord(
                prediction_id=p.id,
                target_symbol=p.target_symbol,
                direction=_direction_from_label(p.label),
                label=p.label,
                target_kind=_classify_target(p.target_symbol, sector_etfs),
                # Normalise: identity-mapped predictions keep their
                # tz-aware date while DB-rehydrated ones come back naive;
                # downstream sorts can't compare mixed tz status.
                entry_date=_as_utc(p.prediction_date),
                confidence=p.confidence,
                realised_return=ret,
                pnl_pct=_signed_pnl(p.label, ret),
                hit=hit,
                closed=closed,
            )
        )
    return out


# ---------------- Aggregates / charts data ----------------


def trades_dataframe(trades: Iterable[TradeRecord]) -> pd.DataFrame:
    """Flat DataFrame of all trades — one row per trade."""
    rows = [
        {
            "prediction_id": t.prediction_id,
            "target_symbol": t.target_symbol,
            "target_kind": t.target_kind,
            "direction": t.direction,
            "label": t.label,
            "entry_date": t.entry_date,
            "confidence": t.confidence,
            "realised_return": t.realised_return,
            "pnl_pct": t.pnl_pct,
            "hit": t.hit,
            "closed": t.closed,
        }
        for t in trades
    ]
    columns = [
        "prediction_id", "target_symbol", "target_kind", "direction",
        "label", "entry_date", "confidence", "realised_return", "pnl_pct",
        "hit", "closed",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def cumulative_pnl_series(
    trades: Iterable[TradeRecord],
) -> pd.DataFrame:
    """One row per closed trade, with the running cumulative PnL %.

    Only LONG/SHORT trades contribute to the cumulative sum — FLAT
    trades have pnl_pct == 0 by construction and don't move the curve.
    Open trades (closed=False) are skipped.

    Returned columns: ``entry_date``, ``target_kind``, ``pnl_pct``,
    ``cum_pnl_pct``. The cumulative column is computed globally across
    all targets so the UI can split by ``target_kind`` afterwards.
    """
    closed = [
        t for t in trades if t.closed and t.pnl_pct is not None and t.direction != "FLAT"
    ]
    if not closed:
        return pd.DataFrame(
            columns=["entry_date", "target_kind", "pnl_pct", "cum_pnl_pct"]
        )
    closed.sort(key=lambda t: t.entry_date)

    df = pd.DataFrame(
        [
            {
                "entry_date": t.entry_date,
                "target_kind": t.target_kind,
                "target_symbol": t.target_symbol,
                "pnl_pct": float(t.pnl_pct or 0.0),
            }
            for t in closed
        ]
    )
    df["cum_pnl_pct"] = df["pnl_pct"].cumsum()
    return df


def rolling_hit_rate(
    trades: Iterable[TradeRecord],
    *,
    window: int = 14,
) -> pd.DataFrame:
    """Trailing-``window``-trade rolling hit-rate.

    Excludes FLAT predictions (they're hold-the-line non-trades).
    Returned columns: ``entry_date``, ``hit_rate``.
    """
    closed = [
        t
        for t in trades
        if t.closed and t.direction != "FLAT" and t.hit is not None
    ]
    if not closed:
        return pd.DataFrame(columns=["entry_date", "hit_rate"])
    closed.sort(key=lambda t: t.entry_date)
    df = pd.DataFrame(
        [{"entry_date": t.entry_date, "hit_int": 1 if t.hit else 0} for t in closed]
    )
    df["hit_rate"] = (
        df["hit_int"].rolling(window=window, min_periods=1).mean()
    )
    return df[["entry_date", "hit_rate"]]


def hit_rate_by_target_kind(
    trades: Iterable[TradeRecord],
) -> pd.DataFrame:
    """Hit-rate per target_kind (MARKET / SECTOR / STOCK), excluding FLAT.

    Returned columns: ``target_kind``, ``trades``, ``hits``, ``hit_rate``.
    """
    rows: dict[str, dict[str, int]] = {}
    for t in trades:
        if not t.closed or t.direction == "FLAT" or t.hit is None:
            continue
        bucket = rows.setdefault(
            t.target_kind, {"trades": 0, "hits": 0}
        )
        bucket["trades"] += 1
        if t.hit:
            bucket["hits"] += 1
    if not rows:
        return pd.DataFrame(columns=["target_kind", "trades", "hits", "hit_rate"])
    return pd.DataFrame(
        [
            {
                "target_kind": kind,
                "trades": v["trades"],
                "hits": v["hits"],
                "hit_rate": v["hits"] / v["trades"] if v["trades"] else 0.0,
            }
            for kind, v in rows.items()
        ]
    ).sort_values("target_kind").reset_index(drop=True)


def hit_rate_by_label(
    trades: Iterable[TradeRecord],
) -> pd.DataFrame:
    """Hit-rate per UP/DOWN/FLAT label.

    FLAT is included here because its hit-rule (|return| < 0.25%) is
    well-defined even though no trade is opened. Returned columns:
    ``label``, ``trades``, ``hits``, ``hit_rate``.
    """
    rows: dict[str, dict[str, int]] = {}
    for t in trades:
        if not t.closed or t.hit is None:
            continue
        bucket = rows.setdefault(t.label, {"trades": 0, "hits": 0})
        bucket["trades"] += 1
        if t.hit:
            bucket["hits"] += 1
    if not rows:
        return pd.DataFrame(columns=["label", "trades", "hits", "hit_rate"])
    return pd.DataFrame(
        [
            {
                "label": label,
                "trades": v["trades"],
                "hits": v["hits"],
                "hit_rate": v["hits"] / v["trades"] if v["trades"] else 0.0,
            }
            for label, v in rows.items()
        ]
    ).sort_values("label").reset_index(drop=True)


@dataclass(frozen=True)
class PerformanceSummary:
    """Aggregate stats over a set of trades."""

    total_predictions: int
    closed_trades: int
    open_trades: int
    flat_skipped: int
    wins: int
    losses: int
    hit_rate: float
    cumulative_pnl_pct: float
    avg_pnl_per_trade_pct: float
    best_trade_pnl_pct: Optional[float]
    worst_trade_pnl_pct: Optional[float]


def performance_summary(trades: Iterable[TradeRecord]) -> PerformanceSummary:
    """Roll a flat list of TradeRecord into headline metrics."""
    trades = list(trades)
    total = len(trades)
    open_t = sum(1 for t in trades if not t.closed)
    flat = sum(1 for t in trades if t.direction == "FLAT")
    directional = [
        t for t in trades
        if t.closed and t.direction != "FLAT" and t.pnl_pct is not None
    ]
    closed = len(directional)
    wins = sum(1 for t in directional if t.hit)
    losses = closed - wins
    cum = sum(t.pnl_pct or 0.0 for t in directional)
    avg = cum / closed if closed else 0.0
    best = max((t.pnl_pct for t in directional if t.pnl_pct is not None), default=None)
    worst = min((t.pnl_pct for t in directional if t.pnl_pct is not None), default=None)
    return PerformanceSummary(
        total_predictions=total,
        closed_trades=closed,
        open_trades=open_t,
        flat_skipped=flat,
        wins=wins,
        losses=losses,
        hit_rate=(wins / closed) if closed else 0.0,
        cumulative_pnl_pct=cum,
        avg_pnl_per_trade_pct=avg,
        best_trade_pnl_pct=best,
        worst_trade_pnl_pct=worst,
    )
