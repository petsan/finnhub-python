"""Backtester: pair each Prediction with the realised next-session move."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import (
    PriceBar,
    Prediction,
    PredictionOutcome,
)
from finn_predictor.storage.repo import save_outcome


@dataclass(frozen=True)
class BacktestReport:
    """Summary of a backtest run."""

    total: int
    scored: int
    hits: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.scored if self.scored else 0.0


def _label_matches(label: str, realised_return: float) -> bool:
    """Score the prediction:

    * ``UP``   counts as a hit when realised_return > 0
    * ``DOWN`` counts as a hit when realised_return < 0
    * ``FLAT`` counts as a hit when ``|return| < 0.25%``
    """
    if label == "UP":
        return realised_return > 0.0
    if label == "DOWN":
        return realised_return < 0.0
    return abs(realised_return) < 0.0025


def _next_bar_after(
    session: Session, symbol: str, after: datetime
) -> Optional[PriceBar]:
    """First price bar strictly after ``after`` for ``symbol``, or None."""
    stmt = (
        select(PriceBar)
        .where(PriceBar.symbol == symbol, PriceBar.trade_date > after)
        .order_by(PriceBar.trade_date)
        .limit(1)
    )
    return session.scalars(stmt).first()


def _last_bar_on_or_before(
    session: Session, symbol: str, on_or_before: datetime
) -> Optional[PriceBar]:
    stmt = (
        select(PriceBar)
        .where(
            PriceBar.symbol == symbol,
            PriceBar.trade_date <= on_or_before,
        )
        .order_by(PriceBar.trade_date.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def score_outcomes(session: Session) -> BacktestReport:
    """Walk every prediction without an outcome and try to score it.

    The realised return is ``close_{next session} / close_{prediction day} - 1``.
    Predictions for which the next bar hasn't landed yet are left unscored
    and will be retried on the next backtest invocation.
    """
    preds: list[Prediction] = list(
        session.scalars(
            select(Prediction)
            .outerjoin(PredictionOutcome)
            .where(PredictionOutcome.id.is_(None))
        )
    )
    scored = 0
    hits = 0

    for pred in preds:
        anchor = _last_bar_on_or_before(session, pred.target_symbol, pred.prediction_date)
        if anchor is None:
            continue
        nxt = _next_bar_after(session, pred.target_symbol, anchor.trade_date)
        if nxt is None:
            continue

        if anchor.close == 0:
            continue
        realised = (nxt.close / anchor.close) - 1.0
        hit = _label_matches(pred.label, realised)
        save_outcome(
            session,
            PredictionOutcome(
                prediction_id=pred.id,
                realised_return=realised,
                hit=hit,
                realised_at=datetime.now(timezone.utc),
            ),
        )
        scored += 1
        if hit:
            hits += 1

    return BacktestReport(total=len(preds), scored=scored, hits=hits)
