"""Iteration 1: whole-market predictor.

Algorithm (intentionally crude):

    todays_index   = recency-weighted mean of today's article scores
    baseline_mean  = mean of last 30 daily indices
    baseline_sigma = stddev of last 30 daily indices
    z              = (todays_index - baseline_mean) / max(baseline_sigma, 0.05)

    label = UP   if z >  THRESHOLD
            DOWN if z < -THRESHOLD
            FLAT otherwise

    confidence = min(1.0, |z| / 2.0)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from finn_predictor.predictor.aggregate import (
    daily_sentiment_index,
    rolling_baseline,
)
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import Prediction
from finn_predictor.storage.repo import save_prediction, utc_day_window


logger = logging.getLogger(__name__)

THRESHOLD_SIGMA = 0.5
MIN_BASELINE_SIGMA = 0.05  # floor avoids divide-by-zero on quiet weeks
MIN_ARTICLES_FOR_CALL = 3  # under this we emit FLAT with zero confidence


def classify(z: float, *, threshold: float = THRESHOLD_SIGMA) -> tuple[str, float]:
    """Return ``(label, confidence)`` for a normalised sentiment z-score."""
    confidence = min(1.0, abs(z) / 2.0)
    if z > threshold:
        return "UP", confidence
    if z < -threshold:
        return "DOWN", confidence
    return "FLAT", confidence


def predict_market(
    session: Session,
    *,
    scorer: Scorer,
    on_date: datetime | None = None,
    symbol: str = "^GSPC",
    category: Optional[str] = "general",
    article_symbol: Optional[str] = None,
) -> Optional[Prediction]:
    """Compute and persist a whole-market prediction for ``on_date``.

    Returns the saved :class:`Prediction` (or ``None`` if there's not enough
    data even to call FLAT — i.e. zero articles).
    """
    on_date = on_date or datetime.now(timezone.utc)

    today = daily_sentiment_index(
        session,
        model_version=scorer.model_version,
        day=on_date,
        category=category,
        symbol=article_symbol,
    )
    if today.is_empty:
        logger.info("no articles for %s — skipping prediction", on_date.date())
        return None

    base = rolling_baseline(
        session,
        model_version=scorer.model_version,
        end_day=on_date,
        category=category,
        symbol=article_symbol,
    )
    sigma = max(base.stddev, MIN_BASELINE_SIGMA)
    z = (today.weighted_mean - base.mean) / sigma

    if today.count < MIN_ARTICLES_FOR_CALL:
        label, confidence = "FLAT", 0.0
    else:
        label, confidence = classify(z)

    # Normalise to start-of-UTC-day so two runs on the same calendar day
    # collide on save_prediction's unique key and upsert the same row.
    # Without this, every click of "Run ingestion now" would mint a
    # fresh Prediction row that differs only in the seconds-precision
    # timestamp, double-counting in History and the rolling baseline.
    prediction_day, _ = utc_day_window(on_date)

    pred = Prediction(
        target_symbol=symbol,
        prediction_date=prediction_day,
        label=label,
        confidence=confidence,
        sentiment_index=today.weighted_mean,
        article_count=today.count,
        model_version=scorer.model_version,
    )
    return save_prediction(session, pred)
