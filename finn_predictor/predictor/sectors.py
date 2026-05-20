"""Iteration 2: per-sector predictor.

Reuses the same algorithm as :mod:`market` but filters the article universe
to the sector's tickers (via ``NewsArticle.symbol``) and targets the
sector ETF. For the first cut we don't market-cap-weight; once we have a
``HistoricalMarketCap`` table this would slot in as the ``weights`` argument
to :func:`aggregate_sentiment`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from finn_predictor.predictor.aggregate import (
    aggregate_sentiment,
    rolling_baseline,
)
from finn_predictor.predictor.market import (
    MIN_ARTICLES_FOR_CALL,
    MIN_BASELINE_SIGMA,
    classify,
)
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import Prediction, Sector, SentimentScore
from finn_predictor.storage.repo import (
    all_sectors,
    articles_in_window,
    save_prediction,
    utc_day_window,
)


logger = logging.getLogger(__name__)


def _sector_articles_scores(
    session: Session,
    *,
    sector_symbols: Iterable[str],
    model_version: str,
    day: datetime,
) -> tuple[list[float], int]:
    """Return ``(scores, article_count)`` for one sector on ``day``.

    Iterates symbol-by-symbol so a sector with thousands of tickers stays
    streamable; the call sites we have today pass at most a few dozen.
    """
    start, end = utc_day_window(day)
    all_scores: list[float] = []
    for sym in sector_symbols:
        arts = articles_in_window(session, start, end, symbol=sym, category="company")
        if not arts:
            continue
        ids = [a.id for a in arts]
        from sqlalchemy import select

        score_rows = session.execute(
            select(SentimentScore.score).where(
                SentimentScore.article_id.in_(ids),
                SentimentScore.model_version == model_version,
            )
        ).all()
        all_scores.extend(float(r[0]) for r in score_rows)
    return all_scores, len(all_scores)


def predict_sector(
    session: Session,
    *,
    scorer: Scorer,
    sector: Sector,
    sector_symbols: Iterable[str],
    on_date: datetime | None = None,
) -> Optional[Prediction]:
    """Compute and persist a prediction for one sector's ETF on ``on_date``."""
    on_date = on_date or datetime.now(timezone.utc)
    scores, n = _sector_articles_scores(
        session,
        sector_symbols=sector_symbols,
        model_version=scorer.model_version,
        day=on_date,
    )
    if n == 0:
        return None

    today_summary = aggregate_sentiment(scores)
    base = rolling_baseline(
        session,
        model_version=scorer.model_version,
        end_day=on_date,
        category="company",
    )
    sigma = max(base.stddev, MIN_BASELINE_SIGMA)
    z = (today_summary.weighted_mean - base.mean) / sigma

    if n < MIN_ARTICLES_FOR_CALL:
        label, confidence = "FLAT", 0.0
    else:
        label, confidence = classify(z)

    # See note in predict_market: normalise to start-of-UTC-day so the
    # save_prediction upsert collapses same-day runs into one row.
    prediction_day, _ = utc_day_window(on_date)

    pred = Prediction(
        target_symbol=sector.etf_symbol,
        prediction_date=prediction_day,
        label=label,
        confidence=confidence,
        sentiment_index=today_summary.weighted_mean,
        article_count=n,
        model_version=scorer.model_version,
    )
    return save_prediction(session, pred)


def predict_all_sectors(
    session: Session,
    *,
    scorer: Scorer,
    on_date: datetime | None = None,
    sector_universe: Optional[dict[str, Iterable[str]]] = None,
) -> list[Prediction]:
    """Run :func:`predict_sector` for every persisted sector.

    ``sector_universe`` maps a sector ``code`` to its constituent ticker
    symbols. When omitted, sectors without an explicit universe are skipped
    (they will have no articles to aggregate anyway).
    """
    on_date = on_date or datetime.now(timezone.utc)
    sectors = all_sectors(session)
    sector_universe = sector_universe or {}

    out: list[Prediction] = []
    for sector in sectors:
        symbols = list(sector_universe.get(sector.code, []))
        if not symbols:
            continue
        pred = predict_sector(
            session,
            scorer=scorer,
            sector=sector,
            sector_symbols=symbols,
            on_date=on_date,
        )
        if pred is not None:
            out.append(pred)
    return out
