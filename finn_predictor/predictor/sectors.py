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
    _recency_weight,
    aggregate_sentiment,
    rolling_baseline,
)
from finn_predictor.predictor.classifier import (
    LogisticCalibration,
    apply_logreg_classification,
)
from finn_predictor.predictor.magnitude import MagnitudeCalibration
from finn_predictor.predictor.market import (
    MIN_ARTICLES_FOR_CALL,
    MIN_BASELINE_SIGMA,
    THRESHOLD_SIGMA,
    classify,
)
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    RelatedEntity,
    Sector,
    SentimentScore,
)
from finn_predictor.storage.repo import (
    all_sectors,
    articles_in_window,
    latest_market_caps,
    related_entities_for,
    save_prediction,
    utc_day_window,
)


logger = logging.getLogger(__name__)


def _sector_scored_articles(
    session: Session,
    *,
    sector_symbols: Iterable[str],
    model_version: str,
    day: datetime,
) -> list[tuple[NewsArticle, float]]:
    """Return ``[(article, score), ...]`` for one sector on ``day``.

    Iterates symbol-by-symbol so a sector with thousands of tickers stays
    streamable; the call sites we have today pass at most a few dozen.
    Keeping the article around (rather than just the score) lets the
    caller apply recency and per-source weights — same machinery the
    market predictor uses.
    """
    from sqlalchemy import select

    start, end = utc_day_window(day)
    pairs: list[tuple[NewsArticle, float]] = []
    for sym in sector_symbols:
        arts = articles_in_window(session, start, end, symbol=sym, category="company")
        if not arts:
            continue
        ids = [a.id for a in arts]
        rows = session.execute(
            select(SentimentScore.article_id, SentimentScore.score).where(
                SentimentScore.article_id.in_(ids),
                SentimentScore.model_version == model_version,
            )
        ).all()
        score_map = {aid: float(sc) for aid, sc in rows}
        for a in arts:
            if a.id in score_map:
                pairs.append((a, score_map[a.id]))
    return pairs


def predict_sector(
    session: Session,
    *,
    scorer: Scorer,
    sector: Sector,
    sector_symbols: Iterable[str],
    on_date: datetime | None = None,
    threshold_sigma: Optional[float] = None,
    min_baseline_sigma: Optional[float] = None,
    half_life_hours: Optional[float] = None,
    source_weights: Optional[dict[str, float]] = None,
    use_market_cap_weights: bool = True,
    calibration: Optional[LogisticCalibration] = None,
    magnitude_calibration: Optional[MagnitudeCalibration] = None,
) -> Optional[Prediction]:
    """Compute and persist a prediction for one sector's ETF on ``on_date``.

    Accepts the same learnable knobs as :func:`predict_market` — both
    predictors pull them from
    :func:`finn_predictor.learning.active_weights` in the daily ingest.

    When ``use_market_cap_weights`` is True (the default) and the DB has
    cached :class:`HistoricalMarketCap` rows for any of ``sector_symbols``,
    each article's per-ticker weight is multiplied by its company's most
    recent market cap as of ``on_date``. Tickers without a cap row keep
    weight 1.0, so the run silently degrades to the previous uniform
    behaviour for sectors where caps haven't been ingested yet.
    """
    on_date = on_date or datetime.now(timezone.utc)
    threshold = threshold_sigma if threshold_sigma is not None else THRESHOLD_SIGMA
    floor = min_baseline_sigma if min_baseline_sigma is not None else MIN_BASELINE_SIGMA
    half_life = half_life_hours if half_life_hours is not None else 12.0
    sw = source_weights or {}

    sector_symbols = list(sector_symbols)
    paired = _sector_scored_articles(
        session,
        sector_symbols=sector_symbols,
        model_version=scorer.model_version,
        day=on_date,
    )
    n = len(paired)
    if n == 0:
        return None

    # Cap weighting is opt-in by argument, but driven by DB state: if the
    # caller asks for it but no caps are cached, every ticker falls back
    # to 1.0 and the run is indistinguishable from uniform. Normalising
    # by the mean keeps the weighted_mean in the same numeric range as
    # the cap-less path, which matters because the rolling baseline
    # below does NOT see cap weights (sentiment scores stay in [-1, 1]).
    caps: dict[str, float] = {}
    if use_market_cap_weights:
        caps = latest_market_caps(
            session, sector_symbols, on_or_before=on_date
        )
        if caps:
            mean_cap = sum(caps.values()) / len(caps)
            if mean_cap > 0:
                caps = {k: v / mean_cap for k, v in caps.items()}

    _, end = utc_day_window(on_date)
    scores = [s for _, s in paired]
    weights = [
        _recency_weight(a.published_at, end, half_life)
        * sw.get((a.source or "").strip(), 1.0)
        * caps.get((a.symbol or "").strip(), 1.0)
        for a, _ in paired
    ]
    today_summary = aggregate_sentiment(scores, weights=weights)

    base = rolling_baseline(
        session,
        model_version=scorer.model_version,
        end_day=on_date,
        category="company",
        half_life_hours=half_life,
        source_weights=sw,
    )
    sigma = max(base.stddev, floor)
    z = (today_summary.weighted_mean - base.mean) / sigma

    if n < MIN_ARTICLES_FOR_CALL:
        label, confidence = "FLAT", 0.0
    elif calibration is not None:
        label, confidence = apply_logreg_classification(
            today_summary.weighted_mean, calibration
        )
    else:
        label, confidence = classify(z, threshold=threshold)

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
    if magnitude_calibration is not None:
        forecast = magnitude_calibration.predict(today_summary.weighted_mean)
        pred.expected_return_p10 = forecast.p10
        pred.expected_return_p50 = forecast.p50
        pred.expected_return_p90 = forecast.p90
    return save_prediction(session, pred)


def _sector_universe_from_db(session: Session) -> dict[str, list[str]]:
    """Build the {sector_code: [ticker, …]} map from cached ``ETF_HOLDING`` rows.

    The cache is populated by ``refresh_sector_constituents`` (called
    from the Focus tab's *Refresh constituents* button). Sectors with
    no cached holdings end up with an empty list, which
    :func:`predict_all_sectors` treats as "skip".
    """
    sectors = all_sectors(session)
    out: dict[str, list[str]] = {}
    for s in sectors:
        rows = related_entities_for(
            session, s.etf_symbol, relationship="ETF_HOLDING"
        )
        out[s.code] = [r.related_symbol for r in rows]
    return out


def predict_all_sectors(
    session: Session,
    *,
    scorer: Scorer,
    on_date: datetime | None = None,
    sector_universe: Optional[dict[str, Iterable[str]]] = None,
    threshold_sigma: Optional[float] = None,
    min_baseline_sigma: Optional[float] = None,
    half_life_hours: Optional[float] = None,
    source_weights: Optional[dict[str, float]] = None,
    use_market_cap_weights: bool = True,
    calibration: Optional[LogisticCalibration] = None,
    magnitude_calibration: Optional[MagnitudeCalibration] = None,
) -> list[Prediction]:
    """Run :func:`predict_sector` for every persisted sector.

    ``sector_universe`` maps a sector ``code`` to its constituent ticker
    symbols. When omitted (the default), the universe is rebuilt from
    cached ``RelatedEntity(relationship="ETF_HOLDING")`` rows — i.e.
    whatever the Focus tab's *Refresh constituents* button has pulled
    from Finnhub. Sectors with no cached holdings are silently skipped
    (they'd have no articles to aggregate anyway).
    """
    on_date = on_date or datetime.now(timezone.utc)
    sectors = all_sectors(session)
    if sector_universe is None:
        sector_universe = _sector_universe_from_db(session)

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
            threshold_sigma=threshold_sigma,
            min_baseline_sigma=min_baseline_sigma,
            half_life_hours=half_life_hours,
            source_weights=source_weights,
            use_market_cap_weights=use_market_cap_weights,
            calibration=calibration,
            magnitude_calibration=magnitude_calibration,
        )
        if pred is not None:
            out.append(pred)
    return out
