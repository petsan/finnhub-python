"""Sentiment aggregation helpers shared by the market and sector predictors.

The predictor logic is split into three pure functions so we can unit-test
them independently of the database:

  * :func:`aggregate_sentiment` — collapse a list of scored articles to a
    daily summary (mean, count, std-dev, span).
  * :func:`daily_sentiment_index` — pull articles from the DB for one day
    and return a SentimentSummary.
  * :func:`rolling_baseline` — mean/std of daily indices over a trailing
    window, used as the prediction-threshold reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import NewsArticle, SentimentScore
from finn_predictor.storage.repo import articles_in_window, utc_day_window


@dataclass(frozen=True)
class SentimentSummary:
    """Per-day or per-window sentiment aggregate."""

    mean: float
    count: int
    stddev: float
    weighted_mean: float

    @property
    def is_empty(self) -> bool:
        return self.count == 0


EMPTY_SUMMARY = SentimentSummary(mean=0.0, count=0, stddev=0.0, weighted_mean=0.0)


def aggregate_sentiment(
    scores: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
) -> SentimentSummary:
    """Reduce raw scores to a :class:`SentimentSummary`.

    The recency-weighted mean uses the caller-supplied ``weights`` (one per
    score). When ``weights`` is None the weighted_mean equals the plain mean.
    """
    n = len(scores)
    if n == 0:
        return EMPTY_SUMMARY

    s = sum(scores)
    mean = s / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    if weights is None or len(weights) != n or sum(weights) == 0:
        weighted_mean = mean
    else:
        wsum = float(sum(weights))
        weighted_mean = sum(w * x for w, x in zip(weights, scores)) / wsum

    return SentimentSummary(
        mean=mean, count=n, stddev=stddev, weighted_mean=weighted_mean
    )


def _as_utc(d: datetime) -> datetime:
    """SQLite drops tz info on roundtrip — coerce naive datetimes back to UTC."""
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def _recency_weight(published_at: datetime, anchor: datetime, half_life_hours: float) -> float:
    """Exponential decay: weight = 2 ** (-age_hours / half_life)."""
    published_at = _as_utc(published_at)
    anchor = _as_utc(anchor)
    age_hours = max(0.0, (anchor - published_at).total_seconds() / 3600.0)
    return 2.0 ** (-age_hours / half_life_hours)


def _articles_with_scores(
    session: Session,
    *,
    articles: Sequence[NewsArticle],
    model_version: str,
) -> list[tuple[NewsArticle, float]]:
    """Pair each article with its score for ``model_version`` (drop missing)."""
    if not articles:
        return []
    ids = [a.id for a in articles]
    score_rows = session.execute(
        select(SentimentScore.article_id, SentimentScore.score).where(
            SentimentScore.article_id.in_(ids),
            SentimentScore.model_version == model_version,
        )
    ).all()
    score_map = {aid: sc for aid, sc in score_rows}
    return [(a, score_map[a.id]) for a in articles if a.id in score_map]


def daily_sentiment_index(
    session: Session,
    *,
    model_version: str,
    day: datetime,
    symbol: Optional[str] = None,
    category: Optional[str] = "general",
    half_life_hours: float = 12.0,
    source_weights: Optional[dict[str, float]] = None,
) -> SentimentSummary:
    """Aggregate sentiment for one UTC day.

    ``symbol``/``category`` filter the article set; defaults select the
    whole-market ``general`` feed.

    ``source_weights`` is an optional per-source multiplier (e.g.
    ``{"Reuters": 1.3, "WSJ": 0.7}``) applied **on top of** the recency
    weight before the weighted mean is computed. Sources not in the map
    keep weight 1.0. This is the live-scoring counterpart to the same
    knob in :mod:`learning.simulate` — when the learning loop fits a
    set of source weights, the daily ingest passes them through here.
    """
    start, end = utc_day_window(day)
    articles = articles_in_window(
        session, start, end, category=category, symbol=symbol
    )
    paired = _articles_with_scores(session, articles=articles, model_version=model_version)
    if not paired:
        return EMPTY_SUMMARY

    sw = source_weights or {}
    scores = [s for _, s in paired]
    weights = [
        _recency_weight(a.published_at, end, half_life_hours)
        * sw.get((a.source or "").strip(), 1.0)
        for a, _ in paired
    ]
    return aggregate_sentiment(scores, weights=weights)


def rolling_baseline(
    session: Session,
    *,
    model_version: str,
    end_day: datetime,
    window_days: int = 30,
    symbol: Optional[str] = None,
    category: Optional[str] = "general",
    half_life_hours: float = 12.0,
    source_weights: Optional[dict[str, float]] = None,
) -> SentimentSummary:
    """Mean/stddev of daily indices over the trailing window.

    ``half_life_hours`` and ``source_weights`` are forwarded to each
    inner :func:`daily_sentiment_index` call so the baseline reflects
    the same scoring config the caller is about to compare against.
    Otherwise the baseline would silently float against the default
    config even after the learner activated new weights.
    """
    daily_means: list[float] = []
    for i in range(1, window_days + 1):
        day = end_day - timedelta(days=i)
        summary = daily_sentiment_index(
            session,
            model_version=model_version,
            day=day,
            symbol=symbol,
            category=category,
            half_life_hours=half_life_hours,
            source_weights=source_weights,
        )
        if not summary.is_empty:
            daily_means.append(summary.weighted_mean)

    if not daily_means:
        return EMPTY_SUMMARY
    return aggregate_sentiment(daily_means)
