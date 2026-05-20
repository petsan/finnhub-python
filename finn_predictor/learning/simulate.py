"""Replay-the-pipeline simulator used as the training objective.

The simulator takes a :class:`LearnedConfig` (candidate parameter vector)
and replays the predictor over the data we already have in the DB:

1. For each ``(target_symbol, UTC-day)`` group that has at least one
   scored article, recompute the day's sentiment index using the
   candidate's ``half_life_hours`` and per-source weights, plus a fresh
   ``rolling_baseline`` over the same model_version.
2. Re-classify each day using the candidate's ``threshold_sigma`` and
   ``min_baseline_sigma`` floor.
3. Pair each re-classified prediction with the realised next-session
   return that already lives in ``prediction_outcomes`` for the matching
   ``(target, prediction_date)`` — i.e. the historical price data does
   not move with the candidate.
4. Compute ``hit_rate + 0.5 * cumulative_pnl_pct`` (blended objective).

Per-day re-aggregation does require iterating the day's articles, but
everything is loaded once per training session into in-memory lookups so
each skopt evaluation is a tight Python loop. With ~hundreds of trades
in the ledger a 30-call gp_minimize finishes in seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.learning.config import LearnedConfig
from finn_predictor.predictor.aggregate import _as_utc
from finn_predictor.predictor.market import MIN_ARTICLES_FOR_CALL, classify
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    PredictionOutcome,
    SentimentScore,
)


# A FLAT call counts as a hit if the realised move is small enough.
# Mirrors the rule in predictor.backtest._label_matches so the simulator
# stays consistent with the backtester the rest of the system reports on.
FLAT_NOISE_THRESHOLD = 0.0025


@dataclass(frozen=True)
class ScoredArticle:
    """In-memory copy of one (article, score) pair for the simulator."""

    article_id: int
    target_symbol: str            # canonical target this article rolls into
    published_at: datetime
    source: str
    raw_score: float


@dataclass(frozen=True)
class TrainingFrame:
    """The dataset the simulator iterates per skopt evaluation.

    Built once with :func:`build_training_frame` and reused across every
    candidate parameter vector.
    """

    # Per (target, day) → list of articles fed into that prediction
    articles_by_day: dict[tuple[str, datetime], list[ScoredArticle]]
    # Per (target, day) → realised_return from PredictionOutcome
    realised_by_day: dict[tuple[str, datetime], float]
    # Per target → list of (day, weighted_mean_for_default_config) used for
    # the rolling baseline. Built from a ROUND of the simulator with the
    # candidate's half-life because the baseline must move with the
    # candidate too.

    @property
    def closed_prediction_keys(self) -> list[tuple[str, datetime]]:
        return sorted(set(self.articles_by_day) & set(self.realised_by_day))


def _norm_day(d: datetime) -> datetime:
    """Start-of-UTC-day, tz-aware. Mirrors the predictor's prediction_date norm."""
    d = _as_utc(d)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def build_training_frame(
    session: Session, *, model_version: str
) -> TrainingFrame:
    """Materialise everything the simulator needs in-memory.

    Loads:
      * Predictions with outcomes (closed trades only),
      * NewsArticles within the predictions' day windows,
      * SentimentScores for ``model_version``.

    Heavy by design: this happens once per training run.
    """
    # 1) Predictions with outcomes (closed trades only).
    rows = session.execute(
        select(
            Prediction.id,
            Prediction.target_symbol,
            Prediction.prediction_date,
            PredictionOutcome.realised_return,
        ).join(
            PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id
        ).where(Prediction.model_version == model_version)
    ).all()

    realised_by_day: dict[tuple[str, datetime], float] = {}
    keys_by_target: dict[str, set[datetime]] = {}
    for _id, sym, pdate, ret in rows:
        day = _norm_day(pdate)
        realised_by_day[(sym, day)] = float(ret)
        keys_by_target.setdefault(sym, set()).add(day)

    if not realised_by_day:
        return TrainingFrame(articles_by_day={}, realised_by_day={})

    # 2) Articles on the relevant days, scoped by category + (optional) symbol.
    # Market predictions use general news; sector/stock use company.
    # We can't tell which from the Prediction row alone, so we load both
    # universes and let the simulator pick the right slice per target.

    articles_by_day: dict[tuple[str, datetime], list[ScoredArticle]] = {}

    # Pull all SentimentScore for this model_version in one query.
    score_rows = session.execute(
        select(SentimentScore.article_id, SentimentScore.score).where(
            SentimentScore.model_version == model_version
        )
    ).all()
    score_map = {aid: float(sc) for aid, sc in score_rows}

    # Per-target, get the article set that the predictor would have used:
    # ^GSPC  → general news (NewsArticle.symbol is None)
    # other  → company news with NewsArticle.symbol == target
    for target, days in keys_by_target.items():
        if target == "^GSPC":
            arts_stmt = select(NewsArticle).where(NewsArticle.category == "general")
        else:
            arts_stmt = select(NewsArticle).where(
                NewsArticle.category == "company",
                NewsArticle.symbol == target,
            )
        arts = list(session.scalars(arts_stmt))
        # Bucket by start-of-UTC-day, but only keep buckets where we have
        # a known (target, day) prediction.
        for a in arts:
            day = _norm_day(a.published_at)
            if day not in days:
                continue
            score = score_map.get(a.id)
            if score is None:
                continue
            articles_by_day.setdefault((target, day), []).append(
                ScoredArticle(
                    article_id=a.id,
                    target_symbol=target,
                    published_at=_as_utc(a.published_at),
                    source=(a.source or "").strip() or "(unknown)",
                    raw_score=float(score),
                )
            )
    return TrainingFrame(
        articles_by_day=articles_by_day,
        realised_by_day=realised_by_day,
    )


def _weighted_mean(
    arts: Iterable[ScoredArticle],
    *,
    day_anchor: datetime,
    config: LearnedConfig,
) -> tuple[float, int]:
    """Recency-weighted, source-weighted mean of an article-day's scores."""
    arts = list(arts)
    if not arts:
        return 0.0, 0
    half_life = max(config.half_life_hours, 0.5)
    weighted_sum = 0.0
    total_weight = 0.0
    for a in arts:
        age_hours = max(
            0.0, (day_anchor - a.published_at).total_seconds() / 3600.0
        )
        recency = 2.0 ** (-age_hours / half_life)
        src_w = config.weight_for_source(a.source)
        w = recency * src_w
        weighted_sum += a.raw_score * w
        total_weight += w
    if total_weight == 0.0:
        return 0.0, len(arts)
    return weighted_sum / total_weight, len(arts)


def _rolling_baseline_for(
    target: str,
    *,
    end_day: datetime,
    indices: dict[tuple[str, datetime], float],
    window_days: int = 30,
) -> tuple[float, float]:
    """Rolling mean + stddev of pre-computed daily indices for ``target``."""
    # Pull the last `window_days` indices for this target, strictly before
    # end_day.
    samples = [
        v for (sym, day), v in indices.items()
        if sym == target and day < end_day and (end_day - day).days <= window_days
    ]
    if not samples:
        return 0.0, 0.0
    mean = sum(samples) / len(samples)
    if len(samples) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    return mean, math.sqrt(var)


def simulate(
    frame: TrainingFrame, config: LearnedConfig
) -> tuple[float, float, int, int]:
    """Replay every closed prediction in ``frame`` under ``config``.

    Returns ``(hit_rate, cumulative_pnl, hits, directional_count)``.
    ``directional_count`` excludes FLAT (no trade). ``hit_rate`` is over
    directional + FLAT predictions combined (matches the backtester);
    cumulative_pnl is over directional only.
    """
    keys = frame.closed_prediction_keys
    if not keys:
        return 0.0, 0.0, 0, 0

    # Step 1: compute each (target, day)'s recency-weighted index under
    # this config. We need them all first before computing rolling
    # baselines for any single day.
    indices: dict[tuple[str, datetime], float] = {}
    counts: dict[tuple[str, datetime], int] = {}
    for key in keys:
        target, day = key
        # Anchor recency to end-of-day so newer-in-day articles weigh more,
        # mirroring the live predictor.
        anchor = datetime(day.year, day.month, day.day, 23, 59, tzinfo=timezone.utc)
        idx, n = _weighted_mean(
            frame.articles_by_day.get(key, []),
            day_anchor=anchor,
            config=config,
        )
        indices[key] = idx
        counts[key] = n

    hits = 0
    total_evaluated = 0
    directional = 0
    cum_pnl = 0.0
    floor = max(config.min_baseline_sigma, 1e-6)

    for key in keys:
        target, day = key
        idx = indices[key]
        n = counts[key]
        ret = frame.realised_by_day[key]

        if n < MIN_ARTICLES_FOR_CALL:
            label = "FLAT"
        else:
            mean, std = _rolling_baseline_for(
                target, end_day=day, indices=indices
            )
            sigma = max(std, floor)
            z = (idx - mean) / sigma
            label, _ = classify(z, threshold=config.threshold_sigma)

        # Hit rule mirrors backtest._label_matches.
        if label == "UP":
            hit = ret > 0
            cum_pnl += ret
            directional += 1
        elif label == "DOWN":
            hit = ret < 0
            cum_pnl += -ret
            directional += 1
        else:  # FLAT
            hit = abs(ret) < FLAT_NOISE_THRESHOLD

        total_evaluated += 1
        if hit:
            hits += 1

    hit_rate = hits / total_evaluated if total_evaluated else 0.0
    return hit_rate, cum_pnl, hits, directional


def blended_objective(
    frame: TrainingFrame, config: LearnedConfig
) -> float:
    """Hit-rate + 50% × cumulative PnL — the user-selected objective."""
    hit_rate, cum_pnl, _, _ = simulate(frame, config)
    return hit_rate + 0.5 * cum_pnl
