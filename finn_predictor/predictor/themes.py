"""Per-theme directional predictor (PR-6).

Mirrors :mod:`predictor.sectors` semantically — aggregate the day's
recency-weighted sentiment across a theme's constituent tickers,
classify the result against a rolling baseline, emit a directional
call — but **does not** write to the :class:`Prediction` table.

Why not Prediction rows? Theme codes can be longer than the
``target_symbol String(16)`` column accommodates
(``financialExchangesData`` is 22 chars), so persisting themes there
would either require a schema widening (Postgres migration) or a
truncation scheme that risks collisions across operator-added themes.
For PR-6 we sidestep both by returning a :class:`ThemePrediction`
dataclass; the Today / Themes tab renders it from a fresh aggregation
each pageload. When PR-7's affinity-blend ships, the THEME_MEMBER
edges in :class:`RelatedEntity` are the canonical lookup — the
ThemePrediction is just a UI convenience.

Curated theme list :data:`DEFAULT_THEME_CODES` matches the 10–15 codes
sign-off in ``progress.md`` (2026-05-21 session). Operators can add
their own via the ``add-theme`` CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.predictor.aggregate import (
    _recency_weight,
    aggregate_sentiment,
)
from finn_predictor.predictor.market import (
    MIN_ARTICLES_FOR_CALL,
    MIN_BASELINE_SIGMA,
    THRESHOLD_SIGMA,
    classify,
)
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import (
    NewsArticle,
    RelatedEntity,
    SentimentScore,
)
from finn_predictor.storage.repo import (
    articles_in_window,
    related_entities_for,
    utc_day_window,
)


logger = logging.getLogger(__name__)


# Curated seed set. Operator can add more via `add-theme <code>` CLI.
# These are codes Finnhub's /stock/investment-theme accepts as of
# 2026-Q1; collected from the public docs + a few well-known themes
# the predictor's target audience asks for. Keeping the list small
# keeps ingestion cheap (one API call per theme on each refresh).
DEFAULT_THEME_CODES: tuple[str, ...] = (
    "financialExchangesData",
    "cyberSecurity",
    "cleanEnergy",
    "electricVehicles",
    "aiSemis",
    "cloudComputing",
    "robotics",
    "spaceExploration",
    "digitalPayments",
    "nuclearEnergy",
    "semiconductor",
    "futureMobility",
)


@dataclass(frozen=True)
class ThemePrediction:
    """A directional call for a single investment theme.

    Not persisted — recomputed on each render. Carries enough state
    for the UI to surface the call alongside its supporting metadata
    (which sectors did this draw from, how many articles, how strong
    a signal vs the rolling baseline).
    """

    theme_code: str
    theme_name: str
    constituent_count: int
    label: str           # UP / DOWN / FLAT
    confidence: float
    sentiment_index: float
    article_count: int
    model_version: str
    prediction_date: datetime


def _theme_scored_articles(
    session: Session,
    *,
    constituents: list[str],
    model_version: str,
    day: datetime,
) -> list[tuple[NewsArticle, float]]:
    """Pair each scored article for the theme's constituents with its score.

    Mirror of :func:`predictor.sectors._sector_scored_articles` — we
    walk symbol-by-symbol so a theme with hundreds of constituents
    stays streamable. Default curated themes top out at ~50.
    """
    start, end = utc_day_window(day)
    pairs: list[tuple[NewsArticle, float]] = []
    for sym in constituents:
        arts = articles_in_window(
            session, start, end, symbol=sym, category="company"
        )
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


def predict_theme(
    session: Session,
    *,
    scorer: Scorer,
    theme_code: str,
    on_date: Optional[datetime] = None,
    threshold_sigma: Optional[float] = None,
    min_baseline_sigma: Optional[float] = None,
    half_life_hours: float = 24.0,
    baseline_window_days: int = 30,
) -> Optional[ThemePrediction]:
    """Compute a directional call for the theme.

    Returns ``None`` when:
      * no THEME_MEMBER edges exist for ``theme_code`` (the theme
        wasn't ingested yet, or the call was made before
        ``refresh_investment_themes``), or
      * the constituents have no scored articles in the target day's
        window (insufficient signal — same "no data" path as the
        market and sector predictors).

    The classification uses the same z-score rule as the market
    predictor; ``threshold_sigma`` and ``min_baseline_sigma`` fall
    back to the predictor-wide defaults when unset.
    """
    if not theme_code:
        raise ValueError("theme_code must be a non-empty string")

    members = related_entities_for(
        session, theme_code, relationship="THEME_MEMBER"
    )
    if not members:
        return None

    constituents = [m.related_symbol for m in members]

    threshold = (
        THRESHOLD_SIGMA if threshold_sigma is None else float(threshold_sigma)
    )
    floor = (
        MIN_BASELINE_SIGMA if min_baseline_sigma is None
        else float(min_baseline_sigma)
    )

    target_day = on_date or datetime.now(timezone.utc)
    target_day_utc = (
        target_day if target_day.tzinfo else
        target_day.replace(tzinfo=timezone.utc)
    )

    pairs = _theme_scored_articles(
        session,
        constituents=constituents,
        model_version=scorer.model_version,
        day=target_day_utc,
    )
    if len(pairs) < MIN_ARTICLES_FOR_CALL:
        return None

    # Recency weighting anchored at end-of-day-UTC; same machinery the
    # market and sector predictors use.
    _, day_end = utc_day_window(target_day_utc)
    weights = [
        _recency_weight(a.published_at, day_end, half_life_hours)
        for a, _ in pairs
    ]
    scores = [s for _, s in pairs]
    summary = aggregate_sentiment(scores, weights=weights)

    # Rolling baseline across previous days' theme indices. We walk
    # each day in the lookback window, compute its theme-level
    # sentiment_index (mean across all constituents' scored articles),
    # then summarise the series — :func:`aggregate.rolling_baseline`
    # is scoped to a single symbol so we do the per-day walk ourselves
    # and feed the daily means into :func:`aggregate_sentiment`.
    baseline_start = target_day_utc - timedelta(days=baseline_window_days)
    baseline_indices: list[float] = []
    cursor = baseline_start
    while cursor < target_day_utc:
        prior = _theme_scored_articles(
            session,
            constituents=constituents,
            model_version=scorer.model_version,
            day=cursor,
        )
        if prior:
            prior_summary = aggregate_sentiment([s for _, s in prior])
            baseline_indices.append(prior_summary.mean)
        cursor = cursor + timedelta(days=1)

    baseline = aggregate_sentiment(baseline_indices)
    # Z-score of today's weighted mean against the rolling baseline.
    # Floor the denominator so a brand-new theme (no baseline history)
    # doesn't divide by zero and pin confidence at 1.0 on day-1.
    denom = max(baseline.stddev, floor)
    z = (summary.weighted_mean - baseline.mean) / denom if denom > 0 else 0.0
    label, confidence = classify(z, threshold=threshold)

    return ThemePrediction(
        theme_code=theme_code,
        theme_name=theme_code,  # caller (UI) overrides via InvestmentTheme lookup
        constituent_count=len(constituents),
        label=label,
        confidence=confidence,
        sentiment_index=summary.weighted_mean,
        article_count=summary.count,
        model_version=scorer.model_version,
        prediction_date=target_day_utc,
    )


def predict_all_themes(
    session: Session,
    *,
    scorer: Scorer,
    on_date: Optional[datetime] = None,
    theme_codes: Optional[list[str]] = None,
) -> list[ThemePrediction]:
    """Run :func:`predict_theme` across every ingested theme.

    When ``theme_codes`` is None, iterates every :class:`InvestmentTheme`
    in the DB. Themes with no constituents or no fresh articles are
    silently skipped — same pattern as :func:`predict_all_sectors`.
    Output is sorted by descending confidence so the strongest signals
    surface first in the UI.
    """
    from finn_predictor.storage.models import InvestmentTheme

    if theme_codes is None:
        themes = list(
            session.scalars(select(InvestmentTheme).order_by(InvestmentTheme.theme_code))
        )
        codes = [t.theme_code for t in themes]
        name_by_code = {t.theme_code: t.name for t in themes}
    else:
        codes = list(theme_codes)
        # The caller has supplied raw codes — look up names where we
        # have them, fall back to the code itself otherwise.
        themes = list(
            session.scalars(
                select(InvestmentTheme).where(InvestmentTheme.theme_code.in_(codes))
            )
        )
        name_by_code = {t.theme_code: t.name for t in themes}

    out: list[ThemePrediction] = []
    for code in codes:
        try:
            pred = predict_theme(
                session, scorer=scorer, theme_code=code, on_date=on_date
            )
        except Exception:
            # Per-theme failure isolation — don't let one bad code kill
            # the rest of the sweep.
            logger.exception("predict_theme failed for %s", code)
            continue
        if pred is None:
            continue
        # Patch in the operator-friendly name if we have one.
        if code in name_by_code:
            pred = ThemePrediction(
                theme_code=pred.theme_code,
                theme_name=name_by_code[code],
                constituent_count=pred.constituent_count,
                label=pred.label,
                confidence=pred.confidence,
                sentiment_index=pred.sentiment_index,
                article_count=pred.article_count,
                model_version=pred.model_version,
                prediction_date=pred.prediction_date,
            )
        out.append(pred)
    out.sort(key=lambda p: p.confidence, reverse=True)
    return out
