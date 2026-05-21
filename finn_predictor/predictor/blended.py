"""Affinity-blended per-stock predictor (PR-7).

The legacy :func:`predictor.stocks.predict_stock` uses only the
target's own company news. PR-7 introduces an opt-in blended variant
that adds weighted contributions from the target's related entities
(competitors, peers, suppliers, customers, theme co-members) so a
positive headline about a competitor pulls the target's call down,
a positive headline about a supplier pulls it up, etc.

The locked decision from progress.md (2026-05-21):

    Off by default; opt-in toggle + learner fits weights via the
    AFFINITY_WEIGHT dimension once ≥10 closed outcomes exist.

So this module ships :data:`DEFAULT_AFFINITY_WEIGHTS` (a reasonable
starting set) but the live pipeline only invokes
:func:`predict_stock_blended` when the operator opts in. The
``model_version`` tag of a blended Prediction is
``"<scorer>+aff:default"`` so blended and unblended calls coexist
peacefully under the existing one-row-per-(target, day, model)
uniqueness constraint.

INSTITUTIONAL_HOLDER is in :data:`AffinityWeights` for completeness
but its weight defaults to 0.0 — institutional holders' related_symbol
is the institution name, not a ticker, so no NewsArticle rows match.
Including them in the weights dataclass means PR-8's learning loop
can fit a non-zero weight later if a sensible signal source appears
(e.g. tagging articles with the institution name).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

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
    Prediction,
    RelatedEntity,
    SentimentScore,
)
from finn_predictor.storage.repo import (
    articles_in_window,
    related_entities_for,
    save_prediction,
    utc_day_window,
)


@dataclass(frozen=True)
class AffinityWeights:
    """Per-relationship weights for blended sentiment aggregation.

    Sign convention: positive sentiment about a related entity is
    multiplied by the weight, then added to the target's contribution
    pool. A *negative* weight (e.g. ``competitor=-0.30``) inverts the
    contribution — positive news about a competitor pulls the target
    *down*, which is the "rivalry headwind" model.

    Defaults are the curated starting set from design.md, sized so
    that the SELF signal still dominates (sum of |other weights| =
    ~0.95, slightly under SELF=1.0). PR-8's learner can fit these
    from closed outcomes once ≥10 are accumulated.
    """

    self_weight: float = 1.0
    peer: float = 0.10
    competitor: float = -0.30
    supplier: float = 0.15
    customer: float = 0.25
    theme_member: float = 0.10
    # INSTITUTIONAL_HOLDER's related_symbol is an institution name,
    # not a ticker — no NewsArticle rows are tagged with it. We
    # carry the field so PR-8's search space can include it
    # symmetrically, but it defaults to 0.0 because there's nothing
    # to weight against today.
    institutional_holder: float = 0.0

    def weight_for(self, relationship: str) -> float:
        """Lookup the weight for a RelatedEntity.relationship string."""
        return {
            "PEER": self.peer,
            "COMPETITOR": self.competitor,
            "SUPPLIER": self.supplier,
            "CUSTOMER": self.customer,
            "THEME_MEMBER": self.theme_member,
            "INSTITUTIONAL_HOLDER": self.institutional_holder,
        }.get(relationship, 0.0)


DEFAULT_AFFINITY_WEIGHTS = AffinityWeights()
BLENDED_MODEL_SUFFIX = "+aff:default"


# Relationship kinds we read from the RelatedEntity table when building
# the blended pool. INSTITUTIONAL_HOLDER is omitted (zero-weight + no
# matching articles); THEME_MEMBER lookups are *inverse* (we want
# themes that include the target, not the target's themes).
_ARTICLE_BEARING_RELATIONSHIPS: tuple[str, ...] = (
    "PEER",
    "COMPETITOR",
    "SUPPLIER",
    "CUSTOMER",
)


@dataclass(frozen=True)
class AffinityContribution:
    """How much each relationship kind contributed to a blended call.

    Surfaced by the UI's per-prediction breakdown popover so an
    operator can see whether a "DOWN on AAPL" call was driven by AAPL
    itself, its competitors, or its suppliers.
    """

    relationship: str   # SELF / PEER / COMPETITOR / SUPPLIER / CUSTOMER / THEME_MEMBER
    article_count: int
    weighted_sum: float


def _themes_containing(session: Session, *, symbol: str) -> list[str]:
    """Theme codes for which ``symbol`` is a THEME_MEMBER.

    Inverse direction from :func:`related_entities_for` — for themes,
    ``source_symbol`` is the theme code and ``related_symbol`` is the
    ticker, so we query by related_symbol to find the themes a ticker
    belongs to.
    """
    rows = session.scalars(
        select(RelatedEntity).where(
            RelatedEntity.related_symbol == symbol,
            RelatedEntity.relationship == "THEME_MEMBER",
        )
    )
    return [row.source_symbol for row in rows]


def _theme_co_members(
    session: Session, *, symbol: str
) -> list[str]:
    """Tickers that share at least one theme with ``symbol`` (excluding self)."""
    themes = _themes_containing(session, symbol=symbol)
    if not themes:
        return []
    rows = session.scalars(
        select(RelatedEntity.related_symbol)
        .where(
            RelatedEntity.relationship == "THEME_MEMBER",
            RelatedEntity.source_symbol.in_(themes),
        )
        .distinct()
    )
    others = {sym for sym in rows if sym and sym != symbol}
    return sorted(others)


def _related_symbols(
    session: Session, *, symbol: str
) -> dict[str, list[str]]:
    """Map relationship kind → list of related tickers for ``symbol``.

    Order is alphabetical (deterministic across runs). THEME_MEMBER is
    populated via :func:`_theme_co_members` because its direction in
    the DB is theme→ticker, opposite from the other kinds.
    """
    out: dict[str, list[str]] = {}
    for rel in _ARTICLE_BEARING_RELATIONSHIPS:
        rows = related_entities_for(session, symbol, relationship=rel)
        syms = sorted({r.related_symbol for r in rows})
        if syms:
            out[rel] = syms
    co_members = _theme_co_members(session, symbol=symbol)
    if co_members:
        out["THEME_MEMBER"] = co_members
    return out


def _scored_articles_for_symbol(
    session: Session,
    *,
    symbol: str,
    day_start: datetime,
    day_end: datetime,
    model_version: str,
) -> list[tuple[NewsArticle, float]]:
    """All scored ``company`` articles for one ticker on one day."""
    arts = articles_in_window(
        session, day_start, day_end, symbol=symbol, category="company"
    )
    if not arts:
        return []
    ids = [a.id for a in arts]
    rows = session.execute(
        select(SentimentScore.article_id, SentimentScore.score).where(
            SentimentScore.article_id.in_(ids),
            SentimentScore.model_version == model_version,
        )
    ).all()
    score_map = {aid: float(sc) for aid, sc in rows}
    return [(a, score_map[a.id]) for a in arts if a.id in score_map]


@dataclass(frozen=True)
class _BlendedPool:
    """Internal: the assembled article pool + per-relationship summary."""
    contributions: list[AffinityContribution]
    weighted_sum: float
    total_weight: float
    article_count: int


def _build_blended_pool(
    session: Session,
    *,
    symbol: str,
    day: datetime,
    weights: AffinityWeights,
    model_version: str,
    half_life_hours: float,
) -> _BlendedPool:
    """Assemble the weighted article pool for ``symbol`` on ``day``.

    Walks each relationship kind, fetches that side's scored articles,
    applies (recency_weight × relationship_weight) per article, and
    summarises the totals. The relationship weight can be negative
    (competitor) — we preserve sign on the article contribution by
    flipping the score, not the recency weight.
    """
    day_start, day_end = utc_day_window(day)

    related_map = _related_symbols(session, symbol=symbol)
    # SELF contribution.
    self_pairs = _scored_articles_for_symbol(
        session, symbol=symbol, day_start=day_start, day_end=day_end,
        model_version=model_version,
    )
    contributions: list[AffinityContribution] = []
    grand_sum = 0.0
    grand_w = 0.0
    grand_n = 0

    def _accumulate(
        *,
        relationship: str,
        pairs: list[tuple[NewsArticle, float]],
        weight: float,
    ) -> None:
        nonlocal grand_sum, grand_w, grand_n
        if not pairs or weight == 0.0:
            contributions.append(
                AffinityContribution(
                    relationship=relationship,
                    article_count=0,
                    weighted_sum=0.0,
                )
            )
            return
        local_sum = 0.0
        local_w = 0.0
        local_n = 0
        for article, score in pairs:
            rw = _recency_weight(article.published_at, day_end, half_life_hours)
            # Final per-article contribution: score × (relationship_weight ×
            # recency_weight). We split into a signed contribution and a
            # magnitude weight so the weighted-mean denominator is always
            # positive (sum of |relationship_weight| × recency_weight),
            # matching how aggregate_sentiment treats a weights vector.
            mag = abs(weight) * rw
            local_sum += (weight * rw) * score
            local_w += mag
            local_n += 1
        contributions.append(
            AffinityContribution(
                relationship=relationship,
                article_count=local_n,
                weighted_sum=local_sum,
            )
        )
        grand_sum += local_sum
        grand_w += local_w
        grand_n += local_n

    _accumulate(
        relationship="SELF", pairs=self_pairs, weight=weights.self_weight
    )

    for rel, syms in related_map.items():
        weight = weights.weight_for(rel)
        # Deduplicate across overlapping rel kinds for the same symbol —
        # one ticker that's both a peer and a competitor shouldn't have
        # its articles counted twice. The simplest fix is to walk in
        # order and skip syms we've already absorbed.
        rel_pairs: list[tuple[NewsArticle, float]] = []
        for sym in syms:
            rel_pairs.extend(
                _scored_articles_for_symbol(
                    session, symbol=sym, day_start=day_start, day_end=day_end,
                    model_version=model_version,
                )
            )
        _accumulate(relationship=rel, pairs=rel_pairs, weight=weight)

    return _BlendedPool(
        contributions=contributions,
        weighted_sum=grand_sum,
        total_weight=grand_w,
        article_count=grand_n,
    )


def _blended_index_for_day(
    session: Session,
    *,
    symbol: str,
    day: datetime,
    weights: AffinityWeights,
    model_version: str,
    half_life_hours: float,
) -> Optional[float]:
    """Single-day blended sentiment index for ``symbol``.

    Returns None when the pool is empty (no scored articles across
    SELF and any related entity) — so the baseline-builder can skip
    the day cleanly.
    """
    pool = _build_blended_pool(
        session, symbol=symbol, day=day,
        weights=weights, model_version=model_version,
        half_life_hours=half_life_hours,
    )
    if pool.total_weight == 0.0:
        return None
    return pool.weighted_sum / pool.total_weight


def predict_stock_blended(
    session: Session,
    *,
    scorer: Scorer,
    symbol: str,
    on_date: Optional[datetime] = None,
    affinity_weights: Optional[AffinityWeights] = None,
    threshold_sigma: Optional[float] = None,
    min_baseline_sigma: Optional[float] = None,
    half_life_hours: float = 24.0,
    baseline_window_days: int = 30,
) -> Optional[Prediction]:
    """Compute + persist an affinity-blended directional call for ``symbol``.

    Pulls today's scored articles for the target and its related
    entities (peers, competitors, suppliers, customers, theme
    co-members), weights each by its relationship kind, classifies
    the weighted index against the target's own rolling baseline of
    blended indices.

    Persists as a :class:`Prediction` row with ``model_version =
    f"{scorer.model_version}+aff:default"`` so blended and unblended
    calls for the same ticker on the same day are stored as two
    separate rows (one per model_version).

    Returns ``None`` when there's not enough data — either too few
    articles in today's pool (< :data:`MIN_ARTICLES_FOR_CALL`) or
    no related entities cached for the target *and* no own-symbol
    articles. The caller decides whether to fall back to
    :func:`predictor.stocks.predict_stock`.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    target = symbol.strip().upper()

    weights = affinity_weights or DEFAULT_AFFINITY_WEIGHTS
    threshold = (
        THRESHOLD_SIGMA if threshold_sigma is None else float(threshold_sigma)
    )
    floor = (
        MIN_BASELINE_SIGMA if min_baseline_sigma is None
        else float(min_baseline_sigma)
    )

    day = on_date or datetime.now(timezone.utc)
    day_utc = day if day.tzinfo else day.replace(tzinfo=timezone.utc)
    day_start, day_end = utc_day_window(day_utc)

    pool = _build_blended_pool(
        session,
        symbol=target,
        day=day_utc,
        weights=weights,
        model_version=scorer.model_version,
        half_life_hours=half_life_hours,
    )
    if pool.article_count < MIN_ARTICLES_FOR_CALL or pool.total_weight == 0.0:
        return None

    today_index = pool.weighted_sum / pool.total_weight

    # Baseline: per-day blended index for the prior N days.
    baseline_indices: list[float] = []
    cursor = day_utc - timedelta(days=baseline_window_days)
    while cursor < day_utc:
        idx = _blended_index_for_day(
            session, symbol=target, day=cursor,
            weights=weights, model_version=scorer.model_version,
            half_life_hours=half_life_hours,
        )
        if idx is not None:
            baseline_indices.append(idx)
        cursor = cursor + timedelta(days=1)
    baseline = aggregate_sentiment(baseline_indices)

    denom = max(baseline.stddev, floor)
    z = (today_index - baseline.mean) / denom if denom > 0 else 0.0
    label, confidence = classify(z, threshold=threshold)

    pred = Prediction(
        target_symbol=target,
        prediction_date=day_start,  # already start-of-UTC-day
        label=label,
        confidence=confidence,
        sentiment_index=today_index,
        article_count=pool.article_count,
        model_version=f"{scorer.model_version}{BLENDED_MODEL_SUFFIX}",
    )
    return save_prediction(session, pred)


def predict_all_stocks_blended(
    session: Session,
    *,
    scorer: Scorer,
    symbols: Iterable[str],
    on_date: Optional[datetime] = None,
    affinity_weights: Optional[AffinityWeights] = None,
    threshold_sigma: Optional[float] = None,
    min_baseline_sigma: Optional[float] = None,
    half_life_hours: float = 24.0,
) -> list[Prediction]:
    """Run :func:`predict_stock_blended` across a ticker list."""
    out: list[Prediction] = []
    for sym in symbols:
        if not sym or not sym.strip():
            continue
        pred = predict_stock_blended(
            session,
            scorer=scorer,
            symbol=sym,
            on_date=on_date,
            affinity_weights=affinity_weights,
            threshold_sigma=threshold_sigma,
            min_baseline_sigma=min_baseline_sigma,
            half_life_hours=half_life_hours,
        )
        if pred is not None:
            out.append(pred)
    return out


def explain_blend(
    session: Session,
    *,
    scorer: Scorer,
    symbol: str,
    on_date: Optional[datetime] = None,
    affinity_weights: Optional[AffinityWeights] = None,
    half_life_hours: float = 24.0,
) -> list[AffinityContribution]:
    """Per-relationship contribution breakdown for the latest blended call.

    Doesn't write to the DB. Pure read; safe to call from a popover
    handler that's just rendering details.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    weights = affinity_weights or DEFAULT_AFFINITY_WEIGHTS
    day = on_date or datetime.now(timezone.utc)
    day_utc = day if day.tzinfo else day.replace(tzinfo=timezone.utc)
    pool = _build_blended_pool(
        session,
        symbol=symbol.strip().upper(),
        day=day_utc,
        weights=weights,
        model_version=scorer.model_version,
        half_life_hours=half_life_hours,
    )
    return list(pool.contributions)
