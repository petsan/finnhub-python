"""Tests for PR-7: affinity-blended per-stock predictor.

Covers:
* :class:`AffinityWeights` defaults + per-relationship lookup.
* SELF-only blend matches the legacy per-stock predictor when no
  related entities exist.
* COMPETITOR with negative weight inverts the contribution direction
  (positive competitor news → target headwind).
* SUPPLIER / CUSTOMER / PEER positively contribute.
* THEME_MEMBER co-membership: tickers sharing a theme contribute via
  the theme weight.
* Empty pool (no articles anywhere) returns None.
* Persisted Prediction uses the `+aff:default` model_version suffix.
* :func:`explain_blend` returns per-relationship contribution breakdown.
* :func:`predict_all_stocks_blended` skips blanks and gathers the rest.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import pytest
from sqlalchemy.orm import Session

from finn_predictor.predictor.blended import (
    AffinityWeights,
    BLENDED_MODEL_SUFFIX,
    DEFAULT_AFFINITY_WEIGHTS,
    explain_blend,
    predict_all_stocks_blended,
    predict_stock_blended,
)
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import NewsArticle, SentimentScore
from finn_predictor.storage.repo import (
    save_scores,
    upsert_articles,
    upsert_related_entity,
)
from tests.conftest import make_article, make_score


class _StubScorer(Scorer):
    def __init__(self, *, model_version: str = "stub-test"):
        self._mv = model_version

    @property
    def model_version(self) -> str:
        return self._mv

    def score(self, text: str) -> float:
        return 0.0  # unused in these tests; articles are pre-scored


D = datetime(2026, 5, 21, tzinfo=timezone.utc)


def _seed(
    session: Session,
    *,
    symbol: str,
    scores: Iterable[float],
    day: datetime = D,
    finnhub_id_start: int = 1,
) -> None:
    """Seed N scored ``company`` articles for ``symbol`` on ``day``.

    Only the articles inserted by *this call* are scored — calling
    ``_seed`` twice for the same symbol on different days does not
    re-score the earlier articles (which would trip the
    ``(article_id, model_version)`` UNIQUE constraint on
    sentiment_scores).
    """
    scores = list(scores)
    new_ids = list(range(finnhub_id_start, finnhub_id_start + len(scores)))
    arts = [
        make_article(
            finnhub_id=fid,
            category="company",
            symbol=symbol,
            published_at=day.replace(hour=12),
        )
        for fid in new_ids
    ]
    upsert_articles(session, arts)
    just_added = (
        session.query(NewsArticle)
        .filter(NewsArticle.finnhub_id.in_(new_ids))
        .order_by(NewsArticle.finnhub_id)
        .all()
    )
    save_scores(
        session,
        [
            make_score(a.id, sc, model_version="stub-test")
            for a, sc in zip(just_added, scores)
        ],
    )


# ---------------------------------------------------------------------------
# AffinityWeights dataclass
# ---------------------------------------------------------------------------

def test_affinity_weights_defaults_are_curated() -> None:
    """The shipped defaults match the design.md curated values."""
    w = DEFAULT_AFFINITY_WEIGHTS
    assert w.self_weight == 1.0
    assert w.competitor == -0.30
    assert w.supplier == 0.15
    assert w.customer == 0.25
    assert w.peer == 0.10
    assert w.theme_member == 0.10
    # Institutional holders contribute nothing today.
    assert w.institutional_holder == 0.0


@pytest.mark.parametrize(
    "rel,expected",
    [
        ("PEER", 0.10),
        ("COMPETITOR", -0.30),
        ("SUPPLIER", 0.15),
        ("CUSTOMER", 0.25),
        ("THEME_MEMBER", 0.10),
        ("INSTITUTIONAL_HOLDER", 0.0),
        ("UNKNOWN", 0.0),  # fallback for unmapped relationships
    ],
)
def test_weight_for_lookup(rel: str, expected: float) -> None:
    assert DEFAULT_AFFINITY_WEIGHTS.weight_for(rel) == expected


def test_affinity_weights_is_frozen() -> None:
    """Mutation must raise — same hardening pattern as Settings."""
    w = AffinityWeights()
    with pytest.raises(Exception):
        w.self_weight = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# predict_stock_blended — SELF-only behaviour
# ---------------------------------------------------------------------------

def test_blend_self_only_matches_unblended_direction(session: Session) -> None:
    """With no related entities, the blended call should agree on
    direction with the simple per-stock call (the math is just SELF
    contributions with weight 1.0)."""
    _seed(session, symbol="AAPL", scores=[0.5, 0.6, 0.7, 0.8])

    pred = predict_stock_blended(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
        min_baseline_sigma=0.01,
    )
    assert pred is not None
    assert pred.label == "UP"
    assert pred.article_count == 4
    assert pred.sentiment_index > 0


def test_blend_persisted_with_suffix_model_version(session: Session) -> None:
    """The Prediction row gets the `+aff:default` model_version
    suffix so blended and unblended predictions coexist."""
    _seed(session, symbol="AAPL", scores=[0.5, 0.6, 0.7, 0.8])
    pred = predict_stock_blended(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
        min_baseline_sigma=0.01,
    )
    assert pred is not None
    assert pred.model_version == "stub-test" + BLENDED_MODEL_SUFFIX


def test_blend_no_articles_returns_none(session: Session) -> None:
    pred = predict_stock_blended(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
    )
    assert pred is None


def test_blend_empty_symbol_raises(session: Session) -> None:
    with pytest.raises(ValueError):
        predict_stock_blended(
            session, scorer=_StubScorer(), symbol="", on_date=D,
        )
    with pytest.raises(ValueError):
        predict_stock_blended(
            session, scorer=_StubScorer(), symbol="   ", on_date=D,
        )


# ---------------------------------------------------------------------------
# COMPETITOR — negative weight inverts contribution direction
# ---------------------------------------------------------------------------

def test_blend_competitor_positive_news_pulls_target_down(session: Session) -> None:
    """The target has flat-ish news; the competitor has wildly positive
    news. With competitor weight = -0.30, the blended index should
    reflect a net negative pull on the target."""
    # Pre-day baseline: a few mildly-positive days so today's negative
    # blended index is clearly below the mean.
    for i, day in enumerate([D - timedelta(days=2), D - timedelta(days=1)]):
        _seed(
            session, symbol="AAPL", scores=[0.05, 0.04, 0.06],
            day=day, finnhub_id_start=100 + i * 10,
        )

    _seed(session, symbol="AAPL", scores=[0.0, 0.05, 0.0], day=D,
          finnhub_id_start=1)
    _seed(session, symbol="MSFT", scores=[0.9, 0.9, 0.9, 0.9], day=D,
          finnhub_id_start=200)

    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="COMPETITOR", rank=0,
    )

    pred = predict_stock_blended(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
        min_baseline_sigma=0.01,
    )
    assert pred is not None
    # Self contributes ~0; competitor contributes 4 articles × 0.9 ×
    # (-0.30) — net negative blended index.
    assert pred.sentiment_index < 0


def test_blend_supplier_positive_helps_target(session: Session) -> None:
    """Positive supplier news contributes positively to the blend
    (weight = +0.15)."""
    # Baseline: a few mildly-negative days so today's positive blend
    # is clearly above the mean. Different finnhub_id_start per
    # iteration to avoid colliding with the previous day's seed
    # (UNIQUE constraint on sentiment_scores forbids re-scoring an
    # article with the same model_version).
    for i, day in enumerate([D - timedelta(days=2), D - timedelta(days=1)]):
        _seed(
            session, symbol="AAPL", scores=[-0.05, -0.04, -0.06],
            day=day, finnhub_id_start=300 + i * 10,
        )

    _seed(session, symbol="AAPL", scores=[0.0, 0.0, 0.0], day=D,
          finnhub_id_start=1)
    _seed(session, symbol="TSM", scores=[0.7, 0.7, 0.7], day=D,
          finnhub_id_start=400)

    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="TSM",
        relationship="SUPPLIER", rank=0,
    )

    pred = predict_stock_blended(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
        min_baseline_sigma=0.01,
    )
    assert pred is not None
    assert pred.sentiment_index > 0


# ---------------------------------------------------------------------------
# THEME_MEMBER co-membership
# ---------------------------------------------------------------------------

def test_blend_theme_co_member_contributes(session: Session) -> None:
    """AAPL and NVDA are both in the `aiSemis` theme. NVDA's
    sentiment should contribute (small positive weight) to AAPL's
    blended index."""
    for sym in ("AAPL", "NVDA"):
        upsert_related_entity(
            session, source_symbol="aiSemis", related_symbol=sym,
            relationship="THEME_MEMBER",
        )
    _seed(session, symbol="AAPL", scores=[0.0, 0.0, 0.0], day=D)
    _seed(session, symbol="NVDA", scores=[0.8, 0.9, 0.85], day=D,
          finnhub_id_start=500)

    contribs = explain_blend(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
    )
    by_kind = {c.relationship: c for c in contribs}
    assert "THEME_MEMBER" in by_kind
    # Positive weighted_sum (theme weight × NVDA's positive scores).
    assert by_kind["THEME_MEMBER"].weighted_sum > 0
    assert by_kind["THEME_MEMBER"].article_count == 3


def test_blend_theme_excludes_self_from_co_members(session: Session) -> None:
    """The target itself is a member of its own theme — but its
    articles must not be double-counted under THEME_MEMBER."""
    for sym in ("AAPL", "NVDA"):
        upsert_related_entity(
            session, source_symbol="aiSemis", related_symbol=sym,
            relationship="THEME_MEMBER",
        )
    _seed(session, symbol="AAPL", scores=[1.0, 1.0, 1.0], day=D)

    contribs = explain_blend(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
    )
    by_kind = {c.relationship: c for c in contribs}
    # Without NVDA articles, the theme contribution should be zero.
    assert by_kind["THEME_MEMBER"].article_count == 0


def test_blend_no_themes_means_no_theme_contribution(session: Session) -> None:
    _seed(session, symbol="AAPL", scores=[0.5, 0.5, 0.5], day=D)
    contribs = explain_blend(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
    )
    by_kind = {c.relationship: c for c in contribs}
    # THEME_MEMBER not present in contributions when no theme edges exist.
    assert "THEME_MEMBER" not in by_kind


# ---------------------------------------------------------------------------
# explain_blend
# ---------------------------------------------------------------------------

def test_explain_blend_breaks_down_by_relationship(session: Session) -> None:
    _seed(session, symbol="AAPL", scores=[0.4, 0.5, 0.6], day=D)
    _seed(session, symbol="MSFT", scores=[0.2, 0.3], day=D, finnhub_id_start=600)
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="COMPETITOR",
    )

    contribs = explain_blend(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
    )
    by_kind = {c.relationship: c for c in contribs}
    assert by_kind["SELF"].article_count == 3
    assert by_kind["COMPETITOR"].article_count == 2
    # Competitor weight is negative, MSFT's positive scores → negative
    # weighted_sum.
    assert by_kind["COMPETITOR"].weighted_sum < 0


def test_explain_blend_empty_input_raises(session: Session) -> None:
    with pytest.raises(ValueError):
        explain_blend(session, scorer=_StubScorer(), symbol="", on_date=D)


# ---------------------------------------------------------------------------
# predict_all_stocks_blended
# ---------------------------------------------------------------------------

def test_predict_all_stocks_blended_skips_empty(session: Session) -> None:
    _seed(session, symbol="AAPL", scores=[0.5, 0.6, 0.7], day=D,
          finnhub_id_start=1)
    out = predict_all_stocks_blended(
        session, scorer=_StubScorer(),
        symbols=["", " ", "AAPL", "NEVERSEEN"],
        on_date=D, min_baseline_sigma=0.01,
    )
    # Blanks dropped silently; NEVERSEEN returns None (no articles);
    # AAPL gets through.
    assert len(out) == 1
    assert out[0].target_symbol == "AAPL"


def test_predict_all_stocks_blended_empty_list(session: Session) -> None:
    out = predict_all_stocks_blended(
        session, scorer=_StubScorer(), symbols=[], on_date=D,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

def test_blend_accepts_custom_weights(session: Session) -> None:
    """A caller can override the default weights — PR-8's learner uses
    this to inject fitted weights without touching the DB constant."""
    _seed(session, symbol="AAPL", scores=[0.1, 0.1, 0.1], day=D)
    _seed(session, symbol="MSFT", scores=[1.0, 1.0, 1.0], day=D,
          finnhub_id_start=700)
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="COMPETITOR",
    )

    # Heavy positive weight on COMPETITOR — flip the sign convention
    # for this one call to verify the override flows through.
    weights = AffinityWeights(
        self_weight=1.0, competitor=+1.0, peer=0.0, supplier=0.0,
        customer=0.0, theme_member=0.0,
    )
    contribs = explain_blend(
        session, scorer=_StubScorer(), symbol="AAPL", on_date=D,
        affinity_weights=weights,
    )
    by_kind = {c.relationship: c for c in contribs}
    # With +1.0 weight, competitor's positive scores contribute
    # positively (not negatively).
    assert by_kind["COMPETITOR"].weighted_sum > 0
