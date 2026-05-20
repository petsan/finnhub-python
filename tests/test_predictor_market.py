"""Iteration 1: whole-market predictor tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.market import (
    MIN_ARTICLES_FOR_CALL,
    THRESHOLD_SIGMA,
    classify,
    predict_market,
)
from finn_predictor.sentiment.vader import VaderScorer
from finn_predictor.storage.repo import (
    predictions_for,
    save_scores,
    upsert_articles,
)
from tests.conftest import make_article, make_score


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


# ---------------- classify() ----------------


def test_classify_threshold_boundaries() -> None:
    # Inside [-T, +T] should map to FLAT
    label, _ = classify(0.0)
    assert label == "FLAT"
    label, _ = classify(THRESHOLD_SIGMA)
    assert label == "FLAT"
    label, _ = classify(-THRESHOLD_SIGMA)
    assert label == "FLAT"


def test_classify_above_threshold_is_up() -> None:
    label, conf = classify(1.5)
    assert label == "UP"
    assert 0.0 < conf <= 1.0


def test_classify_below_threshold_is_down() -> None:
    label, conf = classify(-1.5)
    assert label == "DOWN"
    assert 0.0 < conf <= 1.0


def test_classify_confidence_clamped_to_one() -> None:
    _, conf = classify(5.0)
    assert conf == pytest.approx(1.0)


# ---------------- predict_market() ----------------


class _FixedScorer:
    """Tiny stand-in scorer with a deterministic model_version."""

    model_version = "vader-test"

    def score(self, text: str) -> float:  # not used by predict_market
        return 0.0

    def score_many(self, texts):
        return [0.0 for _ in texts]


def _seed_day(session, *, ids, scores, day):
    """Insert one article per id with corresponding score, all timestamped at `day`."""
    arts = [make_article(finnhub_id=i, published_at=day) for i in ids]
    upsert_articles(session, arts)
    persisted = session.query(type(arts[0])).filter(type(arts[0]).finnhub_id.in_(ids)).all()
    by_fh = {a.finnhub_id: a for a in persisted}
    save_scores(
        session,
        [make_score(by_fh[i].id, s, model_version="vader-test") for i, s in zip(ids, scores)],
    )


def test_predict_market_returns_none_without_articles(session) -> None:
    out = predict_market(session, scorer=_FixedScorer(), on_date=D)
    assert out is None


def test_predict_market_emits_flat_when_below_min_articles(session) -> None:
    _seed_day(session, ids=[1, 2], scores=[0.9, 0.9], day=D)
    pred = predict_market(session, scorer=_FixedScorer(), on_date=D)
    assert pred is not None
    assert pred.label == "FLAT"
    assert pred.confidence == 0.0
    assert pred.article_count == 2  # below MIN_ARTICLES_FOR_CALL


def test_predict_market_calls_up_for_strongly_positive_day(session) -> None:
    assert MIN_ARTICLES_FOR_CALL <= 3
    _seed_day(session, ids=list(range(10, 20)), scores=[0.8] * 10, day=D)
    pred = predict_market(session, scorer=_FixedScorer(), on_date=D)
    assert pred is not None
    assert pred.label == "UP"
    assert pred.article_count == 10
    assert pred.target_symbol == "^GSPC"
    assert pred.confidence > 0.0


def test_predict_market_calls_down_for_strongly_negative_day(session) -> None:
    _seed_day(session, ids=list(range(20, 30)), scores=[-0.8] * 10, day=D)
    pred = predict_market(session, scorer=_FixedScorer(), on_date=D)
    assert pred is not None
    assert pred.label == "DOWN"


def test_predict_market_is_idempotent_for_same_date(session) -> None:
    _seed_day(session, ids=list(range(40, 50)), scores=[0.8] * 10, day=D)
    a = predict_market(session, scorer=_FixedScorer(), on_date=D)
    b = predict_market(session, scorer=_FixedScorer(), on_date=D)
    assert a is not None and b is not None
    assert a.id == b.id  # upsert path

    rows = predictions_for(session, "^GSPC")
    assert len(rows) == 1


def test_predict_market_normalises_prediction_date_to_utc_midnight(session) -> None:
    """Two calls with different timestamps on the same UTC day must upsert
    the SAME row — fixes the duplicate-predictions bug."""
    _seed_day(session, ids=list(range(60, 70)), scores=[0.8] * 10, day=D)

    morning = D.replace(hour=3, minute=48, second=11, microsecond=399988)
    afternoon = D.replace(hour=16, minute=6, second=2, microsecond=278856)

    a = predict_market(session, scorer=_FixedScorer(), on_date=morning)
    b = predict_market(session, scorer=_FixedScorer(), on_date=afternoon)
    assert a is not None and b is not None
    assert a.id == b.id

    rows = predictions_for(session, "^GSPC")
    assert len(rows) == 1
    # prediction_date stored at start-of-day UTC.
    stored = rows[0].prediction_date
    expected_naive = D.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    assert (stored.replace(tzinfo=None) if stored.tzinfo else stored) == expected_naive


def test_predict_market_separate_rows_on_different_days(session) -> None:
    """Same predictor on two different UTC days → two rows."""
    day1 = D
    day2 = D + timedelta(days=1)
    _seed_day(session, ids=list(range(70, 80)), scores=[0.7] * 10, day=day1)
    _seed_day(session, ids=list(range(80, 90)), scores=[-0.7] * 10, day=day2)
    predict_market(session, scorer=_FixedScorer(), on_date=day1)
    predict_market(session, scorer=_FixedScorer(), on_date=day2)
    rows = predictions_for(session, "^GSPC")
    assert len(rows) == 2


def test_predict_market_uses_baseline_to_dampen_calls(session) -> None:
    # Trailing 10 days are mildly positive (mean ~ 0.4); today is identical.
    # That should NOT trigger UP — the z-score will be ~0.
    for i in range(10):
        _seed_day(
            session,
            ids=[1000 + i * 10 + j for j in range(5)],
            scores=[0.4] * 5,
            day=D - timedelta(days=i + 1),
        )
    _seed_day(session, ids=[9001, 9002, 9003, 9004, 9005], scores=[0.4] * 5, day=D)

    pred = predict_market(session, scorer=_FixedScorer(), on_date=D)
    assert pred is not None
    assert pred.label == "FLAT"


def test_predict_market_with_real_vader(session) -> None:
    """End-to-end sanity check using the real VaderScorer."""
    scorer = VaderScorer()
    arts = [
        make_article(finnhub_id=i, published_at=D, headline="Markets soar to record highs on stellar earnings")
        for i in range(50, 60)
    ]
    upsert_articles(session, arts)

    # Score via the real scorer
    from finn_predictor.ingestion.jobs import score_pending_articles
    n = score_pending_articles(session, scorer)
    assert n == 10

    pred = predict_market(session, scorer=scorer, on_date=D)
    assert pred is not None
    assert pred.label in {"UP", "FLAT"}
    assert pred.article_count == 10
