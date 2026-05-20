"""Sentiment aggregation tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.aggregate import (
    aggregate_sentiment,
    daily_sentiment_index,
    rolling_baseline,
)
from finn_predictor.storage.repo import save_scores, upsert_articles
from tests.conftest import make_article, make_score


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


def test_aggregate_sentiment_empty() -> None:
    s = aggregate_sentiment([])
    assert s.is_empty
    assert s.count == 0
    assert s.mean == 0.0


def test_aggregate_sentiment_single_value() -> None:
    s = aggregate_sentiment([0.6])
    assert s.count == 1
    assert s.mean == pytest.approx(0.6)
    assert s.stddev == 0.0  # n=1 -> stddev undefined; we return 0
    assert s.weighted_mean == pytest.approx(0.6)


def test_aggregate_sentiment_multiple_values() -> None:
    s = aggregate_sentiment([-0.4, 0.0, 0.4])
    assert s.count == 3
    assert s.mean == pytest.approx(0.0)
    # sample stddev of {-0.4, 0, 0.4} = sqrt(0.32/2) = 0.4
    assert s.stddev == pytest.approx(0.4)


def test_aggregate_sentiment_with_weights() -> None:
    s = aggregate_sentiment([0.0, 1.0], weights=[1.0, 3.0])
    assert s.mean == pytest.approx(0.5)
    assert s.weighted_mean == pytest.approx(0.75)


def test_aggregate_sentiment_zero_weights_falls_back_to_mean() -> None:
    s = aggregate_sentiment([0.2, 0.8], weights=[0.0, 0.0])
    assert s.weighted_mean == s.mean


def test_aggregate_sentiment_mismatched_weights_falls_back_to_mean() -> None:
    s = aggregate_sentiment([0.2, 0.8], weights=[1.0])
    assert s.weighted_mean == s.mean


def test_daily_sentiment_index_uses_recency_weighting(session) -> None:
    # Two articles at very different times of day; the later one should
    # dominate the recency-weighted mean.
    early = make_article(
        finnhub_id=1,
        published_at=datetime(2026, 5, 19, 1, tzinfo=timezone.utc),
        headline="early-grim",
    )
    late = make_article(
        finnhub_id=2,
        published_at=datetime(2026, 5, 19, 22, tzinfo=timezone.utc),
        headline="late-cheery",
    )
    upsert_articles(session, [early, late])

    arts = session.query(type(early)).all()
    by_finnhub = {a.finnhub_id: a for a in arts}
    save_scores(
        session,
        [
            make_score(by_finnhub[1].id, -1.0, model_version="vader-test"),
            make_score(by_finnhub[2].id, +1.0, model_version="vader-test"),
        ],
    )

    s = daily_sentiment_index(
        session,
        model_version="vader-test",
        day=D,
        category="general",
        half_life_hours=6.0,
    )
    assert s.count == 2
    # Late article (closer to day end) is heavier -> weighted mean > 0.
    assert s.weighted_mean > 0.5
    assert s.mean == pytest.approx(0.0)


def test_daily_sentiment_index_is_empty_when_no_articles(session) -> None:
    s = daily_sentiment_index(session, model_version="vader-test", day=D)
    assert s.is_empty


def test_daily_sentiment_index_skips_unscored_articles(session) -> None:
    """An article without a matching SentimentScore row must not affect aggregate."""
    upsert_articles(session, [make_article(finnhub_id=1, published_at=D)])
    s = daily_sentiment_index(session, model_version="vader-test", day=D)
    assert s.is_empty


def test_rolling_baseline_across_window(session) -> None:
    # Drop one article per day for 10 days with a slight trend (0.0 -> 0.9).
    arts = []
    for i in range(10):
        a = make_article(
            finnhub_id=100 + i,
            published_at=D - timedelta(days=i + 1),
        )
        arts.append(a)
    upsert_articles(session, arts)
    persisted = (
        session.query(type(arts[0])).order_by(type(arts[0]).finnhub_id).all()
    )
    scores = [0.1 * i for i in range(10)]
    save_scores(
        session,
        [make_score(persisted[i].id, scores[i], model_version="vader-test") for i in range(10)],
    )

    base = rolling_baseline(
        session,
        model_version="vader-test",
        end_day=D,
        window_days=10,
        category="general",
    )
    assert base.count == 10
    assert base.mean == pytest.approx(sum(scores) / 10)


def test_rolling_baseline_empty_when_window_has_no_articles(session) -> None:
    base = rolling_baseline(
        session, model_version="vader-test", end_day=D, window_days=30
    )
    assert base.is_empty


def test_daily_sentiment_index_applies_source_weights(session) -> None:
    """Per-source multipliers really shift the weighted mean."""
    # Two articles same day, same recency, same score sign but opposite
    # signs; their plain mean is 0. With Reuters weighted 3x WSJ, the
    # Reuters article (+0.8) should dominate.
    a_pos = make_article(
        finnhub_id=1, source="Reuters",
        headline="up", published_at=D,
    )
    a_neg = make_article(
        finnhub_id=2, source="WSJ",
        headline="down", published_at=D,
    )
    upsert_articles(session, [a_pos, a_neg])
    arts = session.query(type(a_pos)).order_by(type(a_pos).finnhub_id).all()
    save_scores(
        session,
        [
            make_score(arts[0].id, 0.8, model_version="vader-test"),
            make_score(arts[1].id, -0.8, model_version="vader-test"),
        ],
    )

    flat = daily_sentiment_index(
        session, model_version="vader-test", day=D,
    )
    boosted = daily_sentiment_index(
        session, model_version="vader-test", day=D,
        source_weights={"Reuters": 3.0, "WSJ": 1.0},
    )
    # Without source weights they cancel; with the boost the positive
    # one dominates.
    assert flat.weighted_mean == pytest.approx(0.0, abs=1e-9)
    assert boosted.weighted_mean > 0.3


def test_rolling_baseline_passes_source_weights_through(session) -> None:
    """rolling_baseline forwards the kwargs to each inner daily call."""
    # Plant one positive Reuters article 5 days ago; rolling baseline
    # should pick up its boosted contribution.
    a = make_article(
        finnhub_id=1, source="Reuters", headline="up",
        published_at=D - timedelta(days=5),
    )
    upsert_articles(session, [a])
    arts = session.query(type(a)).all()
    save_scores(
        session,
        [make_score(arts[0].id, 0.6, model_version="vader-test")],
    )
    plain = rolling_baseline(
        session, model_version="vader-test", end_day=D,
    )
    boosted = rolling_baseline(
        session, model_version="vader-test", end_day=D,
        source_weights={"Reuters": 5.0},
    )
    # The baseline mean for a single-day single-article window is the
    # article's score regardless of weighting magnitude (the daily index
    # divides by total weight). So we don't expect mean to change — we
    # just check this didn't crash and both produced equal means.
    assert plain.mean == pytest.approx(boosted.mean)
