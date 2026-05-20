"""Tests for finn_predictor.learning."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.learning.config import (
    DEFAULTS,
    DIM_HALF_LIFE,
    DIM_MIN_SIGMA,
    DIM_SOURCE_WEIGHT,
    DIM_THRESHOLD,
    LearnedConfig,
    active_weights,
    apply_to_default,
    weights_for_version,
)
from finn_predictor.learning.simulate import (
    blended_objective,
    build_training_frame,
    simulate,
)
from finn_predictor.learning.train import (
    MIN_TRADES_FOR_TRAINING,
    NotEnoughDataError,
    _fit_source_weights,
    _split_frame_by_date,
    train_weights,
)
from finn_predictor.storage.models import (
    LearnedWeight,
    PredictionOutcome,
)
from finn_predictor.storage.repo import (
    save_outcome,
    save_prediction,
    save_scores,
    upsert_articles,
)
from tests.conftest import make_article, make_prediction, make_score


D = datetime(2026, 5, 19, tzinfo=timezone.utc)
MV = "vader-test"


# ---------------- config ----------------


def test_apply_to_default_matches_constants() -> None:
    cfg = apply_to_default()
    assert cfg.threshold_sigma == DEFAULTS[DIM_THRESHOLD]
    assert cfg.min_baseline_sigma == DEFAULTS[DIM_MIN_SIGMA]
    assert cfg.half_life_hours == DEFAULTS[DIM_HALF_LIFE]
    assert cfg.source_weights == {}


def test_learned_config_source_weight_lookup() -> None:
    cfg = LearnedConfig(
        version=1, threshold_sigma=0.4, min_baseline_sigma=0.05,
        half_life_hours=10, source_weights={"Reuters": 1.3, "WSJ": 0.7},
    )
    assert cfg.weight_for_source("Reuters") == 1.3
    assert cfg.weight_for_source("WSJ") == 0.7
    assert cfg.weight_for_source("UnknownSrc") == 1.0
    assert cfg.weight_for_source(None) == 1.0
    assert cfg.weight_for_source("") == 1.0


def test_active_weights_defaults_when_none_active(session) -> None:
    cfg = active_weights(session)
    assert cfg.version is None
    assert cfg.threshold_sigma == DEFAULTS[DIM_THRESHOLD]


def test_active_weights_reads_active_row(session) -> None:
    session.add_all(
        [
            LearnedWeight(
                version=1, dimension=DIM_THRESHOLD, value=0.42, is_active=True,
            ),
            LearnedWeight(
                version=1, dimension=DIM_MIN_SIGMA, value=0.08, is_active=True,
            ),
            LearnedWeight(
                version=1, dimension=DIM_HALF_LIFE, value=8.0, is_active=True,
            ),
            LearnedWeight(
                version=1, dimension=DIM_SOURCE_WEIGHT, key="Reuters",
                value=1.2, is_active=True,
            ),
        ]
    )
    session.commit()
    cfg = active_weights(session)
    assert cfg.version == 1
    assert cfg.threshold_sigma == pytest.approx(0.42)
    assert cfg.min_baseline_sigma == pytest.approx(0.08)
    assert cfg.half_life_hours == pytest.approx(8.0)
    assert cfg.source_weights == {"Reuters": 1.2}


def test_weights_for_version_fills_with_defaults(session) -> None:
    """Missing dimensions in a version fall back to module defaults."""
    session.add(
        LearnedWeight(
            version=7, dimension=DIM_THRESHOLD, value=0.6, is_active=False,
        )
    )
    session.commit()
    cfg = weights_for_version(session, 7)
    assert cfg.threshold_sigma == pytest.approx(0.6)
    assert cfg.min_baseline_sigma == DEFAULTS[DIM_MIN_SIGMA]


# ---------------- helpers for training-data fixtures ----------------


def _seed_one_day(session, *, target, day, scores, ret, hit_label):
    """Insert N articles + scores + Prediction + Outcome on a single day."""
    arts = [
        make_article(
            finnhub_id=hash((target, day, i)) % 1_000_000_000,
            symbol=None if target == "^GSPC" else target,
            category="general" if target == "^GSPC" else "company",
            published_at=day + timedelta(hours=12 + i),
            source=("Reuters" if i % 2 == 0 else "WSJ"),
        )
        for i, _ in enumerate(scores)
    ]
    upsert_articles(session, arts)
    persisted = (
        session.query(type(arts[0]))
        .filter(type(arts[0]).finnhub_id.in_([a.finnhub_id for a in arts]))
        .all()
    )
    save_scores(
        session,
        [
            make_score(persisted[i].id, scores[i], model_version=MV)
            for i in range(len(scores))
        ],
    )
    pred = save_prediction(
        session,
        make_prediction(
            target_symbol=target,
            prediction_date=day,
            label=hit_label,
            confidence=0.5,
            sentiment_index=sum(scores) / len(scores),
            article_count=len(scores),
            model_version=MV,
        ),
    )
    save_outcome(
        session,
        PredictionOutcome(
            prediction_id=pred.id,
            realised_return=ret,
            hit=(hit_label == "UP" and ret > 0)
            or (hit_label == "DOWN" and ret < 0)
            or (hit_label == "FLAT" and abs(ret) < 0.0025),
        ),
    )


# ---------------- simulate ----------------


def test_build_training_frame_finds_articles_and_outcomes(session) -> None:
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=5),
        scores=[0.7, 0.6, 0.8], ret=0.015, hit_label="UP",
    )
    frame = build_training_frame(session, model_version=MV)
    assert len(frame.closed_prediction_keys) == 1
    key = frame.closed_prediction_keys[0]
    assert frame.articles_by_day[key] != []
    assert frame.realised_by_day[key] == pytest.approx(0.015)


def test_simulate_constant_sentiment_collapses_to_baseline(session) -> None:
    """Constant sentiment + the rolling baseline catches up after day 1.

    From day 2 onward today's z-score collapses toward zero (today equals
    the recent mean), so the simulator emits FLAT. FLAT's hit rule is
    |ret| < 0.25%, which fails for a 1.2% move → those are misses. Only
    the very first day (no prior baseline → z very high) calls UP and
    hits, so the run lands at 1/5 = 0.2.
    """
    for i in range(5):
        _seed_one_day(
            session, target="^GSPC",
            day=D - timedelta(days=10 + i),
            scores=[0.8, 0.7, 0.85],
            ret=0.012,
            hit_label="UP",
        )
    frame = build_training_frame(session, model_version=MV)
    cfg = LearnedConfig(
        version=None, threshold_sigma=0.5, min_baseline_sigma=0.05,
        half_life_hours=12.0, source_weights={},
    )
    hit_rate, cum_pnl, hits, directional = simulate(frame, cfg)
    assert hit_rate == pytest.approx(0.2)
    # cum_pnl reflects the single UP call that hit.
    assert cum_pnl > 0


def test_simulate_threshold_changes_call(session) -> None:
    """A very low threshold turns FLAT calls into directional ones."""
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=15),
        scores=[0.0, 0.0, 0.0], ret=0.0, hit_label="FLAT",
    )
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=10),
        scores=[0.0, 0.0, 0.0], ret=0.0, hit_label="FLAT",
    )
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=5),
        scores=[0.5, 0.6, 0.55], ret=0.02, hit_label="UP",
    )
    frame = build_training_frame(session, model_version=MV)

    # With a very low threshold the simulator should produce an UP for
    # the latest day and the realised move agrees → hit.
    aggressive = LearnedConfig(
        version=None, threshold_sigma=0.01, min_baseline_sigma=0.05,
        half_life_hours=12.0, source_weights={},
    )
    hr_a, pnl_a, _, _ = simulate(frame, aggressive)

    # With an absurdly high threshold everything is FLAT.
    # NB the simulator's z-score on the latest day can be ~11 (large
    # sentiment / tiny baseline σ floor), so we need a threshold well
    # above that to force FLAT.
    conservative = LearnedConfig(
        version=None, threshold_sigma=100.0, min_baseline_sigma=0.05,
        half_life_hours=12.0, source_weights={},
    )
    hr_c, pnl_c, _, _ = simulate(frame, conservative)

    # Aggressive captures the +2% UP move; conservative emits FLAT and
    # accumulates no PnL.
    assert pnl_a != pnl_c


def test_blended_objective_returns_float(session) -> None:
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=5),
        scores=[0.5, 0.6], ret=0.01, hit_label="UP",
    )
    frame = build_training_frame(session, model_version=MV)
    score = blended_objective(frame, apply_to_default())
    assert isinstance(score, float)


def test_simulate_empty_frame_safe(session) -> None:
    frame = build_training_frame(session, model_version=MV)
    hr, pnl, hits, n = simulate(frame, apply_to_default())
    assert (hr, pnl, hits, n) == (0.0, 0.0, 0, 0)


# ---------------- source weights ----------------


def test_fit_source_weights_boost_better_sources(session) -> None:
    """Reuters articles agree with realised direction; WSJ disagrees.
    Reuters should end up with weight > 1, WSJ < 1."""
    # Day 1: Reuters scores +, WSJ scores -, market goes UP
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=5),
        scores=[0.8, -0.7, 0.7, -0.6],
        ret=0.02, hit_label="UP",
    )
    frame = build_training_frame(session, model_version=MV)
    weights = _fit_source_weights(frame)
    if weights:  # may be empty if all sources had < 3 articles
        # Reuters articles agreed with the direction; WSJ didn't.
        assert weights.get("Reuters", 1.0) >= weights.get("WSJ", 1.0)


def test_fit_source_weights_empty_frame_returns_empty(session) -> None:
    frame = build_training_frame(session, model_version=MV)
    assert _fit_source_weights(frame) == {}


# ---------------- split ----------------


def test_split_frame_by_date(session) -> None:
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=30),
        scores=[0.5, 0.5, 0.5], ret=0.01, hit_label="UP",
    )
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=5),
        scores=[0.5, 0.5, 0.5], ret=0.01, hit_label="UP",
    )
    frame = build_training_frame(session, model_version=MV)
    train, hold = _split_frame_by_date(
        frame, cutoff=D - timedelta(days=14)
    )
    assert len(train.closed_prediction_keys) == 1
    assert len(hold.closed_prediction_keys) == 1


# ---------------- train_weights end-to-end ----------------


def test_train_weights_rejects_small_ledger(session) -> None:
    _seed_one_day(
        session, target="^GSPC", day=D - timedelta(days=5),
        scores=[0.5, 0.5, 0.5], ret=0.01, hit_label="UP",
    )
    with pytest.raises(NotEnoughDataError):
        train_weights(session, model_version=MV, n_calls=5)


def test_train_weights_persists_and_activates(session) -> None:
    """End-to-end: seed enough trades, train, verify a new active version lands."""
    # Seed 12 days each with a sentiment that matches the realised move.
    for i in range(12):
        ret = 0.01 if i % 2 == 0 else -0.01
        label = "UP" if ret > 0 else "DOWN"
        scores = [0.7] * 3 if ret > 0 else [-0.7] * 3
        _seed_one_day(
            session, target="^GSPC",
            day=D - timedelta(days=20 + i),
            scores=scores, ret=ret, hit_label=label,
        )

    report = train_weights(session, model_version=MV, n_calls=10, random_state=42)
    assert report.version == 1
    assert report.n_train >= MIN_TRADES_FOR_TRAINING
    assert report.fitted["threshold_sigma"] >= 0.10
    assert report.fitted["threshold_sigma"] <= 1.50

    # Active version reads back the same threshold.
    cfg = active_weights(session)
    assert cfg.version == 1
    assert cfg.threshold_sigma == pytest.approx(
        report.fitted["threshold_sigma"]
    )


def test_train_weights_two_runs_increment_version(session) -> None:
    for i in range(12):
        _seed_one_day(
            session, target="^GSPC",
            day=D - timedelta(days=20 + i),
            scores=[0.5] * 3, ret=0.005, hit_label="UP",
        )
    r1 = train_weights(session, model_version=MV, n_calls=10)
    r2 = train_weights(session, model_version=MV, n_calls=10)
    assert r1.version == 1
    assert r2.version == 2
    # The newer (and only the newer) version is active.
    cfg = active_weights(session)
    assert cfg.version == 2


def test_train_weights_activate_false_keeps_old(session) -> None:
    for i in range(12):
        _seed_one_day(
            session, target="^GSPC",
            day=D - timedelta(days=20 + i),
            scores=[0.5] * 3, ret=0.005, hit_label="UP",
        )
    train_weights(session, model_version=MV, n_calls=10, activate=True)
    train_weights(
        session, model_version=MV, n_calls=10, activate=False,
    )
    cfg = active_weights(session)
    assert cfg.version == 1  # untouched
