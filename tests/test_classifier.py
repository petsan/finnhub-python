"""Logistic-regression classification mode tests."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.classifier import (
    DEFAULT_DECISION_BAND,
    LogisticCalibration,
    NotEnoughCalibrationDataError,
    apply_logreg_classification,
    fit_logreg_calibration,
    load_calibration,
    resolve_classifier_mode,
    save_calibration,
)
from finn_predictor.predictor.market import predict_market
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    PredictionOutcome,
    SentimentScore,
)
from finn_predictor.storage.repo import save_outcome, save_prediction


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


# --- Pure-math sanity tests --------------------------------------------


def test_logistic_calibration_probability_is_sigmoid() -> None:
    cal = LogisticCalibration(beta=2.0, intercept=0.0)
    # At sentiment=0 → probability should sit at exactly 0.5.
    assert cal.probability_up(0.0) == pytest.approx(0.5)
    # At sentiment=1 → P = sigmoid(2) ≈ 0.8808.
    assert cal.probability_up(1.0) == pytest.approx(1 / (1 + math.exp(-2.0)))


def test_apply_logreg_classification_emits_up_above_band() -> None:
    cal = LogisticCalibration(beta=5.0, intercept=0.0)  # steep, decisive
    label, conf = apply_logreg_classification(0.5, cal)
    assert label == "UP"
    assert conf > 0.5


def test_apply_logreg_classification_emits_down_below_band() -> None:
    cal = LogisticCalibration(beta=5.0, intercept=0.0)
    label, conf = apply_logreg_classification(-0.5, cal)
    assert label == "DOWN"
    assert conf > 0.5


def test_apply_logreg_classification_emits_flat_inside_band() -> None:
    """Probability just above 0.5 + band stays in the FLAT zone."""
    cal = LogisticCalibration(beta=0.01, intercept=0.0)  # nearly flat
    label, _conf = apply_logreg_classification(0.1, cal)
    assert label == "FLAT"


def test_apply_logreg_classification_confidence_is_calibrated() -> None:
    """Confidence = |2P − 1| not normalised-z."""
    cal = LogisticCalibration(beta=1.0, intercept=0.0)
    _label, conf = apply_logreg_classification(1.0, cal)
    p = 1.0 / (1.0 + math.exp(-1.0))
    assert conf == pytest.approx(abs(2 * p - 1))


# --- Fitting -----------------------------------------------------------


def _seed_calibration_dataset(
    session, *, target_symbol: str = "^GSPC", model_version: str = "vader-test"
) -> None:
    """Drop 20 closed predictions where positive sentiment → up outcomes.

    The relationship is monotonic but noisy; Newton-Raphson should
    recover a positive beta.
    """
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    # 10 positive-sentiment predictions, 9 went up
    # 10 negative-sentiment predictions, 9 went down
    series = (
        [(0.5, +0.01)] * 9 + [(0.4, -0.005)]  # bullish but one missed
        + [(-0.5, -0.012)] * 9 + [(-0.45, 0.004)]  # bearish but one missed
    )
    for i, (sentiment, realised) in enumerate(series):
        pred = save_prediction(
            session,
            Prediction(
                target_symbol=target_symbol,
                prediction_date=base + timedelta(days=i),
                label="UP" if sentiment > 0 else "DOWN",
                confidence=0.5,
                sentiment_index=sentiment,
                article_count=5,
                model_version=model_version,
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(
                prediction_id=pred.id,
                realised_return=realised,
                hit=(sentiment > 0) == (realised > 0),
            ),
        )


def test_fit_logreg_calibration_recovers_positive_beta(session) -> None:
    """Positive sentiment correlates with positive realised return → beta > 0."""
    _seed_calibration_dataset(session)
    cal = fit_logreg_calibration(session, model_version="vader-test")
    assert cal.beta > 0
    assert cal.n_samples == 20
    # And confidence at sentiment +0.5 should now be > the band.
    label, _ = apply_logreg_classification(0.5, cal)
    assert label == "UP"


def test_fit_logreg_calibration_filters_to_model_version(session) -> None:
    """Predictions from another model_version don't pollute the fit."""
    _seed_calibration_dataset(session, model_version="vader-test")
    # Add 5 predictions under a different model_version that all went DOWN
    # on positive sentiment — if we accidentally include them, beta flips
    # in sign.
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for i in range(5):
        pred = save_prediction(
            session,
            Prediction(
                target_symbol="^GSPC",
                prediction_date=base + timedelta(days=i),
                label="UP",
                confidence=0.5,
                sentiment_index=0.6,
                article_count=5,
                model_version="finbert-test",
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(prediction_id=pred.id, realised_return=-0.01, hit=False),
        )

    cal = fit_logreg_calibration(session, model_version="vader-test")
    assert cal.beta > 0
    assert cal.n_samples == 20


def test_fit_logreg_calibration_raises_on_small_dataset(session) -> None:
    """Below MIN_FIT_SAMPLES closed predictions → NotEnoughCalibrationDataError."""
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i in range(3):
        pred = save_prediction(
            session,
            Prediction(
                target_symbol="^GSPC",
                prediction_date=base + timedelta(days=i),
                label="UP",
                confidence=0.5,
                sentiment_index=0.5,
                article_count=5,
                model_version="vader-test",
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(prediction_id=pred.id, realised_return=0.01, hit=True),
        )

    with pytest.raises(NotEnoughCalibrationDataError):
        fit_logreg_calibration(session, model_version="vader-test")


def test_fit_logreg_calibration_raises_on_one_class(session) -> None:
    """20 closed predictions all up → no decision boundary to fit."""
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i in range(20):
        pred = save_prediction(
            session,
            Prediction(
                target_symbol="^GSPC",
                prediction_date=base + timedelta(days=i),
                label="UP",
                confidence=0.5,
                sentiment_index=0.4 + 0.01 * i,
                article_count=5,
                model_version="vader-test",
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(prediction_id=pred.id, realised_return=0.01, hit=True),
        )

    with pytest.raises(NotEnoughCalibrationDataError, match="single class"):
        fit_logreg_calibration(session, model_version="vader-test")


# --- Persistence -------------------------------------------------------


def test_save_then_load_calibration_roundtrips(session) -> None:
    cal = LogisticCalibration(beta=1.7, intercept=-0.2, n_samples=42)
    save_calibration(session, cal)
    loaded = load_calibration(session)
    assert loaded is not None
    assert loaded.beta == pytest.approx(1.7)
    assert loaded.intercept == pytest.approx(-0.2)
    assert loaded.n_samples == 42


def test_load_calibration_returns_none_when_missing(session) -> None:
    assert load_calibration(session) is None


def test_load_calibration_returns_none_for_malformed_blob(session) -> None:
    from finn_predictor.storage.repo import set_setting
    set_setting(session, "logreg_calibration", "not-json")
    assert load_calibration(session) is None


# --- Predictor integration --------------------------------------------


def test_predict_market_with_calibration_overrides_rule(session) -> None:
    """A high-beta calibration flips a marginal call from FLAT to UP.

    Without calibration, three articles at sentiment ~0.4 don't cross
    the 0.5σ z-threshold against an empty baseline → rule emits UP at
    capped confidence. With a calibration that strongly favours UP at
    positive sentiment, the call becomes UP with calibrated confidence
    that matches |2P − 1|.
    """
    # Seed 3 positive articles today, scored.
    arts = []
    for i in range(3):
        a = NewsArticle(
            finnhub_id=1000 + i,
            category="general",
            headline=f"good news {i}",
            published_at=D,
            source="Reuters",
        )
        session.add(a)
        arts.append(a)
    session.commit()
    for a in arts:
        session.add(
            SentimentScore(article_id=a.id, score=0.4, model_version="vader-test")
        )
    session.commit()

    class _Scorer:
        model_version = "vader-test"

        def score(self, text):
            return 0.0

        def score_many(self, texts):
            return [0.0 for _ in texts]

    cal = LogisticCalibration(beta=5.0, intercept=0.0, n_samples=50)
    pred = predict_market(
        session, scorer=_Scorer(), on_date=D, symbol="^GSPC",
        calibration=cal,
    )
    assert pred is not None
    assert pred.label == "UP"
    # |2P-1| where P = sigmoid(5*0.4) = sigmoid(2) ≈ 0.881 → conf ≈ 0.762
    assert pred.confidence == pytest.approx(
        abs(2 * (1 / (1 + math.exp(-2.0))) - 1), abs=1e-3
    )


# --- Env-driven mode selection ----------------------------------------


def test_resolve_classifier_mode_defaults_to_rule(monkeypatch) -> None:
    monkeypatch.delenv("FINN_PREDICTOR_CLASSIFIER", raising=False)
    assert resolve_classifier_mode() == "rule"


def test_resolve_classifier_mode_honours_logreg(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_CLASSIFIER", "LogReg")
    assert resolve_classifier_mode() == "logreg"


def test_resolve_classifier_mode_falls_back_on_unknown(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_CLASSIFIER", "xgb")
    assert resolve_classifier_mode() == "rule"


def test_default_decision_band_is_small_but_nonzero() -> None:
    assert 0 < DEFAULT_DECISION_BAND < 0.5
