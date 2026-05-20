"""Quantile-band magnitude prediction tests."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.magnitude import (
    DEFAULT_QUANTILES,
    MagnitudeCalibration,
    MagnitudeForecast,
    NotEnoughMagnitudeDataError,
    QuantileFit,
    _fit_quantile_regression_1d,
    _pinball_loss,
    fit_quantile_calibration,
    load_calibration,
    resolve_magnitude_mode,
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


# --- Pure-math sanity ---------------------------------------------------


def test_pinball_loss_asymmetric() -> None:
    """Pinball loss penalises the wrong tail more for tail quantiles."""
    # At tau=0.9, under-predicting by 1 (residual=+1) should hurt more
    # than over-predicting by 1 (residual=-1).
    assert _pinball_loss(+1.0, 0.9) > _pinball_loss(-1.0, 0.9)
    # At tau=0.1, the asymmetry flips.
    assert _pinball_loss(+1.0, 0.1) < _pinball_loss(-1.0, 0.1)
    # At tau=0.5, the loss is symmetric.
    assert _pinball_loss(+1.0, 0.5) == pytest.approx(_pinball_loss(-1.0, 0.5))


def test_pinball_loss_zero_residual() -> None:
    assert _pinball_loss(0.0, 0.1) == 0.0
    assert _pinball_loss(0.0, 0.9) == 0.0


def test_quantile_fit_recovers_known_intercept() -> None:
    """y = 0.5 + noise → tau=0.5 fit intercept ≈ 0.5; tail fits bracket it."""
    random.seed(42)
    xs = [random.uniform(-1.0, 1.0) for _ in range(400)]
    ys = [0.5 + random.gauss(0.0, 0.3) for _ in xs]  # constant signal + symmetric noise

    p10 = _fit_quantile_regression_1d(xs, ys, 0.10)
    p50 = _fit_quantile_regression_1d(xs, ys, 0.50)
    p90 = _fit_quantile_regression_1d(xs, ys, 0.90)

    assert p50.intercept == pytest.approx(0.5, abs=0.05)
    assert p10.intercept < p50.intercept < p90.intercept


def test_quantile_fit_recovers_known_slope() -> None:
    """Linear-in-x signal: betas of all quantile fits should match the slope."""
    random.seed(7)
    xs = [random.uniform(-1.0, 1.0) for _ in range(500)]
    ys = [0.0 + 1.2 * x + random.gauss(0.0, 0.2) for x in xs]

    fits = [_fit_quantile_regression_1d(xs, ys, t) for t in (0.1, 0.5, 0.9)]
    for f in fits:
        # Loose tolerance: subgradient descent + non-smooth loss isn't
        # going to get four decimal places, but a few is enough to
        # confirm the recovery.
        assert f.beta == pytest.approx(1.2, abs=0.15)


# --- MagnitudeForecast invariants --------------------------------------


def test_magnitude_forecast_sorts_crossing_quantiles() -> None:
    """Construction with a crossed band silently reorders to the valid one."""
    f = MagnitudeForecast(p10=0.5, p50=0.0, p90=-0.5)
    assert f.p10 == -0.5
    assert f.p50 == 0.0
    assert f.p90 == 0.5


def test_magnitude_forecast_preserves_ordered_band() -> None:
    f = MagnitudeForecast(p10=-0.02, p50=0.001, p90=0.025)
    assert (f.p10, f.p50, f.p90) == (-0.02, 0.001, 0.025)


# --- Calibration apply --------------------------------------------------


def test_calibration_predict_threads_all_three_fits() -> None:
    cal = MagnitudeCalibration(
        fits=(
            QuantileFit(tau=0.10, intercept=-0.01, beta=0.05),
            QuantileFit(tau=0.50, intercept=0.001, beta=0.10),
            QuantileFit(tau=0.90, intercept=0.012, beta=0.15),
        ),
        n_samples=42,
    )
    out = cal.predict(0.5)
    assert out.p10 == pytest.approx(-0.01 + 0.5 * 0.05)
    assert out.p50 == pytest.approx(0.001 + 0.5 * 0.10)
    assert out.p90 == pytest.approx(0.012 + 0.5 * 0.15)


# --- DB-driven fit ------------------------------------------------------


def _seed_outcomes(session, n: int, *, mv: str = "vader-test") -> None:
    """Drop ``n`` closed predictions with sentiment correlated to return."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    random.seed(123)
    for i in range(n):
        sentiment = random.uniform(-0.8, 0.8)
        realised = 0.01 * sentiment + random.gauss(0.0, 0.015)
        p = save_prediction(
            session,
            Prediction(
                target_symbol="^GSPC",
                prediction_date=base + timedelta(days=i),
                label="UP" if sentiment > 0 else "DOWN",
                confidence=0.5,
                sentiment_index=sentiment,
                article_count=5,
                model_version=mv,
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(
                prediction_id=p.id,
                realised_return=realised,
                hit=(sentiment > 0) == (realised > 0),
            ),
        )


def test_fit_quantile_calibration_recovers_ordered_band(session) -> None:
    _seed_outcomes(session, 100)
    cal = fit_quantile_calibration(session, model_version="vader-test")
    assert cal.n_samples == 100
    assert cal.quantiles() == DEFAULT_QUANTILES
    # The fits at a representative sentiment should be ordered.
    f = cal.predict(0.5)
    assert f.p10 <= f.p50 <= f.p90
    # Median should track the underlying signal — sentiment +0.5 implies
    # a small positive realised return on the synthetic data.
    assert f.p50 > 0


def test_fit_quantile_calibration_raises_on_small_dataset(session) -> None:
    _seed_outcomes(session, 5)
    with pytest.raises(NotEnoughMagnitudeDataError):
        fit_quantile_calibration(session, model_version="vader-test")


def test_fit_quantile_calibration_raises_on_zero_variance(session) -> None:
    """Realised returns all the same → degenerate fit → refuse."""
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(20):
        p = save_prediction(
            session,
            Prediction(
                target_symbol="^GSPC",
                prediction_date=base + timedelta(days=i),
                label="UP",
                confidence=0.5,
                sentiment_index=0.2,
                article_count=5,
                model_version="vader-test",
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(prediction_id=p.id, realised_return=0.005, hit=True),
        )
    with pytest.raises(NotEnoughMagnitudeDataError, match="zero variance"):
        fit_quantile_calibration(session, model_version="vader-test")


def test_fit_quantile_calibration_filters_by_model_version(session) -> None:
    """Predictions under a different model_version don't pollute the fit."""
    _seed_outcomes(session, 100, mv="vader-test")
    # Add 30 FinBERT predictions with intentionally extreme returns —
    # if the fit accidentally included them, the band widens dramatically.
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for i in range(30):
        p = save_prediction(
            session,
            Prediction(
                target_symbol="^GSPC",
                prediction_date=base + timedelta(days=i),
                label="UP",
                confidence=0.5,
                sentiment_index=0.2,
                article_count=5,
                model_version="finbert-test",
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(prediction_id=p.id, realised_return=0.5, hit=True),
        )

    cal = fit_quantile_calibration(session, model_version="vader-test")
    assert cal.n_samples == 100
    # The p90 at sentiment +0.5 should still be a small move, not 0.5.
    assert cal.predict(0.5).p90 < 0.1


# --- Persistence --------------------------------------------------------


def test_save_then_load_calibration_roundtrips(session) -> None:
    original = MagnitudeCalibration(
        fits=(
            QuantileFit(tau=0.10, intercept=-0.012, beta=0.04),
            QuantileFit(tau=0.50, intercept=0.001, beta=0.07),
            QuantileFit(tau=0.90, intercept=0.015, beta=0.12),
        ),
        n_samples=88,
    )
    save_calibration(session, original)
    loaded = load_calibration(session)
    assert loaded is not None
    assert loaded.n_samples == 88
    for orig_f, load_f in zip(original.fits, loaded.fits):
        assert orig_f.tau == load_f.tau
        assert orig_f.intercept == pytest.approx(load_f.intercept)
        assert orig_f.beta == pytest.approx(load_f.beta)


def test_load_calibration_returns_none_when_missing(session) -> None:
    assert load_calibration(session) is None


def test_load_calibration_returns_none_for_malformed_blob(session) -> None:
    from finn_predictor.storage.repo import set_setting
    set_setting(session, "magnitude_calibration", "{not json")
    assert load_calibration(session) is None


# --- Env-driven resolver -----------------------------------------------


def test_resolve_magnitude_mode_defaults_to_off(monkeypatch) -> None:
    monkeypatch.delenv("FINN_PREDICTOR_MAGNITUDE", raising=False)
    assert resolve_magnitude_mode() == "off"


def test_resolve_magnitude_mode_honours_quantile(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_MAGNITUDE", "Quantile")
    assert resolve_magnitude_mode() == "quantile"


def test_resolve_magnitude_mode_falls_back_on_unknown(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_MAGNITUDE", "garch")
    assert resolve_magnitude_mode() == "off"


# --- Predictor integration ---------------------------------------------


def test_predict_market_writes_band_when_calibration_passed(session) -> None:
    """The three magnitude columns get populated when calibration is supplied."""
    # Seed today's articles + scores so the predictor has data to chew on.
    arts = []
    for i in range(5):
        a = NewsArticle(
            finnhub_id=2000 + i,
            category="general",
            headline=f"market headline {i}",
            published_at=D,
            source="Reuters",
        )
        session.add(a)
        arts.append(a)
    session.commit()
    for a in arts:
        session.add(
            SentimentScore(article_id=a.id, score=0.3, model_version="vader-test")
        )
    session.commit()

    class _Scorer:
        model_version = "vader-test"
        def score(self, t):  # noqa: ARG002
            return 0.0
        def score_many(self, ts):  # noqa: ARG002
            return [0.0 for _ in ts]

    cal = MagnitudeCalibration(
        fits=(
            QuantileFit(tau=0.10, intercept=-0.02, beta=0.0),
            QuantileFit(tau=0.50, intercept=0.005, beta=0.01),
            QuantileFit(tau=0.90, intercept=0.025, beta=0.0),
        ),
        n_samples=50,
    )
    pred = predict_market(
        session, scorer=_Scorer(), on_date=D, symbol="^GSPC",
        magnitude_calibration=cal,
    )
    assert pred is not None
    assert pred.expected_return_p10 is not None
    assert pred.expected_return_p50 is not None
    assert pred.expected_return_p90 is not None
    assert pred.expected_return_p10 <= pred.expected_return_p50 <= pred.expected_return_p90


def test_predict_market_leaves_band_null_without_calibration(session) -> None:
    """Default path (no magnitude_calibration arg) doesn't touch the columns."""
    arts = []
    for i in range(3):
        a = NewsArticle(
            finnhub_id=3000 + i,
            category="general",
            headline=f"market headline {i}",
            published_at=D,
            source="Reuters",
        )
        session.add(a)
        arts.append(a)
    session.commit()
    for a in arts:
        session.add(
            SentimentScore(article_id=a.id, score=0.3, model_version="vader-test")
        )
    session.commit()

    class _Scorer:
        model_version = "vader-test"
        def score(self, t):  # noqa: ARG002
            return 0.0
        def score_many(self, ts):  # noqa: ARG002
            return [0.0 for _ in ts]

    pred = predict_market(session, scorer=_Scorer(), on_date=D, symbol="^GSPC")
    assert pred is not None
    assert pred.expected_return_p10 is None
    assert pred.expected_return_p50 is None
    assert pred.expected_return_p90 is None


def test_save_prediction_preserves_band_on_upsert_without_band(session) -> None:
    """A re-run without magnitude data must NOT erase a band stored earlier."""
    cal = MagnitudeCalibration(
        fits=(
            QuantileFit(tau=0.10, intercept=-0.02, beta=0.0),
            QuantileFit(tau=0.50, intercept=0.005, beta=0.0),
            QuantileFit(tau=0.90, intercept=0.025, beta=0.0),
        ),
        n_samples=50,
    )
    # First run: with calibration
    arts = []
    for i in range(4):
        a = NewsArticle(
            finnhub_id=4000 + i,
            category="general",
            headline=f"upsert headline {i}",
            published_at=D,
            source="Reuters",
        )
        session.add(a)
        arts.append(a)
    session.commit()
    for a in arts:
        session.add(
            SentimentScore(article_id=a.id, score=0.25, model_version="vader-test")
        )
    session.commit()

    class _Scorer:
        model_version = "vader-test"
        def score(self, t):  # noqa: ARG002
            return 0.0
        def score_many(self, ts):  # noqa: ARG002
            return [0.0 for _ in ts]

    first = predict_market(
        session, scorer=_Scorer(), on_date=D, symbol="^GSPC",
        magnitude_calibration=cal,
    )
    assert first is not None and first.expected_return_p50 is not None
    band_before = (first.expected_return_p10, first.expected_return_p50, first.expected_return_p90)

    # Second run on the same day, no calibration. The upsert must NOT
    # null the band that was just written.
    second = predict_market(session, scorer=_Scorer(), on_date=D, symbol="^GSPC")
    assert second.id == first.id
    band_after = (second.expected_return_p10, second.expected_return_p50, second.expected_return_p90)
    assert band_after == band_before


def test_default_quantiles_are_p10_p50_p90() -> None:
    assert DEFAULT_QUANTILES == (0.10, 0.50, 0.90)
