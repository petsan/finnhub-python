"""Quantile-band magnitude prediction.

The default classifier (rule or logreg) outputs direction + confidence
— never a return number. This module adds an opt-in *magnitude band*:
``p10 / p50 / p90`` quantiles of next-bar return conditional on
today's ``sentiment_index``. Fit on closed outcomes via pinball-loss
minimisation, persisted as a single JSON blob in ``app_settings``,
applied at predict time only when ``FINN_PREDICTOR_MAGNITUDE=quantile``
AND a calibration exists.

Why quantiles instead of a point estimate? Sentiment alone explains a
single-digit fraction of next-day return variance, so a single number
would imply precision the data can't support. A ``p10–p90`` band tells
the truth about uncertainty (the band is wide), which is the only
useful magnitude output a sentiment-only model can honestly produce.

Why a separate module from :mod:`classifier`? Same DB tables, same
training-set query, but a different objective (quantile pinball loss
vs Bernoulli logit). Keeping them apart means a deploy can run
logistic-regression direction and quantile-band magnitude
independently; either, both, or neither.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import Prediction, PredictionOutcome


# AppSetting row that stores the active magnitude calibration. One row
# per deploy, refreshed by ``save_calibration``; predictors load it at
# decision time when the env opts in.
SETTING_KEY = "magnitude_calibration"

# Quantiles we fit. Fixed at three: p10 / p50 / p90 covers the honest
# story (10th-and-90th-percentile range + median). Bumping this to
# more quantiles would require a calibration-version bump because the
# stored JSON shape would change.
DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Pinball-loss subgradient-descent controls. Convergence is loose
# enough that 2000 iterations is comfortable for our scale
# (hundreds-to-low-thousands of training points, 1 feature + intercept).
MAX_DESCENT_ITERS = 2000
LEARNING_RATE = 0.02
GRADIENT_TOL = 1e-6

# Minimum closed-outcome rows before we'll fit. Higher than the
# classifier's 10 because we're fitting three regressions and the
# tail quantiles need samples in the tails to be meaningful.
MIN_FIT_SAMPLES = 15


class NotEnoughMagnitudeDataError(RuntimeError):
    """Raised when there aren't enough closed outcomes to fit the band."""


@dataclass(frozen=True)
class QuantileFit:
    """Linear fit for one quantile: ``y_tau = intercept + beta * x``."""

    tau: float
    intercept: float
    beta: float


@dataclass(frozen=True)
class MagnitudeForecast:
    """Per-prediction band: realised-return quantiles."""

    p10: float
    p50: float
    p90: float

    def __post_init__(self) -> None:
        # Defensive: quantile fits can cross on small / noisy data even
        # though the population quantiles can't. Sorting protects the
        # downstream UI from showing a band whose lower edge sits above
        # the upper edge, which would read as a bug to a user.
        ordered = sorted((self.p10, self.p50, self.p90))
        if (self.p10, self.p50, self.p90) != tuple(ordered):
            object.__setattr__(self, "p10", ordered[0])
            object.__setattr__(self, "p50", ordered[1])
            object.__setattr__(self, "p90", ordered[2])


@dataclass(frozen=True)
class MagnitudeCalibration:
    """Persisted ``MagnitudeForecast`` model.

    Holds one :class:`QuantileFit` per quantile in :data:`DEFAULT_QUANTILES`.
    ``feature_name`` is recorded so future versions of this module can
    diff against the stored calibration and refuse mismatches.
    """

    fits: tuple[QuantileFit, ...]
    n_samples: int = 0
    feature_name: str = "sentiment_index"

    def quantiles(self) -> tuple[float, ...]:
        return tuple(f.tau for f in self.fits)

    def predict(self, sentiment_index: float) -> MagnitudeForecast:
        """Apply the three fits to a single ``sentiment_index``."""
        x = float(sentiment_index)
        by_tau = {f.tau: f.intercept + f.beta * x for f in self.fits}
        return MagnitudeForecast(
            p10=by_tau.get(0.10, 0.0),
            p50=by_tau.get(0.50, 0.0),
            p90=by_tau.get(0.90, 0.0),
        )


# --- Pinball-loss quantile regression -----------------------------------


def _pinball_loss(residual: float, tau: float) -> float:
    """Quantile (pinball) loss for one residual."""
    return tau * residual if residual >= 0 else (tau - 1.0) * residual


def _fit_quantile_regression_1d(
    xs: list[float],
    ys: list[float],
    tau: float,
    *,
    n_iters: int = MAX_DESCENT_ITERS,
    lr: float = LEARNING_RATE,
) -> QuantileFit:
    """Pinball-loss MLE for a 1-feature linear quantile regression.

    Subgradient descent on ``intercept`` and ``beta``. The pinball
    loss is convex but non-differentiable at zero residual, so we use
    the subgradient ``-tau if residual > 0 else (1 - tau)``. Tracks
    the best objective seen and returns those parameters — pure SGD
    on a non-smooth objective oscillates around the minimum.
    """
    n = len(xs)
    if n == 0:
        return QuantileFit(tau=tau, intercept=0.0, beta=0.0)

    # Warm start: median(ys) as the intercept gets us close for tau=0.5,
    # and somewhere reasonable for the other quantiles too.
    sorted_ys = sorted(ys)
    intercept = sorted_ys[n // 2]
    beta = 0.0

    best_loss = float("inf")
    best_intercept = intercept
    best_beta = beta

    prev_loss = float("inf")
    for _ in range(n_iters):
        grad_a = 0.0
        grad_b = 0.0
        total_loss = 0.0
        for x, y in zip(xs, ys):
            residual = y - (intercept + beta * x)
            # Subgradient: derivative of pinball loss w.r.t. residual.
            if residual > 0:
                sub = -tau
            elif residual < 0:
                sub = 1.0 - tau
            else:
                sub = 0.0
            # Loss-vs-params chain: dL/da = sub * dResidual/da = -sub
            # (because residual = y - a - b*x), but we want the descent
            # direction so we minimise — flip the sign.
            grad_a += sub
            grad_b += sub * x
            total_loss += _pinball_loss(residual, tau)

        grad_a /= n
        grad_b /= n
        total_loss /= n

        if total_loss < best_loss:
            best_loss = total_loss
            best_intercept = intercept
            best_beta = beta

        if abs(prev_loss - total_loss) < GRADIENT_TOL and prev_loss != float("inf"):
            break
        prev_loss = total_loss

        intercept -= lr * grad_a
        beta -= lr * grad_b

    return QuantileFit(tau=tau, intercept=best_intercept, beta=best_beta)


# --- DB-driven fitting --------------------------------------------------


def _closed_returns_with_sentiment(
    session: Session,
    *,
    model_version: str,
    target_symbol: Optional[str] = None,
) -> list[tuple[float, float]]:
    """Return ``[(sentiment_index, realised_return), ...]`` for closed predictions.

    Unlike the classifier's training set, we keep FLAT calls — magnitude
    is meaningful regardless of direction (a FLAT call that realised a
    big move is informative about magnitude even though the direction
    bucketed it differently).
    """
    stmt = (
        select(Prediction.sentiment_index, PredictionOutcome.realised_return)
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(Prediction.model_version == model_version)
    )
    if target_symbol is not None:
        stmt = stmt.where(Prediction.target_symbol == target_symbol)

    out: list[tuple[float, float]] = []
    for sentiment, realised in session.execute(stmt):
        out.append((float(sentiment), float(realised)))
    return out


def fit_quantile_calibration(
    session: Session,
    *,
    model_version: str,
    target_symbol: Optional[str] = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> MagnitudeCalibration:
    """Fit one :class:`QuantileFit` per quantile from closed outcomes.

    Raises :class:`NotEnoughMagnitudeDataError` when fewer than
    :data:`MIN_FIT_SAMPLES` closed predictions are available, or when
    the realised-return column has zero variance (degenerate dataset
    that yields a flat band).
    """
    pairs = _closed_returns_with_sentiment(
        session, model_version=model_version, target_symbol=target_symbol
    )
    n = len(pairs)
    if n < MIN_FIT_SAMPLES:
        raise NotEnoughMagnitudeDataError(
            f"need at least {MIN_FIT_SAMPLES} closed predictions to fit "
            f"a magnitude band, have {n}"
        )

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]

    # A constant target gives a degenerate fit (all quantiles collapse
    # to the same intercept) which would mislead the UI into reporting
    # a zero-width band as if it were a real prediction.
    if max(ys) - min(ys) < 1e-9:
        raise NotEnoughMagnitudeDataError(
            "realised_return has zero variance — no magnitude signal to fit"
        )

    fits = tuple(
        _fit_quantile_regression_1d(xs, ys, tau) for tau in quantiles
    )
    return MagnitudeCalibration(fits=fits, n_samples=n)


# --- Persistence --------------------------------------------------------


def save_calibration(session: Session, calibration: MagnitudeCalibration) -> None:
    """Persist ``calibration`` as the active magnitude calibration.

    Stored in the same single-row ``app_settings`` pattern as the
    logreg classifier calibration (different key). The predictors look
    this up once per call when the env opts in.
    """
    from finn_predictor.storage.repo import set_setting

    payload = json.dumps(
        {
            "fits": [
                {"tau": f.tau, "intercept": f.intercept, "beta": f.beta}
                for f in calibration.fits
            ],
            "n_samples": int(calibration.n_samples),
            "feature_name": calibration.feature_name,
        }
    )
    set_setting(session, SETTING_KEY, payload)


def load_calibration(session: Session) -> Optional[MagnitudeCalibration]:
    """Return the saved calibration, or ``None`` if none has been fitted."""
    from finn_predictor.storage.repo import get_setting

    raw = get_setting(session, SETTING_KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    try:
        fits = tuple(
            QuantileFit(
                tau=float(f["tau"]),
                intercept=float(f["intercept"]),
                beta=float(f["beta"]),
            )
            for f in data["fits"]
        )
        return MagnitudeCalibration(
            fits=fits,
            n_samples=int(data.get("n_samples", 0)),
            feature_name=str(data.get("feature_name", "sentiment_index")),
        )
    except (KeyError, TypeError, ValueError):
        return None


# --- Env-driven mode selection -----------------------------------------


def resolve_magnitude_mode() -> str:
    """Return ``'off'`` (default) or ``'quantile'`` from the env.

    Anything unrecognised falls back to ``off`` — quiet fallbacks for
    every model toggle in this project, so a typo at deploy time
    never blanks the UI or silently activates an experimental mode.
    """
    raw = os.environ.get("FINN_PREDICTOR_MAGNITUDE", "off")
    name = (raw or "off").strip().lower() or "off"
    if name in {"off", "quantile"}:
        return name
    return "off"
