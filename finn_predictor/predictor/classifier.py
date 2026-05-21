"""Logistic-regression classification mode.

Default predictor mode is rule-based: ``classify(z)`` in
:mod:`finn_predictor.predictor.market` thresholds a z-score directly. The
shipped confidence is just normalised z-distance, which is fine as a
relative ranking but isn't calibrated — a confidence of 0.8 doesn't mean
"80% chance the call is right."

This module fits a tiny 1-feature logistic regression
``P(up_realised | sentiment_index)`` on closed predictions
(predictions whose :class:`PredictionOutcome` rows exist) and exposes:

* :class:`LogisticCalibration` — the persisted ``(beta, intercept)`` pair.
* :func:`fit_logreg_calibration` — Newton-Raphson fit from the DB.
* :func:`apply_logreg_classification` — at-predict-time decision rule
  that returns ``(label, confidence)`` with confidence ∈ [0, 1] equal to
  ``|2P − 1|``. The label is ``UP``/``DOWN``/``FLAT`` using a small
  decision band around 0.5 so the FLAT zone doesn't collapse.

The whole thing is gated behind ``FINN_PREDICTOR_CLASSIFIER=logreg`` so
the default deploy keeps the rule classifier (which is what every
existing test asserts against).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import Prediction, PredictionOutcome


# AppSetting row that stores the currently-active calibration. One row
# per deploy, refreshed by ``save_calibration``; the predictors load it
# at decision time.
SETTING_KEY = "logreg_calibration"


# Tunables ---------------------------------------------------------------

# Default decision band around 0.5 — outside this band we emit UP/DOWN,
# inside we emit FLAT. The band keeps the model from labelling marginal
# probabilities; matches the FLAT slot the rule classifier exposes.
DEFAULT_DECISION_BAND = 0.05

# Newton-Raphson controls. The objective is convex so a handful of
# iterations converges to within float precision for any reasonable
# starting point; we cap to keep pathological inputs bounded.
MAX_NEWTON_ITERS = 50
NEWTON_TOL = 1e-7

# Below this many closed predictions we refuse to fit: a one-or-two-
# point fit either blows up or collapses to the prior, neither of
# which is useful.
MIN_FIT_SAMPLES = 10


class NotEnoughCalibrationDataError(RuntimeError):
    """Raised when the DB doesn't have enough closed outcomes to fit."""


@dataclass(frozen=True)
class LogisticCalibration:
    """A fitted ``P(up) = sigmoid(intercept + beta * sentiment_index)`` model."""

    beta: float
    intercept: float
    n_samples: int = 0

    def probability_up(self, sentiment_index: float) -> float:
        """Return ``P(up_realised | sentiment_index)`` ∈ (0, 1)."""
        logit = self.intercept + self.beta * float(sentiment_index)
        return _sigmoid(logit)


# --- Math ---------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid (avoids overflow at extreme logits)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_logreg_newton(
    xs: list[float], ys: list[int]
) -> tuple[float, float]:
    """Newton-Raphson MLE for the two-parameter logistic regression.

    ``xs`` is the feature vector, ``ys`` is 0/1 labels. Returns
    ``(beta, intercept)``. The two-by-two Hessian inverse is computed
    in-line so we don't drag in numpy just for a 1-feature model.
    """
    beta = 0.0
    intercept = 0.0
    n = len(xs)

    for _ in range(MAX_NEWTON_ITERS):
        # Compute residuals and Hessian components.
        # Gradient = sum( (p - y) * [1, x] )
        # Hessian  = sum( p*(1-p) * [[1, x], [x, x^2]] )
        g0 = 0.0  # ∂L/∂intercept
        g1 = 0.0  # ∂L/∂beta
        h00 = 0.0
        h01 = 0.0
        h11 = 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(intercept + beta * x)
            diff = p - y
            g0 += diff
            g1 += diff * x
            w = p * (1.0 - p)
            h00 += w
            h01 += w * x
            h11 += w * x * x

        # Regularise the Hessian very lightly — without this a perfectly
        # separable training set sends |beta| → ∞ and we never converge.
        # Equivalent to a Gaussian prior with variance 1/eps on each
        # parameter; keeps fits stable on small or clean datasets.
        eps = 1e-3
        h00 += eps
        h11 += eps
        # Add the L2 gradient as well so the optimum matches the
        # regularised Hessian we just built.
        g0 += eps * intercept
        g1 += eps * beta

        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break  # singular Hessian — bail with whatever we have.

        # Inverse-Hessian times gradient (2x2 closed form).
        inv00 = h11 / det
        inv01 = -h01 / det
        inv11 = h00 / det
        step0 = inv00 * g0 + inv01 * g1
        step1 = inv01 * g0 + inv11 * g1

        intercept -= step0
        beta -= step1

        if max(abs(step0), abs(step1)) < NEWTON_TOL:
            break

    return beta, intercept


# --- Fitting from the DB -----------------------------------------------


def _closed_predictions_with_sentiment(
    session: Session,
    *,
    model_version: str,
    target_symbol: Optional[str] = None,
) -> list[tuple[float, int]]:
    """Return ``[(sentiment_index, up_realised), ...]`` for closed predictions.

    ``up_realised = 1`` when the next-bar return is positive, else 0.
    FLAT predictions are dropped — we're fitting the directional call,
    not the FLAT-vs-non-FLAT decision.
    """
    stmt = (
        select(Prediction.sentiment_index, PredictionOutcome.realised_return)
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(Prediction.model_version == model_version)
        .where(Prediction.label.in_(("UP", "DOWN")))
    )
    if target_symbol is not None:
        stmt = stmt.where(Prediction.target_symbol == target_symbol)

    out: list[tuple[float, int]] = []
    for sentiment, realised in session.execute(stmt):
        out.append((float(sentiment), 1 if float(realised) > 0 else 0))
    return out


def fit_logreg_calibration(
    session: Session,
    *,
    model_version: str,
    target_symbol: Optional[str] = None,
) -> LogisticCalibration:
    """Fit a calibration from this model's closed predictions.

    Raises :class:`NotEnoughCalibrationDataError` when fewer than
    :data:`MIN_FIT_SAMPLES` closed predictions are available, or when
    all of them landed on a single outcome (a one-class dataset has no
    decision boundary to fit).
    """
    pairs = _closed_predictions_with_sentiment(
        session, model_version=model_version, target_symbol=target_symbol
    )
    n = len(pairs)
    if n < MIN_FIT_SAMPLES:
        raise NotEnoughCalibrationDataError(
            f"need at least {MIN_FIT_SAMPLES} closed UP/DOWN predictions to fit, "
            f"have {n}"
        )

    ys = [y for _, y in pairs]
    if len(set(ys)) < 2:
        raise NotEnoughCalibrationDataError(
            "all closed outcomes landed on a single class — can't fit a "
            "decision boundary"
        )

    xs = [x for x, _ in pairs]
    beta, intercept = _fit_logreg_newton(xs, ys)
    return LogisticCalibration(beta=beta, intercept=intercept, n_samples=n)


# --- At-predict-time helpers -------------------------------------------


def apply_logreg_classification(
    sentiment_index: float,
    calibration: LogisticCalibration,
    *,
    decision_band: float = DEFAULT_DECISION_BAND,
) -> tuple[str, float]:
    """Map ``sentiment_index`` through ``calibration`` to ``(label, confidence)``.

    ``decision_band`` is the half-width around 0.5 in probability space
    within which we emit ``FLAT``. With the default 0.05, ``P(up)`` must
    be ≥ 0.55 for ``UP`` and ≤ 0.45 for ``DOWN``.

    Confidence is the calibrated ``|2P − 1|``, so a probability of 0.8
    or 0.2 both map to confidence 0.6 — and the magnitude is now a
    real probability gap rather than an arbitrary z-distance.
    """
    p_up = calibration.probability_up(sentiment_index)
    confidence = abs(2.0 * p_up - 1.0)
    if p_up > 0.5 + decision_band:
        return "UP", confidence
    if p_up < 0.5 - decision_band:
        return "DOWN", confidence
    return "FLAT", confidence


def save_calibration(
    session: Session, calibration: LogisticCalibration
) -> None:
    """Persist ``calibration`` as the active classifier calibration.

    Stored as a JSON blob in a single :class:`AppSetting` row so we
    don't need a dedicated table for two floats. The predictors look
    this key up once per call when ``FINN_PREDICTOR_CLASSIFIER=logreg``.
    """
    from finn_predictor.storage.repo import set_setting

    payload = json.dumps(
        {
            "beta": float(calibration.beta),
            "intercept": float(calibration.intercept),
            "n_samples": int(calibration.n_samples),
        }
    )
    set_setting(session, SETTING_KEY, payload)


def load_calibration(session: Session) -> Optional[LogisticCalibration]:
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
        return LogisticCalibration(
            beta=float(data["beta"]),
            intercept=float(data["intercept"]),
            n_samples=int(data.get("n_samples", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def resolve_classifier_mode() -> str:
    """Return ``'rule'`` (default) or ``'logreg'`` from the env.

    Anything unrecognised falls back to ``rule`` for the same reason as
    :func:`finn_predictor.sentiment.resolve_active_scorer`: a typo
    shouldn't take the UI offline.
    """
    raw = os.environ.get("FINN_PREDICTOR_CLASSIFIER", "rule")
    name = (raw or "rule").strip().lower() or "rule"
    if name in {"rule", "logreg"}:
        return name
    return "rule"
