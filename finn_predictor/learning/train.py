"""Train + persist learned weights using Bayesian optimisation.

Public API:
* :func:`train_weights(session, ...)` runs gp_minimize over the
  learnable dimensions, persists the result as a new :class:`LearnedWeight`
  version, auto-activates it, and returns a :class:`TrainingReport`.

The search splits trades into train / holdout by date so we can detect
overfitting: the objective minimised by skopt is the **training-set**
blended score, but the report also computes the same blend on the
holdout window.

For source weights we sidestep skopt — a closed-form per-source
hit-rate ratio is good enough and means we don't blow the search
dimensionality up with O(n_sources) extra knobs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.learning.config import (
    DIM_HALF_LIFE,
    DIM_MIN_SIGMA,
    DIM_SOURCE_WEIGHT,
    DIM_THRESHOLD,
    LearnedConfig,
    active_weights,
)
from finn_predictor.learning.simulate import (
    TrainingFrame,
    blended_objective,
    build_training_frame,
    simulate,
)
from finn_predictor.sentiment.vader import VaderScorer
from finn_predictor.storage.models import LearnedWeight
from finn_predictor.storage.repo import (
    POLICY_AUTO,
    activate_learned_version,
    get_activation_policy,
    get_holdout_tolerance,
)


logger = logging.getLogger(__name__)


# Search bounds. Hand-picked from the existing hand-tuned values to give
# the optimiser room without letting it wander to absurd places.
SEARCH_SPACE = (
    ("threshold_sigma", 0.10, 1.50),
    ("min_baseline_sigma", 0.01, 0.30),
    ("half_life_hours", 2.0, 36.0),
)

# Need at least this many closed trades to bother training. Anything
# less and the holdout split has zero trades.
MIN_TRADES_FOR_TRAINING = 10
HOLDOUT_DAYS = 14
DEFAULT_N_CALLS = 30


class NotEnoughDataError(RuntimeError):
    """Raised when the trade ledger is too small to fit weights reliably."""


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a single training run; returned by :func:`train_weights`."""

    version: int
    training_score: float
    holdout_score: Optional[float]
    baseline_training_score: float
    baseline_holdout_score: Optional[float]
    n_train: int
    n_holdout: int
    n_calls: int
    fitted: dict[str, float] = field(default_factory=dict)
    source_weights: dict[str, float] = field(default_factory=dict)
    # Whether this version is the live one after train_weights returned.
    activated: bool = False
    # Gate diagnostics. Populated only when activation was eligible
    # (policy=AUTO and an explicit override wasn't passed).
    active_holdout_score_at_decision: Optional[float] = None
    holdout_tolerance: Optional[float] = None
    gate_blocked: bool = False
    gate_reason: Optional[str] = None


# ---------------- frame splitting ----------------


def _split_frame_by_date(
    frame: TrainingFrame, *, cutoff: datetime
) -> tuple[TrainingFrame, TrainingFrame]:
    """Split into (train, holdout) by ``prediction_date < cutoff``."""
    train_keys = {k for k in frame.closed_prediction_keys if k[1] < cutoff}
    hold_keys = {k for k in frame.closed_prediction_keys if k[1] >= cutoff}
    return (
        TrainingFrame(
            articles_by_day={k: v for k, v in frame.articles_by_day.items() if k in train_keys},
            realised_by_day={k: v for k, v in frame.realised_by_day.items() if k in train_keys},
        ),
        TrainingFrame(
            articles_by_day={k: v for k, v in frame.articles_by_day.items() if k in hold_keys},
            realised_by_day={k: v for k, v in frame.realised_by_day.items() if k in hold_keys},
        ),
    )


# ---------------- source weights (closed form) ----------------


def _fit_source_weights(frame: TrainingFrame) -> dict[str, float]:
    """Closed-form per-source weight = source_hit_rate / overall_hit_rate.

    Renormalised to mean 1 so the aggregate scale doesn't drift. Sources
    with fewer than 3 articles use the default weight (1.0) so we don't
    chase noise.
    """
    # Tally per-source: total scored articles + whether the day's
    # prediction "hit" (using the realised_return sign as proxy).
    source_counts: dict[str, int] = {}
    source_hits: dict[str, int] = {}
    total = 0
    hits = 0
    for key, arts in frame.articles_by_day.items():
        ret = frame.realised_by_day.get(key)
        if ret is None:
            continue
        ret_sign = 1 if ret > 0 else (-1 if ret < 0 else 0)
        # "Hit" proxy: an article from this source backs the direction
        # of the realised move if its score sign agrees.
        for a in arts:
            total += 1
            score_sign = 1 if a.raw_score > 0 else (-1 if a.raw_score < 0 else 0)
            agreed = score_sign != 0 and score_sign == ret_sign
            source_counts[a.source] = source_counts.get(a.source, 0) + 1
            if agreed:
                source_hits[a.source] = source_hits.get(a.source, 0) + 1
                hits += 1

    if total == 0:
        return {}
    overall = hits / total
    weights: dict[str, float] = {}
    for src, n in source_counts.items():
        if n < 3:
            continue
        src_rate = source_hits.get(src, 0) / n
        if overall == 0:
            continue
        weights[src] = src_rate / overall

    if not weights:
        return {}
    mean_w = sum(weights.values()) / len(weights)
    if mean_w == 0:
        return {}
    return {src: w / mean_w for src, w in weights.items()}


# ---------------- skopt loop ----------------


def _next_version(session: Session) -> int:
    max_v = session.scalar(
        select(LearnedWeight.version).order_by(LearnedWeight.version.desc()).limit(1)
    )
    return (max_v or 0) + 1


def _activate_version(session: Session, version: int) -> None:
    """Set ``is_active = True`` for the given version, False for the rest.

    Kept as a private wrapper so callers don't need to import the repo
    helper directly. The shared implementation lives in
    :func:`storage.repo.activate_learned_version`.
    """
    activate_learned_version(session, version)


def _persist_weights(
    session: Session,
    *,
    version: int,
    threshold: float,
    min_sigma: float,
    half_life: float,
    source_weights: dict[str, float],
    training_score: float,
    holdout_score: Optional[float],
) -> None:
    """Write one row per dimension. Caller decides whether to activate."""
    rows = [
        LearnedWeight(
            version=version, dimension=DIM_THRESHOLD, key=None,
            value=float(threshold),
            training_score=training_score, holdout_score=holdout_score,
            is_active=False,
        ),
        LearnedWeight(
            version=version, dimension=DIM_MIN_SIGMA, key=None,
            value=float(min_sigma),
            training_score=training_score, holdout_score=holdout_score,
            is_active=False,
        ),
        LearnedWeight(
            version=version, dimension=DIM_HALF_LIFE, key=None,
            value=float(half_life),
            training_score=training_score, holdout_score=holdout_score,
            is_active=False,
        ),
    ]
    for src, w in source_weights.items():
        rows.append(
            LearnedWeight(
                version=version, dimension=DIM_SOURCE_WEIGHT, key=src,
                value=float(w),
                training_score=training_score, holdout_score=holdout_score,
                is_active=False,
            )
        )
    session.add_all(rows)
    session.commit()


def train_weights(
    session: Session,
    *,
    model_version: Optional[str] = None,
    n_calls: int = DEFAULT_N_CALLS,
    holdout_days: int = HOLDOUT_DAYS,
    activate: Optional[bool] = None,
    random_state: int = 0,
) -> TrainingReport:
    """Run Bayesian optimisation over the learnable dimensions.

    Steps:
        1. Materialise a :class:`TrainingFrame` (articles + outcomes).
        2. Time-split into (train, holdout) by ``holdout_days``.
        3. Fit per-source weights via closed-form hit-rate ratio (on the
           training half only).
        4. Pass the training frame + source weights into skopt's
           ``gp_minimize``; the search space is the three scalar knobs
           (threshold, min_sigma, half_life) listed in :data:`SEARCH_SPACE`.
        5. Persist the fitted weights as a new :class:`LearnedWeight`
           version. The ``activate`` argument decides whether to flip
           it live: ``True`` always activates, ``False`` never activates,
           and the default ``None`` reads the persisted
           ``activation_policy`` setting (AUTO ⇒ activate, MANUAL ⇒
           leave inactive — user must click *Activate* in the UI).

    Raises :class:`NotEnoughDataError` when fewer than
    :data:`MIN_TRADES_FOR_TRAINING` closed predictions exist.
    """
    model_version = model_version or VaderScorer().model_version
    frame = build_training_frame(session, model_version=model_version)
    total = len(frame.closed_prediction_keys)
    if total < MIN_TRADES_FOR_TRAINING:
        raise NotEnoughDataError(
            f"need at least {MIN_TRADES_FOR_TRAINING} closed predictions to train, "
            f"have {total}"
        )

    cutoff = datetime.now(timezone.utc) - timedelta(days=holdout_days)
    train_frame, hold_frame = _split_frame_by_date(frame, cutoff=cutoff)

    # If the holdout would be empty (e.g. all predictions older than
    # holdout_days), fall back to training on the whole set with no holdout.
    use_holdout = len(hold_frame.closed_prediction_keys) > 0

    source_weights = _fit_source_weights(train_frame)

    # Score the baseline (no weights) on both splits for the report.
    baseline = LearnedConfig(
        version=None,
        threshold_sigma=0.5,
        min_baseline_sigma=0.05,
        half_life_hours=12.0,
        source_weights={},
    )
    baseline_train = blended_objective(train_frame, baseline)
    baseline_hold = (
        blended_objective(hold_frame, baseline) if use_holdout else None
    )

    # Build the objective function for skopt.
    def _objective(params: list[float]) -> float:
        cand = LearnedConfig(
            version=None,
            threshold_sigma=float(params[0]),
            min_baseline_sigma=float(params[1]),
            half_life_hours=float(params[2]),
            source_weights=source_weights,
        )
        # skopt minimises; we negate.
        return -blended_objective(train_frame, cand)

    # Lazy-import skopt so the rest of the package boots on systems that
    # don't have it (the UI feature is opt-in).
    from skopt import gp_minimize
    from skopt.space import Real

    space = [
        Real(low, high, name=name)
        for name, low, high in SEARCH_SPACE
    ]
    result = gp_minimize(
        func=_objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=random_state,
        acq_func="EI",
    )
    fitted_threshold = float(result.x[0])
    fitted_min_sigma = float(result.x[1])
    fitted_half_life = float(result.x[2])
    training_score = float(-result.fun)

    fitted_config = LearnedConfig(
        version=None,
        threshold_sigma=fitted_threshold,
        min_baseline_sigma=fitted_min_sigma,
        half_life_hours=fitted_half_life,
        source_weights=source_weights,
    )
    holdout_score = (
        blended_objective(hold_frame, fitted_config) if use_holdout else None
    )

    version = _next_version(session)
    _persist_weights(
        session,
        version=version,
        threshold=fitted_threshold,
        min_sigma=fitted_min_sigma,
        half_life=fitted_half_life,
        source_weights=source_weights,
        training_score=training_score,
        holdout_score=holdout_score,
    )

    # Resolve the activation decision in two stages.
    #
    # 1. Policy stage: explicit True/False from the caller wins; None
    #    means "consult the persisted policy".
    # 2. Gate stage: when the policy says AUTO, re-score the currently-
    #    active config on the SAME holdout window we just used and
    #    refuse to activate if the new candidate scores worse than
    #    (active_holdout - tolerance). The gate is a safety net for
    #    auto runs; explicit `activate=True` from a caller skips it.
    explicit_decision = activate is not None
    if activate is None:
        activate = get_activation_policy(session) == POLICY_AUTO

    active_holdout_at_decision: Optional[float] = None
    tolerance = get_holdout_tolerance(session)
    gate_blocked = False
    gate_reason: Optional[str] = None

    if activate and use_holdout and not explicit_decision:
        # Re-score the live config on this same holdout window so the
        # comparison is apples-to-apples. The live config might be
        # baseline defaults if nothing has been activated yet.
        active_cfg = active_weights(session)
        # The newly-persisted version's is_active flag is False until we
        # flip it, so active_weights() correctly returns the prior live
        # config. Score it on the holdout we just built.
        active_holdout_at_decision = blended_objective(hold_frame, active_cfg)

        if holdout_score is not None and (
            holdout_score < active_holdout_at_decision - tolerance
        ):
            gate_blocked = True
            gate_reason = (
                f"holdout score {holdout_score:+.4f} is "
                f"{active_holdout_at_decision - holdout_score:+.4f} "
                f"below the live config's {active_holdout_at_decision:+.4f} "
                f"on the same holdout window (tolerance {tolerance:.4f}). "
                "The new version is saved but left inactive — flip it "
                "manually in the UI if you want to override the gate."
            )
            activate = False

    if activate:
        _activate_version(session, version)

    return TrainingReport(
        version=version,
        training_score=training_score,
        holdout_score=holdout_score,
        baseline_training_score=baseline_train,
        baseline_holdout_score=baseline_hold,
        n_train=len(train_frame.closed_prediction_keys),
        n_holdout=len(hold_frame.closed_prediction_keys),
        n_calls=n_calls,
        fitted={
            "threshold_sigma": fitted_threshold,
            "min_baseline_sigma": fitted_min_sigma,
            "half_life_hours": fitted_half_life,
        },
        source_weights=source_weights,
        activated=activate,
        active_holdout_score_at_decision=active_holdout_at_decision,
        holdout_tolerance=tolerance,
        gate_blocked=gate_blocked,
        gate_reason=gate_reason,
    )
