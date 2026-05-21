"""Pluggable sentiment scoring.

Iteration 1 ships :class:`~finn_predictor.sentiment.vader.VaderScorer`.
Iteration 2 introduces :class:`~finn_predictor.sentiment.finbert.FinBertScorer`
behind the same interface.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.sentiment.base import Scorer, get_scorer
from finn_predictor.sentiment.vader import VaderScorer


logger = logging.getLogger(__name__)


def resolve_active_scorer() -> Scorer:
    """Return the scorer named by ``FINN_PREDICTOR_SCORER`` (default ``vader``).

    This is the single source of truth for *which* scorer the live
    pipeline (UI ingest, CLI ingest, backfill scoring, training-version
    detection) should construct. Reading the env var here — rather than
    threading a config object through every call site — keeps the
    sentiment package usable from short-lived helpers (Streamlit
    callbacks, batch scripts) without a Settings dance.

    Falls back to ``vader`` for any unset / blank / unrecognised value
    rather than raising, because mis-configuration shouldn't break the
    UI on startup; the more rigorous validation lives in
    :func:`finn_predictor.config.load_settings`.
    """
    raw = os.environ.get("FINN_PREDICTOR_SCORER", "vader")
    name = (raw or "vader").strip().lower() or "vader"
    try:
        return get_scorer(name)
    except ValueError:
        return get_scorer("vader")


def detect_scorer_mismatch(
    session: Session, *, active_scorer: Optional[Scorer] = None
) -> Optional[str]:
    """Return a one-line warning when the live scorer doesn't match recent predictions.

    Compares ``active_scorer.model_version`` (default:
    :func:`resolve_active_scorer`) against the ``model_version`` on the
    most recently created :class:`Prediction` row. A mismatch usually
    means someone flipped ``FINN_PREDICTOR_SCORER`` without retraining
    — the live pipeline will from this point forward write predictions
    under the new model_version, but the rolling baseline carries
    history under the previous one until enough fresh days accumulate.

    Returns the message string when a mismatch is detected, ``None``
    otherwise (and ``None`` when no predictions exist at all — a fresh
    DB has nothing to compare against).
    """
    # Lazy import: avoid a circular dependency between sentiment and
    # storage during ``import finn_predictor.sentiment``.
    from finn_predictor.storage.models import Prediction

    scorer = active_scorer if active_scorer is not None else resolve_active_scorer()
    active_version = scorer.model_version

    recent = session.scalar(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(1)
    )
    if recent is None:
        return None
    if recent.model_version == active_version:
        return None
    return (
        f"scorer mismatch: live pipeline configured for "
        f"{active_version!r} but most recent prediction was written "
        f"with {recent.model_version!r}. The rolling baseline will be "
        f"stale until enough fresh days accumulate. Run "
        f"`python -m finn_predictor.cli retrain` after a few sessions "
        f"to refit weights against the new scorer."
    )


def warn_if_scorer_mismatch(
    session: Session, *, active_scorer: Optional[Scorer] = None
) -> Optional[str]:
    """Side-effecting wrapper around :func:`detect_scorer_mismatch`.

    Emits the mismatch message at WARNING level on the package logger
    and returns it (or ``None`` when no mismatch). Idempotent: callers
    can safely invoke this at every startup; the logger filters
    duplicates per process if needed.
    """
    msg = detect_scorer_mismatch(session, active_scorer=active_scorer)
    if msg is not None:
        logger.warning(msg)
    return msg


__all__ = [
    "Scorer",
    "VaderScorer",
    "detect_scorer_mismatch",
    "get_scorer",
    "resolve_active_scorer",
    "warn_if_scorer_mismatch",
]
