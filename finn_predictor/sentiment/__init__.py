"""Pluggable sentiment scoring.

Iteration 1 ships :class:`~finn_predictor.sentiment.vader.VaderScorer`.
Iteration 2 introduces :class:`~finn_predictor.sentiment.finbert.FinBertScorer`
behind the same interface.
"""

from __future__ import annotations

import os

from finn_predictor.sentiment.base import Scorer, get_scorer
from finn_predictor.sentiment.vader import VaderScorer


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


__all__ = ["Scorer", "VaderScorer", "get_scorer", "resolve_active_scorer"]
