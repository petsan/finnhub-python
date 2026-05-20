"""Pluggable sentiment scoring.

Iteration 1 ships :class:`~finn_predictor.sentiment.vader.VaderScorer`.
Iteration 2 introduces :class:`~finn_predictor.sentiment.finbert.FinBertScorer`
behind the same interface.
"""

from finn_predictor.sentiment.base import Scorer, get_scorer
from finn_predictor.sentiment.vader import VaderScorer

__all__ = ["Scorer", "VaderScorer", "get_scorer"]
