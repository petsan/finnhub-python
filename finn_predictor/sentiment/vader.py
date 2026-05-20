"""VADER-based sentiment scorer.

VADER is rule-/lexicon-based, ships as a small wheel, and scores text on the
fly without any model loading. Its ``compound`` field is already in
``[-1, +1]``, which lines up with our :class:`Scorer` contract.
"""

from __future__ import annotations

from importlib import metadata
from typing import Iterable

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _vader_version() -> str:
    """Look up the installed vaderSentiment version; fall back to ``unknown``."""
    try:
        return metadata.version("vaderSentiment")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


class VaderScorer:
    """Wraps :class:`SentimentIntensityAnalyzer` behind the Scorer Protocol."""

    def __init__(self, analyzer: SentimentIntensityAnalyzer | None = None) -> None:
        self._analyzer = analyzer or SentimentIntensityAnalyzer()
        self._version = f"vader-{_vader_version()}"

    @property
    def model_version(self) -> str:
        return self._version

    def score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        scores = self._analyzer.polarity_scores(text)
        compound = float(scores.get("compound", 0.0))
        # VADER guarantees [-1, 1] but be defensive against future changes.
        return max(-1.0, min(1.0, compound))

    def score_many(self, texts: Iterable[str]) -> list[float]:
        return [self.score(t) for t in texts]
