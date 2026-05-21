"""Scorer interface + factory.

A Scorer maps a piece of text (typically ``headline + " " + summary``) to a
real-valued sentiment score in ``[-1.0, +1.0]`` and reports a stable
``model_version`` string used as a primary-key component in the DB.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    """Protocol that all sentiment scorers must implement."""

    @property
    def model_version(self) -> str:
        """Stable identifier (e.g. ``vader-3.3.2``)."""
        ...

    def score(self, text: str) -> float:
        """Return a sentiment score in ``[-1.0, +1.0]``."""
        ...

    def score_many(self, texts: Iterable[str]) -> list[float]:
        """Convenience batch wrapper. Default impl loops over ``score``."""
        ...


def get_scorer(name: str = "vader") -> Scorer:
    """Resolve a scorer by name.

    ``vader`` is always available. ``finbert`` is lazy-imported so the heavy
    torch/transformers stack only loads when explicitly requested.
    """
    name = name.lower()
    if name == "vader":
        from finn_predictor.sentiment.vader import VaderScorer

        return VaderScorer()
    if name == "finbert":
        # Lazy import: keeps torch out of the default dependency tree.
        from finn_predictor.sentiment.finbert import FinBertScorer

        return FinBertScorer()
    raise ValueError(f"Unknown scorer: {name!r}")
