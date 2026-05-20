"""FinBERT scorer (iteration 2).

This module is intentionally a small shim. Loading the actual ``ProsusAI/finbert``
weights pulls in ``torch`` + ``transformers`` (~hundreds of MB) and is too
heavy for the default test environment. The class therefore lazy-imports
those libraries on first use and exposes the same :class:`Scorer` interface
as :class:`VaderScorer`, making it a drop-in replacement.

We accept an injected ``pipeline_fn`` to keep this unit-testable without
network downloads.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

_PipelineFn = Callable[[str], list[dict[str, float | str]]]


class FinBertScorer:
    """Map FinBERT's {positive, negative, neutral} output to ``[-1, +1]``."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, pipeline_fn: Optional[_PipelineFn] = None) -> None:
        self._pipeline: Optional[_PipelineFn] = pipeline_fn
        self._version = "finbert-prosusai-1.0"

    @property
    def model_version(self) -> str:
        return self._version

    def _ensure_pipeline(self) -> _PipelineFn:
        if self._pipeline is not None:
            return self._pipeline

        # Lazy import: don't pull torch unless someone actually invokes the model.
        from transformers import pipeline  # type: ignore[import-not-found]

        hf_pipe = pipeline("sentiment-analysis", model=self.MODEL_NAME)

        def _call(text: str) -> list[dict[str, float | str]]:
            return hf_pipe(text)

        self._pipeline = _call
        return _call

    @staticmethod
    def _map_label_to_score(label: str, prob: float) -> float:
        label = label.lower()
        if label == "positive":
            return float(prob)
        if label == "negative":
            return -float(prob)
        return 0.0  # neutral

    def score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        results = self._ensure_pipeline()(text)
        if not results:
            return 0.0
        top = results[0]
        label = str(top.get("label", "neutral"))
        prob = float(top.get("score", 0.0))
        return self._map_label_to_score(label, prob)

    def score_many(self, texts: Iterable[str]) -> list[float]:
        return [self.score(t) for t in texts]
