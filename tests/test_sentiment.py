"""Sentiment scorer tests."""

from __future__ import annotations

import pytest

from finn_predictor.sentiment import (
    Scorer,
    VaderScorer,
    get_scorer,
    resolve_active_scorer,
)
from finn_predictor.sentiment.finbert import FinBertScorer


def test_vader_scorer_satisfies_protocol() -> None:
    s = VaderScorer()
    assert isinstance(s, Scorer)
    assert s.model_version.startswith("vader-")


def test_vader_positive_negative_neutral_ordering() -> None:
    # VADER's lexicon is built around general-purpose affect words; we
    # deliberately use ones it knows so the predictor's iter-1 baseline isn't
    # at the mercy of finance-domain neologisms (which is also why iter-2
    # swaps in FinBERT).
    s = VaderScorer()
    pos = s.score("Amazing record-breaking quarter, profits soared")
    neg = s.score("Markets crash as panic selling and bankruptcies sweep Wall Street")
    neu = s.score("The company filed its annual report on Tuesday.")
    assert pos > 0.2
    assert neg < -0.2
    assert abs(neu) < 0.5
    assert pos > neu > neg


def test_vader_score_is_zero_for_empty_text() -> None:
    s = VaderScorer()
    assert s.score("") == 0.0
    assert s.score("   ") == 0.0


def test_vader_score_clipped_to_unit_interval() -> None:
    s = VaderScorer()
    for sample in [
        "fantastic incredible amazing record-breaking",
        "horrible disaster collapse bankruptcy",
        "neutral note",
    ]:
        assert -1.0 <= s.score(sample) <= 1.0


def test_vader_score_many_matches_individual() -> None:
    s = VaderScorer()
    samples = ["great quarter!", "terrible loss", "the company filed"]
    assert s.score_many(samples) == [s.score(t) for t in samples]


def test_get_scorer_resolves_known_and_rejects_unknown() -> None:
    assert isinstance(get_scorer("vader"), VaderScorer)
    with pytest.raises(ValueError):
        get_scorer("nope")


def test_finbert_scorer_uses_injected_pipeline() -> None:
    """We never load torch in tests — inject a fake pipeline."""

    def fake_pipeline(text: str):
        return [{"label": "positive", "score": 0.92}]

    fb = FinBertScorer(pipeline_fn=fake_pipeline)
    assert fb.score("the company beat estimates") == pytest.approx(0.92)
    assert fb.model_version == "finbert-prosusai-1.0"


def test_finbert_handles_negative_and_neutral_labels() -> None:
    pos = FinBertScorer(pipeline_fn=lambda t: [{"label": "positive", "score": 0.9}])
    neg = FinBertScorer(pipeline_fn=lambda t: [{"label": "negative", "score": 0.8}])
    neu = FinBertScorer(pipeline_fn=lambda t: [{"label": "neutral", "score": 0.7}])
    assert pos.score("x") == pytest.approx(0.9)
    assert neg.score("x") == pytest.approx(-0.8)
    assert neu.score("x") == 0.0


def test_finbert_empty_text_returns_zero() -> None:
    """Even with a pipeline that would crash, empty text short-circuits."""

    def bomb(text: str):
        raise AssertionError("should not be called for empty text")

    fb = FinBertScorer(pipeline_fn=bomb)
    assert fb.score("") == 0.0


def test_finbert_handles_empty_pipeline_response() -> None:
    fb = FinBertScorer(pipeline_fn=lambda t: [])
    assert fb.score("some text") == 0.0


def test_finbert_get_scorer_lazy_import() -> None:
    """get_scorer('finbert') only imports the heavy class — instantiating
    without a fake pipeline shouldn't run inference until score() is called.
    """
    fb = get_scorer("finbert")
    assert isinstance(fb, FinBertScorer)
    # We do NOT call .score() to avoid pulling torch.


def test_resolve_active_scorer_defaults_to_vader(monkeypatch) -> None:
    monkeypatch.delenv("FINN_PREDICTOR_SCORER", raising=False)
    assert isinstance(resolve_active_scorer(), VaderScorer)


def test_resolve_active_scorer_honours_finbert_env(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_SCORER", "finbert")
    assert isinstance(resolve_active_scorer(), FinBertScorer)


def test_resolve_active_scorer_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_SCORER", "  FinBERT  ")
    assert isinstance(resolve_active_scorer(), FinBertScorer)


def test_resolve_active_scorer_falls_back_on_unknown(monkeypatch) -> None:
    """Unrecognised env values shouldn't crash the UI at startup."""
    monkeypatch.setenv("FINN_PREDICTOR_SCORER", "magic-llm")
    assert isinstance(resolve_active_scorer(), VaderScorer)


def test_resolve_active_scorer_falls_back_on_blank(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_SCORER", "")
    assert isinstance(resolve_active_scorer(), VaderScorer)
