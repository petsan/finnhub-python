"""Tests for the pluggable story-clustering layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.storage.clustering import (
    DEFAULT_COSINE_THRESHOLD,
    Clusterer,
    EmbeddingClusterer,
    PrefixClusterer,
    _cosine,
    resolve_active_clusterer,
)
from finn_predictor.storage.repo import upsert_articles
from tests.conftest import make_article


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


# --- Pure-math sanity --------------------------------------------------


def test_cosine_orthogonal_vectors_zero() -> None:
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_identical_vectors_one() -> None:
    assert _cosine([0.3, 0.4], [0.3, 0.4]) == pytest.approx(1.0)


def test_cosine_zero_vector_returns_zero() -> None:
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_mismatched_lengths_returns_zero() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


# --- PrefixClusterer protocol compliance -------------------------------


def test_prefix_clusterer_satisfies_protocol() -> None:
    c = PrefixClusterer()
    assert isinstance(c, Clusterer)
    assert c.name == "prefix"


def _as_naive(ts):
    """SQLite drops tz info on roundtrip; normalise for comparison."""
    if ts is None:
        return None
    return ts.replace(tzinfo=None)


def test_prefix_clusterer_delegates_to_legacy_helper(session) -> None:
    """PrefixClusterer reproduces the same result as the original function."""
    earlier = D - timedelta(hours=2)
    later = D
    upsert_articles(
        session,
        [
            make_article(finnhub_id=1, headline="Fed cuts rates by 25 bps",
                         published_at=earlier),
            make_article(finnhub_id=2, headline="Fed cuts rates by 25 bps in surprise",
                         published_at=later),
        ],
    )
    out = PrefixClusterer().earliest_times(
        session,
        ["Fed cuts rates by 25 bps in surprise"],
        now=D + timedelta(hours=1),
    )
    got = out["Fed cuts rates by 25 bps in surprise"]
    assert _as_naive(got) == _as_naive(earlier)


# --- EmbeddingClusterer with injected embed_fn -------------------------


def _fake_embedder(table: dict[str, list[float]]):
    """Return a stable embed_fn that maps headlines via ``table``.

    Unknown headlines map to a zero vector — they cosine to 0 against
    everything, which mimics the "no signal" path without involving the
    real model.
    """

    def _embed(texts: list[str]) -> list[list[float]]:
        return [list(table.get(t, [0.0, 0.0, 0.0])) for t in texts]

    return _embed


def test_embedding_clusterer_groups_paraphrased_headlines(session) -> None:
    """Two headlines with identical embeddings cluster, picking earliest time."""
    earlier = D - timedelta(hours=3)
    later = D
    upsert_articles(
        session,
        [
            make_article(
                finnhub_id=10,
                headline="Apple ships record iPhone quarter",
                published_at=earlier,
            ),
            make_article(
                finnhub_id=11,
                headline="Apple posts blockbuster iPhone earnings",
                published_at=later,
            ),
        ],
    )

    table = {
        "Apple ships record iPhone quarter": [1.0, 0.0, 0.0],
        "Apple posts blockbuster iPhone earnings": [0.95, 0.05, 0.0],
        "Unrelated bond rally": [0.0, 0.0, 1.0],
    }
    clusterer = EmbeddingClusterer(embed_fn=_fake_embedder(table), threshold=0.7)
    out = clusterer.earliest_times(
        session,
        ["Apple posts blockbuster iPhone earnings", "Unrelated bond rally"],
        now=D + timedelta(hours=1),
    )
    # The blockbuster headline clusters with the earlier 'ships record'
    # one because their cosine ≈ 0.998 ≥ 0.7. The bond rally has no
    # candidate in the DB it embeds close to → None.
    assert _as_naive(out["Apple posts blockbuster iPhone earnings"]) == _as_naive(earlier)
    assert out["Unrelated bond rally"] is None


def test_embedding_clusterer_respects_threshold(session) -> None:
    """A cosine just below the threshold doesn't cluster."""
    other_time = D - timedelta(hours=5)
    upsert_articles(
        session,
        [
            make_article(
                finnhub_id=20,
                headline="Some headline",
                published_at=other_time,
            ),
        ],
    )
    # Two unit vectors with cosine = 0.6 — below the default 0.7.
    import math
    table = {
        "Some headline": [1.0, 0.0],
        "Different topic": [0.6, math.sqrt(1 - 0.6 ** 2)],  # cosine=0.6 w/ above
    }
    clusterer = EmbeddingClusterer(embed_fn=_fake_embedder(table))
    out = clusterer.earliest_times(
        session, ["Different topic"], now=D + timedelta(hours=1)
    )
    assert out == {"Different topic": None}


def test_embedding_clusterer_handles_empty_inputs(session) -> None:
    out = EmbeddingClusterer(embed_fn=lambda xs: [[0.0]] * len(xs)).earliest_times(
        session, [], now=D
    )
    assert out == {}


def test_embedding_clusterer_handles_empty_window(session) -> None:
    """No candidates in the lookback window → all Nones, no embed call."""
    called: list[list[str]] = []

    def _embed(texts: list[str]) -> list[list[float]]:
        called.append(list(texts))
        return [[1.0, 0.0]] * len(texts)

    clusterer = EmbeddingClusterer(embed_fn=_embed)
    out = clusterer.earliest_times(session, ["any headline"], now=D)
    assert out == {"any headline": None}
    # The candidate fast-path short-circuits before any embed call.
    assert called == []


def test_embedding_clusterer_dedupes_input_batch(session) -> None:
    """Two identical input headlines result in only one embed entry."""
    upsert_articles(
        session,
        [make_article(finnhub_id=30, headline="Same story", published_at=D - timedelta(hours=1))],
    )

    seen: list[list[str]] = []

    def _embed(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return [[1.0, 0.0]] * len(texts)

    EmbeddingClusterer(embed_fn=_embed).earliest_times(
        session, ["Repeat", "Repeat"], now=D + timedelta(hours=1)
    )
    # One batched call: 1 unique input + 1 candidate = 2 items
    assert seen == [["Repeat", "Same story"]]


def test_embedding_clusterer_falls_back_to_prefix_on_embed_failure(session) -> None:
    """An embed exception logs a warning and falls back to prefix matching."""
    earlier = D - timedelta(hours=2)
    upsert_articles(
        session,
        [
            make_article(
                finnhub_id=40,
                headline="Fed cuts rates by 25 bps",
                published_at=earlier,
            )
        ],
    )

    def _bomb(texts):
        raise RuntimeError("embedding model died")

    clusterer = EmbeddingClusterer(embed_fn=_bomb)
    out = clusterer.earliest_times(
        session,
        ["Fed cuts rates by 25 bps in surprise"],
        now=D + timedelta(hours=1),
    )
    # PrefixClusterer would match these via the prefix-aware fallback
    assert _as_naive(out["Fed cuts rates by 25 bps in surprise"]) == _as_naive(earlier)


# --- Env-driven resolver ----------------------------------------------


def test_resolve_active_clusterer_defaults_to_prefix(monkeypatch) -> None:
    monkeypatch.delenv("FINN_PREDICTOR_CLUSTERER", raising=False)
    assert isinstance(resolve_active_clusterer(), PrefixClusterer)


def test_resolve_active_clusterer_honours_embedding(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_CLUSTERER", "Embedding")
    assert isinstance(resolve_active_clusterer(), EmbeddingClusterer)


def test_resolve_active_clusterer_falls_back_on_unknown(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_CLUSTERER", "magic-rnn")
    assert isinstance(resolve_active_clusterer(), PrefixClusterer)


def test_resolve_active_clusterer_falls_back_on_blank(monkeypatch) -> None:
    monkeypatch.setenv("FINN_PREDICTOR_CLUSTERER", "")
    assert isinstance(resolve_active_clusterer(), PrefixClusterer)


def test_default_cosine_threshold_in_paraphrase_range() -> None:
    assert 0.5 <= DEFAULT_COSINE_THRESHOLD <= 0.9
