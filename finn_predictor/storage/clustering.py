"""Pluggable story-clustering.

Two clusterer implementations:

* :class:`PrefixClusterer` — the default, dependency-free 8-word-prefix
  matcher that shipped with iteration 1. Delegates to
  :func:`finn_predictor.storage.stories.earliest_story_times`.

* :class:`EmbeddingClusterer` — embeds every candidate headline via a
  caller-supplied (or lazy-loaded ``sentence-transformers``) embedding
  function and clusters by cosine similarity. Catches paraphrased
  / rewritten reposts that the prefix matcher misses; conversely
  suppresses the prefix matcher's accidental collisions between
  unrelated stories that happen to share a few opening words.

Selection is via the ``FINN_PREDICTOR_CLUSTERER`` env var
(``prefix`` / ``embedding``) and ``resolve_active_clusterer``. Both
implementations share the same :class:`Clusterer` Protocol so callers
(the UI and any future jobs) need no other branching.

Why a Protocol + factory rather than a flag on the prefix function:
keeps the heavy ``sentence-transformers`` import out of the default
dependency tree, matches the :class:`FinBertScorer` pattern, and
leaves the door open for a third clusterer (e.g. a fine-tuned model)
without changing call sites.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional, Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import NewsArticle
from finn_predictor.storage.stories import earliest_story_times, story_key


logger = logging.getLogger(__name__)


# Default model — small (~80 MB), fast, ships with sentence-transformers'
# default pipeline. Caller can override via the `EmbeddingClusterer(model_name=…)`
# kwarg without touching the env.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine threshold above which two headlines are considered the same
# story. 0.7 is the standard "paraphrase" cutoff for MiniLM-class
# encoders; below this you start clustering unrelated finance stories
# that share the same domain vocabulary.
DEFAULT_COSINE_THRESHOLD = 0.7


EmbedFn = Callable[[list[str]], list[list[float]]]


@runtime_checkable
class Clusterer(Protocol):
    """Maps a batch of headlines to each story's earliest known timestamp."""

    def earliest_times(
        self,
        session: Session,
        headlines: Iterable[str],
        *,
        lookback_days: int = 14,
        now: datetime | None = None,
    ) -> dict[str, datetime | None]:
        ...


# --- Prefix clusterer (default) ----------------------------------------


class PrefixClusterer:
    """Thin wrapper around the dependency-free 8-word-prefix matcher.

    Exists so the call sites can hold a single :class:`Clusterer`
    instance regardless of which implementation is active.
    """

    name = "prefix"

    def earliest_times(
        self,
        session: Session,
        headlines: Iterable[str],
        *,
        lookback_days: int = 14,
        now: datetime | None = None,
    ) -> dict[str, datetime | None]:
        return earliest_story_times(
            session, headlines, lookback_days=lookback_days, now=now,
        )


# --- Embedding clusterer ----------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 on a zero vector."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class EmbeddingClusterer:
    """Sentence-embedding cosine-similarity clusterer.

    Pulls every article in the lookback window, embeds the input
    headlines and the candidate headlines in one batched call, and
    clusters by cosine similarity ≥ ``threshold``. The earliest
    ``published_at`` over each cluster is returned per input headline.

    ``embed_fn`` is injectable so tests can swap in a deterministic
    fake. When ``embed_fn`` is None the default path lazy-imports
    ``sentence-transformers`` on first use and constructs a
    SentenceTransformer with ``model_name``.
    """

    name = "embedding"

    def __init__(
        self,
        *,
        embed_fn: Optional[EmbedFn] = None,
        threshold: float = DEFAULT_COSINE_THRESHOLD,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._embed_fn = embed_fn
        self._threshold = float(threshold)
        self._model_name = model_name

    def _resolve_embed_fn(self) -> EmbedFn:
        if self._embed_fn is not None:
            return self._embed_fn

        # Lazy import: don't pull sentence-transformers (or its torch
        # transitive) unless we actually need to embed.
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        model = SentenceTransformer(self._model_name)

        def _embed(texts: list[str]) -> list[list[float]]:
            vecs = model.encode(texts)
            # Avoid a hard numpy dependency at the type boundary —
            # convert to native lists. SentenceTransformer returns a
            # numpy array; ``.tolist()`` is always available.
            return [list(v) for v in vecs.tolist()]

        self._embed_fn = _embed
        return _embed

    def earliest_times(
        self,
        session: Session,
        headlines: Iterable[str],
        *,
        lookback_days: int = 14,
        now: datetime | None = None,
    ) -> dict[str, datetime | None]:
        headlines = list(headlines)
        result: dict[str, datetime | None] = {h: None for h in headlines}

        # De-dupe the input batch so we don't embed the same headline
        # multiple times — the UI sometimes passes repeats when several
        # rows share a wire.
        non_empty_inputs = [h for h in headlines if h and h.strip()]
        if not non_empty_inputs:
            return result

        anchor = now or datetime.now(timezone.utc)
        cutoff = anchor - timedelta(days=lookback_days)

        rows = session.execute(
            select(NewsArticle.headline, NewsArticle.published_at).where(
                NewsArticle.published_at >= cutoff
            )
        ).all()

        # When the lookback window is empty (e.g. fresh DB) there is
        # nothing to compare against; return the default Nones rather
        # than spin up the embedding model just to embed the inputs.
        if not rows:
            return result

        candidate_headlines: list[str] = []
        candidate_times: list[datetime] = []
        for hl, ts in rows:
            if hl and hl.strip():
                candidate_headlines.append(hl)
                candidate_times.append(ts)
        if not candidate_headlines:
            return result

        # One batched embed call: inputs + all candidates. The fake
        # embed_fn injected in tests gets the full text list so it can
        # return deterministic vectors.
        unique_inputs = list(dict.fromkeys(non_empty_inputs))
        embed = self._resolve_embed_fn()
        try:
            all_vecs = embed(unique_inputs + candidate_headlines)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "embedding clusterer failed (%s); falling back to prefix matcher",
                exc,
            )
            return earliest_story_times(
                session, headlines, lookback_days=lookback_days, now=now,
            )

        input_vecs = all_vecs[: len(unique_inputs)]
        candidate_vecs = all_vecs[len(unique_inputs) :]

        # For each unique input, scan the candidate list once and keep
        # the earliest published_at over the matches.
        min_by_input: dict[str, datetime] = {}
        for ih, iv in zip(unique_inputs, input_vecs):
            best_ts: datetime | None = None
            for cv, ct in zip(candidate_vecs, candidate_times):
                if _cosine(iv, cv) >= self._threshold:
                    if best_ts is None or ct < best_ts:
                        best_ts = ct
            if best_ts is not None:
                min_by_input[ih] = best_ts

        for h in headlines:
            if h in min_by_input:
                result[h] = min_by_input[h]
        return result


# --- Selection ---------------------------------------------------------


def resolve_active_clusterer() -> Clusterer:
    """Pick the clusterer named by ``FINN_PREDICTOR_CLUSTERER``.

    ``prefix`` (default) → :class:`PrefixClusterer`.
    ``embedding`` → :class:`EmbeddingClusterer` with no injected
    ``embed_fn`` (lazy-loads sentence-transformers on first call).

    Unknown / blank values fall back to the prefix matcher rather than
    raising — same pattern as :func:`finn_predictor.sentiment.resolve_active_scorer`.
    """
    raw = os.environ.get("FINN_PREDICTOR_CLUSTERER", "prefix")
    name = (raw or "prefix").strip().lower() or "prefix"
    if name == "embedding":
        return EmbeddingClusterer()
    return PrefixClusterer()


__all__ = [
    "Clusterer",
    "DEFAULT_COSINE_THRESHOLD",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbedFn",
    "EmbeddingClusterer",
    "PrefixClusterer",
    "resolve_active_clusterer",
    "story_key",
]
