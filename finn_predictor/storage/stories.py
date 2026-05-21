"""Story-clustering helpers.

A "story" here is a group of NewsArticle rows that report the same event,
typically across multiple wire services. We don't have headline embeddings
or a clustering library — instead we use a deterministic, dependency-free
**story key**: the first ``N`` lowercase alphanumeric words of the
headline. That catches the most common reposting pattern (Reuters and AP
publishing identical or near-identical leads within minutes of each
other) without false-clustering unrelated stories together.

Two articles have the same ``story_key`` ⇒ they are treated as the same
story for the purpose of reporting **first-seen** time.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import NewsArticle


# Drop anything that isn't a letter, digit, or whitespace. Punctuation,
# em-dashes, quotation marks, etc. get collapsed to spaces.
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE_RE = re.compile(r"\s+")

# A small stop-list so headlines starting with "The " / "A " don't collide
# with the same-event headline missing those words.
_LEADING_STOPWORDS = frozenset({"the", "a", "an"})


def story_key(headline: str, *, n_words: int = 8) -> str:
    """Normalise a headline to a comparison key.

    Steps:
        * lowercase the headline,
        * replace any non-alphanumeric run with a single space,
        * collapse whitespace,
        * drop a leading definite/indefinite article (``the``, ``a``, ``an``),
        * keep the first ``n_words`` whitespace-separated tokens.

    Returns the empty string for empty / None / whitespace-only input so
    callers can short-circuit cleanly.
    """
    if not headline:
        return ""
    lowered = headline.lower()
    cleaned = _PUNCT_RE.sub(" ", lowered)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ""
    tokens = cleaned.split(" ")
    if tokens and tokens[0] in _LEADING_STOPWORDS:
        tokens = tokens[1:]
    return " ".join(tokens[:n_words])


def earliest_story_times(
    session: Session,
    headlines: Iterable[str],
    *,
    lookback_days: int = 14,
    now: datetime | None = None,
) -> dict[str, datetime | None]:
    """For each headline, return the earliest matching article's ``published_at``.

    Matching is by :func:`story_key` equality. The search window is the
    last ``lookback_days`` (default 14) — long enough to find the original
    wire report behind a republished story, short enough to keep the scan
    bounded on a busy DB.

    Returns a dict keyed by the *input* headline strings (preserves them
    verbatim so callers can look up by what they passed in). Headlines
    whose story key is empty or that have no matching article in the
    window map to ``None``.
    """
    headlines = list(headlines)
    keys_by_input = {h: story_key(h) for h in headlines}
    needed_keys = {k for k in keys_by_input.values() if k}
    if not needed_keys:
        return {h: None for h in headlines}

    anchor = now or datetime.now(timezone.utc)
    cutoff = anchor - timedelta(days=lookback_days)

    rows = session.execute(
        select(NewsArticle.headline, NewsArticle.published_at).where(
            NewsArticle.published_at >= cutoff
        )
    ).all()

    # Prefix-aware clustering: two stories cluster when one's story_key
    # is a word-prefix of the other's. That catches the case where a
    # short headline ("Fed cuts rates") is the original wire of a later,
    # longer headline ("Fed cuts rates by 25 bps in surprise move").
    def _matches(needed: str, candidate: str) -> bool:
        if not needed or not candidate:
            return False
        if needed == candidate:
            return True
        return needed.startswith(candidate + " ") or candidate.startswith(needed + " ")

    min_by_key: dict[str, datetime] = {}
    for hl, ts in rows:
        k = story_key(hl)
        if not k:
            continue
        for needed_key in needed_keys:
            if not _matches(needed_key, k):
                continue
            cur = min_by_key.get(needed_key)
            if cur is None or ts < cur:
                min_by_key[needed_key] = ts

    return {h: min_by_key.get(keys_by_input[h]) for h in headlines}
