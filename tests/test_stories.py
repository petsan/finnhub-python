"""Tests for finn_predictor.storage.stories."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.storage.repo import upsert_articles
from finn_predictor.storage.stories import earliest_story_times, story_key
from tests.conftest import make_article


D = datetime(2026, 5, 19, 14, tzinfo=timezone.utc)


# ---------------- story_key ----------------


def test_story_key_empty_inputs() -> None:
    assert story_key("") == ""
    assert story_key("   ") == ""
    assert story_key("!!!") == ""


def test_story_key_normalises_case_and_punctuation() -> None:
    a = story_key("Apple Inc. Reports Strong Q3 Earnings — Stock Surges")
    b = story_key("apple inc reports strong q3 earnings stock surges")
    assert a == b


def test_story_key_drops_leading_articles() -> None:
    """'The Fed cuts rates' must cluster with 'Fed cuts rates'."""
    assert story_key("The Fed cuts rates by 25 bps") == story_key(
        "Fed cuts rates by 25 bps"
    )
    assert story_key("A new chip from Apple") == story_key("new chip from Apple")
    assert story_key("An unusual move by the SEC") == story_key(
        "unusual move by the SEC"
    )


def test_story_key_keeps_first_n_words() -> None:
    key = story_key("Apple Microsoft Nvidia Tesla Amazon Meta Google Adobe Cisco")
    # Default n_words=8 → "apple microsoft nvidia tesla amazon meta google adobe"
    assert key.split() == [
        "apple", "microsoft", "nvidia", "tesla",
        "amazon", "meta", "google", "adobe",
    ]


def test_story_key_collapses_whitespace() -> None:
    assert story_key("Apple\n\tbeats\n\nestimates") == story_key("Apple beats estimates")


def test_story_key_treats_unicode_punctuation_as_separator() -> None:
    """Em-dashes, smart quotes, etc. shouldn't carry over into the key."""
    key1 = story_key("Apple beats — analysts cheer")
    key2 = story_key("Apple beats analysts cheer")
    assert key1 == key2


# ---------------- earliest_story_times ----------------


def test_earliest_story_times_empty_input(session) -> None:
    assert earliest_story_times(session, []) == {}


def test_earliest_story_times_finds_earliest_match(session) -> None:
    """Three articles about the same story; the earliest published_at wins."""
    upsert_articles(
        session,
        [
            make_article(
                finnhub_id=1,
                headline="Fed cuts rates by 25 bps",
                published_at=D - timedelta(hours=3),
            ),
            make_article(
                finnhub_id=2,
                headline="The Fed cuts rates by 25 bps, analysts say",
                published_at=D - timedelta(hours=1),
            ),
            make_article(
                finnhub_id=3,
                headline="Fed cuts rates by 25 bps in surprise move",
                published_at=D,
            ),
        ],
    )

    out = earliest_story_times(
        session,
        ["Fed cuts rates by 25 bps in surprise move"],
        now=D + timedelta(hours=1),
    )
    earliest = out["Fed cuts rates by 25 bps in surprise move"]
    assert earliest is not None
    # SQLite roundtrip can strip the tz; compare as naive UTC.
    expected = (D - timedelta(hours=3)).replace(tzinfo=None)
    actual = earliest.replace(tzinfo=None) if earliest.tzinfo else earliest
    assert actual == expected


def test_earliest_story_times_unrelated_stories_stay_separate(session) -> None:
    upsert_articles(
        session,
        [
            make_article(
                finnhub_id=1,
                headline="Apple beats Q3 earnings",
                published_at=D - timedelta(hours=5),
            ),
            make_article(
                finnhub_id=2,
                headline="Microsoft launches new cloud product",
                published_at=D - timedelta(hours=2),
            ),
        ],
    )

    out = earliest_story_times(
        session,
        ["Apple beats Q3 earnings", "Microsoft launches new cloud product"],
        now=D,
    )
    assert out["Apple beats Q3 earnings"] is not None
    assert out["Microsoft launches new cloud product"] is not None
    assert out["Apple beats Q3 earnings"] != out["Microsoft launches new cloud product"]


def test_earliest_story_times_respects_lookback_window(session) -> None:
    """Articles published before the lookback cutoff must not match."""
    upsert_articles(
        session,
        [
            make_article(
                finnhub_id=1,
                headline="Fed cuts rates",
                published_at=D - timedelta(days=30),  # outside default 14d window
            ),
            make_article(
                finnhub_id=2,
                headline="Fed cuts rates",
                published_at=D - timedelta(hours=2),  # inside
            ),
        ],
    )

    out = earliest_story_times(
        session,
        ["Fed cuts rates"],
        lookback_days=14,
        now=D,
    )
    earliest = out["Fed cuts rates"]
    assert earliest is not None
    # Should be the 2h-old article, NOT the 30-day-old one.
    expected = (D - timedelta(hours=2)).replace(tzinfo=None)
    actual = earliest.replace(tzinfo=None) if earliest.tzinfo else earliest
    assert actual == expected


def test_earliest_story_times_missing_headline_returns_none(session) -> None:
    out = earliest_story_times(session, ["A headline nobody published"], now=D)
    assert out == {"A headline nobody published": None}


def test_earliest_story_times_empty_headline_returns_none(session) -> None:
    out = earliest_story_times(session, ["", "  "], now=D)
    assert out == {"": None, "  ": None}


def test_earliest_story_times_prefix_clusters_short_to_long(session) -> None:
    """A short headline (< n_words) must cluster with longer variants of
    the same story."""
    upsert_articles(
        session,
        [
            # Original wire: 5 words. Story key is shorter than n_words=8.
            make_article(
                finnhub_id=1,
                headline="Fed cuts rates by 25",
                published_at=D - timedelta(hours=4),
            ),
            # Later expansion: 9 words. Story key extends the original.
            make_article(
                finnhub_id=2,
                headline="Fed cuts rates by 25 bps in surprise move overnight",
                published_at=D,
            ),
        ],
    )
    out = earliest_story_times(
        session,
        ["Fed cuts rates by 25 bps in surprise move overnight"],
        now=D + timedelta(hours=1),
    )
    earliest = out["Fed cuts rates by 25 bps in surprise move overnight"]
    assert earliest is not None
    expected = (D - timedelta(hours=4)).replace(tzinfo=None)
    actual = earliest.replace(tzinfo=None) if earliest.tzinfo else earliest
    assert actual == expected


def test_earliest_story_times_handles_duplicate_input_headlines(session) -> None:
    upsert_articles(
        session,
        [make_article(finnhub_id=1, headline="Apple beats", published_at=D - timedelta(hours=4))],
    )
    out = earliest_story_times(session, ["Apple beats", "Apple beats"], now=D)
    assert len(out) == 1  # dict dedupes by key
    assert out["Apple beats"] is not None
