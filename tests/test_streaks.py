"""Tests for :func:`finn_predictor.storage.repo.streaks_for` (PR-3).

Streaks power the new ``Streak`` and ``Flipped`` columns on the
per-stock table. The helper must:

* Return one ``StreakInfo`` per requested symbol that has any history,
  and **omit** symbols with no predictions (so callers can use
  ``streaks.get(symbol)`` without checking length first).
* Count the current label's run from the most recent prediction backward.
* Surface the first different label as ``previous_label`` (None when
  the symbol has only ever carried one label).
* Set ``flipped`` exactly when ``previous_label is not None``.
* Honour the ``model_version`` filter so a mixed-model history doesn't
  produce phantom flips.

Backed by the standard in-memory SQLite ``session`` fixture.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from finn_predictor.storage.models import Prediction
from finn_predictor.storage.repo import (
    StreakInfo,
    save_prediction,
    streaks_for,
)


def _day(offset: int) -> datetime:
    """UTC midnight at ``base + offset days`` — matches the predictor's
    start-of-day normalisation so save_prediction's upsert works."""
    return datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(days=offset)


def _save(
    session: Session,
    *,
    symbol: str,
    label: str,
    day_offset: int,
    model_version: str = "vader-test",
) -> Prediction:
    return save_prediction(
        session,
        Prediction(
            target_symbol=symbol,
            prediction_date=_day(day_offset),
            label=label,
            confidence=0.5,
            sentiment_index=0.0,
            article_count=1,
            model_version=model_version,
        ),
    )


# ---------------------------------------------------------------------------
# Shape contracts
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_dict(session: Session) -> None:
    assert streaks_for(session, []) == {}


def test_blank_symbols_dropped(session: Session) -> None:
    """Empty / falsy symbols filter out — same defensive style as the
    rest of repo.py — so a caller doing ``['', None]`` doesn't crash
    inside the IN clause."""
    assert streaks_for(session, ["", None]) == {}  # type: ignore[list-item]


def test_missing_symbol_omitted_not_synthesized(session: Session) -> None:
    """A symbol with no predictions should NOT appear in the result.

    UI callers depend on this to distinguish "no data yet" (streak
    column blank) from "1-day streak" (the brand-new pick case)."""
    _save(session, symbol="AAPL", label="UP", day_offset=0)
    out = streaks_for(session, ["AAPL", "NEVER_INGESTED"])
    assert "NEVER_INGESTED" not in out
    assert "AAPL" in out


def test_single_prediction_streak_of_one(session: Session) -> None:
    _save(session, symbol="AAPL", label="UP", day_offset=0)
    info = streaks_for(session, ["AAPL"])["AAPL"]
    assert info == StreakInfo(
        symbol="AAPL",
        current_label="UP",
        streak=1,
        previous_label=None,
        flipped=False,
    )


# ---------------------------------------------------------------------------
# Streak counting
# ---------------------------------------------------------------------------

def test_two_day_same_label_streak(session: Session) -> None:
    _save(session, symbol="AAPL", label="UP", day_offset=0)
    _save(session, symbol="AAPL", label="UP", day_offset=1)
    info = streaks_for(session, ["AAPL"])["AAPL"]
    assert info.streak == 2
    assert info.previous_label is None
    assert info.flipped is False


def test_five_day_streak(session: Session) -> None:
    for i in range(5):
        _save(session, symbol="AAPL", label="DOWN", day_offset=i)
    info = streaks_for(session, ["AAPL"])["AAPL"]
    assert info.current_label == "DOWN"
    assert info.streak == 5
    assert info.previous_label is None
    assert info.flipped is False


def test_flip_one_day_breaks_streak(session: Session) -> None:
    """Old: UP, UP, UP. Today: DOWN. Streak goes to 1; previous is UP."""
    for i in range(3):
        _save(session, symbol="AAPL", label="UP", day_offset=i)
    _save(session, symbol="AAPL", label="DOWN", day_offset=3)

    info = streaks_for(session, ["AAPL"])["AAPL"]
    assert info.current_label == "DOWN"
    assert info.streak == 1
    assert info.previous_label == "UP"
    assert info.flipped is True


def test_flip_with_multi_day_new_streak(session: Session) -> None:
    """UP, UP, DOWN, DOWN → today is DOWN day 2, prev UP."""
    _save(session, symbol="AAPL", label="UP", day_offset=0)
    _save(session, symbol="AAPL", label="UP", day_offset=1)
    _save(session, symbol="AAPL", label="DOWN", day_offset=2)
    _save(session, symbol="AAPL", label="DOWN", day_offset=3)

    info = streaks_for(session, ["AAPL"])["AAPL"]
    assert info.current_label == "DOWN"
    assert info.streak == 2
    assert info.previous_label == "UP"
    assert info.flipped is True


def test_flat_label_streak(session: Session) -> None:
    """FLAT is a normal label — same semantics as UP / DOWN."""
    for i in range(4):
        _save(session, symbol="MSFT", label="FLAT", day_offset=i)
    info = streaks_for(session, ["MSFT"])["MSFT"]
    assert info.current_label == "FLAT"
    assert info.streak == 4
    assert info.flipped is False


def test_flip_from_flat_to_up(session: Session) -> None:
    _save(session, symbol="MSFT", label="FLAT", day_offset=0)
    _save(session, symbol="MSFT", label="FLAT", day_offset=1)
    _save(session, symbol="MSFT", label="UP", day_offset=2)

    info = streaks_for(session, ["MSFT"])["MSFT"]
    assert info.current_label == "UP"
    assert info.streak == 1
    assert info.previous_label == "FLAT"
    assert info.flipped is True


def test_streak_uses_db_order_not_calendar(session: Session) -> None:
    """Day-7 written before day-5; the helper must order by
    prediction_date DESC regardless of insertion order."""
    _save(session, symbol="AAPL", label="UP", day_offset=7)
    _save(session, symbol="AAPL", label="DOWN", day_offset=5)
    _save(session, symbol="AAPL", label="UP", day_offset=6)

    info = streaks_for(session, ["AAPL"])["AAPL"]
    # Latest by date is day-7 (UP). Day-6 is UP too, day-5 is DOWN.
    # Streak should be 2 (day-7 and day-6 both UP), previous DOWN.
    assert info.current_label == "UP"
    assert info.streak == 2
    assert info.previous_label == "DOWN"
    assert info.flipped is True


# ---------------------------------------------------------------------------
# Multi-symbol behaviour
# ---------------------------------------------------------------------------

def test_independent_streaks_per_symbol(session: Session) -> None:
    _save(session, symbol="AAPL", label="UP", day_offset=0)
    _save(session, symbol="AAPL", label="UP", day_offset=1)

    _save(session, symbol="MSFT", label="DOWN", day_offset=0)
    _save(session, symbol="MSFT", label="UP", day_offset=1)

    out = streaks_for(session, ["AAPL", "MSFT"])
    assert out["AAPL"].streak == 2
    assert out["AAPL"].flipped is False

    assert out["MSFT"].streak == 1
    assert out["MSFT"].previous_label == "DOWN"
    assert out["MSFT"].flipped is True


def test_symbol_filter_excludes_others(session: Session) -> None:
    """If only AAPL is requested, MSFT must not appear even with data."""
    _save(session, symbol="AAPL", label="UP", day_offset=0)
    _save(session, symbol="MSFT", label="UP", day_offset=0)
    out = streaks_for(session, ["AAPL"])
    assert list(out) == ["AAPL"]


def test_dedupes_requested_symbols(session: Session) -> None:
    """Requesting AAPL twice shouldn't double-query or produce
    duplicated keys (set semantics)."""
    _save(session, symbol="AAPL", label="UP", day_offset=0)
    out = streaks_for(session, ["AAPL", "AAPL", "aapl"])
    # Only the exact-case "AAPL" matches DB rows; aapl is a different
    # symbol per SQL (case-sensitive). So just one entry.
    assert list(out) == ["AAPL"]


# ---------------------------------------------------------------------------
# Model-version filter
# ---------------------------------------------------------------------------

def test_model_version_filter_isolates_history(session: Session) -> None:
    """vader-test UP UP UP / logreg DOWN. When filtering to vader-test
    the streak is 3 UPs. When filtering to logreg, just 1 DOWN."""
    for i in range(3):
        _save(session, symbol="AAPL", label="UP", day_offset=i,
              model_version="vader-test")
    _save(session, symbol="AAPL", label="DOWN", day_offset=3,
          model_version="logreg")

    vader = streaks_for(session, ["AAPL"], model_version="vader-test")["AAPL"]
    assert vader.current_label == "UP"
    assert vader.streak == 3
    assert vader.flipped is False

    logreg = streaks_for(session, ["AAPL"], model_version="logreg")["AAPL"]
    assert logreg.current_label == "DOWN"
    assert logreg.streak == 1
    assert logreg.flipped is False  # no prior logreg row


def test_no_model_version_filter_mixes_models(session: Session) -> None:
    """When ``model_version`` is None the helper considers every model's
    rows — useful for a 'flipped across any model' view."""
    _save(session, symbol="AAPL", label="UP", day_offset=0,
          model_version="vader-test")
    _save(session, symbol="AAPL", label="DOWN", day_offset=1,
          model_version="logreg")

    info = streaks_for(session, ["AAPL"])["AAPL"]
    assert info.current_label == "DOWN"
    assert info.previous_label == "UP"
    assert info.flipped is True
