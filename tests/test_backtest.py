"""Backtester tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.backtest import (
    BacktestReport,
    _label_matches,
    score_outcomes,
)
from finn_predictor.storage.repo import (
    save_prediction,
    upsert_price_bars,
)
from tests.conftest import make_prediction, make_price_bar


D = datetime(2026, 5, 19, tzinfo=timezone.utc)


def test_label_matches_up_down_flat() -> None:
    assert _label_matches("UP", 0.01) is True
    assert _label_matches("UP", -0.01) is False
    assert _label_matches("DOWN", -0.01) is True
    assert _label_matches("DOWN", 0.01) is False
    assert _label_matches("FLAT", 0.001) is True
    assert _label_matches("FLAT", 0.01) is False


def test_backtest_report_hit_rate() -> None:
    assert BacktestReport(total=0, scored=0, hits=0).hit_rate == 0.0
    assert BacktestReport(total=10, scored=4, hits=3).hit_rate == pytest.approx(0.75)


def test_score_outcomes_scores_predictions_with_two_bars(session) -> None:
    # Anchor + next-session bars for ^GSPC
    upsert_price_bars(
        session,
        [
            make_price_bar("^GSPC", D, close=100.0),
            make_price_bar("^GSPC", D + timedelta(days=1), close=101.0),
        ],
    )
    pred = save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )

    report = score_outcomes(session)
    assert report.scored == 1
    assert report.hits == 1
    session.refresh(pred)
    assert pred.outcome is not None
    assert pred.outcome.realised_return == pytest.approx(0.01)
    assert pred.outcome.hit is True


def test_score_outcomes_skips_when_no_next_bar(session) -> None:
    upsert_price_bars(session, [make_price_bar("^GSPC", D, close=100.0)])
    save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )
    report = score_outcomes(session)
    assert report.scored == 0


def test_score_outcomes_does_not_rescore(session) -> None:
    upsert_price_bars(
        session,
        [
            make_price_bar("^GSPC", D, close=100.0),
            make_price_bar("^GSPC", D + timedelta(days=1), close=99.0),
        ],
    )
    save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="DOWN"),
    )
    first = score_outcomes(session)
    second = score_outcomes(session)
    assert first.scored == 1
    assert second.scored == 0


def test_score_outcomes_handles_flat_correctly(session) -> None:
    upsert_price_bars(
        session,
        [
            make_price_bar("^GSPC", D, close=100.0),
            # +0.1% move counts as FLAT hit (threshold 0.25%)
            make_price_bar("^GSPC", D + timedelta(days=1), close=100.1),
        ],
    )
    save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="FLAT"),
    )
    report = score_outcomes(session)
    assert report.scored == 1
    assert report.hits == 1


def test_score_outcomes_skips_prediction_with_no_anchor(session) -> None:
    """If no PriceBar exists at-or-before the prediction date, skip silently."""
    save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )
    # Future bars only — anchor lookup returns None.
    upsert_price_bars(
        session,
        [make_price_bar("^GSPC", D + timedelta(days=2), close=99.0)],
    )
    report = score_outcomes(session)
    assert report.scored == 0


def test_score_outcomes_handles_zero_anchor_price(session) -> None:
    upsert_price_bars(
        session,
        [
            make_price_bar("^GSPC", D, close=0.0),
            make_price_bar("^GSPC", D + timedelta(days=1), close=10.0),
        ],
    )
    save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )
    report = score_outcomes(session)
    assert report.scored == 0  # division-by-zero guard
