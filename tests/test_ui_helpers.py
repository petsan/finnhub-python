"""Tests for the pure data helpers behind the Streamlit UI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from finn_predictor.storage.models import PredictionOutcome, Sector
from finn_predictor.storage.repo import (
    save_outcome,
    save_prediction,
    save_scores,
    upsert_articles,
)
from finn_predictor.ui.app import (
    latest_market_prediction,
    prediction_history,
    recent_headlines,
    sector_grid,
)
from tests.conftest import make_article, make_prediction, make_score


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


def test_latest_market_prediction_returns_most_recent(session) -> None:
    older = save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D - timedelta(days=2)),
    )
    newer = save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="DOWN"),
    )
    out = latest_market_prediction(session)
    assert out is not None
    assert out.id == newer.id != older.id


def test_latest_market_prediction_none_when_empty(session) -> None:
    assert latest_market_prediction(session) is None


def test_recent_headlines_returns_score_when_available(session) -> None:
    upsert_articles(
        session,
        [make_article(finnhub_id=i, headline=f"hl-{i}", published_at=D + timedelta(minutes=i)) for i in range(3)],
    )
    arts = session.query(type(make_article(finnhub_id=999))).all()
    save_scores(
        session,
        [make_score(arts[0].id, 0.4, model_version="vader-test")],
    )
    rows = recent_headlines(session, limit=10, model_version="vader-test")
    # Newest first.
    assert rows[0]["headline"] == "hl-2"
    by_headline = {r["headline"]: r for r in rows}
    assert by_headline["hl-0"]["sentiment"] == 0.4
    # NaN for unscored — represented as float('nan').
    assert by_headline["hl-1"]["sentiment"] != by_headline["hl-1"]["sentiment"]  # NaN != NaN


def test_recent_headlines_handles_no_articles(session) -> None:
    assert recent_headlines(session) == []


def test_prediction_history_returns_frame_with_outcomes(session) -> None:
    pred = save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )
    save_outcome(
        session,
        PredictionOutcome(prediction_id=pred.id, realised_return=0.012, hit=True),
    )
    df = prediction_history(session, "^GSPC")
    assert not df.empty
    assert df.iloc[0]["realised_return"] == 0.012
    # pandas may upcast bools to numpy.bool_; coerce before identity comparison.
    assert bool(df.iloc[0]["hit"]) is True


def test_prediction_history_empty_returns_typed_frame(session) -> None:
    df = prediction_history(session, "^GSPC")
    assert df.empty
    assert "label" in df.columns


def test_sector_grid_one_row_per_sector(session) -> None:
    sectors = [
        Sector(code="TECH", name="Tech", etf_symbol="XLK"),
        Sector(code="ENERGY", name="Energy", etf_symbol="XLE"),
    ]
    session.add_all(sectors)
    session.commit()

    # Only TECH has a prediction.
    save_prediction(
        session,
        make_prediction(target_symbol="XLK", prediction_date=D, label="UP"),
    )

    df = sector_grid(session, sectors)
    assert len(df) == 2
    tech = df[df["etf"] == "XLK"].iloc[0]
    energy = df[df["etf"] == "XLE"].iloc[0]
    assert tech["label"] == "UP"
    assert energy["label"] == "—"
