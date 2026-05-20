"""Tests for the pure data helpers behind the Streamlit UI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from finn_predictor.storage.models import NewsArticle, PredictionOutcome, Sector
from finn_predictor.storage.repo import (
    save_outcome,
    save_prediction,
    save_scores,
    upsert_articles,
)
from finn_predictor.ui.app import (
    _parse_symbols,
    latest_market_prediction,
    prediction_history,
    recent_headlines,
    run_ingestion_with_key,
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


# ---------------- sidebar-supplied API key plumbing ----------------


def test_parse_symbols_handles_csv_variants() -> None:
    assert _parse_symbols("") == []
    assert _parse_symbols("   ") == []
    assert _parse_symbols("aapl") == ["AAPL"]
    assert _parse_symbols("AAPL, msft , , NVDA") == ["AAPL", "MSFT", "NVDA"]


def test_run_ingestion_with_key_rejects_empty_key(session) -> None:
    with pytest.raises(ValueError):
        run_ingestion_with_key(session, api_key="")
    with pytest.raises(ValueError):
        run_ingestion_with_key(session, api_key="   ")


def test_run_ingestion_with_key_builds_client_and_runs(session) -> None:
    """The helper must construct a Finnhub client with the in-session key
    and route through run_daily_ingest. We patch both the Client constructor
    and run_daily_ingest to keep the test offline.
    """
    fake_counts = {
        "general_news": 7,
        "company_news": 0,
        "market_prices": 1,
        "sector_prices": 11,
        "company_prices": 0,
        "scored": 7,
        "predictions": 1,
    }

    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, patch(
        "finn_predictor.ui.app.run_daily_ingest", return_value=fake_counts
    ) as mock_run:
        mock_cls.return_value.close = lambda: None
        out = run_ingestion_with_key(
            session,
            api_key="sk-session-only",
            rate_limit_per_minute=10,
            company_symbols=["AAPL", "MSFT"],
        )

    mock_cls.assert_called_once_with(api_key="sk-session-only")
    assert mock_run.called
    kwargs = mock_run.call_args.kwargs
    assert kwargs["company_symbols"] == ["AAPL", "MSFT"]
    assert out == fake_counts


def test_run_ingestion_with_key_closes_client_on_exception(session) -> None:
    """Even if run_daily_ingest raises, the key-bearing client must be closed."""
    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, patch(
        "finn_predictor.ui.app.run_daily_ingest", side_effect=RuntimeError("boom")
    ):
        client_instance = mock_cls.return_value
        with pytest.raises(RuntimeError, match="boom"):
            run_ingestion_with_key(session, api_key="sk-x")
        client_instance.close.assert_called_once_with()


def test_api_key_is_never_persisted_to_db(session) -> None:
    """No table should contain the API key after a successful ingestion."""
    api_key = "sk-leak-canary-9f7c"

    fake_counts = {
        "general_news": 0,
        "company_news": 0,
        "market_prices": 0,
        "sector_prices": 0,
        "company_prices": 0,
        "scored": 0,
        "predictions": 0,
    }

    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, patch(
        "finn_predictor.ui.app.run_daily_ingest", return_value=fake_counts
    ):
        mock_cls.return_value.close = lambda: None
        run_ingestion_with_key(session, api_key=api_key)

    # Walk every text column we have and assert the canary doesn't appear.
    cols_to_scan = [
        (NewsArticle, "headline"),
        (NewsArticle, "summary"),
        (NewsArticle, "url"),
        (NewsArticle, "source"),
    ]
    for model, attr in cols_to_scan:
        for row in session.query(model).all():
            assert api_key not in (getattr(row, attr) or "")
