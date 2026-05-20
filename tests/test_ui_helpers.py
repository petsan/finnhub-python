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
    _escape_markdown,
    _format_headline_markdown,
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


def test_recent_headlines_includes_url_field(session) -> None:
    """The Today tab links each headline to its source URL."""
    upsert_articles(
        session,
        [
            make_article(
                finnhub_id=1,
                headline="With URL",
                url="https://example.com/story",
                published_at=D,
            ),
            # Article with no URL — UI should fall back gracefully.
            make_article(finnhub_id=2, headline="No URL", url="", published_at=D),
        ],
    )
    rows = recent_headlines(session)
    by_headline = {r["headline"]: r for r in rows}
    assert by_headline["With URL"]["url"] == "https://example.com/story"
    assert by_headline["No URL"]["url"] == ""


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


# ---------------- headline markdown formatting ----------------


def test_escape_markdown_handles_brackets_and_backticks() -> None:
    assert _escape_markdown("a [b] c") == "a \\[b\\] c"
    assert _escape_markdown("use `pip`") == "use \\`pip\\`"
    assert _escape_markdown("back\\slash") == "back\\\\slash"


def test_format_headline_markdown_with_http_url() -> None:
    row = {
        "headline": "Markets close higher",
        "url": "https://example.com/a",
        "source": "Reuters",
        "symbol": "*",
        "published_at": datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc),
        "sentiment": 0.42,
    }
    line = _format_headline_markdown(row)
    assert "[Markets close higher](https://example.com/a)" in line
    assert "*Reuters*" in line
    assert "2026-05-19 14:30 UTC" in line
    assert "sentiment +0.42" in line


def test_format_headline_markdown_falls_back_when_url_missing() -> None:
    row = {
        "headline": "Plain",
        "url": "",
        "source": "",
        "symbol": "",
        "published_at": None,
        "sentiment": float("nan"),
    }
    line = _format_headline_markdown(row)
    assert "(" not in line.split("**Plain**")[0]  # no malformed link
    assert "**Plain**" in line  # bold fallback when no URL
    assert "sentiment —" in line  # NaN renders as em-dash


def test_format_headline_markdown_rejects_unsafe_url_schemes() -> None:
    """Only http(s) URLs become clickable links — anything else is plain bold."""
    row = {
        "headline": "Suspicious",
        "url": "javascript:alert(1)",
        "source": "",
        "symbol": "*",
        "published_at": None,
        "sentiment": float("nan"),
    }
    line = _format_headline_markdown(row)
    assert "javascript:" not in line
    assert "**Suspicious**" in line


def test_format_headline_markdown_escapes_brackets_in_headline() -> None:
    row = {
        "headline": "Stocks [really] surge",
        "url": "https://x.example/y",
        "source": "",
        "symbol": "*",
        "published_at": None,
        "sentiment": float("nan"),
    }
    line = _format_headline_markdown(row)
    # Bracketed text must be escaped so the Markdown parser doesn't
    # interpret it as an inline link.
    assert "[Stocks \\[really\\] surge](https://x.example/y)" in line


def test_format_headline_markdown_handles_empty_headline() -> None:
    row = {
        "headline": "",
        "url": "https://x.example/y",
        "source": "",
        "symbol": "*",
        "published_at": None,
        "sentiment": float("nan"),
    }
    line = _format_headline_markdown(row)
    assert "(no title)" in line


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
    from finn_predictor.ingestion.client import IngestionError

    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, patch(
        "finn_predictor.ui.app.run_daily_ingest", side_effect=RuntimeError("boom")
    ):
        client_instance = mock_cls.return_value
        # IngestionError wraps the original; original RuntimeError doesn't escape.
        with pytest.raises(IngestionError):
            run_ingestion_with_key(session, api_key="sk-x")
        client_instance.close.assert_called_once_with()


def test_run_ingestion_with_key_disables_env_proxy_trust(session) -> None:
    """Per deployment decision: bypass HTTPS_PROXY so an intercepting dev
    proxy doesn't break TLS verification."""
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
        client_instance = mock_cls.return_value
        client_instance._session.trust_env = True  # initial
        client_instance.close = lambda: None
        run_ingestion_with_key(session, api_key="sk-x")
        assert client_instance._session.trust_env is False


def test_run_ingestion_with_key_scrubs_leaked_token_from_message(session) -> None:
    """If anything in the pipeline leaks the api_key in an exception message,
    run_ingestion_with_key must redact it before re-raising."""
    from finn_predictor.ingestion.client import IngestionError, REDACTED

    leaky_msg = "boom token=sk-leak-9f7c in url"
    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, patch(
        "finn_predictor.ui.app.run_daily_ingest", side_effect=RuntimeError(leaky_msg)
    ):
        mock_cls.return_value.close = lambda: None
        with pytest.raises(IngestionError) as excinfo:
            run_ingestion_with_key(session, api_key="sk-leak-9f7c")
        msg = str(excinfo.value)
        assert "sk-leak-9f7c" not in msg
        assert REDACTED in msg
        # Context chain suppressed -> default tracebacks won't re-leak via __cause__.
        assert excinfo.value.__suppress_context__ is True


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
