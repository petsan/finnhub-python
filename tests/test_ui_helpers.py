"""Tests for the pure data helpers behind the Streamlit UI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from finn_predictor.storage.models import NewsArticle, PredictionOutcome, Sector
from finn_predictor.storage.repo import (
    save_outcome,
    save_prediction,
    save_scores,
    upsert_articles,
)
from finn_predictor.predictor.explain import ArticleContribution
from finn_predictor.ui.app import (
    _escape_markdown,
    _format_headline_markdown,
    _parse_symbols,
    build_contribution_chart,
    contribution_chart_data,
    headlines_from_contributions,
    latest_market_prediction,
    latest_predictions,
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


# ---------------- latest_predictions & headlines_from_contributions ----------------


def test_latest_predictions_one_per_target_symbol_market_first(session) -> None:
    """Multiple targets across days; we expect one row per symbol, ^GSPC first."""
    save_prediction(
        session,
        make_prediction(
            target_symbol="XLK", prediction_date=D - timedelta(days=2), label="UP"
        ),
    )
    save_prediction(
        session,
        make_prediction(
            target_symbol="XLK", prediction_date=D, label="DOWN"
        ),
    )
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC", prediction_date=D - timedelta(days=1), label="UP"
        ),
    )
    save_prediction(
        session,
        make_prediction(
            target_symbol="XLE", prediction_date=D, label="FLAT"
        ),
    )

    rows = latest_predictions(session)
    syms = [r.target_symbol for r in rows]
    # ^GSPC first, then sectors alphabetically.
    assert syms == ["^GSPC", "XLE", "XLK"]
    # Each target's *latest* row is returned.
    assert {r.target_symbol: r.label for r in rows}["XLK"] == "DOWN"


def test_latest_predictions_empty(session) -> None:
    assert latest_predictions(session) == []


def test_headlines_from_contributions_preserves_order_and_fields(session) -> None:
    a1 = make_article(finnhub_id=1, headline="A", url="https://x/a", published_at=D)
    a2 = make_article(finnhub_id=2, headline="B", url="", published_at=D)
    from finn_predictor.storage.repo import upsert_articles
    upsert_articles(session, [a1, a2])
    persisted = session.query(type(a1)).order_by(type(a1).finnhub_id).all()

    contribs = [
        ArticleContribution(
            article=persisted[0], score=0.8, weight=1.0,
            contribution=0.6, supports_call=True,
        ),
        ArticleContribution(
            article=persisted[1], score=-0.4, weight=1.0,
            contribution=-0.3, supports_call=False,
        ),
    ]
    rows = headlines_from_contributions(contribs, limit=10)
    assert [r["headline"] for r in rows] == ["A", "B"]
    assert rows[0]["contribution"] == pytest.approx(0.6)
    assert rows[0]["supports_call"] is True
    assert rows[0]["url"] == "https://x/a"
    assert rows[1]["url"] == ""


def test_headlines_from_contributions_respects_limit(session) -> None:
    arts = [
        make_article(finnhub_id=i, headline=f"hl-{i}", published_at=D)
        for i in range(20)
    ]
    from finn_predictor.storage.repo import upsert_articles
    upsert_articles(session, arts)
    persisted = session.query(type(arts[0])).all()
    contribs = [
        ArticleContribution(
            article=a, score=0.5, weight=1.0,
            contribution=0.5 / 20, supports_call=True,
        )
        for a in persisted
    ]
    rows = headlines_from_contributions(contribs, limit=5)
    assert len(rows) == 5


# ---------------- contribution chart ----------------


def _make_contribs(session, items):
    """Insert articles + return ArticleContribution stubs in the same order."""
    from finn_predictor.storage.repo import upsert_articles

    arts = []
    for i, (headline, score, contrib) in enumerate(items):
        arts.append(
            make_article(
                finnhub_id=i + 1,
                headline=headline,
                symbol="AAPL" if i == 0 else "",
                published_at=D,
            )
        )
    upsert_articles(session, arts)
    persisted = (
        session.query(type(arts[0]))
        .order_by(type(arts[0]).finnhub_id)
        .all()
    )
    contribs = []
    for art, (_, score, contrib) in zip(persisted, items):
        contribs.append(
            ArticleContribution(
                article=art,
                score=score,
                weight=1.0,
                contribution=contrib,
                supports_call=contrib > 0,
            )
        )
    return contribs


def test_contribution_chart_data_empty_returns_typed_frame(session) -> None:
    df = contribution_chart_data([])
    assert df.empty
    for col in ("x", "contribution", "direction", "supports_call"):
        assert col in df.columns


def test_contribution_chart_data_sorted_ascending_by_contribution(session) -> None:
    contribs = _make_contribs(
        session,
        [
            ("middle", 0.0, 0.0),
            ("very negative", -0.9, -0.5),
            ("slightly positive", 0.2, 0.1),
            ("very positive", 0.95, 0.4),
        ],
    )
    df = contribution_chart_data(contribs)
    # Ascending order: most negative first.
    assert list(df["contribution"]) == sorted(c.contribution for c in contribs)
    assert list(df["x"]) == [0, 1, 2, 3]


def test_contribution_chart_data_direction_labels(session) -> None:
    contribs = _make_contribs(
        session,
        [
            ("neg", -0.5, -0.3),
            ("zero", 0.0, 0.0),
            ("pos", 0.5, 0.2),
        ],
    )
    df = contribution_chart_data(contribs)
    dirs = dict(zip(df["headline"], df["direction"]))
    assert dirs["neg"] == "negative"
    assert dirs["zero"] == "neutral"
    assert dirs["pos"] == "positive"


def test_contribution_chart_data_expands_ticker_when_session_passed(session) -> None:
    contribs = _make_contribs(session, [("Apple news", 0.6, 0.3)])
    df = contribution_chart_data(contribs, session=session)
    assert df.iloc[0]["company"] == "Apple Inc."


def test_contribution_chart_data_no_session_leaves_company_blank(session) -> None:
    contribs = _make_contribs(session, [("Apple news", 0.6, 0.3)])
    df = contribution_chart_data(contribs, session=None)
    assert df.iloc[0]["company"] == ""


def test_build_contribution_chart_returns_altair_chart() -> None:
    """Smoke test: the chart builder produces something Streamlit can render."""
    import altair as alt

    df = pd.DataFrame(
        [
            {
                "x": 0,
                "contribution": -0.3,
                "sentiment": -0.5,
                "headline": "A",
                "source": "S",
                "symbol": "",
                "company": "",
                "direction": "negative",
                "supports_call": False,
            },
            {
                "x": 1,
                "contribution": 0.3,
                "sentiment": 0.5,
                "headline": "B",
                "source": "S",
                "symbol": "",
                "company": "",
                "direction": "positive",
                "supports_call": True,
            },
        ]
    )
    chart = build_contribution_chart(df, title="t")
    # Round-trip through Vega-Lite JSON to confirm the spec is valid.
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["field"] == "x"
    assert spec["encoding"]["y"]["field"] == "contribution"
    assert spec["encoding"]["color"]["field"] == "direction"
    # Tooltip lists must include headline & contribution.
    tooltip_fields = {t["field"] for t in spec["encoding"]["tooltip"]}
    assert "headline" in tooltip_fields
    assert "contribution" in tooltip_fields


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


def test_format_headline_markdown_supports_call_marker() -> None:
    """🟢 when the article supports the Call, 🔴 when it opposes."""
    base = {
        "headline": "h",
        "url": "https://x/y",
        "source": "",
        "symbol": "*",
        "published_at": None,
        "sentiment": 0.5,
        "contribution": 0.123,
    }
    pro = _format_headline_markdown({**base, "supports_call": True})
    con = _format_headline_markdown({**base, "supports_call": False})
    assert pro.startswith("🟢 ")
    assert con.startswith("🔴 ")
    # Contribution shows in both, signed to 3dp.
    assert "contrib +0.123" in pro
    assert "contrib +0.123" in con


def test_format_headline_markdown_omits_marker_when_supports_unknown() -> None:
    """Legacy rows (no supports_call/contribution) render without markers."""
    line = _format_headline_markdown(
        {
            "headline": "h",
            "url": "https://x/y",
            "source": "",
            "symbol": "*",
            "published_at": None,
            "sentiment": 0.5,
        }
    )
    assert not line.startswith("🟢") and not line.startswith("🔴")
    assert "contrib" not in line


def test_format_headline_markdown_uses_company_name_when_present() -> None:
    """The headline row should show 'Apple Inc. (`AAPL`)' instead of '`AAPL`'."""
    line = _format_headline_markdown(
        {
            "headline": "Apple ships chip",
            "url": "https://x/y",
            "source": "",
            "symbol": "AAPL",
            "company": "Apple Inc.",
            "published_at": None,
            "sentiment": 0.5,
        }
    )
    assert "Apple Inc. (`AAPL`)" in line
    # Make sure we don't ALSO have a bare `AAPL` token elsewhere.
    assert "· `AAPL` ·" not in line


def test_format_headline_markdown_falls_back_when_no_company() -> None:
    line = _format_headline_markdown(
        {
            "headline": "h",
            "url": "https://x/y",
            "source": "",
            "symbol": "ZZZ",
            "published_at": None,
            "sentiment": 0.5,
        }
    )
    assert "`ZZZ`" in line


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
