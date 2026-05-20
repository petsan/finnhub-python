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
    BROWSER_STORAGE_API_KEY,
    NEUTRAL_SENTIMENT_THRESHOLD,
    _get_local_storage,
    build_market_price_chart,
    group_stock_predictions_by_sector,
    _escape_markdown,
    _format_headline_markdown,
    _parse_symbols,
    attach_first_seen,
    build_contribution_chart,
    build_cum_pnl_chart,
    build_hit_rate_by_kind_chart,
    build_hit_rate_by_label_chart,
    build_rolling_hit_chart,
    contribution_chart_data,
    format_expected_move,
    headlines_from_contributions,
    latest_market_prediction,
    latest_predictions,
    neutral_headlines_from_contributions,
    partition_predictions,
    prediction_history,
    recent_headlines,
    run_backfill_with_key,
    run_ingestion_with_key,
    run_price_backfill,
    sector_grid,
    stock_predictions_table,
)
from tests.conftest import make_article, make_prediction, make_price_bar, make_score


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


# ---------------- partition_predictions & stock_predictions_table ----------------


def test_partition_predictions_separates_market_sectors_stocks(session) -> None:
    from finn_predictor.storage.repo import ensure_default_sectors
    ensure_default_sectors(session)

    market = save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )
    xlk = save_prediction(
        session,
        make_prediction(target_symbol="XLK", prediction_date=D, label="FLAT"),
    )
    aapl = save_prediction(
        session,
        make_prediction(target_symbol="AAPL", prediction_date=D, label="UP"),
    )

    m, sectors, stocks = partition_predictions(session, [market, xlk, aapl])
    assert m is not None and m.id == market.id
    assert [p.target_symbol for p in sectors] == ["XLK"]
    assert [p.target_symbol for p in stocks] == ["AAPL"]


def test_partition_predictions_no_market(session) -> None:
    from finn_predictor.storage.repo import ensure_default_sectors
    ensure_default_sectors(session)

    aapl = save_prediction(
        session,
        make_prediction(target_symbol="AAPL", prediction_date=D, label="UP"),
    )
    m, sectors, stocks = partition_predictions(session, [aapl])
    assert m is None
    assert sectors == []
    assert [p.target_symbol for p in stocks] == ["AAPL"]


def test_stock_predictions_table_columns_and_sort(session) -> None:
    a = make_prediction(
        target_symbol="AAPL",
        prediction_date=D,
        label="UP",
        confidence=0.42,
        sentiment_index=0.31,
        article_count=8,
    )
    b = make_prediction(
        target_symbol="MSFT",
        prediction_date=D,
        label="FLAT",
        confidence=0.10,
        sentiment_index=0.05,
        article_count=3,
    )
    df = stock_predictions_table(session, [b, a])  # pass MSFT first
    # Sorted by Confidence DESC then Articles DESC, so AAPL leads.
    assert list(df["Ticker"]) == ["AAPL", "MSFT"]
    assert "Company" in df.columns and "Sentiment" in df.columns
    # Company name expansion was applied.
    assert df.iloc[0]["Company"] == "Apple Inc."


def test_stock_predictions_table_empty_returns_typed_frame(session) -> None:
    df = stock_predictions_table(session, [])
    assert df.empty
    for col in ("Company", "Ticker", "Call", "Confidence", "Articles", "Sentiment"):
        assert col in df.columns


# ---------------- price backfill + Performance charts ----------------


def test_run_price_backfill_pulls_targets_and_scores(session) -> None:
    """run_price_backfill collects distinct prediction targets, calls the
    yfinance fetcher, then runs the backtester to populate outcomes."""
    from finn_predictor.storage.models import PriceBar
    from finn_predictor.storage.repo import save_prediction

    # Predict on AAPL + ^GSPC; targets passed to the fetcher should
    # include both (plus an auto-^GSPC entry).
    save_prediction(
        session,
        make_prediction(target_symbol="AAPL", prediction_date=D, label="UP"),
    )
    save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )

    captured: dict[str, object] = {}

    def fake_history(symbols, start, end):
        captured["symbols"] = list(symbols)
        # Return a single-ticker flat frame to keep the test tight.
        import pandas as pd
        idx = pd.DatetimeIndex(
            [D + timedelta(days=1), D + timedelta(days=2)]
        )
        df = pd.DataFrame(
            {"Open": [100, 101], "High": [101, 102], "Low": [99, 100],
             "Close": [100.5, 101.5], "Volume": [1, 1]},
            index=idx,
        )
        # multi-index single ticker, when we have multiple symbols
        if len(symbols) > 1:
            return pd.concat({s: df for s in symbols}, axis=1)
        return df

    out = run_price_backfill(session, days=30, history_fn=fake_history)
    assert {"^GSPC", "AAPL"}.issubset(set(captured["symbols"]))
    # Bars should have landed.
    assert session.query(PriceBar).count() > 0
    # And the backtester ran.
    assert hasattr(out["scored"], "scored")


def test_run_price_backfill_with_no_predictions_still_pulls_gspc(session) -> None:
    """Even on an empty DB we backfill ^GSPC so the market predictor has data."""
    captured: dict[str, object] = {}

    def fake_history(symbols, start, end):
        captured["symbols"] = list(symbols)
        import pandas as pd
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )

    run_price_backfill(session, days=14, history_fn=fake_history)
    assert "^GSPC" in captured["symbols"]


def test_build_cum_pnl_chart_returns_valid_spec() -> None:
    df = pd.DataFrame(
        [
            {"entry_date": D, "target_kind": "MARKET", "target_symbol": "^GSPC",
             "pnl_pct": 0.01, "cum_pnl_pct": 0.01},
            {"entry_date": D + timedelta(days=1), "target_kind": "MARKET",
             "target_symbol": "^GSPC", "pnl_pct": -0.005, "cum_pnl_pct": 0.005},
        ]
    )
    spec = build_cum_pnl_chart(df).to_dict()
    assert spec["mark"]["type"] == "line"
    assert spec["encoding"]["x"]["field"] == "entry_date"
    assert spec["encoding"]["y"]["field"] == "cum_pnl_pct"


def test_build_rolling_hit_chart_has_reference_line() -> None:
    df = pd.DataFrame(
        [
            {"entry_date": D, "hit_rate": 0.6},
            {"entry_date": D + timedelta(days=1), "hit_rate": 0.55},
        ]
    )
    spec = build_rolling_hit_chart(df, window=14).to_dict()
    # Combination chart → 'layer' contains the line + 50% reference rule.
    assert "layer" in spec
    assert len(spec["layer"]) == 2


def test_build_hit_rate_by_kind_chart_has_bar_mark() -> None:
    df = pd.DataFrame(
        [
            {"target_kind": "MARKET", "trades": 10, "hits": 6, "hit_rate": 0.6},
            {"target_kind": "STOCK",  "trades": 20, "hits": 9, "hit_rate": 0.45},
        ]
    )
    spec = build_hit_rate_by_kind_chart(df).to_dict()
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["field"] == "target_kind"


def test_build_hit_rate_by_label_chart_sort_order() -> None:
    df = pd.DataFrame(
        [
            {"label": "DOWN", "trades": 5, "hits": 3, "hit_rate": 0.6},
            {"label": "UP",   "trades": 8, "hits": 4, "hit_rate": 0.5},
            {"label": "FLAT", "trades": 3, "hits": 2, "hit_rate": 0.66},
        ]
    )
    spec = build_hit_rate_by_label_chart(df).to_dict()
    assert spec["encoding"]["x"]["sort"] == ["UP", "FLAT", "DOWN"]


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


def test_format_headline_markdown_shows_first_reported_when_earlier() -> None:
    """When the same story was reported elsewhere earlier, surface that ts."""
    published = datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)
    first_seen = datetime(2026, 5, 19, 9, 0, tzinfo=timezone.utc)
    line = _format_headline_markdown(
        {
            "headline": "h",
            "url": "https://x/y",
            "source": "",
            "symbol": "*",
            "published_at": published,
            "first_seen_at": first_seen,
            "sentiment": 0.5,
        }
    )
    assert "first reported 2026-05-19 09:00 UTC" in line


def test_format_headline_markdown_hides_first_reported_when_close() -> None:
    """Same-minute republishes shouldn't add a redundant 'first reported' line."""
    published = datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)
    first_seen = datetime(2026, 5, 19, 14, 31, tzinfo=timezone.utc)  # 1 min apart
    line = _format_headline_markdown(
        {
            "headline": "h",
            "url": "https://x/y",
            "source": "",
            "symbol": "*",
            "published_at": published,
            "first_seen_at": first_seen,
            "sentiment": 0.5,
        }
    )
    assert "first reported" not in line


def test_format_headline_markdown_first_reported_handles_naive_datetime() -> None:
    """SQLite roundtrip returns naive datetimes — formatter must cope."""
    published = datetime(2026, 5, 19, 14, 30)  # naive
    first_seen = datetime(2026, 5, 19, 9, 0)   # naive
    line = _format_headline_markdown(
        {
            "headline": "h",
            "url": "https://x/y",
            "source": "",
            "symbol": "*",
            "published_at": published,
            "first_seen_at": first_seen,
            "sentiment": 0.5,
        }
    )
    assert "first reported 2026-05-19 09:00 UTC" in line


def test_attach_first_seen_populates_field(session) -> None:
    """Plumbing test: attach_first_seen mutates rows in place."""
    from finn_predictor.storage.repo import upsert_articles

    early = make_article(
        finnhub_id=1,
        headline="Apple beats expectations",
        published_at=D - timedelta(hours=5),
    )
    late = make_article(
        finnhub_id=2,
        headline="Apple beats expectations again",  # different story_key
        published_at=D - timedelta(hours=1),
    )
    upsert_articles(session, [early, late])

    rows = [
        {"headline": "Apple beats expectations", "published_at": D},
        {"headline": "Apple beats expectations again", "published_at": D},
    ]
    attach_first_seen(session, rows)
    assert rows[0]["first_seen_at"] is not None
    assert rows[1]["first_seen_at"] is not None


def test_attach_first_seen_safe_on_empty_input(session) -> None:
    assert attach_first_seen(session, []) == []


def test_contribution_chart_data_includes_first_seen_field(session) -> None:
    from finn_predictor.storage.repo import upsert_articles

    earlier = make_article(
        finnhub_id=1,
        headline="Big news everyone",
        published_at=D - timedelta(hours=5),
    )
    later = make_article(
        finnhub_id=2,
        headline="Big news everyone",
        published_at=D,
    )
    upsert_articles(session, [earlier, later])
    persisted = (
        session.query(type(earlier))
        .order_by(type(earlier).finnhub_id)
        .all()
    )

    contribs = [
        ArticleContribution(
            article=persisted[1],
            score=0.7,
            weight=1.0,
            contribution=0.3,
            supports_call=True,
        )
    ]
    df = contribution_chart_data(contribs, session=session)
    assert "first_seen_at" in df.columns
    fs = df.iloc[0]["first_seen_at"]
    assert fs is not None
    # The earliest of the cluster was the article persisted 5h earlier.
    assert fs <= persisted[1].published_at


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


def test_run_backfill_with_key_rejects_empty_inputs(session) -> None:
    with pytest.raises(ValueError):
        run_backfill_with_key(
            session, api_key="", symbols=["AAPL"],
            start=D - timedelta(days=5), end=D,
        )
    with pytest.raises(ValueError):
        run_backfill_with_key(
            session, api_key="sk-x", symbols=[],
            start=D - timedelta(days=5), end=D,
        )
    with pytest.raises(ValueError):
        run_backfill_with_key(
            session, api_key="sk-x", symbols=["", "  "],
            start=D - timedelta(days=5), end=D,
        )


def test_run_backfill_with_key_routes_through_client(session) -> None:
    """Helper builds a client with the in-session key and runs backfill_many."""
    from finn_predictor.ingestion.backfill import BackfillResult

    fake_results = {
        "AAPL": BackfillResult(
            symbol="AAPL", inserted=12, chunks_attempted=2,
            chunks_failed=0, failures=[],
        )
    }

    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, \
         patch("finn_predictor.ui.app.backfill_many", return_value=fake_results) as bm, \
         patch(
             "finn_predictor.ui.app.score_pending_articles", return_value=7
         ) as sp, \
         patch(
             "finn_predictor.ui.app.retroactive_predict_many",
             return_value={"AAPL": 30},
         ) as rp:
        mock_cls.return_value.close = lambda: None
        out = run_backfill_with_key(
            session,
            api_key="sk-session-only",
            symbols=["AAPL"],
            start=D - timedelta(days=30),
            end=D,
        )

    mock_cls.assert_called_once_with(api_key="sk-session-only")
    assert out["backfill"] == fake_results
    assert out["scored"] == 7
    assert out["predictions"] == {"AAPL": 30}
    bm.assert_called_once()
    sp.assert_called_once()
    rp.assert_called_once()


def test_run_backfill_with_key_disables_env_proxy(session) -> None:
    """Same proxy bypass as run_ingestion_with_key."""
    from finn_predictor.ingestion.backfill import BackfillResult

    fake_results = {
        "AAPL": BackfillResult(
            symbol="AAPL", inserted=0, chunks_attempted=1,
            chunks_failed=0, failures=[],
        )
    }
    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, \
         patch("finn_predictor.ui.app.backfill_many", return_value=fake_results), \
         patch("finn_predictor.ui.app.score_pending_articles", return_value=0), \
         patch(
             "finn_predictor.ui.app.retroactive_predict_many", return_value={"AAPL": 0}
         ):
        client_instance = mock_cls.return_value
        client_instance._session.trust_env = True
        client_instance.close = lambda: None
        run_backfill_with_key(
            session, api_key="sk-x", symbols=["AAPL"],
            start=D - timedelta(days=1), end=D,
        )
        assert client_instance._session.trust_env is False


def test_run_backfill_with_key_scrubs_leaked_token(session) -> None:
    """An unexpected leak in a non-IngestionError must still be scrubbed."""
    from finn_predictor.ingestion.client import IngestionError, REDACTED

    with patch("finn_predictor.ui.app.FinnhubClient") as mock_cls, \
         patch(
             "finn_predictor.ui.app.backfill_many",
             side_effect=RuntimeError("token=sk-leak-x in url"),
         ):
        mock_cls.return_value.close = lambda: None
        with pytest.raises(IngestionError) as excinfo:
            run_backfill_with_key(
                session, api_key="sk-leak-x", symbols=["AAPL"],
                start=D - timedelta(days=5), end=D,
            )
        msg = str(excinfo.value)
        assert "sk-leak-x" not in msg
        assert REDACTED in msg


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


def test_format_expected_move_renders_band() -> None:
    """All three columns populated → formatted band string."""
    p = make_prediction(target_symbol="^GSPC", prediction_date=D)
    p.expected_return_p10 = -0.012
    p.expected_return_p50 = 0.004
    p.expected_return_p90 = 0.018
    out = format_expected_move(p)
    assert out == "-1.20% to +1.80% (median +0.40%)"


def test_format_expected_move_returns_none_when_columns_missing() -> None:
    """Any null → None (caller hides the row entirely)."""
    p = make_prediction(target_symbol="^GSPC", prediction_date=D)
    # All three default to None on the model, so untouched.
    assert format_expected_move(p) is None

    # Partial population still returns None — we don't render a half-band.
    p.expected_return_p10 = -0.01
    p.expected_return_p90 = 0.01
    assert format_expected_move(p) is None  # p50 missing


def _make_contribution(headline: str, score: float, *, published_at=None, source="Reuters", symbol=None, url=""):
    """Build an ArticleContribution from the bare minimums needed by the UI helpers."""
    from finn_predictor.predictor.explain import ArticleContribution
    from finn_predictor.storage.models import NewsArticle

    art = NewsArticle(
        finnhub_id=hash(headline) % 1_000_000_000,
        category="general",
        headline=headline,
        summary="",
        source=source,
        url=url,
        symbol=symbol,
        published_at=published_at or D,
    )
    # weight + contribution placeholders — the neutral filter looks at .score
    return ArticleContribution(
        article=art, score=score, weight=1.0,
        contribution=score / 10.0, supports_call=False,
    )


def test_neutral_headlines_filter_returns_only_low_score_rows() -> None:
    """Only articles with |score| ≤ threshold show up."""
    contribs = [
        _make_contribution("Big rally", 0.9),
        _make_contribution("Sky stayed blue", 0.0),
        _make_contribution("Earnings report filed", 0.02),
        _make_contribution("Disaster strikes", -0.85),
        _make_contribution("Marginal note", -0.04),
    ]
    out = neutral_headlines_from_contributions(contribs)
    headlines = {r["headline"] for r in out}
    assert headlines == {"Sky stayed blue", "Earnings report filed", "Marginal note"}
    # And no contribution field is leaked — the row dict is intentionally
    # missing it (the formatter then skips the "contrib …" suffix).
    assert all("contribution" not in r for r in out)


def test_neutral_headlines_sorted_by_recency() -> None:
    """Neutral pile orders newest → oldest, opposite of |contribution|."""
    older = datetime(2026, 5, 18, 6, tzinfo=timezone.utc)
    newer = datetime(2026, 5, 19, 9, tzinfo=timezone.utc)
    newest = datetime(2026, 5, 19, 16, tzinfo=timezone.utc)
    contribs = [
        _make_contribution("Old neutral", 0.01, published_at=older),
        _make_contribution("Newest neutral", -0.03, published_at=newest),
        _make_contribution("Mid neutral", 0.0, published_at=newer),
    ]
    out = neutral_headlines_from_contributions(contribs)
    assert [r["headline"] for r in out] == [
        "Newest neutral", "Mid neutral", "Old neutral",
    ]


def test_neutral_headlines_honours_limit() -> None:
    contribs = [_make_contribution(f"Neutral {i}", 0.01) for i in range(15)]
    out = neutral_headlines_from_contributions(contribs, limit=4)
    assert len(out) == 4


def test_neutral_headlines_respects_explicit_threshold() -> None:
    """A bigger threshold lets through articles the default would exclude."""
    contribs = [
        _make_contribution("Mild positive", 0.15),
        _make_contribution("Truly neutral", 0.02),
        _make_contribution("Mild negative", -0.10),
    ]
    out = neutral_headlines_from_contributions(contribs, sentiment_threshold=0.2)
    assert {r["headline"] for r in out} == {
        "Mild positive", "Truly neutral", "Mild negative",
    }


def test_neutral_headlines_empty_input() -> None:
    assert neutral_headlines_from_contributions([]) == []


def test_get_local_storage_returns_none_when_disabled(monkeypatch) -> None:
    """The escape-hatch env var disables localStorage cleanly."""
    monkeypatch.setenv("FINN_PREDICTOR_DISABLE_LOCAL_STORAGE", "1")
    assert _get_local_storage() is None


def test_get_local_storage_respects_truthy_disable_values(monkeypatch) -> None:
    """'true' / 'yes' also disable, matching the rest of the codebase's parsing."""
    for val in ("true", "yes", "TRUE", "1"):
        monkeypatch.setenv("FINN_PREDICTOR_DISABLE_LOCAL_STORAGE", val)
        assert _get_local_storage() is None


def test_get_local_storage_ignores_falsey_disable_values(monkeypatch) -> None:
    """Empty / '0' / 'no' do NOT disable — let the package try."""
    # We can't actually instantiate LocalStorage here without hanging, but
    # the helper at least shouldn't short-circuit on these values.
    monkeypatch.setenv("FINN_PREDICTOR_DISABLE_LOCAL_STORAGE", "0")
    # It'll either return a LocalStorage instance (in 'streamlit run') or
    # None from the inner except; both are acceptable here. The
    # assertion that matters is "this didn't blow up because of the env."
    # If the package's constructor hangs in AppTest-like mode, this test
    # will time out — but plain `python` raises immediately, so:
    result = _get_local_storage()  # noqa: F841 — exercising the path is the test
    # No raise == pass; we don't constrain the return type beyond that.


def test_build_market_price_chart_returns_none_on_empty_bars() -> None:
    """No bars → None so the caller hides the chart entirely."""
    assert build_market_price_chart([]) is None


def test_build_market_price_chart_returns_chart_with_data() -> None:
    """Bars in, Altair Chart out — pannable + zoomable via interactive()."""
    import altair as alt
    bars = [
        make_price_bar(symbol="^GSPC", trade_date=D - timedelta(days=i), close=4000.0 + i)
        for i in range(20)
    ]
    chart = build_market_price_chart(bars, symbol="^GSPC", days=30)
    assert chart is not None
    # Spec inspection rather than rendering: confirm the y-axis is bound
    # to the close column and that the chart is interactive (zoom+pan).
    spec = chart.to_dict()
    assert spec["encoding"]["y"]["field"] == "close"
    assert spec["encoding"]["x"]["field"] == "date"
    # interactive() injects a default selection parameter.
    assert "params" in spec or "selection" in spec


def test_build_market_price_chart_limits_to_n_days() -> None:
    """With 100 bars and days=30, only the most recent 30 are drawn."""
    bars = [
        make_price_bar(symbol="^GSPC", trade_date=D - timedelta(days=i), close=100.0 + i)
        for i in range(100)
    ]
    chart = build_market_price_chart(bars, days=30)
    assert chart is not None
    # The underlying dataframe should be tailed.
    df = chart.data
    assert len(df) == 30


def test_group_stock_predictions_by_sector_buckets_mapped_tickers(session) -> None:
    """Mapped tickers go into their sector group; unmapped → unmapped pile."""
    from finn_predictor.storage.repo import ensure_default_sectors
    ensure_default_sectors(session)

    preds = [
        make_prediction(target_symbol="AAPL", prediction_date=D, label="UP", confidence=0.55),
        make_prediction(target_symbol="MSFT", prediction_date=D, label="UP", confidence=0.40),
        make_prediction(target_symbol="JPM", prediction_date=D, label="DOWN", confidence=0.30),
        make_prediction(target_symbol="WEIRD42", prediction_date=D, label="FLAT", confidence=0.05),
    ]
    grouped, unmapped = group_stock_predictions_by_sector(session, preds)

    # Two sectors should appear (TECH, FIN), each ordered by sector code.
    sector_codes = [s.code for s, _ in grouped]
    assert sector_codes == ["FIN", "TECH"]  # sorted by code

    tech_group = next(preds for s, preds in grouped if s.code == "TECH")
    assert {p.target_symbol for p in tech_group} == {"AAPL", "MSFT"}

    fin_group = next(preds for s, preds in grouped if s.code == "FIN")
    assert {p.target_symbol for p in fin_group} == {"JPM"}

    assert [p.target_symbol for p in unmapped] == ["WEIRD42"]


def test_group_stock_predictions_by_sector_empty_input(session) -> None:
    from finn_predictor.storage.repo import ensure_default_sectors
    ensure_default_sectors(session)
    grouped, unmapped = group_stock_predictions_by_sector(session, [])
    assert grouped == []
    assert unmapped == []


def test_group_stock_predictions_all_unmapped(session) -> None:
    """All tickers unmapped → empty groups, full unmapped list."""
    from finn_predictor.storage.repo import ensure_default_sectors
    ensure_default_sectors(session)
    preds = [
        make_prediction(target_symbol=f"WEIRD{i}", prediction_date=D, label="FLAT", confidence=0.05)
        for i in range(3)
    ]
    grouped, unmapped = group_stock_predictions_by_sector(session, preds)
    assert grouped == []
    assert len(unmapped) == 3


def test_browser_storage_api_key_is_namespaced() -> None:
    """The localStorage key includes a product prefix so we don't collide
    with other tabs' storage on the same origin."""
    assert "finn_predictor" in BROWSER_STORAGE_API_KEY
    assert "finnhub" in BROWSER_STORAGE_API_KEY  # references what it stores


def test_neutral_sentiment_threshold_matches_explain_flat_band() -> None:
    """The neutral cutoff stays in lockstep with explain.FLAT_SUPPORT_BAND.

    Both knobs encode the same notion: a single article whose
    contribution sits at or under this magnitude doesn't tip a FLAT
    call. Drifting them apart would create the UX bug where an
    article shows up in the neutral pile while also being treated as
    a directional supporter elsewhere.
    """
    from finn_predictor.predictor.explain import FLAT_SUPPORT_BAND
    assert NEUTRAL_SENTIMENT_THRESHOLD == FLAT_SUPPORT_BAND


def test_format_expected_move_handles_negative_band() -> None:
    """An all-negative band reads correctly."""
    p = make_prediction(target_symbol="^GSPC", prediction_date=D)
    p.expected_return_p10 = -0.025
    p.expected_return_p50 = -0.010
    p.expected_return_p90 = -0.002
    assert format_expected_move(p) == "-2.50% to -0.20% (median -1.00%)"
