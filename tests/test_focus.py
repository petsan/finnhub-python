"""Tests for the focus module + related-entities repo helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from finn_predictor.ingestion.client import IngestionError
from finn_predictor.predictor.focus import (
    CompanyFocus,
    EventFocus,
    SectorFocus,
    compose_company_focus,
    compose_event_focus,
    compose_sector_focus,
    refresh_company_relationships,
    refresh_sector_constituents,
)
from finn_predictor.storage.models import Sector
from finn_predictor.storage.repo import (
    ensure_default_sectors,
    related_entities_for,
    save_prediction,
    save_scores,
    upsert_articles,
    upsert_related_entity,
)
from tests.conftest import make_article, make_prediction, make_score


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


# ---------------- repo: upsert_related_entity ----------------


def test_upsert_related_entity_inserts_and_updates(session) -> None:
    a = upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="PEER", rank=2,
    )
    assert a.rank == 2
    b = upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="PEER", rank=5, metadata_text="updated",
    )
    assert b.id == a.id  # same row
    assert b.rank == 5
    assert b.metadata_text == "updated"


def test_upsert_related_entity_rejects_unknown_relationship(session) -> None:
    with pytest.raises(ValueError):
        upsert_related_entity(
            session, source_symbol="AAPL", related_symbol="MSFT",
            relationship="FRIEND",
        )


def test_related_entities_for_orders_by_rank(session) -> None:
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="GOOG",
        relationship="PEER", rank=2,
    )
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="PEER", rank=0,
    )
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="META",
        relationship="PEER", rank=1,
    )
    rows = related_entities_for(session, "AAPL", relationship="PEER")
    assert [r.related_symbol for r in rows] == ["MSFT", "META", "GOOG"]


# ---------------- compose_company_focus ----------------


def test_compose_company_focus_basic(session) -> None:
    save_prediction(
        session,
        make_prediction(target_symbol="AAPL", prediction_date=D, label="UP"),
    )
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="PEER", rank=0,
    )
    save_prediction(
        session,
        make_prediction(target_symbol="MSFT", prediction_date=D, label="FLAT"),
    )

    bundle = compose_company_focus(session, "AAPL")
    assert isinstance(bundle, CompanyFocus)
    assert bundle.symbol == "AAPL"
    assert bundle.own_prediction is not None
    assert bundle.own_prediction.label == "UP"
    assert [p.related_symbol for p in bundle.peers] == ["MSFT"]
    assert bundle.peers[0].prediction is not None
    assert bundle.peers[0].prediction.label == "FLAT"


def test_compose_company_focus_no_data(session) -> None:
    bundle = compose_company_focus(session, "ZZZZ")
    assert bundle.symbol == "ZZZZ"
    assert bundle.own_prediction is None
    assert bundle.peers == []
    assert bundle.recent_articles == []


def test_compose_company_focus_includes_supply_chain(session) -> None:
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="TSM",
        relationship="SUPPLIER", rank=0, metadata_text="Taiwan Semi",
    )
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="T",
        relationship="CUSTOMER", rank=0,
    )
    bundle = compose_company_focus(session, "AAPL")
    assert [s.related_symbol for s in bundle.suppliers] == ["TSM"]
    assert [c.related_symbol for c in bundle.customers] == ["T"]


def test_compose_company_focus_rejects_empty(session) -> None:
    with pytest.raises(ValueError):
        compose_company_focus(session, "")


def test_compose_company_focus_recent_articles_filter(session) -> None:
    """Articles for the subject + peers (in the lookback window) appear."""
    upsert_articles(
        session,
        [
            make_article(finnhub_id=1, symbol="AAPL", headline="A1",
                          published_at=D - timedelta(days=2)),
            make_article(finnhub_id=2, symbol="MSFT", headline="A2",
                          published_at=D - timedelta(days=3)),
            # outside ticker universe (peers don't include GOOG)
            make_article(finnhub_id=3, symbol="GOOG", headline="A3",
                          published_at=D - timedelta(days=1)),
        ],
    )
    upsert_related_entity(
        session, source_symbol="AAPL", related_symbol="MSFT",
        relationship="PEER",
    )
    bundle = compose_company_focus(session, "AAPL", article_days=30)
    headlines = {row["headline"] for row in bundle.recent_articles}
    assert headlines == {"A1", "A2"}


# ---------------- compose_sector_focus ----------------


def test_compose_sector_focus_returns_constituents(session) -> None:
    ensure_default_sectors(session)
    upsert_related_entity(
        session, source_symbol="XLK", related_symbol="AAPL",
        relationship="ETF_HOLDING", rank=0,
    )
    upsert_related_entity(
        session, source_symbol="XLK", related_symbol="MSFT",
        relationship="ETF_HOLDING", rank=1,
    )
    save_prediction(
        session,
        make_prediction(target_symbol="XLK", prediction_date=D, label="UP"),
    )

    bundle = compose_sector_focus(session, "TECH")
    assert isinstance(bundle, SectorFocus)
    assert bundle.etf_symbol == "XLK"
    assert bundle.own_prediction is not None and bundle.own_prediction.label == "UP"
    assert [c.related_symbol for c in bundle.constituents] == ["AAPL", "MSFT"]


def test_compose_sector_focus_unknown_sector(session) -> None:
    assert compose_sector_focus(session, "NOSUCH") is None


def test_compose_sector_focus_rejects_empty(session) -> None:
    with pytest.raises(ValueError):
        compose_sector_focus(session, "")


# ---------------- compose_event_focus ----------------


def test_compose_event_focus_matches_terms_case_insensitive(session) -> None:
    upsert_articles(
        session,
        [
            make_article(finnhub_id=1, headline="Fed cuts rates by 25 bps",
                          published_at=D - timedelta(days=1)),
            make_article(finnhub_id=2, headline="Apple ships new chip",
                          published_at=D - timedelta(days=2)),
            make_article(finnhub_id=3, headline="Federal Reserve hikes rates",
                          summary="The fed continues tightening",
                          published_at=D - timedelta(days=3)),
        ],
    )
    bundle = compose_event_focus(session, "Fed rates", lookback_days=14)
    assert bundle.article_count == 2  # 1 and 3 match both "fed" + "rates"


def test_compose_event_focus_empty_query_returns_zero(session) -> None:
    upsert_articles(
        session,
        [make_article(finnhub_id=1, headline="anything", published_at=D)],
    )
    bundle = compose_event_focus(session, "")
    assert bundle.article_count == 0
    assert bundle.implied_label == "FLAT"


def test_compose_event_focus_aggregates_sentiment(session) -> None:
    upsert_articles(
        session,
        [
            make_article(finnhub_id=i, headline="iran war update",
                          published_at=D - timedelta(days=1))
            for i in range(1, 4)
        ],
    )
    arts = session.query(type(make_article(finnhub_id=999))).all()
    save_scores(
        session,
        [make_score(a.id, 0.8, model_version="vader-test") for a in arts],
    )

    bundle = compose_event_focus(
        session, "iran war", model_version="vader-test"
    )
    assert bundle.article_count == 3
    assert bundle.aggregate_sentiment == pytest.approx(0.8)
    assert bundle.implied_label == "UP"


def test_compose_event_focus_negative_aggregate(session) -> None:
    upsert_articles(
        session,
        [make_article(finnhub_id=i, headline="bank panic",
                       published_at=D - timedelta(hours=i))
         for i in range(1, 4)],
    )
    arts = session.query(type(make_article(finnhub_id=999))).all()
    save_scores(
        session,
        [make_score(a.id, -0.7, model_version="vader-test") for a in arts],
    )
    bundle = compose_event_focus(
        session, "bank panic", model_version="vader-test"
    )
    assert bundle.implied_label == "DOWN"


def test_compose_event_focus_respects_lookback(session) -> None:
    upsert_articles(
        session,
        [
            make_article(finnhub_id=1, headline="layoffs at MegaCorp",
                          published_at=D - timedelta(days=30)),
            make_article(finnhub_id=2, headline="layoffs continue",
                          published_at=D - timedelta(days=1)),
        ],
    )
    bundle = compose_event_focus(session, "layoffs", lookback_days=7)
    assert bundle.article_count == 1  # only the recent one


# ---------------- refresh_company_relationships ----------------


def test_refresh_company_relationships_persists_peers(session) -> None:
    gw = MagicMock()
    gw.company_peers.return_value = ["MSFT", "GOOG", "META"]
    gw.stock_supply_chain.side_effect = IngestionError("FinnhubAPI 403: gated")

    result = refresh_company_relationships(session, gw, symbol="AAPL")
    assert result.peers_added == 3
    assert any("supply_chain" in f["op"] for f in result.failures)
    rows = related_entities_for(session, "AAPL", relationship="PEER")
    assert [r.related_symbol for r in rows] == ["MSFT", "GOOG", "META"]


def test_refresh_company_relationships_parses_supply_chain(session) -> None:
    gw = MagicMock()
    gw.company_peers.return_value = []
    gw.stock_supply_chain.return_value = {
        "data": [
            {"symbol": "TSM", "name": "Taiwan Semi", "relation": "Supplier"},
            {"symbol": "MU",  "name": "Micron",       "relation": "Supplier"},
            {"symbol": "T",   "name": "AT&T",        "relation": "Customer"},
            {"symbol": "",   "name": "ignored",      "relation": "Supplier"},
        ]
    }
    result = refresh_company_relationships(session, gw, symbol="AAPL")
    assert result.suppliers_added == 2
    assert result.customers_added == 1
    assert {r.related_symbol for r in related_entities_for(session, "AAPL", relationship="SUPPLIER")} == {"TSM", "MU"}


def test_refresh_company_relationships_peers_failure_isolated(session) -> None:
    gw = MagicMock()
    gw.company_peers.side_effect = IngestionError("FinnhubAPI 403")
    gw.stock_supply_chain.return_value = {"data": []}
    result = refresh_company_relationships(session, gw, symbol="AAPL")
    assert result.peers_added == 0
    assert len(result.failures) == 1
    assert "peers" in result.failures[0]["op"]


def test_refresh_company_relationships_rejects_empty_symbol(session) -> None:
    with pytest.raises(ValueError):
        refresh_company_relationships(session, MagicMock(), symbol="")


# ---------------- refresh_sector_constituents ----------------


def test_refresh_sector_constituents_persists_holdings(session) -> None:
    gw = MagicMock()
    gw.etfs_holdings.return_value = {
        "holdings": [
            {"symbol": "AAPL", "name": "Apple Inc.", "percent": 22.0},
            {"symbol": "MSFT", "name": "Microsoft",  "percent": 20.5},
            {"symbol": "",     "name": "ignored",    "percent": 0.0},
        ]
    }
    result = refresh_sector_constituents(session, gw, etf_symbol="XLK")
    assert result.holdings_added == 2
    rows = related_entities_for(session, "XLK", relationship="ETF_HOLDING")
    assert [r.related_symbol for r in rows] == ["AAPL", "MSFT"]


def test_refresh_sector_constituents_failure_isolated(session) -> None:
    gw = MagicMock()
    gw.etfs_holdings.side_effect = IngestionError("FinnhubAPI 403")
    result = refresh_sector_constituents(session, gw, etf_symbol="XLK")
    assert result.holdings_added == 0
    assert len(result.failures) == 1


def test_refresh_sector_constituents_rejects_empty(session) -> None:
    with pytest.raises(ValueError):
        refresh_sector_constituents(session, MagicMock(), etf_symbol="")
