"""Iteration 2: per-sector predictor tests."""

from __future__ import annotations

from datetime import datetime, timezone

from finn_predictor.predictor.sectors import predict_all_sectors, predict_sector
from finn_predictor.storage.models import Sector
from finn_predictor.storage.repo import (
    ensure_default_sectors,
    predictions_for,
    save_scores,
    upsert_articles,
)
from tests.conftest import make_article, make_score


D = datetime(2026, 5, 19, 12, tzinfo=timezone.utc)


class _FixedScorer:
    model_version = "vader-test"

    def score(self, text: str) -> float:
        return 0.0

    def score_many(self, texts):
        return [0.0 for _ in texts]


def _seed_company_articles(session, *, symbol, scores, day=D):
    """Drop `len(scores)` company-news articles for `symbol` with given scores."""
    ids = list(range(1, len(scores) + 1))
    arts = [
        make_article(
            finnhub_id=hash((symbol, i)) % 1_000_000_000,
            symbol=symbol,
            category="company",
            published_at=day,
        )
        for i in ids
    ]
    upsert_articles(session, arts)
    persisted = (
        session.query(type(arts[0]))
        .filter(type(arts[0]).symbol == symbol, type(arts[0]).category == "company")
        .all()
    )
    save_scores(
        session,
        [
            make_score(art.id, sc, model_version="vader-test")
            for art, sc in zip(persisted, scores)
        ],
    )


def test_predict_sector_returns_none_without_articles(session) -> None:
    sector = Sector(code="TECH", name="Tech", etf_symbol="XLK")
    session.add(sector)
    session.commit()
    out = predict_sector(
        session,
        scorer=_FixedScorer(),
        sector=sector,
        sector_symbols=["AAPL", "MSFT"],
        on_date=D,
    )
    assert out is None


def test_predict_sector_calls_up_for_positive_company_news(session) -> None:
    sector = Sector(code="TECH", name="Tech", etf_symbol="XLK")
    session.add(sector)
    session.commit()

    _seed_company_articles(session, symbol="AAPL", scores=[0.8, 0.7, 0.9])
    _seed_company_articles(session, symbol="MSFT", scores=[0.6, 0.85, 0.75])

    pred = predict_sector(
        session,
        scorer=_FixedScorer(),
        sector=sector,
        sector_symbols=["AAPL", "MSFT"],
        on_date=D,
    )
    assert pred is not None
    assert pred.target_symbol == "XLK"
    assert pred.label == "UP"
    assert pred.article_count == 6


def test_predict_sector_calls_down_for_negative_company_news(session) -> None:
    sector = Sector(code="ENERGY", name="Energy", etf_symbol="XLE")
    session.add(sector)
    session.commit()
    _seed_company_articles(session, symbol="XOM", scores=[-0.8, -0.85, -0.7])
    _seed_company_articles(session, symbol="CVX", scores=[-0.6, -0.75, -0.9])

    pred = predict_sector(
        session,
        scorer=_FixedScorer(),
        sector=sector,
        sector_symbols=["XOM", "CVX"],
        on_date=D,
    )
    assert pred is not None
    assert pred.label == "DOWN"


def test_predict_all_sectors_skips_sectors_without_universe(session) -> None:
    ensure_default_sectors(session)
    _seed_company_articles(session, symbol="AAPL", scores=[0.9, 0.8, 0.7])

    preds = predict_all_sectors(
        session,
        scorer=_FixedScorer(),
        on_date=D,
        sector_universe={"TECH": ["AAPL"]},
    )
    assert {p.target_symbol for p in preds} == {"XLK"}


def test_predict_all_sectors_empty_when_no_universe(session) -> None:
    ensure_default_sectors(session)
    out = predict_all_sectors(session, scorer=_FixedScorer(), on_date=D)
    assert out == []


def test_predict_sector_emits_flat_below_min_articles(session) -> None:
    """Fewer than MIN_ARTICLES_FOR_CALL articles -> FLAT with zero confidence."""
    sector = Sector(code="UTIL", name="Utilities", etf_symbol="XLU")
    session.add(sector)
    session.commit()
    _seed_company_articles(session, symbol="DUK", scores=[0.9, 0.8])  # only 2 articles

    pred = predict_sector(
        session,
        scorer=_FixedScorer(),
        sector=sector,
        sector_symbols=["DUK"],
        on_date=D,
    )
    assert pred is not None
    assert pred.label == "FLAT"
    assert pred.confidence == 0.0


def test_predict_sector_persists_through_repo(session) -> None:
    sector = Sector(code="FIN", name="Financials", etf_symbol="XLF")
    session.add(sector)
    session.commit()
    _seed_company_articles(session, symbol="JPM", scores=[0.5, 0.6, 0.55])

    predict_sector(
        session,
        scorer=_FixedScorer(),
        sector=sector,
        sector_symbols=["JPM"],
        on_date=D,
    )
    rows = predictions_for(session, "XLF")
    assert len(rows) == 1
    assert rows[0].article_count == 3
