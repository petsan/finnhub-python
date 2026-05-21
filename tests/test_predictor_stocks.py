"""Per-stock predictor tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.stocks import (
    predict_all_stocks,
    predict_stock,
    retroactive_predict_many,
    retroactive_predict_stock,
)
from finn_predictor.storage.repo import (
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


def _seed_company(session, *, symbol, scores, day=D):
    arts = [
        make_article(
            finnhub_id=hash((symbol, i, day)) % 1_000_000_000,
            symbol=symbol,
            category="company",
            headline=f"{symbol} news {i}",
            published_at=day,
        )
        for i in range(len(scores))
    ]
    upsert_articles(session, arts)
    persisted = (
        session.query(type(arts[0]))
        .filter(
            type(arts[0]).symbol == symbol,
            type(arts[0]).category == "company",
        )
        .all()
    )
    by_finnhub = {a.finnhub_id: a for a in persisted}
    save_scores(
        session,
        [
            make_score(by_finnhub[a.finnhub_id].id, sc, model_version="vader-test")
            for a, sc in zip(arts, scores)
            if a.finnhub_id in by_finnhub
        ],
    )


def test_predict_stock_returns_none_without_articles(session) -> None:
    pred = predict_stock(session, scorer=_FixedScorer(), symbol="AAPL", on_date=D)
    assert pred is None


def test_predict_stock_rejects_empty_symbol(session) -> None:
    with pytest.raises(ValueError):
        predict_stock(session, scorer=_FixedScorer(), symbol="", on_date=D)


def test_predict_stock_writes_row_with_ticker_as_target(session) -> None:
    _seed_company(session, symbol="AAPL", scores=[0.7, 0.8, 0.6, 0.9, 0.85])
    pred = predict_stock(session, scorer=_FixedScorer(), symbol="AAPL", on_date=D)
    assert pred is not None
    assert pred.target_symbol == "AAPL"
    assert pred.label in {"UP", "DOWN", "FLAT"}
    assert pred.article_count == 5

    rows = predictions_for(session, "AAPL")
    assert len(rows) == 1


def test_predict_stock_scopes_to_own_company_news(session) -> None:
    """AAPL's prediction must ignore MSFT articles even though both have
    category='company'."""
    _seed_company(session, symbol="AAPL", scores=[0.9, 0.8, 0.85])
    _seed_company(session, symbol="MSFT", scores=[-0.9, -0.8, -0.85])

    aapl = predict_stock(session, scorer=_FixedScorer(), symbol="AAPL", on_date=D)
    msft = predict_stock(session, scorer=_FixedScorer(), symbol="MSFT", on_date=D)
    assert aapl is not None and msft is not None
    # AAPL is overwhelmingly positive; MSFT overwhelmingly negative.
    assert aapl.label == "UP"
    assert msft.label == "DOWN"
    # Each prediction's article_count is exactly its own ticker's articles.
    assert aapl.article_count == 3
    assert msft.article_count == 3


def test_predict_stock_normalises_prediction_date(session) -> None:
    """Two runs on the same UTC day → one row (inherits market predictor's fix)."""
    _seed_company(session, symbol="AAPL", scores=[0.7, 0.8, 0.6, 0.9, 0.85])
    morning = D.replace(hour=3)
    afternoon = D.replace(hour=20)
    a = predict_stock(session, scorer=_FixedScorer(), symbol="AAPL", on_date=morning)
    b = predict_stock(session, scorer=_FixedScorer(), symbol="AAPL", on_date=afternoon)
    assert a is not None and b is not None
    assert a.id == b.id
    assert len(predictions_for(session, "AAPL")) == 1


def test_predict_all_stocks_runs_one_per_symbol(session) -> None:
    _seed_company(session, symbol="AAPL", scores=[0.9, 0.8, 0.85])
    _seed_company(session, symbol="MSFT", scores=[0.7, 0.6, 0.8])
    # NVDA has no articles — should be skipped.

    out = predict_all_stocks(
        session,
        scorer=_FixedScorer(),
        symbols=["AAPL", "MSFT", "NVDA"],
        on_date=D,
    )
    targets = {p.target_symbol for p in out}
    assert targets == {"AAPL", "MSFT"}


def test_predict_all_stocks_empty_input(session) -> None:
    assert predict_all_stocks(session, scorer=_FixedScorer(), symbols=[]) == []


def test_predict_all_stocks_skips_empty_string(session) -> None:
    _seed_company(session, symbol="AAPL", scores=[0.9, 0.8, 0.85])
    out = predict_all_stocks(
        session, scorer=_FixedScorer(), symbols=["AAPL", ""], on_date=D
    )
    assert [p.target_symbol for p in out] == ["AAPL"]


# ---------------- retroactive_predict_stock ----------------


def test_retroactive_predict_stock_one_per_day_with_articles(session) -> None:
    """Seed articles on day 1, day 3, day 5. Retro-predict over a 5-day
    range → 3 predictions, one per day-with-data."""
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_company(session, symbol="AAPL", scores=[0.9, 0.8, 0.85], day=base)
    _seed_company(
        session, symbol="AAPL", scores=[0.7, 0.8, 0.6],
        day=base + timedelta(days=2),
    )
    _seed_company(
        session, symbol="AAPL", scores=[-0.7, -0.8, -0.9],
        day=base + timedelta(days=4),
    )

    n = retroactive_predict_stock(
        session,
        scorer=_FixedScorer(),
        symbol="AAPL",
        start=base,
        end=base + timedelta(days=4),
    )
    assert n == 3

    rows = predictions_for(session, "AAPL")
    assert len(rows) == 3
    # Predictions are at start-of-UTC-day (normalised).
    days = sorted(
        (r.prediction_date.replace(tzinfo=None) if r.prediction_date.tzinfo else r.prediction_date)
        for r in rows
    )
    assert days == [
        datetime(2026, 5, 1),
        datetime(2026, 5, 3),
        datetime(2026, 5, 5),
    ]


def test_retroactive_predict_stock_idempotent(session) -> None:
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_company(session, symbol="AAPL", scores=[0.9, 0.8, 0.85], day=base)
    a = retroactive_predict_stock(
        session, scorer=_FixedScorer(), symbol="AAPL",
        start=base, end=base,
    )
    b = retroactive_predict_stock(
        session, scorer=_FixedScorer(), symbol="AAPL",
        start=base, end=base,
    )
    assert a == 1 and b == 1
    assert len(predictions_for(session, "AAPL")) == 1


def test_retroactive_predict_stock_empty_range(session) -> None:
    """end < start → 0 predictions, no error."""
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    n = retroactive_predict_stock(
        session, scorer=_FixedScorer(), symbol="AAPL",
        start=base, end=base - timedelta(days=1),
    )
    assert n == 0


def test_retroactive_predict_stock_rejects_empty_symbol(session) -> None:
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        retroactive_predict_stock(
            session, scorer=_FixedScorer(), symbol="",
            start=base, end=base,
        )


def test_retroactive_predict_stock_handles_naive_datetimes(session) -> None:
    naive = datetime(2026, 5, 1)
    _seed_company(
        session, symbol="AAPL", scores=[0.9, 0.8, 0.85],
        day=naive.replace(tzinfo=timezone.utc),
    )
    n = retroactive_predict_stock(
        session, scorer=_FixedScorer(), symbol="AAPL",
        start=naive, end=naive,
    )
    assert n == 1


def test_retroactive_predict_many(session) -> None:
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_company(session, symbol="AAPL", scores=[0.9, 0.8, 0.85], day=base)
    _seed_company(session, symbol="MSFT", scores=[-0.7, -0.6, -0.8], day=base)

    out = retroactive_predict_many(
        session,
        scorer=_FixedScorer(),
        symbols=["AAPL", "MSFT", "NVDA"],  # NVDA has no articles
        start=base,
        end=base,
    )
    assert out == {"AAPL": 1, "MSFT": 1, "NVDA": 0}


def test_retroactive_predict_many_skips_blank_entries(session) -> None:
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    _seed_company(session, symbol="AAPL", scores=[0.9, 0.8, 0.85], day=base)
    out = retroactive_predict_many(
        session, scorer=_FixedScorer(),
        symbols=["AAPL", "", "  "],
        start=base, end=base,
    )
    assert list(out.keys()) == ["AAPL"]
