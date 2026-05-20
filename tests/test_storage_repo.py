"""Storage / repository tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.storage.repo import (
    DEFAULT_SECTORS,
    all_sectors,
    articles_in_window,
    ensure_default_sectors,
    latest_price_bar,
    price_bars,
    predictions_for,
    save_outcome,
    save_prediction,
    save_scores,
    unscored_articles,
    upsert_articles,
    upsert_price_bars,
    utc_day_window,
)
from finn_predictor.storage.models import PredictionOutcome
from tests.conftest import (
    make_article,
    make_prediction,
    make_price_bar,
    make_score,
)


D0 = datetime(2026, 5, 19, 14, tzinfo=timezone.utc)


def test_upsert_articles_inserts_then_dedupes(session) -> None:
    a = make_article(finnhub_id=1)
    assert upsert_articles(session, [a]) == 1
    # Re-insert same finnhub_id -> 0 new rows
    dup = make_article(finnhub_id=1, headline="changed")
    assert upsert_articles(session, [dup]) == 0
    rows = articles_in_window(session, D0 - timedelta(days=1), D0 + timedelta(days=1))
    assert len(rows) == 1
    # Confirm original headline retained (we DO NOTHING on conflict).
    assert rows[0].headline == "headline"


def test_articles_in_window_filters_by_category_and_symbol(session) -> None:
    upsert_articles(
        session,
        [
            make_article(finnhub_id=1, category="general", symbol=None, published_at=D0),
            make_article(finnhub_id=2, category="company", symbol="AAPL", published_at=D0),
            make_article(finnhub_id=3, category="company", symbol="MSFT", published_at=D0),
        ],
    )
    rows = articles_in_window(
        session, D0 - timedelta(hours=1), D0 + timedelta(hours=1), category="company"
    )
    assert sorted(r.symbol for r in rows) == ["AAPL", "MSFT"]

    rows = articles_in_window(
        session, D0 - timedelta(hours=1), D0 + timedelta(hours=1), symbol="AAPL"
    )
    assert [r.symbol for r in rows] == ["AAPL"]


def test_unscored_articles_skips_already_scored(session) -> None:
    upsert_articles(session, [make_article(finnhub_id=i) for i in (1, 2, 3)])
    arts = articles_in_window(session, D0 - timedelta(days=1), D0 + timedelta(days=1))
    save_scores(session, [make_score(arts[0].id, 0.2)])

    pending = unscored_articles(session, "vader-test")
    assert {a.finnhub_id for a in pending} == {arts[1].finnhub_id, arts[2].finnhub_id}


def test_upsert_price_bars_unique_by_symbol_and_date(session) -> None:
    bar = make_price_bar("^GSPC", D0, close=4000.0)
    assert upsert_price_bars(session, [bar]) == 1
    assert upsert_price_bars(session, [bar]) == 0  # dedupe

    latest = latest_price_bar(session, "^GSPC")
    assert latest is not None
    assert latest.close == 4000.0

    upsert_price_bars(session, [make_price_bar("^GSPC", D0 + timedelta(days=1), 4050.0)])
    rows = price_bars(session, "^GSPC", D0, D0 + timedelta(days=5))
    assert [r.close for r in rows] == [4000.0, 4050.0]


def test_save_prediction_upserts_same_triple(session) -> None:
    p1 = make_prediction(target_symbol="^GSPC", prediction_date=D0, label="UP")
    saved = save_prediction(session, p1)
    assert saved.id is not None

    p2 = make_prediction(target_symbol="^GSPC", prediction_date=D0, label="DOWN")
    again = save_prediction(session, p2)
    assert again.id == saved.id
    assert again.label == "DOWN"  # updated in place

    rows = predictions_for(session, "^GSPC")
    assert len(rows) == 1


def test_save_outcome_upserts(session) -> None:
    pred = save_prediction(
        session, make_prediction(target_symbol="^GSPC", prediction_date=D0)
    )
    save_outcome(
        session,
        PredictionOutcome(prediction_id=pred.id, realised_return=0.01, hit=True),
    )
    save_outcome(
        session,
        PredictionOutcome(prediction_id=pred.id, realised_return=-0.005, hit=False),
    )
    session.refresh(pred)
    assert pred.outcome is not None
    assert pred.outcome.realised_return == pytest.approx(-0.005)
    assert pred.outcome.hit is False


def test_ensure_default_sectors_is_idempotent(session) -> None:
    first = ensure_default_sectors(session)
    second = ensure_default_sectors(session)
    assert {s.code for s in first} == {code for code, *_ in DEFAULT_SECTORS}
    assert len(first) == len(second) == len(DEFAULT_SECTORS)
    assert {s.code for s in all_sectors(session)} == {s.code for s in first}


def test_utc_day_window_normalises_to_midnight() -> None:
    start, end = utc_day_window(D0)
    assert start.tzinfo is timezone.utc
    assert start.hour == 0 and start.minute == 0
    assert end - start == timedelta(days=1)
