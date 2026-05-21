"""Storage / repository tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.storage.repo import (
    DEFAULT_SECTORS,
    all_sectors,
    articles_in_window,
    ensure_default_sectors,
    latest_market_caps,
    latest_price_bar,
    migrate_predictions_to_daily,
    price_bars,
    predictions_for,
    save_outcome,
    save_prediction,
    save_scores,
    unscored_articles,
    upsert_articles,
    upsert_market_caps,
    upsert_price_bars,
    utc_day_window,
)
from finn_predictor.storage.models import HistoricalMarketCap, PredictionOutcome
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


def test_migrate_predictions_collapses_same_day_duplicates(session) -> None:
    """4 predictions across one UTC day → 1 row with prediction_date at midnight."""
    base_day = datetime(2026, 5, 20, tzinfo=timezone.utc)
    timestamps = [
        base_day.replace(hour=3, minute=48),
        base_day.replace(hour=3, minute=55),
        base_day.replace(hour=3, minute=59),
        base_day.replace(hour=4, minute=6),
    ]
    for i, ts in enumerate(timestamps):
        save_prediction(
            session,
            make_prediction(
                target_symbol="^GSPC",
                prediction_date=ts,
                label="DOWN",
                confidence=1.0,
                sentiment_index=-0.463,
                article_count=3,
                model_version="vader-test",
            ),
        )

    assert len(predictions_for(session, "^GSPC")) == 4
    deleted = migrate_predictions_to_daily(session)
    assert deleted == 3
    rows = predictions_for(session, "^GSPC")
    assert len(rows) == 1

    # The kept row's prediction_date is now midnight UTC for that day.
    stored = rows[0].prediction_date
    actual = stored.replace(tzinfo=None) if stored.tzinfo else stored
    assert actual == datetime(2026, 5, 20, 0, 0, 0)


def test_migrate_predictions_keeps_separate_days(session) -> None:
    """Predictions on different UTC days must NOT be collapsed."""
    d1 = datetime(2026, 5, 19, 14, tzinfo=timezone.utc)
    d2 = datetime(2026, 5, 20, 14, tzinfo=timezone.utc)
    save_prediction(session, make_prediction(target_symbol="^GSPC", prediction_date=d1))
    save_prediction(session, make_prediction(target_symbol="^GSPC", prediction_date=d2))
    migrate_predictions_to_daily(session)
    rows = predictions_for(session, "^GSPC")
    assert len(rows) == 2


def test_migrate_predictions_idempotent_on_clean_db(session) -> None:
    """Re-running the migration after it's clean must be a no-op."""
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=datetime(2026, 5, 19, tzinfo=timezone.utc),
        ),
    )
    assert migrate_predictions_to_daily(session) == 0
    assert migrate_predictions_to_daily(session) == 0
    assert len(predictions_for(session, "^GSPC")) == 1


def test_migrate_predictions_keeps_separate_target_symbols(session) -> None:
    """Same day, different target_symbols → don't collapse."""
    d = datetime(2026, 5, 19, 14, tzinfo=timezone.utc)
    save_prediction(session, make_prediction(target_symbol="^GSPC", prediction_date=d))
    save_prediction(session, make_prediction(target_symbol="XLK", prediction_date=d))
    save_prediction(session, make_prediction(target_symbol="XLE", prediction_date=d))
    migrate_predictions_to_daily(session)
    assert len(predictions_for(session, "^GSPC")) == 1
    assert len(predictions_for(session, "XLK")) == 1
    assert len(predictions_for(session, "XLE")) == 1


def test_dialect_insert_picks_sqlite_by_default(session) -> None:
    """The helper returns the SQLite insert builder for our test session."""
    from finn_predictor.storage.repo import _dialect_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    assert _dialect_insert(session) is sqlite_insert


def test_dialect_insert_picks_postgres_when_dialect_is_postgresql() -> None:
    """When the session is bound to a postgresql engine, use postgres_insert."""
    from sqlalchemy.dialects.postgresql import insert as postgres_insert
    from finn_predictor.storage.repo import _dialect_insert
    from unittest.mock import MagicMock

    fake_session = MagicMock()
    fake_dialect = MagicMock()
    fake_dialect.name = "postgresql"
    fake_session.get_bind.return_value.dialect = fake_dialect

    assert _dialect_insert(fake_session) is postgres_insert


def test_app_settings_get_set_roundtrip(session) -> None:
    from finn_predictor.storage.repo import (
        get_setting, set_setting,
    )
    assert get_setting(session, "missing", default="d") == "d"
    set_setting(session, "k", "v1")
    assert get_setting(session, "k") == "v1"
    # Upsert: overwriting same key updates rather than duplicates.
    set_setting(session, "k", "v2")
    assert get_setting(session, "k") == "v2"


def test_activation_policy_defaults_and_set(session) -> None:
    from finn_predictor.storage.repo import (
        POLICY_AUTO, POLICY_MANUAL,
        get_activation_policy, set_activation_policy,
    )
    # Default is AUTO when nothing has been set.
    assert get_activation_policy(session) == POLICY_AUTO
    set_activation_policy(session, POLICY_MANUAL)
    assert get_activation_policy(session) == POLICY_MANUAL


def test_activation_policy_rejects_unknown_value(session) -> None:
    from finn_predictor.storage.repo import set_activation_policy
    with pytest.raises(ValueError):
        set_activation_policy(session, "WHATEVER")


def test_activation_policy_unknown_persisted_falls_back_to_auto(session) -> None:
    """If the DB somehow holds a junk value, the getter coerces to AUTO."""
    from finn_predictor.storage.repo import (
        POLICY_AUTO, get_activation_policy, set_setting,
        SETTING_ACTIVATION_POLICY,
    )
    set_setting(session, SETTING_ACTIVATION_POLICY, "GARBAGE")
    assert get_activation_policy(session) == POLICY_AUTO


def test_activate_learned_version_flips_flags(session) -> None:
    from finn_predictor.storage.models import LearnedWeight
    from finn_predictor.storage.repo import activate_learned_version

    session.add_all(
        [
            LearnedWeight(version=1, dimension="THRESHOLD_SIGMA",
                          value=0.4, is_active=True),
            LearnedWeight(version=1, dimension="MIN_BASELINE_SIGMA",
                          value=0.05, is_active=True),
            LearnedWeight(version=2, dimension="THRESHOLD_SIGMA",
                          value=0.6, is_active=False),
            LearnedWeight(version=2, dimension="MIN_BASELINE_SIGMA",
                          value=0.08, is_active=False),
        ]
    )
    session.commit()
    n = activate_learned_version(session, 2)
    assert n == 2  # both v2 rows now active

    actives = session.query(LearnedWeight).filter_by(is_active=True).all()
    assert {a.version for a in actives} == {2}


def test_activate_learned_version_raises_on_missing_version(session) -> None:
    from finn_predictor.storage.repo import activate_learned_version
    with pytest.raises(ValueError):
        activate_learned_version(session, 99)


def test_utc_day_window_normalises_to_midnight() -> None:
    start, end = utc_day_window(D0)
    assert start.tzinfo is timezone.utc
    assert start.hour == 0 and start.minute == 0
    assert end - start == timedelta(days=1)


# -- Market caps ---------------------------------------------------------


def _cap(symbol: str, day: datetime, value: float) -> HistoricalMarketCap:
    return HistoricalMarketCap(symbol=symbol, as_of_date=day, market_cap=value)


def test_upsert_market_caps_dedupes_on_symbol_date(session) -> None:
    """The (symbol, as_of_date) unique constraint silences duplicate inserts."""
    d = D0
    rows = [_cap("AAPL", d, 1.0), _cap("AAPL", d, 9.9), _cap("AAPL", d, 2.5)]
    n = upsert_market_caps(session, rows)
    assert n == 1
    caps = latest_market_caps(session, ["AAPL"])
    # First write wins; subsequent values for the same day no-op (matches
    # the no-update semantics of upsert_price_bars).
    assert caps == {"AAPL": pytest.approx(1.0)}


def test_latest_market_caps_returns_latest_per_symbol(session) -> None:
    """Multiple snapshots per symbol → only the most recent appears."""
    day_old = D0 - timedelta(days=5)
    day_new = D0
    upsert_market_caps(
        session,
        [
            _cap("AAPL", day_old, 2_800_000.0),
            _cap("AAPL", day_new, 2_910_000.0),
            _cap("MSFT", day_old, 3_000_000.0),
        ],
    )
    caps = latest_market_caps(session, ["AAPL", "MSFT", "NONE"])
    assert caps == {
        "AAPL": pytest.approx(2_910_000.0),
        "MSFT": pytest.approx(3_000_000.0),
    }
    assert "NONE" not in caps


def test_latest_market_caps_honours_on_or_before(session) -> None:
    """Look-ahead bias guard: only snapshots ≤ cutoff are considered."""
    day_old = D0 - timedelta(days=5)
    day_new = D0
    upsert_market_caps(
        session,
        [
            _cap("AAPL", day_old, 2_800_000.0),
            _cap("AAPL", day_new, 2_910_000.0),
        ],
    )
    caps = latest_market_caps(
        session, ["AAPL"], on_or_before=D0 - timedelta(days=1)
    )
    assert caps == {"AAPL": pytest.approx(2_800_000.0)}


def test_latest_market_caps_skips_non_positive_values(session) -> None:
    """A zero / negative cap row is treated as missing."""
    upsert_market_caps(session, [_cap("WEIRD", D0, 0.0)])
    caps = latest_market_caps(session, ["WEIRD"])
    assert caps == {}
