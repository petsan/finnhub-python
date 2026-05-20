"""Hypothetical-trade simulator + accuracy aggregation tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finn_predictor.predictor.trades import (
    PerformanceSummary,
    TradeRecord,
    cumulative_pnl_series,
    hit_rate_by_label,
    hit_rate_by_target_kind,
    hypothetical_trades,
    performance_summary,
    rolling_hit_rate,
    trades_dataframe,
)
from finn_predictor.storage.models import PredictionOutcome, Sector
from finn_predictor.storage.repo import save_outcome, save_prediction
from tests.conftest import make_prediction


D = datetime(2026, 5, 19, tzinfo=timezone.utc)


def _make_market(session, label, ret=None, hit=None, *, day_offset: int = 0):
    """Persist a market Prediction on day D + day_offset.

    Distinct offsets are required across the same test because the
    dedupe migration upserts on (target, UTC-day, model).
    """
    p = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D + timedelta(days=day_offset),
            label=label,
            confidence=0.7,
        ),
    )
    if ret is not None:
        save_outcome(
            session,
            PredictionOutcome(prediction_id=p.id, realised_return=ret, hit=hit),
        )
    return p


# ---------------- hypothetical_trades ----------------


def test_hypothetical_trades_empty(session) -> None:
    assert hypothetical_trades(session) == []


def test_hypothetical_trades_pnl_signed_by_direction(session) -> None:
    """UP gets +realised, DOWN gets -realised, FLAT gets 0."""
    _make_market(session, "UP", ret=0.012, hit=True, day_offset=0)
    _make_market(session, "DOWN", ret=-0.008, hit=True, day_offset=1)
    _make_market(session, "FLAT", ret=0.001, hit=True, day_offset=2)

    trades = hypothetical_trades(session)
    by_label = {t.label: t for t in trades}
    assert by_label["UP"].pnl_pct == pytest.approx(0.012)
    assert by_label["DOWN"].pnl_pct == pytest.approx(0.008)   # -(-0.008)
    assert by_label["FLAT"].pnl_pct == pytest.approx(0.0)


def test_hypothetical_trades_open_when_no_outcome(session) -> None:
    _make_market(session, "UP")
    trades = hypothetical_trades(session)
    assert len(trades) == 1
    assert trades[0].closed is False
    assert trades[0].pnl_pct is None
    assert trades[0].hit is None


def test_hypothetical_trades_target_kind_classification(session) -> None:
    """^GSPC → MARKET, sector ETF → SECTOR, anything else → STOCK."""
    session.add(Sector(code="TECH", name="Tech", etf_symbol="XLK"))
    session.commit()

    _make_market(session, "UP", ret=0.01, hit=True)
    p = save_prediction(
        session, make_prediction(
            target_symbol="XLK", prediction_date=D, label="DOWN",
            confidence=0.6,
        ),
    )
    save_outcome(session, PredictionOutcome(prediction_id=p.id, realised_return=-0.01, hit=True))
    p2 = save_prediction(
        session, make_prediction(
            target_symbol="AAPL", prediction_date=D, label="UP",
            confidence=0.5,
        ),
    )
    save_outcome(session, PredictionOutcome(prediction_id=p2.id, realised_return=0.02, hit=True))

    kinds = {t.target_symbol: t.target_kind for t in hypothetical_trades(session)}
    assert kinds == {"^GSPC": "MARKET", "XLK": "SECTOR", "AAPL": "STOCK"}


def test_hypothetical_trades_filter_by_target(session) -> None:
    _make_market(session, "UP", ret=0.01, hit=True)
    save_prediction(
        session,
        make_prediction(target_symbol="AAPL", prediction_date=D, label="UP"),
    )
    trades = hypothetical_trades(session, target_symbol="AAPL")
    assert len(trades) == 1
    assert trades[0].target_symbol == "AAPL"


def test_hypothetical_trades_filter_by_since(session) -> None:
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC", prediction_date=D - timedelta(days=10),
            label="UP",
        ),
    )
    save_prediction(
        session,
        make_prediction(target_symbol="^GSPC", prediction_date=D, label="UP"),
    )
    trades = hypothetical_trades(session, since=D - timedelta(days=5))
    assert len(trades) == 1


def test_hypothetical_trades_ordered_chronologically(session) -> None:
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC", prediction_date=D + timedelta(days=2),
            label="UP",
        ),
    )
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC", prediction_date=D, label="UP",
        ),
    )
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC", prediction_date=D + timedelta(days=1),
            label="UP",
        ),
    )
    dates = [t.entry_date for t in hypothetical_trades(session)]
    assert dates == sorted(dates)


# ---------------- aggregates ----------------


def test_cumulative_pnl_series_excludes_open_and_flat(session) -> None:
    _make_market(session, "UP", ret=0.01, hit=True, day_offset=0)       # closed UP, +1%
    _make_market(session, "FLAT", ret=0.001, hit=True, day_offset=1)    # FLAT - excluded
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC", prediction_date=D + timedelta(days=2),
            label="UP",
        ),
    )  # open - excluded
    df = cumulative_pnl_series(hypothetical_trades(session))
    assert len(df) == 1
    assert df.iloc[0]["pnl_pct"] == pytest.approx(0.01)
    assert df.iloc[0]["cum_pnl_pct"] == pytest.approx(0.01)


def test_cumulative_pnl_series_cumulative(session) -> None:
    # Three closed UP trades: +1%, -0.5%, +2%
    for i, ret in enumerate([0.01, -0.005, 0.02]):
        p = save_prediction(
            session,
            make_prediction(
                target_symbol="^GSPC",
                prediction_date=D + timedelta(days=i),
                label="UP",
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(
                prediction_id=p.id, realised_return=ret, hit=ret > 0
            ),
        )
    df = cumulative_pnl_series(hypothetical_trades(session))
    assert list(df["cum_pnl_pct"]) == pytest.approx(
        [0.01, 0.005, 0.025], abs=1e-9
    )


def test_cumulative_pnl_series_empty(session) -> None:
    df = cumulative_pnl_series([])
    assert df.empty
    assert "cum_pnl_pct" in df.columns


def test_rolling_hit_rate_window(session) -> None:
    # 5 closed UP trades, hits pattern: T F T T F → rolling mean over window=3
    pattern = [True, False, True, True, False]
    for i, hit in enumerate(pattern):
        p = save_prediction(
            session,
            make_prediction(
                target_symbol="^GSPC",
                prediction_date=D + timedelta(days=i),
                label="UP",
            ),
        )
        save_outcome(
            session,
            PredictionOutcome(
                prediction_id=p.id,
                realised_return=0.01 if hit else -0.01,
                hit=hit,
            ),
        )
    df = rolling_hit_rate(hypothetical_trades(session), window=3)
    # Last value is mean of [T,T,F] = 2/3
    assert df.iloc[-1]["hit_rate"] == pytest.approx(2 / 3)


def test_hit_rate_by_target_kind_excludes_flat_and_open(session) -> None:
    session.add(Sector(code="TECH", name="Tech", etf_symbol="XLK"))
    session.commit()

    _make_market(session, "UP", ret=0.01, hit=True, day_offset=0)
    _make_market(session, "FLAT", ret=0.001, hit=True, day_offset=1)  # excluded from directional
    p = save_prediction(
        session, make_prediction(
            target_symbol="XLK", prediction_date=D, label="DOWN",
        ),
    )  # open - excluded
    save_prediction(
        session, make_prediction(
            target_symbol="AAPL", prediction_date=D, label="UP",
        ),
    )
    # close the AAPL trade with a loss
    apple_p = (
        session.query(type(p)).filter_by(target_symbol="AAPL").first()
    )
    save_outcome(
        session,
        PredictionOutcome(
            prediction_id=apple_p.id, realised_return=-0.005, hit=False
        ),
    )

    df = hit_rate_by_target_kind(hypothetical_trades(session))
    by_kind = {row["target_kind"]: row for _, row in df.iterrows()}
    assert by_kind["MARKET"]["hit_rate"] == 1.0   # 1/1
    assert by_kind["STOCK"]["hit_rate"] == 0.0    # 0/1
    assert "SECTOR" not in by_kind  # XLK trade was open


def test_hit_rate_by_label_includes_flat(session) -> None:
    _make_market(session, "UP", ret=0.01, hit=True, day_offset=0)
    _make_market(session, "FLAT", ret=0.001, hit=True, day_offset=1)
    df = hit_rate_by_label(hypothetical_trades(session))
    labels = {row["label"]: row for _, row in df.iterrows()}
    assert labels["UP"]["hit_rate"] == 1.0
    assert labels["FLAT"]["hit_rate"] == 1.0


def test_performance_summary_full(session) -> None:
    # Closed: UP (+1%) hit, UP (-0.5%) miss, DOWN (-0.6%) hit
    # FLAT skipped, plus one open prediction
    p1 = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D, label="UP",
        ),
    )
    save_outcome(
        session, PredictionOutcome(prediction_id=p1.id, realised_return=0.01, hit=True)
    )
    p2 = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D + timedelta(days=1), label="UP",
        ),
    )
    save_outcome(
        session, PredictionOutcome(prediction_id=p2.id, realised_return=-0.005, hit=False)
    )
    p3 = save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D + timedelta(days=2), label="DOWN",
        ),
    )
    save_outcome(
        session, PredictionOutcome(prediction_id=p3.id, realised_return=-0.006, hit=True)
    )
    _make_market(session, "FLAT", ret=0.0001, hit=True, day_offset=3)
    save_prediction(
        session,
        make_prediction(
            target_symbol="^GSPC",
            prediction_date=D + timedelta(days=4),
            label="UP",
        ),
    )  # open

    summary = performance_summary(hypothetical_trades(session))
    assert summary.total_predictions == 5
    assert summary.closed_trades == 3
    assert summary.open_trades == 1
    assert summary.flat_skipped == 1
    assert summary.wins == 2
    assert summary.losses == 1
    assert summary.hit_rate == pytest.approx(2 / 3)
    # PnL: +1% + (-0.5%) + (-(-0.6%)) = +1.1%
    assert summary.cumulative_pnl_pct == pytest.approx(0.011)
    assert summary.best_trade_pnl_pct == pytest.approx(0.01)
    assert summary.worst_trade_pnl_pct == pytest.approx(-0.005)


def test_performance_summary_empty() -> None:
    s = performance_summary([])
    assert s.total_predictions == 0
    assert s.closed_trades == 0
    assert s.hit_rate == 0.0
    assert s.cumulative_pnl_pct == 0.0
    assert s.best_trade_pnl_pct is None


def test_trades_dataframe_columns_and_empty() -> None:
    df = trades_dataframe([])
    expected = {
        "prediction_id", "target_symbol", "target_kind", "direction",
        "label", "entry_date", "confidence", "realised_return", "pnl_pct",
        "hit", "closed",
    }
    assert expected.issubset(set(df.columns))
    assert df.empty
