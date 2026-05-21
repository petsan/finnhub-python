"""Per-stock predictor — directional call for an individual ticker.

Reuses the z-score classifier from :mod:`predictor.market`, scoped to a
single ticker's own ``company`` news. Stored as a :class:`Prediction`
row with ``target_symbol = <ticker>`` so it flows through the existing
:func:`storage.repo.save_prediction` upsert (one row per ticker per UTC
day per model) and the duplicate-collapsing migration.

The market and sector predictors live in their own modules for
discoverability; this one is small enough to be a wrapper but it gets
its own module so callers (jobs, UI, tests) import the API by intent
rather than by reaching into ``market`` for a non-market call.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from finn_predictor.predictor.classifier import LogisticCalibration
from finn_predictor.predictor.magnitude import MagnitudeCalibration
from finn_predictor.predictor.market import predict_market
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import Prediction
from finn_predictor.storage.repo import utc_day_window


def predict_stock(
    session: Session,
    *,
    scorer: Scorer,
    symbol: str,
    on_date: datetime | None = None,
    threshold_sigma: Optional[float] = None,
    min_baseline_sigma: Optional[float] = None,
    half_life_hours: Optional[float] = None,
    source_weights: Optional[dict[str, float]] = None,
    calibration: Optional[LogisticCalibration] = None,
    magnitude_calibration: Optional[MagnitudeCalibration] = None,
) -> Optional[Prediction]:
    """Compute and persist a per-ticker directional call.

    All four learnable parameters forward to :func:`predict_market`;
    pass them explicitly when you want a per-call override of the
    active learned weights.

    Returns ``None`` if no scored articles exist for ``symbol`` in the
    target day — matches the market predictor's "not enough data" path.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    return predict_market(
        session,
        scorer=scorer,
        on_date=on_date,
        symbol=symbol,
        category="company",
        article_symbol=symbol,
        threshold_sigma=threshold_sigma,
        min_baseline_sigma=min_baseline_sigma,
        half_life_hours=half_life_hours,
        source_weights=source_weights,
        calibration=calibration,
        magnitude_calibration=magnitude_calibration,
    )


def predict_all_stocks(
    session: Session,
    *,
    scorer: Scorer,
    symbols: Iterable[str],
    on_date: datetime | None = None,
    threshold_sigma: Optional[float] = None,
    min_baseline_sigma: Optional[float] = None,
    half_life_hours: Optional[float] = None,
    source_weights: Optional[dict[str, float]] = None,
    calibration: Optional[LogisticCalibration] = None,
    magnitude_calibration: Optional[MagnitudeCalibration] = None,
) -> list[Prediction]:
    """Run :func:`predict_stock` across a list of tickers.

    Skips tickers with no scored articles in the day's window. Returns
    only the predictions that were actually written.
    """
    out: list[Prediction] = []
    for sym in symbols:
        if not sym:
            continue
        pred = predict_stock(
            session, scorer=scorer, symbol=sym, on_date=on_date,
            threshold_sigma=threshold_sigma,
            min_baseline_sigma=min_baseline_sigma,
            half_life_hours=half_life_hours,
            source_weights=source_weights,
            calibration=calibration,
            magnitude_calibration=magnitude_calibration,
        )
        if pred is not None:
            out.append(pred)
    return out


def retroactive_predict_stock(
    session: Session,
    *,
    scorer: Scorer,
    symbol: str,
    start: datetime,
    end: datetime,
) -> int:
    """Generate one :class:`Prediction` per UTC day in ``[start, end]``.

    Used after a historical backfill to populate the *History* tab and
    establish a real rolling baseline. Days with no scored articles in
    the window are skipped (predict_stock returns None). Re-running is
    safe — the start-of-UTC-day prediction_date normalization plus
    save_prediction's upsert collapse same-day re-runs.

    ``start`` and ``end`` are normalised to UTC if naive.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
    if end_utc < start_utc:
        return 0

    day, _ = utc_day_window(start_utc)
    last_day, _ = utc_day_window(end_utc)
    written = 0
    while day <= last_day:
        pred = predict_stock(session, scorer=scorer, symbol=symbol, on_date=day)
        if pred is not None:
            written += 1
        day = day + timedelta(days=1)
    return written


def retroactive_predict_many(
    session: Session,
    *,
    scorer: Scorer,
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
) -> dict[str, int]:
    """Run :func:`retroactive_predict_stock` for each ticker.

    Returns ``{symbol: predictions_written}``.
    """
    out: dict[str, int] = {}
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        out[sym] = retroactive_predict_stock(
            session, scorer=scorer, symbol=sym, start=start, end=end
        )
    return out
