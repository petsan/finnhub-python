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

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from finn_predictor.predictor.market import predict_market
from finn_predictor.sentiment.base import Scorer
from finn_predictor.storage.models import Prediction


def predict_stock(
    session: Session,
    *,
    scorer: Scorer,
    symbol: str,
    on_date: datetime | None = None,
) -> Optional[Prediction]:
    """Compute and persist a per-ticker directional call.

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
    )


def predict_all_stocks(
    session: Session,
    *,
    scorer: Scorer,
    symbols: Iterable[str],
    on_date: datetime | None = None,
) -> list[Prediction]:
    """Run :func:`predict_stock` across a list of tickers.

    Skips tickers with no scored articles in the day's window. Returns
    only the predictions that were actually written.
    """
    out: list[Prediction] = []
    for sym in symbols:
        if not sym:
            continue
        pred = predict_stock(session, scorer=scorer, symbol=sym, on_date=on_date)
        if pred is not None:
            out.append(pred)
    return out
