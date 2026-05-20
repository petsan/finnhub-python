"""Prediction layer.

* :mod:`market`   - whole-market predictor (iteration 1)
* :mod:`sectors`  - per-sector predictor (iteration 2)
* :mod:`backtest` - replay predictions against realised price moves
"""

from finn_predictor.predictor.aggregate import (
    SentimentSummary,
    aggregate_sentiment,
    daily_sentiment_index,
    rolling_baseline,
)
from finn_predictor.predictor.market import classify, predict_market
from finn_predictor.predictor.sectors import predict_all_sectors, predict_sector

__all__ = [
    "SentimentSummary",
    "aggregate_sentiment",
    "classify",
    "daily_sentiment_index",
    "predict_all_sectors",
    "predict_market",
    "predict_sector",
    "rolling_baseline",
]
