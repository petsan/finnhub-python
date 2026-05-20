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
from finn_predictor.predictor.explain import (
    ArticleContribution,
    article_contributions,
    explain_prediction,
)
from finn_predictor.predictor.market import classify, predict_market
from finn_predictor.predictor.sectors import predict_all_sectors, predict_sector
from finn_predictor.predictor.stocks import (
    predict_all_stocks,
    predict_stock,
    retroactive_predict_many,
    retroactive_predict_stock,
)
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

__all__ = [
    "ArticleContribution",
    "PerformanceSummary",
    "SentimentSummary",
    "TradeRecord",
    "aggregate_sentiment",
    "article_contributions",
    "classify",
    "cumulative_pnl_series",
    "daily_sentiment_index",
    "explain_prediction",
    "hit_rate_by_label",
    "hit_rate_by_target_kind",
    "hypothetical_trades",
    "performance_summary",
    "predict_all_sectors",
    "predict_all_stocks",
    "predict_market",
    "predict_sector",
    "predict_stock",
    "retroactive_predict_many",
    "retroactive_predict_stock",
    "rolling_baseline",
    "rolling_hit_rate",
    "trades_dataframe",
]
