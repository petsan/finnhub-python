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
from finn_predictor.predictor.classifier import (
    LogisticCalibration,
    NotEnoughCalibrationDataError,
    apply_logreg_classification,
    fit_logreg_calibration,
    resolve_classifier_mode,
)
from finn_predictor.predictor.magnitude import (
    MagnitudeCalibration,
    MagnitudeForecast,
    NotEnoughMagnitudeDataError,
    QuantileFit,
    fit_quantile_calibration,
    resolve_magnitude_mode,
)
from finn_predictor.predictor.explain import (
    ArticleContribution,
    article_contributions,
    explain_prediction,
)
from finn_predictor.predictor.focus import (
    CompanyFocus,
    EventFocus,
    RefreshResult,
    RelatedPrediction,
    SectorFocus,
    compose_company_focus,
    compose_event_focus,
    compose_sector_focus,
    refresh_company_relationships,
    refresh_sector_constituents,
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
    "CompanyFocus",
    "EventFocus",
    "LogisticCalibration",
    "MagnitudeCalibration",
    "MagnitudeForecast",
    "NotEnoughCalibrationDataError",
    "NotEnoughMagnitudeDataError",
    "PerformanceSummary",
    "QuantileFit",
    "RefreshResult",
    "RelatedPrediction",
    "SectorFocus",
    "SentimentSummary",
    "TradeRecord",
    "aggregate_sentiment",
    "apply_logreg_classification",
    "article_contributions",
    "compose_company_focus",
    "compose_event_focus",
    "compose_sector_focus",
    "refresh_company_relationships",
    "refresh_sector_constituents",
    "classify",
    "cumulative_pnl_series",
    "daily_sentiment_index",
    "explain_prediction",
    "fit_logreg_calibration",
    "fit_quantile_calibration",
    "hit_rate_by_label",
    "hit_rate_by_target_kind",
    "hypothetical_trades",
    "performance_summary",
    "predict_all_sectors",
    "predict_all_stocks",
    "predict_market",
    "predict_sector",
    "predict_stock",
    "resolve_classifier_mode",
    "resolve_magnitude_mode",
    "retroactive_predict_many",
    "retroactive_predict_stock",
    "rolling_baseline",
    "rolling_hit_rate",
    "trades_dataframe",
]
