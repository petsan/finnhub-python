"""Runtime configuration loaded from environment variables.

We deliberately keep this module dependency-free so it can be imported from
both the Streamlit UI process and short-lived scripts (jobs, backtests, tests)
without surprises. The Settings dataclass is frozen so it cannot be mutated
after construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DB_URL = "sqlite:///finn_predictor.db"
DEFAULT_SCORER = "vader"
SUPPORTED_SCORERS = ("vader", "finbert")


@dataclass(frozen=True)
class Settings:
    """Application configuration snapshot.

    Attributes:
        finnhub_api_key: API token sent to api.finnhub.io. Never logged.
        database_url:    SQLAlchemy URL. Defaults to a file-backed SQLite db.
        request_timeout: HTTP timeout in seconds for Finnhub calls.
        rate_limit_per_minute: Soft cap on calls per minute (Finnhub free tier
            is ~60/min; we leave headroom).
        scorer_name: Which :class:`Scorer` implementation to use for live
            scoring. ``vader`` (default) needs no extra dependencies;
            ``finbert`` requires ``torch`` + ``transformers`` and downloads
            the ``ProsusAI/finbert`` weights on first use.
    """

    finnhub_api_key: str
    database_url: str = DEFAULT_DB_URL
    request_timeout: float = 15.0
    rate_limit_per_minute: int = 55
    scorer_name: str = DEFAULT_SCORER


def load_settings(
    env: dict[str, str] | None = None,
    *,
    require_api_key: bool = True,
) -> Settings:
    """Build a Settings from an env mapping (defaults to ``os.environ``).

    Args:
        env: Override mapping. Useful in tests; production callers pass nothing.
        require_api_key: When True (default), refuse to return a Settings
            with an empty key — CLI / batch scripts always want this.
            When False, return a Settings with an empty key; the caller is
            then responsible for supplying it through another channel
            (e.g. the Streamlit UI accepts it per-session in memory).

    Raises:
        RuntimeError: if ``require_api_key`` is True and ``FINNHUB_API_KEY``
            is not set. We refuse to start up silently with a placeholder
            because Finnhub would just 401 and we'd write nothing to the DB.
    """
    source = os.environ if env is None else env
    api_key = source.get("FINNHUB_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError(
            "FINNHUB_API_KEY is not set. Export it before starting "
            "finn-predictor (e.g. `export FINNHUB_API_KEY=...`)."
        )

    scorer_name = source.get("FINN_PREDICTOR_SCORER", DEFAULT_SCORER).strip().lower()
    if scorer_name not in SUPPORTED_SCORERS:
        raise RuntimeError(
            f"FINN_PREDICTOR_SCORER={scorer_name!r} is not supported. "
            f"Choose one of: {', '.join(SUPPORTED_SCORERS)}."
        )

    return Settings(
        finnhub_api_key=api_key,
        database_url=source.get("FINN_PREDICTOR_DB_URL", DEFAULT_DB_URL),
        request_timeout=float(source.get("FINN_PREDICTOR_TIMEOUT", "15")),
        rate_limit_per_minute=int(source.get("FINN_PREDICTOR_RATE_LIMIT", "55")),
        scorer_name=scorer_name,
    )
