"""News ingestion — converts Finnhub payloads to :class:`NewsArticle` rows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import FinnhubGateway
from finn_predictor.storage.models import NewsArticle
from finn_predictor.storage.repo import upsert_articles


def _coerce_article(payload: dict, *, category: str, symbol: str | None) -> NewsArticle:
    """Map a Finnhub article dict to a :class:`NewsArticle` ORM instance.

    Finnhub returns ``datetime`` as a unix epoch (seconds, UTC).
    """
    finnhub_id = int(payload.get("id") or 0)
    ts = int(payload.get("datetime") or 0)
    published = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)
    return NewsArticle(
        finnhub_id=finnhub_id,
        category=category,
        symbol=symbol,
        headline=str(payload.get("headline") or "").strip(),
        summary=str(payload.get("summary") or "").strip(),
        source=str(payload.get("source") or "").strip(),
        url=str(payload.get("url") or "").strip(),
        published_at=published,
    )


def ingest_general_news(
    session: Session,
    gateway: FinnhubGateway,
    *,
    category: str = "general",
    min_id: int = 0,
) -> int:
    """Pull whole-market news and persist new articles. Returns insert count."""
    payloads: Sequence[dict] = gateway.general_news(category=category, min_id=min_id)
    articles = [
        _coerce_article(p, category=category, symbol=None)
        for p in payloads
        if int(p.get("id") or 0) > 0
    ]
    return upsert_articles(session, articles)


def ingest_company_news(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
) -> int:
    """Pull per-company news for ``symbol`` between ``start`` and ``end``."""
    payloads: Sequence[dict] = gateway.company_news(
        symbol=symbol,
        _from=start.date().isoformat(),
        to=end.date().isoformat(),
    )
    articles = [
        _coerce_article(p, category="company", symbol=symbol)
        for p in payloads
        if int(p.get("id") or 0) > 0
    ]
    return upsert_articles(session, articles)
