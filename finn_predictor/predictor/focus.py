"""Focus on a company / sector / event.

Three composers, all pure functions over the session — no writes
(except :func:`refresh_company_relationships`, which is explicitly
about cache population from Finnhub):

* :func:`compose_company_focus` returns everything we know about a
  ticker: its own latest Prediction, its peers' Predictions, supply-
  chain neighbours (when the plan allows), its sector membership +
  sector ETF prediction, plus recent articles tagged with that ticker.
* :func:`compose_sector_focus` returns the sector ETF's Prediction + its
  cached top constituents' Predictions + recent articles tagged with any
  of those tickers.
* :func:`compose_event_focus` is free-text: it searches the news table
  for articles whose headline + summary match the query, aggregates
  sentiment across the matches, and computes an "implied call".

The relationship cache lives in :class:`RelatedEntity` so we can render
focus pages without re-hitting the API every time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import FinnhubGateway, IngestionError
from finn_predictor.predictor.aggregate import aggregate_sentiment
from finn_predictor.predictor.market import (
    MIN_BASELINE_SIGMA,
    THRESHOLD_SIGMA,
    classify,
)
from finn_predictor.storage.models import (
    NewsArticle,
    Prediction,
    RelatedEntity,
    Sector,
    SentimentScore,
)
from finn_predictor.storage.repo import (
    related_entities_for,
    upsert_related_entity,
)


# ---------------- Bundle dataclasses ----------------


@dataclass(frozen=True)
class RelatedPrediction:
    """A related entity (ticker) + its most recent prediction if any."""

    related_symbol: str
    relationship: str
    rank: Optional[int]
    metadata_text: Optional[str]
    prediction: Optional[Prediction]


@dataclass(frozen=True)
class CompanyFocus:
    symbol: str
    company_name: str
    sector_code: Optional[str]
    sector_etf: Optional[str]
    own_prediction: Optional[Prediction]
    sector_prediction: Optional[Prediction]
    peers: list[RelatedPrediction] = field(default_factory=list)
    suppliers: list[RelatedPrediction] = field(default_factory=list)
    customers: list[RelatedPrediction] = field(default_factory=list)
    recent_articles: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class SectorFocus:
    sector_code: str
    sector_name: str
    etf_symbol: str
    own_prediction: Optional[Prediction]
    constituents: list[RelatedPrediction] = field(default_factory=list)
    recent_articles: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class EventFocus:
    query: str
    lookback_days: int
    matched_articles: list[dict]
    target_breakdown: list[dict]   # per-ticker counts + mean sentiment
    aggregate_sentiment: float
    article_count: int
    implied_label: str             # UP / DOWN / FLAT
    implied_confidence: float


# ---------------- Helpers ----------------


def _latest_prediction(
    session: Session, target_symbol: str
) -> Optional[Prediction]:
    return session.scalar(
        select(Prediction)
        .where(Prediction.target_symbol == target_symbol)
        .order_by(Prediction.prediction_date.desc())
        .limit(1)
    )


def _attach_predictions(
    session: Session, related: Iterable[RelatedEntity]
) -> list[RelatedPrediction]:
    """For each related entity, attach the most recent matching Prediction."""
    out: list[RelatedPrediction] = []
    for r in related:
        pred = _latest_prediction(session, r.related_symbol)
        out.append(
            RelatedPrediction(
                related_symbol=r.related_symbol,
                relationship=r.relationship,
                rank=r.rank,
                metadata_text=r.metadata_text,
                prediction=pred,
            )
        )
    return out


def _recent_articles_for_symbols(
    session: Session,
    symbols: Iterable[str],
    *,
    limit: int = 20,
    days: int = 30,
    model_version: Optional[str] = None,
) -> list[dict]:
    """Recent NewsArticle rows whose ``symbol`` is in ``symbols``.

    Each row carries the per-article sentiment for ``model_version``
    if provided, else NaN.
    """
    syms = [s for s in symbols if s]
    if not syms:
        return []
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = (
        select(NewsArticle)
        .where(NewsArticle.symbol.in_(syms), NewsArticle.published_at >= since)
        .order_by(NewsArticle.published_at.desc())
        .limit(limit)
    )
    arts = list(session.scalars(stmt))
    if not arts:
        return []
    score_map: dict[int, float] = {}
    if model_version is not None:
        rows = session.execute(
            select(SentimentScore.article_id, SentimentScore.score).where(
                SentimentScore.article_id.in_([a.id for a in arts]),
                SentimentScore.model_version == model_version,
            )
        ).all()
        score_map = {aid: float(sc) for aid, sc in rows}
    return [
        {
            "published_at": a.published_at,
            "headline": a.headline,
            "source": a.source,
            "symbol": a.symbol or "*",
            "url": a.url or "",
            "sentiment": score_map.get(a.id, float("nan")),
        }
        for a in arts
    ]


def _sector_for_symbol(
    session: Session, symbol: str
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Best-effort lookup of (sector_code, sector_name, etf_symbol).

    Today we only know a ticker's sector if we've cached an ETF_HOLDING
    relationship pointing at it. Returns ``(None, None, None)`` otherwise.
    """
    rel = session.scalar(
        select(RelatedEntity).where(
            RelatedEntity.related_symbol == symbol,
            RelatedEntity.relationship == "ETF_HOLDING",
        )
    )
    if rel is None:
        return None, None, None
    sec = session.scalar(
        select(Sector).where(Sector.etf_symbol == rel.source_symbol)
    )
    if sec is None:
        return None, None, rel.source_symbol
    return sec.code, sec.name, sec.etf_symbol


# ---------------- Compose: company ----------------


def compose_company_focus(
    session: Session,
    symbol: str,
    *,
    article_limit: int = 20,
    article_days: int = 30,
    model_version: Optional[str] = None,
) -> CompanyFocus:
    """Bundle the prediction + cached relationships + recent news for a ticker."""
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    symbol = symbol.strip().upper()

    own = _latest_prediction(session, symbol)
    peers = _attach_predictions(
        session, related_entities_for(session, symbol, relationship="PEER")
    )
    suppliers = _attach_predictions(
        session, related_entities_for(session, symbol, relationship="SUPPLIER")
    )
    customers = _attach_predictions(
        session, related_entities_for(session, symbol, relationship="CUSTOMER")
    )

    sector_code, sector_name, etf_symbol = _sector_for_symbol(session, symbol)
    sector_pred = _latest_prediction(session, etf_symbol) if etf_symbol else None

    # Universe of tickers we want articles for: the subject + its peers +
    # supply-chain neighbours. We keep the article window short and bounded.
    universe = {symbol}
    universe.update(p.related_symbol for p in peers)
    universe.update(p.related_symbol for p in suppliers)
    universe.update(p.related_symbol for p in customers)
    recent = _recent_articles_for_symbols(
        session, universe,
        limit=article_limit, days=article_days, model_version=model_version,
    )

    return CompanyFocus(
        symbol=symbol,
        company_name=symbol,  # caller can expand via symbol_names.expand_symbol
        sector_code=sector_code,
        sector_etf=etf_symbol,
        own_prediction=own,
        sector_prediction=sector_pred,
        peers=peers,
        suppliers=suppliers,
        customers=customers,
        recent_articles=recent,
    )


# ---------------- Compose: sector ----------------


def compose_sector_focus(
    session: Session,
    sector_code: str,
    *,
    article_limit: int = 20,
    article_days: int = 30,
    model_version: Optional[str] = None,
) -> Optional[SectorFocus]:
    """Bundle for a sector: ETF prediction + cached top constituents + news."""
    if not sector_code:
        raise ValueError("sector_code must be a non-empty string")
    sector = session.scalar(
        select(Sector).where(Sector.code == sector_code)
    )
    if sector is None:
        return None

    own = _latest_prediction(session, sector.etf_symbol)
    constituents = _attach_predictions(
        session,
        related_entities_for(
            session, sector.etf_symbol, relationship="ETF_HOLDING"
        ),
    )
    universe = {c.related_symbol for c in constituents}
    universe.add(sector.etf_symbol)
    recent = _recent_articles_for_symbols(
        session, universe,
        limit=article_limit, days=article_days, model_version=model_version,
    )
    return SectorFocus(
        sector_code=sector.code,
        sector_name=sector.name,
        etf_symbol=sector.etf_symbol,
        own_prediction=own,
        constituents=constituents,
        recent_articles=recent,
    )


# ---------------- Compose: event (free-text) ----------------


_WORD_BOUNDARY = re.compile(r"[^\w]+")


def _normalised_query_terms(query: str) -> list[str]:
    """Split the query into non-empty lowercase terms."""
    if not query:
        return []
    return [t.lower() for t in _WORD_BOUNDARY.split(query) if t]


def compose_event_focus(
    session: Session,
    query: str,
    *,
    lookback_days: int = 14,
    article_limit: int = 50,
    model_version: Optional[str] = None,
) -> EventFocus:
    """Free-text search for an event/theme across recent news.

    All non-empty tokens in ``query`` must appear (case-insensitively) in
    either the headline or the summary. Returned bundle includes:

      * matched articles with their per-article sentiment
      * per-ticker breakdown (count + mean sentiment)
      * aggregate sentiment + an "implied call" using the same classifier
        used by the daily predictor (threshold = ±0.5σ, but here the
        baseline is just zero — we don't have a query-specific baseline,
        so the implied call is sign-of-mean: UP if > 0.1, DOWN if < -0.1,
        else FLAT).
    """
    terms = _normalised_query_terms(query)
    if not terms:
        return EventFocus(
            query=query,
            lookback_days=lookback_days,
            matched_articles=[],
            target_breakdown=[],
            aggregate_sentiment=0.0,
            article_count=0,
            implied_label="FLAT",
            implied_confidence=0.0,
        )

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    stmt = select(NewsArticle).where(NewsArticle.published_at >= since)
    for t in terms:
        like = f"%{t}%"
        stmt = stmt.where(
            or_(
                NewsArticle.headline.ilike(like),
                NewsArticle.summary.ilike(like),
            )
        )
    stmt = stmt.order_by(NewsArticle.published_at.desc()).limit(article_limit)
    arts = list(session.scalars(stmt))

    score_map: dict[int, float] = {}
    if model_version is not None and arts:
        score_rows = session.execute(
            select(SentimentScore.article_id, SentimentScore.score).where(
                SentimentScore.article_id.in_([a.id for a in arts]),
                SentimentScore.model_version == model_version,
            )
        ).all()
        score_map = {aid: float(sc) for aid, sc in score_rows}

    matched_dicts = [
        {
            "published_at": a.published_at,
            "headline": a.headline,
            "source": a.source,
            "symbol": a.symbol or "*",
            "url": a.url or "",
            "sentiment": score_map.get(a.id, float("nan")),
        }
        for a in arts
    ]

    # Per-ticker breakdown
    by_ticker: dict[str, list[float]] = {}
    for a in arts:
        sym = a.symbol or "*"
        if a.id in score_map:
            by_ticker.setdefault(sym, []).append(score_map[a.id])
        else:
            by_ticker.setdefault(sym, [])
    breakdown = []
    for sym, scores in sorted(
        by_ticker.items(), key=lambda kv: (-len([a for a in arts if (a.symbol or "*") == kv[0]]), kv[0])
    ):
        total_for_sym = sum(1 for a in arts if (a.symbol or "*") == sym)
        mean_sent = sum(scores) / len(scores) if scores else float("nan")
        breakdown.append(
            {
                "symbol": sym,
                "articles": total_for_sym,
                "scored": len(scores),
                "mean_sentiment": mean_sent,
            }
        )

    # Aggregate sentiment and implied call
    scores_flat = [s for s in score_map.values()]
    summary = aggregate_sentiment(scores_flat)
    mean = summary.weighted_mean
    # Sign-of-mean classifier (no rolling baseline for ad-hoc queries):
    # the dead-band is +/- 0.1 on the [-1, +1] sentiment scale.
    if mean > 0.1:
        implied = "UP"
    elif mean < -0.1:
        implied = "DOWN"
    else:
        implied = "FLAT"
    implied_conf = min(1.0, abs(mean) / 0.5)

    return EventFocus(
        query=query,
        lookback_days=lookback_days,
        matched_articles=matched_dicts,
        target_breakdown=breakdown,
        aggregate_sentiment=mean,
        article_count=len(arts),
        implied_label=implied,
        implied_confidence=implied_conf,
    )


# ---------------- Cache refresh (Finnhub-backed) ----------------


@dataclass(frozen=True)
class RefreshResult:
    """Counts + failures from a single relationship-refresh run."""

    symbol: str
    peers_added: int
    suppliers_added: int
    customers_added: int
    holdings_added: int
    failures: list[dict[str, str]] = field(default_factory=list)


def _parse_supply_chain_payload(payload: object) -> tuple[list[dict], list[dict]]:
    """Pull suppliers + customers out of Finnhub's supply-chain response.

    The endpoint shape (per Finnhub docs) is::

        {"symbol": "AAPL", "data": [
            {"symbol": "TSM", "name": "Taiwan Semi", "relation": "Supplier", ...},
            {"symbol": "MU",  "name": "Micron",      "relation": "Supplier", ...},
            {"symbol": "T",   "name": "AT&T",        "relation": "Customer", ...},
            ...
        ]}

    We only keep entries whose ``symbol`` looks like a real ticker.
    """
    if not isinstance(payload, dict):
        return [], []
    data = payload.get("data") or []
    suppliers: list[dict] = []
    customers: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sym = (item.get("symbol") or "").strip()
        if not sym:
            continue
        rel = (item.get("relation") or item.get("relationship") or "").lower()
        bucket = suppliers if "suppl" in rel else customers if "cust" in rel else None
        if bucket is not None:
            bucket.append(item)
    return suppliers, customers


def _parse_etf_holdings_payload(payload: object) -> list[dict]:
    """Top constituents from ``etfs_holdings`` output."""
    if not isinstance(payload, dict):
        return []
    holdings = payload.get("holdings") or []
    out = []
    for item in holdings:
        if not isinstance(item, dict):
            continue
        sym = (item.get("symbol") or "").strip()
        if sym:
            out.append(item)
    return out


def refresh_company_relationships(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbol: str,
) -> RefreshResult:
    """Pull peers + supply chain for ``symbol`` and cache in related_entities.

    Each Finnhub call is wrapped so a 403 on supply-chain (paid endpoint)
    doesn't prevent the peer cache from updating.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    failures: list[dict[str, str]] = []
    peers_added = 0
    suppliers_added = 0
    customers_added = 0

    try:
        peers = gateway.company_peers(symbol) or []
        for i, p in enumerate(peers):
            if not p or not isinstance(p, str):
                continue
            upsert_related_entity(
                session,
                source_symbol=symbol,
                related_symbol=p.strip().upper(),
                relationship="PEER",
                rank=i,
            )
            peers_added += 1
    except IngestionError as exc:
        failures.append({"op": f"company_peers:{symbol}", "error": str(exc)})

    try:
        sc = gateway.stock_supply_chain(symbol)
        suppliers, customers = _parse_supply_chain_payload(sc)
        for i, item in enumerate(suppliers):
            upsert_related_entity(
                session,
                source_symbol=symbol,
                related_symbol=(item.get("symbol") or "").strip().upper(),
                relationship="SUPPLIER",
                rank=i,
                metadata_text=item.get("name") or None,
            )
            suppliers_added += 1
        for i, item in enumerate(customers):
            upsert_related_entity(
                session,
                source_symbol=symbol,
                related_symbol=(item.get("symbol") or "").strip().upper(),
                relationship="CUSTOMER",
                rank=i,
                metadata_text=item.get("name") or None,
            )
            customers_added += 1
    except IngestionError as exc:
        failures.append({"op": f"stock_supply_chain:{symbol}", "error": str(exc)})

    return RefreshResult(
        symbol=symbol,
        peers_added=peers_added,
        suppliers_added=suppliers_added,
        customers_added=customers_added,
        holdings_added=0,
        failures=failures,
    )


def refresh_sector_constituents(
    session: Session,
    gateway: FinnhubGateway,
    *,
    etf_symbol: str,
    limit: int = 25,
) -> RefreshResult:
    """Pull top constituents of ``etf_symbol`` and cache as ETF_HOLDING rows."""
    if not etf_symbol:
        raise ValueError("etf_symbol must be a non-empty string")
    etf_symbol = etf_symbol.strip().upper()
    failures: list[dict[str, str]] = []
    holdings_added = 0

    try:
        payload = gateway.etfs_holdings(etf_symbol)
        holdings = _parse_etf_holdings_payload(payload)[:limit]
        for i, item in enumerate(holdings):
            upsert_related_entity(
                session,
                source_symbol=etf_symbol,
                related_symbol=(item.get("symbol") or "").strip().upper(),
                relationship="ETF_HOLDING",
                rank=i,
                metadata_text=item.get("name") or None,
            )
            holdings_added += 1
    except IngestionError as exc:
        failures.append({"op": f"etfs_holdings:{etf_symbol}", "error": str(exc)})

    return RefreshResult(
        symbol=etf_symbol,
        peers_added=0,
        suppliers_added=0,
        customers_added=0,
        holdings_added=holdings_added,
        failures=failures,
    )
