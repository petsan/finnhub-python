"""Affinity-relationship helpers — curation and auto-seed.

Sits above :mod:`finn_predictor.predictor.focus` (the read-side, used
by the UI Focus tab) and below the UI / CLI. Three responsibilities:

* **Auto-seed COMPETITOR** — :func:`refresh_competitors` walks the
  existing PEER rows for a ticker, looks up each peer's
  ``finnhubIndustry`` via ``company_profile2``, and writes a sibling
  COMPETITOR row when the peer's industry matches the target's. PEER
  rows are not deleted — the underlying Finnhub data stays cached;
  COMPETITOR is a curation layer on top.
* **Manual curation** — :func:`promote_peer_to_competitor` and
  :func:`demote_competitor` let the operator override the auto-seed
  decision from the CLI or UI. Promotion writes a COMPETITOR row even
  when the industries don't match (operator's call); demotion removes
  the COMPETITOR row but leaves PEER untouched.
* **Ingest 13-F holders** — :func:`refresh_institutional_holders`
  pulls ``/institutional/ownership`` and caches the result as
  INSTITUTIONAL_HOLDER :class:`RelatedEntity` rows. (PR-5)

These are pure-ish (one DB roundtrip per call) so the affinity-blend
in PR-7 can read straight from :class:`RelatedEntity` without
re-fetching from Finnhub.

PR-6 will add ``refresh_investment_themes``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import FinnhubGateway, IngestionError
from finn_predictor.storage.models import RelatedEntity
from finn_predictor.storage.repo import (
    related_entities_for,
    upsert_related_entity,
)


# Maximum length of an institution name we'll store as a
# RelatedEntity.related_symbol. The column is String(64); we truncate
# longer names rather than reject them, since well-known holders
# already fit (e.g. "VANGUARD GROUP INC", "BLACKROCK INC", "STATE
# STREET CORP" are all well under 64).
INSTITUTION_NAME_MAX_LEN = 64


@dataclass(frozen=True)
class CompetitorRefreshResult:
    """Outcome of :func:`refresh_competitors`.

    Attributes:
        symbol: The target ticker the refresh was run for.
        competitors_added: Count of COMPETITOR rows newly written
            (idempotent — re-running on the same data inserts zero).
        peers_considered: Count of PEER rows examined this run.
        skipped_no_industry: Peers we couldn't classify because their
            profile lookup didn't return ``finnhubIndustry`` (free-tier
            quirks, delisted tickers, ETF peers, etc.).
        failures: Per-peer error records so the UI can surface what
            went wrong without aborting the whole sweep.
    """

    symbol: str
    competitors_added: int = 0
    peers_considered: int = 0
    skipped_no_industry: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


def _industry_of(profile: object) -> Optional[str]:
    """Pull ``finnhubIndustry`` out of a profile-2 payload, defensively.

    Returns None when:
      * ``profile`` isn't a dict (gateway returned something weird)
      * the field is missing
      * the field is empty or non-string
    """
    if not isinstance(profile, dict):
        return None
    raw = profile.get("finnhubIndustry")
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    return cleaned or None


def refresh_competitors(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbol: str,
) -> CompetitorRefreshResult:
    """Auto-seed COMPETITOR rows from existing PEER rows for ``symbol``.

    Decision rule: a peer is a COMPETITOR iff Finnhub's
    ``company_profile2`` reports the same ``finnhubIndustry`` as the
    target's. The target's own profile is fetched once; each peer's
    profile is fetched in turn. PEER rows are not modified.

    Idempotent — re-running with no upstream changes inserts zero new
    rows (``upsert_related_entity`` updates ``fetched_at`` on the
    existing row but doesn't duplicate it). Per-peer failure
    isolation: a profile-fetch error on one peer doesn't abort the
    rest of the sweep, just lands in ``failures``.

    Preconditions:
      * ``refresh_company_relationships`` has already run for
        ``symbol`` (so PEER rows exist). When the PEER set is empty,
        this returns a result with zero counts and no failures —
        callers don't need to special-case it.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    symbol = symbol.strip().upper()

    failures: list[dict[str, str]] = []

    # Target's own industry — one fetch, used to compare every peer.
    try:
        target_profile = gateway.company_profile2(symbol)
    except IngestionError as exc:
        # Without the target's industry there's nothing to compare
        # against — return immediately, but record the failure so the
        # UI can show it (rather than silently emitting an empty result).
        return CompetitorRefreshResult(
            symbol=symbol,
            failures=[{"op": f"company_profile2:{symbol}", "error": str(exc)}],
        )

    target_industry = _industry_of(target_profile)
    if target_industry is None:
        # The target has no industry tag — we can't auto-seed. This is
        # informational, not a failure (free-tier ETF tickers, indices,
        # delisted symbols all land here).
        return CompetitorRefreshResult(
            symbol=symbol,
            failures=[{
                "op": f"company_profile2:{symbol}",
                "error": "target has no finnhubIndustry; cannot auto-seed",
            }],
        )

    peers = related_entities_for(session, symbol, relationship="PEER")
    competitors_added = 0
    skipped_no_industry = 0

    for peer in peers:
        peer_symbol = peer.related_symbol
        try:
            peer_profile = gateway.company_profile2(peer_symbol)
        except IngestionError as exc:
            failures.append({
                "op": f"company_profile2:{peer_symbol}",
                "error": str(exc),
            })
            continue

        peer_industry = _industry_of(peer_profile)
        if peer_industry is None:
            skipped_no_industry += 1
            continue

        if peer_industry != target_industry:
            continue

        # Industry match → write COMPETITOR row. The (source, related,
        # relationship) tuple is the uniqueness key, so this coexists
        # with the PEER row peacefully.
        was_new = session.scalar(
            select(RelatedEntity).where(
                RelatedEntity.source_symbol == symbol,
                RelatedEntity.related_symbol == peer_symbol,
                RelatedEntity.relationship == "COMPETITOR",
            )
        ) is None
        upsert_related_entity(
            session,
            source_symbol=symbol,
            related_symbol=peer_symbol,
            relationship="COMPETITOR",
            rank=peer.rank,
            metadata_text=f"auto-seeded from PEER on industry {peer_industry!r}",
        )
        if was_new:
            competitors_added += 1

    return CompetitorRefreshResult(
        symbol=symbol,
        competitors_added=competitors_added,
        peers_considered=len(peers),
        skipped_no_industry=skipped_no_industry,
        failures=failures,
    )


def promote_peer_to_competitor(
    session: Session, *, symbol: str, peer_symbol: str
) -> RelatedEntity:
    """Manually mark ``peer_symbol`` as a COMPETITOR of ``symbol``.

    Doesn't require the PEER row to exist — the operator may want to
    pin a competitor relationship without one. Idempotent: re-promoting
    refreshes ``fetched_at`` on the existing COMPETITOR row.

    Validates both symbols via :func:`finn_predictor.ingestion.symbols.valid_ticker`
    so a typo can't write garbage into the relationship cache.
    """
    from finn_predictor.ingestion.symbols import valid_ticker

    if not symbol or not peer_symbol:
        raise ValueError("symbol and peer_symbol must be non-empty")
    src = symbol.strip().upper()
    dst = peer_symbol.strip().upper()
    if not valid_ticker(src):
        raise ValueError(f"symbol {symbol!r} is not a valid ticker")
    if not valid_ticker(dst):
        raise ValueError(f"peer_symbol {peer_symbol!r} is not a valid ticker")
    return upsert_related_entity(
        session,
        source_symbol=src,
        related_symbol=dst,
        relationship="COMPETITOR",
        rank=None,
        metadata_text="operator-curated",
    )


# ---------------------------------------------------------------------------
# Institutional holders (PR-5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstitutionalHoldersResult:
    """Outcome of :func:`refresh_institutional_holders`.

    Attributes:
        symbol: The target ticker the refresh was run for.
        holders_added: Count of INSTITUTIONAL_HOLDER rows inserted
            (i.e. holders not previously seen for this symbol).
        holders_refreshed: Count of pre-existing INSTITUTIONAL_HOLDER
            rows whose percentage / filing-date metadata was updated.
        failures: Per-call error records.
    """

    symbol: str
    holders_added: int = 0
    holders_refreshed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


def _normalise_institution_name(raw: object) -> Optional[str]:
    """Clean an institution name into a stable, length-bounded key.

    Returns None when the value isn't a usable string. Uppercases for
    stability across filings (Finnhub returns "Vanguard Group, Inc."
    one quarter and "VANGUARD GROUP INC" the next), strips outer
    whitespace, collapses internal whitespace, truncates to
    :data:`INSTITUTION_NAME_MAX_LEN`. The truncation is deterministic
    so two filings of the same holder produce the same key.
    """
    if not isinstance(raw, str):
        return None
    cleaned = " ".join(raw.split()).upper()
    if not cleaned:
        return None
    return cleaned[:INSTITUTION_NAME_MAX_LEN]


def _coerce_float(raw: object) -> Optional[float]:
    """Best-effort numeric coercion for Finnhub payload fields."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def refresh_institutional_holders(
    session: Session,
    gateway: FinnhubGateway,
    *,
    symbol: str,
    lookback_days: int = 180,
    limit: int = 25,
    today: Optional[datetime] = None,
) -> InstitutionalHoldersResult:
    """Pull 13-F holders for ``symbol`` and cache as INSTITUTIONAL_HOLDER rows.

    The endpoint is gated on most paid plans; :class:`IngestionError`
    (403 / 429 / network) is captured into ``failures`` rather than
    propagated, matching the resilient-ingest convention used by
    ``refresh_company_relationships``.

    Payload semantics — the upstream returns
    ``{"symbol": ..., "data": [ {name, share, value, percentage,
    filingDate, ...}, ... ]}``. We persist the top ``limit`` entries
    by Finnhub's natural order (by share value descending). The
    holder's ownership % and filing date land in ``metadata_text``
    as JSON so callers can sort / display without re-fetching.

    ``rank`` is set to the position in Finnhub's response so
    :func:`related_entities_for`'s ascending-by-rank sort naturally
    surfaces the biggest holder first.
    """
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    target = symbol.strip().upper()

    if limit <= 0:
        return InstitutionalHoldersResult(symbol=target)

    now = today or datetime.now(timezone.utc)
    to_iso = now.strftime("%Y-%m-%d")
    from_iso = (now - timedelta(days=max(1, int(lookback_days)))).strftime(
        "%Y-%m-%d"
    )

    try:
        payload = gateway.institutional_ownership(target, from_iso, to_iso)
    except IngestionError as exc:
        return InstitutionalHoldersResult(
            symbol=target,
            failures=[{
                "op": f"institutional_ownership:{target}",
                "error": str(exc),
            }],
        )

    rows = _parse_institutional_payload(payload)
    if not rows:
        return InstitutionalHoldersResult(symbol=target)

    holders_added = 0
    holders_refreshed = 0
    seen_names: set[str] = set()

    for position, raw in enumerate(rows[:limit]):
        name = _normalise_institution_name(raw.get("name"))
        if name is None or name in seen_names:
            # Empty / duplicate name in the payload — drop. Finnhub
            # occasionally repeats a filer across filing types; we keep
            # the highest-share row (which comes first by virtue of
            # the sort) and skip the rest.
            continue
        seen_names.add(name)

        pct = _coerce_float(raw.get("percentage"))
        share = _coerce_float(raw.get("share"))
        value = _coerce_float(raw.get("value"))
        filing_date = raw.get("filingDate") if isinstance(
            raw.get("filingDate"), str
        ) else None

        metadata = json.dumps(
            {
                "percentage": pct,
                "share": share,
                "value": value,
                "filing_date": filing_date,
            },
            sort_keys=True,
        )

        existing = session.scalar(
            select(RelatedEntity).where(
                RelatedEntity.source_symbol == target,
                RelatedEntity.related_symbol == name,
                RelatedEntity.relationship == "INSTITUTIONAL_HOLDER",
            )
        )
        upsert_related_entity(
            session,
            source_symbol=target,
            related_symbol=name,
            relationship="INSTITUTIONAL_HOLDER",
            rank=position,
            metadata_text=metadata,
        )
        if existing is None:
            holders_added += 1
        else:
            holders_refreshed += 1

    return InstitutionalHoldersResult(
        symbol=target,
        holders_added=holders_added,
        holders_refreshed=holders_refreshed,
        failures=[],
    )


def _parse_institutional_payload(payload: object) -> list[dict]:
    """Defensively extract the holder list from Finnhub's payload.

    The upstream returns ``{"symbol": ..., "data": [...]}`` on success
    and various other shapes on edge cases (empty plan, no filings,
    delisted ticker). This helper returns an empty list for anything
    that isn't a dict with a ``data`` list of dicts — keeps the
    refresh path crash-free regardless of upstream weirdness.
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Investment themes (PR-6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThemeRefreshResult:
    """Outcome of :func:`refresh_investment_themes`.

    Attributes:
        themes_processed: Codes we attempted to ingest.
        themes_added: New :class:`InvestmentTheme` rows inserted.
        members_added: New THEME_MEMBER edges across all themes.
        failures: Per-theme error records.
    """

    themes_processed: list[str] = field(default_factory=list)
    themes_added: int = 0
    members_added: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


def _humanise_theme_code(code: str) -> str:
    """Convert ``cyberSecurity`` → ``Cyber Security`` for display."""
    # Insert a space before each uppercase letter that follows a lowercase
    # one, then title-case. Operator can override via the CLI.
    import re as _re
    spaced = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", code)
    return spaced.title()


def refresh_investment_themes(
    session: Session,
    gateway: FinnhubGateway,
    *,
    theme_codes: Optional[list[str]] = None,
) -> ThemeRefreshResult:
    """Pull constituents for each theme code and cache as THEME_MEMBER rows.

    ``theme_codes`` defaults to
    :data:`finn_predictor.predictor.themes.DEFAULT_THEME_CODES`. Per-theme
    failure isolation: a 403 on one theme doesn't block the rest.

    For each code:

    1. Upsert an :class:`InvestmentTheme` row (idempotent — re-running
       just refreshes ``fetched_at``).
    2. Call ``stock_investment_theme(code)``.
    3. Upsert one :class:`RelatedEntity(relationship="THEME_MEMBER")`
       per constituent, with ``source_symbol = theme_code`` and
       ``related_symbol = ticker``.

    Returns a :class:`ThemeRefreshResult` summarising what was touched.
    """
    from finn_predictor.predictor.themes import DEFAULT_THEME_CODES
    from finn_predictor.storage.models import InvestmentTheme

    codes = list(theme_codes) if theme_codes else list(DEFAULT_THEME_CODES)

    themes_added = 0
    members_added = 0
    failures: list[dict[str, str]] = []
    processed: list[str] = []

    for code in codes:
        code = code.strip()
        if not code:
            continue
        processed.append(code)

        # 1. Upsert the InvestmentTheme row.
        existing_theme = session.scalar(
            select(InvestmentTheme).where(InvestmentTheme.theme_code == code)
        )
        if existing_theme is None:
            session.add(
                InvestmentTheme(
                    theme_code=code,
                    name=_humanise_theme_code(code),
                    fetched_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
            themes_added += 1
        else:
            existing_theme.fetched_at = datetime.now(timezone.utc)
            session.commit()

        # 2. Fetch constituents.
        try:
            payload = gateway.stock_investment_theme(code)
        except IngestionError as exc:
            failures.append({
                "op": f"stock_investment_theme:{code}",
                "error": str(exc),
            })
            continue

        constituents = _parse_theme_payload(payload)
        if not constituents:
            continue

        # 3. Upsert membership edges.
        seen: set[str] = set()
        for position, raw_sym in enumerate(constituents):
            sym = (raw_sym or "").strip().upper()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            was_new = session.scalar(
                select(RelatedEntity).where(
                    RelatedEntity.source_symbol == code,
                    RelatedEntity.related_symbol == sym,
                    RelatedEntity.relationship == "THEME_MEMBER",
                )
            ) is None
            upsert_related_entity(
                session,
                source_symbol=code,
                related_symbol=sym,
                relationship="THEME_MEMBER",
                rank=position,
            )
            if was_new:
                members_added += 1

    return ThemeRefreshResult(
        themes_processed=processed,
        themes_added=themes_added,
        members_added=members_added,
        failures=failures,
    )


def _parse_theme_payload(payload: object) -> list[str]:
    """Extract the constituent ticker list from a Finnhub theme response.

    Defensive: returns an empty list for anything that isn't a dict
    with a ``symbols`` list. Each entry is expected to be a string,
    but the helper tolerates dicts with a ``symbol`` key (some
    Finnhub variants return ``[{"symbol": "AAPL"}, ...]``).
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("symbols")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            sym = item.get("symbol")
            if isinstance(sym, str):
                out.append(sym)
    return out


def add_investment_theme(
    session: Session,
    *,
    theme_code: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> "InvestmentTheme":
    """Register a theme without an immediate Finnhub fetch.

    Useful for operator-curated themes that pre-exist the ``refresh``
    sweep. Idempotent — re-registering an existing code updates the
    ``name`` / ``description`` and refreshes ``fetched_at``.
    """
    from finn_predictor.storage.models import InvestmentTheme

    code = (theme_code or "").strip()
    if not code:
        raise ValueError("theme_code must be non-empty")

    cleaned_name = (name or "").strip() or _humanise_theme_code(code)
    cleaned_desc = (description or "").strip() or None

    existing = session.scalar(
        select(InvestmentTheme).where(InvestmentTheme.theme_code == code)
    )
    if existing is not None:
        existing.name = cleaned_name
        existing.description = cleaned_desc
        existing.fetched_at = datetime.now(timezone.utc)
        session.commit()
        return existing

    row = InvestmentTheme(
        theme_code=code,
        name=cleaned_name,
        description=cleaned_desc,
        fetched_at=datetime.now(timezone.utc),
    )
    session.add(row)
    session.commit()
    return row


def demote_competitor(
    session: Session, *, symbol: str, peer_symbol: str
) -> bool:
    """Remove a COMPETITOR row (if any). Returns True iff a row was deleted.

    Idempotent — a missing row returns False rather than raising, so
    "Undo" buttons can be safely double-clicked.
    """
    if not symbol or not peer_symbol:
        raise ValueError("symbol and peer_symbol must be non-empty")
    src = symbol.strip().upper()
    dst = peer_symbol.strip().upper()
    row = session.scalar(
        select(RelatedEntity).where(
            RelatedEntity.source_symbol == src,
            RelatedEntity.related_symbol == dst,
            RelatedEntity.relationship == "COMPETITOR",
        )
    )
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True
