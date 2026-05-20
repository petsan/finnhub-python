"""Curated ticker → GICS-sector membership.

This module backs the **free-tier workaround** for sector aggregation:
Finnhub's ``/etf/holdings`` is paid-only, so users on the free tier
can't populate ``RelatedEntity(ETF_HOLDING)`` rows the normal way.

The mapping here is a static, public-knowledge approximation of the
top constituents of the 11 Sector Select SPDR ETFs (XLK / XLE / XLF /
XLV / XLY / XLP / XLI / XLB / XLU / XLRE / XLC). It is intentionally
small (mega-caps + a few obvious mid-caps per sector) — the goal is
to cover the tickers a user is most likely to list in the sidebar,
not to be an authoritative ETF-holdings table.

How it's used:

* :func:`sector_for_ticker` resolves a ticker to a sector ``code``
  (one of ``DEFAULT_SECTORS`` codes — ``"TECH"`` etc.) or ``None``
  for the long tail of unmapped symbols.
* :func:`sector_universe_from_tickers` builds the
  ``{sector_code: [tickers]}`` shape that
  :func:`predict_all_sectors` consumes, derived from the user's
  list of company tickers.
* The daily-ingest job merges this with whatever happens to be in
  ``RelatedEntity(ETF_HOLDING)`` (in case some sectors *are*
  populated via a paid key), so the two sources combine cleanly
  rather than override each other.

Updating the table is a one-line edit. We keep this in code rather
than the DB so a fresh deploy works immediately — no manual seeding
required, no race with the first ingest, and the data is reviewable
in version control alongside the rest of the codebase.
"""

from __future__ import annotations

from typing import Iterable, Optional


# Ticker → sector-code. Codes match the ``code`` column of
# DEFAULT_SECTORS in :mod:`finn_predictor.storage.repo`, NOT the ETF
# symbol — callers can resolve code → ETF via the ``Sector`` table.
#
# Curation policy:
#   * Top ~10 holdings of each Sector SPDR ETF as of mid-2025 (~110
#     tickers total). Heavy mega-cap bias on purpose; that's also
#     what shows up in user sidebars.
#   * No ADRs / dual listings (no ".B" share classes, no "RY.TO"
#     style foreign tickers) — the news ingest path doesn't deal with
#     those cleanly today and surfacing them here would just confuse
#     the synthesis.
#   * Single sector per ticker: companies that span sectors (BRK.B
#     across most of finance + industrials) go to their *primary*
#     SPDR classification, matching what an index provider would do.
SECTOR_MEMBERSHIP: dict[str, str] = {
    # ---- Information Technology (TECH / XLK) ----
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "AVGO": "TECH",
    "ORCL": "TECH", "CRM": "TECH", "ADBE": "TECH", "CSCO": "TECH",
    "AMD": "TECH", "QCOM": "TECH", "ACN": "TECH", "INTU": "TECH",
    "IBM": "TECH", "TXN": "TECH", "NOW": "TECH", "AMAT": "TECH",
    "INTC": "TECH", "MU": "TECH", "ADI": "TECH", "PANW": "TECH",
    "LRCX": "TECH", "KLAC": "TECH", "PLTR": "TECH",

    # ---- Energy (ENERGY / XLE) ----
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY", "EOG": "ENERGY",
    "SLB": "ENERGY", "PSX": "ENERGY", "MPC": "ENERGY", "VLO": "ENERGY",
    "OXY": "ENERGY", "PXD": "ENERGY", "WMB": "ENERGY", "KMI": "ENERGY",
    "OKE": "ENERGY",

    # ---- Financials (FIN / XLF) ----
    "JPM": "FIN", "BAC": "FIN", "WFC": "FIN", "GS": "FIN",
    "MS": "FIN", "BLK": "FIN", "C": "FIN", "SCHW": "FIN",
    "AXP": "FIN", "SPGI": "FIN", "MMC": "FIN", "CB": "FIN",
    "PGR": "FIN", "BRK.B": "FIN", "USB": "FIN", "PNC": "FIN",
    "TFC": "FIN", "COF": "FIN", "AON": "FIN", "ICE": "FIN",

    # ---- Health Care (HEALTH / XLV) ----
    "LLY": "HEALTH", "JNJ": "HEALTH", "UNH": "HEALTH", "MRK": "HEALTH",
    "ABBV": "HEALTH", "TMO": "HEALTH", "ABT": "HEALTH", "PFE": "HEALTH",
    "DHR": "HEALTH", "AMGN": "HEALTH", "BMY": "HEALTH", "GILD": "HEALTH",
    "ISRG": "HEALTH", "MDT": "HEALTH", "VRTX": "HEALTH", "ELV": "HEALTH",
    "CVS": "HEALTH", "REGN": "HEALTH", "BSX": "HEALTH",

    # ---- Consumer Discretionary (DISCRETIONARY / XLY) ----
    "AMZN": "DISCRETIONARY", "TSLA": "DISCRETIONARY", "HD": "DISCRETIONARY",
    "MCD": "DISCRETIONARY", "NKE": "DISCRETIONARY", "LOW": "DISCRETIONARY",
    "BKNG": "DISCRETIONARY", "TJX": "DISCRETIONARY", "SBUX": "DISCRETIONARY",
    "CMG": "DISCRETIONARY", "ORLY": "DISCRETIONARY", "AZO": "DISCRETIONARY",
    "MAR": "DISCRETIONARY", "F": "DISCRETIONARY", "GM": "DISCRETIONARY",

    # ---- Consumer Staples (STAPLES / XLP) ----
    "PG": "STAPLES", "COST": "STAPLES", "WMT": "STAPLES", "KO": "STAPLES",
    "PEP": "STAPLES", "PM": "STAPLES", "MO": "STAPLES", "MDLZ": "STAPLES",
    "CL": "STAPLES", "TGT": "STAPLES", "KMB": "STAPLES", "GIS": "STAPLES",
    "SYY": "STAPLES", "STZ": "STAPLES", "EL": "STAPLES",

    # ---- Industrials (INDUSTRIAL / XLI) ----
    "GE": "INDUSTRIAL", "CAT": "INDUSTRIAL", "RTX": "INDUSTRIAL",
    "HON": "INDUSTRIAL", "UNP": "INDUSTRIAL", "BA": "INDUSTRIAL",
    "LMT": "INDUSTRIAL", "DE": "INDUSTRIAL", "UPS": "INDUSTRIAL",
    "ETN": "INDUSTRIAL", "ADP": "INDUSTRIAL", "MMM": "INDUSTRIAL",
    "GD": "INDUSTRIAL", "NOC": "INDUSTRIAL", "WM": "INDUSTRIAL",
    "FDX": "INDUSTRIAL", "CSX": "INDUSTRIAL", "NSC": "INDUSTRIAL",

    # ---- Materials (MATERIAL / XLB) ----
    "LIN": "MATERIAL", "SHW": "MATERIAL", "APD": "MATERIAL",
    "ECL": "MATERIAL", "FCX": "MATERIAL", "NEM": "MATERIAL",
    "DOW": "MATERIAL", "DD": "MATERIAL", "NUE": "MATERIAL", "CTVA": "MATERIAL",

    # ---- Utilities (UTILITIES / XLU) ----
    "NEE": "UTILITIES", "SO": "UTILITIES", "DUK": "UTILITIES", "SRE": "UTILITIES",
    "AEP": "UTILITIES", "D": "UTILITIES", "EXC": "UTILITIES", "XEL": "UTILITIES",
    "ED": "UTILITIES", "PEG": "UTILITIES", "WEC": "UTILITIES",

    # ---- Real Estate (REAL_ESTATE / XLRE) ----
    "PLD": "REAL_ESTATE", "AMT": "REAL_ESTATE", "EQIX": "REAL_ESTATE",
    "WELL": "REAL_ESTATE", "DLR": "REAL_ESTATE", "SPG": "REAL_ESTATE",
    "PSA": "REAL_ESTATE", "O": "REAL_ESTATE", "CCI": "REAL_ESTATE",
    "EXR": "REAL_ESTATE", "AVB": "REAL_ESTATE", "VICI": "REAL_ESTATE",

    # ---- Communication Services (COMM / XLC) ----
    "GOOGL": "COMM", "GOOG": "COMM", "META": "COMM", "NFLX": "COMM",
    "DIS": "COMM", "T": "COMM", "VZ": "COMM", "CMCSA": "COMM",
    "TMUS": "COMM", "CHTR": "COMM", "EA": "COMM", "TTWO": "COMM",
    "WBD": "COMM", "ROKU": "COMM",
}


def sector_for_ticker(symbol: Optional[str]) -> Optional[str]:
    """Return the sector code for ``symbol``, or ``None`` if unmapped.

    Normalises whitespace and case so a user pasting ``" aapl "`` in
    the sidebar maps cleanly. Returns ``None`` for the long tail of
    tickers we don't bother to cover — the UI then groups those into
    an "Unmapped" bucket.
    """
    if not symbol:
        return None
    return SECTOR_MEMBERSHIP.get(symbol.strip().upper())


def sector_universe_from_tickers(
    tickers: Iterable[str],
) -> dict[str, list[str]]:
    """Group ``tickers`` by sector code.

    Returns a dict shaped for :func:`predict_all_sectors`'s
    ``sector_universe`` argument: ``{sector_code: [ticker, …]}``.
    Unmapped tickers are silently dropped from the result — callers
    that want to surface them (e.g. an "Unmapped" UI section) should
    keep their own copy of the input list.

    Order is preserved within each sector so the UI can render
    tickers in the order the user typed them.
    """
    out: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for raw in tickers:
        sym = (raw or "").strip().upper()
        if not sym:
            continue
        code = SECTOR_MEMBERSHIP.get(sym)
        if code is None:
            continue
        if (code, sym) in seen:
            continue
        seen.add((code, sym))
        out.setdefault(code, []).append(sym)
    return out


def merge_sector_universes(
    primary: dict[str, Iterable[str]] | None,
    secondary: dict[str, Iterable[str]] | None,
) -> dict[str, list[str]]:
    """Union two sector-universe dicts; ``primary`` wins on overlap.

    Used by the daily-ingest job: cached ``RelatedEntity(ETF_HOLDING)``
    rows are the primary source (a paid Finnhub plan populates these
    exactly), and the curated map derived from the user's company
    tickers is the secondary fallback (free-tier deploys without
    ``/etf/holdings`` access).
    """
    merged: dict[str, list[str]] = {}
    for source in (primary or {}, secondary or {}):
        for code, tickers in source.items():
            current = merged.setdefault(code, [])
            for sym in tickers:
                if sym not in current:
                    current.append(sym)
    return merged
