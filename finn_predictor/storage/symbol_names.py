"""Ticker-symbol → human-readable name resolution.

Two layers, in order of priority:

  1. :data:`WELL_KNOWN_NAMES` — a hand-curated dict covering ``^GSPC``,
     the 11 sector SPDR ETFs we predict on, and the ~30 mega-caps a
     typical user is most likely to feed into the per-company news pull.
     This avoids spending API quota on names that almost never change.
  2. :class:`Sector` rows in the database — supplies sector-ETF names for
     anything not pre-baked above.

If you need a name that isn't in either layer, add it to
:data:`WELL_KNOWN_NAMES`. We deliberately do **not** look names up on
Finnhub at render time — the UI must remain offline (no API key required
just to render).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import Sector


# Curated mapping: ticker → display name. Lowercased lookups are NOT
# attempted automatically — tickers like ``^GSPC`` are case-sensitive in
# Finnhub's data, so we treat keys as exact.
WELL_KNOWN_NAMES: dict[str, str] = {
    # --- Indices -------------------------------------------------------
    "^GSPC": "S&P 500 Index",
    "^DJI": "Dow Jones Industrial Average",
    "^IXIC": "Nasdaq Composite",
    # --- Sector SPDR ETFs (mirrors storage.repo.DEFAULT_SECTORS) ------
    "XLK": "Information Technology (XLK)",
    "XLE": "Energy (XLE)",
    "XLF": "Financials (XLF)",
    "XLV": "Health Care (XLV)",
    "XLY": "Consumer Discretionary (XLY)",
    "XLP": "Consumer Staples (XLP)",
    "XLI": "Industrials (XLI)",
    "XLB": "Materials (XLB)",
    "XLU": "Utilities (XLU)",
    "XLRE": "Real Estate (XLRE)",
    "XLC": "Communication Services (XLC)",
    # --- Common ETFs --------------------------------------------------
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    # --- Mega-cap stocks ----------------------------------------------
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG": "Alphabet Inc. (Class C)",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "AVGO": "Broadcom Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "WMT": "Walmart Inc.",
    "XOM": "Exxon Mobil Corp.",
    "PG": "Procter & Gamble Co.",
    "HD": "The Home Depot Inc.",
    "CVX": "Chevron Corp.",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co. Inc.",
    "PEP": "PepsiCo Inc.",
    "KO": "The Coca-Cola Co.",
    "BAC": "Bank of America Corp.",
    "LLY": "Eli Lilly and Co.",
    "PFE": "Pfizer Inc.",
    "COST": "Costco Wholesale Corp.",
    "ABT": "Abbott Laboratories",
    "ADBE": "Adobe Inc.",
    "CSCO": "Cisco Systems Inc.",
    "ORCL": "Oracle Corp.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "MCD": "McDonald's Corp.",
    "ACN": "Accenture plc",
    "CRM": "Salesforce Inc.",
    "NFLX": "Netflix Inc.",
    "DIS": "The Walt Disney Co.",
    "INTC": "Intel Corp.",
    "AMD": "Advanced Micro Devices Inc.",
    "IBM": "International Business Machines Corp.",
    "BA": "The Boeing Co.",
    "GE": "General Electric Co.",
    "F": "Ford Motor Co.",
    "GM": "General Motors Co.",
}


def expand_symbol(session: Session, symbol: str) -> str:
    """Return a display string for ``symbol``.

    Resolution order:

    1. Exact match in :data:`WELL_KNOWN_NAMES`.
    2. A :class:`Sector` row whose ``etf_symbol`` matches → ``"<name> (<symbol>)"``.
    3. The raw symbol unchanged.

    The function never raises and is safe to call with ``None`` or empty
    strings (returns ``""`` in that case so the UI can omit the line).
    """
    if not symbol:
        return ""

    known = WELL_KNOWN_NAMES.get(symbol)
    if known is not None:
        return known

    sector = session.scalar(select(Sector).where(Sector.etf_symbol == symbol))
    if sector is not None:
        return f"{sector.name} ({sector.etf_symbol})"

    return symbol


def expand_symbol_short(session: Session, symbol: str) -> str:
    """Like :func:`expand_symbol` but with the bracketed ticker stripped.

    Useful when the ticker is already shown separately and we just want
    the company/sector name in flowing prose.
    """
    full = expand_symbol(session, symbol)
    if full.endswith(f" ({symbol})"):
        return full[: -(len(symbol) + 3)]
    return full
