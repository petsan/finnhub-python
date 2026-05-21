"""Tests for :mod:`finn_predictor.predictor.affinity` — PR-4 COMPETITOR layer.

Covers:

* `refresh_competitors` auto-seed from PEER rows whose target shares
  Finnhub's ``finnhubIndustry``.
* Per-peer failure isolation (one bad profile doesn't kill the sweep).
* Target without industry → no rows written + informational failure.
* `promote_peer_to_competitor` / `demote_competitor` manual curation,
  symbol-validation, idempotent re-runs.
* PEER rows are *not* mutated by competitor operations.
* `compose_company_focus` exposes the new ``competitors`` field.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from finn_predictor.ingestion.client import IngestionError
from finn_predictor.predictor.affinity import (
    CompetitorRefreshResult,
    demote_competitor,
    promote_peer_to_competitor,
    refresh_competitors,
)
from finn_predictor.predictor.focus import compose_company_focus
from finn_predictor.storage.repo import (
    related_entities_for,
    upsert_related_entity,
)


# ---------------------------------------------------------------------------
# refresh_competitors — auto-seed
# ---------------------------------------------------------------------------

def test_refresh_competitors_returns_empty_when_no_peers(session: Session) -> None:
    gw = MagicMock()
    gw.company_profile2.return_value = {"finnhubIndustry": "Technology"}
    result = refresh_competitors(session, gw, symbol="AAPL")
    assert isinstance(result, CompetitorRefreshResult)
    assert result.competitors_added == 0
    assert result.peers_considered == 0
    assert result.failures == []


def test_refresh_competitors_seeds_industry_matches(session: Session) -> None:
    """AAPL has peers MSFT, GOOG (Technology) and KO (Beverages).
    Only the Technology peers should become COMPETITOR rows."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER", rank=0)
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="GOOG", relationship="PEER", rank=1)
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="KO",  relationship="PEER", rank=2)

    gw = MagicMock()
    gw.company_profile2.side_effect = lambda sym: {
        "AAPL": {"finnhubIndustry": "Technology"},
        "MSFT": {"finnhubIndustry": "Technology"},
        "GOOG": {"finnhubIndustry": "Technology"},
        "KO":   {"finnhubIndustry": "Beverages"},
    }[sym]

    result = refresh_competitors(session, gw, symbol="AAPL")
    assert result.competitors_added == 2
    assert result.peers_considered == 3
    assert result.skipped_no_industry == 0
    assert result.failures == []

    comp_syms = {
        r.related_symbol
        for r in related_entities_for(session, "AAPL", relationship="COMPETITOR")
    }
    assert comp_syms == {"MSFT", "GOOG"}


def test_refresh_competitors_preserves_peer_rows(session: Session) -> None:
    """PEER rows must survive auto-seed — COMPETITOR is additive."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER", rank=0)
    gw = MagicMock()
    gw.company_profile2.return_value = {"finnhubIndustry": "Technology"}

    refresh_competitors(session, gw, symbol="AAPL")
    peers = related_entities_for(session, "AAPL", relationship="PEER")
    assert [p.related_symbol for p in peers] == ["MSFT"]


def test_refresh_competitors_idempotent_on_rerun(session: Session) -> None:
    """Second run with identical data adds zero new COMPETITOR rows."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER", rank=0)
    gw = MagicMock()
    gw.company_profile2.return_value = {"finnhubIndustry": "Technology"}

    r1 = refresh_competitors(session, gw, symbol="AAPL")
    r2 = refresh_competitors(session, gw, symbol="AAPL")
    assert r1.competitors_added == 1
    assert r2.competitors_added == 0
    # Still exactly one COMPETITOR row.
    rows = related_entities_for(session, "AAPL", relationship="COMPETITOR")
    assert len(rows) == 1


def test_refresh_competitors_target_profile_failure_returns_empty(
    session: Session,
) -> None:
    """If the target's own profile fetch fails, abort cleanly with
    a recorded failure — don't run any peer queries."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER", rank=0)
    gw = MagicMock()
    gw.company_profile2.side_effect = IngestionError("FinnhubAPI 403: gated")

    result = refresh_competitors(session, gw, symbol="AAPL")
    assert result.competitors_added == 0
    assert any("company_profile2:AAPL" in f["op"] for f in result.failures)
    # Peer profile lookups never happened — saves quota.
    assert gw.company_profile2.call_count == 1


def test_refresh_competitors_target_has_no_industry(session: Session) -> None:
    """Target with no finnhubIndustry → no rows + informational failure
    explaining why. (^GSPC, ETFs, delisted symbols all land here.)"""
    upsert_related_entity(session, source_symbol="^GSPC",
                          related_symbol="AAPL", relationship="PEER", rank=0)
    gw = MagicMock()
    gw.company_profile2.return_value = {}  # no finnhubIndustry

    result = refresh_competitors(session, gw, symbol="^GSPC")
    assert result.competitors_added == 0
    assert result.peers_considered == 0
    assert any("no finnhubIndustry" in f["error"] for f in result.failures)


def test_refresh_competitors_peer_profile_failure_isolated(session: Session) -> None:
    """One bad profile fetch shouldn't abort the whole sweep."""
    for sym in ["MSFT", "GOOG", "NVDA"]:
        upsert_related_entity(session, source_symbol="AAPL",
                              related_symbol=sym, relationship="PEER")

    def profile_lookup(sym):
        if sym == "AAPL":
            return {"finnhubIndustry": "Technology"}
        if sym == "GOOG":
            raise IngestionError("FinnhubAPI 502")
        return {"finnhubIndustry": "Technology"}

    gw = MagicMock()
    gw.company_profile2.side_effect = profile_lookup

    result = refresh_competitors(session, gw, symbol="AAPL")
    assert result.competitors_added == 2  # MSFT + NVDA
    assert any("company_profile2:GOOG" in f["op"] for f in result.failures)
    assert result.peers_considered == 3


def test_refresh_competitors_peer_without_industry_skipped(session: Session) -> None:
    """A peer profile that has no finnhubIndustry is counted as skipped,
    not failed."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MYSTERY", relationship="PEER")
    gw = MagicMock()
    gw.company_profile2.side_effect = lambda sym: {
        "AAPL": {"finnhubIndustry": "Technology"},
        "MYSTERY": {"name": "Mystery Co"},  # no industry
    }[sym]

    result = refresh_competitors(session, gw, symbol="AAPL")
    assert result.competitors_added == 0
    assert result.skipped_no_industry == 1
    assert result.failures == []


def test_refresh_competitors_non_dict_profile_payload(session: Session) -> None:
    """Defensive: gateway returns a non-dict profile (rare; treat as
    no-industry rather than crash)."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER")
    gw = MagicMock()
    gw.company_profile2.side_effect = lambda sym: (
        {"finnhubIndustry": "Technology"} if sym == "AAPL" else ["not", "a", "dict"]
    )
    result = refresh_competitors(session, gw, symbol="AAPL")
    assert result.competitors_added == 0
    assert result.skipped_no_industry == 1


def test_refresh_competitors_empty_industry_string_skipped(session: Session) -> None:
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER")
    gw = MagicMock()
    gw.company_profile2.side_effect = lambda sym: {
        "AAPL": {"finnhubIndustry": "Technology"},
        "MSFT": {"finnhubIndustry": "   "},  # whitespace-only
    }[sym]
    result = refresh_competitors(session, gw, symbol="AAPL")
    assert result.competitors_added == 0
    assert result.skipped_no_industry == 1


def test_refresh_competitors_normalises_symbol_case(session: Session) -> None:
    """Lower-case input should be uppercased before the DB read."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER")
    gw = MagicMock()
    gw.company_profile2.return_value = {"finnhubIndustry": "Technology"}
    result = refresh_competitors(session, gw, symbol="  aapl  ")
    assert result.symbol == "AAPL"
    assert result.competitors_added == 1


def test_refresh_competitors_empty_symbol_raises(session: Session) -> None:
    with pytest.raises(ValueError):
        refresh_competitors(session, MagicMock(), symbol="")


# ---------------------------------------------------------------------------
# promote_peer_to_competitor — manual curation
# ---------------------------------------------------------------------------

def test_promote_peer_to_competitor_basic(session: Session) -> None:
    row = promote_peer_to_competitor(
        session, symbol="aapl", peer_symbol="msft"
    )
    assert row.source_symbol == "AAPL"
    assert row.related_symbol == "MSFT"
    assert row.relationship == "COMPETITOR"
    assert row.metadata_text == "operator-curated"


def test_promote_peer_to_competitor_works_without_existing_peer(session: Session) -> None:
    """Operator can pin a COMPETITOR even when no PEER row exists.

    The Finnhub peer list may miss real competitors (especially across
    industries — e.g. Tesla vs traditional auto). The operator's call wins."""
    promote_peer_to_competitor(session, symbol="TSLA", peer_symbol="F")
    rows = related_entities_for(session, "TSLA", relationship="COMPETITOR")
    assert [r.related_symbol for r in rows] == ["F"]


def test_promote_idempotent_refreshes_existing_row(session: Session) -> None:
    a = promote_peer_to_competitor(session, symbol="AAPL", peer_symbol="MSFT")
    b = promote_peer_to_competitor(session, symbol="AAPL", peer_symbol="MSFT")
    assert a.id == b.id
    # Single row in DB.
    rows = related_entities_for(session, "AAPL", relationship="COMPETITOR")
    assert len(rows) == 1


@pytest.mark.parametrize(
    "src,peer",
    [
        ("", "MSFT"),
        ("AAPL", ""),
        ("AAPL", "INVALID;CHAR"),
        ("BAD/CHAR", "MSFT"),
        ("AAPL", "X" * 17),
    ],
)
def test_promote_validates_symbols(session: Session, src: str, peer: str) -> None:
    with pytest.raises(ValueError):
        promote_peer_to_competitor(session, symbol=src, peer_symbol=peer)


# ---------------------------------------------------------------------------
# demote_competitor
# ---------------------------------------------------------------------------

def test_demote_competitor_removes_row(session: Session) -> None:
    promote_peer_to_competitor(session, symbol="AAPL", peer_symbol="MSFT")
    assert demote_competitor(session, symbol="AAPL", peer_symbol="MSFT") is True
    assert related_entities_for(session, "AAPL", relationship="COMPETITOR") == []


def test_demote_competitor_returns_false_when_missing(session: Session) -> None:
    """Idempotent — double-click on Undo should not raise."""
    assert demote_competitor(session, symbol="AAPL", peer_symbol="MSFT") is False


def test_demote_competitor_leaves_peer_row_intact(session: Session) -> None:
    """Demotion only touches the COMPETITOR row; the underlying PEER
    row stays in the cache."""
    upsert_related_entity(session, source_symbol="AAPL",
                          related_symbol="MSFT", relationship="PEER", rank=0)
    promote_peer_to_competitor(session, symbol="AAPL", peer_symbol="MSFT")

    demote_competitor(session, symbol="AAPL", peer_symbol="MSFT")
    peers = related_entities_for(session, "AAPL", relationship="PEER")
    assert [p.related_symbol for p in peers] == ["MSFT"]


def test_demote_competitor_empty_args_raise(session: Session) -> None:
    with pytest.raises(ValueError):
        demote_competitor(session, symbol="", peer_symbol="MSFT")
    with pytest.raises(ValueError):
        demote_competitor(session, symbol="AAPL", peer_symbol="")


# ---------------------------------------------------------------------------
# compose_company_focus exposes the new competitors field
# ---------------------------------------------------------------------------

def test_compose_company_focus_returns_competitors(session: Session) -> None:
    """The Focus tab read path must surface COMPETITOR rows alongside
    PEER / SUPPLIER / CUSTOMER. Empty list when none cached."""
    promote_peer_to_competitor(session, symbol="AAPL", peer_symbol="MSFT")
    promote_peer_to_competitor(session, symbol="AAPL", peer_symbol="GOOG")

    bundle = compose_company_focus(session, "AAPL")
    # The new field exists and contains the rows we just wrote.
    comp_syms = sorted(c.related_symbol for c in bundle.competitors)
    assert comp_syms == ["GOOG", "MSFT"]
    # Peers + suppliers + customers default to empty.
    assert bundle.peers == []
    assert bundle.suppliers == []


def test_compose_company_focus_no_competitors_is_empty_list(session: Session) -> None:
    bundle = compose_company_focus(session, "AAPL")
    assert bundle.competitors == []


# ---------------------------------------------------------------------------
# RELATIONSHIPS allowed-set is extended
# ---------------------------------------------------------------------------

def test_relationships_allowed_set_includes_competitor() -> None:
    """The repo's allowed-set must accept COMPETITOR so future
    upserts don't trip the validation in upsert_related_entity."""
    from finn_predictor.storage.repo import RELATIONSHIPS
    assert "COMPETITOR" in RELATIONSHIPS
