"""Tests for the Watchlist + WatchlistMember storage layer (PR-2).

Covers every public helper in :mod:`finn_predictor.storage.repo` that
touches the new tables, plus the model-level relationships (cascade
delete, uniqueness, cross-list membership).

The ``session`` fixture is the in-memory SQLite one from
``tests/conftest.py`` — each test starts with an empty DB so ordering
doesn't matter.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.storage.models import Watchlist, WatchlistMember
from finn_predictor.storage.repo import (
    WatchlistError,
    add_to_watchlist,
    create_watchlist,
    delete_watchlist,
    get_watchlist,
    list_watchlists,
    remove_from_watchlist,
    rename_watchlist,
    replace_watchlist_symbols,
    update_watchlist_description,
    watchlist_members,
    watchlist_symbols,
)


def _to_utc(d: datetime) -> datetime:
    """Normalise any datetime to UTC-naive for ordered comparison.

    SQLite stores ``DateTime(timezone=True)`` as an ISO string and
    returns a naive datetime on read. The in-memory Python object set
    via ``_utcnow()`` is tz-aware. Comparing the two directly raises
    ``TypeError: can't compare offset-naive and offset-aware datetimes``.
    This helper coerces both sides into the same shape so the
    "updated_at moved forward" assertions can be expressed plainly.
    """
    if d.tzinfo is None:
        return d
    return d.astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Watchlist CRUD
# ---------------------------------------------------------------------------

def test_create_watchlist_minimal(session: Session) -> None:
    wl = create_watchlist(session, name="Tech bets")
    assert wl.id is not None
    assert wl.name == "Tech bets"
    assert wl.description is None
    # Both stamps come from the same ``_utcnow()`` call in
    # ``create_watchlist`` so they're identical at creation.
    assert _to_utc(wl.created_at) == _to_utc(wl.updated_at)


def test_create_watchlist_with_description(session: Session) -> None:
    wl = create_watchlist(
        session, name="Energy short list", description="  high-conv shorts  "
    )
    # Description is trimmed.
    assert wl.description == "high-conv shorts"


def test_create_watchlist_empty_description_is_stored_as_none(session: Session) -> None:
    wl = create_watchlist(session, name="X", description="   ")
    assert wl.description is None


def test_create_watchlist_strips_outer_whitespace(session: Session) -> None:
    wl = create_watchlist(session, name="  Tech  ")
    assert wl.name == "Tech"


def test_create_watchlist_rejects_duplicate(session: Session) -> None:
    create_watchlist(session, name="Tech")
    with pytest.raises(WatchlistError, match="already exists"):
        create_watchlist(session, name="Tech")


def test_create_watchlist_rejects_duplicate_case_insensitive_input(session: Session) -> None:
    """Names are case-sensitive at the DB layer; we don't auto-collapse case.

    Two distinct names (``Tech`` and ``tech``) coexist. We document that
    deliberately so the operator can disambiguate watchlists by capitalisation.
    """
    create_watchlist(session, name="Tech")
    # 'tech' is a different name; creation should succeed.
    create_watchlist(session, name="tech")
    assert {w.name for w in list_watchlists(session)} == {"Tech", "tech"}


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "tech/bad",       # slash
        "tech;DROP",      # semicolon
        "tech\nbad",      # newline
        "list<script>",   # angle bracket
        "x" * 65,         # over 64 chars
    ],
)
def test_create_watchlist_rejects_bad_names(session: Session, bad: str) -> None:
    with pytest.raises(WatchlistError):
        create_watchlist(session, name=bad)


def test_create_watchlist_rejects_non_string(session: Session) -> None:
    with pytest.raises(WatchlistError, match="must be a string"):
        create_watchlist(session, name=42)  # type: ignore[arg-type]


def test_get_watchlist_returns_none_for_missing(session: Session) -> None:
    assert get_watchlist(session, "Nope") is None


def test_get_watchlist_returns_existing(session: Session) -> None:
    created = create_watchlist(session, name="Tech")
    fetched = get_watchlist(session, "  Tech  ")  # extra whitespace tolerated
    assert fetched is not None
    assert fetched.id == created.id


def test_get_watchlist_validates_name_shape(session: Session) -> None:
    # We don't silently swallow malformed inputs; the caller should
    # know they typed something invalid.
    with pytest.raises(WatchlistError):
        get_watchlist(session, "with/slash")


def test_list_watchlists_orders_alphabetically(session: Session) -> None:
    create_watchlist(session, name="Zeta")
    create_watchlist(session, name="Alpha")
    create_watchlist(session, name="Mu")
    names = [w.name for w in list_watchlists(session)]
    assert names == ["Alpha", "Mu", "Zeta"]


def test_list_watchlists_empty_returns_empty_list(session: Session) -> None:
    assert list_watchlists(session) == []


# ---------------------------------------------------------------------------
# Rename + update_description
# ---------------------------------------------------------------------------

def test_rename_watchlist(session: Session) -> None:
    created = create_watchlist(session, name="Tech")
    initial_updated_at = created.updated_at
    renamed = rename_watchlist(session, old_name="Tech", new_name="AI bets")
    assert renamed.id == created.id
    assert renamed.name == "AI bets"
    assert renamed.updated_at > initial_updated_at


def test_rename_watchlist_to_same_name_is_noop(session: Session) -> None:
    created = create_watchlist(session, name="Tech")
    initial = created.updated_at
    same = rename_watchlist(session, old_name="Tech", new_name="  Tech  ")
    assert same.id == created.id
    # updated_at didn't move — true no-op.
    assert same.updated_at == initial


def test_rename_watchlist_missing_raises(session: Session) -> None:
    with pytest.raises(WatchlistError, match="not found"):
        rename_watchlist(session, old_name="Missing", new_name="Other")


def test_rename_watchlist_collision_raises(session: Session) -> None:
    create_watchlist(session, name="Tech")
    create_watchlist(session, name="Energy")
    with pytest.raises(WatchlistError, match="already exists"):
        rename_watchlist(session, old_name="Tech", new_name="Energy")


def test_update_description_sets_and_clears(session: Session) -> None:
    create_watchlist(session, name="Tech", description="initial")
    updated = update_watchlist_description(session, name="Tech", description="new")
    assert updated.description == "new"
    cleared = update_watchlist_description(session, name="Tech", description=None)
    assert cleared.description is None
    cleared_via_blank = update_watchlist_description(
        session, name="Tech", description="   "
    )
    assert cleared_via_blank.description is None


def test_update_description_missing_raises(session: Session) -> None:
    with pytest.raises(WatchlistError, match="not found"):
        update_watchlist_description(session, name="Missing", description="x")


# ---------------------------------------------------------------------------
# Delete + cascade
# ---------------------------------------------------------------------------

def test_delete_watchlist_returns_true_on_existing(session: Session) -> None:
    create_watchlist(session, name="Tech")
    assert delete_watchlist(session, name="Tech") is True
    assert get_watchlist(session, "Tech") is None


def test_delete_watchlist_returns_false_on_missing(session: Session) -> None:
    # Idempotent double-click semantics — never raises on "already gone".
    assert delete_watchlist(session, name="Missing") is False


def test_delete_watchlist_cascades_to_members(session: Session) -> None:
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    add_to_watchlist(session, name="Tech", symbol="MSFT")
    assert len(watchlist_members(session, name="Tech")) == 2

    delete_watchlist(session, name="Tech")
    # The members must be gone too — cascade delete.
    remaining = session.scalars(select(WatchlistMember)).all()
    assert remaining == []


# ---------------------------------------------------------------------------
# WatchlistMember CRUD
# ---------------------------------------------------------------------------

def test_add_to_watchlist_basic(session: Session) -> None:
    create_watchlist(session, name="Tech")
    member = add_to_watchlist(session, name="Tech", symbol="aapl")
    assert member.symbol == "AAPL"     # uppercased
    assert member.notes is None
    assert member.watchlist_id is not None


def test_add_to_watchlist_stores_notes(session: Session) -> None:
    create_watchlist(session, name="Tech")
    member = add_to_watchlist(
        session, name="Tech", symbol="AAPL", notes="  core position  "
    )
    assert member.notes == "core position"  # trimmed


def test_add_to_watchlist_idempotent_refreshes_notes(session: Session) -> None:
    create_watchlist(session, name="Tech")
    first = add_to_watchlist(session, name="Tech", symbol="AAPL", notes="first")
    again = add_to_watchlist(session, name="Tech", symbol="AAPL", notes="second")
    # Same row, refreshed notes.
    assert first.id == again.id
    assert again.notes == "second"
    # And only one row exists.
    members = watchlist_members(session, name="Tech")
    assert len(members) == 1


def test_add_to_watchlist_missing_list_raises(session: Session) -> None:
    with pytest.raises(WatchlistError, match="not found"):
        add_to_watchlist(session, name="Missing", symbol="AAPL")


@pytest.mark.parametrize(
    "bad_symbol",
    ["", "  ", "AAPL;DROP", "aapl/bad", "a" * 17, None],
)
def test_add_to_watchlist_rejects_bad_symbol(session: Session, bad_symbol) -> None:
    create_watchlist(session, name="Tech")
    with pytest.raises(WatchlistError):
        add_to_watchlist(session, name="Tech", symbol=bad_symbol)


def test_add_to_watchlist_bumps_parent_updated_at(session: Session) -> None:
    parent = create_watchlist(session, name="Tech")
    before = _to_utc(parent.updated_at)
    # Microsecond-level resolution is enough on Linux/CPython, but be
    # generous with a tiny sleep so the assertion isn't load-sensitive.
    time.sleep(0.002)
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    session.refresh(parent)
    assert _to_utc(parent.updated_at) > before


def test_same_symbol_in_multiple_watchlists(session: Session) -> None:
    """AAPL can belong to both 'Tech' and 'Mega-cap' simultaneously."""
    create_watchlist(session, name="Tech")
    create_watchlist(session, name="Mega-cap")
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    add_to_watchlist(session, name="Mega-cap", symbol="AAPL")
    assert watchlist_symbols(session, name="Tech") == ["AAPL"]
    assert watchlist_symbols(session, name="Mega-cap") == ["AAPL"]
    # Union returns it once.
    assert watchlist_symbols(session) == ["AAPL"]


def test_remove_from_watchlist(session: Session) -> None:
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    add_to_watchlist(session, name="Tech", symbol="MSFT")

    assert remove_from_watchlist(session, name="Tech", symbol="aapl") is True
    assert watchlist_symbols(session, name="Tech") == ["MSFT"]


def test_remove_from_watchlist_missing_member_returns_false(session: Session) -> None:
    create_watchlist(session, name="Tech")
    assert remove_from_watchlist(session, name="Tech", symbol="AAPL") is False


def test_remove_from_watchlist_missing_list_raises(session: Session) -> None:
    with pytest.raises(WatchlistError, match="not found"):
        remove_from_watchlist(session, name="Missing", symbol="AAPL")


def test_watchlist_members_orders_by_symbol(session: Session) -> None:
    create_watchlist(session, name="Tech")
    for sym in ["MSFT", "AAPL", "NVDA", "GOOG"]:
        add_to_watchlist(session, name="Tech", symbol=sym)
    members = watchlist_members(session, name="Tech")
    assert [m.symbol for m in members] == ["AAPL", "GOOG", "MSFT", "NVDA"]


def test_watchlist_members_missing_list_raises(session: Session) -> None:
    with pytest.raises(WatchlistError, match="not found"):
        watchlist_members(session, name="Missing")


# ---------------------------------------------------------------------------
# watchlist_symbols (union + named)
# ---------------------------------------------------------------------------

def test_watchlist_symbols_union_dedupes(session: Session) -> None:
    create_watchlist(session, name="Tech")
    create_watchlist(session, name="Mega")
    for sym in ["AAPL", "MSFT"]:
        add_to_watchlist(session, name="Tech", symbol=sym)
    for sym in ["AAPL", "GOOG"]:
        add_to_watchlist(session, name="Mega", symbol=sym)
    assert watchlist_symbols(session) == ["AAPL", "GOOG", "MSFT"]


def test_watchlist_symbols_empty_universe(session: Session) -> None:
    # No watchlists at all → empty list, never None.
    assert watchlist_symbols(session) == []


def test_watchlist_symbols_named_missing_raises(session: Session) -> None:
    with pytest.raises(WatchlistError, match="not found"):
        watchlist_symbols(session, name="Missing")


def test_watchlist_symbols_named_empty(session: Session) -> None:
    create_watchlist(session, name="Tech")
    assert watchlist_symbols(session, name="Tech") == []


# ---------------------------------------------------------------------------
# replace_watchlist_symbols (the sidebar "paste + save" flow)
# ---------------------------------------------------------------------------

def test_replace_watchlist_symbols_initial_population(session: Session) -> None:
    create_watchlist(session, name="Tech")
    count = replace_watchlist_symbols(
        session, name="Tech", symbols=["aapl", "msft", "NVDA"]
    )
    assert count == 3
    assert watchlist_symbols(session, name="Tech") == ["AAPL", "MSFT", "NVDA"]


def test_replace_watchlist_symbols_replaces_existing_set(session: Session) -> None:
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="AAPL", notes="legacy")
    add_to_watchlist(session, name="Tech", symbol="MSFT")
    add_to_watchlist(session, name="Tech", symbol="OLD")

    count = replace_watchlist_symbols(
        session, name="Tech", symbols=["AAPL", "GOOG"]
    )
    assert count == 2
    members = watchlist_members(session, name="Tech")
    assert {m.symbol for m in members} == {"AAPL", "GOOG"}
    # The legacy notes are lost — replacement is total, not partial.
    aapl = next(m for m in members if m.symbol == "AAPL")
    assert aapl.notes is None


def test_replace_watchlist_symbols_dedupes_within_input(session: Session) -> None:
    create_watchlist(session, name="Tech")
    count = replace_watchlist_symbols(
        session, name="Tech", symbols=["AAPL", "aapl", "MSFT", "AAPL"]
    )
    assert count == 2
    assert watchlist_symbols(session, name="Tech") == ["AAPL", "MSFT"]


def test_replace_watchlist_symbols_with_empty_iterable_clears(session: Session) -> None:
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    assert watchlist_symbols(session, name="Tech") == ["AAPL"]
    count = replace_watchlist_symbols(session, name="Tech", symbols=[])
    assert count == 0
    assert watchlist_symbols(session, name="Tech") == []


def test_replace_watchlist_symbols_rejects_bad_symbol_atomically(
    session: Session,
) -> None:
    """A bad symbol mid-stream raises *before* we write anything.

    Otherwise the watchlist would land in a partially-replaced state
    with no surfaced error. We validate every symbol up front."""
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="EXISTING")
    with pytest.raises(WatchlistError):
        replace_watchlist_symbols(
            session,
            name="Tech",
            symbols=["AAPL", "BAD;SYMBOL", "MSFT"],
        )
    # Pre-existing state untouched.
    assert watchlist_symbols(session, name="Tech") == ["EXISTING"]


def test_replace_watchlist_symbols_missing_list_raises(session: Session) -> None:
    with pytest.raises(WatchlistError, match="not found"):
        replace_watchlist_symbols(session, name="Missing", symbols=["AAPL"])


def test_replace_watchlist_symbols_bumps_parent_updated_at(session: Session) -> None:
    parent = create_watchlist(session, name="Tech")
    before = _to_utc(parent.updated_at)
    time.sleep(0.002)
    replace_watchlist_symbols(session, name="Tech", symbols=["AAPL"])
    session.refresh(parent)
    assert _to_utc(parent.updated_at) > before


# ---------------------------------------------------------------------------
# Foreign-key + relationship sanity
# ---------------------------------------------------------------------------

def test_orm_relationship_back_populates(session: Session) -> None:
    """``Watchlist.members`` should round-trip via the ORM relationship."""
    create_watchlist(session, name="Tech")
    add_to_watchlist(session, name="Tech", symbol="AAPL")
    add_to_watchlist(session, name="Tech", symbol="MSFT")

    parent = get_watchlist(session, "Tech")
    assert parent is not None
    syms = sorted(m.symbol for m in parent.members)
    assert syms == ["AAPL", "MSFT"]
    # Each member points back at the same Watchlist.
    for m in parent.members:
        assert m.watchlist.id == parent.id


def test_dunder_repr_does_not_raise(session: Session) -> None:
    """Defensive: ``repr(Watchlist)`` is debug-only but must not blow up."""
    wl = create_watchlist(session, name="Tech")
    # The model marks __repr__ pragma: no cover — call it directly so it
    # at least gets exercised and we'd notice a future regression.
    assert "Tech" in repr(wl)
