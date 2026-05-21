"""Tests for PR-8: AFFINITY_WEIGHT learning dimension.

Scope covered by this PR:

* :data:`DIM_AFFINITY_WEIGHT` is registered in
  :data:`finn_predictor.learning.config.DIMENSIONS`.
* :class:`LearnedConfig` carries an ``affinity_weights`` dict and
  exposes a :meth:`to_affinity_weights` conversion.
* :func:`_config_from_rows` loads :data:`DIM_AFFINITY_WEIGHT` rows.
* :func:`train_weights._persist_weights` writes one row per
  relationship by default (curated baseline) — this is the
  versioning hook future training iterations can fit against.
* :func:`weights_for_version` round-trips affinity weights.

Out of scope for this PR (filed as the documented follow-up): actually
fitting affinity weights via ``gp_minimize`` — that requires
:mod:`learning.simulate` to compute a blended objective, which is a
substantial refactor of the training loop. The persistence + read
machinery shipped here is the foundation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from finn_predictor.learning.config import (
    DIM_AFFINITY_WEIGHT,
    DIM_HALF_LIFE,
    DIM_MIN_SIGMA,
    DIM_SOURCE_WEIGHT,
    DIM_THRESHOLD,
    DIMENSIONS,
    LearnedConfig,
    active_weights,
    apply_to_default,
    weights_for_version,
)
from finn_predictor.learning.train import (
    _curated_affinity_baseline,
    _persist_weights,
)
from finn_predictor.predictor.blended import AffinityWeights
from finn_predictor.storage.models import LearnedWeight


# ---------------------------------------------------------------------------
# DIMENSIONS allowlist
# ---------------------------------------------------------------------------

def test_dimensions_includes_affinity_weight() -> None:
    assert DIM_AFFINITY_WEIGHT in DIMENSIONS
    # Sanity: every legacy dim is still there.
    for d in (DIM_THRESHOLD, DIM_MIN_SIGMA, DIM_HALF_LIFE, DIM_SOURCE_WEIGHT):
        assert d in DIMENSIONS


def test_dim_affinity_weight_value_string() -> None:
    """Stable serialised name so future migrations can grep for it."""
    assert DIM_AFFINITY_WEIGHT == "AFFINITY_WEIGHT"


# ---------------------------------------------------------------------------
# LearnedConfig.affinity_weights field
# ---------------------------------------------------------------------------

def test_learned_config_defaults_empty_affinity_weights() -> None:
    cfg = apply_to_default()
    assert cfg.affinity_weights == {}


def test_learned_config_carries_affinity_weights() -> None:
    cfg = LearnedConfig(
        version=1,
        threshold_sigma=0.5,
        min_baseline_sigma=0.05,
        half_life_hours=12.0,
        affinity_weights={"COMPETITOR": -0.4, "SUPPLIER": 0.2},
    )
    assert cfg.affinity_weights["COMPETITOR"] == -0.4
    assert cfg.affinity_weights["SUPPLIER"] == 0.2


def test_to_affinity_weights_empty_returns_defaults() -> None:
    """Empty dict on LearnedConfig → AffinityWeights() defaults."""
    cfg = apply_to_default()
    weights = cfg.to_affinity_weights()
    expected = AffinityWeights()
    assert weights == expected


def test_to_affinity_weights_partial_overrides_default() -> None:
    """A LearnedConfig with only COMPETITOR set keeps the curated
    defaults for every other relationship — non-overridden fields
    don't get zeroed out."""
    cfg = LearnedConfig(
        version=2,
        threshold_sigma=0.5, min_baseline_sigma=0.05, half_life_hours=12.0,
        affinity_weights={"COMPETITOR": -0.5},
    )
    weights = cfg.to_affinity_weights()
    assert weights.competitor == -0.5
    # Other fields stayed at the default values.
    defaults = AffinityWeights()
    assert weights.peer == defaults.peer
    assert weights.supplier == defaults.supplier
    assert weights.customer == defaults.customer
    assert weights.theme_member == defaults.theme_member
    assert weights.self_weight == defaults.self_weight


def test_to_affinity_weights_full_override() -> None:
    cfg = LearnedConfig(
        version=3,
        threshold_sigma=0.5, min_baseline_sigma=0.05, half_life_hours=12.0,
        affinity_weights={
            "SELF": 0.9,
            "PEER": 0.2,
            "COMPETITOR": -0.5,
            "SUPPLIER": 0.3,
            "CUSTOMER": 0.4,
            "THEME_MEMBER": 0.15,
            "INSTITUTIONAL_HOLDER": 0.01,
        },
    )
    weights = cfg.to_affinity_weights()
    assert weights.self_weight == 0.9
    assert weights.peer == 0.2
    assert weights.competitor == -0.5
    assert weights.supplier == 0.3
    assert weights.customer == 0.4
    assert weights.theme_member == 0.15
    assert weights.institutional_holder == 0.01


# ---------------------------------------------------------------------------
# Round-trip via the LearnedWeight table
# ---------------------------------------------------------------------------

def test_persist_weights_writes_affinity_baseline(session: Session) -> None:
    """``_persist_weights`` writes the curated affinity baseline by default."""
    _persist_weights(
        session,
        version=1,
        threshold=0.6,
        min_sigma=0.04,
        half_life=18.0,
        source_weights={"Reuters": 1.2},
        training_score=0.55,
        holdout_score=0.52,
    )

    rows = list(
        session.query(LearnedWeight)
        .filter(
            LearnedWeight.version == 1,
            LearnedWeight.dimension == DIM_AFFINITY_WEIGHT,
        )
    )
    # One row per relationship in the curated baseline.
    relationships = {r.key for r in rows}
    assert "SELF" in relationships
    assert "COMPETITOR" in relationships
    assert "SUPPLIER" in relationships
    assert "CUSTOMER" in relationships
    assert "PEER" in relationships
    assert "THEME_MEMBER" in relationships
    assert "INSTITUTIONAL_HOLDER" in relationships

    # Values match the curated AffinityWeights() defaults.
    by_key = {r.key: r.value for r in rows}
    expected = _curated_affinity_baseline()
    for rel, expected_val in expected.items():
        assert by_key[rel] == pytest.approx(expected_val)


def test_persist_weights_explicit_affinity_overrides_baseline(
    session: Session,
) -> None:
    """When the caller supplies ``affinity_weights``, those values
    are persisted instead of the curated baseline (PR-8 follow-up hook
    for when ``train_weights`` fits via gp_minimize)."""
    custom = {"COMPETITOR": -0.99, "SUPPLIER": 0.99}
    _persist_weights(
        session,
        version=5,
        threshold=0.5, min_sigma=0.05, half_life=12.0,
        source_weights={},
        training_score=0.5, holdout_score=None,
        affinity_weights=custom,
    )
    rows = list(
        session.query(LearnedWeight)
        .filter(
            LearnedWeight.version == 5,
            LearnedWeight.dimension == DIM_AFFINITY_WEIGHT,
        )
    )
    by_key = {r.key: r.value for r in rows}
    # Only the two keys we passed are persisted — no curated leakage.
    assert set(by_key) == {"COMPETITOR", "SUPPLIER"}
    assert by_key["COMPETITOR"] == pytest.approx(-0.99)
    assert by_key["SUPPLIER"] == pytest.approx(0.99)


def test_weights_for_version_round_trips_affinity(session: Session) -> None:
    """LearnedConfig.affinity_weights survives a write + read cycle."""
    _persist_weights(
        session,
        version=7,
        threshold=0.7, min_sigma=0.04, half_life=18.0,
        source_weights={"Reuters": 1.3},
        training_score=0.6, holdout_score=0.58,
        affinity_weights={"COMPETITOR": -0.4, "CUSTOMER": 0.3},
    )
    cfg = weights_for_version(session, 7)
    assert cfg.version == 7
    assert cfg.affinity_weights["COMPETITOR"] == pytest.approx(-0.4)
    assert cfg.affinity_weights["CUSTOMER"] == pytest.approx(0.3)
    # Source weights still round-trip alongside affinity weights.
    assert cfg.source_weights["Reuters"] == pytest.approx(1.3)


def test_active_weights_reads_affinity(session: Session) -> None:
    """``active_weights`` (used by the predictor) returns the affinity
    weights of whichever version is active."""
    _persist_weights(
        session,
        version=9,
        threshold=0.5, min_sigma=0.05, half_life=12.0,
        source_weights={},
        training_score=0.5, holdout_score=None,
        affinity_weights={"COMPETITOR": -0.42},
    )
    # Mark version 9 active.
    rows = list(
        session.query(LearnedWeight).filter(LearnedWeight.version == 9)
    )
    for r in rows:
        r.is_active = True
    session.commit()

    cfg = active_weights(session)
    assert cfg.version == 9
    assert cfg.affinity_weights["COMPETITOR"] == pytest.approx(-0.42)
    # And ``to_affinity_weights`` materialises the override.
    aw = cfg.to_affinity_weights()
    assert aw.competitor == pytest.approx(-0.42)
    # Untouched fields keep the curated defaults.
    assert aw.supplier == AffinityWeights().supplier


def test_active_weights_no_active_version_returns_defaults(
    session: Session,
) -> None:
    """No persisted version → apply_to_default → empty affinity dict
    → curated AffinityWeights via to_affinity_weights()."""
    cfg = active_weights(session)
    assert cfg.affinity_weights == {}
    assert cfg.to_affinity_weights() == AffinityWeights()


# ---------------------------------------------------------------------------
# Curated baseline helper
# ---------------------------------------------------------------------------

def test_curated_affinity_baseline_matches_default_dataclass() -> None:
    """The persisted baseline must exactly match the AffinityWeights
    defaults — otherwise the active predictor would silently drift
    from the curated values on first train_weights run."""
    baseline = _curated_affinity_baseline()
    defaults = AffinityWeights()
    assert baseline["SELF"] == defaults.self_weight
    assert baseline["PEER"] == defaults.peer
    assert baseline["COMPETITOR"] == defaults.competitor
    assert baseline["SUPPLIER"] == defaults.supplier
    assert baseline["CUSTOMER"] == defaults.customer
    assert baseline["THEME_MEMBER"] == defaults.theme_member
    assert baseline["INSTITUTIONAL_HOLDER"] == defaults.institutional_holder
