"""Read + apply learned weights at predictor call time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finn_predictor.predictor.market import (
    MIN_ARTICLES_FOR_CALL,
    MIN_BASELINE_SIGMA,
    THRESHOLD_SIGMA,
)
from finn_predictor.storage.models import LearnedWeight


# Dimension names used in the LearnedWeight table.
DIM_THRESHOLD = "THRESHOLD_SIGMA"
DIM_MIN_SIGMA = "MIN_BASELINE_SIGMA"
DIM_HALF_LIFE = "HALF_LIFE_HOURS"
DIM_SOURCE_WEIGHT = "SOURCE_WEIGHT"
# PR-8: per-relationship multiplier used by the affinity-blended
# predictor (:mod:`predictor.blended`). ``key`` is the relationship
# string (PEER / COMPETITOR / SUPPLIER / CUSTOMER / THEME_MEMBER /
# INSTITUTIONAL_HOLDER); ``value`` is the float weight. We carry these
# in the LearnedWeight table even when ``train_weights`` is not yet
# fitting them via gp_minimize so the curated defaults can be the v1
# baseline; future training iterations can fit against this baseline.
DIM_AFFINITY_WEIGHT = "AFFINITY_WEIGHT"

DIMENSIONS = frozenset({
    DIM_THRESHOLD, DIM_MIN_SIGMA, DIM_HALF_LIFE,
    DIM_SOURCE_WEIGHT, DIM_AFFINITY_WEIGHT,
})

# Hand-coded defaults — used when no learned set is active.
DEFAULTS = {
    DIM_THRESHOLD: THRESHOLD_SIGMA,
    DIM_MIN_SIGMA: MIN_BASELINE_SIGMA,
    DIM_HALF_LIFE: 12.0,
}


@dataclass(frozen=True)
class LearnedConfig:
    """Typed snapshot of the active learned weights.

    Source weights default to 1.0 for any source not in the map.
    Affinity weights default to the curated set in
    :data:`finn_predictor.predictor.blended.DEFAULT_AFFINITY_WEIGHTS`
    (resolved lazily by :meth:`to_affinity_weights`) so a fresh deploy
    behaves identically to the pre-PR-8 affinity-blend defaults.
    """

    version: Optional[int]
    threshold_sigma: float
    min_baseline_sigma: float
    half_life_hours: float
    source_weights: dict[str, float] = field(default_factory=dict)
    # PR-8: per-relationship affinity weights. Empty dict means "use
    # the curated defaults" — see :meth:`to_affinity_weights`. When
    # ``train_weights`` populates this, the keys are the relationship
    # strings used by RelatedEntity (PEER / COMPETITOR / etc.).
    affinity_weights: dict[str, float] = field(default_factory=dict)

    def weight_for_source(self, source: Optional[str]) -> float:
        """Return the multiplier for an article from ``source`` (default 1.0)."""
        if not source:
            return 1.0
        return self.source_weights.get(source, 1.0)

    def to_affinity_weights(self):
        """Materialise an :class:`AffinityWeights` for the blended predictor.

        Lazy-imports :mod:`finn_predictor.predictor.blended` so this
        module remains importable in environments without the predictor
        installed (CLI hash-password, smoke tests). The ``affinity_weights``
        dict overrides the corresponding field on the dataclass; missing
        keys fall back to the curated default.
        """
        from finn_predictor.predictor.blended import AffinityWeights

        defaults = AffinityWeights()
        if not self.affinity_weights:
            return defaults
        # Map RelatedEntity.relationship strings → dataclass field names.
        return AffinityWeights(
            self_weight=self.affinity_weights.get("SELF", defaults.self_weight),
            peer=self.affinity_weights.get("PEER", defaults.peer),
            competitor=self.affinity_weights.get("COMPETITOR", defaults.competitor),
            supplier=self.affinity_weights.get("SUPPLIER", defaults.supplier),
            customer=self.affinity_weights.get("CUSTOMER", defaults.customer),
            theme_member=self.affinity_weights.get(
                "THEME_MEMBER", defaults.theme_member
            ),
            institutional_holder=self.affinity_weights.get(
                "INSTITUTIONAL_HOLDER", defaults.institutional_holder
            ),
        )


def apply_to_default() -> LearnedConfig:
    """Config that reproduces the existing hand-tuned predictor."""
    return LearnedConfig(
        version=None,
        threshold_sigma=DEFAULTS[DIM_THRESHOLD],
        min_baseline_sigma=DEFAULTS[DIM_MIN_SIGMA],
        half_life_hours=DEFAULTS[DIM_HALF_LIFE],
        source_weights={},
    )


def _rows_for_version(
    session: Session, version: int
) -> list[LearnedWeight]:
    return list(
        session.scalars(
            select(LearnedWeight).where(LearnedWeight.version == version)
        )
    )


def _config_from_rows(
    version: Optional[int], rows: list[LearnedWeight]
) -> LearnedConfig:
    threshold = DEFAULTS[DIM_THRESHOLD]
    min_sigma = DEFAULTS[DIM_MIN_SIGMA]
    half_life = DEFAULTS[DIM_HALF_LIFE]
    sources: dict[str, float] = {}
    affinity: dict[str, float] = {}
    for r in rows:
        if r.dimension == DIM_THRESHOLD:
            threshold = float(r.value)
        elif r.dimension == DIM_MIN_SIGMA:
            min_sigma = float(r.value)
        elif r.dimension == DIM_HALF_LIFE:
            half_life = float(r.value)
        elif r.dimension == DIM_SOURCE_WEIGHT and r.key:
            sources[r.key] = float(r.value)
        elif r.dimension == DIM_AFFINITY_WEIGHT and r.key:
            affinity[r.key] = float(r.value)
    return LearnedConfig(
        version=version,
        threshold_sigma=threshold,
        min_baseline_sigma=min_sigma,
        half_life_hours=half_life,
        source_weights=sources,
        affinity_weights=affinity,
    )


def weights_for_version(
    session: Session, version: int
) -> LearnedConfig:
    """Load a specific learned-weight version. Missing dims fall back to defaults."""
    rows = _rows_for_version(session, version)
    return _config_from_rows(version, rows)


def active_weights(session: Session) -> LearnedConfig:
    """The single :class:`LearnedConfig` flagged ``is_active=True``.

    Returns the hand-tuned defaults when no version has been activated yet.
    """
    active_row = session.scalar(
        select(LearnedWeight).where(LearnedWeight.is_active.is_(True)).limit(1)
    )
    if active_row is None:
        return apply_to_default()
    return weights_for_version(session, active_row.version)
