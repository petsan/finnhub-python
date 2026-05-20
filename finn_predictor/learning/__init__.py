"""Self-improvement: learn weights from hypothetical-trade outcomes.

Public surface:

* :func:`train_weights` — runs a Bayesian-optimisation search over the
  learnable dimensions on the historical trade ledger, persists the
  resulting weights as a new versioned row set, and auto-activates the
  newest version.
* :func:`active_weights` / :func:`weights_for_version` — read helpers.
* :class:`LearnedConfig` — the typed bundle the predictor reads at
  call time.
"""

from finn_predictor.learning.config import (
    LearnedConfig,
    active_weights,
    apply_to_default,
    weights_for_version,
)
from finn_predictor.learning.train import (
    TrainingReport,
    train_weights,
)

__all__ = [
    "LearnedConfig",
    "TrainingReport",
    "active_weights",
    "apply_to_default",
    "train_weights",
    "weights_for_version",
]
