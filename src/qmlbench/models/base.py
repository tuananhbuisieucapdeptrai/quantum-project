"""Shared conventions for all models.

Every model is a scikit-learn estimator constructed as ``factory(feature_types, task, **hp)``:

* ``feature_types`` lets the model build the right encoding (``AngleScaler`` for quantum
  models, ``PeriodicExpander`` for classical ones) inside its own pipeline, so the runner
  always passes *raw* features and preprocessing is fitted on training data only.
* ``task == "probability"``: the model must return probabilities in [0, 1] from
  ``predict`` (quantum models: map the expectation value <M> in [-1, 1] via (1 + <M>) / 2,
  or a sigmoid; classical: classifiers' ``predict_proba[:, 1]``).
* Hyperparameter search spaces live next to each factory (``SEARCH_SPACES``) so every
  model gets the same tuning budget from ``qmlbench.tuning``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

ModelFactory = Callable[..., BaseEstimator]


class ProbaAsPredict(RegressorMixin, BaseEstimator):
    """Wrap a classifier so ``predict`` returns P(y=1) -- used for the soccer xG task."""

    def __init__(self, clf: ClassifierMixin):
        self.clf = clf

    def fit(self, X, y):
        self.clf_ = clone(self.clf).fit(X, np.asarray(y).astype(int))
        return self

    def predict(self, X):
        return self.clf_.predict_proba(X)[:, 1]


def n_parameters(model: Any) -> int | None:
    """Best-effort trainable-parameter count (reported next to accuracy)."""
    fn = getattr(model, "n_parameters", None)
    return fn() if callable(fn) else None


Types = Sequence[str]
