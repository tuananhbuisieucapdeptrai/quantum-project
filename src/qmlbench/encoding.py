"""Feature encodings that respect each feature's type.

Two transformers, both fitted on training data only:

``AngleScaler`` -- for quantum models (inputs become rotation angles).
    angle  : kept as is (already radians in [-pi, pi)). A rotation R(phi) is 2pi-periodic,
             so the circuit sees the true wrap-around of the input for free.
    real   : robustly min-max scaled to an interval of width pi (default [-pi/2, pi/2]).
             Width pi rather than 2pi on purpose: mapping onto the full circle would make the
             smallest and largest values *the same rotation* and invent a fake periodicity.
             Values beyond the fitted quantiles are clipped (handles outliers such as
             glitched SDSS photometry).
    binary : {0, 1} -> the two ends of the same interval.

``PeriodicExpander`` -- for classical models.
    Replaces every angle column by (sin, cos) and standardises real columns, so classical
    models get the same knowledge of periodicity that the quantum encoding has built in.
    Without this the comparison would be unfair to classical models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AngleScaler(TransformerMixin, BaseEstimator):
    def __init__(self, feature_types: Sequence[str], real_range: tuple[float, float] = (
            -np.pi / 2, np.pi / 2), quantiles: tuple[float, float] = (0.005, 0.995)):
        self.feature_types = feature_types
        self.real_range = real_range
        self.quantiles = quantiles

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.quantile(X, self.quantiles[0], axis=0)
        self.hi_ = np.quantile(X, self.quantiles[1], axis=0)
        same = self.hi_ <= self.lo_
        self.hi_ = np.where(same, self.lo_ + 1.0, self.hi_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        a, b = self.real_range
        for j, t in enumerate(self.feature_types):
            if t == "angle":
                X[:, j] = (X[:, j] + np.pi) % (2 * np.pi) - np.pi
            elif t == "binary":
                X[:, j] = np.where(X[:, j] > 0.5, b, a)
            else:
                u = np.clip((X[:, j] - self.lo_[j]) / (self.hi_[j] - self.lo_[j]), 0.0, 1.0)
                X[:, j] = a + u * (b - a)
        return X


class PeriodicExpander(TransformerMixin, BaseEstimator):
    def __init__(self, feature_types: Sequence[str], standardize: bool = True):
        self.feature_types = feature_types
        self.standardize = standardize

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cols = []
        for j, t in enumerate(self.feature_types):
            if t == "angle":
                cols += [np.sin(X[:, j]), np.cos(X[:, j])]
            elif t == "real" and self.standardize:
                cols.append((X[:, j] - self.mean_[j]) / self.std_[j])
            else:
                cols.append(X[:, j])
        return np.column_stack(cols)

    def get_feature_names_out(self, names: Sequence[str]) -> list[str]:
        out = []
        for n, t in zip(names, self.feature_types):
            out += [f"sin_{n}", f"cos_{n}"] if t == "angle" else [n]
        return out
