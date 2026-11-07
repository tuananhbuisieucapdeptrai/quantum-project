"""Tests for the data re-uploading regressor, including its Fourier-degree property."""

import numpy as np
import pytest

from qmlbench.metrics import r2
from qmlbench.models.quantum_vqc import ReuploadingRegressor, vqc

rng = np.random.default_rng(0)
X1 = rng.uniform(-np.pi, np.pi, (150, 1))
GRID = np.linspace(-np.pi, np.pi, 200)[:, None]


def _fit(target, n_layers, epochs=150):
    return vqc(["angle"], "regression", n_layers=n_layers, epochs=epochs).fit(X1, target(X1))


def test_fits_scaled_shifted_sine():
    # Output scaling + target standardisation: a target outside [-1, 1] must still fit.
    f = lambda x: 3 * np.sin(x[:, 0]) + 1
    assert r2(f(GRID), _fit(f, 1).predict(GRID)) > 0.99


def test_fourier_degree_limited_by_layers():
    # Schuld, Sweke & Meyer (2021): L layers of RY encoding -> frequencies |k| <= L.
    f = lambda x: np.cos(2 * x[:, 0])
    assert r2(f(GRID), _fit(f, 1).predict(GRID)) < 0.2
    assert r2(f(GRID), _fit(f, 2).predict(GRID)) > 0.95


def test_probability_task_outputs_probabilities():
    X = rng.uniform(-1, 1, (200, 2))
    y = (rng.random(200) < 1 / (1 + np.exp(-3 * X[:, 0]))).astype(float)
    p = vqc(["real", "real"], "probability", n_layers=1, epochs=30).fit(X, y).predict(X)
    assert p.shape == (200,) and np.all((p > 0) & (p < 1))


def test_parameter_count_and_determinism():
    est = ReuploadingRegressor(["real", "angle", "real"], n_layers=2)
    assert est.n_parameters() == 3 * 3 * 3 + 2
    a = _fit(lambda x: np.sin(x[:, 0]), 1, epochs=5).predict(GRID)
    b = _fit(lambda x: np.sin(x[:, 0]), 1, epochs=5).predict(GRID)
    assert np.allclose(a, b)


def test_rejects_wrong_feature_count():
    with pytest.raises(ValueError):
        ReuploadingRegressor(["real"]).fit(np.zeros((30, 2)), np.zeros(30))
