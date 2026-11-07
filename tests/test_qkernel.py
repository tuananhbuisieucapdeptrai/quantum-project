"""Tests for the quantum kernel model."""

import numpy as np
import pytest

from qmlbench.metrics import r2
from qmlbench.models.quantum_kernel import QuantumKernelRegressor, angle_kernel_closed_form, qkernel

rng = np.random.default_rng(0)


def test_angle_kernel_matches_closed_form():
    A, B = rng.uniform(-np.pi, np.pi, (25, 3)), rng.uniform(-np.pi, np.pi, (15, 3))
    m = QuantumKernelRegressor(["angle"] * 3, feature_map="angle")
    assert np.allclose(m.kernel(A, B), angle_kernel_closed_form(A, B))


@pytest.mark.parametrize("fmap", ["angle", "iqp"])
def test_kernel_is_valid(fmap):
    A = rng.uniform(-1, 1, (30, 3))
    K = QuantumKernelRegressor(["real"] * 3, feature_map=fmap, reps=2).kernel(A)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9   # positive semi-definite


def test_bandwidth_leaves_angles_untouched():
    m = QuantumKernelRegressor(["angle", "real"], bandwidth=0.1)
    X = np.array([[3.0, 3.0]])
    assert np.allclose(m._scale(X), [[3.0, 0.3]])


def test_fits_periodic_function():
    X = rng.uniform(-np.pi, np.pi, (200, 2))
    f = lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) + 0.5
    Xt = rng.uniform(-np.pi, np.pi, (200, 2))
    p = qkernel(["angle", "angle"], "regression", feature_map="angle", alpha=1e-4).fit(X, f(X))
    assert r2(f(Xt), p.predict(Xt)) > 0.99


def test_probability_task():
    X = rng.uniform(-1, 1, (200, 2))
    y = (rng.random(200) < 1 / (1 + np.exp(-3 * X[:, 0]))).astype(float)
    p = qkernel(["real", "real"], "probability").fit(X, y).predict(X)
    assert np.all((p >= 0) & (p <= 1))
