"""Offline unit tests: encodings, splits, metrics, soccer geometry, classical factories."""

import numpy as np
import pandas as pd
import pytest

from qmlbench.data.base import RegressionDataset, wrap_angle
from qmlbench.data.soccer import shot_geometry
from qmlbench.data.tides import constituent_phases
from qmlbench.encoding import AngleScaler, PeriodicExpander
from qmlbench.metrics import brier_skill, outlier_fraction, r2, rmse, sigma_nmad
from qmlbench.models.classical import MODELS
from qmlbench.splits import make_split, subsample


def test_wrap_angle_range():
    a = wrap_angle(np.linspace(-20, 20, 1001))
    assert a.min() >= -np.pi and a.max() < np.pi


def test_angle_scaler_types():
    X = np.column_stack([np.linspace(0, 10, 100), np.linspace(-3, 3, 100),
                         np.r_[np.zeros(50), np.ones(50)]])
    Z = AngleScaler(["real", "angle", "binary"]).fit(X).transform(X)
    assert np.all(np.abs(Z[:, 0]) <= np.pi / 2 + 1e-12)   # reals on a half circle
    assert np.allclose(Z[:, 1], X[:, 1])                   # angles untouched
    assert set(np.round(Z[:, 2], 6)) == {round(-np.pi / 2, 6), round(np.pi / 2, 6)}


def test_angle_scaler_clips_outliers():
    X = np.r_[np.linspace(0, 1, 1000), [1e6]][:, None]
    Z = AngleScaler(["real"]).fit(X).transform(X)
    assert Z.max() <= np.pi / 2 + 1e-12


def test_periodic_expander():
    X = np.column_stack([np.arange(5.0), np.zeros(5)])
    pe = PeriodicExpander(["real", "angle"]).fit(X)
    Z = pe.transform(X)
    assert Z.shape == (5, 3) and np.allclose(Z[:, 2], 1.0)  # cos(0)
    assert pe.get_feature_names_out(["a", "b"]) == ["a", "sin_b", "cos_b"]


def test_tidal_phase_period():
    # M2 returns to the same phase after one M2 period (12.4206 h).
    period = 360 / 28.9841042
    p = constituent_phases(np.array([0.0, period]), ["M2"])
    assert np.isclose(np.cos(p[0, 0]), np.cos(p[1, 0]))


def test_shot_geometry():
    d, ang, bearing = shot_geometry(np.array([108.0, 119.0]), np.array([40.0, 40.0]))
    assert np.allclose(d, [12, 1]) and np.allclose(bearing, 0)
    assert np.isclose(ang[0], 2 * np.arctan(4 / 12))     # penalty spot
    assert ang[1] > ang[0]                               # closer -> wider angle


def test_metrics():
    y = np.array([0.1, 0.2, 0.3])
    assert rmse(y, y) == 0 and r2(y, y) == 1
    assert sigma_nmad(y, y) == 0 and outlier_fraction(y, y + 1) == 1
    assert brier_skill(np.array([0, 1, 0, 1.0]), np.array([0, 1, 0, 1.0])) == 1


def _toy(kind):
    n = 200
    t = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return RegressionDataset("toy", np.random.rand(n, 2), np.random.rand(n), ["a", "b"],
                             ["real", "angle"], "regression", kind, time=t.to_numpy(),
                             groups=np.repeat(np.arange(20), 10))


def test_splits_disjoint():
    ds = _toy("temporal")
    tr, te = make_split(ds, {"kind": "temporal", "test_start": "2020-01-06"})
    assert len(set(tr) & set(te)) == 0 and ds.time[tr].max() < ds.time[te].min()
    ds = _toy("group")
    tr, te = make_split(ds, {"kind": "group", "test_size": 0.25, "seed": 0})
    assert not set(ds.groups[tr]) & set(ds.groups[te])
    assert len(subsample(tr, 10, 0)) == 10


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("task", ["regression", "probability"])
def test_classical_factories(name, task):
    rng = np.random.default_rng(0)
    X = rng.random((60, 2))
    y = (X[:, 0] > 0.5).astype(float) if task == "probability" else X[:, 0]
    p = MODELS[name](["real", "angle"], task).fit(X, y).predict(X)
    assert p.shape == (60,) and np.all(np.isfinite(p))
