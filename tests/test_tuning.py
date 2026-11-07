"""Tests for the equal-budget tuner and the tuned runner."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from qmlbench.data.base import RegressionDataset, processed_path
from qmlbench.tuning import inner_folds, suggest, tune


def _ds(kind, n=120):
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, (n, 1))
    t = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC").to_numpy()
    return RegressionDataset("toy", X, np.sin(3 * X[:, 0]), ["a"], ["angle"], "regression", kind,
                             time=t, groups=np.repeat(np.arange(n // 6), 6))


def test_temporal_folds_are_contiguous_blocks():
    ds = _ds("temporal")
    idx = np.random.default_rng(1).permutation(120)[:90]
    for tr, va in inner_folds(ds, idx, 3, 0):
        assert not set(tr) & set(va)
        t_va = ds.time[va]
        # no training point lies strictly inside the validation block's time range
        inside = (ds.time[tr] > t_va.min()) & (ds.time[tr] < t_va.max())
        assert not inside.any()


def test_group_folds_keep_groups_together():
    ds = _ds("group")
    for tr, va in inner_folds(ds, np.arange(120), 3, 0):
        assert not set(ds.groups[tr]) & set(ds.groups[va])


def test_suggest_kinds():
    import optuna
    space = {"a": ("log", 1e-3, 1.0), "b": ("int", 1, 3), "c": ("cat", ["x", "y"]),
             "d": ("float", 0.0, 1.0)}
    p = suggest(optuna.trial.FixedTrial({"a": 0.01, "b": 2, "c": "y", "d": 0.5}), space)
    assert p == {"a": 0.01, "b": 2, "c": "y", "d": 0.5}


def test_tuning_finds_better_params():
    # Ridge on raw angle features can't fit sin(3x); a well-tuned RFF can. Check the tuner
    # beats the RFF default in inner-CV loss.
    from qmlbench.models.classical import rff
    from qmlbench.tuning import cv_loss
    ds = _ds("random")
    idx = np.arange(ds.n_samples)

    def fit_predict(p, tr, ev):
        return rff(ds.feature_types, ds.task, **p).fit(ds.X[tr], ds.y[tr]).predict(ds.X[ev])

    space = {"gamma": ("log", 1e-2, 1e1), "alpha": ("log", 1e-6, 1e0)}
    res = tune(fit_predict, space, ds, idx, trials=15, cv_folds=3, seed=0)
    default = cv_loss(fit_predict, {}, ds, inner_folds(ds, idx, 3, 0))
    assert res["n_trials"] == 15 and res["cv_loss"] < default


def test_empty_space_skips_tuning():
    res = tune(lambda p, a, b: np.zeros(len(b)), {}, _ds("random"), np.arange(10), 5, 3, 0)
    assert res["params"] == {} and res["n_trials"] == 0 and math.isnan(res["cv_loss"])


@pytest.mark.network
def test_run_one_tuned_on_real_data(tmp_path):
    if not processed_path("wind").exists():
        pytest.skip("run `qmlbench-fetch --only wind` first")
    from qmlbench.data import get_dataset, load_config
    from qmlbench.run import run_one
    cfg = load_config("wind")
    row = run_one(get_dataset("wind", cfg), cfg, "domain:power_curve", 100, 0, 500,
                  trials=4, cv_folds=3)
    assert row["family"] == "domain" and row["n_trials"] == 4
    assert "bin_width" in json.loads(row["params"]) and row["r2"] > 0.5
