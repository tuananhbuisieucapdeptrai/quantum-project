"""Checks on the downloaded, processed datasets. Skipped if a dataset hasn't been fetched."""

import numpy as np
import pytest

from qmlbench.data import DATASETS, get_dataset, load_config
from qmlbench.data.base import processed_path
from qmlbench.splits import make_split

EXPECTED = {"tides": (5, "regression"), "wind": (2, "regression"),
            "soccer": (7, "probability"), "photoz": (5, "regression")}


@pytest.mark.network
@pytest.mark.parametrize("name", DATASETS)
def test_processed_dataset(name):
    if not processed_path(name).exists():
        pytest.skip(f"run `qmlbench-fetch --only {name}` first")
    cfg = load_config(name)
    ds = get_dataset(name, cfg)
    d, task = EXPECTED[name]
    assert ds.n_features == d and ds.task == task
    assert ds.n_samples > 1000
    assert np.isfinite(ds.X).all() and np.isfinite(ds.y).all()
    for j, t in enumerate(ds.feature_types):
        if t == "angle":
            assert ds.X[:, j].min() >= -np.pi and ds.X[:, j].max() < np.pi
    if task == "probability":
        assert set(np.unique(ds.y)) <= {0.0, 1.0}
    tr, te = make_split(ds, cfg["split"])
    assert len(tr) > 0 and len(te) > 0 and not set(tr) & set(te)
