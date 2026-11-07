"""Domain baselines on the real datasets (skipped if a dataset hasn't been fetched)."""

import numpy as np
import pytest

from qmlbench.data import get_dataset, load_config
from qmlbench.data.base import processed_path
from qmlbench.metrics import evaluate
from qmlbench.models.domain import DOMAIN_BASELINES
from qmlbench.splits import cap_test, make_split, subsample

CASES = [(d, b) for d, bs in DOMAIN_BASELINES.items() for b in bs]


@pytest.mark.network
@pytest.mark.parametrize("dataset,baseline", CASES)
def test_domain_baseline_runs_and_beats_mean(dataset, baseline):
    if not processed_path(dataset).exists():
        pytest.skip(f"run `qmlbench-fetch --only {dataset}` first")
    cfg = load_config(dataset)
    ds = get_dataset(dataset, cfg)
    pool, test = make_split(ds, cfg["split"])
    tr, te = subsample(pool, 300, 0), cap_test(test, 1000)
    model = DOMAIN_BASELINES[dataset][baseline](ds).fit(tr)
    pred = model.predict(te)
    assert pred.shape == te.shape and np.isfinite(pred).all()
    m = evaluate(dataset, ds.task, ds.y[te], pred)
    # Every expert method should beat predicting the mean / base rate.
    assert (m["brier_skill"] > 0) if ds.task == "probability" else (m["r2"] > 0)
