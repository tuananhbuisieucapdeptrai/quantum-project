"""Train/test splitting that matches each domain's structure, plus training-size subsampling.

temporal : train strictly before ``test_start``, test on or after it (tides, wind).
           A random split would leak: neighbouring hours/10-min rows are almost identical.
group    : whole groups (soccer matches) go to either train or test.
random   : i.i.d. split (photoz).

The benchmark varies the training-set size n_train; ``subsample`` draws n_train points from
the training pool with a given seed, while the test set stays fixed across all runs of a
domain so every model is scored on exactly the same points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data.base import RegressionDataset


def make_split(ds: RegressionDataset, split_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_pool_idx, test_idx)."""
    idx = np.arange(ds.n_samples)
    kind = split_cfg["kind"]
    if kind != ds.split_kind:
        raise ValueError(f"{ds.name}: config split {kind!r} != dataset split {ds.split_kind!r}")
    if kind == "temporal":
        t = pd.to_datetime(ds.time, utc=True)
        cut = pd.Timestamp(split_cfg["test_start"], tz="UTC")
        return idx[t < cut], idx[t >= cut]
    if kind == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=split_cfg["test_size"],
                                random_state=split_cfg.get("seed", 0))
        tr, te = next(gss.split(ds.X, ds.y, groups=ds.groups))
        return tr, te
    if kind == "random":
        tr, te = train_test_split(idx, test_size=split_cfg["test_size"],
                                  random_state=split_cfg.get("seed", 0))
        return np.sort(tr), np.sort(te)
    raise ValueError(f"unknown split kind {kind!r}")


def subsample(pool: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Draw ``n`` indices from ``pool`` without replacement (all of it if n is None/too big)."""
    if n is None or n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=n, replace=False))


def cap_test(test: np.ndarray, n_max: int | None, seed: int = 0) -> np.ndarray:
    """Fixed random cap on the test set (quantum simulation cost grows with test size)."""
    return subsample(test, n_max, seed)
