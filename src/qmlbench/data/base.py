"""Common dataset interface for every benchmark domain.

Each domain module (tides, wind, soccer, photoz) provides two functions:

    fetch(cfg, refresh=False) -> Path
        Download the raw data (cached under data/raw/<name>/) and write one tidy table
        to data/processed/<name>.parquet. Returns that path.

    load(cfg) -> RegressionDataset
        Read the processed table and turn it into model-ready arrays.

``RegressionDataset`` carries per-feature *types* so that encodings can treat genuinely
periodic inputs (tidal phases, wind direction, shot bearing) differently from ordinary
real-valued inputs. This is central to the benchmark's research question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from .io import PROCESSED_DIR, PROJECT_ROOT

FeatureType = Literal["angle", "real", "binary"]
Task = Literal["regression", "probability"]
SplitKind = Literal["temporal", "group", "random"]

CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class RegressionDataset:
    """Model-ready view of one domain.

    Attributes
    ----------
    X : (n, d) float array of features. ``angle`` features are in radians in [-pi, pi).
    y : (n,) float array of targets (for ``probability`` tasks, y is 0/1 outcomes).
    feature_names, feature_types : per-column metadata, len d.
    task : "regression" (squared-error style metrics) or "probability" (Brier / log-loss).
    split_kind : how train/test must be separated for this domain (see ``qmlbench.splits``).
    time : optional (n,) datetime64 array, required for temporal splits.
    groups : optional (n,) array of group ids, required for group splits.
    extras : optional DataFrame of extra columns aligned with X (e.g. domain baselines
             such as NOAA's official tide prediction or StatsBomb's own xG).
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    feature_types: list[FeatureType]
    task: Task
    split_kind: SplitKind
    time: np.ndarray | None = None
    groups: np.ndarray | None = None
    extras: pd.DataFrame | None = None
    target_name: str = "y"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, d = self.X.shape
        assert self.y.shape == (n,), "y must be 1-D and aligned with X"
        assert len(self.feature_names) == d == len(self.feature_types)
        assert set(self.feature_types) <= {"angle", "real", "binary"}
        if self.split_kind == "temporal":
            assert self.time is not None, "temporal split needs a time column"
        if self.split_kind == "group":
            assert self.groups is not None, "group split needs group ids"

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def summary(self) -> str:
        types = ", ".join(f"{n}:{t}" for n, t in zip(self.feature_names, self.feature_types))
        return (f"[{self.name}] n={self.n_samples} d={self.n_features} task={self.task} "
                f"split={self.split_kind} target={self.target_name}\n  features: {types}")


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


def processed_path(name: str) -> Path:
    return PROCESSED_DIR / f"{name}.parquet"


def read_processed(name: str) -> pd.DataFrame:
    path = processed_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `qmlbench-fetch --only {name}` first."
        )
    return pd.read_parquet(path)


def write_processed(df: pd.DataFrame, name: str) -> Path:
    path = processed_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap radians into [-pi, pi)."""
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi
