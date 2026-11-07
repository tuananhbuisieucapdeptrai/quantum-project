"""Dataset registry. ``get_dataset("tides")`` returns a ``RegressionDataset``."""

from __future__ import annotations

from . import photoz, soccer, tides, wind
from .base import RegressionDataset, load_config

REGISTRY = {"tides": tides, "wind": wind, "soccer": soccer, "photoz": photoz}
DATASETS = list(REGISTRY)


def get_dataset(name: str, cfg: dict | None = None) -> RegressionDataset:
    cfg = cfg or load_config(name)
    return REGISTRY[name].load(cfg)


def fetch_dataset(name: str, cfg: dict | None = None, refresh: bool = False):
    cfg = cfg or load_config(name)
    return REGISTRY[name].fetch(cfg, refresh=refresh)


__all__ = ["DATASETS", "REGISTRY", "RegressionDataset", "fetch_dataset", "get_dataset",
           "load_config"]
