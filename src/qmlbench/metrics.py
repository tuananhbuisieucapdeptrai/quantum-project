"""Metrics: generic regression, probability scoring, and domain-standard metrics."""

from __future__ import annotations

import numpy as np


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# --- probability tasks (soccer xG) -------------------------------------------------------

def brier(y, p) -> float:
    return float(np.mean((np.asarray(y, float) - np.clip(p, 0, 1)) ** 2))


def log_loss(y, p, eps: float = 1e-6) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_skill(y, p) -> float:
    """1 - Brier / Brier(constant base rate). >0 means better than predicting the average."""
    y = np.asarray(y, float)
    ref = np.mean((y - y.mean()) ** 2)
    return float(1 - brier(y, p) / ref) if ref > 0 else float("nan")


def calibration_curve(y, p, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean predicted, observed frequency, count) per quantile bin of p."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    mp, fy, n = [], [], []
    for b in range(len(edges) - 1):
        m = which == b
        if m.any():
            mp.append(p[m].mean()); fy.append(y[m].mean()); n.append(m.sum())
    return np.array(mp), np.array(fy), np.array(n)


# --- photometric redshift ------------------------------------------------------------------

def sigma_nmad(z_true, z_pred) -> float:
    """Normalised median absolute deviation of dz/(1+z)."""
    dz = (np.asarray(z_pred) - np.asarray(z_true)) / (1 + np.asarray(z_true))
    return float(1.4826 * np.median(np.abs(dz - np.median(dz))))


def outlier_fraction(z_true, z_pred, threshold: float = 0.15) -> float:
    dz = np.abs(np.asarray(z_pred) - np.asarray(z_true)) / (1 + np.asarray(z_true))
    return float(np.mean(dz > threshold))


def evaluate(dataset_name: str, task: str, y, p) -> dict[str, float]:
    """All metrics relevant to one dataset, as a flat dict (one row of the results table)."""
    if task == "probability":
        return {"brier": brier(y, p), "log_loss": log_loss(y, p), "brier_skill": brier_skill(y, p)}
    out = {"rmse": rmse(y, p), "mae": mae(y, p), "r2": r2(y, p)}
    if dataset_name == "photoz":
        out |= {"sigma_nmad": sigma_nmad(y, p), "outlier_frac": outlier_fraction(y, p)}
    return out
