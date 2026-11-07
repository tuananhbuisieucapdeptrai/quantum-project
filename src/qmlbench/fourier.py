"""Frequency analysis of data and fitted models on the torus of angle features.

Why
---
A data re-uploading circuit with L layers and one Pauli-rotation encoding per feature is a
truncated Fourier series (Schuld, Sweke & Meyer 2021):

    f(x) = sum_{omega in Omega_L} c_omega e^{i omega . x},   Omega_L = {-L..L}^d.

The angle-map quantum kernel prod_j cos^2((x_j - x'_j)/2) has the same spectrum as L = 1.
So for purely-angle data (tides) we can ask two exact questions:

1. **Ceiling.** What is the best test score *any* model restricted to a frequency set can
   reach? Fit the target by least squares on that set (``spectrum_ceiling``). If the
   1-layer ceiling is already high, a 1-layer VQC that scores lower is limited by training
   or by which coefficients the circuit can reach -- not by its frequencies.
2. **What did the model learn?** Sample a fitted model on a regular grid and take the FFT
   (``model_spectrum``). For a model whose spectrum is inside {-K..K}^d a grid of
   n >= 2K + 1 points per axis gives the coefficients *exactly*. For models that are not
   band-limited (GP, MLP) the result is their spectrum up to aliasing.

Conventions
-----------
Frequencies are stored on a half-lattice (first non-zero entry positive), because omega and
-omega describe the same real cosine. ``amplitude`` is that cosine's amplitude in target
units: y ~ a cos(omega.x) + b sin(omega.x) -> amplitude = hypot(a, b) = 2 |c_omega|.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

# --- frequency sets ------------------------------------------------------------------------

def frequency_set(d: int, l_inf: int, l1: int | None = None, additive: bool = False) -> np.ndarray:
    """Non-zero integer frequency vectors on the half-lattice, shape (m, d).

    l_inf    : max |omega_j| (a re-uploading circuit with L layers has l_inf = L).
    l1       : optional cap on sum_j |omega_j| (the "order" of a cross term).
    additive : only one non-zero entry (a sum of per-feature series; no feature interactions).
    """
    out = []
    for w in itertools.product(range(-l_inf, l_inf + 1), repeat=d):
        w = np.asarray(w)
        nz = np.flatnonzero(w)
        if nz.size == 0 or w[nz[0]] < 0:
            continue
        if l1 is not None and np.abs(w).sum() > l1:
            continue
        if additive and nz.size > 1:
            continue
        out.append(w)
    return np.array(out, dtype=int).reshape(-1, d)


def fourier_design(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """[1, cos(X W^T), sin(X W^T)], shape (n, 1 + 2m)."""
    P = np.asarray(X, dtype=float) @ np.asarray(W, dtype=float).T
    return np.hstack([np.ones((len(P), 1)), np.cos(P), np.sin(P)])


# --- data side -----------------------------------------------------------------------------

def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def data_spectrum(X: np.ndarray, y: np.ndarray, W: np.ndarray, ridge: float = 0.0) -> dict:
    """Least-squares Fourier coefficients of ``y`` on the frequency set ``W``.

    Returns {"freqs" (m, d), "amplitude" (m,), "phase" (m,), "mean", "coef" (1 + 2m,)}.
    ``ridge`` > 0 adds a small L2 penalty (useful when frequencies are nearly collinear,
    e.g. tidal phases sampled along a single time line).
    """
    A = fourier_design(X, W)
    if ridge > 0:
        coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
    else:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    m = len(W)
    a, b = coef[1:m + 1], coef[m + 1:]
    return {"freqs": W, "amplitude": np.hypot(a, b), "phase": np.arctan2(b, a),
            "mean": float(coef[0]), "coef": coef}


def spectrum_ceiling(X_train, y_train, X_test, y_test, W: np.ndarray, ridge: float = 0.0) -> dict:
    """Best least-squares fit on frequency set ``W``: train and test R^2, number of terms."""
    spec = data_spectrum(X_train, y_train, W, ridge=ridge)
    fit = lambda X: fourier_design(X, W) @ spec["coef"]
    return {"n_freqs": len(W), "train_r2": _r2(y_train, fit(X_train)),
            "test_r2": _r2(y_test, fit(X_test))}


# --- model side ----------------------------------------------------------------------------

def model_spectrum(predict: Callable[[np.ndarray], np.ndarray], d: int, k_max: int,
                   n_grid: int | None = None, batch: int = 4096) -> dict:
    """Fourier coefficients of a fitted model over the angle torus [-pi, pi)^d.

    ``predict`` maps an (n, d) array of angles to (n,) outputs. The model is sampled on a
    regular grid of ``n_grid`` (default 2*k_max + 1) points per axis and transformed with an
    n-dimensional FFT. Exact when the model's spectrum lies in {-k_max..k_max}^d.
    Returns the same keys as ``data_spectrum`` (freqs on the half-lattice with
    max |omega_j| <= k_max, amplitude, phase, mean).
    """
    n = n_grid or 2 * k_max + 1
    if n < 2 * k_max + 1:
        raise ValueError("n_grid must be >= 2*k_max + 1 to resolve frequencies up to k_max")
    axis = -np.pi + 2 * np.pi * np.arange(n) / n
    grid = np.stack(np.meshgrid(*[axis] * d, indexing="ij"), axis=-1).reshape(-1, d)
    vals = np.concatenate([np.asarray(predict(grid[i:i + batch]), dtype=float).ravel()
                           for i in range(0, len(grid), batch)])
    # The grid starts at -pi, so shift each coefficient by e^{i omega pi} to refer to x = 0.
    c = np.fft.fftn(vals.reshape([n] * d)) / n ** d
    W = frequency_set(d, k_max)
    idx = tuple((W % n).T)
    cw = c[idx] * np.exp(1j * np.pi * W.sum(axis=1))
    # sum_omega c_omega e^{i omega x}: pair (omega, -omega) -> 2 Re(c e^{i omega x})
    #   = 2|c| cos(omega x + arg c)  -> amplitude 2|c|; as a cos + b sin: phase = -arg c.
    return {"freqs": W, "amplitude": 2 * np.abs(cw), "phase": -np.angle(cw),
            "mean": float(np.real(c[(0,) * d]))}


def power_by_order(freqs: np.ndarray, amplitude: np.ndarray, norm: str = "linf") -> dict[int, float]:
    """Total power (sum of amplitude^2 / 2, the variance it explains) per frequency order.

    norm "linf": order = max_j |omega_j| (the number of layers needed to express it);
    norm "l1"  : order = sum_j |omega_j|.
    """
    a = np.abs(freqs)
    order = a.max(axis=1) if norm == "linf" else a.sum(axis=1)
    power = amplitude ** 2 / 2
    return {int(k): float(power[order == k].sum()) for k in np.unique(order)}


def top_frequencies(spec: dict, names: list[str], k: int = 10) -> list[tuple[str, float]]:
    """The k strongest components as readable labels, e.g. ('2*M2', 0.08) or ('M2-S2', ...)."""
    out = []
    for i in np.argsort(-spec["amplitude"])[:k]:
        terms = []
        for w, nm in zip(spec["freqs"][i], names):
            if w == 0:
                continue
            sign = "-" if w < 0 else "+"
            mag = "" if abs(w) == 1 else f"{abs(w)}*"
            terms.append(f"{sign}{mag}{nm}")
        out.append(("".join(terms).lstrip("+"), float(spec["amplitude"][i])))
    return out
