"""Quantum kernel regression.

Model
-----
A feature map prepares |phi(x)> = U(x)|0>^n (one qubit per feature) and the fidelity kernel is

    k(x, x') = |<phi(x)|phi(x')>|^2 .

Predictions come from kernel ridge regression (regression tasks) or L2-regularised logistic
regression on the kernel features [k(x, x_1), ..., k(x, x_n)] (probability task). By Schuld
(2021), "supervised quantum models are kernel methods", so this is the natural second family
next to the variational model.

Feature maps
------------
``angle`` : RY(x_j) on qubit j. The overlap factorises:
            k(x, x') = prod_j cos^2((x_j - x'_j) / 2),
            i.e. frequencies {-1, 0, 1} per feature -- classically trivial, kept as a sanity
            check (``angle_kernel_closed_form``) and as the "how much does entanglement add?"
            control.
``iqp``   : ``qml.IQPEmbedding`` (Havlicek et al. 2019): H, RZ(x_j), and ZZ(x_j * x_k) for all
            feature pairs, repeated ``reps`` times. Not known to be classically efficient in
            general, and it creates products between features.

Bandwidth
---------
``bandwidth`` multiplies the *real* and *binary* inputs before encoding (smaller -> smoother
kernel; Shaydulin & Wild 2022 show bandwidth tuning is essential for quantum kernels).
*Angle* features are never rescaled: a non-integer multiple of an angle would destroy its exact
2*pi periodicity, which is the property this benchmark is testing.

Known interaction (observed on tides): IQP's ZZ(x_j * x_k) terms take *products* of inputs.
For angle features spanning [-pi, pi) these products reach ~pi^2 and are not 2*pi-periodic in
either angle, so the IQP kernel becomes nearly diagonal (it memorises) and bandwidth cannot
fix it, because angles are not rescaled. The angle feature map has no such problem
(untuned R^2 0.87-0.91 on tides vs about 0 for IQP). Leave ``feature_map`` to the tuner, and
report this as a finding: the standard IQP map is a poor fit for periodic inputs.

Cost
----
On hardware each kernel entry needs its own circuit, so the Gram matrix costs O(n^2) circuits.
In noiseless simulation we can compute each state vector once and take inner products, i.e. O(n)
circuits. The result is the same kernel exactly; only wall-clock time differs. Report this
caveat next to any timing comparison.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from ..encoding import AngleScaler
from .base import Types

FEATURE_MAPS = ("angle", "iqp")


def _state_fn(n_qubits: int, feature_map: str, reps: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state(x):
        if feature_map == "angle":
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        elif feature_map == "iqp":
            qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=reps)
        else:
            raise ValueError(f"unknown feature map {feature_map!r}; choose from {FEATURE_MAPS}")
        return qml.state()

    return state


def angle_kernel_closed_form(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact angle-map kernel, prod_j cos^2((a_j - b_j)/2). Used to validate the simulator path."""
    diff = A[:, None, :] - B[None, :, :]
    return np.prod(np.cos(diff / 2) ** 2, axis=-1)


class QuantumKernelRegressor(RegressorMixin, BaseEstimator):
    """scikit-learn regressor; expects inputs already mapped to angles (see ``qkernel``)."""

    def __init__(self, feature_types: Types, task: str = "regression",
                 feature_map: str = "iqp", reps: int = 1, bandwidth: float = 1.0,
                 alpha: float = 1e-3, seed: int = 0, batch_size: int = 512):
        self.feature_types = feature_types
        self.task = task
        self.feature_map = feature_map
        self.reps = reps
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.seed = seed  # unused (the model is deterministic); accepted for a uniform API
        self.batch_size = batch_size

    # --- kernel ----------------------------------------------------------------------------

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        mask = np.array([t != "angle" for t in self.feature_types])
        X[:, mask] *= self.bandwidth
        return X

    def states(self, X: np.ndarray) -> np.ndarray:
        """State vectors |phi(x)> for each row, shape (n, 2**n_qubits)."""
        if getattr(self, "_state", None) is None:
            self._state = _state_fn(len(self.feature_types), self.feature_map, self.reps)
        Xs = self._scale(X)
        out = [np.asarray(self._state(Xs[i: i + self.batch_size]))
               for i in range(0, len(Xs), self.batch_size)]
        return np.concatenate(out).reshape(len(Xs), -1)

    @staticmethod
    def kernel_from_states(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
        return np.abs(SA @ SB.conj().T) ** 2

    def kernel(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        SA = self.states(A)
        return self.kernel_from_states(SA, SA if B is None else self.states(B))

    # --- scikit-learn API ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[1] != len(self.feature_types):
            raise ValueError(f"X has {X.shape[1]} columns but {len(self.feature_types)} feature types")
        self._state = None
        self.train_states_ = self.states(X)
        K = self.kernel_from_states(self.train_states_, self.train_states_)
        if self.task == "probability":
            self.head_ = LogisticRegression(C=1.0 / self.alpha, max_iter=5000)
            self.head_.fit(K, y.astype(int))
        else:
            self.y_mean_ = float(y.mean())
            self.head_ = KernelRidge(alpha=self.alpha, kernel="precomputed")
            self.head_.fit(K, y - self.y_mean_)
        return self

    def predict(self, X) -> np.ndarray:
        K = self.kernel_from_states(self.states(np.asarray(X, dtype=float)), self.train_states_)
        if self.task == "probability":
            return self.head_.predict_proba(K)[:, 1]
        return self.head_.predict(K) + self.y_mean_

    def n_parameters(self) -> int:
        """Number of fitted dual coefficients (one per training point)."""
        return len(self.train_states_)

    def __getstate__(self):  # QNodes don't pickle; rebuilt lazily
        state = self.__dict__.copy()
        state["_state"] = None
        return state


def qkernel(feature_types: Types, task: str, **hp):
    """Factory used by the runner: angle encoding + quantum kernel model."""
    return make_pipeline(AngleScaler(feature_types), QuantumKernelRegressor(feature_types, task, **hp))


SEARCH_SPACES = {"qkernel": {"feature_map": ("cat", list(FEATURE_MAPS)), "reps": ("int", 1, 3),
                             "bandwidth": ("log", 0.05, 2.0), "alpha": ("log", 1e-6, 1e0)}}
