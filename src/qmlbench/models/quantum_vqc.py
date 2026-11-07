"""Variational quantum regressor with data re-uploading.

Model
-----
For input x in R^d (mapped to angles by ``AngleScaler``) and n = d qubits:

    |psi(x, theta)> = W_L(theta_L) S(x) ... W_1(theta_1) S(x) W_0(theta_0) |0>^n
    z(x)  = <psi| M |psi>,      M = (1/n) sum_i Z_i          (z in [-1, 1])
    f(x)  = a * z(x) + c                     regression  (in standardised target units)
    p(x)  = sigmoid(a * z(x) + c)            probability task (soccer xG)

S(x) = prod_j RY(x_j) on qubit j (one Pauli rotation per feature per layer). W_l are
trainable ``qml.StronglyEntanglingLayers`` blocks (an arbitrary single-qubit rotation per
qubit plus a ring of CNOTs). The extra block W_0 before the first encoding matters: without
it the first encoding acts on |0> and the model loses some of its Fourier coefficients.

Why this matters for the benchmark (Schuld, Sweke & Meyer 2021): z is a truncated Fourier
series  z(x) = sum_{omega in Omega} c_omega(theta) e^{i omega . x}, with integer
frequencies |omega_j| <= L (L = ``n_layers``). So
  * tides: the model can express each constituent phase up to harmonic L;
  * wind: direction effects up to the L-th harmonic of the angle;
  * photoz / soccer: real features live in [-pi/2, pi/2], so the model is a low-order
    trigonometric polynomial there -- a strong smoothness prior.

Training
--------
* Simulator: ``default.qubit`` with backprop through PyTorch. For <= 8 qubits and batched
  inputs this is faster than ``lightning.qubit`` + adjoint, which doesn't batch.
* Targets are standardised inside ``fit`` (and un-standardised in ``predict``), so the
  [-1, 1] range of z never limits what the model can reach (tide levels span -1.6..1.4 m).
* Loss: mean squared error (regression) or binary cross-entropy (probability).
* Adam on mini-batches; a random ``val_frac`` of the training points is held out for early
  stopping, and the weights with the best validation loss are kept.
* Budget (v2): mini-batches of 128, up to 400 epochs, stop after 40 epochs without
  improvement. The first benchmark (v1) used batches of 32, 100 epochs and patience 15;
  docs/FOURIER.md shows that stopped training far too early (tides, n=1000: R^2 0.83 vs
  0.90 with the same circuit). Pass ``epochs=100, patience=15, batch_size=32`` for v1.
  A halve-on-plateau learning-rate schedule was tried and did not help (3 seeds).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline

from ..encoding import AngleScaler
from .base import Types

DTYPE = torch.float64


def _trainable_block(w_l: torch.Tensor, wires: list[int]) -> None:
    """One trainable block W_l; ``w_l`` has shape (n_qubits, 3)."""
    if len(wires) == 1:  # StronglyEntanglingLayers needs >= 2 wires for its CNOT ring
        qml.Rot(w_l[0, 0], w_l[0, 1], w_l[0, 2], wires=wires[0])
    else:
        qml.StronglyEntanglingLayers(w_l.unsqueeze(0), wires=wires)


def circuit_reupload(x: torch.Tensor, weights: torch.Tensor, n_qubits: int):
    """W_0, then n_layers x [S(x), W_l]. ``weights`` has shape (n_layers + 1, n_qubits, 3).

    ``x`` may be a single sample (n_qubits,) or a batch (B, n_qubits); PennyLane broadcasts.
    Returns one <Z_i> per qubit; the caller averages them.
    """
    wires = list(range(n_qubits))
    _trainable_block(weights[0], wires)
    for l in range(1, weights.shape[0]):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        _trainable_block(weights[l], wires)
    return [qml.expval(qml.PauliZ(i)) for i in wires]


def weight_shape(n_qubits: int, n_layers: int) -> tuple[int, int, int]:
    return (n_layers + 1, n_qubits, 3)


class ReuploadingRegressor(RegressorMixin, BaseEstimator):
    """scikit-learn regressor; expects inputs already mapped to angles (see ``vqc``)."""

    def __init__(self, feature_types: Types, task: str = "regression", n_layers: int = 2,
                 lr: float = 0.05, epochs: int = 400, batch_size: int = 128,
                 val_frac: float = 0.15, patience: int = 40, seed: int = 0):
        self.feature_types = feature_types
        self.task = task
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.patience = patience
        self.seed = seed

    # --- circuit ---------------------------------------------------------------------------

    def _get_qnode(self):
        if getattr(self, "_qnode", None) is None:
            n_qubits = len(self.feature_types)
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def qnode(x, w):
                return circuit_reupload(x, w, n_qubits)

            self._qnode = qnode
        return self._qnode

    def _z(self, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Mean <Z_i> over qubits, shape (B,)."""
        out = self._get_qnode()(X, w)
        return torch.stack(list(out)).mean(dim=0)

    def _logits(self, X, w, ac):
        return ac[0] * self._z(X, w) + ac[1]

    def _loss(self, X, y, w, ac):
        out = self._logits(X, w, ac)
        if self.task == "probability":
            return torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        return torch.mean((out - y) ** 2)

    # --- scikit-learn API ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, n_qubits = X.shape
        if n_qubits != len(self.feature_types):
            raise ValueError(f"X has {n_qubits} columns but {len(self.feature_types)} feature types")
        self._qnode = None
        gen = torch.Generator().manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Target handling: standardise for regression; for probability start at the base rate.
        if self.task == "probability":
            self.y_mean_, self.y_std_ = 0.0, 1.0
            rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
            c0 = np.log(rate / (1 - rate))
        else:
            self.y_mean_ = float(y.mean())
            self.y_std_ = float(y.std()) or 1.0
            c0 = 0.0
        y_fit = (y - self.y_mean_) / self.y_std_

        # Hold out a validation set for early stopping (skipped for tiny training sets).
        n_val = round(self.val_frac * n) if n >= 20 else 0
        order = rng.permutation(n)
        val_idx, tr_idx = order[:n_val], order[n_val:]
        Xt = torch.as_tensor(X[tr_idx], dtype=DTYPE)
        yt = torch.as_tensor(y_fit[tr_idx], dtype=DTYPE)
        Xv = torch.as_tensor(X[val_idx], dtype=DTYPE)
        yv = torch.as_tensor(y_fit[val_idx], dtype=DTYPE)

        w = torch.nn.Parameter(torch.empty(weight_shape(n_qubits, self.n_layers), dtype=DTYPE)
                               .uniform_(0, 2 * np.pi, generator=gen))
        ac = torch.nn.Parameter(torch.tensor([1.0, c0], dtype=DTYPE))  # output scale, offset
        opt = torch.optim.Adam([w, ac], lr=self.lr)

        self.history_ = {"train_loss": [], "val_loss": []}
        best_loss, best_state, bad_epochs = np.inf, None, 0
        for epoch in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=gen)
            total = 0.0
            for start in range(0, len(Xt), self.batch_size):
                idx = perm[start: start + self.batch_size]
                loss = self._loss(Xt[idx], yt[idx], w, ac)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.history_["train_loss"].append(total / len(Xt))

            if n_val:
                with torch.no_grad():
                    v = self._loss(Xv, yv, w, ac).item()
                self.history_["val_loss"].append(v)
            else:
                v = self.history_["train_loss"][-1]
            if v < best_loss - 1e-6:
                best_loss, bad_epochs = v, 0
                best_state = (w.detach().clone(), ac.detach().clone(), epoch)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.weights_, self.ac_, self.best_epoch_ = best_state
        self.n_epochs_run_ = epoch + 1
        return self

    def predict(self, X, batch_size: int = 1024) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        outs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[start: start + batch_size], dtype=DTYPE)
                outs.append(self._logits(xb, self.weights_, self.ac_))
        out = torch.cat(outs).numpy()
        if self.task == "probability":
            return 1.0 / (1.0 + np.exp(-out))
        return out * self.y_std_ + self.y_mean_

    def n_parameters(self) -> int:
        n_qubits = len(self.feature_types)
        return int(np.prod(weight_shape(n_qubits, self.n_layers))) + 2  # + output scale, offset

    def __getstate__(self):  # QNodes don't pickle; rebuild lazily after unpickling
        state = self.__dict__.copy()
        state["_qnode"] = None
        return state


def vqc(feature_types: Types, task: str, **hp):
    """Factory used by the runner: angle encoding + re-uploading regressor."""
    return make_pipeline(AngleScaler(feature_types), ReuploadingRegressor(feature_types, task, **hp))


SEARCH_SPACES = {"vqc": {"n_layers": ("int", 1, 4), "lr": ("log", 1e-3, 1e-1)}}
