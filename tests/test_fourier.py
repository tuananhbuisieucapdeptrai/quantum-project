import numpy as np
import torch

from qmlbench.fourier import (
    data_spectrum,
    frequency_set,
    model_spectrum,
    power_by_order,
    spectrum_ceiling,
    top_frequencies,
)
from qmlbench.models.quantum_vqc import ReuploadingRegressor


def test_frequency_set_sizes():
    assert len(frequency_set(2, 1)) == (3 ** 2 - 1) // 2
    assert len(frequency_set(5, 1)) == (3 ** 5 - 1) // 2
    assert len(frequency_set(3, 2, additive=True)) == 6
    assert np.abs(frequency_set(3, 2, l1=2)).sum(axis=1).max() == 2


def _target(X):
    return 0.7 * np.cos(X[:, 0] - 0.3) + 0.2 * np.sin(2 * X[:, 0] + X[:, 1]) + 0.1


def test_data_spectrum_recovers_known_series():
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, size=(2000, 2))
    spec = data_spectrum(X, _target(X), frequency_set(2, 2))
    top = dict(top_frequencies(spec, ["a", "b"], k=2))
    assert np.isclose(top["a"], 0.7, atol=1e-8) and np.isclose(top["2*a+b"], 0.2, atol=1e-8)
    assert np.isclose(spec["mean"], 0.1)
    assert np.isclose(sum(power_by_order(spec["freqs"], spec["amplitude"]).values()),
                      0.7 ** 2 / 2 + 0.2 ** 2 / 2)


def test_ceiling_shows_missing_frequency():
    rng = np.random.default_rng(1)
    X = rng.uniform(-np.pi, np.pi, size=(1500, 2))
    y = _target(X)
    low = spectrum_ceiling(X[:1000], y[:1000], X[1000:], y[1000:], frequency_set(2, 1))
    high = spectrum_ceiling(X[:1000], y[:1000], X[1000:], y[1000:], frequency_set(2, 2))
    assert high["test_r2"] > 0.999 and low["test_r2"] < high["test_r2"] - 0.05


def test_model_spectrum_matches_data_spectrum():
    spec = model_spectrum(_target, d=2, k_max=2)
    amp = dict(top_frequencies(spec, ["a", "b"], k=2))
    assert np.isclose(amp["a"], 0.7) and np.isclose(amp["2*a+b"], 0.2)
    i = [tuple(w) for w in spec["freqs"]].index((1, 0))
    # 0.7 cos(x - 0.3) = a cos x + b sin x with phase arctan2(b, a) = 0.3
    assert np.isclose(spec["phase"][i], 0.3)


def test_vqc_spectrum_is_band_limited():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, size=(60, 2))
    m = ReuploadingRegressor(["angle", "angle"], n_layers=1, epochs=3, seed=0).fit(X, _target(X))
    spec = model_spectrum(m.predict, d=2, k_max=2)
    p = power_by_order(spec["freqs"], spec["amplitude"])
    assert p[2] < 1e-10 * max(p[1], 1e-12) + 1e-14  # 1 layer -> nothing at |omega_j| = 2
