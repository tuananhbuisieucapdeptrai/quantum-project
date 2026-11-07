"""Fourier analysis of the tides benchmark: is the VQC limited by its frequencies?

    python scripts/fourier_tides.py              # ceilings + figure (about 1 min)
    python scripts/fourier_tides.py --probe-vqc  # also retrain VQCs (about 15 min, 6 processes)

Outputs (results/fourier/):
  tides_ceilings.csv     best least-squares test R^2 per frequency set, fitted on all 17,520
                         training hours (and on n_train=1000, 5 seeds, for the small sets)
  tides_vqc_probe.csv    (--probe-vqc) VQC depth / training-budget / teacher experiments
  tides_ceiling_vs_models.png
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from qmlbench.data import get_dataset, load_config
from qmlbench.data.io import PROJECT_ROOT
from qmlbench.fourier import (
    data_spectrum,
    fourier_design,
    frequency_set,
    model_spectrum,
    power_by_order,
    spectrum_ceiling,
)
from qmlbench.splits import cap_test, make_split, subsample

OUT = PROJECT_ROOT / "results" / "fourier"
D = 5  # five constituent phases
SETS = {
    "additive, |w|<=1": {"l_inf": 1, "additive": True}, # what ridge on sin/cos can express
    "full, |w_j|<=1": {"l_inf": 1},          # 1-layer VQC and the angle kernel
    "full, |w_j|<=2": {"l_inf": 2},          # 2-layer VQC
}
RIDGE = 1e-6  # tiny: tidal phases lie on one time line, so many frequency vectors are collinear


def load():
    cfg = load_config("tides")
    ds = get_dataset("tides", cfg)
    pool, test = make_split(ds, cfg["split"])
    return cfg, ds, pool, cap_test(test, 2000)


def ceilings(ds, pool, test, n_train, seeds) -> pd.DataFrame:
    rows = []
    for name, kw in SETS.items():
        W = frequency_set(D, **kw)
        r = spectrum_ceiling(ds.X[pool], ds.y[pool], ds.X[test], ds.y[test], W, RIDGE)
        rows.append({"set": name, "n_train": len(pool), "seed": -1, **r})
        for n in n_train:
            for s in seeds:
                tr = subsample(pool, n, s)
                if len(W) * 2 + 1 > n:  # far more coefficients than points: not meaningful
                    continue
                r = spectrum_ceiling(ds.X[tr], ds.y[tr], ds.X[test], ds.y[test], W, ridge=1e-3)
                rows.append({"set": name, "n_train": n, "seed": s, **r})
    return pd.DataFrame(rows)


# --- VQC probes ----------------------------------------------------------------------------

PROBES = [  # (n_layers, epochs, patience, target)
    (1, 100, 15, "data"), (1, 600, 60, "data"), (2, 600, 60, "data"), (3, 600, 60, "data"),
    (1, 600, 60, "teacher"), (2, 600, 60, "teacher"),
]


def _probe(args) -> dict:
    import torch
    torch.set_num_threads(1)
    from qmlbench.models.quantum_vqc import vqc
    L, epochs, patience, target = args
    _, ds, pool, test = load()
    X, y = ds.X, ds.y.copy()
    if target == "teacher":  # noise-free target inside the 1-layer spectrum
        W1 = frequency_set(D, 1)
        y = fourier_design(X, W1) @ data_spectrum(X[pool], y[pool], W1, RIDGE)["coef"]
    tr = subsample(pool, 1000, 0)
    m = vqc(ds.feature_types, "regression", n_layers=L, lr=0.046, epochs=epochs,
            patience=patience, seed=0).fit(X[tr], y[tr])
    r2 = lambda a, b: float(1 - np.mean((a - b) ** 2) / np.var(a))
    spec = model_spectrum(m.predict, d=D, k_max=L + 1)
    power = power_by_order(spec["freqs"], spec["amplitude"])
    est = m[-1]
    return {"n_layers": L, "epochs": epochs, "target": target, "epochs_run": est.n_epochs_run_,
            "train_r2": r2(y[tr], m.predict(X[tr])), "test_r2": r2(y[test], m.predict(X[test])),
            "n_parameters": est.n_parameters(),
            # share of the model's variance at each frequency order; order L+1 is a band-limit
            # check and must be ~0
            **{f"power_share_order{k}": v / sum(power.values()) for k, v in power.items()}}


# --- figure --------------------------------------------------------------------------------

def plot(ceil: pd.DataFrame, path, runs: pd.DataFrame | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if runs is None:
        runs = pd.read_parquet(PROJECT_ROOT / "results" / "all_runs.parquet")
    runs = runs[runs.dataset == "tides"]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = [("gp", "GP (best classical)", "#2a78d6", 6), ("qkernel", "Quantum kernel", "#eb6834", -11),
              ("vqc", "VQC", "#1baf7a", -3)]  # last: vertical label offset, avoids collisions
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    full = ceil[ceil.seed < 0].set_index("set").test_r2
    for key, label, ls, x in [("full, |w_j|<=1", "1-layer ceiling", "--", 130),
                              ("additive, |w|<=1", "no-interaction ceiling", ":", 380)]:
        ax.axhline(full[key], color=muted, lw=1.2, ls=ls, zorder=1)
        ax.annotate(f"{label} {full[key]:.3f}", (x, full[key]), xytext=(0, 3),
                    textcoords="offset points", color=muted, fontsize=8, ha="left")
    for model, label, color, dy in series:
        g = runs[runs.model == model].groupby("n_train").r2
        m, s = g.mean(), g.std()
        ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.12, lw=0)
        ax.plot(m.index, m.values, color=color, lw=2, marker="o", ms=4.5, label=label, zorder=3)
        if m.min() < 0.70:  # keep the scale readable; say what is cut off
            ax.annotate(f"{label} at n={m.idxmin()}: {m.min():.2f} (off scale)",
                        (m.idxmin(), 0.70), xytext=(4, 4), textcoords="offset points",
                        color=muted, fontsize=7.5)
        ax.annotate(label, (m.index[-1], m.values[-1]), xytext=(8, dy),
                    textcoords="offset points", color=ink, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 250, 500, 1000], ["50", "100", "250", "500", "1000"])
    ax.set_ylim(0.70, 0.95)
    ax.set_xlim(45, 2600)  # room for end labels
    ax.set_xlabel("training points", color=muted)
    ax.set_ylabel("test $R^2$ (2023, mean ± 1 sd over 5 seeds)", color=muted)
    ax.set_title("Tides: the VQC sits below what its own frequencies allow",
                 loc="left", color=ink, fontsize=11)
    ax.grid(axis="y", color=grid, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=ink,
              bbox_to_anchor=(1.0, 0.04))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-vqc", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, ds, pool, test = load()

    ceil = ceilings(ds, pool, test, [1000], range(5))
    ceil.to_csv(OUT / "tides_ceilings.csv", index=False)
    print(ceil.groupby(["set", "n_train"]).test_r2.agg(["mean", "std"]).round(4).to_string())

    # Note: we do not split the *data's* variance by frequency order. Tidal speeds are
    # integer combinations of a few astronomical ones (K1 + O1 = M2 exactly), so along the
    # time line many frequency vectors coincide and their individual coefficients are not
    # identifiable. Nested ceilings are: they only compare spans.
    plot(ceil, OUT / "tides_ceiling_vs_models.png")

    if args.probe_vqc:
        with ProcessPoolExecutor(args.jobs) as ex:
            probe = pd.DataFrame(list(ex.map(_probe, PROBES)))
        probe.to_csv(OUT / "tides_vqc_probe.csv", index=False)
        print(probe.round(4).to_string())


if __name__ == "__main__":
    main()
