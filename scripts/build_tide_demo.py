"""Export the data behind the tides demo website (demo/).

    python scripts/build_tide_demo.py

Refits each model on the benchmark's n=1000, seed-0 training subsample with the
hyperparameters the tuner chose in that run (read from results/), predicts every hour of 2023,
and writes demo/data/tides.json:

* test hours, observed water level and each model's prediction (whole of 2023);
* a sample of training hours and levels, so the browser can train the angle quantum kernel
  itself (its kernel has the closed form prod_j cos^2((x_j - x'_j)/2), identical to the
  simulated circuit -- tested in tests/test_qkernel.py);
* benchmark scores and spectrum ceilings for the summary panels.

Model scores in the summary panel are the benchmark means over five seeds (results/raw).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qmlbench.data import get_dataset, load_config
from qmlbench.data.tides import CONSTITUENT_SPEEDS_DEG_PER_HOUR
from qmlbench.models.quantum_kernel import angle_kernel_closed_form
from qmlbench.run import make_fit_predict, run_path
from qmlbench.splits import make_split, subsample

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "demo" / "data" / "tides.json"
N, SEED, POOL_SAMPLE = 1000, 0, 3000
MODELS = {  # key in the JSON -> (model id in the benchmark, label)
    "noaa": ("domain:noaa", "NOAA official prediction"),
    "utide": ("domain:harmonic", "UTide harmonic analysis"),
    "gp": ("gp", "Gaussian process"),
    "qkernel": ("qkernel", "Quantum kernel"),
    "vqc": ("vqc", "Re-uploading circuit"),
}


def tuned_params(model: str) -> dict:
    path = run_path(RAW, "tides", model, N, SEED)
    return json.loads(json.loads(path.read_text())["params"])


def r2(y, p) -> float:
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def main() -> None:
    cfg = load_config("tides")
    ds = get_dataset("tides", cfg)
    pool, test = make_split(ds, cfg["split"])
    tr = subsample(pool, N, SEED)
    hours = ds.extras["hours_since_epoch"].to_numpy()
    y = ds.y

    preds, params = {}, {}
    for key, (model, _) in MODELS.items():
        p = tuned_params(model)
        t0 = time.perf_counter()
        preds[key], _ = make_fit_predict(ds, model, SEED)(p, tr, test)
        params[key] = p
        print(f"{key:8s} R2(2023) = {r2(y[test], preds[key]):.4f}  "
              f"({time.perf_counter() - t0:.0f}s)  {p}")

    # Benchmark scores (capped test set, 5 seeds) for the summary panel.
    runs = pd.DataFrame([json.loads(f.read_text()) for f in sorted((RAW / "tides").glob("*.json"))])
    curves = {key: runs[runs.model == model].groupby("n_train").r2.mean().round(4).to_dict()
              for key, (model, _) in MODELS.items() if key != "noaa"}
    ceil = pd.read_csv(ROOT / "results" / "fourier" / "tides_ceilings.csv")
    ceil = ceil[ceil.seed < 0].set_index("set").test_r2.round(4).to_dict()

    # Training sample for the in-browser quantum kernel (subsampled pool, deterministic).
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(pool, size=min(POOL_SAMPLE, len(pool)), replace=False))

    # Self-check: the JS kernel ridge uses this closed form; confirm it reproduces the model.
    qk = params["qkernel"]
    if qk.get("feature_map") == "angle":
        K = angle_kernel_closed_form(ds.X[tr], ds.X[tr])
        ym = y[tr].mean()
        coef = np.linalg.solve(K + qk["alpha"] * np.eye(len(tr)), y[tr] - ym)
        p_cf = angle_kernel_closed_form(ds.X[test], ds.X[tr]) @ coef + ym
        print(f"closed-form kernel vs simulated circuit: max |diff| = "
              f"{np.abs(p_cf - preds['qkernel']).max():.2e} m")

    data = {
        "station": {"id": cfg["station"], "name": cfg.get("station_name", "San Francisco"),
                    "latitude": cfg.get("latitude")},
        "epoch": "2000-01-01T00:00:00Z",
        "constituents": cfg["constituents"],
        "speeds_deg_per_hour": [CONSTITUENT_SPEEDS_DEG_PER_HOUR[c] for c in cfg["constituents"]],
        "units": "m (relative to mean sea level)",
        "test": {"hours": np.round(hours[test]).astype(int).tolist(),
                 "observed": np.round(y[test], 3).tolist(),
                 "predictions": {k: np.round(v, 3).tolist() for k, v in preds.items()},
                 "r2": {k: round(r2(y[test], v), 4) for k, v in preds.items()}},
        "train_pool": {"hours": np.round(hours[sample]).astype(int).tolist(),
                       "level": np.round(y[sample], 3).tolist()},
        "models": {k: lab for k, (_, lab) in MODELS.items()},
        "params": params,
        "benchmark": {"n_train": N, "seed": SEED, "curves": curves, "ceilings": ceil},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"-> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
