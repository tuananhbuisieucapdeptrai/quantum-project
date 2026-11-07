"""Benchmark runner.

    python -m qmlbench.run --dry-run                           # grid size, what's already done
    python -m qmlbench.run --datasets tides --models ridge vqc domain --n-train 100 --seeds 0
    python -m qmlbench.run --jobs 6                             # full grid from configs/benchmark.yaml
    python -m qmlbench.run --collect                            # merge finished runs into one table

One run = (dataset, model, n_train, seed):
  1. fixed split from configs/<dataset>.yaml (temporal / group / random), test set capped;
  2. subsample n_train training points with ``seed``;
  3. tune with the shared budget (``qmlbench.tuning``: inner CV on the training subsample);
  4. refit the best parameters on the whole subsample, predict the test set, score it;
  5. write results/raw/<dataset>/<model>__n<n>__s<seed>.json.

One file per run makes long benchmarks resumable: finished runs are skipped on the next call
(``--force`` re-runs them), and parallel workers never write to the same file.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .data import get_dataset, load_config
from .data.base import CONFIG_DIR, RegressionDataset
from .data.io import PROJECT_ROOT
from .metrics import evaluate
from .models import classical, quantum_kernel, quantum_vqc
from .models import domain as domain_models
from .models.domain import DOMAIN_BASELINES
from .splits import cap_test, make_split, subsample
from .tuning import tune

# Models built from (feature_types, task) and fitted on X.
FACTORIES = {**classical.MODELS, "vqc": quantum_vqc.vqc, "qkernel": quantum_kernel.qkernel}
FAMILY = {**{m: "classical" for m in classical.MODELS}, "vqc": "quantum", "qkernel": "quantum"}
SPACES = {**classical.SEARCH_SPACES, **quantum_vqc.SEARCH_SPACES,
          **quantum_kernel.SEARCH_SPACES}
# Rough relative cost, used only to schedule expensive runs first (better load balancing).
COST = {"vqc": 100, "gp": 5, "mlp": 5, "gbm": 3}


def load_benchmark_config() -> dict:
    with open(CONFIG_DIR / "benchmark.yaml") as fh:
        return yaml.safe_load(fh)


def expand_models(models: list[str], dataset: str) -> list[str]:
    """``domain`` -> every domain baseline for this dataset, as ``domain:<name>``."""
    out = []
    for m in models:
        if m == "domain":
            out += [f"domain:{k}" for k in DOMAIN_BASELINES[dataset]]
        else:
            out.append(m)
    return out


def search_space(dataset: str, model_name: str) -> dict:
    if model_name.startswith("domain:"):
        return domain_models.SEARCH_SPACES[dataset][model_name.split(":", 1)[1]]
    return SPACES[model_name]


def family_of(dataset: str, model_name: str) -> str:
    if model_name.startswith("domain:"):
        cls = DOMAIN_BASELINES[dataset][model_name.split(":", 1)[1]]
        return "reference" if cls.reference else "domain"
    return FAMILY[model_name]


def make_fit_predict(ds: RegressionDataset, model_name: str, seed: int):
    """``fit_predict(params, train_idx, eval_idx) -> (pred, fitted_estimator)`` for any model."""
    def fit_predict(params: dict, tr: np.ndarray, ev: np.ndarray):
        if model_name.startswith("domain:"):
            cls = DOMAIN_BASELINES[ds.name][model_name.split(":", 1)[1]]
            model = cls(ds, seed=seed, **params).fit(tr)
            return model.predict(ev), model
        model = FACTORIES[model_name](ds.feature_types, ds.task, seed=seed, **params)
        model.fit(ds.X[tr], ds.y[tr])
        est = model[-1] if hasattr(model, "steps") else model
        return model.predict(ds.X[ev]), est
    return fit_predict


def run_one(ds: RegressionDataset, dcfg: dict, model_name: str, n_train: int, seed: int,
            test_cap: int, trials: int = 0, cv_folds: int = 3, hp: dict | None = None) -> dict:
    """One benchmark run. ``hp`` fixes the parameters (no tuning); else tune with ``trials``."""
    pool, test = make_split(ds, dcfg["split"])
    test = cap_test(test, test_cap)
    tr = subsample(pool, n_train, seed)
    fit_predict = make_fit_predict(ds, model_name, seed)

    if hp is None:
        tuned = tune(lambda p, a, b: fit_predict(p, a, b)[0], search_space(ds.name, model_name),
                     ds, tr, trials=trials, cv_folds=cv_folds, seed=seed)
    else:
        tuned = {"params": hp, "cv_loss": math.nan, "n_trials": 0, "tune_seconds": 0.0}

    t0 = time.perf_counter()
    pred, est = fit_predict(tuned["params"], tr, test)
    fit_s = time.perf_counter() - t0
    n_params = est.n_parameters() if hasattr(est, "n_parameters") else None
    return {"dataset": ds.name, "model": model_name, "family": family_of(ds.name, model_name),
            "n_train": len(tr), "seed": seed, "n_test": len(test),
            "params": json.dumps(tuned["params"], default=float), "cv_loss": tuned["cv_loss"],
            "n_trials": tuned["n_trials"], "tune_seconds": tuned["tune_seconds"],
            "fit_seconds": fit_s, "n_parameters": n_params,
            "n_epochs_run": getattr(est, "n_epochs_run_", None),
            **evaluate(ds.name, ds.task, ds.y[test], pred)}


# --- grid execution -----------------------------------------------------------------------

def run_path(results_dir: Path, dataset: str, model: str, n: int, seed: int) -> Path:
    return Path(results_dir) / dataset / f"{model.replace(':', '-')}__n{n}__s{seed}.json"


@functools.cache
def _cached_dataset(name: str) -> tuple[RegressionDataset, dict]:
    cfg = load_config(name)
    return get_dataset(name, cfg), cfg


def _init_worker() -> None:
    # One thread per process: the VQC is no faster with more threads, so parallelism comes
    # from running several fits at once.
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    np.seterr(all="ignore")
    import warnings
    warnings.filterwarnings("ignore")


def _run_task(task: dict) -> tuple[dict, str | None]:
    try:
        ds, dcfg = _cached_dataset(task["dataset"])
        row = run_one(ds, dcfg, task["model"], task["n_train"], task["seed"], task["test_cap"],
                      trials=task["trials"], cv_folds=task["cv_folds"])
        path = Path(task["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(row, default=float, indent=1))
        tmp.replace(path)
        return task, None
    except Exception:  # noqa: BLE001 -- record the failure, keep the grid running
        return task, traceback.format_exc()


def collect(results_dir: Path) -> pd.DataFrame:
    """All finished runs as one DataFrame; also written to <results_dir>/../all_runs.parquet."""
    files = sorted(Path(results_dir).glob("*/*.json"))
    if not files:
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(f.read_text()) for f in files])
    out = Path(results_dir).parent / "all_runs.parquet"
    df.to_parquet(out, index=False)
    return df


def main(argv: list[str] | None = None) -> None:
    bcfg = load_benchmark_config()
    tcfg = bcfg["tuning"]
    default_models = [*bcfg["models"]["classical"], *bcfg["models"]["quantum"],
                      *bcfg["models"]["domain"]]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=bcfg["datasets"])
    ap.add_argument("--models", nargs="+", default=default_models,
                    help="model names, or 'domain' for all domain baselines of each dataset")
    ap.add_argument("--n-train", nargs="+", type=int, default=bcfg["n_train"])
    ap.add_argument("--seeds", nargs="+", type=int, default=bcfg["seeds"])
    ap.add_argument("--trials", type=int, default=tcfg["trials"], help="tuning trials per run")
    ap.add_argument("--cv-folds", type=int, default=tcfg["cv_folds"])
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--force", action="store_true", help="re-run finished runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", action="store_true", help="only merge finished runs")
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / bcfg["results_dir"],
                    help="where per-run JSON files go (default from benchmark.yaml)")
    args = ap.parse_args(argv)
    results_dir = args.results_dir

    if args.collect:
        df = collect(results_dir)
        print(f"{len(df)} runs -> {results_dir.parent / 'all_runs.parquet'}")
        return

    unknown = [m for m in args.models if m not in FACTORIES and m != "domain"
               and not m.startswith("domain:")]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")

    tasks = []
    for name in args.datasets:
        for m, n, s in itertools.product(expand_models(args.models, name), args.n_train, args.seeds):
            path = run_path(results_dir, name, m, n, s)
            if path.exists() and not args.force:
                continue
            tasks.append({"dataset": name, "model": m, "n_train": n, "seed": s,
                          "test_cap": bcfg["test_cap"], "trials": args.trials,
                          "cv_folds": args.cv_folds, "path": str(path)})
    tasks.sort(key=lambda t: -COST.get(t["model"], 1) * t["n_train"])
    total = sum(len(expand_models(args.models, d)) for d in args.datasets) \
        * len(args.n_train) * len(args.seeds)
    print(f"grid: {total} runs, {total - len(tasks)} already done, {len(tasks)} to run "
          f"({args.trials} trials x {args.cv_folds} folds each, {args.jobs} jobs)")
    if args.dry_run or not tasks:
        return

    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="runs"):
            task, err = fut.result()
            if err:
                failures.append((task, err))
                tqdm.write(f"FAILED {task['dataset']} {task['model']} n={task['n_train']} "
                           f"s={task['seed']}:\n{err.splitlines()[-1]}")
    df = collect(results_dir)
    print(f"{len(df)} finished runs collected; {len(failures)} failed this time")


if __name__ == "__main__":
    main()
