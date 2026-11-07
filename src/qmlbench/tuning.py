"""Equal-budget hyperparameter search.

Every tunable model gets the same number of Optuna trials (TPE sampler, fixed seed). Each trial
is scored by inner K-fold cross-validation on the *training subsample only*; the test set is
never touched during tuning. The best parameters are then refitted on the whole training
subsample by the runner.

Inner folds respect each domain's structure, as the outer split does:
  temporal -> blocked folds: training points sorted by time and cut into contiguous blocks,
              so a fold is never validated on hours that sit between its own training hours;
  group    -> GroupKFold on match ids (a match never sits on both sides);
  random   -> shuffled KFold.

Search-space format (each model module's ``SEARCH_SPACES``):
  ("log", lo, hi) log-uniform float   ("float", lo, hi) uniform float
  ("int", lo, hi) integer              ("cat", [choices]) categorical
A model with an empty space (e.g. ``gp``, which fits its own kernel hyperparameters by
marginal likelihood, or UTide) is not tuned and is fitted once.

The tuner is model-agnostic: it only needs ``fit_predict(params, train_idx, eval_idx)``,
so the same code tunes quantum, classical and domain models.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold, KFold

from .data.base import RegressionDataset
from .metrics import brier, rmse

optuna.logging.set_verbosity(optuna.logging.WARNING)

FitPredict = Callable[[dict, np.ndarray, np.ndarray], np.ndarray]


def inner_folds(ds: RegressionDataset, idx: np.ndarray, k: int,
                seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K (train, validation) index pairs inside ``idx``, matching the dataset's split kind."""
    idx = np.asarray(idx)
    if ds.split_kind == "temporal":
        ordered = idx[np.argsort(ds.time[idx], kind="stable")]
        blocks = np.array_split(ordered, k)
        return [(np.concatenate(blocks[:i] + blocks[i + 1:]), blocks[i]) for i in range(k)]
    if ds.split_kind == "group":
        groups = ds.groups[idx]
        k = min(k, len(np.unique(groups)))
        return [(idx[a], idx[b]) for a, b in GroupKFold(n_splits=k).split(idx, groups=groups)]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(idx[a], idx[b]) for a, b in kf.split(idx)]


def loss_fn(task: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Tuning objective (lower is better): RMSE for regression, Brier score for probabilities."""
    return brier if task == "probability" else rmse


def suggest(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "cat":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"unknown search-space kind {kind!r} for {name}")
    return params


def cv_loss(fit_predict: FitPredict, params: dict, ds: RegressionDataset,
            folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean validation loss over the folds; inf if the model fails to fit."""
    loss, losses = loss_fn(ds.task), []
    for tr, va in folds:
        try:
            pred = fit_predict(params, tr, va)
        except Exception:  # noqa: BLE001 -- a bad hyperparameter combination shouldn't kill the search
            return math.inf
        if not np.all(np.isfinite(pred)):
            return math.inf
        losses.append(loss(ds.y[va], pred))
    return float(np.mean(losses))


def tune(fit_predict: FitPredict, space: dict, ds: RegressionDataset, train_idx: np.ndarray,
         trials: int, cv_folds: int, seed: int) -> dict:
    """Search ``space`` with ``trials`` Optuna trials of inner CV on ``train_idx``.

    Returns {"params", "cv_loss", "n_trials", "tune_seconds"}; for an empty space, params={}
    and no CV is run.
    """
    t0 = time.perf_counter()
    if not space or trials <= 0:
        return {"params": {}, "cv_loss": math.nan, "n_trials": 0, "tune_seconds": 0.0}
    folds = inner_folds(ds, train_idx, cv_folds, seed)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda tr: cv_loss(fit_predict, suggest(tr, space), ds, folds),
                   n_trials=trials)
    best = study.best_trial
    return {"params": dict(best.params), "cv_loss": float(best.value), "n_trials": len(study.trials),
            "tune_seconds": time.perf_counter() - t0}
