"""Domain-expert baselines: what a practitioner in each field would actually use.

Unlike the quantum and classical models, these need information that isn't in ``X``
(timestamps for tidal analysis, reference columns, which column is wind speed). So they are
built from the whole ``RegressionDataset`` and fitted on **row indices**:

    model = DOMAIN_BASELINES["tides"]["harmonic"](ds)
    model.fit(train_idx)
    pred = model.predict(test_idx)

The runner uses exactly the same train/test indices as for every other model.

Two kinds of baseline:
* **trained** -- fitted on the same n_train points as every other model (fair comparison);
* **reference** (``reference = True``) -- a published, externally fitted prediction that we only
  look up: NOAA's official tide forecast and StatsBomb's own xG. They were fitted on far more
  data than any of our models, so they mark the practical ceiling, not a fair competitor.

tides  : ``harmonic``     UTide harmonic analysis (8 major constituents + nodal corrections)
         ``noaa``         NOAA official harmonic prediction (reference)
wind   : ``power_curve``  binned power curve, IEC 61400-12-1 style (ignores direction)
soccer : ``xg_logistic``  classic xG: logistic regression on distance and goal angle
         ``statsbomb_xg`` StatsBomb's xG model (reference)
photoz : ``knn``          k-nearest-neighbour empirical photo-z on colours + magnitude
"""

from __future__ import annotations

import numpy as np
import utide
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.base import RegressionDataset


class DomainBaseline:
    """Base class: ``fit(train_idx)`` / ``predict(idx)`` on rows of the dataset."""

    reference = False

    def __init__(self, ds: RegressionDataset, seed: int = 0, **hp):
        self.ds = ds
        self.seed = seed
        self.hp = hp

    def col(self, name: str) -> int:
        return self.ds.feature_names.index(name)

    def fit(self, idx: np.ndarray) -> DomainBaseline:
        return self

    def predict(self, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n_parameters(self) -> int | None:
        return None


# --- tides ---------------------------------------------------------------------------------

class HarmonicTides(DomainBaseline):
    """Classical tidal harmonic analysis with UTide (ordinary least squares)."""

    def fit(self, idx):
        meta = self.ds.meta
        self.constit_ = list(meta["harmonic_constituents"])
        self.coef_ = utide.solve(
            self.ds.time[idx], self.ds.y[idx], lat=meta["latitude"], constit=self.constit_,
            method="ols", conf_int="none", trend=False, nodal=True, verbose=False)
        return self

    def predict(self, idx):
        return utide.reconstruct(self.ds.time[idx], self.coef_, verbose=False).h

    def n_parameters(self):
        return 2 * len(self.constit_) + 1  # amplitude + phase per constituent, + mean


class NOAAReference(DomainBaseline):
    reference = True

    def predict(self, idx):
        return self.ds.extras["noaa_prediction"].to_numpy()[idx]


# --- wind ----------------------------------------------------------------------------------

class PowerCurve(DomainBaseline):
    """Mean power per wind-speed bin, linearly interpolated between bin centres."""

    def fit(self, idx):
        ws = self.ds.X[idx, self.col("wind_speed")]
        y = self.ds.y[idx]
        width = float(self.hp.get("bin_width", 0.5))
        bins = np.floor(ws / width).astype(int)
        centres, means = [], []
        for b in np.unique(bins):
            m = bins == b
            centres.append(ws[m].mean())
            means.append(y[m].mean())
        self.centres_, self.means_ = np.array(centres), np.array(means)
        return self

    def predict(self, idx):
        ws = self.ds.X[idx, self.col("wind_speed")]
        return np.interp(ws, self.centres_, self.means_)  # flat beyond the observed range

    def n_parameters(self):
        return len(self.centres_)


# --- soccer --------------------------------------------------------------------------------

class LogisticXG(DomainBaseline):
    """The textbook xG model: P(goal) = sigmoid(a + b*distance + c*goal_angle)."""

    def _X(self, idx):
        return self.ds.X[idx][:, [self.col("distance"), self.col("goal_angle")]]

    def fit(self, idx):
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=2000))
        self.model_.fit(self._X(idx), self.ds.y[idx].astype(int))
        return self

    def predict(self, idx):
        return self.model_.predict_proba(self._X(idx))[:, 1]

    def n_parameters(self):
        return 3


class StatsBombReference(DomainBaseline):
    reference = True

    def predict(self, idx):
        return self.ds.extras["statsbomb_xg"].to_numpy()[idx]


# --- astronomy -----------------------------------------------------------------------------

class KNNPhotoZ(DomainBaseline):
    """Distance-weighted k-nearest neighbours on standardised colours + r magnitude."""

    def fit(self, idx):
        k = min(int(self.hp.get("k", 10)), len(idx))
        self.model_ = make_pipeline(StandardScaler(),
                                    KNeighborsRegressor(n_neighbors=k, weights="distance"))
        self.model_.fit(self.ds.X[idx], self.ds.y[idx])
        return self

    def predict(self, idx):
        return self.model_.predict(self.ds.X[idx])


DOMAIN_BASELINES: dict[str, dict[str, type[DomainBaseline]]] = {
    "tides": {"harmonic": HarmonicTides, "noaa": NOAAReference},
    "wind": {"power_curve": PowerCurve},
    "soccer": {"xg_logistic": LogisticXG, "statsbomb_xg": StatsBombReference},
    "photoz": {"knn": KNNPhotoZ},
}

# Hyperparameters tuned with the same budget as every other model (see qmlbench.tuning).
# UTide and the textbook xG model have nothing to tune; references are never fitted.
SEARCH_SPACES = {
    "tides": {"harmonic": {}, "noaa": {}},
    "wind": {"power_curve": {"bin_width": ("float", 0.25, 1.5)}},
    "soccer": {"xg_logistic": {}, "statsbomb_xg": {}},
    "photoz": {"knn": {"k": ("int", 3, 50)}},
}
