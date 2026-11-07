"""Classical baselines. Inputs go through ``PeriodicExpander`` (angles -> sin/cos, reals
standardised) so classical models know about periodicity too.

Random Fourier Features (``rff``) is the key comparison: Landman et al. (2022) showed that
many variational quantum models can be approximated by linear models on random Fourier
features, so if ``rff`` matches the quantum models, the quantum advantage claim fails.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline

from ..encoding import PeriodicExpander
from .base import ProbaAsPredict, Types


def ridge(feature_types: Types, task: str, alpha: float = 1.0, **_):
    if task == "probability":
        return make_pipeline(PeriodicExpander(feature_types),
                             ProbaAsPredict(LogisticRegression(C=1 / alpha, max_iter=2000)))
    return make_pipeline(PeriodicExpander(feature_types), Ridge(alpha=alpha))


def rff(feature_types: Types, task: str, n_components: int = 200, gamma: float = 1.0,
        alpha: float = 1.0, seed: int = 0, **_):
    head = (ProbaAsPredict(LogisticRegression(C=1 / alpha, max_iter=2000))
            if task == "probability" else Ridge(alpha=alpha))
    return make_pipeline(PeriodicExpander(feature_types),
                         RBFSampler(gamma=gamma, n_components=n_components, random_state=seed),
                         head)


def mlp(feature_types: Types, task: str, hidden: int = 32, layers: int = 2,
        alpha: float = 1e-4, lr: float = 1e-3, seed: int = 0, **_):
    kw = {"hidden_layer_sizes": (hidden,) * layers, "alpha": alpha, "learning_rate_init": lr,
          "max_iter": 2000, "early_stopping": True, "random_state": seed}
    est = ProbaAsPredict(MLPClassifier(**kw)) if task == "probability" else MLPRegressor(**kw)
    return make_pipeline(PeriodicExpander(feature_types), est)


def gbm(feature_types: Types, task: str, max_depth: int = 4, lr: float = 0.05,
        max_iter: int = 300, seed: int = 0, **_):
    kw = {"max_depth": max_depth, "learning_rate": lr, "max_iter": max_iter, "random_state": seed}
    est = (ProbaAsPredict(HistGradientBoostingClassifier(**kw)) if task == "probability"
           else HistGradientBoostingRegressor(**kw))
    return make_pipeline(PeriodicExpander(feature_types), est)


def gp(feature_types: Types, task: str, **_):
    kernel = ConstantKernel() * RBF() + WhiteKernel()
    return make_pipeline(PeriodicExpander(feature_types),
                         GaussianProcessRegressor(kernel=kernel, normalize_y=True))


MODELS = {"ridge": ridge, "rff": rff, "mlp": mlp, "gbm": gbm, "gp": gp}

SEARCH_SPACES = {
    "ridge": {"alpha": ("log", 1e-4, 1e2)},
    "rff": {"n_components": ("int", 50, 500), "gamma": ("log", 1e-2, 1e1),
            "alpha": ("log", 1e-4, 1e2)},
    "mlp": {"hidden": ("int", 8, 128), "layers": ("int", 1, 3), "alpha": ("log", 1e-6, 1e-1),
            "lr": ("log", 1e-4, 1e-2)},
    "gbm": {"max_depth": ("int", 2, 8), "lr": ("log", 1e-2, 3e-1), "max_iter": ("int", 50, 500)},
    "gp": {},
}
