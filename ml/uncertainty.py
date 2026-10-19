"""Approximate prediction uncertainty.

Two model-specific estimates are supported; both are APPROXIMATE and must not
be presented as calibrated confidence intervals unless checked:

* Gaussian Process: the predictive standard deviation of the posterior.
* Random Forest / Extra Trees: the standard deviation of the individual tree
  predictions (ensemble spread). This reflects model disagreement, not noise,
  and is known to be only a rough indicator.

:func:`interval_coverage` provides the empirical check used throughout the
application: the fraction of held-out test points inside mean +/- z*sigma.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline

UNCERTAINTY_DISCLAIMER = (
    "Uncertainty values are APPROXIMATE model-based indicators (GP predictive std or tree-ensemble "
    "spread). They are not calibrated confidence intervals and do not include errors of the "
    "synthetic data-generating formula or of real EM physics."
)


def uncertainty_method(pipeline: Pipeline) -> str | None:
    model = pipeline.named_steps["model"]
    if isinstance(model, GaussianProcessRegressor):
        return "gp_predictive_std"
    if isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
        return "tree_ensemble_spread"
    return None


def predict_with_uncertainty(pipeline: Pipeline, X) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """Return (mean, std or None, method name)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    method = uncertainty_method(pipeline)
    model = pipeline.named_steps["model"]
    if method == "gp_predictive_std":
        Xt = pipeline[:-1].transform(X)
        mean, std = model.predict(Xt, return_std=True)
        return np.asarray(mean), np.asarray(std), method
    if method == "tree_ensemble_spread":
        Xt = np.asarray(pipeline[:-1].transform(X), dtype=np.float32)
        per_tree = np.stack([tree.predict(Xt) for tree in model.estimators_])
        return model.predict(Xt), per_tree.std(axis=0), method
    return pipeline.predict(X), None, None


def interval_coverage(y_true, mean, std, z: float = 1.96) -> float | None:
    if std is None:
        return None
    y_true, mean, std = (np.asarray(a, dtype=float) for a in (y_true, mean, std))
    inside = np.abs(y_true - mean) <= z * std
    return float(inside.mean())
