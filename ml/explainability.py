"""Explainability tools for the surrogate models.

* Impurity (MDI) importance - tree models only; biased toward high-cardinality
  features and computed on training data.
* Permutation importance - model-agnostic drop in test R^2 when one feature is
  shuffled (Breiman, 2001; Fisher et al., 2019).
* One-at-a-time sensitivity - sweep one parameter, hold others fixed.
* Partial dependence - average model response over the data distribution
  (Friedman, 2001).
* Design-space maps - mean and uncertainty over a 2-D grid, used to point out
  where new EM simulations / measurements would be most informative.

All of these describe what the MODEL has learned (associations in the data).
They do not establish physical causation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance

from ml.uncertainty import predict_with_uncertainty

CAUSATION_NOTE = (
    "These explanations describe associations captured by the trained model. A high importance "
    "means the model's predictions depend strongly on that parameter; it does not prove a physical "
    "cause-and-effect mechanism. For synthetic data they recover the assumed formula, not real physics."
)


def impurity_importance(pipeline, features: list[str]) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    return (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def permutation_importance_frame(pipeline, X, y, features: list[str], seed: int = 42,
                                 n_repeats: int = 10) -> pd.DataFrame:
    res = permutation_importance(pipeline, X, y, n_repeats=n_repeats, random_state=seed, scoring="r2")
    return (pd.DataFrame({"feature": features, "importance_mean": res.importances_mean,
                          "importance_std": res.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def sensitivity_sweep(pipeline, features: list[str], base_point: dict, feature: str,
                      lo: float, hi: float, n_points: int = 80) -> pd.DataFrame:
    grid = np.linspace(lo, hi, n_points)
    X = np.tile([float(base_point[f]) for f in features], (n_points, 1))
    X[:, features.index(feature)] = grid
    mean, std, _ = predict_with_uncertainty(pipeline, X)
    return pd.DataFrame({feature: grid, "prediction": mean, "std": std if std is not None else np.nan})


def local_sensitivities(pipeline, features: list[str], base_point: dict,
                        ranges: dict[str, tuple[float, float]], rel_step: float = 0.05) -> pd.DataFrame:
    """Finite-difference change in prediction for a +/- step of 5 % of each parameter's range."""
    rows = []
    x0 = np.array([[float(base_point[f]) for f in features]])
    for i, f in enumerate(features):
        lo, hi = ranges[f]
        h = rel_step * (hi - lo)
        xp, xm = x0.copy(), x0.copy()
        xp[0, i] = min(hi, x0[0, i] + h)
        xm[0, i] = max(lo, x0[0, i] - h)
        yp, ym = pipeline.predict(xp)[0], pipeline.predict(xm)[0]
        span = xp[0, i] - xm[0, i]
        rows.append({"feature": f, "delta_prediction": float(yp - ym), "per_unit": float((yp - ym) / span) if span else 0.0})
    return pd.DataFrame(rows)


def partial_dependence_frame(pipeline, X_background, features: list[str], feature: str,
                             grid_resolution: int = 50) -> pd.DataFrame:
    idx = features.index(feature)
    pd_res = partial_dependence(pipeline, np.asarray(X_background, dtype=float), [idx],
                                grid_resolution=grid_resolution, kind="average")
    grid = pd_res["grid_values"][0] if "grid_values" in pd_res else pd_res["values"][0]
    return pd.DataFrame({feature: grid, "partial_dependence": pd_res["average"][0]})


def design_space_grid(pipeline, features: list[str], fx: str, fy: str,
                      ranges: dict[str, tuple[float, float]], fixed: dict[str, float],
                      n: int = 60) -> dict:
    """Evaluate mean/std on an n x n grid over (fx, fy), other features fixed."""
    xs = np.linspace(*ranges[fx], n)
    ys = np.linspace(*ranges[fy], n)
    XX, YY = np.meshgrid(xs, ys)
    X = np.tile([float(fixed.get(f, 0.0)) for f in features], (XX.size, 1))
    X[:, features.index(fx)] = XX.ravel()
    X[:, features.index(fy)] = YY.ravel()
    mean, std, method = predict_with_uncertainty(pipeline, X)
    return {
        "x": xs, "y": ys, "fx": fx, "fy": fy,
        "mean": np.asarray(mean).reshape(XX.shape),
        "std": None if std is None else np.asarray(std).reshape(XX.shape),
        "method": method,
    }


def high_uncertainty_regions(grid: dict, top_k: int = 5, min_separation: float = 0.15) -> pd.DataFrame:
    """Pick up to ``top_k`` well-separated grid points with the largest predictive std.

    These are suggestions for where additional EM simulations or measurements
    would most reduce *model* uncertainty; they are not guaranteed improvements.
    """
    if grid.get("std") is None:
        return pd.DataFrame()
    xs, ys, std = grid["x"], grid["y"], grid["std"]
    order = np.argsort(std.ravel())[::-1]
    span_x, span_y = xs[-1] - xs[0], ys[-1] - ys[0]
    chosen: list[tuple[float, float, float, float]] = []
    for flat in order:
        iy, ix = np.unravel_index(flat, std.shape)
        px, py = xs[ix], ys[iy]
        if all(np.hypot((px - cx) / span_x, (py - cy) / span_y) >= min_separation for cx, cy, _, _ in chosen):
            chosen.append((px, py, float(std[iy, ix]), float(grid["mean"][iy, ix])))
        if len(chosen) >= top_k:
            break
    return pd.DataFrame(chosen, columns=[grid["fx"], grid["fy"], "predicted_std", "predicted_mean"])
