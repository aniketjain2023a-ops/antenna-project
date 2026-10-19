"""Model catalogue and artefact persistence.

``MODEL_SPECS`` is the single place where model types, defaults and tuning
search spaces are defined. Every estimator is wrapped in a scikit-learn
``Pipeline`` (scaler -> model) so preprocessing is fitted on the training fold
only, which prevents data leakage during cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

ARTIFACT_FORMAT_VERSION = 1


def _gp_kernel(n_features: int, kind: str = "rbf"):
    ls = np.ones(n_features)
    base = RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3)) if kind == "rbf" else \
        Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5 if kind == "matern52" else 1.5)
    return ConstantKernel(1.0, (1e-2, 1e3)) * base + WhiteKernel(1e-2, (1e-6, 1e1))


@dataclass
class ModelSpec:
    key: str
    label: str
    build: Callable[[int, int], object]          # (seed, n_features) -> estimator
    needs_scaling: bool
    search_space: Callable[[int], dict]           # n_features -> param distributions
    description: str
    uncertainty: str | None = None
    max_train_samples: int | None = None
    tunable: bool = True
    notes: list[str] = field(default_factory=list)


MODEL_SPECS: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        key="dummy", label="Dummy baseline (mean)",
        build=lambda seed, d: DummyRegressor(strategy="mean"),
        needs_scaling=False, search_space=lambda d: {}, tunable=False,
        description="Always predicts the training mean. Any useful model must beat it.",
    ),
    "random_forest": ModelSpec(
        key="random_forest", label="Random Forest",
        build=lambda seed, d: RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=1),
        needs_scaling=False,
        search_space=lambda d: {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, 0.7],
        },
        description="Bagged ensemble of deep decision trees (Breiman, 2001).",
        uncertainty="Spread (std) of individual tree predictions - approximate, uncalibrated.",
    ),
    "extra_trees": ModelSpec(
        key="extra_trees", label="Extra Trees",
        build=lambda seed, d: ExtraTreesRegressor(n_estimators=150, random_state=seed, n_jobs=1),
        needs_scaling=False,
        search_space=lambda d: {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, 0.7],
        },
        description="Extremely randomized trees (Geurts et al., 2006); random split thresholds.",
        uncertainty="Spread (std) of individual tree predictions - approximate, uncalibrated.",
    ),
    "gradient_boosting": ModelSpec(
        key="gradient_boosting", label="Gradient Boosting",
        build=lambda seed, d: GradientBoostingRegressor(random_state=seed),
        needs_scaling=False,
        search_space=lambda d: {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        },
        description="Stage-wise additive trees fitted to residuals (Friedman, 2001).",
    ),
    "hist_gradient_boosting": ModelSpec(
        key="hist_gradient_boosting", label="Histogram Gradient Boosting",
        build=lambda seed, d: HistGradientBoostingRegressor(random_state=seed),
        needs_scaling=False,
        search_space=lambda d: {
            "model__max_iter": [100, 200, 400],
            "model__learning_rate": [0.03, 0.1],
            "model__max_leaf_nodes": [15, 31],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
        description="Binned gradient boosting; designed for large datasets (>10k rows).",
        notes=["Primarily intended for large datasets; on a few hundred samples it offers little "
               "advantage over classic gradient boosting."],
    ),
    "gaussian_process": ModelSpec(
        key="gaussian_process", label="Gaussian Process",
        build=lambda seed, d: GaussianProcessRegressor(
            kernel=_gp_kernel(d, "rbf"), normalize_y=True, n_restarts_optimizer=2, random_state=seed),
        needs_scaling=True,
        search_space=lambda d: {
            "model__kernel": [_gp_kernel(d, "rbf"), _gp_kernel(d, "matern52"), _gp_kernel(d, "matern32")],
        },
        description="Bayesian non-parametric regression with ARD kernel + white noise "
                    "(Rasmussen & Williams, 2006). O(n^3) cost: only for small/medium datasets.",
        uncertainty="GP predictive standard deviation (includes learned noise); calibration checked on test set.",
        max_train_samples=config.GP_MAX_TRAIN_SAMPLES,
    ),
}


def build_pipeline(model_key: str, seed: int, n_features: int) -> Pipeline:
    spec = MODEL_SPECS[model_key]
    scaler = StandardScaler() if spec.needs_scaling else "passthrough"
    return Pipeline([("scaler", scaler), ("model", spec.build(seed, n_features))])


def model_label(model_key: str) -> str:
    spec = MODEL_SPECS.get(model_key)
    return spec.label if spec else model_key


def artifact_filename(experiment_id: int, model_key: str, variant: str, target: str) -> str:
    safe_target = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in target)
    return f"exp{experiment_id:05d}_{model_key}_{variant}_{safe_target}.joblib"


def save_artifact(payload: dict, filename: str, models_dir: str | Path | None = None) -> Path:
    """Persist a model payload (pipeline + metadata) with joblib."""
    directory = Path(models_dir or config.MODELS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    payload = {**payload, "artifact_format_version": ARTIFACT_FORMAT_VERSION}
    joblib.dump(payload, path, compress=3)
    return path


def load_artifact(path: str | Path) -> dict:
    """Load a saved payload and check that it has the expected structure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found: {path}")
    payload = joblib.load(path)
    required = {"pipeline", "features", "target", "model_type"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"File {path} is not a valid MetaOpt-AI model artefact.")
    return payload
