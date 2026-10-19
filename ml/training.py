"""Training, validation and hyper-parameter tuning.

Leakage-prevention protocol (applied to every model and target):

1. A single reproducible train/test split (``random_state=seed``) is made
   BEFORE any fitting. The test set is touched exactly once, for the final
   metrics.
2. Scaling lives inside the ``Pipeline``, so cross-validation refits the scaler
   on each training fold.
3. Cross-validation (K-fold, shuffled, seeded) and ``RandomizedSearchCV`` run on
   the TRAINING portion only. Model *selection* in the application uses the CV
   score, never the test score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split

import config
from ml.evaluation import dataset_size_warning, regression_metrics
from ml.model_registry import MODEL_SPECS, build_pipeline
from ml.preprocessing import xy_for_target
from ml.uncertainty import interval_coverage, predict_with_uncertainty
from utils.logging_config import get_logger

log = get_logger("training")


@dataclass
class TrainingConfig:
    features: list[str]
    targets: list[str]
    model_keys: list[str]
    test_size: float = config.DEFAULT_TEST_SIZE
    seed: int = config.RANDOM_SEED
    cv_folds: int = config.DEFAULT_CV_FOLDS
    tune: bool = False
    tuning_iterations: int = config.DEFAULT_TUNING_ITERATIONS
    custom_search_spaces: dict[str, dict] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.features:
            raise ValueError("Select at least one input feature.")
        if not self.targets:
            raise ValueError("Select at least one target variable.")
        if not self.model_keys:
            raise ValueError("Select at least one model.")
        unknown = [m for m in self.model_keys if m not in MODEL_SPECS]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}")
        if not 0.05 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.05 and 0.5.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2.")
        if self.tuning_iterations < 1:
            raise ValueError("tuning_iterations must be >= 1.")


@dataclass
class TrainingResult:
    model_key: str
    variant: str
    target: str
    features: list[str]
    pipeline: object
    hyperparameters: dict
    metrics: dict
    cv: dict
    extra_metrics: dict
    train_ranges: dict
    validation_config: dict
    n_train: int
    n_test: int
    seconds: float
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_std: np.ndarray | None
    warnings: list[str] = field(default_factory=list)


def _jsonable_params(estimator) -> dict:
    out = {}
    for key, value in estimator.get_params(deep=False).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _cv_scores(pipeline, X, y, folds: int, seed: int) -> dict:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        clone(pipeline), X, y, cv=kf,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"}, error_score="raise",
    )
    return {
        "cv_r2_mean": float(np.mean(scores["test_r2"])),
        "cv_r2_std": float(np.std(scores["test_r2"])),
        "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
        "folds": folds,
    }


def _finalise(model_key, variant, target, features, pipe, cv, X_tr, X_te, y_te, cfg, val_cfg, t0, warnings):
    mean, std, method = predict_with_uncertainty(pipe, X_te)
    metrics = regression_metrics(y_te, mean)
    extra = {"uncertainty_method": method}
    if std is not None:
        extra["coverage95"] = interval_coverage(y_te, mean, std)
        extra["mean_test_std"] = float(np.mean(std))
    return TrainingResult(
        model_key=model_key, variant=variant, target=target, features=list(features), pipeline=pipe,
        hyperparameters=_jsonable_params(pipe.named_steps["model"]), metrics=metrics, cv=cv,
        extra_metrics=extra,
        train_ranges={f: (float(X_tr[:, i].min()), float(X_tr[:, i].max())) for i, f in enumerate(features)},
        validation_config=val_cfg, n_train=len(X_tr), n_test=len(X_te), seconds=time.perf_counter() - t0,
        X_train=X_tr, X_test=X_te, y_test=y_te, y_pred=np.asarray(mean), y_std=std, warnings=list(warnings),
    )


def train_models(df: pd.DataFrame, cfg: TrainingConfig, progress=None) -> list[TrainingResult]:
    """Train every (target, model[, tuned]) combination described by ``cfg``."""
    cfg.validate()
    results: list[TrainingResult] = []
    jobs = [(t, m) for t in cfg.targets for m in cfg.model_keys]
    for job_idx, (target, model_key) in enumerate(jobs):
        if progress:
            progress(job_idx / len(jobs), f"Training {model_key} for {target}")
        X, y, _ = xy_for_target(df, cfg.features, target)
        warnings: list[str] = []
        size_msg = dataset_size_warning(len(y))
        if len(y) < config.MIN_SAMPLES_TO_TRAIN:
            raise ValueError(f"Target '{target}': {size_msg}")
        if size_msg:
            warnings.append(size_msg)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed)
        spec = MODEL_SPECS[model_key]
        if spec.max_train_samples and len(y_tr) > spec.max_train_samples:
            warnings.append(f"{spec.label} skipped: {len(y_tr)} training rows exceed its limit "
                            f"of {spec.max_train_samples} (cubic cost).")
            log.warning(warnings[-1])
            continue
        warnings.extend(spec.notes)
        use_cv = len(y_tr) >= config.MIN_SAMPLES_FOR_CV
        folds = min(cfg.cv_folds, len(y_tr) // 5) if use_cv else 0
        val_cfg = {
            "split": "train_test_split", "test_size": cfg.test_size, "shuffle": True,
            "cv": f"KFold(n_splits={folds}, shuffle=True)" if folds >= 2 else "none (too few samples)",
            "selection_metric": "cv_r2_mean (training data only)",
            "n_samples_used": int(len(y)),
        }
        if folds < 2:
            warnings.append("Cross-validation disabled: training set too small.")

        # --- default configuration ------------------------------------------
        t0 = time.perf_counter()
        pipe = build_pipeline(model_key, cfg.seed, len(cfg.features))
        cv = _cv_scores(pipe, X_tr, y_tr, folds, cfg.seed) if folds >= 2 else {}
        pipe.fit(X_tr, y_tr)
        results.append(_finalise(model_key, "default", target, cfg.features, pipe, cv,
                                 X_tr, X_te, y_te, cfg, val_cfg, t0, warnings))
        log.info("Trained %s/%s default: R2=%.3f", model_key, target, results[-1].metrics["r2"])

        # --- tuned configuration --------------------------------------------
        if cfg.tune and spec.tunable and folds >= 2:
            space = cfg.custom_search_spaces.get(model_key) or spec.search_space(len(cfg.features))
            if not space:
                continue
            t0 = time.perf_counter()
            grid_size = int(np.prod([len(v) for v in space.values()]))
            n_iter = min(cfg.tuning_iterations, grid_size)
            search = RandomizedSearchCV(
                build_pipeline(model_key, cfg.seed, len(cfg.features)), space,
                n_iter=n_iter, cv=KFold(n_splits=folds, shuffle=True, random_state=cfg.seed),
                scoring="r2", random_state=cfg.seed, n_jobs=1, refit=True, error_score="raise",
            )
            search.fit(X_tr, y_tr)
            best_idx = search.best_index_
            cv_t = {
                "cv_r2_mean": float(search.best_score_),
                "cv_r2_std": float(search.cv_results_["std_test_score"][best_idx]),
                "folds": folds,
                "search": "RandomizedSearchCV",
                "n_iter": n_iter,
                "best_params": {k: str(v) for k, v in search.best_params_.items()},
            }
            tuned_cfg = {**val_cfg, "tuning": "RandomizedSearchCV on training folds only",
                         "search_space": {k: [str(v) for v in vals] for k, vals in space.items()}}
            results.append(_finalise(model_key, "tuned", target, cfg.features, search.best_estimator_, cv_t,
                                     X_tr, X_te, y_te, cfg, tuned_cfg, t0, warnings))
            log.info("Trained %s/%s tuned: R2=%.3f", model_key, target, results[-1].metrics["r2"])
    if progress:
        progress(1.0, "Training complete")
    return results
