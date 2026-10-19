"""Regression metrics and model-comparison helpers.

* R^2  - fraction of target variance explained (1 = perfect, 0 = as good as
  predicting the mean, < 0 = worse than the mean).
* MAE  - mean absolute error, in the target's own unit (dB or dBi).
* RMSE - root mean squared error, same unit, penalises large errors more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

METRIC_EXPLANATIONS = {
    "R2": "Coefficient of determination on the held-out test set. 1.0 is perfect; 0.0 equals "
          "always predicting the training mean (the Dummy baseline); negative is worse than that.",
    "MAE": "Mean absolute error in the target's unit (dB for S11, dBi for gain). Typical size of an error.",
    "RMSE": "Root mean squared error in the target's unit. Larger than MAE when a few errors are big.",
    "CV R2": "Mean (and std) R2 across K cross-validation folds of the TRAINING set only. Used to "
             "select models so the test set is never used for selection or tuning.",
    "Coverage95": "Fraction of test points inside mean +/- 1.96*sigma. ~0.95 suggests reasonable "
                  "uncertainty scale on THIS data; it is not a proof of calibration.",
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and of equal shape.")
    return {
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def dataset_size_warning(n_samples: int) -> str | None:
    if n_samples < config.MIN_SAMPLES_TO_TRAIN:
        return (f"Only {n_samples} samples: too few to train and validate a model "
                f"(minimum {config.MIN_SAMPLES_TO_TRAIN}).")
    if n_samples < config.SMALL_DATASET_WARNING:
        return (f"Small dataset ({n_samples} samples): estimates of predictive performance may be "
                "unstable; a different random split could give noticeably different metrics.")
    return None


def comparison_table(experiments: list[dict]) -> pd.DataFrame:
    """Tabulate experiments for side-by-side comparison."""
    if not experiments:
        return pd.DataFrame()
    rows = []
    for e in experiments:
        rows.append(
            {
                "id": e.get("id"),
                "target": e.get("target_variable"),
                "model": e.get("model_type"),
                "variant": e.get("variant"),
                "test_R2": e.get("r2"),
                "test_MAE": e.get("mae"),
                "test_RMSE": e.get("rmse"),
                "cv_R2_mean": e.get("cv_r2_mean"),
                "cv_R2_std": e.get("cv_r2_std"),
                "coverage95": (e.get("extra_metrics") or {}).get("coverage95"),
                "n_train": e.get("n_train"),
                "n_test": e.get("n_test"),
                "train_s": e.get("training_seconds"),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "cv_R2_mean"], ascending=[True, False], na_position="last")


def improvement_statement(default_metrics: dict, tuned_metrics: dict, target: str, model: str) -> str:
    """Honest, measured statement on whether tuning helped (by test RMSE)."""
    d, t = default_metrics.get("rmse"), tuned_metrics.get("rmse")
    if d is None or t is None:
        return f"{model}/{target}: comparison unavailable."
    change = (d - t) / d * 100 if d else 0.0
    if t < d * 0.99:
        return f"{model}/{target}: tuning reduced test RMSE from {d:.3f} to {t:.3f} ({change:.1f}% lower)."
    if t > d * 1.01:
        return f"{model}/{target}: tuning did NOT help; test RMSE rose from {d:.3f} to {t:.3f}."
    return f"{model}/{target}: tuning made no meaningful difference (RMSE {d:.3f} vs {t:.3f})."
