"""Training, metrics, persistence, prediction, uncertainty and explainability."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from datasets import synthetic
from ml.evaluation import comparison_table, dataset_size_warning, improvement_statement, regression_metrics
from ml.explainability import (
    design_space_grid,
    high_uncertainty_regions,
    impurity_importance,
    partial_dependence_frame,
    permutation_importance_frame,
    sensitivity_sweep,
)
from ml.model_registry import MODEL_SPECS, build_pipeline, load_artifact, save_artifact
from ml.prediction import predict_point
from ml.training import TrainingConfig, train_models
from ml.uncertainty import interval_coverage, predict_with_uncertainty
from services import workflow
from utils.validation import ValidationError


def test_metric_calculations_known_values():
    m = regression_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0])
    assert m["mae"] == pytest.approx(0.25)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 1.0 / 5.0)
    perfect = regression_metrics([1, 2, 3], [1, 2, 3])
    assert perfect["r2"] == 1.0 and perfect["mae"] == 0.0
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])


def test_models_train_and_beat_dummy(trained):
    table = comparison_table(trained["experiments"])
    for target in ("S11_dB", "Gain_dBi"):
        sub = table[table["target"] == target].set_index("model")
        assert sub.loc["random_forest", "test_R2"] > sub.loc["dummy", "test_R2"] + 0.5
        assert sub.loc["dummy", "test_R2"] < 0.05
    assert all(e["dataset_checksum"] == trained["dataset"]["checksum"] for e in trained["experiments"])
    assert all(e["source_type"] == "SYNTHETIC" for e in trained["experiments"])


def test_split_is_reproducible_and_disjoint(trained):
    res = trained["results"][0]
    assert res.n_train + res.n_test == 500
    assert res.validation_config["selection_metric"].startswith("cv_r2_mean")
    # same seed -> identical test set
    df = synthetic.generate_synthetic_dataset(500, 42)
    again = train_models(df, TrainingConfig(features=["L_mm", "G_mm"], targets=["S11_dB"], model_keys=["dummy"], cv_folds=3))
    assert np.array_equal(again[0].X_test, trained["results"][0].X_test)
    train_rows = {tuple(r) for r in res.X_train}
    assert not any(tuple(r) in train_rows for r in res.X_test)


def test_active_model_selected_by_cv_not_test(trained):
    active = workflow.get_active_experiments(trained["repo"], trained["dataset"]["id"])
    for target, exp in active.items():
        pool = [e for e in trained["experiments"] if e["target_variable"] == target and e["model_type"] != "dummy"]
        assert exp["cv_r2_mean"] == max(e["cv_r2_mean"] for e in pool)
        assert exp["model_type"] != "dummy"


def test_hyperparameter_tuning_records_settings():
    df = synthetic.generate_synthetic_dataset(200, 3)
    res = train_models(df, TrainingConfig(features=["L_mm", "G_mm"], targets=["Gain_dBi"],
                                          model_keys=["gradient_boosting"], tune=True, tuning_iterations=3, cv_folds=3))
    variants = {r.variant: r for r in res}
    assert set(variants) == {"default", "tuned"}
    assert variants["tuned"].cv["n_iter"] == 3 and "best_params" in variants["tuned"].cv
    stmt = improvement_statement(variants["default"].metrics, variants["tuned"].metrics, "Gain_dBi", "GB")
    assert any(word in stmt for word in ("reduced", "did NOT help", "no meaningful difference"))


def test_model_saving_and_loading_roundtrip(trained, tmp_path):
    res = next(r for r in trained["results"] if r.model_key == "random_forest")
    path = save_artifact({"pipeline": res.pipeline, "features": res.features, "target": res.target,
                          "model_type": res.model_key}, "rt.joblib", tmp_path)
    loaded = load_artifact(path)
    assert np.allclose(loaded["pipeline"].predict(res.X_test), res.pipeline.predict(res.X_test))
    joblib.dump({"not": "a model"}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        load_artifact(tmp_path / "bad.joblib")
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "missing.joblib")
    # artefacts referenced from the database exist and load
    models = workflow.get_active_models(trained["repo"], trained["dataset"]["id"])
    assert set(models) == {"S11_dB", "Gain_dBi"}


def test_prediction_with_uncertainty_and_extrapolation(trained):
    models = trained["models"]
    res = predict_point(models, {"L_mm": 10.1, "G_mm": 1.2})
    assert set(res.predictions) == {"S11_dB", "Gain_dBi"}
    assert all(v is not None and v >= 0 for v in res.uncertainty.values())     # RF/ET provide spread
    assert not res.extrapolation
    ext = predict_point(models, {"L_mm": 14.0, "G_mm": 1.2})
    assert ext.extrapolation and ext.warnings
    with pytest.raises(ValidationError):
        predict_point(models, {"L_mm": -3.0, "G_mm": 1.0})
    rec = workflow.save_prediction(trained["repo"], models, res)
    assert rec["predicted_outputs"] == pytest.approx(res.predictions)


def test_uncertainty_only_for_supported_models():
    df = synthetic.generate_synthetic_dataset(120, 5)
    X, y = df[["L_mm", "G_mm"]].to_numpy(), df["Gain_dBi"].to_numpy()
    for key, expected in [("gaussian_process", "gp_predictive_std"), ("random_forest", "tree_ensemble_spread"),
                          ("gradient_boosting", None), ("dummy", None)]:
        pipe = build_pipeline(key, 0, 2).fit(X, y)
        mean, std, method = predict_with_uncertainty(pipe, X[:5])
        assert method == expected and len(mean) == 5
        assert (std is None) == (expected is None)
    assert interval_coverage([0, 0], [0, 0], None) is None
    assert interval_coverage([0.0, 10.0], [0.0, 0.0], [1.0, 1.0]) == 0.5


def test_missing_targets_handled():
    df = synthetic.generate_synthetic_dataset(150, 9)
    df.loc[:59, "Gain_dBi"] = np.nan                        # 60 missing gains: rows excluded only for gain
    res = train_models(df, TrainingConfig(features=["L_mm", "G_mm"], targets=["S11_dB", "Gain_dBi"],
                                          model_keys=["random_forest"], cv_folds=3))
    n = {r.target: r.n_train + r.n_test for r in res}
    assert n == {"S11_dB": 150, "Gain_dBi": 90}
    with pytest.raises(KeyError):
        train_models(df.drop(columns=["Gain_dBi"]), TrainingConfig(features=["L_mm", "G_mm"], targets=["Gain_dBi"],
                                                                   model_keys=["dummy"]))


def test_small_dataset_warning_and_refusal():
    assert dataset_size_warning(10) and "too few" in dataset_size_warning(10)
    assert "unstable" in dataset_size_warning(60)
    assert dataset_size_warning(500) is None
    with pytest.raises(ValueError):
        train_models(synthetic.generate_synthetic_dataset(12, 1),
                     TrainingConfig(features=["L_mm", "G_mm"], targets=["S11_dB"], model_keys=["dummy"]))
    res = train_models(synthetic.generate_synthetic_dataset(60, 1),
                       TrainingConfig(features=["L_mm", "G_mm"], targets=["S11_dB"], model_keys=["random_forest"]))
    assert any("unstable" in w for w in res[0].warnings)


def test_training_config_validation():
    with pytest.raises(ValueError):
        TrainingConfig(features=["L_mm"], targets=["S11_dB"], model_keys=["svm"]).validate()
    with pytest.raises(ValueError):
        TrainingConfig(features=["L_mm"], targets=["S11_dB"], model_keys=["dummy"], test_size=0.9).validate()
    assert {"dummy", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting",
            "gaussian_process"} == set(MODEL_SPECS)


def test_explainability_outputs(trained):
    lm = trained["models"]["S11_dB"]
    p = lm.payload
    imp = impurity_importance(lm.pipeline, lm.features)
    assert imp is not None and imp["importance"].sum() == pytest.approx(1.0)
    perm = permutation_importance_frame(lm.pipeline, p["X_test"], p["y_test"], lm.features, n_repeats=3)
    assert set(perm["feature"]) == {"L_mm", "G_mm"}
    assert perm.iloc[0]["feature"] == "L_mm"                 # resonance is driven mostly by L in the synthetic formula
    sweep = sensitivity_sweep(lm.pipeline, lm.features, {"L_mm": 10, "G_mm": 1}, "L_mm", 8.0, 12.0, 20)
    assert len(sweep) == 20 and sweep["std"].notna().all()
    pdf = partial_dependence_frame(lm.pipeline, p["X_train"], lm.features, "G_mm", grid_resolution=15)
    assert len(pdf) == 15
    grid = design_space_grid(lm.pipeline, lm.features, "L_mm", "G_mm", lm.train_ranges, {"L_mm": 10, "G_mm": 1}, n=20)
    assert grid["mean"].shape == (20, 20) and grid["std"].shape == (20, 20)
    hot = high_uncertainty_regions(grid, top_k=3)
    assert len(hot) == 3 and hot["predicted_std"].is_monotonic_decreasing
