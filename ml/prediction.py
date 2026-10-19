"""Inference with saved surrogate models (one model per target)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.model_registry import load_artifact
from ml.uncertainty import predict_with_uncertainty
from utils.validation import validate_geometry


@dataclass
class LoadedModel:
    experiment: dict
    payload: dict

    @property
    def target(self) -> str:
        return self.payload["target"]

    @property
    def features(self) -> list[str]:
        return list(self.payload["features"])

    @property
    def pipeline(self):
        return self.payload["pipeline"]

    @property
    def train_ranges(self) -> dict:
        return {k: tuple(v) for k, v in self.payload.get("train_ranges", {}).items()}


def load_model(experiment: dict) -> LoadedModel:
    if not experiment.get("artifact_path"):
        raise FileNotFoundError(f"Experiment #{experiment.get('id')} has no saved artefact.")
    return LoadedModel(experiment=experiment, payload=load_artifact(experiment["artifact_path"]))


@dataclass
class PredictionResult:
    inputs: dict
    predictions: dict
    uncertainty: dict
    uncertainty_method: dict
    warnings: list[str] = field(default_factory=list)
    extrapolation: bool = False


def check_feature_consistency(models: dict[str, LoadedModel]) -> list[str]:
    feature_sets = {tuple(m.features) for m in models.values()}
    if len(feature_sets) != 1:
        raise ValueError("Selected models use different input features; choose a consistent model set.")
    return list(next(iter(feature_sets)))


def predict_point(models: dict[str, LoadedModel], params: dict[str, float]) -> PredictionResult:
    """Validate ``params`` and predict every target with its model."""
    if not models:
        raise ValueError("No trained models available.")
    features = check_feature_consistency(models)
    ranges: dict[str, tuple[float, float]] = {}
    for m in models.values():
        for f, (lo, hi) in m.train_ranges.items():
            cur = ranges.get(f, (lo, hi))
            ranges[f] = (max(cur[0], lo), min(cur[1], hi))
    check = validate_geometry(params, features, ranges)
    check.raise_if_invalid()
    x = np.array([[float(params[f]) for f in features]])
    preds, unc, methods = {}, {}, {}
    for target, model in models.items():
        mean, std, method = predict_with_uncertainty(model.pipeline, x)
        preds[target] = float(mean[0])
        unc[target] = None if std is None else float(std[0])
        methods[target] = method
    return PredictionResult(
        inputs={f: float(params[f]) for f in features}, predictions=preds, uncertainty=unc,
        uncertainty_method=methods, warnings=check.warnings, extrapolation=bool(check.warnings),
    )


def predict_batch(models: dict[str, LoadedModel], X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Vectorised prediction for optimizers and design-space maps (no validation)."""
    out = {}
    for target, model in models.items():
        mean, std, _ = predict_with_uncertainty(model.pipeline, X)
        out[target] = (np.asarray(mean, dtype=float), None if std is None else np.asarray(std, dtype=float))
    return out
