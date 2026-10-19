"""Glue between the REST API and the existing MetaOpt-AI modules.

Everything scientific (training, prediction, optimization, provenance) is delegated to the existing
packages: ``services.workflow``, ``ml``, ``optimization``, ``database``. This module only shapes data
for JSON and caches loaded models.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import config
from database.repository import Repository
from datasets import synthetic
from datasets.public import SOURCES, read_status
from ml.explainability import impurity_importance, permutation_importance_frame
from ml.model_registry import model_label
from ml.prediction import LoadedModel
from optimization.objectives import DIRECTION
from services import workflow

DEFAULT_MODELS = ["dummy", "random_forest", "gradient_boosting"]

FREQUENCY_NOTES = {
    workflow.SYNTHETIC_DATASET_NAME: f"S11 refers to the single design frequency {synthetic.F0_GHZ:g} GHz assumed by the "
                                     "synthetic generator.",
    SOURCES["metu_simpl"]["dataset_name"]: "Frequency is not an input. S11 targets come from the HFSS sweep "
                                           "(23–33 GHz, 0.05 GHz steps); “S11 at 28 GHz” is read at exactly 28.00 GHz.",
}


@lru_cache(maxsize=1)
def repo() -> Repository:
    config.ensure_directories()
    r = Repository()
    workflow.ensure_demo_dataset(r)
    return r


def _clean(value):
    """JSON-safe floats (NaN/inf -> None)."""
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# ------------------------------------------------------------------------------ datasets
def dataset_key(ds: dict) -> str:
    for key, meta in SOURCES.items():
        if meta["dataset_name"] == ds["name"]:
            return key
    return "synthetic" if ds["dataset_type"] == "SYNTHETIC" else f"dataset-{ds['id']}"


def dataset_label(ds: dict) -> str:
    key = dataset_key(ds)
    if key in SOURCES:
        return SOURCES[key]["label"]
    return "Synthetic demo" if key == "synthetic" else ds["name"]


def visible_datasets() -> list[dict]:
    """Public datasets only when their whole pipeline completed; other registered datasets always."""
    out = []
    for ds in repo().list_datasets():
        key = dataset_key(ds)
        if key in SOURCES and not read_status(key).get("complete"):
            continue
        out.append(ds)
    order = {k: i for i, k in enumerate(list(SOURCES) + ["synthetic"])}
    return sorted(out, key=lambda d: order.get(dataset_key(d), 99))


def unavailable_sources() -> list[dict]:
    rows = []
    for key, meta in SOURCES.items():
        status = read_status(key)
        if not status.get("complete"):
            rows.append({"key": key, "label": meta["label"], "message": status.get("message") or "Pipeline not run.",
                         "stages": {k: v.get("ok") for k, v in status.get("stages", {}).items()}})
    return rows


@lru_cache(maxsize=16)
def _frame(dataset_id: int, checksum: str) -> pd.DataFrame:
    df, _, _ = workflow.load_dataset_frame(repo(), dataset_id)
    return df


def frame(ds: dict) -> pd.DataFrame:
    return _frame(ds["id"], ds["checksum"])


def get_dataset(dataset_id: int) -> dict:
    ds = repo().get_dataset(dataset_id)
    if ds is None or ds["id"] not in {d["id"] for d in visible_datasets()}:
        raise LookupError(f"Dataset {dataset_id} not found or not available.")
    return ds


def feature_meta(ds: dict, df: pd.DataFrame, models: dict[str, LoadedModel]) -> list[dict]:
    ranges = workflow.model_training_ranges(models) if models else {}
    out = []
    for f in ds["input_features"]:
        meta = config.KNOWN_COLUMNS.get(f, {})
        lo, hi = ranges.get(f, (float(df[f].min()), float(df[f].max())))
        out.append({"name": f, "short": config.short_label(f), "unit": config.unit_of(f),
                    "description": meta.get("description", f), "group": meta.get("group", "Parameters"),
                    "min": lo, "max": hi, "median": float(df[f].median()),
                    "optimize_default": bool(meta.get("optimize_default"))})
    return out


def target_meta(ds: dict, df: pd.DataFrame) -> list[dict]:
    return [{"name": t, "short": config.short_label(t), "unit": config.unit_of(t),
             "description": config.KNOWN_COLUMNS.get(t, {}).get("description", t),
             "direction": DIRECTION.get(t), "n_values": int(df[t].notna().sum())}
            for t in ds["target_names"]]


def experiment_row(e: dict, active_ids: set[int] | None = None) -> dict:
    return {
        "id": e["id"], "target": e["target_variable"], "target_short": config.short_label(e["target_variable"]),
        "unit": config.unit_of(e["target_variable"]), "model": e["model_type"], "model_label": model_label(e["model_type"]),
        "variant": e["variant"], "r2": _clean(e["r2"]), "mae": _clean(e["mae"]), "rmse": _clean(e["rmse"]),
        "cv_r2_mean": _clean(e["cv_r2_mean"]), "cv_r2_std": _clean(e["cv_r2_std"]), "n_train": e["n_train"],
        "n_test": e["n_test"], "created_at": e["created_at"], "run_group": e["run_group"],
        "active": bool(active_ids and e["id"] in active_ids),
    }


# --------------------------------------------------------------------------------- models
@lru_cache(maxsize=64)
def _load(experiment_id: int, artifact_path: str) -> LoadedModel:
    exp = repo().get_experiment(experiment_id)
    return workflow.load_models({exp["target_variable"]: exp})[exp["target_variable"]]


def active_models(dataset_id: int) -> dict[str, LoadedModel]:
    exps = workflow.get_active_experiments(repo(), dataset_id)
    return {t: _load(e["id"], e["artifact_path"]) for t, e in exps.items()}


def load_experiment(experiment_id: int) -> LoadedModel:
    exp = repo().get_experiment(experiment_id)
    if exp is None or not exp["artifact_path"]:
        raise LookupError(f"Experiment {experiment_id} has no saved model.")
    return _load(exp["id"], exp["artifact_path"])


def dataset_view(ds: dict) -> dict:
    df = frame(ds)
    models = active_models(ds["id"])
    active = {t: experiment_row(m.experiment, {m.experiment["id"]}) for t, m in models.items()}
    for t, m in models.items():
        active[t]["n_samples_used"] = (m.experiment.get("validation_config") or {}).get("n_samples_used")
    cols = ds["input_features"] + ds["target_names"]
    return {
        "id": ds["id"], "key": dataset_key(ds), "label": dataset_label(ds), "name": ds["name"],
        "type": ds["dataset_type"], "type_description": config.DATASET_TYPES[ds["dataset_type"]],
        "provenance": ds["provenance"], "description": ds["description"], "sample_count": ds["sample_count"],
        "checksum": ds["checksum"], "created_at": ds["created_at"],
        "frequency_note": FREQUENCY_NOTES.get(ds["name"], "Frequency is not an input column in this dataset."),
        "features": feature_meta(ds, df, models), "targets": target_meta(ds, df),
        "quality": {"rows": len(df), "missing_values": int(df[cols].isna().sum().sum()),
                    "duplicate_rows": int(df[cols].duplicated().sum())},
        "active_models": active, "ready": bool(models),
    }


# ---------------------------------------------------------------------------- diagnostics
@lru_cache(maxsize=64)
def diagnostics(experiment_id: int) -> dict:
    lm = load_experiment(experiment_id)
    p, exp = lm.payload, lm.experiment
    y, yhat = np.asarray(p["y_test"], float), np.asarray(p["y_pred"], float)
    points = [{"actual": float(a), "predicted": float(b), "residual": float(a - b)} for a, b in zip(y, yhat)]
    importance = {"permutation": None, "impurity": None}
    if exp["model_type"] != "dummy":
        perm = permutation_importance_frame(lm.pipeline, p["X_test"], p["y_test"], lm.features, seed=p.get("seed", 42))
        importance["permutation"] = [{"feature": r.feature, "short": config.short_label(r.feature),
                                      "mean": float(r.importance_mean), "std": float(r.importance_std)}
                                     for r in perm.itertuples()]
        imp = impurity_importance(lm.pipeline, lm.features)
        if imp is not None:
            importance["impurity"] = [{"feature": r.feature, "short": config.short_label(r.feature),
                                       "mean": float(r.importance)} for r in imp.itertuples()]
    return {"experiment": experiment_row(exp), "points": points, "importance": importance}


def nearest_simulated(df: pd.DataFrame, features: list[str], targets: list[str], designs: list[dict]) -> list[dict]:
    """Closest design that exists in the dataset (normalized distance) — a reference, not a validation."""
    X = df[features].to_numpy(float)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    rows = []
    for d in designs:
        x = np.array([d["parameters"][f] for f in features])
        dist = np.sqrt((((X - x) / scale) ** 2).mean(axis=1))
        i = int(dist.argmin())
        rows.append({"design": d["label"], "row": int(i) + 1, "distance": float(dist[i]),
                     "values": [{"target": t, "short": config.short_label(t), "unit": config.unit_of(t),
                                 "dataset": _clean(df.iloc[i][t]), "predicted": _clean(d["predictions"].get(t))}
                                for t in targets]})
    return rows
