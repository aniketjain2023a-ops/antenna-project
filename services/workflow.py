"""Application services: orchestrate datasets, training, prediction and optimization.

The Streamlit pages, the headless demo script and the tests all call these
functions, so the same logic (and the same database logging) is used everywhere.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

import config
from database.repository import DuplicateRecordError, Repository
from datasets import synthetic
from ml.model_registry import artifact_filename, save_artifact
from ml.prediction import LoadedModel, PredictionResult, load_model, predict_point
from ml.training import TrainingConfig, TrainingResult, train_models
from optimization import ALGORITHMS
from optimization.base import OptimizationOutcome
from optimization.constraints import bounds_array, check_bounds
from optimization.objectives import ObjectiveConfig, SurrogateEvaluator, describe_point
from utils.logging_config import get_logger
from utils.validation import ValidationError, file_checksum, validate_geometry

log = get_logger("workflow")

SYNTHETIC_DATASET_NAME = "Synthetic demonstration dataset v1 (L_mm, G_mm)"
ACTIVE_DATASET_KEY = "active_dataset_id"


# --------------------------------------------------------------------- paths
def _portable(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(config.BASE_DIR))
    except ValueError:
        return str(path)


def resolve_path(stored: str | None) -> Path | None:
    if not stored:
        return None
    p = Path(stored)
    return p if p.is_absolute() else config.BASE_DIR / p


# ------------------------------------------------------------------ datasets
def ensure_demo_dataset(repo: Repository, csv_path: Path | None = None, n_samples: int = 500,
                        seed: int = config.RANDOM_SEED) -> dict:
    """Create (if needed) the synthetic CSV and register it in SQLite. Idempotent."""
    csv_path = Path(csv_path or config.SYNTHETIC_CSV_PATH)
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.generate_synthetic_dataset(n_samples, seed).to_csv(csv_path, index=False)
        log.info("Generated synthetic dataset at %s", csv_path)
    checksum = file_checksum(csv_path)
    existing = repo.find_dataset_by_checksum(checksum)
    if existing:
        return existing
    by_name = repo.get_dataset_by_name(SYNTHETIC_DATASET_NAME)
    if by_name:
        log.warning("Synthetic CSV checksum differs from the registered dataset #%s; keeping the "
                    "registered record. Delete the database to re-register.", by_name["id"])
        return by_name
    df = pd.read_csv(csv_path)
    ds = repo.add_dataset(
        name=SYNTHETIC_DATASET_NAME, dataset_type="SYNTHETIC", provenance=synthetic.PROVENANCE,
        description=(f"{len(df)} Latin-hypercube samples, seed={seed}, L_mm in {synthetic.L_RANGE_MM}, "
                     f"G_mm in {synthetic.G_RANGE_MM}; noise sigma S11={synthetic.S11_NOISE_DB} dB, "
                     f"gain={synthetic.GAIN_NOISE_DB} dB. Demonstrates the workflow only."),
        input_features=synthetic.FEATURES, target_names=synthetic.TARGETS, checksum=checksum,
        file_path=_portable(csv_path), designs=df,
    )
    if repo.get_setting(ACTIVE_DATASET_KEY) is None:
        repo.set_setting(ACTIVE_DATASET_KEY, ds["id"])
    return ds


def register_dataframe(repo: Repository, df: pd.DataFrame, *, name: str, dataset_type: str, provenance: str,
                       description: str, features: list[str], targets: list[str],
                       upload_dir: Path | None = None) -> dict:
    """Store a cleaned copy of user data as CSV and register it (with designs) in SQLite."""
    if dataset_type not in config.DATASET_TYPES:
        raise ValueError(f"Unknown dataset type {dataset_type}")
    if not name.strip():
        raise ValueError("Dataset name is required.")
    missing = [c for c in features + targets if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
    for c in features:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Input feature '{c}' must be numeric.")
    payload = df.to_csv(index=False).encode("utf-8")
    import hashlib

    checksum = hashlib.sha256(payload).hexdigest()
    dup = repo.find_dataset_by_checksum(checksum)
    if dup:
        raise DuplicateRecordError(f"Identical data already registered as dataset #{dup['id']} ('{dup['name']}').")
    upload_dir = Path(upload_dir or config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:60]
    path = upload_dir / f"{safe}_{checksum[:8]}.csv"
    path.write_bytes(payload)
    try:
        ds = repo.add_dataset(name=name, dataset_type=dataset_type, provenance=provenance, description=description,
                              input_features=features, target_names=targets, checksum=checksum,
                              file_path=_portable(path), designs=df)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return ds


def load_dataset_frame(repo: Repository, dataset_id: int) -> tuple[pd.DataFrame, dict, list[str]]:
    """Load a registered dataset's CSV and verify its checksum."""
    ds = repo.get_dataset(dataset_id)
    if ds is None:
        raise LookupError(f"Dataset #{dataset_id} not found.")
    path = resolve_path(ds["file_path"])
    warnings = []
    if path is None or not path.exists():
        raise FileNotFoundError(f"Data file for dataset #{dataset_id} is missing: {path}")
    if file_checksum(path) != ds["checksum"]:
        warnings.append("Checksum mismatch: the CSV on disk differs from the registered version. "
                        "Results may not be reproducible.")
    return pd.read_csv(path), ds, warnings


def get_active_dataset_id(repo: Repository) -> int | None:
    value = repo.get_setting(ACTIVE_DATASET_KEY)
    if value is not None and repo.get_dataset(int(value)):
        return int(value)
    datasets = repo.list_datasets()
    return datasets[0]["id"] if datasets else None


def set_active_dataset_id(repo: Repository, dataset_id: int) -> None:
    repo.set_setting(ACTIVE_DATASET_KEY, int(dataset_id))


# ------------------------------------------------------------------ training
def _active_key(dataset_id: int) -> str:
    return f"active_models:{dataset_id}"


def train_and_log(repo: Repository, dataset_id: int, cfg: TrainingConfig, progress=None,
                  models_dir: Path | None = None, notes: str = "") -> tuple[list[dict], list[TrainingResult]]:
    """Train models, persist artefacts, log experiments; auto-select active models by CV R2."""
    df, ds, warns = load_dataset_frame(repo, dataset_id)
    missing = [c for c in cfg.features + cfg.targets if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in dataset: {missing}")
    results = train_models(df, cfg, progress=progress)
    run_group = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    experiments = []
    for r in results:
        exp = repo.add_experiment(
            dataset_id=dataset_id, run_group=run_group, model_type=r.model_key, variant=r.variant,
            target_variable=r.target, input_features=r.features, hyperparameters=r.hyperparameters,
            random_seed=cfg.seed, validation_config=r.validation_config, r2=r.metrics["r2"],
            mae=r.metrics["mae"], rmse=r.metrics["rmse"], cv_r2_mean=r.cv.get("cv_r2_mean"),
            cv_r2_std=r.cv.get("cv_r2_std"), cv_rmse_mean=r.cv.get("cv_rmse_mean"),
            extra_metrics={**r.extra_metrics, "cv": r.cv, "warnings": r.warnings},
            train_ranges={k: list(v) for k, v in r.train_ranges.items()}, n_train=r.n_train, n_test=r.n_test,
            training_seconds=r.seconds, dataset_checksum=ds["checksum"], source_type=ds["dataset_type"],
            notes=notes,
        )
        payload = {
            "pipeline": r.pipeline, "features": r.features, "target": r.target, "model_type": r.model_key,
            "variant": r.variant, "train_ranges": r.train_ranges, "seed": cfg.seed,
            "X_train": r.X_train[:2000], "X_test": r.X_test, "y_test": r.y_test, "y_pred": r.y_pred,
            "y_std": r.y_std, "dataset_id": dataset_id, "dataset_checksum": ds["checksum"],
            "source_type": ds["dataset_type"], "experiment_id": exp["id"],
            "sklearn_version": sklearn.__version__, "created": datetime.now().isoformat(timespec="seconds"),
        }
        path = save_artifact(payload, artifact_filename(exp["id"], r.model_key, r.variant, r.target), models_dir)
        experiments.append(repo.update_experiment(exp["id"], artifact_path=_portable(path)))
    select_best_active_models(repo, dataset_id, features=cfg.features)
    return experiments, results


def select_best_active_models(repo: Repository, dataset_id: int, candidate_ids: list[int] | None = None,
                              features: list[str] | None = None) -> dict:
    """Pick, per target, the non-dummy experiment with the best CV R2 (test set is NOT used).

    Only experiments using the same input-feature list are compared, because CV scores of
    models with different inputs answer different questions.
    """
    exps = repo.list_experiments(dataset_id=dataset_id)
    if candidate_ids is not None:
        exps = [e for e in exps if e["id"] in set(candidate_ids)]
    if features is not None:
        exps = [e for e in exps if list(e["input_features"]) == list(features)]
    active = dict(repo.get_setting(_active_key(dataset_id), {}) or {})
    for target in sorted({e["target_variable"] for e in exps}):
        pool = [e for e in exps if e["target_variable"] == target and e["model_type"] != "dummy"
                and e["artifact_path"]]
        if not pool:
            continue
        best = max(pool, key=lambda e: (e["cv_r2_mean"] if e["cv_r2_mean"] is not None else -np.inf))
        active[target] = best["id"]
    repo.set_setting(_active_key(dataset_id), active)
    return active


def set_active_model(repo: Repository, dataset_id: int, target: str, experiment_id: int) -> None:
    exp = repo.get_experiment(experiment_id)
    if exp is None or exp["dataset_id"] != dataset_id or exp["target_variable"] != target:
        raise ValueError("Experiment does not match dataset/target.")
    active = dict(repo.get_setting(_active_key(dataset_id), {}) or {})
    active[target] = experiment_id
    repo.set_setting(_active_key(dataset_id), active)


def get_active_experiments(repo: Repository, dataset_id: int) -> dict[str, dict]:
    active = repo.get_setting(_active_key(dataset_id), {}) or {}
    out = {}
    for target, exp_id in active.items():
        exp = repo.get_experiment(int(exp_id))
        if exp and exp["artifact_path"] and resolve_path(exp["artifact_path"]).exists():
            out[target] = exp
    return out


def load_models(experiments: dict[str, dict]) -> dict[str, LoadedModel]:
    loaded = {}
    for target, exp in experiments.items():
        exp = {**exp, "artifact_path": str(resolve_path(exp["artifact_path"]))}
        loaded[target] = load_model(exp)
    return loaded


def get_active_models(repo: Repository, dataset_id: int) -> dict[str, LoadedModel]:
    return load_models(get_active_experiments(repo, dataset_id))


# ---------------------------------------------------------------- prediction
def predict_and_maybe_save(repo: Repository, models: dict[str, LoadedModel], params: dict,
                           save: bool = False) -> tuple[PredictionResult, dict | None]:
    result = predict_point(models, params)
    record = None
    if save:
        record = save_prediction(repo, models, result)
    return result, record


def save_prediction(repo: Repository, models: dict[str, LoadedModel], result: PredictionResult) -> dict:
    first = next(iter(models.values()))
    return repo.add_prediction(
        model_experiment_id=first.experiment["id"],
        model_experiment_ids={t: m.experiment["id"] for t, m in models.items()},
        input_parameters=result.inputs, predicted_outputs=result.predictions, uncertainty=result.uncertainty,
        extrapolation=result.extrapolation, data_source_type=first.experiment.get("source_type", ""),
    )


# -------------------------------------------------------------- optimization
def target_ranges(df: pd.DataFrame, targets: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for t in targets:
        s = pd.to_numeric(df[t], errors="coerce").dropna()
        if len(s):
            out[t] = (float(s.min()), float(s.max()))
    return out


def model_training_ranges(models: dict[str, LoadedModel]) -> dict[str, tuple[float, float]]:
    """Intersection of the feature ranges seen by every model."""
    ranges: dict[str, tuple[float, float]] = {}
    for m in models.values():
        for f, (lo, hi) in m.train_ranges.items():
            cur = ranges.get(f, (lo, hi))
            ranges[f] = (max(cur[0], lo), min(cur[1], hi))
    return ranges


def run_optimization(repo: Repository, models: dict[str, LoadedModel], cfg: ObjectiveConfig,
                     bounds: dict[str, tuple[float, float]], baseline: dict[str, float],
                     algorithms: list[str], seed: int = config.RANDOM_SEED,
                     budget: int = config.DEFAULT_EVAL_BUDGET, allow_extrapolation: bool = False,
                     save: bool = True) -> list[tuple[OptimizationOutcome, dict | None, dict]]:
    """Run each requested optimizer on the surrogate, compare to the baseline and persist."""
    if not algorithms:
        raise ValueError("Select at least one optimization algorithm.")
    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}")
    if budget < 50:
        raise ValueError("Evaluation budget must be at least 50.")
    evaluator = SurrogateEvaluator(models)
    cfg.validate(list(models))
    train_ranges = model_training_ranges(models)
    bound_warnings = check_bounds(bounds, train_ranges, allow_extrapolation)
    base_check = validate_geometry(baseline, evaluator.features, None)
    base_check.raise_if_invalid()
    bnds = bounds_array(bounds, evaluator.features)
    x_base = np.array([float(baseline[f]) for f in evaluator.features])
    base_desc = describe_point(evaluator, cfg, x_base)

    first = next(iter(models.values())).experiment
    outputs = []
    for algo in algorithms:
        _, fn = ALGORITHMS[algo]
        outcome = fn(evaluator, cfg, bounds, seed=seed, budget=budget)
        if not outcome.within_bounds:
            raise ValidationError(f"{algo} returned a candidate outside the bounds.")
        cand = outcome.candidate
        message = outcome.message
        if bound_warnings:
            message += " | " + " ".join(bound_warnings)
        record = None
        if save:
            record = repo.add_optimization_run(
                model_experiment_id=first["id"],
                model_experiment_ids={t: m.experiment["id"] for t, m in models.items()},
                algorithm=algo, random_seed=seed, evaluation_budget=budget, evaluations_used=outcome.n_evals,
                objective_function=cfg.kind, objective_config={**cfg.to_dict(), "algorithm_settings": outcome.settings},
                parameter_bounds={f: list(bnds[i]) for i, f in enumerate(evaluator.features)},
                constraints=[c.to_dict() for c in cfg.constraints], baseline_design=base_desc["parameters"],
                baseline_predictions=base_desc["predictions"], candidate_parameters=cand["parameters"],
                candidate_predictions=cand["predictions"], candidate_uncertainty=cand["uncertainty"],
                baseline_objective=base_desc["objective_penalised"], objective_value=cand["objective_penalised"],
                constraint_violations=cand["violations"], feasible=cand["feasible"],
                within_bounds=outcome.within_bounds, dataset_checksum=first.get("dataset_checksum", ""),
                data_source_type=first.get("source_type", ""), status="COMPLETED" if outcome.success else "FINISHED_WITH_WARNING",
                message=message, runtime_seconds=outcome.runtime_s,
            )
        outputs.append((outcome, record, base_desc))
    return outputs
