"""MetaOpt-AI REST API (FastAPI).

Run in development:   uvicorn backend.main:app --reload --port 8000
The React frontend (frontend/) calls these endpoints under /api. When ``frontend/dist`` exists
(after ``npm run build``) the API also serves the built website at ``/``.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from backend import service
from database.repository import records_to_frame
from ml.model_registry import MODEL_SPECS
from ml.prediction import predict_point
from ml.training import TrainingConfig
from optimization.constraints import Constraint
from optimization.objectives import DIRECTION, ObjectiveConfig
from services import workflow
from utils.validation import ValidationError

app = FastAPI(title="MetaOpt-AI API", version=config.APP_VERSION,
              description="Antenna performance prediction and geometry optimization with ML surrogates.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _dataset(dataset_id: int) -> dict:
    try:
        return service.get_dataset(dataset_id)
    except LookupError as exc:
        raise _not_found(exc) from exc


# ---------------------------------------------------------------------------- overview
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": config.APP_VERSION}


@app.get("/api/datasets")
def list_datasets() -> dict:
    return {"datasets": [service.dataset_view(ds) for ds in service.visible_datasets()],
            "unavailable": service.unavailable_sources(),
            "models": [{"key": k, "label": s.label, "description": s.description} for k, s in MODEL_SPECS.items()
                       if k in ("dummy", "random_forest", "gradient_boosting", "extra_trees")],
            "default_models": service.DEFAULT_MODELS}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int) -> dict:
    return service.dataset_view(_dataset(dataset_id))


@app.get("/api/datasets/{dataset_id}/preview")
def preview(dataset_id: int, limit: int = 12) -> dict:
    ds = _dataset(dataset_id)
    df = service.frame(ds)[ds["input_features"] + ds["target_names"]].head(max(1, min(limit, 100)))
    return {"columns": list(df.columns),
            "rows": [[service._clean(v) for v in row] for row in df.itertuples(index=False)]}


@app.get("/api/datasets/{dataset_id}/experiments")
def experiments(dataset_id: int) -> dict:
    ds = _dataset(dataset_id)
    exps = service.repo().list_experiments(dataset_id=ds["id"])
    active_ids = {m.experiment["id"] for m in service.active_models(ds["id"]).values()}
    return {"experiments": [service.experiment_row(e, active_ids) for e in exps]}


# ------------------------------------------------------------------------------ training
class TrainRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: list(service.DEFAULT_MODELS))
    targets: list[str] | None = None
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    cv_folds: int = Field(5, ge=3, le=10)
    seed: int = Field(config.RANDOM_SEED, ge=0, le=100_000)


@app.post("/api/datasets/{dataset_id}/train")
def train(dataset_id: int, req: TrainRequest) -> dict:
    ds = _dataset(dataset_id)
    unknown = [m for m in req.models if m not in MODEL_SPECS]
    if unknown or not req.models:
        raise HTTPException(422, f"Unknown or missing models: {unknown or req.models}")
    targets = req.targets or ds["target_names"]
    t0 = time.perf_counter()
    try:
        cfg = TrainingConfig(features=ds["input_features"], targets=targets, model_keys=req.models,
                             test_size=req.test_size, seed=req.seed, cv_folds=req.cv_folds)
        exps, _ = workflow.train_and_log(service.repo(), ds["id"], cfg, notes="trained from the website")
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    active_ids = {m.experiment["id"] for m in service.active_models(ds["id"]).values()}
    return {"experiments": [service.experiment_row(e, active_ids) for e in exps],
            "seconds": round(time.perf_counter() - t0, 1)}


@app.get("/api/experiments/{experiment_id}/diagnostics")
def diagnostics(experiment_id: int) -> dict:
    try:
        return service.diagnostics(experiment_id)
    except LookupError as exc:
        raise _not_found(exc) from exc


# ---------------------------------------------------------------------------- prediction
class PredictRequest(BaseModel):
    params: dict[str, float]
    save: bool = False


@app.post("/api/datasets/{dataset_id}/predict")
def predict(dataset_id: int, req: PredictRequest) -> dict:
    ds = _dataset(dataset_id)
    models = service.active_models(ds["id"])
    if not models:
        raise HTTPException(409, "No trained models for this dataset. Train them on the Home page.")
    try:
        result = predict_point(models, req.params)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    saved = workflow.save_prediction(service.repo(), models, result)["id"] if req.save else None
    preds = []
    for t in ds["target_names"]:
        if t not in models:
            continue
        e = models[t].experiment
        n_used = (e.get("validation_config") or {}).get("n_samples_used")
        preds.append({"target": t, "short": config.short_label(t), "unit": config.unit_of(t),
                      "value": service._clean(result.predictions[t]),
                      "uncertainty": service._clean(result.uncertainty.get(t)),
                      "model": e["model_type"], "model_label": service.model_label(e["model_type"]),
                      "r2": service._clean(e["r2"]), "rmse": service._clean(e["rmse"]),
                      "partial_training": bool(n_used and n_used < ds["sample_count"]), "n_samples_used": n_used})
    return {"inputs": result.inputs, "predictions": preds, "extrapolation": result.extrapolation,
            "warnings": result.warnings, "saved_id": saved}


# -------------------------------------------------------------------------- optimization
class ConstraintIn(BaseModel):
    target: str
    operator: Literal["<=", ">="]
    limit: float


class OptimizeRequest(BaseModel):
    objective: str
    constraint: ConstraintIn | None = None
    bounds: dict[str, tuple[float, float]]
    baseline: dict[str, float]
    budget: int = Field(config.DEFAULT_EVAL_BUDGET, ge=100, le=20_000)
    seed: int = Field(config.RANDOM_SEED, ge=0, le=100_000)


def _design(d: dict, label: str, targets: list[str]) -> dict:
    return {"label": label, "parameters": d["parameters"],
            "predictions": {t: service._clean(d["predictions"].get(t)) for t in targets},
            "uncertainty": {t: service._clean(d["uncertainty"].get(t)) for t in targets},
            "feasible": d["feasible"], "violations": d["violations"]}


@app.post("/api/datasets/{dataset_id}/optimize")
def optimize(dataset_id: int, req: OptimizeRequest) -> dict:
    ds = _dataset(dataset_id)
    models = service.active_models(ds["id"])
    if not models:
        raise HTTPException(409, "No trained models for this dataset. Train them on the Home page.")
    if req.objective not in models or req.objective not in DIRECTION:
        raise HTTPException(422, f"'{req.objective}' cannot be optimized for this dataset.")
    if not req.bounds:
        raise HTTPException(422, "Choose at least one parameter to optimize.")
    df = service.frame(ds)
    constraints = [Constraint(req.constraint.target, req.constraint.operator, req.constraint.limit)] \
        if req.constraint else []
    cfg = ObjectiveConfig(kind=f"{DIRECTION[req.objective]}_target", target=req.objective, constraints=constraints,
                          normalization=workflow.target_ranges(df, list(models)))
    try:
        outs = workflow.run_optimization(service.repo(), models, cfg, {k: tuple(v) for k, v in req.bounds.items()},
                                         req.baseline, ["differential_evolution"], seed=req.seed, budget=req.budget,
                                         allow_extrapolation=False)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    outcome, record, base = outs[0]
    targets = [t for t in ds["target_names"] if t in models]
    baseline, candidate = _design(base, "Baseline", targets), _design(outcome.candidate, "Candidate", targets)
    features = ds["input_features"]
    synthetic_oracle = None
    if ds["name"] == workflow.SYNTHETIC_DATASET_NAME:
        from datasets import synthetic

        synthetic_oracle = [{"design": d["label"],
                             "S11_dB": float(synthetic.s11_db_clean(d["parameters"]["L_mm"], d["parameters"]["G_mm"])),
                             "Gain_dBi": float(synthetic.gain_dbi_clean(d["parameters"]["L_mm"], d["parameters"]["G_mm"]))}
                            for d in (baseline, candidate)]
    history = outcome.history
    step = max(1, len(history) // 300)
    return {
        "run_id": record["id"], "objective": req.objective, "direction": DIRECTION[req.objective],
        "constraint": req.constraint.model_dump() if req.constraint else None,
        "free_parameters": outcome.candidate.get("free_parameters", list(req.bounds)),
        "fixed_parameters": record["objective_config"].get("fixed_parameters", {}),
        "baseline": baseline, "candidate": candidate,
        "history": [{"evaluation": i + 1, "best": float(v)} for i, v in enumerate(history) if i % step == 0],
        "n_evals": outcome.n_evals, "runtime_s": round(outcome.runtime_s, 3),
        "nearest_simulated": None if synthetic_oracle else service.nearest_simulated(df, features, targets,
                                                                                       [baseline, candidate]),
        "synthetic_oracle": synthetic_oracle,
    }


# ------------------------------------------------------------------------------- results
@app.get("/api/results/summary")
def results_summary() -> dict:
    r = service.repo()
    rows, runs = [], []
    visible = service.visible_datasets()
    exp_to_ds = {}
    for ds in visible:
        latest = {}
        active_ids = {m.experiment["id"] for m in service.active_models(ds["id"]).values()}
        for e in r.list_experiments(dataset_id=ds["id"]):          # newest first
            exp_to_ds[e["id"]] = ds
            latest.setdefault((e["target_variable"], e["model_type"], e["variant"]), e)
        for e in latest.values():
            rows.append({**service.experiment_row(e, active_ids), "dataset_id": ds["id"],
                         "dataset_label": service.dataset_label(ds), "dataset_type": ds["dataset_type"]})
    for run in r.list_optimization_runs(limit=200):
        ds = exp_to_ds.get(run["model_experiment_id"])
        if ds is None:
            continue
        legacy = {"max_gain": "Gain_dBi", "min_s11": "S11_dB"}      # runs saved before target-based objectives
        objective = run["objective_config"].get("target") or legacy.get(run["objective_function"], run["objective_function"])
        runs.append({"id": run["id"], "created_at": run["created_at"], "dataset_label": service.dataset_label(ds),
                     "dataset_id": ds["id"], "algorithm": run["algorithm"], "objective": objective,
                     "objective_short": config.short_label(objective),
                     "direction": DIRECTION.get(objective) or {"Gain_dBi": "maximize"}.get(objective),
                     "unit": config.unit_of(objective),
                     "baseline": run["baseline_design"], "candidate": run["candidate_parameters"],
                     "baseline_value": service._clean(run["baseline_predictions"].get(objective)),
                     "candidate_value": service._clean(run["candidate_predictions"].get(objective)),
                     "constrained": bool(run["constraints"]), "feasible": run["feasible"],
                     "evaluations": run["evaluations_used"]})
    return {"metrics": rows, "optimization_runs": runs,
            "counts": {k: v for k, v in r.counts().items() if k in ("experiments", "predictions", "optimizations")}}


EXPORTS = {"datasets": "list_datasets", "experiments": "list_experiments", "predictions": "list_predictions",
           "optimizations": "list_optimization_runs"}


@app.get("/api/export/{table}.csv")
def export_csv(table: str) -> StreamingResponse:
    if table not in EXPORTS:
        raise HTTPException(404, f"Unknown table '{table}'. Choose one of {list(EXPORTS)}.")
    frame = records_to_frame(getattr(service.repo(), EXPORTS[table])())
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="metaopt_{table}.csv"'})


# ------------------------------------------------------------------ built website (optional)
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        target = DIST / path
        if path and target.is_file() and DIST in target.resolve().parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
