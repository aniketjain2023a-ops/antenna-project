"""Data-access layer. The UI and services never touch SQLAlchemy sessions directly.

Every public method opens its own transaction via :func:`session_scope`, and
returns plain dictionaries so callers are decoupled from ORM state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError

import config
from database.db import init_db, make_session_factory, session_scope
from database.models import (
    AntennaDesign,
    AppSetting,
    Dataset,
    ModelExperiment,
    OptimizationRun,
    PredictionRecord,
)
from utils.validation import to_jsonable


class DuplicateRecordError(ValueError):
    """Raised when a record would violate a uniqueness rule."""


class RecordNotFoundError(LookupError):
    """Raised when a referenced record does not exist."""


class Repository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or config.DB_PATH)
        self.engine = init_db(self.db_path)
        self._factory = make_session_factory(self.engine)

    def session(self):
        return session_scope(self._factory)

    # ------------------------------------------------------------------ datasets
    def add_dataset(
        self,
        *,
        name: str,
        dataset_type: str,
        provenance: str,
        description: str,
        input_features: list[str],
        target_names: list[str],
        checksum: str,
        file_path: str | None,
        designs: pd.DataFrame | None = None,
    ) -> dict:
        """Insert a dataset (and optionally its design rows) in one transaction."""
        if dataset_type not in config.DATASET_TYPES:
            raise ValueError(f"Unknown dataset type '{dataset_type}'.")
        if not provenance or not provenance.strip():
            raise ValueError("Provenance is mandatory.")
        if not input_features:
            raise ValueError("At least one input feature is required.")
        existing = self.find_dataset_by_checksum(checksum)
        if existing:
            raise DuplicateRecordError(
                f"Identical data already registered as dataset #{existing['id']} ('{existing['name']}')."
            )
        if self.get_dataset_by_name(name):
            raise DuplicateRecordError(f"A dataset named '{name}' already exists.")
        sample_count = 0 if designs is None else int(len(designs))
        try:
            with self.session() as s:
                ds = Dataset(
                    name=name.strip(),
                    dataset_type=dataset_type,
                    provenance=provenance.strip(),
                    description=description or "",
                    sample_count=sample_count,
                    input_features=list(input_features),
                    target_names=list(target_names),
                    file_path=file_path,
                    checksum=checksum,
                )
                s.add(ds)
                s.flush()
                if designs is not None and len(designs):
                    status = "SYNTHETIC_UNVALIDATED" if dataset_type == "SYNTHETIC" else "AS_PROVIDED"
                    records = []
                    for idx, row in enumerate(designs.to_dict(orient="records")):
                        records.append(
                            AntennaDesign(
                                dataset_id=ds.id,
                                row_index=idx,
                                parameters=to_jsonable({f: row.get(f) for f in input_features}),
                                targets=to_jsonable({t: row.get(t) for t in target_names}),
                                source_type=dataset_type,
                                validation_status=status,
                            )
                        )
                    s.add_all(records)
                return ds.to_dict()
        except IntegrityError as exc:
            raise DuplicateRecordError(f"Dataset violates a uniqueness/integrity rule: {exc.orig}") from exc

    def get_dataset(self, dataset_id: int) -> dict | None:
        with self.session() as s:
            ds = s.get(Dataset, dataset_id)
            return ds.to_dict() if ds else None

    def get_dataset_by_name(self, name: str) -> dict | None:
        with self.session() as s:
            ds = s.execute(select(Dataset).where(Dataset.name == name.strip())).scalar_one_or_none()
            return ds.to_dict() if ds else None

    def find_dataset_by_checksum(self, checksum: str) -> dict | None:
        with self.session() as s:
            ds = s.execute(select(Dataset).where(Dataset.checksum == checksum)).scalar_one_or_none()
            return ds.to_dict() if ds else None

    def list_datasets(self) -> list[dict]:
        with self.session() as s:
            return [d.to_dict() for d in s.execute(select(Dataset).order_by(Dataset.id)).scalars()]

    def delete_dataset(self, dataset_id: int) -> None:
        with self.session() as s:
            ds = s.get(Dataset, dataset_id)
            if ds is None:
                raise RecordNotFoundError(f"Dataset #{dataset_id} not found.")
            s.delete(ds)

    def get_designs(self, dataset_id: int, limit: int | None = None) -> list[dict]:
        with self.session() as s:
            q = select(AntennaDesign).where(AntennaDesign.dataset_id == dataset_id).order_by(AntennaDesign.row_index)
            if limit:
                q = q.limit(limit)
            return [d.to_dict() for d in s.execute(q).scalars()]

    def count_designs(self, dataset_id: int | None = None) -> int:
        with self.session() as s:
            q = select(func.count(AntennaDesign.id))
            if dataset_id is not None:
                q = q.where(AntennaDesign.dataset_id == dataset_id)
            return int(s.execute(q).scalar_one())

    # --------------------------------------------------------------- experiments
    def add_experiment(self, **fields: Any) -> dict:
        if self.get_dataset(fields["dataset_id"]) is None:
            raise RecordNotFoundError(f"Dataset #{fields['dataset_id']} not found.")
        fields = {k: to_jsonable(v) if isinstance(v, (dict, list)) else v for k, v in fields.items()}
        with self.session() as s:
            exp = ModelExperiment(**fields)
            s.add(exp)
            s.flush()
            return exp.to_dict()

    def update_experiment(self, experiment_id: int, **fields: Any) -> dict:
        with self.session() as s:
            exp = s.get(ModelExperiment, experiment_id)
            if exp is None:
                raise RecordNotFoundError(f"Experiment #{experiment_id} not found.")
            for k, v in fields.items():
                setattr(exp, k, to_jsonable(v) if isinstance(v, (dict, list)) else v)
            s.flush()
            return exp.to_dict()

    def get_experiment(self, experiment_id: int) -> dict | None:
        with self.session() as s:
            exp = s.get(ModelExperiment, experiment_id)
            return exp.to_dict() if exp else None

    def list_experiments(
        self,
        dataset_id: int | None = None,
        target: str | None = None,
        model_type: str | None = None,
        search: str | None = None,
    ) -> list[dict]:
        with self.session() as s:
            q = select(ModelExperiment).order_by(ModelExperiment.id.desc())
            if dataset_id is not None:
                q = q.where(ModelExperiment.dataset_id == dataset_id)
            if target:
                q = q.where(ModelExperiment.target_variable == target)
            if model_type:
                q = q.where(ModelExperiment.model_type == model_type)
            if search:
                like = f"%{search}%"
                q = q.where(or_(ModelExperiment.model_type.like(like),
                                ModelExperiment.target_variable.like(like),
                                ModelExperiment.notes.like(like),
                                ModelExperiment.run_group.like(like)))
            return [e.to_dict() for e in s.execute(q).scalars()]

    # --------------------------------------------------------------- predictions
    def add_prediction(self, **fields: Any) -> dict:
        if self.get_experiment(fields["model_experiment_id"]) is None:
            raise RecordNotFoundError(f"Experiment #{fields['model_experiment_id']} not found.")
        fields = {k: to_jsonable(v) if isinstance(v, (dict, list)) else v for k, v in fields.items()}
        with self.session() as s:
            rec = PredictionRecord(**fields)
            s.add(rec)
            s.flush()
            return rec.to_dict()

    def list_predictions(self, limit: int | None = None) -> list[dict]:
        with self.session() as s:
            q = select(PredictionRecord).order_by(PredictionRecord.id.desc())
            if limit:
                q = q.limit(limit)
            return [p.to_dict() for p in s.execute(q).scalars()]

    # ------------------------------------------------------------- optimizations
    def add_optimization_run(self, **fields: Any) -> dict:
        if self.get_experiment(fields["model_experiment_id"]) is None:
            raise RecordNotFoundError(f"Experiment #{fields['model_experiment_id']} not found.")
        fields = {k: to_jsonable(v) if isinstance(v, (dict, list)) else v for k, v in fields.items()}
        with self.session() as s:
            run = OptimizationRun(**fields)
            s.add(run)
            s.flush()
            return run.to_dict()

    def get_optimization_run(self, run_id: int) -> dict | None:
        with self.session() as s:
            run = s.get(OptimizationRun, run_id)
            return run.to_dict() if run else None

    def list_optimization_runs(self, limit: int | None = None, algorithm: str | None = None) -> list[dict]:
        with self.session() as s:
            q = select(OptimizationRun).order_by(OptimizationRun.id.desc())
            if algorithm:
                q = q.where(OptimizationRun.algorithm == algorithm)
            if limit:
                q = q.limit(limit)
            return [r.to_dict() for r in s.execute(q).scalars()]

    # ------------------------------------------------------------------ settings
    def set_setting(self, key: str, value: Any) -> None:
        with self.session() as s:
            row = s.get(AppSetting, key)
            if row is None:
                s.add(AppSetting(key=key, value=to_jsonable(value)))
            else:
                row.value = to_jsonable(value)

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.session() as s:
            row = s.get(AppSetting, key)
            return default if row is None else row.value

    # ------------------------------------------------------------------- summary
    def counts(self) -> dict[str, int]:
        with self.session() as s:
            return {
                "datasets": int(s.execute(select(func.count(Dataset.id))).scalar_one()),
                "designs": int(s.execute(select(func.count(AntennaDesign.id))).scalar_one()),
                "experiments": int(s.execute(select(func.count(ModelExperiment.id))).scalar_one()),
                "predictions": int(s.execute(select(func.count(PredictionRecord.id))).scalar_one()),
                "optimizations": int(s.execute(select(func.count(OptimizationRun.id))).scalar_one()),
            }

    def recent_activity(self, limit: int = 10) -> pd.DataFrame:
        rows: list[dict] = []
        for d in self.list_datasets():
            rows.append({"timestamp": d["created_at"], "type": "Dataset registered",
                         "details": f"#{d['id']} {d['name']} [{d['dataset_type']}]"})
        for e in self.list_experiments()[:limit]:
            rows.append({"timestamp": e["created_at"], "type": "Model trained",
                         "details": f"#{e['id']} {e['model_type']} ({e['variant']}) -> {e['target_variable']}, "
                                    f"test R2={_fmt(e['r2'])}"})
        for p in self.list_predictions(limit):
            rows.append({"timestamp": p["created_at"], "type": "Prediction saved",
                         "details": f"#{p['id']} inputs={p['input_parameters']}"})
        for r in self.list_optimization_runs(limit):
            rows.append({"timestamp": r["created_at"], "type": "Optimization run",
                         "details": f"#{r['id']} {r['algorithm']} / {r['objective_function']} "
                                    f"feasible={r['feasible']}"})
        if not rows:
            return pd.DataFrame(columns=["timestamp", "type", "details"])
        return pd.DataFrame(rows).sort_values("timestamp", ascending=False).head(limit).reset_index(drop=True)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Flatten JSON columns into strings so tables display/export cleanly."""
    import json

    df = pd.DataFrame(list(records))
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (dict, list))).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    return df
