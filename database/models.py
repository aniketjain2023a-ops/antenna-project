"""SQLAlchemy ORM models (schema version 1).

Entity overview
---------------
* ``SchemaVersion``    - applied schema migrations
* ``Dataset``          - one row per registered dataset, with provenance + checksum
* ``AntennaDesign``    - individual geometry/performance rows belonging to a dataset
* ``ModelExperiment``  - one trained/evaluated model (one target per experiment)
* ``PredictionRecord`` - saved interactive predictions
* ``OptimizationRun``  - saved surrogate-based optimization runs
* ``AppSetting``       - persistent key/value settings (e.g. active models)

JSON columns hold variable-length structures (feature lists, hyper-parameters,
predictions) so new geometry parameters can be added without schema changes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    def to_dict(self) -> dict:
        out = {}
        for col in self.__table__.columns:
            value = getattr(self, col.name)
            out[col.name] = value.isoformat(sep=" ", timespec="seconds") if isinstance(value, datetime) else value
        return out


DATASET_TYPE_CHECK = "dataset_type IN ('SYNTHETIC','PUBLIC','SIMULATION','MEASUREMENT')"


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    description: Mapped[str] = mapped_column(String(255))
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("checksum", name="uq_dataset_checksum"),
        CheckConstraint(DATASET_TYPE_CHECK, name="ck_dataset_type"),
        CheckConstraint("sample_count >= 0", name="ck_dataset_sample_count"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(20), nullable=False)
    provenance: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    input_features: Mapped[list] = mapped_column(JSON, nullable=False)
    target_names: Mapped[list] = mapped_column(JSON, nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(500))
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    designs: Mapped[list["AntennaDesign"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )
    experiments: Mapped[list["ModelExperiment"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_type == "SYNTHETIC"


class AntennaDesign(Base):
    __tablename__ = "antenna_designs"
    __table_args__ = (UniqueConstraint("dataset_id", "row_index", name="uq_design_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    row_index: Mapped[int | None] = mapped_column(Integer)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    targets: Mapped[dict] = mapped_column(JSON, default=dict)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(40), default="UNVALIDATED")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    dataset: Mapped[Dataset | None] = relationship(back_populates="designs")


class ModelExperiment(Base):
    __tablename__ = "model_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_group: Mapped[str] = mapped_column(String(64), index=True)
    model_type: Mapped[str] = mapped_column(String(60), nullable=False)
    variant: Mapped[str] = mapped_column(String(20), default="default")  # default | tuned
    target_variable: Mapped[str] = mapped_column(String(100), nullable=False)
    input_features: Mapped[list] = mapped_column(JSON, nullable=False)
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_config: Mapped[dict] = mapped_column(JSON, default=dict)
    r2: Mapped[float | None] = mapped_column(Float)
    mae: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    cv_r2_mean: Mapped[float | None] = mapped_column(Float)
    cv_r2_std: Mapped[float | None] = mapped_column(Float)
    cv_rmse_mean: Mapped[float | None] = mapped_column(Float)
    extra_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    train_ranges: Mapped[dict] = mapped_column(JSON, default=dict)
    n_train: Mapped[int | None] = mapped_column(Integer)
    n_test: Mapped[int | None] = mapped_column(Integer)
    training_seconds: Mapped[float | None] = mapped_column(Float)
    artifact_path: Mapped[str | None] = mapped_column(String(500))
    dataset_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    dataset: Mapped[Dataset] = relationship(back_populates="experiments")


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_experiment_id: Mapped[int] = mapped_column(
        ForeignKey("model_experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_experiment_ids: Mapped[dict] = mapped_column(JSON, default=dict)  # target -> experiment id
    input_parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_outputs: Mapped[dict] = mapped_column(JSON, nullable=False)
    uncertainty: Mapped[dict] = mapped_column(JSON, default=dict)
    extrapolation: Mapped[bool] = mapped_column(Boolean, default=False)
    source_type: Mapped[str] = mapped_column(String(40), default="ML_PREDICTION")
    data_source_type: Mapped[str] = mapped_column(String(20), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_experiment_id: Mapped[int] = mapped_column(
        ForeignKey("model_experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_experiment_ids: Mapped[dict] = mapped_column(JSON, default=dict)
    algorithm: Mapped[str] = mapped_column(String(60), nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_budget: Mapped[int] = mapped_column(Integer)
    evaluations_used: Mapped[int | None] = mapped_column(Integer)
    objective_function: Mapped[str] = mapped_column(String(60), nullable=False)
    objective_config: Mapped[dict] = mapped_column(JSON, default=dict)
    parameter_bounds: Mapped[dict] = mapped_column(JSON, nullable=False)
    constraints: Mapped[list] = mapped_column(JSON, default=list)
    baseline_design: Mapped[dict] = mapped_column(JSON, nullable=False)
    baseline_predictions: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_predictions: Mapped[dict] = mapped_column(JSON, default=dict)
    candidate_uncertainty: Mapped[dict] = mapped_column(JSON, default=dict)
    baseline_objective: Mapped[float | None] = mapped_column(Float)
    objective_value: Mapped[float | None] = mapped_column(Float)
    constraint_violations: Mapped[dict] = mapped_column(JSON, default=dict)
    feasible: Mapped[bool | None] = mapped_column(Boolean)
    within_bounds: Mapped[bool | None] = mapped_column(Boolean)
    dataset_checksum: Mapped[str] = mapped_column(String(64), default="")
    data_source_type: Mapped[str] = mapped_column(String(20), default="")
    status: Mapped[str] = mapped_column(String(20), default="COMPLETED")
    message: Mapped[str] = mapped_column(Text, default="")
    runtime_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AppSetting(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict | list | str | int | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
