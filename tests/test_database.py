"""Database initialization, CRUD, integrity, transactions and persistence."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from database.db import dispose_engine, make_session_factory, session_scope
from database.migrations import CURRENT_SCHEMA_VERSION, current_version, run_migrations
from database.models import Dataset
from database.repository import DuplicateRecordError, RecordNotFoundError, Repository


def _add(repo, name="ds", checksum="abc", dataset_type="SIMULATION", designs=None):
    return repo.add_dataset(name=name, dataset_type=dataset_type, provenance="unit test", description="",
                            input_features=["L_mm", "G_mm"], target_names=["S11_dB"], checksum=checksum,
                            file_path=None, designs=designs)


def test_initialization_creates_all_tables_and_schema_version(repo):
    tables = set(inspect(repo.engine).get_table_names())
    assert {"datasets", "antenna_designs", "model_experiments", "prediction_records", "optimization_runs",
            "app_settings", "schema_version"} <= tables
    assert current_version(repo.engine) == CURRENT_SCHEMA_VERSION


def test_migrations_are_idempotent(repo):
    assert run_migrations(repo.engine) == CURRENT_SCHEMA_VERSION
    with repo.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM schema_version")).scalar_one() == 1


def test_foreign_keys_enforced(repo):
    with repo.engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1


def test_dataset_create_and_retrieve_with_designs(repo):
    designs = pd.DataFrame({"L_mm": [10.0, 11.0], "G_mm": [1.0, 0.5], "S11_dB": [-12.0, None]})
    ds = _add(repo, designs=designs)
    assert ds["sample_count"] == 2
    got = repo.get_dataset(ds["id"])
    assert got["provenance"] == "unit test" and got["input_features"] == ["L_mm", "G_mm"]
    rows = repo.get_designs(ds["id"])
    assert len(rows) == 2 and rows[0]["parameters"] == {"L_mm": 10.0, "G_mm": 1.0}
    assert rows[1]["targets"]["S11_dB"] is None  # missing value stored as missing, never fabricated
    assert rows[0]["validation_status"] == "AS_PROVIDED"


def test_duplicate_checksum_and_name_rejected(repo):
    _add(repo, name="a", checksum="same")
    with pytest.raises(DuplicateRecordError):
        _add(repo, name="b", checksum="same")
    with pytest.raises(DuplicateRecordError):
        _add(repo, name="a", checksum="other")


def test_invalid_dataset_type_and_missing_provenance_rejected(repo):
    with pytest.raises(ValueError):
        _add(repo, dataset_type="MADE_UP")
    with pytest.raises(ValueError):
        repo.add_dataset(name="x", dataset_type="SYNTHETIC", provenance="  ", description="",
                         input_features=["L_mm"], target_names=[], checksum="z", file_path=None)


def test_check_constraint_at_database_level(repo):
    factory = make_session_factory(repo.engine)
    with pytest.raises(IntegrityError):
        with session_scope(factory) as s:
            s.add(Dataset(name="bad", dataset_type="FAKE", provenance="p", sample_count=0, input_features=[],
                          target_names=[], checksum="q"))


def test_referential_integrity(repo):
    with pytest.raises(RecordNotFoundError):
        repo.add_experiment(dataset_id=999, run_group="g", model_type="random_forest", target_variable="S11_dB",
                            input_features=["L_mm"], random_seed=1, dataset_checksum="x", source_type="SIMULATION")
    with pytest.raises(RecordNotFoundError):
        repo.add_prediction(model_experiment_id=999, input_parameters={}, predicted_outputs={})
    with pytest.raises(RecordNotFoundError):
        repo.add_optimization_run(model_experiment_id=999, algorithm="x", random_seed=1, objective_function="f",
                                  parameter_bounds={}, baseline_design={})


def test_cascade_delete_removes_dependent_rows(repo):
    ds = _add(repo, designs=pd.DataFrame({"L_mm": [1.0], "G_mm": [1.0], "S11_dB": [-3.0]}))
    exp = repo.add_experiment(dataset_id=ds["id"], run_group="g", model_type="dummy", target_variable="S11_dB",
                              input_features=["L_mm", "G_mm"], random_seed=1, dataset_checksum="abc",
                              source_type="SIMULATION")
    repo.add_prediction(model_experiment_id=exp["id"], input_parameters={"L_mm": 1}, predicted_outputs={"S11_dB": -3})
    repo.delete_dataset(ds["id"])
    assert repo.counts() == {"datasets": 0, "designs": 0, "experiments": 0, "predictions": 0, "optimizations": 0}


def test_transaction_rollback_on_error(repo):
    factory = make_session_factory(repo.engine)
    with pytest.raises(RuntimeError):
        with session_scope(factory) as s:
            s.add(Dataset(name="tx", dataset_type="SIMULATION", provenance="p", sample_count=0, input_features=["a"],
                          target_names=[], checksum="tx"))
            s.flush()
            raise RuntimeError("boom")
    assert repo.get_dataset_by_name("tx") is None


def test_persistence_across_restart(tmp_path):
    db = tmp_path / "persist.db"
    r1 = Repository(db)
    ds = _add(r1, name="persist", checksum="p1")
    r1.set_setting("active_dataset_id", ds["id"])
    dispose_engine(db)                       # simulate application shutdown
    r2 = Repository(db)                      # restart
    assert r2.get_dataset_by_name("persist")["id"] == ds["id"]
    assert r2.get_setting("active_dataset_id") == ds["id"]
    dispose_engine(db)


def test_experiment_history_filters(repo):
    ds = _add(repo)
    for model, target in [("random_forest", "S11_dB"), ("extra_trees", "Gain_dBi")]:
        repo.add_experiment(dataset_id=ds["id"], run_group="g1", model_type=model, target_variable=target,
                            input_features=["L_mm"], random_seed=1, dataset_checksum="abc", source_type="SIMULATION")
    assert len(repo.list_experiments(dataset_id=ds["id"])) == 2
    assert len(repo.list_experiments(target="Gain_dBi")) == 1
    assert len(repo.list_experiments(search="random")) == 1
    assert not repo.recent_activity().empty
