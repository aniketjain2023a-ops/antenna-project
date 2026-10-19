"""Shared fixtures. Every test session uses its own temporary database and folders,
so the real application database in ``data/`` is never touched."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = Path(tempfile.mkdtemp(prefix="metaopt_tests_"))
os.environ.setdefault("METAOPT_DB_PATH", str(_TMP / "default.db"))
os.environ.setdefault("METAOPT_MODELS_DIR", str(_TMP / "models"))
os.environ.setdefault("METAOPT_REPORTS_DIR", str(_TMP / "reports"))
os.environ.setdefault("METAOPT_UPLOAD_DIR", str(_TMP / "uploads"))

import pytest  # noqa: E402

from database.db import dispose_engine  # noqa: E402
from database.repository import Repository  # noqa: E402
from ml.training import TrainingConfig  # noqa: E402
from services import workflow  # noqa: E402


@pytest.fixture()
def repo(tmp_path):
    db = tmp_path / "test.db"
    r = Repository(db)
    yield r
    dispose_engine(db)


@pytest.fixture()
def demo_dataset(repo, tmp_path):
    return workflow.ensure_demo_dataset(repo, csv_path=tmp_path / "synthetic_demo.csv")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Repository with the demo dataset and quick RF / ET / Dummy models (module-scoped for speed)."""
    base = tmp_path_factory.mktemp("trained")
    r = Repository(base / "trained.db")
    ds = workflow.ensure_demo_dataset(r, csv_path=base / "synthetic_demo.csv")
    cfg = TrainingConfig(features=["L_mm", "G_mm"], targets=["S11_dB", "Gain_dBi"],
                         model_keys=["dummy", "random_forest", "extra_trees"], cv_folds=3)
    exps, results = workflow.train_and_log(r, ds["id"], cfg, models_dir=base / "models")
    models = workflow.get_active_models(r, ds["id"])
    yield {"repo": r, "dataset": ds, "experiments": exps, "results": results, "models": models, "base": base}
    dispose_engine(base / "trained.db")
