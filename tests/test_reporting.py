"""PDF and CSV report generation."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from optimization.constraints import Constraint
from optimization.objectives import ObjectiveConfig
from reporting.report_generator import export_csv_bundle, generate_pdf_report
from services import workflow


def _pdf_text(path) -> str:
    pypdf = pytest.importorskip("pypdf")
    return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(path)).pages)


def test_pdf_report_full_workflow(trained, tmp_path):
    repo, ds = trained["repo"], trained["dataset"]
    df, _, _ = workflow.load_dataset_frame(repo, ds["id"])
    cfg = ObjectiveConfig("max_gain", [Constraint("S11_dB", "<=", -10.0)], normalization=workflow.target_ranges(df, ["S11_dB", "Gain_dBi"]))
    workflow.run_optimization(repo, trained["models"], cfg, workflow.model_training_ranges(trained["models"]),
                              {"L_mm": 11.0, "G_mm": 0.6}, ["differential_evolution"], budget=300)
    path = generate_pdf_report(repo, ds["id"], output_dir=tmp_path)
    assert path.exists() and path.read_bytes()[:4] == b"%PDF" and path.stat().st_size > 20_000
    text = _pdf_text(path)
    for phrase in ("SYNTHETIC", "Dataset and provenance", "Model configurations", "Optimization results",
                   "Limitations", "Reproducibility", "NOT physically validated", ds["checksum"][:16]):
        assert phrase in text, phrase
    assert list(tmp_path.glob("figures_*/*.png"))


def test_pdf_report_without_models_or_gain_target(repo, tmp_path):
    data = pd.DataFrame({"L_mm": [9.0 + 0.1 * i for i in range(30)], "G_mm": [1.0] * 30,
                         "S11_dB": [-3.0 - (i % 7) for i in range(30)]})
    ds = workflow.register_dataframe(repo, data, name="S11 only", dataset_type="SIMULATION", provenance="unit test",
                                     description="", features=["L_mm", "G_mm"], targets=["S11_dB"], upload_dir=tmp_path)
    path = generate_pdf_report(repo, ds["id"], output_dir=tmp_path)
    text = _pdf_text(path)
    assert "No models have been trained" in text and "SIMULATION" in text
    assert "SYNTHETIC DATA:" not in text


def test_csv_export_bundle(trained, tmp_path):
    data, paths = export_csv_bundle(trained["repo"], tmp_path)
    assert len(paths) == 4 and all(p.exists() for p in paths)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert {"datasets.csv", "model_experiments.csv", "predictions.csv", "optimization_runs.csv"} <= names
        exps = pd.read_csv(zf.open("model_experiments.csv"))
    assert len(exps) == len(trained["repo"].list_experiments())
    assert (exps["source_type"] == "SYNTHETIC").all()
