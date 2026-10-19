"""Synthetic dataset, provenance, data validation and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from database.repository import DuplicateRecordError
from datasets import synthetic
from datasets.io import read_table
from ml.preprocessing import clean_dataframe, detect_column_types, inspect_quality, xy_for_target
from services import workflow
from utils.validation import ValidationError, file_checksum, validate_bounds, validate_geometry


def test_synthetic_dataset_is_deterministic_and_large_enough():
    a = synthetic.generate_synthetic_dataset(500, seed=42)
    b = synthetic.generate_synthetic_dataset(500, seed=42)
    c = synthetic.generate_synthetic_dataset(500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert len(a) >= 400
    assert set(synthetic.FEATURES + synthetic.TARGETS) <= set(a.columns)


def test_synthetic_values_are_physically_plausible():
    df = synthetic.generate_synthetic_dataset(500, seed=42)
    assert (df["S11_dB"] <= -0.05 + 1e-9).all()             # passive: |Gamma| <= 1
    assert df["L_mm"].between(*synthetic.L_RANGE_MM).all()
    assert df["G_mm"].between(*synthetic.G_RANGE_MM).all()
    assert (df["S11_dB"] <= -10).sum() > 20                    # resonance region is sampled
    assert df["Gain_dBi"].max() < 10                           # no unrealistic gains
    # predictions not artificially perfect: noise present
    resid = df["S11_dB"] - synthetic.s11_db_clean(df["L_mm"], df["G_mm"])
    assert resid.std() > 0.1


def test_synthetic_labeling_and_provenance(repo, demo_dataset):
    ds = demo_dataset
    assert ds["dataset_type"] == "SYNTHETIC"
    assert "SYNTHETIC" in ds["provenance"] and "NOT a measurement" in ds["provenance"]
    assert ds["sample_count"] == 500
    designs = repo.get_designs(ds["id"], limit=3)
    assert all(d["source_type"] == "SYNTHETIC" for d in designs)
    assert all(d["validation_status"] == "SYNTHETIC_UNVALIDATED" for d in designs)


def test_demo_dataset_registration_is_idempotent_and_checksummed(repo, tmp_path):
    csv = tmp_path / "demo.csv"
    first = workflow.ensure_demo_dataset(repo, csv_path=csv)
    second = workflow.ensure_demo_dataset(repo, csv_path=csv)
    assert first["id"] == second["id"]
    assert first["checksum"] == file_checksum(csv)
    df, ds, warnings = workflow.load_dataset_frame(repo, first["id"])
    assert len(df) == 500 and not warnings


def test_checksum_mismatch_detected(repo, tmp_path):
    csv = tmp_path / "demo.csv"
    ds = workflow.ensure_demo_dataset(repo, csv_path=csv)
    with open(csv, "a") as fh:
        fh.write("999,10,1,-5,5,SYNTHETIC\n")
    _, _, warnings = workflow.load_dataset_frame(repo, ds["id"])
    assert any("Checksum mismatch" in w for w in warnings)


def test_missing_value_detection_and_no_imputation():
    df = pd.DataFrame({"L_mm": [10, 11, np.nan, 12], "G_mm": [1, 1, 1, 1.5],
                       "S11_dB": [-12, np.nan, -5, -3], "Gain_dBi": [7, 6, np.nan, np.nan]})
    report = inspect_quality(df)
    assert report.missing_per_column == {"L_mm": 1, "G_mm": 0, "S11_dB": 1, "Gain_dBi": 2}
    original = df.copy()
    clean, summary = clean_dataframe(df, ["L_mm", "G_mm"], ["S11_dB", "Gain_dBi"])
    pd.testing.assert_frame_equal(df, original)              # source never modified
    assert len(clean) == 3                                    # row with missing feature dropped
    assert clean["S11_dB"].isna().sum() == 1                  # missing target kept missing (no imputation)
    assert any("no imputation" in a for a in summary.actions)
    X, y, _ = xy_for_target(clean, ["L_mm", "G_mm"], "Gain_dBi")
    assert len(y) == 2 and not np.isnan(y).any()


def test_duplicates_and_outliers_flagged():
    df = pd.DataFrame({"L_mm": [10, 10, 11, 12, 13], "G_mm": [1, 1, 1, 1, 1], "S11_dB": [-1, -1, -2, -3, -80]})
    report = inspect_quality(df)
    assert report.duplicate_rows == 1
    assert report.outliers_per_column["S11_dB"] == 1
    assert "G_mm" in report.constant_columns
    clean, _ = clean_dataframe(df, ["L_mm", "G_mm"], ["S11_dB"])
    assert len(clean) == 4 and (clean["S11_dB"] == -80).any()   # outlier not removed


def test_column_type_detection_and_coercion():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["1.5", "2.5"], "c": ["x", "y"], "d": [None, None]})
    kinds = detect_column_types(df)
    assert kinds == {"a": "numeric", "b": "numeric-like text", "c": "categorical/text", "d": "empty"}
    clean, _ = clean_dataframe(df, ["a", "b"], [], drop_rows_without_any_target=False)
    assert clean["b"].dtype.kind == "f"


def test_cleaning_rejects_bad_selection():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        clean_dataframe(df, [], ["b"])
    with pytest.raises(ValueError):
        clean_dataframe(df, ["a"], ["a"])
    with pytest.raises(ValueError):
        clean_dataframe(df, ["zzz"], ["b"])


def test_geometry_validation():
    ranges = {"L_mm": (8.0, 12.5), "G_mm": (0.2, 2.0)}
    assert validate_geometry({"L_mm": 10, "G_mm": 1}, ["L_mm", "G_mm"], ranges).ok
    assert not validate_geometry({"L_mm": -1, "G_mm": 1}, ["L_mm", "G_mm"]).ok
    assert not validate_geometry({"L_mm": float("nan"), "G_mm": 1}, ["L_mm", "G_mm"]).ok
    assert not validate_geometry({"L_mm": "abc", "G_mm": 1}, ["L_mm", "G_mm"]).ok
    assert not validate_geometry({"L_mm": 10}, ["L_mm", "G_mm"]).ok
    res = validate_geometry({"L_mm": 14, "G_mm": 1}, ["L_mm", "G_mm"], ranges)
    assert res.ok and any("EXTRAPOLATION" in w for w in res.warnings)
    with pytest.raises(ValidationError):
        validate_geometry({"L_mm": 0}, ["L_mm"]).raise_if_invalid()


def test_invalid_parameter_bounds():
    ranges = {"L_mm": (8.0, 12.5)}
    assert not validate_bounds({"L_mm": (12.0, 9.0)}).ok
    assert not validate_bounds({"L_mm": (9.0, 9.0)}).ok
    assert not validate_bounds({"L_mm": (float("nan"), 9.0)}).ok
    assert not validate_bounds({"L_mm": (-1.0, 9.0)}).ok
    assert not validate_bounds({}).ok
    assert not validate_bounds({"L_mm": (7.0, 12.0)}, ranges).ok                    # extrapolation = error
    lenient = validate_bounds({"L_mm": (7.0, 12.0)}, ranges, allow_extrapolation=True)
    assert lenient.ok and lenient.warnings                                       # ... unless explicitly allowed


def test_read_csv_and_excel(tmp_path):
    df = pd.DataFrame({" L_mm ": [10.0, 11.0], "S11_dB": [-12.0, -3.0]})
    df.to_csv(tmp_path / "x.csv", index=False)
    df.to_excel(tmp_path / "x.xlsx", index=False)
    for name in ("x.csv", "x.xlsx"):
        got = read_table(tmp_path / name)
        assert list(got.columns) == ["L_mm", "S11_dB"] and len(got) == 2
    with pytest.raises(ValueError):
        read_table(tmp_path / "x.txt")


def test_register_uploaded_dataset_with_missing_target(repo, tmp_path):
    df = pd.DataFrame({"L_mm": [9.0, 10.0, 11.0], "G_mm": [0.5, 1.0, 1.5], "S11_dB": [-4.0, -15.0, -3.0]})
    ds = workflow.register_dataframe(repo, df, name="EM sweep", dataset_type="SIMULATION",
                                     provenance="unit-test EM export", description="", features=["L_mm", "G_mm"],
                                     targets=["S11_dB"], upload_dir=tmp_path)
    assert ds["dataset_type"] == "SIMULATION" and ds["target_names"] == ["S11_dB"]
    assert "Gain_dBi" not in ds["target_names"]          # no target is invented
    with pytest.raises(DuplicateRecordError):
        workflow.register_dataframe(repo, df, name="EM sweep 2", dataset_type="SIMULATION", provenance="x",
                                    description="", features=["L_mm", "G_mm"], targets=["S11_dB"], upload_dir=tmp_path)
    with pytest.raises(ValueError):
        workflow.register_dataframe(repo, df.assign(L_mm=["a", "b", "c"]), name="bad", dataset_type="SIMULATION",
                                    provenance="x", description="", features=["L_mm", "G_mm"], targets=["S11_dB"],
                                    upload_dir=tmp_path)
