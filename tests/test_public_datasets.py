"""Public-dataset pipelines: target derivation, censoring, status gating, failure reporting, fixed-parameter optimization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config
from datasets import public
from datasets.public import metu_simpl, zenodo
from optimization.constraints import Constraint
from optimization.differential_evolution import run_differential_evolution
from optimization.objectives import ObjectiveConfig, SurrogateEvaluator

F = metu_simpl.FREQUENCIES_GHZ


def _curve(center_ghz: float, depth_db: float, width_ghz: float) -> np.ndarray:
    """A smooth resonance dip: 0 dB far away, ``depth_db`` at ``center_ghz``."""
    return depth_db * np.exp(-((F - center_ghz) / width_ghz) ** 2)


def test_frequency_axis_matches_hfss_sweep():
    assert len(F) == 201 and F[0] == 23.0 and F[-1] == 33.0
    assert np.allclose(np.diff(F), 0.05)
    assert np.isclose(F, 28.0).sum() == 1


def test_target_derivation_and_censoring():
    db = np.vstack([
        _curve(28.0, -20.0, 0.5),    # interior resonance at 28 GHz, deep enough for a -10 dB band
        _curve(35.0, -30.0, 2.0),    # true resonance above the band: minimum sits on the 33 GHz edge
        _curve(26.0, -6.0, 0.5),     # interior resonance that never reaches -10 dB
    ])
    t = metu_simpl.derive_targets(db)
    assert t.loc[0, "S11_28GHz_dB"] == pytest.approx(-20.0)
    assert t.loc[0, "f_res_GHz"] == pytest.approx(28.0)
    # -10 dB where exp(-(df/0.5)^2) >= 0.5  ->  |df| <= 0.5*sqrt(ln 2) = 0.416 GHz, sampled on a 50 MHz grid
    assert t.loc[0, "BW10_MHz"] == pytest.approx(800.0)
    assert np.isnan(t.loc[1, "f_res_GHz"])                 # censored: resonance outside the band
    assert t.loc[1, "S11_min_dB"] == pytest.approx(db[1].min())
    assert np.isnan(t.loc[1, "BW10_MHz"])                  # -10 dB region reaches the band edge
    assert t.loc[2, "f_res_GHz"] == pytest.approx(26.0)
    assert np.isnan(t.loc[2, "BW10_MHz"])                  # never reaches -10 dB: no bandwidth invented


def test_schema_metadata_complete():
    assert len(metu_simpl.COLUMN_MAP) == 11 and len(set(metu_simpl.FEATURES)) == 11
    for c in metu_simpl.FEATURES + metu_simpl.TARGETS:
        assert c in config.KNOWN_COLUMNS, c
        assert config.KNOWN_COLUMNS[c].get("short")
    assert config.unit_of("feed_position_ratio") == ""        # dimensionless (c_probe)
    assert config.unit_of("S11_28GHz_dB") == "dB" and config.unit_of("f_res_GHz") == "GHz"
    assert "SIMULATION" in metu_simpl.PROVENANCE
    assert "HFSS" in metu_simpl.PROVENANCE and "10.1007/s00521-024-10860-5" in metu_simpl.PROVENANCE


def test_pipeline_status_gates_availability(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_PROCESSED_DIR", tmp_path)
    assert public.ready_sources() == {}
    public.write_stage("metu_simpl", "download", True)
    public.write_stage("metu_simpl", "preprocess", True)
    assert "metu_simpl" not in public.ready_sources()          # training not done yet
    public.write_stage("metu_simpl", "train", True)
    assert "metu_simpl" in public.ready_sources()
    public.write_stage("zenodo", "download", False, "blocked")
    st = public.read_status("zenodo")
    assert not st["complete"] and st["message"] == "blocked" and "zenodo" not in public.ready_sources()


def test_zenodo_unreachable_is_reported_not_faked(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("Tunnel connection failed: 403 Forbidden")
    monkeypatch.setattr(zenodo, "_get", boom)
    with pytest.raises(public.DatasetUnavailable, match="403"):
        zenodo.download(tmp_path)
    assert not any(tmp_path.iterdir())                        # nothing pretends to be downloaded


def test_optimizer_holds_fixed_parameters(trained):
    models = trained["models"]
    df = pd.read_csv(trained["base"] / "synthetic_demo.csv")
    norm = {t: (float(df[t].min()), float(df[t].max())) for t in models}
    ev = SurrogateEvaluator(models, fixed={"G_mm": 1.0})
    assert ev.features == ["L_mm"] and ev.all_features == ["L_mm", "G_mm"]
    cfg = ObjectiveConfig("maximize_target", target="Gain_dBi", constraints=[Constraint("S11_dB", "<=", -10.0)],
                          normalization=norm)
    out = run_differential_evolution(ev, cfg, {"L_mm": (8.5, 12.0)}, seed=0, budget=300)
    assert out.candidate["parameters"]["G_mm"] == 1.0          # held, not optimised
    assert 8.5 <= out.candidate["parameters"]["L_mm"] <= 12.0
    assert out.candidate["free_parameters"] == ["L_mm"]
    with pytest.raises(ValueError):
        SurrogateEvaluator(models, fixed={"L_mm": 10.0, "G_mm": 1.0})   # nothing left to optimize


RAW = config.PUBLIC_RAW_DIR / "metu_simpl"


@pytest.mark.skipif(not (RAW / "MANIFEST.json").exists(), reason="METU-SIMPL raw data not downloaded "
                    "(run scripts/prepare_public_datasets.py)")
def test_real_metu_files_verify_and_preprocess(tmp_path):
    manifest = metu_simpl.verify()                                # checksums of untouched originals
    assert set(manifest["files"]) == set(metu_simpl.FILES)
    summary = metu_simpl.preprocess(dest=tmp_path)
    assert summary["rows_raw"] == 5000 and summary["rows_processed"] == 5000
    table = pd.read_csv(tmp_path / "metu_simpl_processed.csv")
    assert list(table.columns) == ["design_id"] + metu_simpl.FEATURES + metu_simpl.TARGETS
    assert table["S11_28GHz_dB"].notna().all() and (table["S11_min_dB"] <= 0).all()
    fres = table["f_res_GHz"].dropna()
    assert fres.between(23.05, 32.95).all()                        # censored values never sit on a band edge
    assert summary["target_non_missing"]["f_res_GHz"] < 5000
    assert json.loads((RAW / "MANIFEST.json").read_text())["license_files_found"] == []
