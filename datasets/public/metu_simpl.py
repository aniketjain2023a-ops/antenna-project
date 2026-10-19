"""METU-SIMPL Microstrip-Patch dataset (https://github.com/METU-SIMPL/Microstrip-Patch).

Verified facts (inspected 2026-10-19, commit 7113818a)
------------------------------------------------------
* Content: 5,000 probe-fed microstrip patch designs simulated with Ansys HFSS (full-wave EM
  SIMULATION, not measurement). The HFSS project ``antenna_model.aedt`` is included.
* Inputs (``input_parameters.xlsx``, 11 columns): patch length / width / copper thickness,
  substrate and solder-resist thickness, probe radius, pad and antipad clearances (mm), the
  probe position ratio ``c_probe`` (dimensionless, d_probe = c_probe × l_patch) and two relative
  permittivities. Units and meanings come from the ``VariableProp`` definitions in the .aedt file.
* Outputs (``S11_re.xlsx``, ``S11_im.xlsx``): real and imaginary S11 at 201 points. The sweep in
  the .aedt file is 23 GHz to 33 GHz in 0.05 GHz steps (201 points), matching the 201 columns.
* License: the repository contains NO license file. The README asks users to cite
  Saçın & Durgun, Neural Computing and Applications 37, 3827–3837 (2025),
  doi:10.1007/s00521-024-10860-5, and acknowledge the authors. The raw and processed files are
  therefore downloaded locally and NOT committed to this project's repository.

Derived targets (all from the simulated |S11| curve, |S11| = sqrt(re² + im²))
------------------------------------------------------------------------------
* ``S11_28GHz_dB``  : 20·log10|S11| at exactly 28.00 GHz (inside the sweep; defined for every design).
* ``S11_min_dB``    : deepest |S11| within 23–33 GHz (a band-limited quantity by definition).
* ``f_res_GHz``     : frequency of that minimum, ONLY when it lies strictly inside the band. When the
                      minimum sits on 23 or 33 GHz the true resonance is outside the simulated band,
                      so the value is censored (left missing) instead of reporting a false frequency.
* ``BW10_MHz``      : width of the contiguous |S11| ≤ −10 dB region containing the minimum, ONLY when
                      that region lies fully inside the band; otherwise missing (truncated or no match).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.public import DatasetUnavailable, processed_dir, raw_dir
from utils.validation import file_checksum

KEY = "metu_simpl"
REPO_URL = "https://github.com/METU-SIMPL/Microstrip-Patch.git"
FILES = ["README.md", "input_parameters.xlsx", "S11_re.xlsx", "S11_im.xlsx", "antenna_model.aedt"]
CITATION = ("E. S. Saçın and A. C. Durgun, \"Microstrip Patch Antenna Modeling Using Neural Networks with "
            "Knowledge-Based Regularization,\" Neural Computing and Applications, vol. 37, pp. 3827–3837, 2025, "
            "doi:10.1007/s00521-024-10860-5.")
LICENSE_NOTE = ("No license file in the repository (checked at the downloaded commit). The README requests citation "
                "of the associated publication and acknowledgement of the authors. Used here for non-commercial "
                "academic analysis with citation; files are not redistributed.")

FREQ_START_GHZ, FREQ_STEP_GHZ, N_FREQ = 23.0, 0.05, 201
FREQUENCIES_GHZ = np.round(FREQ_START_GHZ + FREQ_STEP_GHZ * np.arange(N_FREQ), 4)
DESIGN_FREQ_GHZ = 28.0

COLUMN_MAP = {
    "length of patch": "patch_length_mm",
    "width of patch": "patch_width_mm",
    "height of patch": "patch_thickness_mm",
    "height of substrate": "substrate_height_mm",
    "height of solder resist layer": "solder_resist_height_mm",
    "radius of the probe": "probe_radius_mm",
    "c_pad": "pad_clearance_mm",
    "c_antipad": "antipad_clearance_mm",
    "c_probe": "feed_position_ratio",
    "dielectric constant of substrate": "substrate_eps_r",
    "dielectric constant of solder resist layer": "solder_resist_eps_r",
}
FEATURES = list(COLUMN_MAP.values())
TARGETS = ["S11_28GHz_dB", "S11_min_dB", "f_res_GHz", "BW10_MHz"]

PROVENANCE = (f"Public EM-SIMULATION dataset (Ansys HFSS full-wave), METU-SIMPL Microstrip-Patch, {REPO_URL}. "
              f"Cite: {CITATION} License: {LICENSE_NOTE}")


def download(dest: Path | None = None) -> dict:
    """Clone the repository, copy the data files unchanged into data/raw/metu_simpl, write a manifest."""
    dest = Path(dest or raw_dir(KEY))
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, tmp + "/repo"],
                              capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise DatasetUnavailable(f"git clone failed: {proc.stderr.strip()[-400:]}")
        repo = Path(tmp) / "repo"
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        present = sorted(p.name for p in repo.iterdir() if p.is_file())
        missing = [f for f in FILES if f not in present]
        if missing:
            raise DatasetUnavailable(f"Expected files missing from the repository: {missing}")
        license_files = [p for p in present if p.lower().startswith(("license", "licence", "copying"))]
        for f in FILES:
            shutil.copy2(repo / f, dest / f)
    manifest = {
        "source": REPO_URL, "commit": commit,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {f: {"bytes": (dest / f).stat().st_size, "sha256": file_checksum(dest / f)} for f in FILES},
        "repository_files": present, "license_files_found": license_files, "license_note": LICENSE_NOTE,
        "citation": CITATION, "data_kind": "EM simulation (Ansys HFSS)",
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(src: Path | None = None) -> dict:
    """Re-hash the raw files against the manifest (detects any modification of the originals)."""
    src = Path(src or raw_dir(KEY))
    manifest = json.loads((src / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [f for f, meta in manifest["files"].items() if file_checksum(src / f) != meta["sha256"]]
    if bad:
        raise DatasetUnavailable(f"Raw files changed since download: {bad}")
    return manifest


def s11_db_matrix(src: Path | None = None) -> np.ndarray:
    src = Path(src or raw_dir(KEY))
    re_ = pd.read_excel(src / "S11_re.xlsx").to_numpy(float)
    im_ = pd.read_excel(src / "S11_im.xlsx").to_numpy(float)
    if re_.shape != im_.shape or re_.shape[1] != N_FREQ:
        raise DatasetUnavailable(f"Unexpected S11 shape {re_.shape} / {im_.shape}; expected (n, {N_FREQ}).")
    mag = np.sqrt(re_ ** 2 + im_ ** 2)
    if np.any(mag > 1.0 + 1e-6):
        raise DatasetUnavailable("|S11| > 1 found; the files do not look like passive S11 data.")
    return 20.0 * np.log10(np.clip(mag, 1e-12, None))


def derive_targets(db: np.ndarray) -> pd.DataFrame:
    f = FREQUENCIES_GHZ
    i28 = int(np.flatnonzero(np.isclose(f, DESIGN_FREQ_GHZ))[0])
    imin = db.argmin(axis=1)
    interior = (imin > 0) & (imin < N_FREQ - 1)
    bw = np.full(len(db), np.nan)
    for r in np.flatnonzero(db.min(axis=1) <= -10.0):
        below = db[r] <= -10.0
        lo = hi = imin[r]
        while lo > 0 and below[lo - 1]:
            lo -= 1
        while hi < N_FREQ - 1 and below[hi + 1]:
            hi += 1
        if lo > 0 and hi < N_FREQ - 1:          # region fully inside the band
            bw[r] = (f[hi] - f[lo]) * 1000.0
    return pd.DataFrame({
        "S11_28GHz_dB": db[:, i28],
        "S11_min_dB": db.min(axis=1),
        "f_res_GHz": np.where(interior, f[imin], np.nan),
        "BW10_MHz": bw,
    })


def preprocess(src: Path | None = None, dest: Path | None = None) -> dict:
    """Build the modelling table in data/processed/metu_simpl (the raw files are only read)."""
    src, dest = Path(src or raw_dir(KEY)), Path(dest or processed_dir(KEY))
    manifest = verify(src)
    X = pd.read_excel(src / "input_parameters.xlsx")
    unknown = [c for c in X.columns if c not in COLUMN_MAP]
    if unknown or len(X.columns) != len(COLUMN_MAP):
        raise DatasetUnavailable(f"Unexpected input columns: {list(X.columns)}")
    X = X.rename(columns=COLUMN_MAP)[FEATURES]
    db = s11_db_matrix(src)
    if len(db) != len(X):
        raise DatasetUnavailable(f"Row mismatch: {len(X)} input rows vs {len(db)} S11 rows.")
    targets = derive_targets(db)
    table = pd.concat([pd.DataFrame({"design_id": np.arange(1, len(X) + 1)}), X, targets], axis=1)
    n_raw = len(table)
    feature_dups = int(X.duplicated().sum())
    missing_inputs = int(X.isna().any(axis=1).sum())
    table = table.loc[~X.isna().any(axis=1) & ~X.duplicated()].reset_index(drop=True)
    dest.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest / "metu_simpl_processed.csv", index=False, float_format="%.6g")
    np.savez_compressed(dest / "s11_db_curves.npz", s11_db=db.astype(np.float32), frequency_ghz=FREQUENCIES_GHZ)
    summary = {
        "source": manifest["source"], "commit": manifest["commit"], "downloaded_at": manifest["downloaded_at"],
        "rows_raw": n_raw, "rows_processed": len(table), "duplicate_input_rows_removed": feature_dups,
        "rows_with_missing_inputs_removed": missing_inputs,
        "features": FEATURES, "targets": TARGETS,
        "target_non_missing": {t: int(table[t].notna().sum()) for t in TARGETS},
        "censoring": {
            "f_res_GHz": "missing where the |S11| minimum lies on a band edge (resonance outside 23–33 GHz)",
            "BW10_MHz": "missing where min |S11| > −10 dB or the −10 dB region reaches a band edge",
        },
        "frequency_sweep_GHz": [float(FREQUENCIES_GHZ[0]), float(FREQUENCIES_GHZ[-1]), FREQ_STEP_GHZ],
        "feature_ranges": {c: [float(table[c].min()), float(table[c].max())] for c in FEATURES},
    }
    (dest / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
