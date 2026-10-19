"""Central configuration for MetaOpt-AI.

All paths are derived from the project root so the application runs from any
working directory. Environment variables allow tests (and advanced users) to
redirect storage without editing code:

* ``METAOPT_DB_PATH``      - SQLite database file
* ``METAOPT_MODELS_DIR``   - directory for saved model artefacts (.joblib)
* ``METAOPT_REPORTS_DIR``  - directory for generated PDF/CSV reports
* ``METAOPT_UPLOAD_DIR``   - directory for registered (cleaned) dataset copies
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------
PROJECT_NAME = "MetaOpt-AI"
APP_TITLE = "MetaOpt-AI | Antenna Design Explorer"
PROJECT_TITLE = "Explainable Machine Learning for Metasurface Antenna Optimization"
STUDENT_NAME = "Aniket Jain"
REGISTRATION_NUMBER = "23BEC0470"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_CSV_PATH = DATA_DIR / "synthetic_demo.csv"
UPLOAD_DIR = Path(os.environ.get("METAOPT_UPLOAD_DIR", DATA_DIR / "registered"))
DB_PATH = Path(os.environ.get("METAOPT_DB_PATH", DATA_DIR / "metaopt_ai.db"))
MODELS_DIR = Path(os.environ.get("METAOPT_MODELS_DIR", BASE_DIR / "models"))
REPORTS_DIR = Path(os.environ.get("METAOPT_REPORTS_DIR", BASE_DIR / "reports"))
LOGS_DIR = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data provenance vocabulary
# ---------------------------------------------------------------------------
DATASET_TYPES: dict[str, str] = {
    "SYNTHETIC": "Synthetic demonstration data (analytical formula + noise, NOT measured)",
    "PUBLIC": "Publicly sourced data (third-party; cite the original source)",
    "SIMULATION": "Electromagnetic (full-wave) simulation data, e.g. exported from an EM solver",
    "MEASUREMENT": "Experimental measurement data, e.g. VNA / anechoic chamber",
}
PREDICTION_SOURCE = "ML_PREDICTION"
OPTIMIZATION_SOURCE = "ML_SELECTED_CANDIDATE"

# Physical meaning and units of known columns. Unknown columns are allowed but
# must be explicitly confirmed by the user before registration.
KNOWN_COLUMNS: dict[str, dict] = {
    "L_mm": {"unit": "mm", "kind": "input", "description": "Effective patch/metasurface length"},
    "G_mm": {"unit": "mm", "kind": "input", "description": "Metasurface gap"},
    "S11_dB": {"unit": "dB", "kind": "target", "direction": "minimize",
               "description": "Input reflection coefficient magnitude at the design frequency"},
    "Gain_dBi": {"unit": "dBi", "kind": "target", "direction": "maximize",
                 "description": "Realized broadside gain"},
    "Bandwidth_MHz": {"unit": "MHz", "kind": "target", "direction": "maximize",
                      "description": "-10 dB impedance bandwidth (only if supplied by real data)"},
}

# Sanity limits for geometry parameters (mm). These are NOT design bounds; they
# only reject physically meaningless inputs such as negative lengths.
GEOMETRY_SANITY_LIMITS = (1e-3, 1000.0)

# ---------------------------------------------------------------------------
# ML defaults (kept small so everything runs quickly on a laptop CPU)
# ---------------------------------------------------------------------------
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_TUNING_ITERATIONS = 8
MIN_SAMPLES_TO_TRAIN = 20
MIN_SAMPLES_FOR_CV = 50
SMALL_DATASET_WARNING = 150
GP_MAX_TRAIN_SAMPLES = 1500

# ---------------------------------------------------------------------------
# Optimization defaults
# ---------------------------------------------------------------------------
DEFAULT_S11_LIMIT_DB = -10.0
DEFAULT_MIN_GAIN_DBI = 6.0
DEFAULT_EVAL_BUDGET = 1500
CONSTRAINT_PENALTY_WEIGHT = 100.0

SYNTHETIC_WARNING = (
    "SYNTHETIC DATA: this dataset was generated from a documented analytical "
    "formula with added noise. It demonstrates the software workflow only. "
    "Results are NOT electromagnetic simulations or laboratory measurements and "
    "say nothing about real-world antenna accuracy."
)


def ensure_directories() -> None:
    """Create all runtime directories if they do not exist."""
    for path in (DATA_DIR, UPLOAD_DIR, DB_PATH.parent, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)


def unit_of(column: str) -> str:
    """Return the physical unit for a known column, or an empty string."""
    return KNOWN_COLUMNS.get(column, {}).get("unit", "")


def label_of(column: str) -> str:
    """Return a human readable axis label with units, e.g. ``S11_dB [dB]``."""
    unit = unit_of(column)
    return f"{column} [{unit}]" if unit else column
