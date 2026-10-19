"""Publicly available antenna datasets: download, verification, preprocessing and pipeline status.

Each source has its own module and its own preprocessing pipeline. A source is offered in the
application only when ``pipeline_status.json`` in its processed folder records every stage
(download, preprocessing, training) as completed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config

SOURCES = {
    "metu_simpl": {"label": "METU-SIMPL", "dataset_name": "METU-SIMPL microstrip patch (HFSS, 23–33 GHz)"},
    "zenodo": {"label": "Zenodo", "dataset_name": "Zenodo record 15866865"},
}
STAGES = ("download", "preprocess", "train")


class DatasetUnavailable(RuntimeError):
    """Raised when a source cannot be downloaded or does not contain usable antenna data."""


def raw_dir(key: str) -> Path:
    return config.PUBLIC_RAW_DIR / key


def processed_dir(key: str) -> Path:
    return config.PUBLIC_PROCESSED_DIR / key


def status_path(key: str) -> Path:
    return processed_dir(key) / "pipeline_status.json"


def read_status(key: str) -> dict:
    path = status_path(key)
    if not path.exists():
        return {"source": key, "stages": {}, "complete": False, "message": "Pipeline has not been run."}
    return json.loads(path.read_text(encoding="utf-8"))


def write_stage(key: str, stage: str, ok: bool, message: str = "", **extra) -> dict:
    status = read_status(key)
    status.setdefault("stages", {})[stage] = {
        "ok": ok, "message": message, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}
    status["complete"] = all(status["stages"].get(s, {}).get("ok") for s in STAGES)
    status["message"] = message if not ok else ("Pipeline complete." if status["complete"] else "In progress.")
    path = status_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def ready_sources() -> dict[str, dict]:
    """Sources whose download, preprocessing and training all completed."""
    return {k: v for k, v in SOURCES.items() if read_status(k).get("complete")}
