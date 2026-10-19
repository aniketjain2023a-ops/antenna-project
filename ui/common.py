"""Shared Streamlit helpers: repository access, active dataset and cached model loading.

Caching policy
--------------
* ``get_repo``      - one Repository per server process (cache_resource).
* ``_load_frame``   - keyed by (dataset id, checksum): a newly registered dataset has a new id,
                      so it is never served stale.
* ``_load_model``   - keyed by (experiment id, artefact path). Artefacts are immutable once
                      written, so trained models stay in memory across reruns instead of being
                      re-read from disk on every interaction.
* Database queries (experiments, runs, active-model settings) are NOT cached, so new records
  appear immediately.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from database.repository import Repository
from ml.prediction import LoadedModel
from services import workflow


@st.cache_resource(show_spinner=False)
def get_repo() -> Repository:
    config.ensure_directories()
    repo = Repository()
    workflow.ensure_demo_dataset(repo)
    return repo


def active_dataset() -> dict | None:
    repo = get_repo()
    ds_id = workflow.get_active_dataset_id(repo)
    return repo.get_dataset(ds_id) if ds_id else None


@st.cache_data(show_spinner=False)
def _load_frame(dataset_id: int, checksum: str) -> tuple[pd.DataFrame, list[str]]:
    df, _, warnings = workflow.load_dataset_frame(get_repo(), dataset_id)
    return df, warnings


def active_frame() -> tuple[pd.DataFrame | None, dict | None, list[str]]:
    """Return (dataframe, dataset record, warnings) for the active dataset."""
    ds = active_dataset()
    if ds is None:
        return None, None, []
    try:
        df, warnings = _load_frame(ds["id"], ds["checksum"])
    except FileNotFoundError as exc:
        return None, ds, [f"The data file for this dataset is missing: {exc}"]
    return df, ds, list(warnings)


@st.cache_resource(show_spinner=False, max_entries=64)
def _load_model(experiment_id: int, artifact_path: str) -> LoadedModel:
    exp = get_repo().get_experiment(experiment_id)
    return workflow.load_models({exp["target_variable"]: exp})[exp["target_variable"]]


def load_model(experiment: dict) -> LoadedModel:
    return _load_model(experiment["id"], experiment["artifact_path"])


def load_active_models(dataset_id: int) -> tuple[dict[str, LoadedModel], str | None]:
    """Active model per target for a dataset, plus an error message if loading failed."""
    try:
        experiments = workflow.get_active_experiments(get_repo(), dataset_id)
        return {t: load_model(e) for t, e in experiments.items()}, None
    except Exception as exc:  # corrupted or deleted artefact
        return {}, f"Could not load the trained models ({exc}). Retrain them on the Dataset & Model page."


def fmt_unc(value, unit: str) -> str:
    return "not available for this model" if value is None else f"± {value:.2f} {unit} (approx. 1σ)"
