"""Shared Streamlit helpers: repository access, active dataset, warnings."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from database.repository import Repository
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


def active_frame() -> tuple[pd.DataFrame | None, dict | None]:
    ds = active_dataset()
    if ds is None:
        return None, None
    try:
        df, warnings = _load_frame(ds["id"], ds["checksum"])
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None, ds
    for w in warnings:
        st.warning(w)
    return df, ds


def provenance_banner(ds: dict | None) -> None:
    if ds is None:
        st.info("No dataset registered yet. Go to **Dataset Management**.")
        return
    if ds["dataset_type"] == "SYNTHETIC":
        st.warning(f"**{config.SYNTHETIC_WARNING}**", icon="⚠️")
    else:
        st.info(f"Active dataset type: **{ds['dataset_type']}** - {config.DATASET_TYPES[ds['dataset_type']]}. "
                "Model outputs are ML predictions, not physical validation.", icon="ℹ️")


def sidebar_dataset_selector() -> None:
    repo = get_repo()
    datasets = repo.list_datasets()
    if not datasets:
        return
    ids = [d["id"] for d in datasets]
    labels = {d["id"]: f"#{d['id']} {d['name']} [{d['dataset_type']}]" for d in datasets}
    current = workflow.get_active_dataset_id(repo)
    choice = st.sidebar.selectbox("Active dataset", ids, index=ids.index(current) if current in ids else 0,
                                  format_func=lambda i: labels[i])
    if choice != current:
        workflow.set_active_dataset_id(repo, choice)
        st.rerun()
    ds = repo.get_dataset(choice)
    if ds["dataset_type"] == "SYNTHETIC":
        st.sidebar.error("SYNTHETIC demonstration data", icon="⚠️")
    else:
        st.sidebar.info(f"Data type: {ds['dataset_type']}")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def active_models_or_stop(ds: dict):
    """Load active models for the dataset or explain how to create them."""
    repo = get_repo()
    try:
        models = workflow.get_active_models(repo, ds["id"])
    except Exception as exc:  # corrupted artefact etc.
        st.error(f"Could not load the active models: {exc}")
        st.stop()
    if not models:
        st.info("No trained models for this dataset yet. Train them in **ML Model Lab** "
                "(or use the quick-start button on the Overview page).")
        st.stop()
    return models


def fmt_unc(value, unit: str) -> str:
    return "not available for this model" if value is None else f"+/- {value:.3f} {unit} (approx., 1 sigma)"
