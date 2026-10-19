"""Page 2 - Dataset management: inspect registered data, upload and register new data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from database.repository import DuplicateRecordError
from datasets.io import excel_sheet_names, read_table
from ml.preprocessing import clean_dataframe, detect_column_types, inspect_quality, iqr_outlier_mask
from services import workflow
from ui.common import active_frame, get_repo, page_header, provenance_banner
from visualization import plots


def _inspect_tab(df: pd.DataFrame, ds: dict) -> None:
    provenance_banner(ds)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", len(df))
    c2.metric("Input features", len(ds["input_features"]))
    c3.metric("Targets", len(ds["target_names"]))
    c4.metric("Type", ds["dataset_type"])
    with st.expander("Provenance and metadata", expanded=True):
        st.write(f"**Name:** {ds['name']}")
        st.write(f"**Provenance:** {ds['provenance']}")
        st.write(f"**Description:** {ds['description']}")
        st.write(f"**SHA-256 checksum:** `{ds['checksum']}`")
        st.write(f"**Stored file:** `{ds['file_path']}` | **Registered:** {ds['created_at']}")
    cols = ds["input_features"] + ds["target_names"]
    st.subheader("Preview")
    st.dataframe(df.head(50), width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(df[cols].describe().T, width="stretch")
    st.subheader("Data quality")
    report = inspect_quality(df, cols)
    st.dataframe(report.as_frame(), width="stretch")
    st.write(f"Duplicate rows (selected columns): **{report.duplicate_rows}**")
    for w in report.warnings:
        st.warning(w)
    outlier_col = st.selectbox("Inspect IQR outliers for column", report.numeric_columns)
    mask = iqr_outlier_mask(df[outlier_col])
    st.caption(f"{int(mask.sum())} row(s) outside Q1-1.5 IQR / Q3+1.5 IQR. Flagged only - nothing is removed. "
               "For S11 a long tail toward deep resonances is physically expected, not an error.")
    if mask.any():
        st.dataframe(df.loc[mask, cols], width="stretch")

    st.subheader("Distributions")
    fig = plots.distribution_grid(df, cols)
    if fig:
        st.plotly_chart(fig, width="stretch")
    fig = plots.correlation_heatmap(df, cols)
    if fig:
        st.plotly_chart(fig, width="stretch")
        st.caption("Pearson r measures LINEAR association only; the S11 resonance is strongly non-linear, so a "
                   "low |r| does not mean a parameter is unimportant.")
    if len(ds["input_features"]) >= 2 and ds["target_names"]:
        color = st.selectbox("Colour design-space scatter by", ds["target_names"])
        fig = plots.scatter_design_space(df, ds["input_features"][0], ds["input_features"][1], color)
        if fig:
            st.plotly_chart(fig, width="stretch")
    st.download_button("Download this dataset (CSV)", df.to_csv(index=False).encode(),
                       file_name=f"dataset_{ds['id']}.csv", mime="text/csv")


def _upload_tab() -> None:
    repo = get_repo()
    st.markdown("Upload **CSV or Excel** data. The original file is never modified; a cleaned copy is stored "
                "after you confirm the column meanings and provenance.")
    up = st.file_uploader("Data file", type=["csv", "xlsx", "xls"])
    if up is None:
        st.info("Tip: to try this page, download the synthetic CSV from the first tab and re-upload it "
                "(it will be detected as a duplicate), or upload your own EM-simulation export.")
        return
    raw = up.getvalue()
    sheet = 0
    if up.name.lower().endswith((".xlsx", ".xls")):
        try:
            sheets = excel_sheet_names(raw)
        except Exception as exc:
            st.error(f"Could not read the Excel file: {exc}")
            return
        sheet = st.selectbox("Sheet", sheets)
    try:
        df = read_table(raw, filename=up.name, sheet_name=sheet)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return
    st.success(f"Loaded {len(df)} rows x {len(df.columns)} columns from {up.name}.")
    st.dataframe(df.head(30), width="stretch")
    st.subheader("Detected column types")
    st.dataframe(pd.DataFrame({"detected type": detect_column_types(df), "missing": df.isna().sum()}), width="stretch")

    st.subheader("Column mapping")
    numeric_like = [c for c, k in detect_column_types(df).items() if k in ("numeric", "numeric-like text")]
    default_feats = [c for c in numeric_like if c in ("L_mm", "G_mm")]
    default_targets = [c for c in numeric_like if c in ("S11_dB", "Gain_dBi", "Bandwidth_MHz")]
    features = st.multiselect("Input (geometry) columns", numeric_like, default=default_feats)
    targets = st.multiselect("Target (performance) columns", [c for c in numeric_like if c not in features],
                             default=[t for t in default_targets if t not in features])
    rename = {}
    with st.expander("Optional: rename columns to canonical names (e.g. 'Length (mm)' -> L_mm)"):
        canon = ["(keep)"] + list(config.KNOWN_COLUMNS)
        for c in features + targets:
            choice = st.selectbox(f"'{c}' maps to", canon, key=f"map_{c}")
            if choice != "(keep)":
                rename[c] = choice
    if not features or not targets:
        st.info("Select at least one input and one target column.")
        return
    new_names = [rename.get(c, c) for c in features + targets]
    if len(set(new_names)) != len(new_names):
        st.error("Column mapping produces duplicate names.")
        return

    st.subheader("Validation and preprocessing")
    report = inspect_quality(df, features + targets)
    st.dataframe(report.as_frame(), width="stretch")
    for w in report.warnings:
        st.warning(w)
    c1, c2 = st.columns(2)
    drop_dups = c1.checkbox("Remove exact duplicate rows", value=True)
    drop_missing = c2.checkbox("Drop rows with missing input features", value=True)
    try:
        clean, summary = clean_dataframe(df, features, targets, rename=rename, drop_duplicates=drop_dups,
                                         drop_missing_features=drop_missing)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.text(summary.as_text())
    unknown = [c for c in clean.columns if c not in config.KNOWN_COLUMNS]
    if unknown:
        st.warning(f"Columns {unknown} have no known physical meaning/unit in MetaOpt-AI. They will be used as "
                   "generic numeric variables only after you confirm their meaning below.")
    neg = [f for f in [rename.get(c, c) for c in features] if f.endswith("_mm") and (clean[f] <= 0).any()]
    if neg:
        st.error(f"Geometry columns {neg} contain non-positive lengths. Fix the data before registering.")
        return
    st.download_button("Download cleaned dataset (CSV)", clean.to_csv(index=False).encode(),
                       file_name=f"cleaned_{up.name.rsplit('.', 1)[0]}.csv", mime="text/csv")

    st.subheader("Provenance (mandatory)")
    with st.form("register_form"):
        name = st.text_input("Dataset name", value=up.name.rsplit(".", 1)[0])
        dtype = st.selectbox("Dataset type", list(config.DATASET_TYPES), format_func=lambda k: f"{k} - {config.DATASET_TYPES[k]}")
        provenance = st.text_area("Provenance: who produced the data, how (solver/instrument, settings), when, "
                                  "and the original source/citation")
        description = st.text_area("Description (optional)")
        confirm = st.checkbox("I confirm the selected columns represent the stated physical quantities and units, "
                              "and the dataset type/provenance above are accurate.")
        submitted = st.form_submit_button("Register dataset", type="primary")
    if submitted:
        if not confirm:
            st.error("Please confirm the column meanings and provenance before registering.")
        elif not provenance.strip():
            st.error("Provenance is mandatory.")
        else:
            try:
                ds = workflow.register_dataframe(
                    repo, clean, name=name, dataset_type=dtype, provenance=provenance, description=description,
                    features=[rename.get(c, c) for c in features], targets=[rename.get(c, c) for c in targets])
                workflow.set_active_dataset_id(repo, ds["id"])
                st.success(f"Registered dataset #{ds['id']} ({ds['sample_count']} rows) and made it active.")
            except DuplicateRecordError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Registration failed: {exc}")


def _registry_tab() -> None:
    repo = get_repo()
    rows = repo.list_datasets()
    st.dataframe(pd.DataFrame(rows)[["id", "name", "dataset_type", "sample_count", "input_features", "target_names",
                                     "checksum", "created_at"]], hide_index=True, width="stretch")
    st.caption("Switch the active dataset in the sidebar. Datasets are never edited in place; register a new "
               "version instead, so earlier experiments stay reproducible.")


def render() -> None:
    page_header("Dataset Management", "Inspect, validate, upload and register antenna datasets with provenance.")
    df, ds = active_frame()
    t1, t2, t3 = st.tabs(["Active dataset", "Upload & register", "All registered datasets"])
    with t1:
        if df is not None:
            _inspect_tab(df, ds)
        else:
            provenance_banner(ds)
    with t2:
        _upload_tab()
    with t3:
        _registry_tab()
