"""Page 7 - Database and experiment history."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from database.repository import records_to_frame
from ui.common import get_repo, page_header


def _filter(df: pd.DataFrame, text: str) -> pd.DataFrame:
    if df.empty or not text:
        return df
    mask = df.astype(str).apply(lambda col: col.str.contains(text, case=False, regex=False)).any(axis=1)
    return df[mask]


def _table_with_export(df: pd.DataFrame, name: str, search: str) -> None:
    shown = _filter(df, search)
    st.caption(f"{len(shown)} of {len(df)} record(s)")
    st.dataframe(shown, hide_index=True, width="stretch")
    st.download_button(f"Export {name} (CSV)", shown.to_csv(index=False).encode(), file_name=f"{name}.csv",
                       mime="text/csv", key=f"dl_{name}")


def render() -> None:
    page_header("Database & Experiment History", f"All records persisted in SQLite: {config.DB_PATH}")
    repo = get_repo()
    counts = repo.counts()
    cols = st.columns(5)
    for col, (k, v) in zip(cols, counts.items()):
        col.metric(k.capitalize(), v)
    search = st.text_input("Search all tables (free text)")
    t1, t2, t3, t4 = st.tabs(["Datasets", "Model experiments", "Predictions", "Optimization runs"])
    with t1:
        _table_with_export(records_to_frame(repo.list_datasets()), "datasets", search)
    with t2:
        exps = repo.list_experiments()
        c1, c2, c3 = st.columns(3)
        ds_f = c1.selectbox("Dataset", ["all"] + sorted({e["dataset_id"] for e in exps}))
        tg_f = c2.selectbox("Target", ["all"] + sorted({e["target_variable"] for e in exps}))
        md_f = c3.selectbox("Model type", ["all"] + sorted({e["model_type"] for e in exps}))
        exps = repo.list_experiments(dataset_id=None if ds_f == "all" else ds_f, target=None if tg_f == "all" else tg_f,
                                     model_type=None if md_f == "all" else md_f)
        df = records_to_frame(exps)
        if not df.empty:
            front = ["id", "dataset_id", "source_type", "model_type", "variant", "target_variable", "r2", "mae", "rmse",
                     "cv_r2_mean", "random_seed", "dataset_checksum", "created_at"]
            df = df[front + [c for c in df.columns if c not in front]]
        _table_with_export(df, "model_experiments", search)
    with t3:
        _table_with_export(records_to_frame(repo.list_predictions()), "predictions", search)
    with t4:
        runs = repo.list_optimization_runs()
        algos = ["all"] + sorted({r["algorithm"] for r in runs})
        a = st.selectbox("Algorithm", algos)
        runs = repo.list_optimization_runs(algorithm=None if a == "all" else a)
        _table_with_export(records_to_frame(runs), "optimization_runs", search)
        if runs:
            rid = st.selectbox("Inspect run", [r["id"] for r in runs])
            st.json(repo.get_optimization_run(rid))
    st.caption("Records persist across application restarts. SYNTHETIC source types denote demonstration data.")
