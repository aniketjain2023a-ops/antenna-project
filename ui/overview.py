"""Page 1 - Overview dashboard."""

from __future__ import annotations

import streamlit as st

import config
from ml.evaluation import comparison_table
from ml.training import TrainingConfig
from services import workflow
from ui.common import active_dataset, get_repo, page_header, provenance_banner

WORKFLOW_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#2a78d6", fontname="Helvetica"];
  data [label="Dataset\\n(synthetic / EM sim /\\nmeasurement)"]; val [label="Validation &\\nprovenance (SQLite)"];
  train [label="ML training\\nCV + tuning"]; eval [label="Evaluation\\nR2 / MAE / RMSE"];
  pred [label="Prediction +\\nuncertainty"]; opt [label="Surrogate\\noptimization"];
  xai [label="Explainability\\n(importance, PDP)"]; rep [label="Reports &\\nexperiment history"];
  verify [label="EM simulation /\\nmeasurement\\n(outside this app)", fillcolor="#fff7ed", color="#eb6834"];
  data -> val -> train -> eval -> pred -> opt -> verify; eval -> xai; opt -> rep; xai -> rep;
  verify -> data [style=dashed, label="new real data"];
}
"""


def render() -> None:
    repo = get_repo()
    ds = active_dataset()
    page_header("MetaOpt-AI | Antenna Design Explorer", config.PROJECT_TITLE)
    provenance_banner(ds)
    st.markdown(
        "**Purpose.** An integrated, *explainable* and *uncertainty-aware* workflow for exploring metasurface "
        "antenna geometries with machine-learning surrogate models: manage data with provenance, train and compare "
        "regressors, predict S11 and gain, optimize geometry on the surrogate, explain what the model learned, and "
        "keep every experiment reproducible in SQLite."
    )
    counts = repo.counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dataset type", ds["dataset_type"] if ds else "-")
    c2.metric("Samples (active)", ds["sample_count"] if ds else 0)
    c3.metric("Model experiments", counts["experiments"])
    c4.metric("Optimization runs", counts["optimizations"])
    c5.metric("Saved predictions", counts["predictions"])

    st.subheader("Current active models")
    if ds:
        active = workflow.get_active_experiments(repo, ds["id"])
        if active:
            st.dataframe(comparison_table(list(active.values())), hide_index=True, width="stretch")
            st.caption("Active models are chosen per target by cross-validated R2 on training data (test set unused).")
        else:
            st.info("No models trained yet for the active dataset.")
            if st.button("Quick start: train demo models (Random Forest, Extra Trees, Gaussian Process; no tuning)",
                         type="primary"):
                cfg = TrainingConfig(features=ds["input_features"], targets=ds["target_names"],
                                     model_keys=["dummy", "random_forest", "extra_trees", "gaussian_process"])
                bar = st.progress(0.0, "Starting")
                with st.spinner("Training..."):
                    workflow.train_and_log(repo, ds["id"], cfg, progress=lambda f, m: bar.progress(min(f, 1.0), m),
                                           notes="quick start")
                st.success("Models trained and logged.")
                st.rerun()

    st.subheader("Workflow")
    st.graphviz_chart(WORKFLOW_DOT, width="stretch")
    st.subheader("Recent activity")
    st.dataframe(repo.recent_activity(10), hide_index=True, width="stretch")
    st.caption(f"Database: `{config.DB_PATH}` | Student: {config.STUDENT_NAME} ({config.REGISTRATION_NUMBER})")
