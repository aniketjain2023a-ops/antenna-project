"""The three application sections. Page objects are created once so pages can link to each other."""

from __future__ import annotations

from functools import cache

import streamlit as st


@cache
def pages() -> dict[str, "st.Page"]:
    from ui import dataset_model, predict_optimize, results_history

    return {
        "dataset": st.Page(dataset_model.render, title="Dataset & Model", icon=":material/database:",
                           url_path="dataset", default=True),
        "predict": st.Page(predict_optimize.render, title="Predict & Optimize", icon=":material/tune:",
                           url_path="predict"),
        "results": st.Page(results_history.render, title="Results & History", icon=":material/insights:",
                           url_path="results"),
    }
