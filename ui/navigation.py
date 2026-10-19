"""The three application sections. Page objects are created once so pages can link to each other."""

from __future__ import annotations

from functools import cache

import streamlit as st


@cache
def pages() -> dict[str, "st.Page"]:
    from ui import dataset_model, design_optimize, results_history

    return {
        "home": st.Page(dataset_model.render, title="Home", icon=":material/home:", url_path="home", default=True),
        "design": st.Page(design_optimize.render, title="Design & Optimize", icon=":material/tune:",
                          url_path="design"),
        "results": st.Page(results_history.render, title="Results", icon=":material/insights:", url_path="results"),
    }
