"""MetaOpt-AI | Antenna Design Explorer - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from ui import (  # noqa: E402
    data_management,
    explainability,
    history,
    model_lab,
    optimization_lab,
    overview,
    prediction,
    project_info,
    reports,
)
from ui.common import get_repo, sidebar_dataset_selector  # noqa: E402

st.set_page_config(page_title=config.APP_TITLE, page_icon="📡", layout="wide")

try:
    get_repo()  # creates the database + synthetic dataset on first start
except Exception as exc:  # pragma: no cover - shown in the UI
    st.error(f"Initialisation failed: {exc}")
    st.stop()

PAGES = [
    st.Page(overview.render, title="Overview Dashboard", icon="🏠", url_path="overview", default=True),
    st.Page(data_management.render, title="Dataset Management", icon="🗂️", url_path="data"),
    st.Page(model_lab.render, title="ML Model Lab", icon="🧠", url_path="models"),
    st.Page(prediction.render, title="Performance Prediction", icon="📈", url_path="predict"),
    st.Page(optimization_lab.render, title="Optimization Lab", icon="🎯", url_path="optimize"),
    st.Page(explainability.render, title="Explainable AI", icon="🔎", url_path="xai"),
    st.Page(history.render, title="Experiment History", icon="🗄️", url_path="history"),
    st.Page(reports.render, title="Report Generator", icon="📄", url_path="reports"),
    st.Page(project_info.render, title="Project Info & Limitations", icon="ℹ️", url_path="about"),
]

nav = st.navigation(PAGES)
st.sidebar.markdown(f"### {config.PROJECT_NAME}")
st.sidebar.caption("Antenna Design Explorer")
sidebar_dataset_selector()
st.sidebar.caption(f"{config.STUDENT_NAME} | {config.REGISTRATION_NUMBER}")
nav.run()
