"""Page 8 - Report generator."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

import config
from reporting.report_generator import export_csv_bundle, generate_pdf_report
from ui.common import active_dataset, get_repo, page_header, provenance_banner


def render() -> None:
    page_header("Report Generator", "Download a PDF report and CSV exports of all experiment records.")
    ds = active_dataset()
    provenance_banner(ds)
    if ds is None:
        return
    st.markdown("The PDF contains project details, dataset provenance and checksum, methodology, model "
                "configurations and metrics, actual-vs-predicted, permutation-importance and uncertainty figures, "
                "optimization results, limitations and reproducibility information for the **active dataset**.")
    n_opt = st.slider("Maximum optimization runs to include", 1, 20, 6)
    if st.button("Generate PDF report", type="primary"):
        with st.spinner("Building report..."):
            try:
                path = generate_pdf_report(get_repo(), ds["id"], max_optimizations=n_opt)
                st.session_state["report_path"] = str(path)
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")
    path = st.session_state.get("report_path")
    if path:
        with open(path, "rb") as fh:
            st.download_button("Download PDF report", fh.read(), file_name=path.split("/")[-1], mime="application/pdf")
        st.caption(f"Saved to `{path}`")

    st.divider()
    if st.button("Prepare CSV export (all tables)"):
        data, paths = export_csv_bundle(get_repo())
        st.session_state["csv_zip"] = data
        st.caption("Also written to: " + ", ".join(f"`{p.name}`" for p in paths) + f" in `{config.REPORTS_DIR}`")
    if "csv_zip" in st.session_state:
        st.download_button("Download CSV bundle (ZIP)", st.session_state["csv_zip"],
                           file_name=f"metaopt_ai_export_{datetime.now():%Y%m%d_%H%M%S}.zip", mime="application/zip")
