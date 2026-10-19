"""Section 3 - Results & History: model performance, explanations, optimization history, provenance, export."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import config
from database.repository import records_to_frame
from ml.explainability import (
    CAUSATION_NOTE,
    design_space_grid,
    high_uncertainty_regions,
    impurity_importance,
    partial_dependence_frame,
    permutation_importance_frame,
    sensitivity_sweep,
)
from ml.model_registry import model_label
from ml.uncertainty import UNCERTAINTY_DISCLAIMER
from reporting.report_generator import LIMITATIONS, export_csv_bundle, export_tables, generate_pdf_report
from services import workflow
from ui import theme
from ui.common import active_frame, get_repo, load_model
from ui.dataset_model import _METRIC_COLUMNS, metrics_frame
from ui.navigation import pages
from visualization import plots


def _exp_label(e: dict, active_ids: set[int]) -> str:
    return (f"#{e['id']} {model_label(e['model_type'])} ({e['variant']}) → {e['target_variable']}"
            + ("  · active" if e["id"] in active_ids else ""))


@st.cache_data(show_spinner=False)
def _permutation(experiment_id: int, artifact_path: str) -> pd.DataFrame:
    lm = load_model(get_repo().get_experiment(experiment_id))
    p = lm.payload
    return permutation_importance_frame(lm.pipeline, p["X_test"], p["y_test"], lm.features, seed=p.get("seed", 42))


# ========================================================================= performance
def _performance_tab(ds: dict, exps: list[dict], active_ids: set[int]) -> None:
    if not exps:
        theme.empty_state("No experiments yet", "Train models on the Dataset & Model page; every run is logged here.",
                          ":material/model_training:")
        st.page_link(pages()["dataset"], label="Go to Dataset & Model", icon=":material/arrow_forward:")
        return
    table = metrics_frame(exps)
    with theme.card("results-1"):
        c1, c2 = st.columns([2.2, 1], vertical_alignment="bottom")
        c1.markdown("**Model performance comparison** · all experiments on this dataset")
        metric = c2.segmented_control("Metric", ["test_R2", "test_RMSE", "test_MAE", "cv_R2_mean"], default="test_R2",
                                      format_func=lambda m: {"test_R2": "R²", "test_RMSE": "RMSE", "test_MAE": "MAE",
                                                             "cv_R2_mean": "CV R²"}[m],
                                      label_visibility="collapsed") or "test_R2"
        latest = metrics_frame([e for e in exps if e["run_group"] == exps[0]["run_group"]])
        theme.chart(plots.metric_comparison(latest, metric),
                    "Latest training run. R² is unitless (1 = perfect); MAE and RMSE are in dB for S11 and dBi for gain.",
                    key="metric_bars")
        st.dataframe(table[["id", "target", "Model", "variant", "test_R2", "test_MAE", "test_RMSE", "cv_R2_mean"]],
                     column_config={"id": st.column_config.NumberColumn("Exp.", format="%d"), **_METRIC_COLUMNS},
                     hide_index=True, width="stretch", height=260)
    with theme.card("results-2"):
        st.markdown("**Diagnostics for one model**")
        choices = [e for e in exps if e["artifact_path"] and e["model_type"] != "dummy"] or exps
        exp = st.selectbox("Experiment", choices, format_func=lambda e: _exp_label(e, active_ids), key="diag_exp")
        try:
            p = load_model(exp).payload
        except Exception as exc:
            st.error(f"Could not load this model: {exc}")
            return
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            theme.chart(plots.actual_vs_predicted(p["y_test"], p["y_pred"], p["target"], y_std=p.get("y_std")),
                        key="diag_avp")
        with c2:
            theme.chart(plots.residual_plot(p["y_test"], p["y_pred"], p["target"]),
                        "Residuals should scatter evenly around zero; a pattern means systematic error.", key="diag_res")


# ===================================================================== explainability
def _explain_tab(ds: dict, exps: list[dict], active_ids: set[int]) -> None:
    usable = [e for e in exps if e["model_type"] != "dummy" and e["artifact_path"]]
    if not usable:
        theme.empty_state("Nothing to explain yet", "Train at least one model other than the Dummy baseline.",
                          ":material/psychology:")
        return
    usable.sort(key=lambda e: (e["id"] not in active_ids, -e["id"]))
    exp = st.selectbox("Model", usable, format_func=lambda e: _exp_label(e, active_ids), key="xai_exp")
    lm = load_model(exp)
    p, features, target, ranges = lm.payload, lm.features, lm.target, lm.train_ranges
    st.caption(CAUSATION_NOTE)

    with theme.card("results-3"):
        st.markdown("**Feature importance**")
        perm = _permutation(exp["id"], exp["artifact_path"])
        imp = impurity_importance(lm.pipeline, features)
        cols = st.columns(2, gap="medium") if imp is not None else [st.container()]
        with cols[0]:
            theme.chart(plots.importance_bar(perm, "importance_mean", f"Permutation importance · {target}",
                                             err_col="importance_std"),
                        "Drop in test R² when one parameter is shuffled (mean ± std of 10 repeats). Model-agnostic.",
                        key="perm")
        if imp is not None:
            with cols[1]:
                theme.chart(plots.importance_bar(imp, "importance", f"Impurity importance · {target}"),
                            "Computed on training data from tree splits; can be biased.", key="mdi")
        else:
            st.caption("Impurity importance is only defined for tree models (Random Forest, Extra Trees, Gradient "
                       "Boosting); select one above to compare both measures.")

    with theme.card("results-4"):
        st.markdown("**Parameter sensitivity**")
        st.caption("Sweep one parameter while the others stay at the base point.")
        cols = st.columns(len(features) + 1)
        sweep = cols[0].segmented_control("Sweep", features, default=features[0], key="sweep_f") or features[0]
        base = {}
        for col, f in zip(cols[1:], features):
            lo, hi = ranges[f]
            base[f] = col.slider(f"Base {config.label_of(f)}", float(lo), float(hi),
                                 float(np.median(p["X_train"][:, features.index(f)])), key=f"base_{exp['id']}_{f}")
        frame = sensitivity_sweep(lm.pipeline, features, base, sweep, *ranges[sweep])
        theme.chart(plots.sensitivity_lines({model_label(exp["model_type"]): frame}, sweep, target, base[sweep]),
                    "The shaded band is the approximate ±1σ model uncertainty, where the model provides one.",
                    key="sens")

    with st.expander("Partial dependence and uncertainty map"):
        pd_f = st.segmented_control("Partial dependence on", features, default=features[0], key="pdp_f") or features[0]
        theme.chart(plots.partial_dependence_line(partial_dependence_frame(lm.pipeline, p["X_train"], features, pd_f),
                                                  pd_f, target),
                    "Average prediction as one parameter varies, over the observed values of the others. Averages can "
                    "hide interactions.", key="pdp")
        if p.get("y_std") is not None and len(features) >= 2:
            cov = (exp["extra_metrics"] or {}).get("coverage95")
            grid = design_space_grid(lm.pipeline, features, features[0], features[1], ranges, base, n=60)
            hot = high_uncertainty_regions(grid, top_k=5)
            marks = hot.assign(label=[f"U{i + 1}" for i in range(len(hot))]) if len(hot) else None
            theme.chart(plots.design_space_map(grid, grid["std"], f"Approximate uncertainty (std) of {target}",
                                               f"std [{config.unit_of(target)}]", scale=plots.SEQUENTIAL_UNC,
                                               points=pd.DataFrame(p["X_train"], columns=features), markers=marks),
                        (f"Empirical 95% interval coverage on the test set: {cov:.0%}. " if cov is not None else "")
                        + "U1–U5 mark where new EM simulations would most reduce model uncertainty — a suggestion, "
                        "not a demonstrated improvement. " + UNCERTAINTY_DISCLAIMER, key="unc")
        else:
            st.caption("The uncertainty map needs a Gaussian Process, Random Forest or Extra Trees model and two inputs.")


# =============================================================================== history
def _history_tab(ds: dict, exp_ids: set[int]) -> None:
    runs = [r for r in get_repo().list_optimization_runs() if r["model_experiment_id"] in exp_ids]
    if not runs:
        theme.empty_state("No optimization runs yet",
                          "Run an optimization on the Predict & Optimize page; each run is saved here.", ":material/tune:")
        st.page_link(pages()["predict"], label="Go to Predict & Optimize", icon=":material/arrow_forward:")
        return
    rows = []
    for r in runs:
        rows.append({"Run": r["id"], "When": r["created_at"], "Algorithm": r["algorithm"].replace("_", " "),
                     "Objective": r["objective_function"].replace("_", " "),
                     **{f"{k}": v for k, v in r["candidate_parameters"].items()},
                     **{f"pred {k}": v for k, v in r["candidate_predictions"].items()},
                     "Feasible": bool(r["feasible"])})
    with theme.card("results-5"):
        st.markdown("**Optimization history**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in rows[0] if c not in ("Run", "When", "Algorithm", "Objective", "Feasible")})
    with theme.card("results-6"):
        run = st.selectbox("Compare baseline and candidate for run", runs,
                           format_func=lambda r: f"Run #{r['id']} · {r['algorithm'].replace('_', ' ')} · "
                                                 f"{r['objective_function'].replace('_', ' ')} · {r['created_at']}")
        targets = list(run["candidate_predictions"])
        frame = pd.DataFrame([
            {"source": "baseline", **run["baseline_predictions"]},
            {"source": "candidate", **run["candidate_predictions"],
             **{f"{t}_std": u for t, u in (run["candidate_uncertainty"] or {}).items()}},
        ])
        cols = st.columns(len(targets), gap="medium")
        for col, t in zip(cols, targets):
            with col:
                theme.chart(plots.baseline_vs_candidates(frame, t), key=f"hist_{t}")
        params = pd.DataFrame({"Baseline": run["baseline_design"], "Candidate": run["candidate_parameters"]})
        params["Change"] = params["Candidate"] - params["Baseline"]
        st.dataframe(params.round(4), width="stretch")
        cons = ", ".join(f"{c['target']} {c['operator']} {c['limit']}" for c in run["constraints"]) or "none"
        st.caption(f"Constraints: {cons} · feasible: {run['feasible']} · bounds {run['parameter_bounds']} · seed "
                   f"{run['random_seed']} · {run['evaluations_used']} evaluations. ML-selected candidate, not "
                   "physically validated.")


# ================================================================================ export
def _records_frame(name: str) -> pd.DataFrame:
    repo = get_repo()
    return records_to_frame({"Datasets": repo.list_datasets, "Model experiments": repo.list_experiments,
                             "Predictions": repo.list_predictions,
                             "Optimization runs": repo.list_optimization_runs}[name]())


def _export_tab(ds: dict) -> None:
    left, right = st.columns([1, 1], gap="large")
    with left, theme.card("results-7"):
        st.markdown("**Dataset provenance**")
        theme.provenance_badge(ds)
        st.write(ds["provenance"])
        st.caption(f"{ds['name']} · {ds['sample_count']} samples · inputs {', '.join(ds['input_features'])} · "
                   f"targets {', '.join(ds['target_names'])}")
        st.caption(f"SHA-256 `{ds['checksum']}`")
    with right, theme.card("results-8"):
        st.markdown("**Export**")
        st.caption("CSV files contain every record in the database. The PDF report covers the active dataset.")
        tables = export_tables(get_repo())
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        with st.container(horizontal=True, gap="small"):
            for name, frame in tables.items():
                st.download_button(name.replace("_", " ").capitalize(), frame.to_csv(index=False).encode(),
                                   file_name=f"{name}_{stamp}.csv", mime="text/csv", icon=":material/download:",
                                   key=f"dl_{name}")
        c1, c2 = st.columns(2)
        if c1.button("Prepare ZIP of all CSVs", icon=":material/folder_zip:", width="stretch"):
            st.session_state["csv_zip"], _ = export_csv_bundle(get_repo())
        if "csv_zip" in st.session_state:
            c1.download_button("Download ZIP", st.session_state["csv_zip"], file_name=f"metaopt_export_{stamp}.zip",
                               mime="application/zip", icon=":material/download:", width="stretch")
        if c2.button("Generate PDF report", icon=":material/picture_as_pdf:", width="stretch"):
            with st.spinner("Building the report…"):
                try:
                    st.session_state["report_path"] = str(generate_pdf_report(get_repo(), ds["id"]))
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")
        path = st.session_state.get("report_path")
        if path:
            with open(path, "rb") as fh:
                c2.download_button("Download PDF", fh.read(), file_name=path.rsplit("/", 1)[-1],
                                   mime="application/pdf", icon=":material/download:", width="stretch")

    with theme.card("results-9"):
        st.markdown("**Experiment records** · SQLite")
        c1, c2 = st.columns([1.4, 1], vertical_alignment="bottom")
        name = c1.segmented_control("Table", ["Model experiments", "Predictions", "Optimization runs", "Datasets"],
                                    default="Model experiments", key="records_table") or "Model experiments"
        search = c2.text_input("Search", placeholder="Filter rows by any text", label_visibility="collapsed")
        frame = _records_frame(name)
        if search and not frame.empty:
            frame = frame[frame.astype(str).apply(lambda c: c.str.contains(search, case=False, regex=False)).any(axis=1)]
        st.caption(f"{len(frame)} record(s) · stored in {config.DB_PATH.name}; records persist across restarts.")
        st.dataframe(frame, hide_index=True, width="stretch", height=280)

    with st.expander("About this project and its limitations", icon=":material/info:"):
        st.markdown(f"**{config.PROJECT_TITLE}** · {config.STUDENT_NAME} · {config.REGISTRATION_NUMBER}")
        st.caption("Contribution: the implementation and integration of an explainable, uncertainty-aware, "
                   "provenance-tracking workflow — not the invention of ML-based antenna optimization.")
        for item in LIMITATIONS:
            st.markdown(f"- {item}")


# ================================================================================== page
def render() -> None:
    _, ds, warnings = active_frame()
    repo = get_repo()
    theme.page_header("Results & History",
                      "Every model, prediction and optimization run is stored in SQLite. Review, explain and "
                      "export them here.")
    if ds is None:
        theme.context_bar(None)
        theme.empty_state("No dataset available", "Load a dataset on the Dataset & Model page first.", ":material/table:")
        return
    exps = repo.list_experiments(dataset_id=ds["id"])
    active_ids = {e["id"] for e in workflow.get_active_experiments(repo, ds["id"]).values()}
    theme.context_bar(ds, len(active_ids))
    theme.synthetic_notice(ds)
    for w in warnings:
        st.warning(w, icon=":material/warning:")
    perf, explain, history, export = st.tabs(["Model performance", "Explainability", "Optimization history",
                                              "Data & export"])
    with perf:
        _performance_tab(ds, exps, active_ids)
    with explain:
        _explain_tab(ds, exps, active_ids)
    with history:
        _history_tab(ds, {e["id"] for e in exps})
    with export:
        _export_tab(ds)
