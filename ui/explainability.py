"""Page 6 - Explainable AI and uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config
from ml.explainability import (
    CAUSATION_NOTE,
    design_space_grid,
    high_uncertainty_regions,
    impurity_importance,
    local_sensitivities,
    partial_dependence_frame,
    permutation_importance_frame,
    sensitivity_sweep,
)
from ml.model_registry import model_label
from ml.uncertainty import UNCERTAINTY_DISCLAIMER
from services import workflow
from ui.common import active_frame, get_repo, page_header, provenance_banner
from visualization import plots


@st.cache_data(show_spinner=False)
def _perm(experiment_id: int, _payload_key: str):
    exp = get_repo().get_experiment(experiment_id)
    lm = workflow.load_models({exp["target_variable"]: exp})[exp["target_variable"]]
    p = lm.payload
    return permutation_importance_frame(lm.pipeline, p["X_test"], p["y_test"], lm.features, seed=p.get("seed", 42))


def render() -> None:
    page_header("Explainable AI & Uncertainty", "What has the surrogate learned, and where is it unsure?")
    df, ds = active_frame()
    provenance_banner(ds)
    if ds is None:
        return
    repo = get_repo()
    exps = [e for e in repo.list_experiments(dataset_id=ds["id"]) if e["model_type"] != "dummy" and e["artifact_path"]]
    if not exps:
        st.info("Train at least one non-baseline model in the ML Model Lab first.")
        return
    active_ids = {e["id"] for e in workflow.get_active_experiments(repo, ds["id"]).values()}
    exps.sort(key=lambda e: (e["id"] not in active_ids, -e["id"]))
    exp_id = st.selectbox("Model experiment", [e["id"] for e in exps],
                          format_func=lambda i: next(f"#{e['id']} {model_label(e['model_type'])} ({e['variant']}) -> "
                                                     f"{e['target_variable']}{' [active]' if e['id'] in active_ids else ''}"
                                                     for e in exps if e["id"] == i))
    exp = repo.get_experiment(exp_id)
    target = exp["target_variable"]
    lm = workflow.load_models({target: exp})[target]
    p, features = lm.payload, lm.features
    ranges = lm.train_ranges
    st.info(CAUSATION_NOTE, icon="🔎")

    st.subheader("A. Feature importance")
    c1, c2 = st.columns(2)
    imp = impurity_importance(lm.pipeline, features)
    fig = plots.importance_bar(imp, "importance", f"Impurity (MDI) importance - {target}")
    if fig:
        c1.plotly_chart(fig, width="stretch")
        c1.caption("MDI: computed on training data; can be biased. Tree models only.")
    else:
        c1.info("Impurity importance is only defined for Random Forest / Extra Trees / Gradient Boosting.")
    perm = _perm(exp_id, exp["artifact_path"])
    c2.plotly_chart(plots.importance_bar(perm, "importance_mean", f"Permutation importance (drop in test R2) - {target}",
                                         err_col="importance_std"), width="stretch")
    c2.caption("Permutation: mean +/- std decrease in R2 on the held-out test set over 10 shuffles. Model-agnostic.")

    st.subheader("B. Parameter sensitivity")
    base = {}
    cols = st.columns(len(features))
    for col, f in zip(cols, features):
        lo, hi = ranges[f]
        base[f] = col.slider(f"Base {config.label_of(f)}", float(lo), float(hi), float(np.median(p["X_train"][:, features.index(f)])),
                             key=f"xai_base_{f}")
    sweep_f = st.selectbox("Parameter to sweep", features)
    frame = sensitivity_sweep(lm.pipeline, features, base, sweep_f, *ranges[sweep_f])
    st.plotly_chart(plots.sensitivity_lines({model_label(exp["model_type"]): frame}, sweep_f, target, base[sweep_f]),
                    width="stretch")
    st.dataframe(local_sensitivities(lm.pipeline, features, base, ranges), hide_index=True, width="stretch")
    st.caption("Local finite-difference sensitivity: change in prediction for a +/-5% step of each parameter's range "
               "around the base point (per_unit = change per mm).")

    st.subheader("C. Partial dependence")
    pd_f = st.selectbox("Partial dependence feature", features, key="pdp_f")
    pdf = partial_dependence_frame(lm.pipeline, p["X_train"], features, pd_f)
    st.plotly_chart(plots.partial_dependence_line(pdf, pd_f, target), width="stretch")
    st.caption("Average prediction when the feature is fixed and other features keep their observed values. "
               "Can hide interactions (e.g. the L-G resonance coupling).")

    st.subheader("D. Uncertainty over the design space")
    if p.get("y_std") is None:
        st.info(f"{model_label(exp['model_type'])} provides no uncertainty estimate. Select a Gaussian Process, "
                "Random Forest or Extra Trees experiment.")
    elif len(features) < 2:
        st.info("A 2-D design-space map needs at least two input features.")
    else:
        cov = (exp["extra_metrics"] or {}).get("coverage95")
        if cov is not None:
            st.metric("Empirical 95% interval coverage on test set", f"{cov:.1%}",
                      help="Ideal approx 95%. Much lower = over-confident; much higher = over-cautious.")
        fx, fy = features[0], features[1]
        grid = design_space_grid(lm.pipeline, features, fx, fy, ranges, base, n=70)
        pts = pd.DataFrame(p["X_train"], columns=features)
        hot = high_uncertainty_regions(grid, top_k=5)
        c1, c2 = st.columns(2)
        c1.plotly_chart(plots.design_space_map(grid, grid["mean"], f"Predicted {target} (mean)", config.label_of(target),
                                               points=pts), width="stretch")
        hot_marks = hot.assign(label=[f"U{i + 1}" for i in range(len(hot))]) if len(hot) else None
        c2.plotly_chart(plots.design_space_map(grid, grid["std"], f"Approx. uncertainty (std) of {target}",
                                               f"std [{config.unit_of(target)}]", scale=plots.SEQUENTIAL_UNC,
                                               points=pts, markers=hot_marks), width="stretch")
        st.markdown("**Where would new EM simulations / measurements be most informative?** "
                    "The marked points U1-U5 have the highest model uncertainty:")
        st.dataframe(hot, hide_index=True, width="stretch")
        st.caption("This is a suggestion based on model uncertainty. This project has NOT demonstrated that "
                   "uncertainty-guided sampling improves accuracy. " + UNCERTAINTY_DISCLAIMER)
