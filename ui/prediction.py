"""Page 4 - Antenna performance prediction."""

from __future__ import annotations

import streamlit as st

import config
from ml.model_registry import model_label
from ml.uncertainty import UNCERTAINTY_DISCLAIMER
from services import workflow
from ui.common import active_frame, active_models_or_stop, fmt_unc, get_repo, page_header, provenance_banner
from utils.validation import ValidationError


def render() -> None:
    page_header("Antenna Performance Prediction", "Predict S11 and gain from geometry with the active surrogate models.")
    df, ds = active_frame()
    provenance_banner(ds)
    if ds is None:
        return
    models = active_models_or_stop(ds)
    features = next(iter(models.values())).features
    ranges = workflow.model_training_ranges(models)

    st.subheader("Geometry inputs")
    cols = st.columns(len(features))
    params = {}
    for col, f in zip(cols, features):
        lo, hi = ranges[f]
        params[f] = col.number_input(f"{config.label_of(f)}  (trained on {lo:.3g}-{hi:.3g})",
                                     value=float(round((lo + hi) / 2, 4)), step=0.01, format="%.4f", key=f"pred_{f}")
    if st.button("Predict", type="primary"):
        try:
            result, _ = workflow.predict_and_maybe_save(get_repo(), models, params, save=False)
            st.session_state["last_prediction"] = (ds["id"], result)
        except ValidationError as exc:
            st.session_state.pop("last_prediction", None)
            st.error(f"Invalid input: {exc}")

    stored = st.session_state.get("last_prediction")
    if stored is None or stored[0] != ds["id"]:
        return
    result = stored[1]
    if result.inputs != {f: float(params[f]) for f in features}:
        st.info("Inputs changed since the last prediction - press **Predict** again.")
    for w in result.warnings:
        st.error(w, icon="🚫")
    cols = st.columns(len(result.predictions))
    for col, (t, v) in zip(cols, result.predictions.items()):
        col.metric(f"Predicted {t}", f"{v:.3f} {config.unit_of(t)}")
        col.caption(f"Uncertainty: {fmt_unc(result.uncertainty.get(t), config.unit_of(t))}")
    st.caption("Label: **ML PREDICTION** - " + ("EXTRAPOLATION, unreliable. " if result.extrapolation else "") +
               ("model trained on SYNTHETIC data; not experimentally validated." if ds["dataset_type"] == "SYNTHETIC"
                else f"model trained on {ds['dataset_type']} data; not a measurement."))
    st.subheader("Model identity and validation metrics")
    rows = []
    for t, m in models.items():
        e = m.experiment
        rows.append({"target": t, "experiment": e["id"], "model": model_label(e["model_type"]), "variant": e["variant"],
                     "dataset": f"#{ds['id']} {ds['dataset_type']}", "test R2": e["r2"], "test MAE": e["mae"],
                     "test RMSE": e["rmse"], "CV R2": e["cv_r2_mean"],
                     "coverage95": (e["extra_metrics"] or {}).get("coverage95"),
                     "uncertainty method": result.uncertainty_method.get(t) or "none"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(UNCERTAINTY_DISCLAIMER)
    if st.button("Save this prediction to the database"):
        rec = workflow.save_prediction(get_repo(), models, result)
        st.success(f"Saved as prediction #{rec['id']}.")
