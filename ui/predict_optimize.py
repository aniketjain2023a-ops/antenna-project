"""Section 2 - Predict & Optimize: the main interactive demonstration.

Prediction (one design -> predicted performance) and optimization (search the bounds
for the best design) are deliberately separate operations with separate actions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

import config
from datasets import synthetic
from ml.explainability import design_space_grid
from ml.model_registry import model_label
from ml.uncertainty import UNCERTAINTY_DISCLAIMER
from optimization import ALGORITHMS
from optimization.analysis import rank_candidates, sample_design_space
from optimization.constraints import Constraint
from optimization.objectives import OBJECTIVE_DESCRIPTIONS, ObjectiveConfig, SurrogateEvaluator
from services import workflow
from ui import theme
from ui.common import active_frame, fmt_unc, get_repo, load_active_models
from ui.navigation import pages
from utils.validation import ValidationError, validate_bounds, validate_geometry
from visualization import plots

S11, GAIN = "S11_dB", "Gain_dBi"
OBJECTIVE_LABELS = {"max_gain": "Max gain", "min_s11": "Min S11", "tradeoff": "Trade-off"}
DIRECTION_COLOR = {S11: "inverse", GAIN: "normal"}   # lower S11 is better; higher gain is better


def _s11_first(targets) -> list[str]:
    """S11 is the headline result; show it first, then the other targets in their stored order."""
    targets = list(targets)
    return sorted(targets, key=lambda t: (t != S11, targets.index(t)))


def _feature_label(f: str) -> str:
    desc = config.KNOWN_COLUMNS.get(f, {}).get("description")
    return f"{desc} ({config.unit_of(f)})" if desc else config.label_of(f)


def _inward(lo: float, hi: float) -> tuple[float, float]:
    """Round bounds inward to 4 decimals so default bounds never extend past the training range."""
    return math.ceil(lo * 1e4) / 1e4, math.floor(hi * 1e4) / 1e4


# ============================================================================ prediction
def _parameter_panel(ds: dict, df: pd.DataFrame, features: list[str], ranges: dict) -> tuple[dict, bool]:
    with theme.card("predict-1"):
        st.markdown("**Antenna parameters**")
        params = {}
        for f in features:
            lo, hi = ranges[f]
            key = f"geom_{ds['id']}_{f}"
            if key not in st.session_state:
                st.session_state[key] = round(float(df[f].median()), 4)
            params[f] = st.number_input(_feature_label(f), key=key, step=0.01, format="%.4f",
                                        help=f"The model was trained on {lo:.4g} – {hi:.4g} {config.unit_of(f)}.")
            st.caption(f"Valid range {lo:.4g} – {hi:.4g} {config.unit_of(f)}")
        check = validate_geometry(params, features, ranges)
        for e in check.errors:
            st.error(e, icon=":material/error:")
        for w in check.warnings:
            st.warning(w, icon=":material/warning:")
        freq_note = (f"{synthetic.F0_GHZ:g} GHz (assumed by the synthetic generator)" if ds["dataset_type"] == "SYNTHETIC"
                     else "the dataset's design frequency")
        st.caption(f"Operating frequency is not an input column in this dataset; S11 refers to {freq_note}.")
        predict = st.button("Predict performance", type="primary", icon=":material/bolt:", width="stretch",
                            disabled=not check.ok)
    return params, predict


def _prediction_panel(ds: dict, models: dict, params: dict) -> None:
    key = f"prediction_{ds['id']}"
    result = st.session_state.get(key)
    with theme.card("predict-2"):
        st.markdown("**Performance prediction**")
        if result is None:
            st.caption("Enter the geometry on the left and press **Predict performance**. Predicted S11 and gain "
                       "appear here, with the model's uncertainty and validation metrics.")
            return
        stale = result.inputs != {f: float(v) for f, v in params.items()}
        with st.container(horizontal=True, gap="small"):
            if stale:
                st.badge("Inputs changed · press Predict", color="gray", icon=":material/refresh:")
            elif result.extrapolation:
                st.badge("Extrapolation · unreliable", color="orange", icon=":material/warning:")
            else:
                st.badge("Within training range", color="green", icon=":material/check_circle:")
            theme.provenance_badge(ds)
        cols = st.columns(len(result.predictions))
        for col, t in zip(cols, _s11_first(result.predictions)):
            v = result.predictions[t]
            col.metric(f"Predicted {t.split('_')[0]}", theme.fmt_value(v, t),
                       help=config.KNOWN_COLUMNS.get(t, {}).get("description"))
            col.caption(f"Uncertainty {fmt_unc(result.uncertainty.get(t), config.unit_of(t))}")
        rows = []
        for t in _s11_first(models):
            e = models[t].experiment
            rows.append({"Target": t, "Model": f"{model_label(e['model_type'])} ({e['variant']})",
                         "Test R²": e["r2"], "Test RMSE": e["rmse"], "CV R²": e["cv_r2_mean"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Test R²": st.column_config.NumberColumn(format="%.3f", help="Variance explained on held-out data (1 = perfect)."),
            "Test RMSE": st.column_config.NumberColumn(format="%.3f", help="Typical error in the target unit (dB / dBi)."),
            "CV R²": st.column_config.NumberColumn(format="%.3f", help="Cross-validated R² on training data."),
        })
        st.caption("ML prediction from a surrogate model — not a simulation or measurement. " + UNCERTAINTY_DISCLAIMER)
        if st.button("Save prediction to history", icon=":material/bookmark_add:", disabled=stale):
            rec = workflow.save_prediction(get_repo(), models, result)
            st.toast(f"Saved as prediction #{rec['id']}.", icon=":material/check_circle:")


# ========================================================================== optimization
def _optimization_form(ds: dict, df: pd.DataFrame, models: dict, features: list[str], ranges: dict,
                       params: dict) -> None:
    targets = list(models)
    norm = workflow.target_ranges(df, targets)
    with theme.card("predict-3"):
        c_obj, c_bounds, c_search = st.columns([1.1, 1.3, 1], gap="large")

        with c_obj:
            st.markdown("**Objective**")
            kinds = [k for k in OBJECTIVE_DESCRIPTIONS if (k == "max_gain" and GAIN in targets)
                     or (k == "min_s11" and S11 in targets) or (k == "tradeoff" and len(targets) > 1)]
            kind = st.segmented_control("Objective", kinds, default=kinds[0], format_func=OBJECTIVE_LABELS.get,
                                        label_visibility="collapsed", key=f"objective_{ds['id']}") or kinds[0]
            constraints, weights = [], {}
            if kind == "max_gain" and S11 in targets:
                lim = st.number_input("Subject to S11 ≤ (dB)", value=config.DEFAULT_S11_LIMIT_DB, step=0.5)
                constraints.append(Constraint(S11, "<=", float(lim)))
            elif kind == "min_s11" and GAIN in targets:
                lim = st.number_input("Subject to gain ≥ (dBi)", value=config.DEFAULT_MIN_GAIN_DBI, step=0.5)
                constraints.append(Constraint(GAIN, ">=", float(lim)))
            elif kind == "tradeoff":
                for t in (t for t in targets if t in (S11, GAIN, "Bandwidth_MHz")):
                    weights[t] = st.slider(f"Weight on {t}", 0.0, 1.0, 0.5, 0.05)
                if S11 in targets:
                    lim = st.number_input("Subject to S11 ≤ (dB)", value=config.DEFAULT_S11_LIMIT_DB, step=0.5)
                    constraints.append(Constraint(S11, "<=", float(lim)))
                st.caption("Each target is scaled to 0–1 over its observed range before weighting, so dB and dBi "
                           "are never added directly.")
            st.caption(OBJECTIVE_DESCRIPTIONS[kind] + ".")

        with c_bounds:
            st.markdown("**Parameter bounds**")
            bounds = {}
            for f in features:
                lo, hi = _inward(*ranges[f])
                b1, b2 = st.columns(2)
                b_lo = b1.number_input(f"{f} min", value=lo, format="%.4f", key=f"blo_{ds['id']}_{f}")
                b_hi = b2.number_input(f"{f} max", value=hi, format="%.4f", key=f"bhi_{ds['id']}_{f}")
                bounds[f] = (b_lo, b_hi)
            baseline_mode = st.radio("Baseline design", ["Parameters entered above", "Median training design"],
                                     horizontal=False, help="The design the candidate is compared against.")

        with c_search:
            st.markdown("**Differential Evolution**")
            budget = st.number_input("Evaluation budget", 100, 20_000, config.DEFAULT_EVAL_BUDGET, step=100,
                                     help="Maximum number of surrogate evaluations.")
            seed = st.number_input("Random seed", 0, 10_000, config.RANDOM_SEED, key=f"opt_seed_{ds['id']}")
            with st.expander("Advanced"):
                algos = st.multiselect("Algorithms", list(ALGORITHMS), default=["differential_evolution"],
                                       format_func=lambda a: ALGORITHMS[a][0],
                                       help="Additional optimizers can cross-check the Differential Evolution result.")
                kappa = st.slider("Risk aversion κ", 0.0, 3.0, 0.0, 0.25,
                                  help="Uses mean ∓ κσ where uncertainty is available, penalising uncertain regions. 0 = off.")
                allow_ext = st.checkbox("Allow bounds outside the training range", value=False,
                                        help="Not recommended: the surrogate is unreliable outside its data.")

        # Inline validation before the action, not after it.
        bound_check = validate_bounds(bounds, ranges, allow_ext)
        cfg = ObjectiveConfig(kind=kind, constraints=constraints, weights=weights, normalization=norm, kappa=kappa)
        problems = list(bound_check.errors)
        try:
            cfg.validate(targets)
        except ValueError as exc:
            problems.append(str(exc))
        if not algos:
            problems.append("Choose at least one algorithm under Advanced.")
        for p in problems:
            st.error(p + (" Narrow the bounds to the valid range, or allow extrapolation under Advanced."
                          if "training range" in p else ""), icon=":material/error:")
        baseline = ({f: float(v) for f, v in params.items()} if baseline_mode.startswith("Parameters")
                    else {f: float(df[f].median()) for f in features})
        base_check = validate_geometry(baseline, features)
        if not base_check.ok:
            problems.extend(base_check.errors)
            st.error("The baseline parameters are invalid: " + "; ".join(base_check.errors), icon=":material/error:")

        run = st.button("Run optimization", type="primary", icon=":material/play_arrow:", disabled=bool(problems))
    if run:
        try:
            with st.spinner("Searching the design space with the surrogate…"):
                outs = workflow.run_optimization(get_repo(), models, cfg, bounds, baseline, algos, seed=int(seed),
                                                 budget=int(budget), allow_extrapolation=allow_ext)
            st.session_state[f"optimization_{ds['id']}"] = {"outs": outs, "cfg": cfg, "bounds": bounds}
            ids = ", ".join(f"#{rec['id']}" for _, rec, _ in outs if rec)
            st.toast(f"Optimization finished · saved as run {ids}.", icon=":material/check_circle:")
        except (ValidationError, ValueError) as exc:
            st.error(f"Optimization could not run: {exc}", icon=":material/error:")
        except Exception as exc:
            st.error(f"Optimization failed unexpectedly: {exc}. Try a smaller budget or retrain the models.",
                     icon=":material/error:")


def _design_card(title: str, design: dict, features: list[str], targets: list[str], badge, ref: dict | None = None) -> None:
    with theme.card("design-baseline" if ref is None else "design-candidate"):
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.markdown(f"**{title}**")
            badge()
        pcols = st.columns(len(features))
        for col, f in zip(pcols, features):
            v = design["parameters"][f]
            delta = None if ref is None else f"{v - ref['parameters'][f]:+.3f} {config.unit_of(f)}"
            col.metric(config.label_of(f), f"{v:.3f}", delta=delta, delta_color="off")
        tcols = st.columns(len(targets))
        for col, t in zip(tcols, targets):
            v = design["predictions"][t]
            delta = None if ref is None else f"{v - ref['predictions'][t]:+.2f} {config.unit_of(t)}"
            col.metric(f"Predicted {t.split('_')[0]}", theme.fmt_value(v, t), delta=delta,
                       delta_color=DIRECTION_COLOR.get(t, "off"))
        unc = [f"{t.split('_')[0]} ± {design['uncertainty'][t]:.2f} {config.unit_of(t)}"
               for t in _s11_first(design["uncertainty"]) if design["uncertainty"][t] is not None]
        if unc:
            st.caption("Approx. model uncertainty (1σ): " + ", ".join(unc))


def _constraint_badge(design: dict, cfg: ObjectiveConfig):
    def _badge():
        if not cfg.constraints:
            st.badge("Unconstrained", color="gray")
        elif design["feasible"]:
            st.badge("Meets " + ", ".join(c.describe() for c in cfg.constraints), color="green", icon=":material/check:")
        else:
            st.badge("Violates " + ", ".join(k for k, v in design["violations"].items() if v > 0), color="red",
                     icon=":material/close:")
    return _badge


def _optimization_results(ds: dict, df: pd.DataFrame, models: dict, features: list[str]) -> None:
    res = st.session_state.get(f"optimization_{ds['id']}")
    if not res:
        theme.empty_state("No optimization run yet",
                          "Set the objective and bounds above and press Run optimization. The baseline and the "
                          "ML-selected candidate will be compared here.", ":material/tune:")
        return
    outs, cfg, bounds = res["outs"], res["cfg"], res["bounds"]
    base = outs[0][2]
    cands = [{**o.candidate, "source": o.algorithm} for o, _, _ in outs]
    ranked = rank_candidates([{**base, "source": "baseline"}] + cands, cfg)
    best = next(c for c in cands if c["source"] == ranked[ranked["source"] != "baseline"].iloc[0]["source"])
    targets = _s11_first(models)

    left, right = st.columns(2, gap="medium")
    with left:
        _design_card("Baseline design", base, features, targets, _constraint_badge(base, cfg))
    with right:
        _design_card(f"ML-selected candidate · {ALGORITHMS[best['source']][0].split(' (')[0]}", best, features,
                     targets, _constraint_badge(best, cfg), ref=base)
    o_best = next(o for o, _, _ in outs if o.algorithm == best["source"])
    rec = next(r for o, r, _ in outs if o.algorithm == best["source"])
    st.caption(f"Changes are relative to the baseline; green means better for that target (lower S11, higher gain). "
               f"{o_best.n_evals} surrogate evaluations in {o_best.runtime_s:.2f} s · saved as run #{rec['id']}. "
               "**This candidate was selected by an ML surrogate and has not been physically validated** — verify it "
               "with full-wave EM simulation and measurement. A deeper S11 alone does not imply better gain, bandwidth "
               "or efficiency.")
    if len(cands) > 1:
        with st.expander("All algorithms, ranked"):
            st.dataframe(ranked, hide_index=True, width="stretch")

    with st.expander("Search diagnostics"):
        theme.chart(plots.convergence_plot({o.algorithm: o.history for o, _, _ in outs}),
                    "Best objective found so far. Early high values are penalties for designs that break a constraint.",
                    key="conv")
        if S11 in models and GAIN in models:
            samples = sample_design_space(SurrogateEvaluator(models), cfg, bounds, n=1200, seed=1)
            cand_df = pd.DataFrame([{"source": c["source"], **c["predictions"]} for c in cands])
            theme.chart(plots.tradeoff_scatter(samples, S11, GAIN, cand_df, base["predictions"]),
                        "Each dot is a predicted design. The Pareto front marks designs where no other sample has both "
                        "lower S11 and higher gain.", key="tradeoff")
        if len(features) >= 2:
            fx, fy = features[0], features[1]
            target = GAIN if GAIN in models else targets[0]
            fixed = {f: float(np.median(df[f])) for f in features}
            grid = design_space_grid(models[target].pipeline, features, fx, fy, bounds, fixed, n=70)
            marks = pd.DataFrame([{fx: c["parameters"][fx], fy: c["parameters"][fy], "label": c["source"][:10]}
                                  for c in [{**base, "source": "baseline"}] + cands])
            theme.chart(plots.design_space_map(grid, grid["mean"], f"Predicted {target} across the search bounds",
                                               config.label_of(target), markers=marks), key="dsmap")
        if ds["dataset_type"] == "SYNTHETIC" and ds["name"] == workflow.SYNTHETIC_DATASET_NAME \
                and set(features) == {"L_mm", "G_mm"}:
            st.markdown("**Synthetic oracle check**")
            rows = [{"design": c["source"], "pred S11 (dB)": c["predictions"].get(S11),
                     "formula S11 (dB)": float(synthetic.s11_db_clean(c["parameters"]["L_mm"], c["parameters"]["G_mm"])),
                     "pred gain (dBi)": c["predictions"].get(GAIN),
                     "formula gain (dBi)": float(synthetic.gain_dbi_clean(c["parameters"]["L_mm"], c["parameters"]["G_mm"]))}
                    for c in [{**base, "source": "baseline"}] + cands]
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
            st.caption("Only possible for synthetic data: compares predictions with the noise-free generating formula. "
                       "It measures error against an assumed formula, not against real antenna physics.")


# ================================================================================== page
def render() -> None:
    df, ds, warnings = active_frame()
    theme.page_header("Predict & Optimize",
                      "Predict the performance of one antenna geometry, or search for a better geometry within "
                      "bounds. Both use the trained surrogate models.")
    if ds is None or df is None:
        theme.context_bar(ds)
        theme.empty_state("No dataset available", "Load a dataset on the Dataset & Model page first.", ":material/table:")
        st.page_link(pages()["dataset"], label="Go to Dataset & Model", icon=":material/arrow_forward:")
        return
    models, error = load_active_models(ds["id"])
    theme.context_bar(ds, len(models))
    theme.synthetic_notice(ds)
    for w in warnings:
        st.warning(w, icon=":material/warning:")
    if error:
        st.error(error, icon=":material/error:")
    if not models:
        theme.empty_state("No trained models yet",
                          "Predictions and optimization need a trained model for each target. Train them on the "
                          "Dataset & Model page (about 30 seconds for the default set).", ":material/model_training:")
        st.page_link(pages()["dataset"], label="Go to Dataset & Model", icon=":material/arrow_forward:")
        return
    features = next(iter(models.values())).features
    ranges = workflow.model_training_ranges(models)

    theme.section("Predict performance")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        params, predict = _parameter_panel(ds, df, features, ranges)
    if predict:
        try:
            result, _ = workflow.predict_and_maybe_save(get_repo(), models, params, save=False)
            st.session_state[f"prediction_{ds['id']}"] = result
        except ValidationError as exc:
            st.session_state.pop(f"prediction_{ds['id']}", None)
            with left:
                st.error(f"Invalid input: {exc}", icon=":material/error:")
    with right:
        _prediction_panel(ds, models, params)

    theme.section("Optimize geometry",
                  "A separate operation: Differential Evolution searches the bounds for the design that best meets "
                  "the objective, calling the same surrogate models. It does not change the prediction above.")
    _optimization_form(ds, df, models, features, ranges, params)
    theme.section("Baseline vs candidate")
    _optimization_results(ds, df, models, features)
