"""Design & Optimize: choose a dataset/model pair, predict one design, optimize with Differential Evolution.

Each dataset keeps its own parameter form: inputs are never passed to a model trained on a
different schema. Prediction and optimization are separate operations with separate actions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

import config
from datasets import synthetic
from datasets.public import SOURCES, read_status
from ml.explainability import design_space_grid
from ml.model_registry import model_label
from ml.uncertainty import UNCERTAINTY_DISCLAIMER
from optimization.constraints import Constraint
from optimization.objectives import DIRECTION, ObjectiveConfig
from services import workflow
from ui import theme
from ui.common import active_frame, fmt_unc, get_repo, load_active_models
from ui.navigation import pages
from utils.validation import ValidationError, validate_bounds, validate_geometry
from visualization import plots

ALGORITHM = "differential_evolution"
FREQUENCY_NOTES = {
    workflow.SYNTHETIC_DATASET_NAME: f"S11 refers to the single design frequency {synthetic.F0_GHZ:g} GHz assumed by "
                                     "the synthetic generator.",
    SOURCES["metu_simpl"]["dataset_name"]: "Frequency is not an input. S11 targets come from the HFSS sweep "
                                           "(23–33 GHz, 0.05 GHz steps); “S11 at 28 GHz” is read at exactly 28.00 GHz.",
}


def _ordered_targets(targets) -> list[str]:
    """S11 targets first (headline result), then the rest in stored order."""
    targets = list(targets)
    return sorted(targets, key=lambda t: (not t.startswith("S11"), targets.index(t)))


def _delta_color(target: str) -> str:
    return {"minimize": "inverse", "maximize": "normal"}.get(DIRECTION.get(target), "off")


def _inward(lo: float, hi: float) -> tuple[float, float]:
    """Round bounds inward to 4 decimals so default bounds never extend past the training range."""
    return math.ceil(lo * 1e4) / 1e4, math.floor(hi * 1e4) / 1e4


def _num(v: float) -> str:
    """Readable number: fixed decimals for small values (never scientific notation in the form)."""
    return f"{v:.4g}" if abs(v) >= 1 else (f"{v:.4f}".rstrip("0").rstrip(".") or "0")


def _fmt_param(f: str, v: float) -> str:
    u = config.unit_of(f)
    return f"{_num(v)} {u}".strip()


# ========================================================================= dataset choice
def _dataset_options() -> dict[int, str]:
    """Datasets whose pipeline is complete (models trained), with short labels."""
    repo = get_repo()
    labels: dict[int, str] = {}
    public_names = {meta["dataset_name"] for meta in SOURCES.values()}
    for key, meta in SOURCES.items():
        ds = repo.get_dataset_by_name(meta["dataset_name"])   # by name: ids differ between machines
        if read_status(key).get("complete") and ds and workflow.get_active_experiments(repo, ds["id"]):
            labels[ds["id"]] = meta["label"]
    for d in repo.list_datasets():
        if d["name"] in public_names:
            continue                                           # public sources appear only when complete
        if d["id"] not in labels and workflow.get_active_experiments(repo, d["id"]):
            labels[d["id"]] = "Synthetic demo" if d["dataset_type"] == "SYNTHETIC" else d["name"]
    return labels


def _dataset_selector(ds: dict | None) -> None:
    options = _dataset_options()
    unavailable = [f"{m['label']}: {read_status(k).get('message') or 'pipeline not run'}"
                   for k, m in SOURCES.items() if not read_status(k).get("complete")]
    if not options:
        return
    ids = list(options)
    current = ds["id"] if ds and ds["id"] in options else ids[0]
    choice = st.segmented_control("Dataset", ids, default=current, format_func=options.get,
                                  key="dataset_selector", help="Each dataset has its own trained models and inputs.")
    if choice is not None and (ds is None or choice != ds["id"]):
        workflow.set_active_dataset_id(get_repo(), choice)
        st.rerun()
    for note in unavailable:
        st.caption(f":material/block: Not available — {note}")


def _data_notice(ds: dict) -> None:
    if ds["dataset_type"] == "SYNTHETIC":
        theme.synthetic_notice(ds)
    else:
        st.info(f"**{config.DATASET_TYPES[ds['dataset_type']].split(',')[0]}.** Models were trained on "
                f"{ds['name']}. Predictions are ML estimates of the simulated response — not new simulations or "
                "measurements.", icon=":material/memory:")


# ============================================================================ prediction
def _parameter_panel(ds: dict, df: pd.DataFrame, features: list[str], ranges: dict) -> tuple[dict, bool]:
    with theme.card("design-params"):
        st.markdown("**Antenna parameters**")
        params = {}
        groups: dict[str, list[str]] = {}
        for f in features:
            groups.setdefault(config.KNOWN_COLUMNS.get(f, {}).get("group", "Parameters"), []).append(f)
        for group, members in groups.items():
            if len(groups) > 1:
                st.caption(f"**{group}**")
            cols = st.columns(2 if len(members) > 1 else 1)
            for i, f in enumerate(members):
                lo, hi = ranges[f]
                key = f"geom_{ds['id']}_{f}"
                if key not in st.session_state:
                    st.session_state[key] = round(float(df[f].median()), 4)
                unit = config.unit_of(f)
                with cols[i % len(cols)]:
                    params[f] = st.number_input(f"{config.short_label(f)}{f' ({unit})' if unit else ''}", key=key,
                                                step=0.01 if hi - lo > 0.5 else 0.001, format="%.4f",
                                                help=f"{config.KNOWN_COLUMNS.get(f, {}).get('description', f)}. "
                                                     f"Trained on {_num(lo)} – {_num(hi)}{f' {unit}' if unit else ''}.")
                    st.caption(f"Valid {_num(lo)} – {_num(hi)}")
        check = validate_geometry(params, features, ranges)
        for e in check.errors:
            st.error(e, icon=":material/error:")
        for w in check.warnings:
            st.warning(w, icon=":material/warning:")
        st.caption(FREQUENCY_NOTES.get(ds["name"], "Frequency is not an input column in this dataset."))
        predict = st.button("Predict performance", type="primary", icon=":material/bolt:", width="stretch",
                            disabled=not check.ok)
    return {f: params[f] for f in features}, predict


def _prediction_panel(ds: dict, models: dict, params: dict) -> None:
    result = st.session_state.get(f"prediction_{ds['id']}")
    with theme.card("design-prediction"):
        st.markdown("**Performance prediction**")
        if result is None:
            st.caption("Enter the parameters on the left and press **Predict performance**. The predicted values "
                       "appear here with the model's uncertainty and validation metrics.")
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
        targets = _ordered_targets(result.predictions)
        cols = st.columns(min(2, len(targets)))
        for i, t in enumerate(targets):
            with cols[i % len(cols)]:
                st.metric(config.short_label(t), theme.fmt_value(result.predictions[t], t, 3 if "GHz" in t else 2),
                          help=config.KNOWN_COLUMNS.get(t, {}).get("description"))
                st.caption(f"Uncertainty {fmt_unc(result.uncertainty.get(t), config.unit_of(t))}")
        rows, partial = [], []
        for t in targets:
            e = models[t].experiment
            n_used = (e["validation_config"] or {}).get("n_samples_used")
            if n_used and n_used < ds["sample_count"]:
                partial.append(f"{config.short_label(t)} ({n_used:,} of {ds['sample_count']:,})")
            rows.append({"Target": config.short_label(t), "Model": model_label(e["model_type"]),
                         "Test R²": e["r2"], "Test RMSE": e["rmse"], "Unit": config.unit_of(t), "CV R²": e["cv_r2_mean"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Test R²": st.column_config.NumberColumn(format="%.3f", help="Variance explained on held-out data (1 = perfect)."),
            "Test RMSE": st.column_config.NumberColumn(format="%.3f", help="Typical error, in the unit shown."),
            "CV R²": st.column_config.NumberColumn(format="%.3f", help="Cross-validated R² on training data."),
        })
        if partial:
            st.caption("Trained only on designs where the quantity is defined inside the simulated band: "
                       + ", ".join(partial) + ". For other designs these predictions are not meaningful.")
        st.caption("ML prediction — not a simulation or measurement. " + UNCERTAINTY_DISCLAIMER)
        if st.button("Save prediction to history", icon=":material/bookmark_add:", disabled=stale):
            rec = workflow.save_prediction(get_repo(), models, result)
            st.toast(f"Saved as prediction #{rec['id']}.", icon=":material/check_circle:")


# ========================================================================== optimization
def _default_objective(targets: list[str]) -> str:
    if "Gain_dBi" in targets:
        return "Gain_dBi"
    return next((t for t in _ordered_targets(targets) if t in DIRECTION), targets[0])


def _optimization_form(ds: dict, df: pd.DataFrame, models: dict, features: list[str], ranges: dict,
                       params: dict) -> None:
    targets = [t for t in _ordered_targets(models)]
    objective_targets = [t for t in targets if t in DIRECTION]
    if not objective_targets:
        st.info("None of this dataset's targets has a defined optimization direction.")
        return
    norm = workflow.target_ranges(df, targets)
    with theme.card("design-optimize"):
        c_obj, c_params, c_search = st.columns([1.15, 1.45, 0.9], gap="large")

        with c_obj:
            st.markdown("**Objective**")
            default_t = _default_objective(objective_targets)
            target = st.selectbox("Objective", objective_targets, index=objective_targets.index(default_t),
                                  format_func=lambda t: f"{DIRECTION[t].capitalize()} {config.short_label(t)}",
                                  key=f"obj_{ds['id']}", label_visibility="collapsed")
            constraints = []
            others = [t for t in targets if t != target]
            if others:
                default_on = "S11_dB" in others and target == "Gain_dBi"
                if st.checkbox("Add a constraint", value=default_on, key=f"con_on_{ds['id']}"):
                    ct = st.selectbox("Constrained target", others, format_func=config.short_label,
                                      index=others.index("S11_dB") if "S11_dB" in others else 0, key=f"con_t_{ds['id']}")
                    c1, c2 = st.columns([1, 1.4])
                    op = c1.selectbox("Rule", ["<=", ">="], index=0 if DIRECTION.get(ct) != "maximize" else 1,
                                      format_func=lambda o: {"<=": "≤", ">=": "≥"}[o], key=f"con_op_{ds['id']}")
                    default_lim = config.DEFAULT_S11_LIMIT_DB if ct.startswith("S11") else round(float(df[ct].median()), 2)
                    lim = c2.number_input(f"Limit ({config.unit_of(ct)})", value=float(default_lim),
                                          key=f"con_lim_{ds['id']}_{ct}")
                    constraints.append(Constraint(ct, op, float(lim)))
            st.caption("Constraints are checked on predictions and enforced with a penalty.")

        with c_params:
            st.markdown("**Parameters and bounds**")
            default_free = [f for f in features if config.KNOWN_COLUMNS.get(f, {}).get("optimize_default")] or features
            free = st.multiselect("Parameters to optimize", features, default=default_free,
                                  format_func=config.short_label, key=f"free_{ds['id']}",
                                  help="The others stay at the baseline value (e.g. materials and layer thicknesses).")
            bounds = {}
            for f in free:
                lo, hi = _inward(*ranges[f])
                b1, b2 = st.columns(2)
                unit = config.unit_of(f)
                b_lo = b1.number_input(f"{config.short_label(f)} min{f' ({unit})' if unit else ''}", value=lo,
                                       format="%.4f", key=f"blo_{ds['id']}_{f}")
                b_hi = b2.number_input(f"{config.short_label(f)} max{f' ({unit})' if unit else ''}", value=hi,
                                       format="%.4f", key=f"bhi_{ds['id']}_{f}")
                bounds[f] = (b_lo, b_hi)
            fixed = [f for f in features if f not in free]
            if fixed:
                st.caption(f"Held at baseline: {', '.join(config.short_label(f) for f in fixed)}.")
            baseline_mode = st.radio("Baseline design", ["Parameters entered above", "Median training design"],
                                     help="The design the candidate is compared against, and the source of the "
                                          "held values.", key=f"base_{ds['id']}")

        with c_search:
            st.markdown("**Differential Evolution**")
            budget = st.number_input("Evaluation budget", 100, 20_000, config.DEFAULT_EVAL_BUDGET, step=100,
                                     help="Maximum number of surrogate evaluations.", key=f"budget_{ds['id']}")
            seed = st.number_input("Random seed", 0, 10_000, config.RANDOM_SEED, key=f"opt_seed_{ds['id']}")
            st.caption("SciPy differential_evolution, Latin-hypercube start, no gradient polish. Bounds may not "
                       "leave the training range.")

        cfg = ObjectiveConfig(kind=f"{DIRECTION[target]}_target", target=target, constraints=constraints,
                              normalization=norm)
        problems = []
        if not free:
            problems.append("Choose at least one parameter to optimize.")
        problems += validate_bounds(bounds, ranges, False).errors if bounds else []
        try:
            cfg.validate(targets)
        except ValueError as exc:
            problems.append(str(exc))
        baseline = ({f: float(v) for f, v in params.items()} if baseline_mode.startswith("Parameters")
                    else {f: float(df[f].median()) for f in features})
        base_check = validate_geometry(baseline, features, ranges)
        problems += base_check.errors
        if fixed and base_check.warnings:
            problems.append("The baseline lies outside the training range; held parameters would be extrapolated. "
                            "Adjust the parameters above or use the median training design.")
        for p in problems:
            st.error(p + (" Narrow the bounds to the valid range." if "training range" in p and "Bounds" in p else ""),
                     icon=":material/error:")
        run = st.button("Run optimization", type="primary", icon=":material/play_arrow:", disabled=bool(problems))
    if run:
        try:
            with st.spinner("Searching the design space with the surrogate…"):
                outs = workflow.run_optimization(get_repo(), models, cfg, bounds, baseline, [ALGORITHM],
                                                 seed=int(seed), budget=int(budget), allow_extrapolation=False)
            st.session_state[f"optimization_{ds['id']}"] = {"outs": outs, "cfg": cfg, "bounds": bounds,
                                                             "baseline": baseline}
            st.toast(f"Optimization finished · saved as run #{outs[0][1]['id']}.", icon=":material/check_circle:")
        except (ValidationError, ValueError) as exc:
            st.error(f"Optimization could not run: {exc}", icon=":material/error:")
        except Exception as exc:
            st.error(f"Optimization failed unexpectedly: {exc}. Try a smaller budget or retrain the models.",
                     icon=":material/error:")


def _design_card(title: str, design: dict, show: list[str], targets: list[str], badge, ref: dict | None = None) -> None:
    with theme.card("design-baseline" if ref is None else "design-candidate"):
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.markdown(f"**{title}**")
            badge()
        pcols = st.columns(min(3, len(show)))
        for i, f in enumerate(show):
            v = design["parameters"][f]
            delta = None if ref is None else f"{v - ref['parameters'][f]:+.4g} {config.unit_of(f)}".strip()
            pcols[i % len(pcols)].metric(config.short_label(f), _fmt_param(f, v), delta=delta, delta_color="off")
        tcols = st.columns(min(2, len(targets)))
        for i, t in enumerate(targets):
            v = design["predictions"][t]
            delta = None if ref is None else f"{v - ref['predictions'][t]:+.2f} {config.unit_of(t)}"
            tcols[i % len(tcols)].metric(f"Predicted {config.short_label(t)}", theme.fmt_value(v, t),
                                         delta=delta, delta_color=_delta_color(t))
        unc = [f"{config.short_label(t)} ± {design['uncertainty'][t]:.2f} {config.unit_of(t)}"
               for t in targets if design["uncertainty"].get(t) is not None]
        if unc:
            st.caption("Approx. model uncertainty (1σ): " + ", ".join(unc))


def _constraint_badge(design: dict, cfg: ObjectiveConfig):
    def _badge():
        if not cfg.constraints:
            st.badge("Unconstrained", color="gray")
        elif design["feasible"]:
            st.badge("Meets " + ", ".join(f"{config.short_label(c.target)} {c.operator.replace('<=', '≤').replace('>=', '≥')} "
                                          f"{c.limit:g}" for c in cfg.constraints), color="green", icon=":material/check:")
        else:
            st.badge("Violates constraint", color="red", icon=":material/close:")
    return _badge


def _nearest_simulated(ds: dict, df: pd.DataFrame, features: list[str], designs: list[dict]) -> None:
    """Closest design that was actually simulated: a sanity reference, not a validation of the candidate."""
    X = df[features].to_numpy(float)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    rows = []
    for d in designs:
        x = np.array([d["parameters"][f] for f in features])
        dist = np.sqrt((((X - x) / scale) ** 2).mean(axis=1))
        i = int(dist.argmin())
        row = {"design": d["source"], "nearest simulated row": int(df.index[i]) + 1, "normalized distance": dist[i]}
        for t in ds["target_names"]:
            row[f"HFSS {config.short_label(t)}"] = df.iloc[i][t]
            row[f"predicted {config.short_label(t)}"] = d["predictions"].get(t)
        rows.append(row)
    st.markdown("**Nearest simulated design**")
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption("The closest design that was actually simulated in the dataset (normalized parameter distance; 0 = "
               "identical). It is a sanity reference only: the candidate itself has not been simulated.")


def _optimization_results(ds: dict, df: pd.DataFrame, models: dict, features: list[str]) -> None:
    res = st.session_state.get(f"optimization_{ds['id']}")
    if not res:
        theme.empty_state("No optimization run yet",
                          "Choose the objective, parameters and bounds above and press Run optimization. The baseline "
                          "and the ML-selected candidate will be compared here.", ":material/tune:")
        return
    outcome, record, base = res["outs"][0]
    cfg, bounds = res["cfg"], res["bounds"]
    cand = {**outcome.candidate, "source": "candidate"}
    targets = _ordered_targets(models)
    free = cand.get("free_parameters") or list(bounds)
    left, right = st.columns(2, gap="medium")
    with left:
        _design_card("Baseline design", base, free, targets, _constraint_badge(base, cfg))
    with right:
        _design_card("ML-selected candidate · Differential Evolution", cand, free, targets,
                     _constraint_badge(cand, cfg), ref=base)
    held = [f for f in features if f not in free]
    st.caption(f"Objective: {DIRECTION.get(cfg.target, '')} {config.short_label(cfg.target)}. "
               + (f"Held at baseline: {', '.join(config.short_label(f) for f in held)}. " if held else "")
               + "Green marks a change that is better for that target. "
               f"{outcome.n_evals} surrogate evaluations in {outcome.runtime_s:.2f} s · saved as run #{record['id']}. "
               "**This candidate was selected by an ML surrogate and has not been simulated or measured** — verify it "
               "with full-wave EM simulation (e.g. the dataset's HFSS model) before relying on it.")
    with st.expander("Search diagnostics"):
        theme.chart(plots.convergence_plot({"Differential Evolution": outcome.history}),
                    "Best objective found so far. Early high values are penalties for designs that break a constraint.",
                    key="conv")
        if len(free) >= 2:
            fx, fy = free[0], free[1]
            fixed_vals = {f: base["parameters"][f] for f in features}
            grid = design_space_grid(models[cfg.target].pipeline, features, fx, fy, bounds, fixed_vals, n=60)
            marks = pd.DataFrame([{fx: d["parameters"][fx], fy: d["parameters"][fy], "label": lbl}
                                  for d, lbl in ((base, "baseline"), (cand, "candidate"))])
            theme.chart(plots.design_space_map(grid, grid["mean"],
                                               f"Predicted {config.short_label(cfg.target)} over {config.short_label(fx)} "
                                               f"and {config.short_label(fy)}", config.label_of(cfg.target),
                                               markers=marks),
                        "Other parameters held at the baseline values.", key="dsmap")
        if ds["dataset_type"] == "SYNTHETIC" and ds["name"] == workflow.SYNTHETIC_DATASET_NAME:
            rows = [{"design": d["source"], "pred S11 (dB)": d["predictions"].get("S11_dB"),
                     "formula S11 (dB)": float(synthetic.s11_db_clean(d["parameters"]["L_mm"], d["parameters"]["G_mm"])),
                     "pred gain (dBi)": d["predictions"].get("Gain_dBi"),
                     "formula gain (dBi)": float(synthetic.gain_dbi_clean(d["parameters"]["L_mm"], d["parameters"]["G_mm"]))}
                    for d in ({**base, "source": "baseline"}, cand)]
            st.markdown("**Synthetic oracle check**")
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
            st.caption("Only possible for synthetic data: compares predictions with the noise-free generating formula.")
        else:
            _nearest_simulated(ds, df, features, [{**base, "source": "baseline"}, cand])


# ================================================================================== page
def render() -> None:
    df, ds, warnings = active_frame()
    theme.page_header("Design & Optimize",
                      "Choose a dataset, predict the performance of one antenna design, or search for a better design "
                      "within bounds. Each dataset uses its own trained models and inputs.")
    _dataset_selector(ds)
    if ds is None or df is None:
        theme.context_bar(ds)
        theme.empty_state("No dataset available", "Load a dataset on the Home page first.", ":material/table:")
        st.page_link(pages()["home"], label="Go to Home", icon=":material/arrow_forward:")
        return
    models, error = load_active_models(ds["id"])
    theme.context_bar(ds, len(models))
    _data_notice(ds)
    for w in warnings:
        st.warning(w, icon=":material/warning:")
    if error:
        st.error(error, icon=":material/error:")
    if not models:
        theme.empty_state("No trained models yet",
                          "Predictions and optimization need a trained model for each target. Train them on the Home "
                          "page.", ":material/model_training:")
        st.page_link(pages()["home"], label="Go to Home", icon=":material/arrow_forward:")
        return
    features = next(iter(models.values())).features
    ranges = workflow.model_training_ranges(models)

    theme.section("Predict performance")
    left, right = st.columns([1.1, 1.2], gap="large")
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

    theme.section("Optimize design",
                  "A separate operation: Differential Evolution searches the bounds for the design that best meets "
                  "the objective, calling the same models. It does not change the prediction above.")
    _optimization_form(ds, df, models, features, ranges, params)
    theme.section("Baseline vs candidate")
    _optimization_results(ds, df, models, features)
