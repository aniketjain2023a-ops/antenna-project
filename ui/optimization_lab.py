"""Page 5 - Optimization Lab: surrogate-based geometry optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config
from datasets import synthetic
from ml.explainability import design_space_grid
from optimization import ALGORITHMS
from optimization.analysis import rank_candidates, sample_design_space
from optimization.constraints import Constraint
from optimization.objectives import OBJECTIVE_DESCRIPTIONS, ObjectiveConfig, SurrogateEvaluator
from services import workflow
from ui.common import active_frame, active_models_or_stop, get_repo, page_header, provenance_banner
from utils.validation import ValidationError
from visualization import plots

S11, GAIN = "S11_dB", "Gain_dBi"


def _objective_form(models: dict, norm: dict) -> ObjectiveConfig | None:
    targets = list(models)
    kinds = [k for k in OBJECTIVE_DESCRIPTIONS
             if (k == "max_gain" and GAIN in targets) or (k == "min_s11" and S11 in targets) or k == "tradeoff"]
    kind = st.selectbox("Objective", kinds, format_func=lambda k: OBJECTIVE_DESCRIPTIONS[k])
    constraints: list[Constraint] = []
    weights: dict[str, float] = {}
    c1, c2, c3 = st.columns(3)
    if kind == "max_gain":
        lim = c1.number_input("Constraint: S11 <= [dB]", value=config.DEFAULT_S11_LIMIT_DB, step=0.5) if S11 in targets else None
        if lim is not None:
            constraints.append(Constraint(S11, "<=", float(lim)))
        else:
            c1.info("No S11 model: unconstrained gain maximisation.")
    elif kind == "min_s11":
        lim = c1.number_input("Constraint: Gain >= [dBi]", value=config.DEFAULT_MIN_GAIN_DBI, step=0.5) if GAIN in targets else None
        if lim is not None:
            constraints.append(Constraint(GAIN, ">=", float(lim)))
    else:
        st.caption("Each target is normalised to [0, 1] over its observed data range (1 = best) before weighting, "
                   "so dB and dBi are never added directly.")
        for i, t in enumerate(t for t in targets if t in (S11, GAIN, "Bandwidth_MHz")):
            weights[t] = [c1, c2, c3][i % 3].slider(f"Weight on {t}", 0.0, 1.0, 0.5, 0.05)
        if S11 in targets and st.checkbox("Also enforce S11 <= limit", value=True):
            constraints.append(Constraint(S11, "<=", float(st.number_input("S11 limit [dB]", value=-10.0, step=0.5))))
        if GAIN in targets and st.checkbox("Also enforce minimum gain", value=False):
            constraints.append(Constraint(GAIN, ">=", float(st.number_input("Gain limit [dBi]", value=6.0, step=0.5))))
        if "Bandwidth_MHz" not in targets:
            st.caption("Bandwidth is not part of this dataset, so it is not used (it is never invented).")
    kappa = st.slider("Risk aversion kappa (use mean -/+ kappa*sigma where uncertainty exists; 0 = off)", 0.0, 3.0, 0.0, 0.25)
    cfg = ObjectiveConfig(kind=kind, constraints=constraints, weights=weights, normalization=norm, kappa=kappa)
    try:
        cfg.validate(targets)
    except ValueError as exc:
        st.error(str(exc))
        return None
    st.caption("Constraint handling: penalised objective = raw objective + "
               f"{cfg.penalty_weight:g} x sum(violation / target data range).")
    return cfg


def render() -> None:
    page_header("Optimization Laboratory", "Search the geometry space using the trained surrogate models.")
    df, ds = active_frame()
    provenance_banner(ds)
    if ds is None or df is None:
        return
    models = active_models_or_stop(ds)
    st.warning("Results are **ML-selected candidate designs** predicted by a surrogate model. They are NOT physically "
               "validated antennas. A deeper S11 alone does not prove better gain, bandwidth, efficiency or radiation "
               "performance. Verify every candidate with full-wave EM simulation and measurement.", icon="🧪")
    features = next(iter(models.values())).features
    train_ranges = workflow.model_training_ranges(models)
    norm = workflow.target_ranges(df, list(models))

    st.subheader("1. Objective and constraints")
    cfg = _objective_form(models, norm)
    if cfg is None:
        return

    st.subheader("2. Parameter bounds and baseline design")
    allow_ext = st.checkbox("Allow bounds outside the training range (extrapolation - not recommended)", value=False)
    bounds, baseline = {}, {}
    cols = st.columns(len(features))
    medians = df[features].median()
    for col, f in zip(cols, features):
        lo, hi = train_ranges[f]
        col.markdown(f"**{config.label_of(f)}** (training range {lo:.4g}-{hi:.4g})")
        b_lo = col.number_input(f"{f} lower", value=float(lo), format="%.4f", key=f"blo_{f}")
        b_hi = col.number_input(f"{f} upper", value=float(hi), format="%.4f", key=f"bhi_{f}")
        bounds[f] = (b_lo, b_hi)
        baseline[f] = col.number_input(f"{f} baseline", value=float(round(medians[f], 4)), format="%.4f", key=f"base_{f}")
    st.caption("Default baseline = median training design. Replace it with your current reference antenna geometry.")

    st.subheader("3. Algorithms and budget")
    c1, c2, c3 = st.columns(3)
    algos = c1.multiselect("Algorithms", list(ALGORITHMS), default=["differential_evolution", "multistart_nelder_mead"],
                           format_func=lambda a: ALGORITHMS[a][0])
    seed = c2.number_input("Random seed", 0, 10_000, config.RANDOM_SEED, key="opt_seed")
    budget = c3.number_input("Evaluation budget per algorithm", 100, 20_000, config.DEFAULT_EVAL_BUDGET, step=100)

    if st.button("Run optimization", type="primary"):
        try:
            with st.spinner("Optimizing on the surrogate..."):
                outs = workflow.run_optimization(get_repo(), models, cfg, bounds, baseline, algos, seed=int(seed),
                                                 budget=int(budget), allow_extrapolation=allow_ext)
            st.session_state["opt_result"] = {"dataset": ds["id"], "outs": outs, "cfg": cfg, "bounds": bounds}
        except (ValidationError, ValueError) as exc:
            st.error(f"Cannot run: {exc}")
    res = st.session_state.get("opt_result")
    if not res or res["dataset"] != ds["id"]:
        return
    _show_results(res, models, df, ds, features)


def _show_results(res: dict, models: dict, df: pd.DataFrame, ds: dict, features: list[str]) -> None:
    outs, cfg, bounds = res["outs"], res["cfg"], res["bounds"]
    base = outs[0][2]
    st.subheader("4. Results")
    st.success("Saved optimization run(s) " + ", ".join(f"#{rec['id']}" for _, rec, _ in outs if rec) + " to SQLite.")
    cands = [{**base, "source": "baseline"}] + [{**o.candidate, "source": o.algorithm} for o, _, _ in outs]
    ranked = rank_candidates(cands, cfg)
    st.markdown("**Ranked designs** (feasible first, then penalised objective; lower is better)")
    st.dataframe(ranked, hide_index=True, width="stretch")
    summary = pd.DataFrame([{"algorithm": o.algorithm, "evaluations": o.n_evals, "budget": o.budget,
                             "runtime_s": round(o.runtime_s, 2), "within_bounds": o.within_bounds,
                             "feasible": o.candidate["feasible"], "violations": o.candidate["violations"],
                             "message": o.message} for o, _, _ in outs])
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("Constraint violations are shown in the target unit (0 = satisfied); evaluated on predictions only.")

    rows = pd.DataFrame([{"source": c["source"], **c["predictions"],
                          **{f"{t}_std": u for t, u in c["uncertainty"].items()}} for c in cands])
    cols = st.columns(len(models))
    for col, t in zip(cols, models):
        fig = plots.baseline_vs_candidates(rows, t)
        if fig:
            col.plotly_chart(fig, width="stretch")

    fig = plots.convergence_plot({o.algorithm: o.history for o, _, _ in outs})
    if fig:
        st.plotly_chart(fig, width="stretch")

    if S11 in models and GAIN in models:
        evaluator = SurrogateEvaluator(models)
        samples = sample_design_space(evaluator, cfg, bounds, n=1200, seed=1)
        cand_df = pd.DataFrame([{"source": c["source"], **c["predictions"]} for c in cands[1:]])
        fig = plots.tradeoff_scatter(samples, S11, GAIN, cand_df, base["predictions"])
        st.plotly_chart(fig, width="stretch")
        st.caption("Pareto front = sampled designs where no other sample has both lower S11 and higher gain "
                   "(on predictions). It shows the trade-off the objective weights choose from.")

    if len(features) >= 2:
        fx, fy = features[0], features[1]
        target = GAIN if GAIN in models else next(iter(models))
        fixed = {f: float(np.median(df[f])) for f in features}
        grid = design_space_grid(models[target].pipeline, features, fx, fy, {f: bounds[f] for f in features}, fixed, n=70)
        marks = pd.DataFrame([{fx: c["parameters"][fx], fy: c["parameters"][fy], "label": c["source"][:10]} for c in cands])
        st.plotly_chart(plots.design_space_map(grid, grid["mean"], f"Predicted {target} over the design space with candidates",
                                               config.label_of(target), markers=marks), width="stretch")

    if ds["dataset_type"] == "SYNTHETIC" and ds["name"] == workflow.SYNTHETIC_DATASET_NAME and set(features) == {"L_mm", "G_mm"}:
        st.subheader("Synthetic oracle check (only possible for synthetic data)")
        oracle = []
        for c in cands:
            l, g = c["parameters"]["L_mm"], c["parameters"]["G_mm"]
            oracle.append({"design": c["source"], "L_mm": l, "G_mm": g,
                           "pred S11": c["predictions"].get(S11), "formula S11 (noise-free)": float(synthetic.s11_db_clean(l, g)),
                           "pred Gain": c["predictions"].get(GAIN), "formula Gain (noise-free)": float(synthetic.gain_dbi_clean(l, g))})
        st.dataframe(pd.DataFrame(oracle), hide_index=True, width="stretch")
        st.caption("Because the synthetic generator is known, we can compare surrogate predictions with the noise-free "
                   "formula. This measures surrogate error against an ASSUMED formula, not against real antenna physics.")
