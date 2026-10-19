"""Page 3 - ML Model Lab: train, tune, compare and select surrogate models."""

from __future__ import annotations

import ast

import pandas as pd
import streamlit as st

import config
from ml.evaluation import METRIC_EXPLANATIONS, comparison_table, dataset_size_warning, improvement_statement
from ml.model_registry import MODEL_SPECS, model_label
from ml.training import TrainingConfig
from services import workflow
from ui.common import active_frame, get_repo, page_header, provenance_banner
from visualization import plots


def _search_space_editor(model_keys: list[str], n_features: int) -> dict:
    custom = {}
    with st.expander("Hyperparameter search spaces (editable)"):
        st.caption("Python-literal dictionaries of lists, e.g. {'model__max_depth': [None, 8]}. "
                   "Gaussian Process kernels are not editable here.")
        for key in model_keys:
            spec = MODEL_SPECS[key]
            if not spec.tunable or key == "gaussian_process":
                continue
            default = {k: v for k, v in spec.search_space(n_features).items()}
            text = st.text_area(spec.label, value=repr(default), key=f"space_{key}")
            try:
                parsed = ast.literal_eval(text)
                if not isinstance(parsed, dict) or not all(isinstance(v, list) and v for v in parsed.values()):
                    raise ValueError
                if parsed != default:
                    custom[key] = parsed
            except (ValueError, SyntaxError):
                st.error(f"Invalid search space for {spec.label}; the default will be used.")
    return custom


def _results_section(ds: dict) -> None:
    repo = get_repo()
    exps = repo.list_experiments(dataset_id=ds["id"])
    if not exps:
        st.info("No experiments yet for this dataset.")
        return
    groups = list(dict.fromkeys(e["run_group"] for e in exps))
    group = st.selectbox("Training run", groups, help="Each click of 'Train' creates a run group.")
    run = [e for e in exps if e["run_group"] == group]
    table = comparison_table(run)
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("What do these metrics mean?"):
        for k, v in METRIC_EXPLANATIONS.items():
            st.markdown(f"**{k}** - {v}")
    warns = {w for e in run for w in (e["extra_metrics"] or {}).get("warnings", [])}
    for w in warns:
        st.warning(w)
    metric = st.radio("Comparison metric", ["test_R2", "test_RMSE", "test_MAE", "cv_R2_mean"], horizontal=True)
    fig = plots.metric_comparison(table, metric)
    if fig:
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Tuned vs default (measured on the held-out test set):**")
    stmts = []
    for e in run:
        if e["variant"] == "tuned":
            d = next((x for x in run if x["model_type"] == e["model_type"] and x["target_variable"] == e["target_variable"]
                      and x["variant"] == "default"), None)
            if d:
                stmts.append(improvement_statement(d, e, e["target_variable"], model_label(e["model_type"])))
    st.write("\n".join(f"- {s}" for s in stmts) if stmts else "No tuned models in this run.")

    st.subheader("Diagnostic plots")
    choice = st.selectbox("Experiment", [e["id"] for e in run if e["artifact_path"]],
                          format_func=lambda i: next(f"#{e['id']} {model_label(e['model_type'])} ({e['variant']}) -> {e['target_variable']}"
                                                     for e in run if e["id"] == i))
    exp = repo.get_experiment(choice)
    try:
        lm = workflow.load_models({exp["target_variable"]: exp})[exp["target_variable"]]
    except Exception as exc:
        st.error(f"Could not load artefact: {exc}")
        return
    p = lm.payload
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.actual_vs_predicted(p["y_test"], p["y_pred"], lm.target, y_std=p.get("y_std")), width="stretch")
    c2.plotly_chart(plots.residual_plot(p["y_test"], p["y_pred"], lm.target), width="stretch")
    with st.expander("Hyperparameters and validation configuration"):
        st.json({"hyperparameters": exp["hyperparameters"], "validation": exp["validation_config"],
                 "cv": (exp["extra_metrics"] or {}).get("cv"), "artifact": exp["artifact_path"]})

    st.subheader("Active model per target")
    st.caption("Used by Prediction, Optimization and XAI pages. Auto-selected by CV R2; you may override.")
    active = workflow.get_active_experiments(repo, ds["id"])
    for target in ds["target_names"]:
        options = [e["id"] for e in exps if e["target_variable"] == target and e["artifact_path"]]
        if not options:
            continue
        current = active.get(target, {}).get("id")
        key = f"active_{target}_{ds['id']}"
        # Keep the widget in sync with the database (e.g. after auto-selection by a new training run);
        # only an explicit user change (on_change) writes a new active model.
        if current in options:
            st.session_state[key] = current

        def _apply(target=target, key=key):
            workflow.set_active_model(repo, ds["id"], target, st.session_state[key])
            st.session_state["active_changed"] = (target, st.session_state[key])

        st.selectbox(f"Active model for {target}", options, key=key, on_change=_apply,
                     format_func=lambda i: next(f"#{e['id']} {model_label(e['model_type'])} ({e['variant']}) "
                                                f"CV R2={e['cv_r2_mean'] if e['cv_r2_mean'] is None else round(e['cv_r2_mean'], 3)}"
                                                for e in exps if e["id"] == i))
    changed = st.session_state.pop("active_changed", None)
    if changed:
        st.success(f"Active model for {changed[0]} set to experiment #{changed[1]}.")
        if next(e for e in exps if e["id"] == changed[1])["model_type"] == "dummy":
            st.warning("The Dummy baseline is not a useful predictor; select it only for comparison.")


def render() -> None:
    page_header("ML Model Lab", "Train, validate, tune and compare regression surrogates.")
    df, ds = active_frame()
    provenance_banner(ds)
    if df is None:
        return
    with st.form("train_form"):
        c1, c2 = st.columns(2)
        features = c1.multiselect("Input features", ds["input_features"], default=ds["input_features"])
        targets = c2.multiselect("Targets", ds["target_names"], default=ds["target_names"])
        model_keys = st.multiselect("Models", list(MODEL_SPECS), default=["dummy", "random_forest", "extra_trees",
                                                                            "gradient_boosting", "gaussian_process"],
                                    format_func=model_label)
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Test split fraction", 0.1, 0.4, config.DEFAULT_TEST_SIZE, 0.05)
        seed = c2.number_input("Random seed", 0, 10_000, config.RANDOM_SEED)
        folds = c3.slider("CV folds (training data)", 3, 10, config.DEFAULT_CV_FOLDS)
        c1, c2 = st.columns(2)
        tune = c1.toggle("Hyperparameter tuning (RandomizedSearchCV)", value=False,
                         help="Off = quick demo. GP tuning compares RBF / Matern kernels and is the slowest.")
        n_iter = c2.slider("Tuning iterations per model", 2, 30, config.DEFAULT_TUNING_ITERATIONS)
        notes = st.text_input("Notes for the experiment log (optional)")
        custom = _search_space_editor(model_keys, len(features) or 1) if tune else {}
        submitted = st.form_submit_button("Train and evaluate", type="primary")
    with st.expander("Model descriptions"):
        for spec in MODEL_SPECS.values():
            st.markdown(f"**{spec.label}** - {spec.description}" + (f" *Uncertainty: {spec.uncertainty}*" if spec.uncertainty else ""))
    if submitted:
        n = int(df[features + targets[:1]].dropna().shape[0]) if features and targets else 0
        msg = dataset_size_warning(n)
        if msg:
            st.warning(msg)
        try:
            cfg = TrainingConfig(features=features, targets=targets, model_keys=model_keys, test_size=test_size,
                                 seed=int(seed), cv_folds=folds, tune=tune, tuning_iterations=n_iter,
                                 custom_search_spaces=custom)
            bar = st.progress(0.0, "Starting")
            with st.spinner("Training models - Gaussian Process and tuning can take up to a minute..."):
                exps, _ = workflow.train_and_log(get_repo(), ds["id"], cfg, notes=notes,
                                                 progress=lambda f, m: bar.progress(min(f, 1.0), m))
            st.success(f"Logged {len(exps)} experiment(s) to SQLite.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    st.divider()
    _results_section(ds)
