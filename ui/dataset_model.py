"""Section 1 - Dataset & Model: load and check data, train surrogate models, read their metrics."""

from __future__ import annotations

import ast

import pandas as pd
import streamlit as st

import config
from database.repository import DuplicateRecordError
from datasets.io import excel_sheet_names, read_table
from ml.evaluation import METRIC_EXPLANATIONS, comparison_table, improvement_statement
from ml.model_registry import MODEL_SPECS, model_label
from ml.preprocessing import clean_dataframe, detect_column_types, inspect_quality, iqr_outlier_mask
from ml.training import TrainingConfig
from services import workflow
from ui import theme
from ui.common import active_frame, get_repo, load_model
from visualization import plots

DEFAULT_MODELS = ["dummy", "random_forest", "gradient_boosting"]


# =============================================================================== dataset
def _dataset_picker(ds: dict) -> None:
    repo = get_repo()
    datasets = repo.list_datasets()
    ids = [d["id"] for d in datasets]
    names = {d["id"]: d["name"] for d in datasets}
    choice = st.selectbox("Active dataset", ids, index=ids.index(ds["id"]), format_func=names.get,
                          help="Every page uses the active dataset. Registered datasets are never edited in place.")
    if choice != ds["id"]:
        workflow.set_active_dataset_id(repo, choice)
        st.rerun()


def _dataset_overview(df: pd.DataFrame, ds: dict) -> None:
    cols = ds["input_features"] + ds["target_names"]
    report = inspect_quality(df, cols)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        with theme.card("dataset-1"):
            _dataset_picker(ds)
            st.markdown("**Input features**")
            with st.container(horizontal=True, gap="small"):
                for f in ds["input_features"]:
                    st.badge(config.label_of(f), color="blue")
            st.markdown("**Target variables**")
            with st.container(horizontal=True, gap="small"):
                for t in ds["target_names"]:
                    st.badge(config.label_of(t), color="violet")
            with st.expander("Provenance"):
                st.write(ds["provenance"])
                if ds["description"]:
                    st.caption(ds["description"])
                st.caption(f"SHA-256 `{ds['checksum'][:16]}…` · registered {ds['created_at']}")
    with right:
        with theme.card("dataset-2"):
            st.markdown("**Data quality**")
            missing = sum(report.missing_per_column.values())
            outliers = sum(report.outliers_per_column.values())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Samples", f"{len(df):,}")
            m2.metric("Missing values", missing, help="Missing target values are never imputed; such rows are "
                                                       "excluded only for the affected target.")
            m3.metric("Duplicate rows", report.duplicate_rows)
            m4.metric("Outliers flagged", outliers, help="IQR rule (1.5 × IQR). Flagged for inspection only; "
                                                        "nothing is removed. Deep S11 resonances are expected outliers.")
            if missing:
                cols_missing = [c for c, n in report.missing_per_column.items() if n]
                st.warning(f"Missing values in {', '.join(cols_missing)}. They are kept as missing, not filled in.",
                           icon=":material/warning:")
            for w in report.warnings:
                if "missing" not in w:
                    st.caption(f"• {w}")
            preview, stats, dist = st.tabs(["Preview", "Statistics", "Distributions"])
            with preview:
                st.dataframe(df.head(100), height=240, width="stretch", hide_index=True)
            with stats:
                st.dataframe(df[cols].describe().T.round(3), width="stretch")
            with dist:
                theme.chart(plots.distribution_grid(df, cols), key="dist")
                if len(df) and any(report.outliers_per_column.values()):
                    col = st.selectbox("Show flagged outliers in", [c for c, n in report.outliers_per_column.items() if n])
                    st.dataframe(df.loc[iqr_outlier_mask(df[col]), cols], width="stretch", hide_index=True)


def _import_dataset() -> None:
    """Upload -> type detection -> column mapping -> quality -> provenance -> register."""
    repo = get_repo()
    up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"],
                          help="The original file is never modified. A cleaned copy is stored after you confirm it.")
    if up is None:
        st.caption("Use EM-simulation exports or measured data with one row per design. You will map columns and "
                   "record provenance before anything is saved.")
        return
    raw = up.getvalue()
    sheet = 0
    if up.name.lower().endswith((".xlsx", ".xls")):
        try:
            sheet = st.selectbox("Sheet", excel_sheet_names(raw))
        except Exception as exc:
            st.error(f"Could not read the Excel file: {exc}")
            return
    try:
        df = read_table(raw, filename=up.name, sheet_name=sheet)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}. Check that it is a CSV or Excel table with a header row.")
        return

    kinds = detect_column_types(df)
    st.success(f"Read {len(df):,} rows × {len(df.columns)} columns from {up.name}.", icon=":material/check:")
    st.dataframe(pd.DataFrame({"detected type": kinds, "missing": df.isna().sum()}), width="stretch")

    numeric_like = [c for c, k in kinds.items() if k in ("numeric", "numeric-like text")]
    c1, c2 = st.columns(2)
    features = c1.multiselect("Input (geometry) columns", numeric_like,
                              default=[c for c in numeric_like if c in ("L_mm", "G_mm")])
    targets = c2.multiselect("Target (performance) columns", [c for c in numeric_like if c not in features],
                             default=[c for c in numeric_like if c in ("S11_dB", "Gain_dBi", "Bandwidth_MHz")
                                      and c not in features])
    rename = {}
    with st.expander("Rename columns to standard names (optional)"):
        st.caption("Standard names carry units, e.g. a column 'Length (mm)' can be mapped to L_mm.")
        canon = ["(keep)"] + list(config.KNOWN_COLUMNS)
        for c in features + targets:
            choice = st.selectbox(f"'{c}' maps to", canon, key=f"map_{c}")
            if choice != "(keep)":
                rename[c] = choice
    if not features or not targets:
        st.info("Select at least one input column and one target column.", icon=":material/arrow_upward:")
        return
    new_names = [rename.get(c, c) for c in features + targets]
    if len(set(new_names)) != len(new_names):
        st.error("Two columns map to the same name. Change one of the mappings.")
        return

    c1, c2 = st.columns(2)
    drop_dups = c1.checkbox("Remove exact duplicate rows", value=True)
    drop_missing = c2.checkbox("Drop rows with missing input features", value=True)
    try:
        clean, summary = clean_dataframe(df, features, targets, rename=rename, drop_duplicates=drop_dups,
                                         drop_missing_features=drop_missing)
    except ValueError as exc:
        st.error(str(exc))
        return
    with st.expander(f"Preprocessing summary · {summary.rows_in} → {summary.rows_out} rows", expanded=True):
        for a in summary.actions:
            st.caption(f"• {a}")
    neg = [f for f in [rename.get(c, c) for c in features] if f.endswith("_mm") and (clean[f] <= 0).any()]
    if neg:
        st.error(f"{', '.join(neg)} contain zero or negative lengths. Fix the source data, then upload again.")
        return
    unknown = [c for c in clean.columns if c not in config.KNOWN_COLUMNS]
    if unknown:
        st.warning(f"{', '.join(unknown)} have no known unit in MetaOpt-AI and will be treated as plain numbers.")

    with st.form("register_form", border=False):
        name = st.text_input("Dataset name", value=up.name.rsplit(".", 1)[0])
        dtype = st.selectbox("Data type", list(config.DATASET_TYPES),
                             format_func=lambda k: f"{k.title()} — {config.DATASET_TYPES[k]}")
        provenance = st.text_area("Provenance", placeholder="Who produced the data, with which solver or instrument "
                                                            "and settings, when, and the original source.")
        description = st.text_input("Description (optional)")
        confirm = st.checkbox("The selected columns are the stated physical quantities and units, and the data type "
                              "and provenance above are accurate.")
        c1, c2 = st.columns([1, 1])
        submitted = c1.form_submit_button("Register dataset", type="primary", icon=":material/upload:")
        c2.download_button("Download cleaned CSV", clean.to_csv(index=False).encode(),
                           file_name=f"cleaned_{up.name.rsplit('.', 1)[0]}.csv", mime="text/csv")
    if submitted:
        if not confirm:
            st.error("Tick the confirmation box to register the dataset.")
        elif not provenance.strip():
            st.error("Provenance is required so results stay traceable.")
        else:
            try:
                ds = workflow.register_dataframe(
                    repo, clean, name=name, dataset_type=dtype, provenance=provenance, description=description,
                    features=[rename.get(c, c) for c in features], targets=[rename.get(c, c) for c in targets])
                workflow.set_active_dataset_id(repo, ds["id"])
                st.toast(f"Registered “{ds['name']}” ({ds['sample_count']} rows) and made it active.",
                         icon=":material/check_circle:")
                st.rerun()
            except DuplicateRecordError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Registration failed: {exc}")


# ================================================================================= model
def _search_space_editor(model_keys: list[str], n_features: int) -> dict:
    custom = {}
    for key in model_keys:
        spec = MODEL_SPECS[key]
        if not spec.tunable or key == "gaussian_process":
            continue
        default = spec.search_space(n_features)
        text = st.text_area(f"{spec.label} search space", value=repr(default), key=f"space_{key}", height=90)
        try:
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, dict) or not all(isinstance(v, list) and v for v in parsed.values()):
                raise ValueError
            if parsed != default:
                custom[key] = parsed
        except (ValueError, SyntaxError):
            st.error(f"Invalid search space for {spec.label}; the default will be used.")
    return custom


def _training_panel(df: pd.DataFrame, ds: dict) -> None:
    with theme.card("dataset-3"):
        with st.form("train_form", border=False):
            model_keys = st.multiselect("Models", list(MODEL_SPECS), default=DEFAULT_MODELS, format_func=model_label,
                                        help="The Dummy baseline always predicts the mean; useful models must beat it.")
            c1, c2 = st.columns(2)
            features = c1.multiselect("Input features", ds["input_features"], default=ds["input_features"])
            targets = c2.multiselect("Targets", ds["target_names"], default=ds["target_names"])
            c1, c2, c3 = st.columns(3)
            test_size = c1.slider("Test split", 0.1, 0.4, config.DEFAULT_TEST_SIZE, 0.05,
                                  help="Fraction held out and used once, for the final metrics only.")
            folds = c2.slider("Cross-validation folds", 3, 10, config.DEFAULT_CV_FOLDS,
                              help="K-fold CV on the training portion; used to choose the active model.")
            seed = c3.number_input("Random seed", 0, 10_000, config.RANDOM_SEED)
            with st.expander("Advanced · hyperparameter tuning"):
                tune = st.toggle("Tune hyperparameters (RandomizedSearchCV, training data only)", value=False)
                n_iter = st.slider("Configurations tried per model", 2, 30, config.DEFAULT_TUNING_ITERATIONS)
                custom = _search_space_editor(model_keys, max(1, len(features)))
            notes = st.text_input("Note for the experiment log (optional)", placeholder="e.g. baseline comparison run")
            submitted = st.form_submit_button("Train models", type="primary", icon=":material/model_training:")
        if submitted:
            if not (model_keys and features and targets):
                st.error("Choose at least one model, one input feature and one target.")
                return
            with st.status("Training models…", expanded=True) as status:
                bar = st.progress(0.0, "Preparing data")
                try:
                    cfg = TrainingConfig(features=features, targets=targets, model_keys=model_keys,
                                         test_size=test_size, seed=int(seed), cv_folds=folds, tune=tune,
                                         tuning_iterations=n_iter, custom_search_spaces=custom if tune else {})
                    exps, _ = workflow.train_and_log(get_repo(), ds["id"], cfg, notes=notes,
                                                     progress=lambda f, m: bar.progress(min(f, 1.0), m))
                    status.update(label=f"Trained and logged {len(exps)} model(s)", state="complete", expanded=False)
                    st.toast("Training complete. Active models updated by cross-validated R².",
                             icon=":material/check_circle:")
                except Exception as exc:
                    status.update(label="Training failed", state="error", expanded=True)
                    st.error(f"{exc}. Check that the selected targets have enough non-missing rows "
                             f"(at least {config.MIN_SAMPLES_TO_TRAIN}).")


# =============================================================================== results
_METRIC_COLUMNS = {
    "target": st.column_config.TextColumn("Target"),
    "Model": st.column_config.TextColumn("Model"),
    "variant": st.column_config.TextColumn("Variant"),
    "test_R2": st.column_config.NumberColumn("R²", format="%.3f", help=METRIC_EXPLANATIONS["R2"]),
    "test_MAE": st.column_config.NumberColumn("MAE", format="%.3f", help=METRIC_EXPLANATIONS["MAE"]),
    "test_RMSE": st.column_config.NumberColumn("RMSE", format="%.3f", help=METRIC_EXPLANATIONS["RMSE"]),
    "cv_R2_mean": st.column_config.NumberColumn("CV R²", format="%.3f", help=METRIC_EXPLANATIONS["CV R2"]),
}


def metrics_frame(exps: list[dict]) -> pd.DataFrame:
    table = comparison_table(exps)
    if table.empty:
        return table
    table["Model"] = table["model"].map(model_label)
    return table


def _active_model_selector(ds: dict, exps: list[dict]) -> None:
    repo = get_repo()
    active = workflow.get_active_experiments(repo, ds["id"])
    st.caption("Used by Predict & Optimize. Chosen automatically by cross-validated R² (the test set is never "
               "used for selection); you can override it here.")
    cols = st.columns(max(1, len(ds["target_names"])))
    changed = None
    for col, target in zip(cols, ds["target_names"]):
        options = [e["id"] for e in exps if e["target_variable"] == target and e["artifact_path"]]
        if not options:
            continue
        current = active.get(target, {}).get("id")
        key = f"active_{target}_{ds['id']}"
        if current in options:
            st.session_state[key] = current  # keep in sync with auto-selection after new training runs

        def _apply(target=target, key=key):
            workflow.set_active_model(repo, ds["id"], target, st.session_state[key])
            st.session_state["active_changed"] = (target, st.session_state[key])

        col.selectbox(target, options, key=key, on_change=_apply,
                      format_func=lambda i: next(f"#{e['id']} {model_label(e['model_type'])} ({e['variant']})"
                                                 for e in exps if e["id"] == i))
    changed = st.session_state.pop("active_changed", None)
    if changed:
        kind = next(e for e in exps if e["id"] == changed[1])["model_type"]
        st.toast(f"Active model for {changed[0]} set to experiment #{changed[1]}.", icon=":material/check_circle:")
        if kind == "dummy":
            st.warning("The Dummy baseline is not a useful predictor; select it only for comparison.")


def _results_panel(ds: dict) -> None:
    repo = get_repo()
    exps = repo.list_experiments(dataset_id=ds["id"])
    if not exps:
        theme.empty_state("No models trained yet",
                          "Choose models above and press Train models. The default set takes under a minute; "
                          "metrics, plots and the active model for each target will appear here.",
                          ":material/model_training:")
        return
    groups = list(dict.fromkeys(e["run_group"] for e in exps))
    with theme.card("dataset-4"):
        c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
        with c1:
            st.markdown("**Model comparison** · held-out test set")
        group = c2.selectbox("Training run", groups, label_visibility="collapsed",
                             format_func=lambda g: f"Run {g[:15]}" + (" (latest)" if g == groups[0] else ""))
        run = [e for e in exps if e["run_group"] == group]
        table = metrics_frame(run)
        st.dataframe(table[["target", "Model", "variant", "test_R2", "test_MAE", "test_RMSE", "cv_R2_mean"]],
                     column_config=_METRIC_COLUMNS, hide_index=True, width="stretch")
        st.caption("MAE and RMSE are in the target's unit: dB for S11, dBi for gain. "
                   + ("Synthetic data: these metrics measure recovery of an assumed formula."
                      if ds["dataset_type"] == "SYNTHETIC" else f"Measured on {ds['dataset_type'].lower()} data."))
        tuned = []
        for e in run:
            if e["variant"] == "tuned":
                d = next((x for x in run if x["model_type"] == e["model_type"] and x["variant"] == "default"
                          and x["target_variable"] == e["target_variable"]), None)
                if d:
                    tuned.append(improvement_statement(d, e, e["target_variable"], model_label(e["model_type"])))
        if tuned:
            with st.expander("Did tuning help? (measured on the test set)"):
                for s in tuned:
                    st.caption(f"• {s}")
        warns = sorted({w for e in run for w in (e["extra_metrics"] or {}).get("warnings", [])})
        for w in warns:
            st.caption(f":orange[●] {w}")

    active = workflow.get_active_experiments(repo, ds["id"])
    if active:
        st.markdown("**Active models · actual vs predicted**")
        cols = st.columns(len(active), gap="medium")
        for col, (target, exp) in zip(cols, active.items()):
            with col, theme.card(f"dataset-avp-{target}"):
                try:
                    p = load_model(exp).payload
                except Exception as exc:
                    st.error(f"Could not load model #{exp['id']}: {exc}")
                    continue
                st.caption(f"{model_label(exp['model_type'])} ({exp['variant']}) · R² {exp['r2']:.3f} · "
                           f"RMSE {exp['rmse']:.3f} {config.unit_of(target)}")
                theme.chart(plots.actual_vs_predicted(p["y_test"], p["y_pred"], target, y_std=p.get("y_std")),
                            key=f"avp_{target}")
        st.caption("Points on the dashed line are perfect predictions. Error bars (where shown) are an approximate "
                   "±1.96σ model uncertainty, not a calibrated interval.")
    with st.expander("Active model per target"):
        _active_model_selector(ds, exps)


# ================================================================================== page
def render() -> None:
    df, ds, warnings = active_frame()
    theme.page_header("MetaOpt-AI",
                      "Machine-learning surrogates for antenna performance prediction and geometry optimization. "
                      "Start here: check the active dataset, then train and compare models.")
    bar = st.empty()  # filled at the end of the run, so it reflects models trained just now
    theme.synthetic_notice(ds)
    for w in warnings:
        st.warning(w, icon=":material/warning:")

    theme.section("Dataset")
    if df is not None:
        _dataset_overview(df, ds)
    else:
        theme.empty_state("No dataset loaded", "Import a CSV or Excel file below to begin.", ":material/table:")
    with st.expander("Import a dataset (CSV or Excel)", icon=":material/upload_file:"):
        _import_dataset()

    if df is not None:
        theme.section("Model", "Train and validate regression surrogates. Scaling, cross-validation and tuning use "
                               "the training split only, so the test set never influences model choice.")
        _training_panel(df, ds)
        theme.section("Results")
        _results_panel(ds)
    n_models = len(workflow.get_active_experiments(get_repo(), ds["id"])) if ds else None
    theme.context_bar(ds, n_models, where=bar)

