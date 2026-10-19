"""PDF (ReportLab) and CSV report generation from the SQLite records."""

from __future__ import annotations

import io
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config
from database.repository import Repository, records_to_frame
from ml.evaluation import comparison_table
from ml.explainability import design_space_grid, high_uncertainty_regions, permutation_importance_frame
from ml.model_registry import model_label
from ml.uncertainty import UNCERTAINTY_DISCLAIMER
from services import workflow
from visualization import static_plots

LIMITATIONS = [
    "The surrogate models are statistical regressions; they do not solve Maxwell's equations and are "
    "not a substitute for full-wave EM simulation or measurement.",
    "When the dataset is SYNTHETIC, all metrics measure how well the models recover a documented "
    "analytical formula plus noise. They say nothing about real antenna accuracy.",
    "Optimized geometries are ML-selected candidates. They must be verified by EM simulation and "
    "prototype measurement before any engineering claim is made.",
    "A deeper S11 alone does not prove better gain, bandwidth, efficiency or radiation pattern.",
    "Uncertainty estimates (GP std, tree spread) are approximate and only empirically checked on the "
    "held-out test split; they are not calibrated confidence intervals.",
    "Predictions outside the training range are extrapolations and are flagged as unreliable.",
    "Only two geometry parameters and two targets are modelled in the demonstration; real designs "
    "depend on substrate, feed, unit-cell shape, frequency and fabrication tolerances.",
]


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8, leading=10))
    ss.add(ParagraphStyle("Warn", parent=ss["BodyText"], textColor=colors.HexColor("#9a3412"),
                          backColor=colors.HexColor("#fff7ed"), borderPadding=6, fontSize=9, leading=12))
    ss.add(ParagraphStyle("CenterTitle", parent=ss["Title"], alignment=TA_CENTER))
    return ss


def _table(df: pd.DataFrame, col_widths=None, font_size: int = 7) -> Table:
    data = [list(map(str, df.columns))] + [[_fmt(v) for v in row] for row in df.itertuples(index=False)]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8c8c8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _latest_per_config(exps: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in sorted(exps, key=lambda e: e["id"], reverse=True):
        key = (e["target_variable"], e["model_type"], e["variant"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def generate_pdf_report(repo: Repository, dataset_id: int, output_dir: Path | None = None,
                        max_optimizations: int = 6) -> Path:
    output_dir = Path(output_dir or config.REPORTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = output_dir / f"figures_{stamp}"
    pdf_path = output_dir / f"MetaOpt-AI_report_dataset{dataset_id}_{stamp}.pdf"
    ss = _styles()
    story: list = []

    df, ds, data_warnings = workflow.load_dataset_frame(repo, dataset_id)
    exps = _latest_per_config(repo.list_experiments(dataset_id=dataset_id))
    active = workflow.get_active_experiments(repo, dataset_id)
    exp_ids = {e["id"] for e in repo.list_experiments(dataset_id=dataset_id)}
    runs = [r for r in repo.list_optimization_runs() if r["model_experiment_id"] in exp_ids][:max_optimizations]

    # ---------------------------------------------------------------- title
    story += [
        Paragraph(config.APP_TITLE, ss["CenterTitle"]),
        Paragraph(config.PROJECT_TITLE, ss["Heading2"]),
        Paragraph(f"Student: {config.STUDENT_NAME} &nbsp;&nbsp; Registration No.: {config.REGISTRATION_NUMBER}", ss["BodyText"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;&nbsp; Software version {config.APP_VERSION}", ss["BodyText"]),
        Spacer(1, 0.3 * cm),
    ]
    if ds["dataset_type"] == "SYNTHETIC":
        story.append(Paragraph("<b>WARNING - SYNTHETIC DATA.</b> " + config.SYNTHETIC_WARNING, ss["Warn"]))
    else:
        story.append(Paragraph(f"<b>Data type: {ds['dataset_type']}.</b> All optimization outputs remain ML "
                               "predictions until verified by EM simulation and measurement.", ss["Warn"]))
    for w in data_warnings:
        story.append(Paragraph(f"<b>Data warning:</b> {w}", ss["Warn"]))

    # ------------------------------------------------------------ provenance
    story += [Spacer(1, 0.3 * cm), Paragraph("1. Dataset and provenance", ss["Heading2"])]
    prov = pd.DataFrame([
        ["Dataset ID", ds["id"]], ["Name", ds["name"]], ["Type", f"{ds['dataset_type']} - {config.DATASET_TYPES[ds['dataset_type']]}"],
        ["Samples", ds["sample_count"]], ["Inputs", ", ".join(ds["input_features"])], ["Targets", ", ".join(ds["target_names"])],
        ["SHA-256 checksum", ds["checksum"]], ["Registered", ds["created_at"]],
    ], columns=["Field", "Value"])
    story.append(_table(prov, col_widths=[4 * cm, 13 * cm], font_size=8))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f"<b>Provenance:</b> {ds['provenance']}", ss["Small"]))
    story.append(Paragraph(f"<b>Description:</b> {ds['description']}", ss["Small"]))
    stats = df[ds["input_features"] + ds["target_names"]].describe().T.reset_index().rename(columns={"index": "variable"})
    story += [Spacer(1, 0.2 * cm), _table(stats[["variable", "count", "mean", "std", "min", "max"]], font_size=8)]

    # ----------------------------------------------------------- methodology
    story += [Paragraph("2. Methodology", ss["Heading2"])]
    for para in [
        "Data are validated (types, missing values, duplicates, IQR outliers flagged but not removed). "
        "Missing target values are never imputed; rows are excluded per target.",
        "For each target a reproducible train/test split is made first. Scaling sits inside a scikit-learn "
        "Pipeline. K-fold cross-validation and RandomizedSearchCV tuning use only the training portion; the "
        "test set is used once for final metrics. The active model per target is chosen by CV R2, not test R2.",
        "Models: Dummy (mean) baseline, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient "
        "Boosting and Gaussian Process regression (ARD kernel + white noise).",
        "Optimization calls the trained surrogates inside SciPy Differential Evolution, LHS multi-start "
        "Nelder-Mead and Dual Annealing. Constraints use a static penalty on range-normalized violations; the "
        "trade-off objective adds range-normalized desirabilities so dB and dBi are never summed directly.",
        "Explainability: impurity and permutation importance, one-at-a-time sensitivity, partial dependence. "
        "Uncertainty: GP predictive std or tree-ensemble spread, with empirical 95% interval coverage on the test set.",
    ]:
        story.append(Paragraph(para, ss["BodyText"]))

    # --------------------------------------------------------------- models
    story += [PageBreak(), Paragraph("3. Model configurations and evaluation metrics", ss["Heading2"])]
    table = comparison_table(exps)
    if table.empty:
        story.append(Paragraph("No models have been trained for this dataset yet.", ss["BodyText"]))
    else:
        cols = ["id", "target", "model", "variant", "test_R2", "test_MAE", "test_RMSE", "cv_R2_mean", "cv_R2_std", "coverage95", "n_train", "n_test"]
        story.append(_table(table[cols], font_size=7))
        story.append(Paragraph("MAE/RMSE are in the target unit (dB for S11, dBi for gain). coverage95 = fraction "
                               "of test points inside +/-1.96 sigma (models with uncertainty only).", ss["Small"]))
        story.append(Paragraph("Active models (selected by cross-validated R2 on training data): " +
                               ", ".join(f"{t}: #{e['id']} {model_label(e['model_type'])} ({e['variant']})" for t, e in active.items()),
                               ss["Small"]))
        bar = static_plots.metric_bars_png(table, "test_R2", fig_dir / "metrics_r2.png")
        if bar:
            story.append(Image(str(bar), width=16 * cm, height=16 * cm * 3.8 / (5 * max(1, table["target"].nunique()))))

    # -------------------------------------------------------------- figures
    if active:
        story += [PageBreak(), Paragraph("4. Figures for the active models", ss["Heading2"])]
        models = workflow.load_models(active)
        imgs = []
        for target, lm in models.items():
            p = lm.payload
            png = static_plots.actual_vs_predicted_png(
                p["y_test"], p["y_pred"], target, f"{target}: {model_label(lm.experiment['model_type'])} (test set)",
                fig_dir / f"avp_{target}.png")
            imgs.append(Image(str(png), width=8 * cm, height=6.4 * cm))
        story.append(Table([imgs[i:i + 2] for i in range(0, len(imgs), 2)]))
        imgs = []
        for target, lm in models.items():
            p = lm.payload
            pim = permutation_importance_frame(lm.pipeline, p["X_test"], p["y_test"], lm.features, seed=p.get("seed", 42))
            png = static_plots.importance_png(pim, "importance_mean", f"Permutation importance (drop in R2): {target}",
                                              fig_dir / f"perm_{target}.png")
            if png:
                imgs.append(Image(str(png), width=8 * cm, height=4.5 * cm))
        if imgs:
            story.append(Table([imgs[i:i + 2] for i in range(0, len(imgs), 2)]))
        unc_model = next((lm for lm in models.values() if lm.payload.get("y_std") is not None), None)
        if unc_model is not None and len(unc_model.features) >= 2:
            feats = unc_model.features
            fixed = {f: float(np.median(unc_model.payload["X_train"][:, i])) for i, f in enumerate(feats)}
            grid = design_space_grid(unc_model.pipeline, feats, feats[0], feats[1], unc_model.train_ranges, fixed, n=50)
            pts = pd.DataFrame(unc_model.payload["X_train"], columns=feats)
            hot = high_uncertainty_regions(grid, top_k=5)
            png = static_plots.heatmap_png(grid, grid["std"], f"Approx. predictive std of {unc_model.target} over the design space",
                                           f"std [{config.unit_of(unc_model.target)}]", fig_dir / "uncertainty_map.png",
                                           cmap="Oranges", points=pts, markers=hot)
            story += [Image(str(png), width=12 * cm, height=9.2 * cm),
                      Paragraph("Crosses mark the highest-uncertainty regions: candidate locations for additional EM "
                                "simulations or measurements. This is a suggestion; no improvement in accuracy from "
                                "such sampling has been demonstrated in this project. " + UNCERTAINTY_DISCLAIMER, ss["Small"])]

    # --------------------------------------------------------- optimization
    story += [PageBreak(), Paragraph("5. Optimization results (ML-selected candidates)", ss["Heading2"])]
    if not runs:
        story.append(Paragraph("No optimization runs recorded for this dataset.", ss["BodyText"]))
    else:
        rows = []
        for r in runs:
            rows.append({"run": r["id"], "algorithm": r["algorithm"], "objective": r["objective_function"],
                         "constraints": "; ".join(f"{c['target']} {c['operator']} {c['limit']}" for c in r["constraints"]) or "-",
                         "candidate": ", ".join(f"{k}={v:.3f}" for k, v in r["candidate_parameters"].items()),
                         "predicted": ", ".join(f"{k}={v:.2f}" for k, v in r["candidate_predictions"].items()),
                         "feasible": r["feasible"], "evals": r["evaluations_used"]})
        story.append(_table(pd.DataFrame(rows), font_size=7))
        r0 = runs[0]
        story.append(Paragraph(
            f"Most recent run #{r0['id']}: baseline {r0['baseline_design']} -> predicted {_dict_fmt(r0['baseline_predictions'])}; "
            f"candidate {_dict_fmt(r0['candidate_parameters'])} -> predicted {_dict_fmt(r0['candidate_predictions'])} "
            f"(approx. std {_dict_fmt(r0['candidate_uncertainty'])}). Bounds: {r0['parameter_bounds']}.", ss["Small"]))
        bc = pd.DataFrame([{"source": "baseline", **r0["baseline_predictions"]}] +
                          [{"source": f"run {r['id']}\n{r['algorithm'][:12]}", **r["candidate_predictions"]}
                           for r in runs if r["objective_function"] == r0["objective_function"]][:4])
        png = static_plots.baseline_candidate_png(bc, list(r0["candidate_predictions"]), fig_dir / "baseline_vs_candidate.png")
        if png:
            story.append(Image(str(png), width=16 * cm, height=6.4 * cm))
        story.append(Paragraph("<b>All candidates are ML-selected designs based on surrogate predictions. They are NOT "
                               "physically validated antenna designs.</b>", ss["Warn"]))

    # ---------------------------------------------------------- limitations
    story += [Paragraph("6. Limitations", ss["Heading2"])]
    for item in LIMITATIONS:
        story.append(Paragraph("&bull; " + item, ss["BodyText"]))

    # ------------------------------------------------------ reproducibility
    import scipy
    import sklearn

    story += [Paragraph("7. Reproducibility information", ss["Heading2"])]
    repro = pd.DataFrame([
        ["Dataset checksum (SHA-256)", ds["checksum"]], ["Global random seed", config.RANDOM_SEED],
        ["Python", platform.python_version()], ["scikit-learn", sklearn.__version__], ["SciPy", scipy.__version__],
        ["NumPy", np.__version__], ["pandas", pd.__version__], ["Platform", platform.platform()],
    ], columns=["Item", "Value"])
    story.append(_table(repro, col_widths=[5 * cm, 12 * cm], font_size=8))
    for t, e in active.items():
        story.append(Paragraph(f"<b>{t}</b> - experiment #{e['id']} {e['model_type']} ({e['variant']}), seed {e['random_seed']}, "
                               f"validation {e['validation_config'].get('cv')}, hyperparameters: {_short(e['hyperparameters'])}", ss["Small"]))
    for r in runs[:3]:
        story.append(Paragraph(f"<b>Optimization #{r['id']}</b> {r['algorithm']}, seed {r['random_seed']}, budget "
                               f"{r['evaluation_budget']} (used {r['evaluations_used']}), settings "
                               f"{_short(r['objective_config'].get('algorithm_settings', {}))}", ss["Small"]))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="MetaOpt-AI report",
                            author=config.STUDENT_NAME)
    label = "SYNTHETIC DATA - workflow demonstration, not physically validated" if ds["dataset_type"] == "SYNTHETIC" \
        else f"{ds['dataset_type']} data - ML predictions are not physical validation"

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#9a3412"))
        canvas.drawString(1.8 * cm, 1.0 * cm, label)
        canvas.setFillColor(colors.black)
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"MetaOpt-AI | page {_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return pdf_path


def _dict_fmt(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {'-' if v is None else f'{v:.3f}'}" for k, v in d.items()) + "}"


def _short(d: dict, limit: int = 300) -> str:
    s = ", ".join(f"{k}={v}" for k, v in d.items())
    return s if len(s) <= limit else s[:limit] + "..."


def export_tables(repo: Repository) -> dict[str, pd.DataFrame]:
    return {
        "datasets": records_to_frame(repo.list_datasets()),
        "model_experiments": records_to_frame(repo.list_experiments()),
        "predictions": records_to_frame(repo.list_predictions()),
        "optimization_runs": records_to_frame(repo.list_optimization_runs()),
    }


def export_csv_bundle(repo: Repository, output_dir: Path | None = None) -> tuple[bytes, list[Path]]:
    """Write one CSV per table into ``output_dir`` and return a ZIP of them as bytes."""
    output_dir = Path(output_dir or config.REPORTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in export_tables(repo).items():
            path = output_dir / f"{name}_{stamp}.csv"
            frame.to_csv(path, index=False)
            paths.append(path)
            zf.writestr(f"{name}.csv", frame.to_csv(index=False))
        zf.writestr("README.txt", "MetaOpt-AI export. Rows with source/data type SYNTHETIC are demonstration data, "
                                  "not measurements. Predictions and optimization candidates are ML outputs.\n")
    return buffer.getvalue(), paths
