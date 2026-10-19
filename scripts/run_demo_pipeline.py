"""Headless end-to-end demonstration of the MetaOpt-AI workflow.

Runs: database init -> synthetic dataset -> training of all models (default +
tuned) -> prediction -> three optimization objectives x three algorithms ->
PDF report + CSV export, and writes a Markdown summary of the MEASURED results
to ``sample_outputs/demo_results_summary.md``.

Usage (from the project root):
    python scripts/run_demo_pipeline.py            # full run incl. tuning (~3-4 min)
    python scripts/run_demo_pipeline.py --quick    # no tuning, fewer models (<1 min)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from database.repository import Repository  # noqa: E402
from datasets import synthetic  # noqa: E402
from ml.evaluation import comparison_table, improvement_statement  # noqa: E402
from ml.explainability import permutation_importance_frame  # noqa: E402
from ml.model_registry import model_label  # noqa: E402
from ml.training import TrainingConfig  # noqa: E402
from optimization.constraints import Constraint  # noqa: E402
from optimization.objectives import ObjectiveConfig  # noqa: E402
from reporting.report_generator import export_csv_bundle, generate_pdf_report  # noqa: E402
from services import workflow  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "-" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    config.ensure_directories()
    repo = Repository()
    ds = workflow.ensure_demo_dataset(repo)
    workflow.set_active_dataset_id(repo, ds["id"])
    print(f"Dataset #{ds['id']} [{ds['dataset_type']}] {ds['sample_count']} samples, checksum {ds['checksum'][:16]}...")

    models_list = ["dummy", "random_forest", "extra_trees", "gaussian_process"] if args.quick else \
        ["dummy", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "gaussian_process"]
    cfg = TrainingConfig(features=ds["input_features"], targets=ds["target_names"], model_keys=models_list,
                         tune=not args.quick, tuning_iterations=8)
    exps, _ = workflow.train_and_log(repo, ds["id"], cfg, notes="scripts/run_demo_pipeline.py",
                                     progress=lambda f, m: print(f"  [{f:4.0%}] {m}"))
    table = comparison_table(exps)
    active = workflow.get_active_experiments(repo, ds["id"])
    models = workflow.get_active_models(repo, ds["id"])

    tuning = []
    for e in exps:
        if e["variant"] == "tuned":
            d = next(x for x in exps if x["model_type"] == e["model_type"] and x["target_variable"] == e["target_variable"]
                     and x["variant"] == "default")
            tuning.append(improvement_statement(d, e, e["target_variable"], model_label(e["model_type"])))

    pred, rec = workflow.predict_and_maybe_save(repo, models, {"L_mm": 10.0, "G_mm": 1.0}, save=True)

    df, _, _ = workflow.load_dataset_frame(repo, ds["id"])
    norm = workflow.target_ranges(df, list(models))
    bounds = workflow.model_training_ranges(models)
    baseline = {"L_mm": float(df["L_mm"].median()), "G_mm": float(df["G_mm"].median())}
    objectives = {
        "max_gain": ObjectiveConfig("max_gain", [Constraint("S11_dB", "<=", -10.0)], normalization=norm),
        "min_s11": ObjectiveConfig("min_s11", [Constraint("Gain_dBi", ">=", 7.0)], normalization=norm),
        "tradeoff": ObjectiveConfig("tradeoff", [Constraint("S11_dB", "<=", -10.0)],
                                    {"S11_dB": 0.5, "Gain_dBi": 0.5}, norm),
    }
    opt_rows = []
    for name, ocfg in objectives.items():
        outs = workflow.run_optimization(repo, models, ocfg, bounds, baseline,
                                         ["differential_evolution", "multistart_nelder_mead", "dual_annealing"],
                                         seed=config.RANDOM_SEED, budget=1500)
        for outcome, record, base in outs:
            c = outcome.candidate
            l, g = c["parameters"]["L_mm"], c["parameters"]["G_mm"]
            opt_rows.append({
                "run": record["id"], "objective": name, "algorithm": outcome.algorithm, "evals": outcome.n_evals,
                "L_mm": l, "G_mm": g, "pred_S11_dB": c["predictions"]["S11_dB"],
                "pred_Gain_dBi": c["predictions"]["Gain_dBi"],
                "std_S11": c["uncertainty"]["S11_dB"], "std_Gain": c["uncertainty"]["Gain_dBi"],
                "feasible": c["feasible"], "formula_S11_dB": float(synthetic.s11_db_clean(l, g)),
                "formula_Gain_dBi": float(synthetic.gain_dbi_clean(l, g)), "runtime_s": outcome.runtime_s,
            })
    opt = pd.DataFrame(opt_rows)
    base_desc = outs[0][2]

    perm_rows = []
    for target, lm in models.items():
        p = lm.payload
        pim = permutation_importance_frame(lm.pipeline, p["X_test"], p["y_test"], lm.features, seed=p["seed"])
        for r in pim.itertuples():
            perm_rows.append({"target": target, "feature": r.feature, "importance_mean": r.importance_mean,
                              "importance_std": r.importance_std})

    pdf = generate_pdf_report(repo, ds["id"])
    _, csv_paths = export_csv_bundle(repo)
    out_dir = ROOT / "sample_outputs"
    out_dir.mkdir(exist_ok=True)
    shutil.copy(pdf, out_dir / "sample_report.pdf")

    import scipy
    import sklearn

    lines = [
        "# MetaOpt-AI - demonstration results (SYNTHETIC DATA)",
        "",
        f"Generated by `scripts/run_demo_pipeline.py{' --quick' if args.quick else ''}` on "
        f"{datetime.now():%Y-%m-%d %H:%M}. Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
        f"SciPy {scipy.__version__}, NumPy {np.__version__}.",
        "",
        "> **All numbers below are measured on the SYNTHETIC demonstration dataset.** They show that the software "
        "workflow functions and how well models recover a documented analytical formula plus noise. They are NOT "
        "electromagnetic simulation or measurement results and say nothing about real antenna accuracy.",
        "",
        f"* Dataset #{ds['id']}: {ds['sample_count']} samples, SHA-256 `{ds['checksum']}`",
        f"* Split: {int(cfg.test_size * 100)} % test, seed {cfg.seed}; {cfg.cv_folds}-fold CV on training data; "
        f"tuning: {'RandomizedSearchCV, 8 iterations' if cfg.tune else 'off'}",
        "",
        "## 1. Model comparison (held-out test set; CV on training data)",
        "",
        md_table(table[["target", "model", "variant", "test_R2", "test_MAE", "test_RMSE", "cv_R2_mean", "cv_R2_std",
                        "coverage95", "train_s"]]),
        "",
        "MAE/RMSE units: dB for S11_dB, dBi for Gain_dBi. coverage95 = fraction of test points within +/-1.96 sigma.",
        "",
        "Active models (selected by CV R2): " + ", ".join(
            f"{t}: #{e['id']} {model_label(e['model_type'])} ({e['variant']})" for t, e in active.items()),
        "",
        "## 2. Effect of hyperparameter tuning (measured)",
        "",
        *([f"* {s}" for s in tuning] or ["* Tuning disabled in this run."]),
        "",
        "## 3. Example prediction",
        "",
        f"Input L_mm=10.0, G_mm=1.0 -> " + ", ".join(
            f"{t} = {v:.3f} (approx. std {pred.uncertainty[t]:.3f})" if pred.uncertainty[t] is not None else f"{t} = {v:.3f}"
            for t, v in pred.predictions.items()) + f". Saved as prediction #{rec['id']}. Noise-free synthetic formula at "
        f"this point: S11 = {float(synthetic.s11_db_clean(10.0, 1.0)):.3f} dB, gain = {float(synthetic.gain_dbi_clean(10.0, 1.0)):.3f} dBi.",
        "",
        "## 4. Permutation importance of the active models (drop in test R2)",
        "",
        md_table(pd.DataFrame(perm_rows)),
        "",
        "## 5. Optimization (surrogate-based, ML-selected candidates)",
        "",
        f"Baseline (median training design) L_mm={baseline['L_mm']:.3f}, G_mm={baseline['G_mm']:.3f} -> predicted "
        f"S11 {base_desc['predictions']['S11_dB']:.2f} dB, gain {base_desc['predictions']['Gain_dBi']:.2f} dBi "
        f"(feasible for S11<=-10: {base_desc['feasible']}). Budget 1500 evaluations per algorithm, seed {config.RANDOM_SEED}.",
        "",
        md_table(opt),
        "",
        "`formula_*` columns evaluate the noise-free synthetic generator at the candidate geometry (an oracle that only "
        "exists for synthetic data). Agreement shows the surrogate recovered the ASSUMED formula; it is not physical validation.",
        "",
        f"Total pipeline runtime: {time.perf_counter() - t_start:.1f} s.",
        "",
        f"Report: `{pdf.relative_to(ROOT)}` (copied to `sample_outputs/sample_report.pdf`). CSV exports: "
        + ", ".join(f"`{p.relative_to(ROOT)}`" for p in csv_paths),
    ]
    (out_dir / "demo_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
