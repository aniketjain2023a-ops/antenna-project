"""Download, verify, preprocess, register and model the public antenna datasets.

Usage (from the project root):
    python scripts/prepare_public_datasets.py            # all sources
    python scripts/prepare_public_datasets.py metu_simpl # one source
    python scripts/prepare_public_datasets.py --redownload

Every stage is recorded in data/processed/<source>/pipeline_status.json. The application offers a
source only when all stages succeeded. Failures are recorded and reported, never hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from database.repository import DuplicateRecordError, Repository  # noqa: E402
from datasets.public import SOURCES, DatasetUnavailable, processed_dir, raw_dir, write_stage  # noqa: E402
from datasets.public import metu_simpl, zenodo  # noqa: E402
from ml.evaluation import comparison_table  # noqa: E402
from ml.explainability import impurity_importance, permutation_importance_frame  # noqa: E402
from ml.model_registry import model_label  # noqa: E402
from ml.training import TrainingConfig  # noqa: E402
from services import workflow  # noqa: E402
from visualization import static_plots  # noqa: E402

MODELS = ["dummy", "random_forest", "gradient_boosting"]


def _register(repo: Repository, key: str, table: pd.DataFrame, features: list[str], targets: list[str],
              provenance: str, description: str) -> dict:
    data = table[features + targets]
    try:
        return workflow.register_dataframe(repo, data, name=SOURCES[key]["dataset_name"], dataset_type="SIMULATION",
                                           provenance=provenance, description=description, features=features,
                                           targets=targets)
    except DuplicateRecordError:
        existing = repo.get_dataset_by_name(SOURCES[key]["dataset_name"])
        if existing is None:
            raise
        return existing


def _export_results(repo: Repository, key: str, ds: dict, exps: list[dict]) -> dict:
    out = config.RESULTS_DIR / key
    (out / "figures").mkdir(parents=True, exist_ok=True)
    table = comparison_table(exps)
    table.insert(0, "dataset", SOURCES[key]["label"])
    table.to_csv(out / "metrics.csv", index=False)
    active = workflow.get_active_experiments(repo, ds["id"])
    models = workflow.load_models({t: e for t, e in active.items()})
    preds, importances = [], []
    for exp in exps:
        lm = workflow.load_models({exp["target_variable"]: exp})[exp["target_variable"]]
        p = lm.payload
        frame = pd.DataFrame(p["X_test"], columns=lm.features)
        frame.insert(0, "experiment_id", exp["id"])
        frame.insert(1, "model", exp["model_type"])
        frame["target"] = exp["target_variable"]
        frame["y_true"] = p["y_test"]
        frame["y_pred"] = p["y_pred"]
        preds.append(frame)
        imp = impurity_importance(lm.pipeline, lm.features)
        if imp is not None:
            importances.append(imp.assign(experiment_id=exp["id"], model=exp["model_type"],
                                          target=exp["target_variable"], method="impurity"))
    for target, lm in models.items():
        p = lm.payload
        perm = permutation_importance_frame(lm.pipeline, p["X_test"], p["y_test"], lm.features, seed=p["seed"])
        importances.append(perm.rename(columns={"importance_mean": "importance"}).assign(
            experiment_id=lm.experiment["id"], model=lm.experiment["model_type"], target=target, method="permutation"))
        name = f"{lm.experiment['model_type']} (exp {lm.experiment['id']})"
        static_plots.actual_vs_predicted_png(p["y_test"], p["y_pred"], target,
                                             f"{SOURCES[key]['label']} · {config.short_label(target)} · {name}",
                                             out / "figures" / f"actual_vs_predicted_{target}.png")
        static_plots.residual_png(p["y_test"], p["y_pred"], target,
                                  f"{SOURCES[key]['label']} · residuals · {config.short_label(target)}",
                                  out / "figures" / f"residuals_{target}.png")
        static_plots.importance_png(perm, "importance_mean",
                                    f"{SOURCES[key]['label']} · permutation importance · {config.short_label(target)}",
                                    out / "figures" / f"permutation_importance_{target}.png")
    static_plots.metric_bars_png(table, "test_R2", out / "figures" / "model_comparison_r2.png")
    pd.concat(preds).to_csv(out / "test_predictions.csv", index=False, float_format="%.6g")
    pd.concat(importances).to_csv(out / "feature_importance.csv", index=False, float_format="%.6g")
    selected = {t: {"experiment_id": e["id"], "model": e["model_type"], "cv_r2": e["cv_r2_mean"], "test_r2": e["r2"],
                    "test_rmse": e["rmse"], "test_mae": e["mae"], "artifact": e["artifact_path"]}
                for t, e in active.items()}
    (out / "selected_models.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


def run_metu(repo: Repository, redownload: bool) -> None:
    key = "metu_simpl"
    print("\n=== METU-SIMPL ===")
    try:
        manifest_path = raw_dir(key) / "MANIFEST.json"
        have_all = all((raw_dir(key) / f).exists() for f in metu_simpl.FILES)
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if redownload or not have_all or recorded is None:
            manifest = metu_simpl.download()
        else:
            manifest = metu_simpl.verify()
        changed = []
        if recorded:
            changed = [f for f, m in manifest["files"].items()
                       if recorded["files"].get(f, {}).get("sha256") != m["sha256"]]
        note = (f" WARNING: upstream files differ from the recorded download ({', '.join(changed)}); results may "
                "not reproduce the documented metrics." if changed else "")
        write_stage(key, "download", True, f"Commit {manifest['commit'][:8]}.{note}", files=list(manifest["files"]),
                    changed_since_recorded=changed)
        print(f"download OK: commit {manifest['commit'][:8]}, files {list(manifest['files'])}{note}")
    except Exception as exc:
        write_stage(key, "download", False, str(exc))
        print(f"download FAILED: {exc}")
        return
    try:
        summary = metu_simpl.preprocess()
        write_stage(key, "preprocess", True, f"{summary['rows_processed']} rows", summary=summary)
        print(f"preprocess OK: {summary['rows_raw']} raw rows -> {summary['rows_processed']} processed; "
              f"non-missing targets {summary['target_non_missing']}")
    except Exception as exc:
        write_stage(key, "preprocess", False, str(exc))
        print(f"preprocess FAILED: {exc}")
        return
    try:
        table = pd.read_csv(processed_dir(key) / "metu_simpl_processed.csv")
        ds = _register(repo, key, table, metu_simpl.FEATURES, metu_simpl.TARGETS, metu_simpl.PROVENANCE,
                       "5,000 HFSS-simulated probe-fed microstrip patch designs; targets derived from the 23–33 GHz "
                       "S11 sweep (see datasets/public/metu_simpl.py for definitions and censoring).")
        cfg = TrainingConfig(features=metu_simpl.FEATURES, targets=metu_simpl.TARGETS, model_keys=MODELS,
                             test_size=0.2, seed=config.RANDOM_SEED, cv_folds=5)
        exps, _ = workflow.train_and_log(repo, ds["id"], cfg, notes="scripts/prepare_public_datasets.py",
                                         progress=lambda f, m: print(f"  [{f:4.0%}] {m}"))
        selected = _export_results(repo, key, ds, exps)
        # verify that every saved model loads and predicts
        loaded = workflow.get_active_models(repo, ds["id"])
        probe = {f: float(table[f].median()) for f in metu_simpl.FEATURES}
        from ml.prediction import predict_point
        check = predict_point(loaded, probe)
        write_stage(key, "train", True, f"{len(exps)} experiments", dataset_id=ds["id"], selected=selected,
                    load_check=check.predictions)
        workflow.set_active_dataset_id(repo, ds["id"])
        print("train OK; selected models:")
        for t, s in selected.items():
            print(f"  {t}: {model_label(s['model'])} exp {s['experiment_id']} test R2={s['test_r2']:.3f} "
                  f"RMSE={s['test_rmse']:.3f} MAE={s['test_mae']:.3f}")
    except Exception as exc:
        write_stage(key, "train", False, str(exc))
        print(f"train FAILED: {exc}")
        raise


def run_zenodo(redownload: bool) -> None:
    key = "zenodo"
    print("\n=== Zenodo 15866865 ===")
    try:
        manifest = zenodo.download()
        write_stage(key, "download", True, f"Downloaded {len(manifest['files'])} file(s)", files=list(manifest["files"]))
        print(f"download OK: {list(manifest['files'])}; license {manifest.get('license')}")
    except DatasetUnavailable as exc:
        write_stage(key, "download", False, str(exc))
        raw_dir(key).mkdir(parents=True, exist_ok=True)
        (raw_dir(key) / "DOWNLOAD_STATUS.json").write_text(json.dumps(
            {"source": zenodo.RECORD_URL, "ok": False, "error": str(exc)}, indent=2), encoding="utf-8")
        print(f"download FAILED: {exc}")
        return
    msg = ("Files downloaded but their schema, units and license have not been inspected; no preprocessing pipeline "
           "exists yet. The dataset is not offered in the app until it has been inspected.")
    write_stage(key, "preprocess", False, msg)
    print(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--redownload", action="store_true")
    args = ap.parse_args()
    config.ensure_directories()
    repo = Repository()
    workflow.ensure_demo_dataset(repo)
    if "metu_simpl" in args.sources:
        run_metu(repo, args.redownload)
    if "zenodo" in args.sources:
        run_zenodo(args.redownload)


if __name__ == "__main__":
    main()
