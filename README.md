# MetaOpt-AI | Antenna Design Explorer

**Explainable Machine Learning for Metasurface Antenna Optimization**
Student: Aniket Jain · Registration No. 23BEC0470

MetaOpt-AI is a locally running research application that integrates data
management with provenance, machine-learning surrogate models, surrogate-based
geometry optimization, explainability and uncertainty estimates, experiment
tracking in SQLite, and PDF/CSV reporting, all in one Streamlit dashboard.

> ⚠️ **Scientific honesty.** The bundled dataset is **SYNTHETIC**: it comes from a
> documented analytical formula plus noise (see `data/README.md`). The ML models
> do **not** solve Maxwell's equations. Optimized geometries are **ML-selected
> candidates**, not physically validated antennas. Results on synthetic data
> show that the software workflow works. They say nothing about real-world
> antenna accuracy.

---

## 1. Quick start (macOS, VS Code terminal)

```bash
git clone https://github.com/aniketjain2023a-ops/antenna-project.git "antenna project"
cd "antenna project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_public_datasets.py   # downloads + models the public datasets (about 2 min)
streamlit run app.py              # opens http://localhost:8501
```

`prepare_public_datasets.py` downloads **METU-SIMPL** (HFSS-simulated microstrip patches) from
GitHub, verifies checksums, preprocesses it into `data/processed/metu_simpl/`, trains Dummy, Random
Forest and Gradient Boosting models, and writes metrics and figures to `results/metu_simpl/`. It
also tries the **Zenodo** record and reports clearly if it cannot be downloaded. A dataset appears in
the app only when its whole pipeline succeeded. See `DATASET_COMPARISON.md`.

On first start the app creates `data/metaopt_ai.db`, generates
`data/synthetic_demo.csv` if missing, and registers it as a SYNTHETIC dataset.
On **Home**, press **Train models** if the active dataset has no models yet (the public-data script already trains METU-SIMPL).
Prediction, optimization and the results pages then have models to use.

Optional headless run of the whole workflow (training with tuning, three
objectives × three optimizers, PDF + CSV export):

```bash
python scripts/run_demo_pipeline.py          # about 2-4 min
python scripts/run_demo_pipeline.py --quick  # under 1 min
```

Tests:

```bash
python -m pytest            # 51 tests, about 80 s
```

## 2. Sections

The interface has three pages, shown in the top navigation bar.

| Page | What it does |
|---|---|
| **Home** | Active dataset with provenance badge; data quality, preview, statistics and distributions; CSV/Excel import with column mapping and mandatory provenance; model training (Dummy, Random Forest, Gradient Boosting by default) with live progress; comparison table (R², MAE, RMSE, CV R²) and actual-vs-predicted plots; active model per target |
| **Design & Optimize** | Dataset selector (only datasets whose pipeline completed). The parameter form matches the selected dataset's inputs, units and ranges, and returns predicted targets with uncertainty and model metrics. Separately, Differential Evolution optimization (objective, optional constraint, parameters to vary, bounds, baseline) shows baseline vs ML-selected candidate side by side, with diagnostics including the nearest actually-simulated design |
| **Results** | Tabs: across-dataset comparison (every metric labelled with its dataset); model performance; explainability (importance, sensitivity, partial dependence, uncertainty); optimization history; data & export (provenance, CSV, ZIP, PDF, searchable records) |

### Design system

The interface follows the Apple Design skill's guidance, translated to Streamlit:
system font with size-specific tracking and leading; one restrained blue accent;
amber, green and red reserved for warnings, success and errors; white cards on a warm
off-white canvas; a translucent navigation bar; instant press feedback; and
reduced-motion, reduced-transparency and high-contrast fallbacks. Tokens live in
`.streamlit/config.toml` (native theme) and `ui/theme.py` (a small CSS layer and
reusable components such as cards, badges, the context bar and empty states).
Charts share one Plotly template defined in `visualization/plots.py`.

## 3. Architecture

```
antenna-project/  (MetaOpt-AI)
├── app.py                     Streamlit entry point (top navigation, 3 sections)
├── config.py                  paths, seeds, provenance vocabulary, defaults
├── database/                  SQLAlchemy models, engine/session, migrations, repository (DAL)
├── datasets/                  synthetic generator (documented formula) + CSV/Excel reader
├── ml/                        preprocessing, training (CV/tuning), evaluation, model registry,
│                              prediction, uncertainty, explainability
├── optimization/              objectives, constraints, DE, Nelder-Mead & dual annealing, Pareto/ranking
├── services/workflow.py       orchestration used by UI, scripts and tests (single source of logic)
├── visualization/             Plotly (dashboard) and Matplotlib (PDF) figures
├── reporting/                 ReportLab PDF + CSV export
├── ui/                        theme.py (design system), one module per section (UI only)
├── utils/                     validation, logging, SciPy compatibility
├── scripts/run_demo_pipeline.py
├── tests/                     pytest suite (database, data, ML, optimization, reporting)
├── data/  models/  reports/  logs/  sample_outputs/
└── *.md                       university documentation
```

Layering: `ui → services → (ml, optimization, reporting) → database`. No SQL in the UI.

## 4. Where things are stored

| Item | Path |
|---|---|
| SQLite database | `data/metaopt_ai.db` (override with `METAOPT_DB_PATH`) |
| Synthetic dataset | `data/synthetic_demo.csv` |
| Uploaded/cleaned datasets | `data/registered/` |
| Trained models (joblib) | `models/` |
| PDF reports, CSV exports, figures | `reports/` |
| Logs | `logs/metaopt.log` |
| Committed example outputs | `sample_outputs/sample_report.pdf`, `sample_outputs/demo_results_summary.md` |

To start completely fresh, delete `data/metaopt_ai.db*`, `models/*.joblib` and `reports/*`.

## 5. Documentation

`PROJECT_REPORT.md` · `RESULTS_INTERPRETATION.md` · `REPRODUCIBILITY.md` ·
`EXPERIMENT_LOG.md` · `VIVA_QUESTIONS.md` · `PRESENTATION_OUTLINE.md` ·
`DEMO_SCRIPT.md` · `REFERENCES.md` · `data/README.md`

## 6. Known limitations

See `PROJECT_REPORT.md` §15. In short: synthetic data only out of the box, two
geometry parameters, surrogate predictions only (no EM solver), uncertainty is
approximate, and uncertainty-guided sampling is suggested but its benefit has not
been demonstrated.
