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

streamlit run app.py              # opens http://localhost:8501
```

On first start the app creates `data/metaopt_ai.db`, generates
`data/synthetic_demo.csv` if missing, and registers it as a SYNTHETIC dataset.
On the **Overview** page, click **Quick start: train demo models** (about 30 s).
Every other page then works.

Optional headless run of the whole workflow (training with tuning, three
objectives × three optimizers, PDF + CSV export):

```bash
python scripts/run_demo_pipeline.py          # about 2-4 min
python scripts/run_demo_pipeline.py --quick  # under 1 min
```

Tests:

```bash
python -m pytest            # 50 tests, about 75 s
```

## 2. Pages

| # | Page | What it does |
|---|---|---|
| A | Overview Dashboard | purpose, counts, active-model metrics, workflow diagram, recent activity, quick-start training |
| B | Dataset Management | inspect active data (stats, quality, IQR outliers, distributions, correlation); upload CSV/Excel → type detection → column mapping → cleaning summary → provenance → register; export |
| C | ML Model Lab | choose features/targets/models/split/seed/CV folds; optional RandomizedSearchCV with editable search spaces; comparison table and charts; actual-vs-predicted & residual plots; active model selection |
| D | Performance Prediction | geometry inputs → S11 & gain with approximate uncertainty, extrapolation warnings, model metrics; save to DB |
| E | Optimization Lab | objective (max gain s.t. S11 ≤ limit / min S11 s.t. gain ≥ limit / normalized trade-off), bounds, baseline, kappa risk-aversion, 3 algorithms, ranking with explanations, convergence, Pareto trade-off, design-space map, synthetic oracle check |
| F | Explainable AI | impurity & permutation importance, OAT sensitivity with uncertainty band, local sensitivities, partial dependence, uncertainty map + suggested sampling regions, empirical interval coverage |
| G | Experiment History | searchable/filterable tables of datasets, experiments, predictions, optimization runs; CSV export |
| H | Report Generator | PDF report (provenance, methods, metrics, figures, optimization, limitations, reproducibility) + CSV ZIP |
| I | Project Info & Limitations | problem, methodology, assumptions, limitations, contribution |

## 3. Architecture

```
antenna-project/  (MetaOpt-AI)
├── app.py                     Streamlit entry point (st.navigation, 9 pages)
├── config.py                  paths, seeds, provenance vocabulary, defaults
├── database/                  SQLAlchemy models, engine/session, migrations, repository (DAL)
├── datasets/                  synthetic generator (documented formula) + CSV/Excel reader
├── ml/                        preprocessing, training (CV/tuning), evaluation, model registry,
│                              prediction, uncertainty, explainability
├── optimization/              objectives, constraints, DE, Nelder-Mead & dual annealing, Pareto/ranking
├── services/workflow.py       orchestration used by UI, scripts and tests (single source of logic)
├── visualization/             Plotly (dashboard) and Matplotlib (PDF) figures
├── reporting/                 ReportLab PDF + CSV export
├── ui/                        one module per dashboard page (UI only)
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
