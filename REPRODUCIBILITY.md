# Reproducibility guide

## 1. Environment
```bash
cd "antenna project"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import sklearn, scipy, numpy; print(sklearn.__version__, scipy.__version__, numpy.__version__)"
```
The recorded results used Python 3.13.16, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.3,
pandas 3.0.5, FastAPI 0.149.0; website: React 19.3, Vite 8.3, Recharts 3.10. Other versions should give the same *workflow*, but
floating-point results (and tree-ensemble or GP fits) may differ slightly between library
versions and CPU architectures (Apple Silicon vs x86).

## 2. Data
- `data/synthetic_demo.csv` is deterministic: `generate_synthetic_dataset(500, seed=42)`.
- Expected SHA-256: `cd5d5ec0f182d2ff9537299e41847020a7bfc9feb3751830448cefef26792b96`
  (check with `shasum -a 256 data/synthetic_demo.csv` on macOS).
- If the file on disk no longer matches the registered checksum, the app shows a "Checksum mismatch" warning.

## 3. Re-running the recorded experiment
```bash
rm -f data/metaopt_ai.db*  models/*.joblib     # fresh state (optional)
python scripts/run_demo_pipeline.py            # writes sample_outputs/demo_results_summary.md
```
Fixed settings: seed 42, test fraction 0.2, 5-fold CV, 8 tuning iterations, optimization budget
1500, baseline = median design, objectives as listed in the summary.

## 4. What is recorded for every experiment (SQLite)
| Record | Stored reproducibility fields |
|---|---|
| Dataset | type, provenance, description, features, targets, SHA-256 checksum, file path, timestamp |
| Model experiment | dataset id + checksum, source type, model type, variant, features, target, all hyperparameters, seed, validation config (split, CV, search space), metrics, CV metrics, coverage, training ranges, artefact path, timestamp |
| Prediction | inputs, outputs, uncertainty, experiment ids per target, extrapolation flag, timestamp |
| Optimization | algorithm + settings, seed, budget and evaluations used, bounds, objective config (weights, normalization, kappa, penalty), constraints, baseline and candidate parameters, predictions and uncertainty, violations, feasibility, within-bounds flag, dataset checksum, runtime, timestamp |

Saved model artefacts (`models/*.joblib`) also contain the fitted pipeline, the test arrays, the
scikit-learn version and the dataset checksum, so diagnostic plots can be redrawn without retraining.

## 5. Reproducing a single run from the database
1. Experiment History → Model experiments → note `random_seed`, `validation_config`, `hyperparameters`.
2. ML Model Lab → choose the same features, targets, models, test fraction, seed and folds → Train.
3. For an optimization: Experiment History → Optimization runs → inspect the run (bounds, objective, seed,
   budget) → enter the same values in the Optimization Lab with the same active models.

## 6. Tests
```bash
python -m pytest -v
```
The tests use temporary databases and folders and never touch `data/metaopt_ai.db`.

## 7. Known sources of non-determinism
- Library version changes (scikit-learn tree construction, the SciPy optimizer random-number API).
  `utils/compat.py` passes `rng=` or `seed=`, whichever the installed SciPy supports.
- Timing values (training seconds, optimizer runtime) naturally vary.
