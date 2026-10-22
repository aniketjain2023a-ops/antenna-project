# MetaOpt-AI — Report Context File

> **What this file is.** Everything needed to write the project report, gathered in one place.
> Every number below was produced by this software, and was re-checked on 2026-10-22 against the
> running application (http://127.0.0.1:8000) and the SQLite database. Nothing is estimated or
> invented. Where something was **not** done, this file says so.
>
> **How to use it.** Give this file, plus the figures in `docs/report/figures/` and
> `results/metu_simpl/figures/`, to whoever (or whatever tool) is drafting the report. Follow the
> honesty rules in §12.

---

## 1. Project identity

| Item | Value |
|---|---|
| Title | MetaOpt-AI — Explainable Machine Learning for Antenna Design Optimization |
| Student | Aniket Jain |
| Registration No. | 23BEC0470 |
| Repository | `aniketjain2023a-ops/antenna-project`, branch `react-fastapi` |
| Type | Software project: ML surrogate modelling and optimization of antenna designs, delivered as a local website |

**One-paragraph summary.**
- MetaOpt-AI is a local website with a React frontend and a FastAPI backend.
- It trains regression surrogate models on antenna design data and predicts performance for a new design, with an approximate uncertainty.
- It searches for improved geometries with Differential Evolution, explains its models with feature importance, and stores every experiment in SQLite.
- Two datasets are used, kept separate:
  - METU-SIMPL: a published dataset of 5,000 microstrip patch antennas simulated in Ansys HFSS.
  - A labelled synthetic demo dataset.
- A second public dataset (Zenodo record 15866865) was requested but could not be downloaded, so it is not used.

**Note on the title.** The original brief said "metasurface antenna". The real data used (METU-SIMPL) are **probe-fed microstrip patch antennas, not metasurfaces**. The report title or abstract should reflect this honestly, for example "…for Microstrip Patch Antenna Optimization", or keep "metasurface" only as future-work motivation.

---

## 2. Problem statement and objectives

**Problem.** Full-wave electromagnetic simulation (HFSS, CST) of a single antenna design takes minutes to hours. Exploring an 11-dimensional design space by simulation alone is expensive.

**Approach.** Train a fast ML surrogate on existing simulations. Use it to predict performance instantly and to search the design space. Then pick candidates for real simulation.

**Objectives (all implemented):**
1. Ingest a verified public antenna dataset with provenance: source, commit and SHA-256 checksums.
2. Derive engineering targets from raw S11 frequency sweeps without inventing values for designs whose resonance falls outside the simulated band (censoring).
3. Train and compare several regressors, with a leakage-safe protocol and a Dummy baseline.
4. Predict all targets for any design, with approximate uncertainty and extrapolation warnings.
5. Optimize the geometry with Differential Evolution under bounds and optional constraints.
6. Explain the models (permutation and impurity importance).
7. Store everything in SQLite, export to CSV and present it in a three-page website.
8. Test it: unit, API and browser end-to-end tests.

---

## 3. System architecture

```
Browser (React 19 + TypeScript, Vite build)
   │  fetch('/api/…')  JSON
   ▼
FastAPI backend (backend/main.py, backend/service.py)  ── also serves the built website
   │
   ├── services/workflow.py   orchestration (train / predict / optimize)
   ├── ml/                    training, evaluation, prediction, uncertainty, importance
   ├── optimization/          objectives + SciPy Differential Evolution
   ├── datasets/              synthetic generator; public/ METU-SIMPL + Zenodo downloaders
   ├── database/              SQLAlchemy models, migrations, repository (SQLite)
   └── models/*.joblib        trained model artifacts
```

**Website pages:**

| Page | Content |
|---|---|
| Home | Workflow strip; dataset cards (provenance badge, sample count, best model per target); unavailable sources with the reason; data quality and checksum; model training panel with test metrics |
| Design & Optimize | Dataset selector. Parameter form generated from the dataset's inputs (units, valid ranges, inline validation). Live prediction (250 ms debounce) with ± uncertainty. Differential Evolution panel: objective, optional constraint, parameters to vary, bounds, baseline, budget, seed. Baseline vs ML-selected candidate cards with improvement deltas, convergence chart, and the nearest actually-simulated design |
| Results | Test R² chart across datasets and targets; full metrics table; per-model actual-vs-predicted, residuals, permutation and impurity importance; optimization history; CSV downloads |

**REST API:**

| Method and path | Purpose |
|---|---|
| GET `/api/health` | health check |
| GET `/api/datasets`, `/api/datasets/{id}` | datasets with features, targets, quality, active models; unavailable sources |
| GET `/api/datasets/{id}/preview`, `/experiments` | first rows; all trained models with metrics |
| POST `/api/datasets/{id}/train` | train models, select active model per target by CV R² |
| POST `/api/datasets/{id}/predict` | predict all targets (`save: true` stores it) |
| POST `/api/datasets/{id}/optimize` | Differential Evolution on the surrogate; stored as a run |
| GET `/api/experiments/{id}/diagnostics` | test-set points and feature importance |
| GET `/api/results/summary` | metrics across datasets and optimization history |
| GET `/api/export/{datasets,experiments,predictions,optimizations}.csv` | CSV export |

**Technology versions (as installed and tested):**
- Backend: Python 3.13.16 (also tested on 3.11.17); scikit-learn 1.9.1; SciPy 1.18.1; NumPy 2.5.3; pandas 3.0.5; FastAPI 0.149.0; uvicorn 0.53.0; SQLAlchemy 2.1.5.
- Frontend: React 19.3.1; Vite 8.3.4; TypeScript 7.0.2; Tailwind CSS 4.3.3; Recharts 3.10.1; react-router-dom 7.18.4; TanStack Query 5.104.5; Radix-based shadcn/ui-style components.

**Code size (lines):**
- Python packages: backend 505, services 345, ml 908, optimization 575, database 648, datasets 494, reporting 316, visualization 115, utils 217, scripts 385, tests 1,010.
- Frontend (`frontend/src`): 2,516.

**Database (current contents):** 38 experiments (trained models), 4 saved predictions, 18 optimization runs (including one from the report-figure capture); 5,500 stored designs (5,000 METU + 500 synthetic); 2 datasets. Tables cover datasets, experiments, predictions and optimization runs.

**UI design.** Design followed the installed Apple Design skill:
- System font with size-specific tracking.
- Translucent navigation bar (backdrop blur).
- Press feedback (scale 0.97).
- Inline validation; live feedback.
- Light and dark palettes.
- Reduced-motion, reduced-transparency and increased-contrast fallbacks.

---

## 4. Datasets

### 4.1 METU-SIMPL Microstrip-Patch (USED — main dataset)

| Item | Value |
|---|---|
| Source | https://github.com/METU-SIMPL/Microstrip-Patch, commit `7113818a` |
| Publication | E. S. Saçın and A. C. Durgun, "Microstrip Patch Antenna Modeling Using Neural Networks with Knowledge-Based Regularization," *Neural Computing and Applications*, vol. 37, pp. 3827–3837, 2025, doi:10.1007/s00521-024-10860-5 |
| Generation | Ansys HFSS full-wave simulation (from the repository's `antenna_model.aedt`) |
| Antenna | Probe-fed rectangular microstrip patch with solder-resist layer |
| Size | 5,000 designs; 0 duplicate rows |
| Frequency sweep | 23–33 GHz in 0.05 GHz steps (S11 in dB) |
| Licence | **No licence file** in the repository. The data are downloaded locally by the script and **not committed** to this repo; the paper is cited |
| Integrity | SHA-256 of each raw file in `data/raw/metu_simpl/MANIFEST.json`; processed-file SHA-256 `4b5d412ecb2679bb77ea9c6dc23684bc0deec4db5101f2fb6fd98ae0b361fe82` |

**11 input features (observed range and median):**

| Feature | Meaning | Min | Max | Median |
|---|---|---|---|---|
| patch_length_mm | patch length | 1.8057 | 5.1999 | 3.4998 |
| patch_width_mm | patch width | 1.8155 | 5.1987 | 3.5002 |
| patch_thickness_mm | copper thickness | 0.0100 | 0.0400 | 0.0250 |
| substrate_height_mm | substrate thickness | 0.1015 | 0.7991 | 0.4500 |
| solder_resist_height_mm | solder-resist thickness | 0.020 | 0.100 | 0.060 |
| probe_radius_mm | probe radius | 0.015 | 0.050 | 0.0325 |
| pad_clearance_mm | pad clearance | 0.000 | 0.025 | 0.0125 |
| antipad_clearance_mm | ground antipad clearance | 0.025 | 0.0998 | 0.0625 |
| feed_position_ratio | probe offset / patch length | 0.0507 | 0.4496 | 0.2500 |
| substrate_eps_r | substrate relative permittivity | 2.0037 | 4.9957 | 3.5000 |
| solder_resist_eps_r | solder-resist relative permittivity | 2.0002 | 4.9957 | 3.5000 |

**Targets derived from each S11 curve (our preprocessing, `datasets/public/metu_simpl.py`):**

| Target | Definition | Available rows |
|---|---|---|
| S11_28GHz_dB | S11 at 28.00 GHz | 5,000 |
| S11_min_dB | minimum S11 over 23–33 GHz | 5,000 |
| f_res_GHz | frequency of the S11 minimum; **missing (censored) if the minimum lies on a band edge**, because the true resonance is then outside the sweep | 2,866 (2,134 censored) |
| BW10_MHz | width of the contiguous S11 ≤ −10 dB region around the minimum; **missing unless the region lies fully inside the band** | 1,201 |

Total missing target values: 5,933. They are censored on purpose and never imputed. Each target is trained only on the rows where it exists.

### 4.2 Zenodo record 15866865 (NOT USED)

- The download was blocked by the build environment's network policy (HTTP 403).
- Its contents, authors and licence were never inspected, so it is not described, not used and must not be cited.
- The downloader exists (`python scripts/prepare_public_datasets.py zenodo`). A pipeline would be built only after the files are inspected.
- The website shows it as "Not available" with the reason.

### 4.3 Synthetic demo dataset (USED — clearly labelled)

- 500 rows, two inputs: `L_mm` (patch length) and `G_mm` (gap). Two targets: `Gain_dBi` and `S11_dB`.
- Generated from a documented analytic formula plus noise. It is **not physical data**.
- It exists to demonstrate and test the workflow. Because the formula is known, optimization results can be checked against the true function (the "oracle").

**Why the datasets are kept separate:** their inputs and targets differ. Pooling them would be scientifically meaningless.

---

## 5. Methodology

**Pipeline:** `scripts/prepare_public_datasets.py` runs these steps:
1. Download (if any file is missing).
2. Verify SHA-256.
3. Preprocess (derive targets).
4. Register in SQLite.
5. Train.
6. Export metrics, importance and figures to `results/metu_simpl/`.

**Leakage-safe ML protocol:**
- Split first: 80/20 train/test, random seed 42.
- 5-fold cross-validation and any tuning run on the **training split only**.
- The test split is used once, for the final metrics.
- The active model per target is chosen by **mean CV R²**, never by test score.
- The Dummy regressor (predicts the training mean) is the baseline every model must beat.
- Models available:
  - Dummy, Random Forest and Gradient Boosting (the defaults trained for METU).
  - Extra Trees, Hist-Gradient Boosting and Gaussian Process (used on the synthetic data).

**Metrics:** R², MAE and RMSE on the held-out test set; CV R² mean ± std.

**Uncertainty (approximate, not calibrated):**
- Tree ensembles: standard deviation across the individual trees' predictions.
- Gaussian Process: predictive standard deviation.
- Shown as "±" and never called a confidence interval.

**Explainability:**
- Permutation importance: drop in test R² when one feature is shuffled.
- Impurity importance: from the tree ensembles.

**Optimization:**
- Algorithm: SciPy `differential_evolution` (vectorized, Latin-hypercube initialization, no gradient polish), run on the surrogate.
- Objectives: minimize or maximize any target.
- Constraints: optional (e.g. S11 ≤ −10 dB), enforced with a penalty normalized by the target range.
- Free parameters: the user chooses which parameters vary. All others are held at the baseline value.
- Bounds must lie inside the training range. Extrapolation is refused (HTTP 422).
- Reproducibility: every run records seed, budget, evaluations, run time, free and fixed parameters.
- **Validation reference:** the nearest design that was actually simulated, by normalized Euclidean distance in input space, is shown next to the baseline and the candidate.

---

## 6. Results — METU-SIMPL (real HFSS data)

### 6.1 Model comparison (held-out test set, seed 42)

| Target | Train / test rows | Model | Test R² | MAE | RMSE | CV R² (5-fold) |
|---|---|---|---|---|---|---|
| S11 at 28 GHz (dB) | 4000 / 1000 | **Random Forest** | **0.441** | 1.110 | 2.117 dB | 0.392 ± 0.023 |
| | | Gradient Boosting | 0.258 | | | |
| | | Dummy | −0.000 | | | |
| Min S11 (dB) | 4000 / 1000 | **Random Forest** | **0.487** | 3.862 | 5.992 dB | 0.484 ± 0.033 |
| | | Gradient Boosting | 0.326 | | | |
| | | Dummy | −0.000 | | | |
| Resonant freq. (GHz) | 2292 / 574 | **Random Forest** | **0.679** | 1.143 | 1.550 GHz | 0.650 ± 0.034 |
| | | Gradient Boosting | 0.515 | | | |
| | | Dummy | −0.009 | | | |
| −10 dB bandwidth (MHz) | 960 / 241 | **Random Forest** | **0.547** | 248.8 | 329.3 MHz | 0.543 ± 0.047 |
| | | Gradient Boosting | 0.534 | | | |
| | | Dummy | −0.001 | | | |

**Interpretation:**
- Random Forest is best on all four targets and far above the Dummy baseline, so the models learn real structure.
- Accuracy is **moderate**: R² 0.44–0.68.
- S11 at a single frequency is hard to predict because it depends sharply on where the resonance falls.
- Resonant frequency is the most learnable target. That is consistent with cavity-model physics: resonance is set mainly by patch length and permittivity.
- These numbers are from our own pipeline. They are **not** comparable to the paper's neural-network results, which use a different target (the full S11 curve) and a different protocol.

### 6.2 Feature importance (permutation, Random Forest)

| Target | Top features (importance = drop in R²) |
|---|---|
| f_res_GHz | patch length 1.17, substrate εr 0.84 |
| BW10_MHz | substrate height 0.60, patch length 0.34 |
| S11_28GHz_dB | patch length 0.42, substrate εr 0.36 |

Physical consistency (textbook patch theory, Balanis ch. 14):
- f_res ≈ c / (2 L √εeff), so length and permittivity dominate the resonance.
- Bandwidth grows with substrate thickness.

The model rediscovered both relationships from data. This is a qualitative sanity check, not proof of correctness.

### 6.3 Prediction example — the median design

| Target | Prediction | ± (tree spread) |
|---|---|---|
| S11 at 28 GHz | −0.549 dB | 0.22 |
| Min S11 | −2.306 dB | 1.74 |
| Resonant frequency | 24.807 GHz | 2.91 |
| −10 dB bandwidth | 827.7 MHz | 304 |

### 6.4 Optimization case study (run #16)

**Settings:**
- Objective: minimize S11 at 28 GHz.
- Algorithm: Differential Evolution, budget 1,500, seed 42.
- Run: 1,485 surrogate evaluations, 4.33 s.
- Free parameters: patch length, patch width, feed position. The other 8 are held at the median baseline.
- The run is deterministic: repeating it with the same seed for the report screenshot (Figure 4) gave identical values.

| Quantity | Baseline (median design) | ML-selected candidate |
|---|---|---|
| Patch length | 3.4998 mm | 2.4417 mm |
| Patch width | 3.5002 mm | 2.6495 mm |
| Feed position | 0.2500 | 0.4467 |
| Predicted S11 @ 28 GHz | −0.549 dB | **−6.446 dB** (± 5.33) |
| Predicted min S11 | −2.306 dB | −18.957 dB |
| Predicted f_res | 24.81 GHz | 27.887 GHz |
| Predicted BW10 | 827.7 MHz | 1,492.3 MHz |

**Nearest actually-simulated designs (HFSS ground truth from the dataset):**

| Design | Dataset row | Normalized distance | HFSS S11 @ 28 GHz | HFSS min S11 | HFSS f_res |
|---|---|---|---|---|---|
| Near baseline | 3254 | 0.436 | −0.560 dB | −1.835 dB | — |
| Near candidate | 1622 | 0.385 | −2.783 dB | −4.092 dB | 26.5 GHz |

**Interpretation:**
- The optimizer shortened the patch, which moved the predicted resonance from about 24.8 GHz toward 28 GHz. That is physically the correct direction, since a shorter patch gives a higher resonance.
- The predicted 28 GHz match improves by about 5.9 dB. However:
  - The uncertainty is large (± 5.33 dB).
  - The nearest real simulation to the candidate is not identical to it, and reaches only −2.78 dB.
- **The candidate itself has not been simulated.** The result is a promising starting point for an HFSS run, not a validated design.

### 6.5 Exported figures (`results/metu_simpl/figures/`)

- `model_comparison_r2.png`
- `actual_vs_predicted_{S11_28GHz_dB, S11_min_dB, f_res_GHz, BW10_MHz}.png`
- `residuals_{…}.png` (the same four targets)
- `permutation_importance_{…}.png` (the same four targets)

Tables: `results/metu_simpl/metrics.csv`, `feature_importance.csv`, `selected_models.json`.

---

## 7. Results — synthetic demo (labelled synthetic)

**Models (test R² for Gain / S11):**
- Gaussian Process (active model): 0.995 / 0.959. RMSE 0.156 dBi for Gain, 0.800 dB for S11.
- Random Forest: 0.983 / 0.907.
- Extra Trees: 0.990 / 0.951.
- Gradient Boosting: 0.983 / 0.844.
- Dummy: ≈ 0.

**Optimization run #17:**
- Objective: maximize gain subject to S11 ≤ −10 dB. 810 evaluations, 0.08 s.
- Geometry: L 10.2536 → 10.1714 mm; G 1.1002 → 1.2812 mm.
- Predicted: gain 7.784 → 8.017 dBi; S11 −14.793 → −15.529 dB (constraint met).
- **True function (oracle):** gain 7.796 → 8.024 dBi; S11 −13.965 → −15.172 dB.
- This confirms that the surrogate-based optimization finds a real improvement **on data where the truth is known**.

Synthetic results show that the workflow works. They say nothing about real antennas.

---

## 8. Testing and verification

| Suite | Tests | Status (2026-10-22) |
|---|---|---|
| test_database | 12 | pass |
| test_data_validation | 13 | pass |
| test_ml_pipeline | 12 | pass |
| test_optimization | 8 | pass |
| test_public_datasets | 7 | pass |
| test_reporting | 3 | pass |
| test_api (FastAPI TestClient) | 5 | pass |
| e2e browser (Playwright + Chromium, live site) | 5 | pass |

- `python -m pytest`: **62 passed, 5 skipped**. The 5 skipped tests are the browser tests, which run only when `METAOPT_E2E_URL` is set.
- The e2e run against the live site passed 5/5.
- The frontend type-check `npx tsc -b` passes, and the production build succeeds.

**Clean-environment reproduction:**
- Setup: a fresh clone and a fresh Python 3.11 virtual environment, then the dataset script and `./start.sh`.
- Result: all e2e tests pass, and the METU metrics reproduced exactly.

**What the API tests check:**
- Models are selected by CV and never the Dummy.
- Invalid inputs are rejected (422).
- Extrapolation is flagged.
- Optimizer results stay within bounds and satisfy the constraint.
- Held parameters stay fixed.
- Out-of-range bounds are refused.
- CSV exports work.

---

## 9. Report figures (website screenshots)

The screenshots are in `docs/report/figures/`, captured from the running application at 1440 × 900.

| File | Suggested caption |
|---|---|
| `fig01_home_hero.png` | Home page: workflow overview and dataset cards with provenance |
| `fig02_home_dataset_training.png` | Home page: dataset quality, provenance and model-training results |
| `fig03_design_prediction.png` | Design & Optimize: parameter form generated from the dataset and live prediction with uncertainty |
| `fig04_design_optimization_result.png` | Design & Optimize: Differential Evolution setup and baseline vs ML-selected candidate |
| `fig05_results_comparison.png` | Results: test R² across datasets, targets and models |
| `fig06_results_diagnostics.png` | Results: actual-vs-predicted, residuals and feature importance for one model |

---

## 10. Limitations (state these in the report)

1. **No physical validation.** No optimized candidate was simulated in HFSS or measured.
2. **Moderate accuracy.** R² ranges from 0.44 to 0.68 on METU. The S11 error at a single frequency (RMSE 2.1 dB) is large relative to the improvements claimed.
3. **Uncertainty is not calibrated.** Tree spread and GP std are heuristics, not confidence intervals.
4. **Censoring.** f_res and BW10 exist only for designs whose resonance or band lies inside 23–33 GHz. Those models are trained on a selected subset, which creates selection bias.
5. **Only one real dataset.** Zenodo could not be obtained, so the cross-dataset comparison was not possible.
6. **Patch, not metasurface.** The real data are microstrip patches.
7. **Gain and efficiency are not available** in METU-SIMPL. Only S11-derived quantities are.
8. Optimization uses the surrogate inside the training bounds only, and one objective at a time (no Pareto front).

---

## 11. Future work

- Simulate the candidate designs in HFSS and add them back to the training data (active learning).
- Obtain and inspect the Zenodo dataset, and build a separate pipeline for it.
- Predict the full S11 curve (multi-output or neural), as in the source paper.
- Calibrate the uncertainty (e.g. conformal prediction, quantile regression forests).
- Multi-objective optimization (e.g. NSGA-II) for match versus bandwidth.
- Apply the method to actual metasurface unit-cell data.

---

## 12. Honesty rules for writing the report

- Use **only** the numbers in this file or in `results/`. Do not round them into stronger claims (e.g. do not say "high accuracy" for R² 0.44).
- Always say "**predicted**" or "**ML-selected candidate**", never "optimized antenna achieves…".
- Label synthetic results as synthetic every time.
- Do not describe or cite Zenodo record 15866865. Say only that it was unavailable.
- References: use `REFERENCES.md` Part A, and check each DOI before citing. Do **not** cite Part B until you have read the paper. The METU paper's DOI is 10.1007/s00521-024-10860-5.
- Write the literature review from papers you have actually read.

---

## 13. Suggested report outline

1. Abstract
2. Introduction: motivation, cost of EM simulation, surrogate modelling
3. Literature review (verified sources only)
4. Datasets: METU-SIMPL, target derivation, censoring; Zenodo status; synthetic data
5. Methodology: ML protocol, models, uncertainty, XAI, Differential Evolution
6. System design: architecture, API, database, UI (with Figures 1–6)
7. Results: model comparison, importance, prediction, optimization case study, synthetic check
8. Testing and verification
9. Discussion and limitations
10. Conclusion and future work
11. References
12. Appendix: run instructions (§14), API table

---

## 14. How to run (Mac)

```bash
git fetch origin && git checkout react-fastapi
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_public_datasets.py   # once: downloads + trains METU-SIMPL (~2 min)
./start.sh                                   # open http://127.0.0.1:8000
```

Requirements: Python 3.11+ and Node.js 20.19+ or 22.12+.

Tests:
```bash
python -m pytest
METAOPT_E2E_URL=http://127.0.0.1:8000 python -m pytest tests/e2e -v   # with the site running
```
