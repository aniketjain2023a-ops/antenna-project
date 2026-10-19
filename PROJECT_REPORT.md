# MetaOpt-AI: Explainable Machine Learning for Metasurface Antenna Optimization

**Student:** Aniket Jain  **Registration No.:** 23BEC0470
**Domain:** Antenna & Microwave Engineering · Machine Learning · Numerical Optimization · Explainable AI

> All quantitative results in this report were produced by this software on a **SYNTHETIC**
> demonstration dataset (`sample_outputs/demo_results_summary.md`). They demonstrate the
> workflow. They are **not** electromagnetic-simulation or measurement results.

---

## 1. Abstract

Designing metasurface-loaded antennas usually means many full-wave electromagnetic (EM)
simulations, because matching (S11) and gain depend non-linearly, and in coupled ways, on
geometry. This project implements **MetaOpt-AI**, a locally running platform that combines
(i) dataset management with provenance and checksums in SQLite, (ii) leakage-safe training
and comparison of six regression surrogates (Dummy, Random Forest, Extra Trees, Gradient
Boosting, Histogram Gradient Boosting and Gaussian Process), (iii) prediction with
approximate uncertainty and extrapolation detection, (iv) constrained surrogate-based
optimization with three SciPy optimizers, (v) explainability (impurity and permutation
importance, sensitivity, partial dependence), and (vi) reproducible reporting.

On a 500-sample synthetic dataset built from a documented, physics-inspired formula, the
Gaussian Process surrogate reached test R² = 0.961 for S11 (RMSE 0.79 dB) and 0.995 for gain
(RMSE 0.16 dBi). Its 95 % interval coverage on the test set was 0.94-0.95. All three
optimizers converged to the same constrained optimum (predicted gain 8.02 dBi at S11 ≈
-15.5 dB), which matches the noise-free generating formula to within 0.01 dBi. These
results show that the workflow is correct and consistent. They do not establish accuracy
for real antennas.

## 2. Introduction

Metasurfaces are thin, engineered arrays of sub-wavelength elements that shape
electromagnetic fields (Holloway et al., 2012). When placed above or around a patch
radiator, they can change impedance matching, bandwidth and gain. Exploring their geometry
with full-wave solvers is accurate but slow. Data-driven surrogate models are a common way
to cut this cost: a regression model learns the map from geometry to performance from a set
of simulations, and an optimizer then searches the cheap surrogate (Jones et al., 1998;
Koziel & Ogurtsov, 2014).

Surrogates bring three practical problems: (1) users cannot easily see *why* a model makes
a prediction, (2) predictions can be confidently wrong away from the training data, and
(3) experiments are hard to reproduce when data, model settings and seeds are not
recorded. MetaOpt-AI addresses all three in one tool.

## 3. Problem statement

Build a local, reproducible software workflow that, given a table of antenna geometry
parameters and performance values:
- predicts S11 and gain for new geometries, with a quantified, honestly-labelled uncertainty;
- finds geometries that optimize an explicit objective under explicit constraints;
- explains which parameters drive the predictions;
- records the provenance of every dataset, model, prediction and optimization; and
- never presents synthetic or predicted results as physical validation.

## 4. Objectives

1. Store datasets and experiment records in SQLite with provenance, checksums and referential integrity.
2. Validate and preprocess data without silently modifying it.
3. Train, validate, tune and compare several regression models without data leakage.
4. Provide prediction with uncertainty and extrapolation warnings.
5. Optimize geometry on the surrogate with two or more algorithms, explicit constraints and documented normalization.
6. Explain predictions (importance, sensitivity, partial dependence) and map uncertainty over the design space.
7. Generate PDF/CSV reports, keep experiments reproducible, and supply automated tests and documentation.

## 5. Literature review (context only)

*This section describes the research area in general terms. Specific papers in REFERENCES.md
Part B are unverified and must be checked before they are cited.*

- **Antenna fundamentals.** A rectangular patch resonates when its effective length is about half
  a guided wavelength. Fringing fields extend the effective length, and the effective
  permittivity lies between 1 and εr (Balanis, 2016). S11 measures reflected power. Mismatch
  reduces realized gain by 10·log10(1-|Γ|²) (Pozar, 2012).
- **Metasurfaces.** Holloway et al. (2012) review metasurfaces as the 2-D counterparts of
  metamaterials. Their reflection and transmission phase can be engineered to improve
  antenna gain or bandwidth.
- **Surrogate-based design.** Replacing expensive simulators with fast approximations, then
  optimizing those approximations, is well established in engineering design (Jones et al.,
  1998; Koziel & Ogurtsov, 2014). Gaussian processes are popular because they also provide a
  predictive variance (Rasmussen & Williams, 2006).
- **Tree ensembles.** Random Forests (Breiman, 2001), Extra Trees (Geurts et al., 2006) and
  gradient boosting (Friedman, 2001) are strong general-purpose regressors for tabular data.
- **Explainability.** Permutation importance (Breiman, 2001; Fisher et al., 2019) and partial
  dependence (Friedman, 2001) are model-agnostic ways to explain a model. Both describe
  associations, not physical causes.
- **Global optimization.** Differential Evolution (Storn & Price, 1997), Nelder-Mead (1965) and
  generalized simulated annealing (Tsallis & Stariolo, 1996) are derivative-free, which suits
  piecewise-constant tree surrogates.
- **ML in antenna design.** This is an active research area (see the reviews listed as UNVERIFIED
  in REFERENCES.md). **This project does not claim to have invented ML-based antenna optimization.**

## 6. Research gap and intended contribution

Many published studies report a surrogate and an optimized design, but the provenance,
uncertainty and explanation steps are often handled separately or only briefly. The
contribution of this project is **engineering and integration**: one reproducible,
open-source workflow in which:
1. every dataset carries a type label (SYNTHETIC / PUBLIC / SIMULATION / MEASUREMENT), provenance text and SHA-256 checksum, and every downstream record inherits them;
2. model selection uses cross-validation only, so the test set is never used for selection or tuning;
3. uncertainty is reported together with an **empirical coverage check** on held-out data instead of being presented as calibrated;
4. optimization uses explicit, range-normalized objectives and constraints, three algorithms for cross-checking, and an optional risk-averse (mean ± κσ) mode;
5. explanations and uncertainty maps point to where new EM simulations would be most informative. The benefit of such sampling is **not** claimed or demonstrated.

## 7. Proposed methodology

```
Dataset → validation & provenance (SQLite) → leakage-safe training (CV, tuning)
        → evaluation (R², MAE, RMSE, coverage) → prediction + uncertainty
        → surrogate optimization (DE / NM / DA) → explainability → reports
        → [outside this app] EM simulation / measurement of candidates → new data
```

## 8. Architecture

Layered Python application (see README §3):

| Layer | Modules | Responsibility |
|---|---|---|
| Presentation | `app.py`, `ui/*` | Streamlit multipage UI (9 pages); no business logic |
| Services | `services/workflow.py` | orchestration shared by UI, demo script and tests |
| Domain | `ml/*`, `optimization/*`, `datasets/*`, `reporting/*`, `visualization/*` | algorithms |
| Data access | `database/repository.py` | transactions, CRUD, duplicate checks |
| Persistence | `database/models.py`, `db.py`, `migrations.py` | SQLAlchemy ORM, SQLite, schema versions |

## 9. Dataset and provenance

The synthetic dataset (500 Latin-hypercube samples, seed 42) uses a half-wave resonance with
assumed εeff = 2.0 at F0 = 10 GHz, gap-dependent capacitive extension of the effective
length, a single-resonance reflection model with loaded Q = 5, best match near G = 0.8 mm, a
metasurface gain term that peaks near G = 1.3 mm, mismatch loss, and Gaussian noise (0.6 dB on
S11, 0.15 dB on gain). All equations are in `datasets/synthetic.py` and `data/README.md`.
Because the best-match gap and the best-gain gap differ, the dataset contains a real
trade-off between matching and gain. Only 58 of 500 samples have S11 ≤ -10 dB, which mimics
the narrow resonance valley of a real antenna.

Dataset SHA-256: `cd5d5ec0f182d2ff9537299e41847020a7bfc9feb3751830448cefef26792b96`.

User data (CSV/Excel) goes through type detection, explicit column mapping, a quality
report, non-destructive cleaning, mandatory provenance, and a confirmation step before
registration.

## 10. Database design

| Table | Key fields | Integrity |
|---|---|---|
| `schema_version` | version, description, applied_at | migration log |
| `datasets` | name (unique), dataset_type (CHECK), provenance, sample_count, input_features, target_names, checksum (unique), file_path | duplicate prevention by checksum and name |
| `antenna_designs` | dataset_id → datasets (CASCADE), row_index, parameters JSON, targets JSON, source_type, validation_status | unique (dataset_id, row_index) |
| `model_experiments` | dataset_id → datasets, model_type, variant, target, features, hyperparameters, seed, validation_config, r2/mae/rmse, cv metrics, coverage, train_ranges, artifact_path, dataset_checksum, source_type | FK + cascade |
| `prediction_records` | model_experiment_id → experiments, inputs, outputs, uncertainty, extrapolation flag | FK |
| `optimization_runs` | model_experiment_id(s), algorithm, seed, budget, bounds, objective config, constraints, baseline & candidate parameters/predictions/uncertainty, violations, feasibility, status | FK |
| `app_settings` | key/value JSON (active dataset, active models) | — |

SQLite foreign keys are enabled on every connection (`PRAGMA foreign_keys=ON`). Every
repository call runs in a commit-or-rollback transaction.

## 11. ML algorithms

All models are scikit-learn `Pipeline`s (scaler → model). A scaler is used only for the GP.
- **Dummy (mean):** the reference any useful model must beat.
- **Random Forest / Extra Trees:** bagged decision-tree ensembles. Uncertainty is approximated by the spread of individual tree predictions.
- **Gradient Boosting / HistGradientBoosting:** stage-wise trees fitted to residuals.
- **Gaussian Process:** constant × RBF/Matérn (ARD) kernel + white noise, `normalize_y`. Provides a predictive standard deviation. Training cost is O(n³), so it is capped at 1500 training rows.

**Protocol:** seeded 80/20 split → 5-fold CV on training data → optional RandomizedSearchCV
(training data only) → single evaluation on the test set. The active model for each target
is the one with the highest CV R². Datasets below 20 rows are refused, below 50 rows CV is
disabled, and below 150 rows the app warns that estimates may be unstable.

## 12. Optimization algorithms

- **A. Differential Evolution** (SciPy, vectorized population evaluation, Latin-hypercube init, no gradient polish).
- **B. LHS multi-start Nelder-Mead** (space-filling screening, then bounded simplex searches from the best points).
- **C. Dual annealing** (generalized simulated annealing, local search disabled).

Objectives: (1) maximize gain s.t. S11 ≤ limit; (2) minimize S11 s.t. gain ≥ limit;
(3) weighted trade-off of range-normalized desirabilities, d = (value - min)/(max - min)
oriented so that 1 is best, so dB and dBi are never added directly. Constraints use a
static penalty: 100 × Σ violation/range. Every candidate is checked against the bounds. An
optional κ > 0 replaces predictions by pessimistic values mean ∓ κσ. Bounds default to the
training range, and extrapolating bounds are rejected unless the user explicitly allows them.

## 13. Experimental setup

Synthetic dataset (above); seed 42; test fraction 0.2; 5-fold CV; tuning with 8
RandomizedSearchCV iterations per model; optimization budget 1500 surrogate evaluations per
algorithm; baseline = median training design (L = 10.254 mm, G = 1.100 mm). Software: Python
3.13.16, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.3 (Linux container used for the recorded
run; the code is platform-independent). Command: `python scripts/run_demo_pipeline.py`.

## 14. Results (synthetic data)

**Model comparison (test set; selected rows - full table in `sample_outputs/demo_results_summary.md`):**

| Target | Model (variant) | Test R² | Test RMSE | CV R² (mean ± sd) | Coverage95 |
|---|---|---|---|---|---|
| S11_dB | Dummy | -0.012 | 3.985 dB | -0.006 ± 0.005 | - |
| S11_dB | Gradient Boosting (default) | 0.844 | 1.566 dB | 0.846 ± 0.041 | - |
| S11_dB | Random Forest (default) | 0.907 | 1.208 dB | 0.913 ± 0.025 | 0.83 |
| S11_dB | Extra Trees (default) | 0.951 | 0.877 dB | 0.947 ± 0.020 | 0.84 |
| S11_dB | Gaussian Process (tuned) | **0.961** | **0.786 dB** | **0.967 ± 0.007** | 0.95 |
| Gain_dBi | Dummy | -0.004 | 2.309 dBi | -0.015 ± 0.015 | - |
| Gain_dBi | Random Forest (default) | 0.983 | 0.303 dBi | 0.980 ± 0.007 | 0.95 |
| Gain_dBi | Extra Trees (default) | 0.990 | 0.229 dBi | 0.989 ± 0.003 | 1.00 |
| Gain_dBi | Gaussian Process (tuned) | **0.995** | **0.156 dBi** | **0.996 ± 0.001** | 0.94 |

**Tuning (measured):** tuning lowered test RMSE for Gradient Boosting (S11 −26.0 %, gain
−17.8 %) and HistGradientBoosting (−19.7 %, −8.9 %). It made **no meaningful difference** for
the GP on gain, gave a 1.8 % gain for the GP on S11, and **made Random Forest and Extra Trees
worse** on both targets (e.g., Extra Trees S11 RMSE rose from 0.877 to 1.041 dB). The
default settings were already well suited, and the small search spaces did not contain them.

**Explainability:** the permutation importance of L_mm (drop in R² ≈ 1.8-1.9) is larger than
that of G_mm (0.37 for gain, 0.68 for S11). This matches the generator, where L sets the
resonance and G mainly shifts it slightly and changes matching and the metasurface gain term.

**Optimization (all three algorithms, predictions):**

| Objective | Candidate (L, G) [mm] | Pred. S11 | Pred. gain | Formula S11 / gain (noise-free) |
|---|---|---|---|---|
| Baseline | (10.254, 1.100) | -14.80 dB | 7.78 dBi | -13.96 / 7.80 |
| Max gain, S11 ≤ -10 dB | (10.171, 1.281) | -15.50 dB | 8.02 dBi | -15.18 / 8.02 |
| Min S11, gain ≥ 7 dBi | (9.936, 0.853) | -22.82 dB | 7.29 dBi | -24.87 / 7.31 |
| Trade-off 0.5/0.5, S11 ≤ -10 | (9.945, 0.876) | -22.77 dB | 7.33 dBi | -24.50 / 7.36 |

DE used 780-1050 evaluations, multi-start NM 871-901, and dual annealing the full 1500. All
three reached the same optimum (to within 0.01 mm). DE was fastest (≈0.1 s versus ≈2 s for
dual annealing) because it evaluates the whole population in one vectorized call.

## 15. Discussion

- The gap between the Dummy model (R² ≈ 0) and the real models shows that the models learn
  structure, not just the mean.
- S11 is harder to learn than gain (R² 0.96 vs 0.995) because of the sharp, narrow
  resonance valley. Tree models approximate it with steps, while the GP fits it smoothly.
- The GP's 95 % interval coverage on the test set (0.94-0.95) is close to nominal *on this
  data*. Tree-spread intervals under-cover S11 (0.83-0.84), meaning they are over-confident.
  This is why uncertainty is labelled approximate.
- At the S11-optimal design, the surrogate is less negative (-22.8 dB) than the noise-free
  formula (-24.9 dB). Very deep nulls are hard to learn from noisy, sparse samples. This
  illustrates that a "deeper S11" prediction can be wrong in magnitude even when the
  location is right.
- The S11-optimal and gain-optimal designs differ: the best match (G ≈ 0.85 mm) gives 0.7 dB
  less gain than the gain-optimal design (G ≈ 1.28 mm). **A deeper S11 does not imply a
  better antenna.**

## 16. Limitations

1. The bundled data is synthetic; no claim about real antennas can be made.
2. Regression surrogates do not solve Maxwell's equations; candidates need full-wave EM verification and measurement.
3. Only two geometry parameters and two targets; real designs also depend on substrate, feed, unit-cell shape, number of cells, frequency and fabrication tolerance.
4. S11 is modelled at a single frequency; there is no bandwidth model, because the bandwidth data needed for one does not exist.
5. Uncertainty is approximate. The coverage check uses one test split and is not proof of calibration.
6. Penalty-based constraint handling and a weighted-sum trade-off cannot reach non-convex parts of a Pareto front. A true multi-objective optimizer (e.g., NSGA-II) is not implemented.
7. Uncertainty-guided sampling is suggested but its benefit has **not** been demonstrated.
8. The recorded run used a Linux container. The code targets macOS too, but timings will differ.

## 17. Future scope

- Import real EM-simulation sweeps (e.g., an open-source solver such as openEMS) or measured data, and repeat the experiments.
- Add frequency-swept S11 to derive the -10 dB bandwidth honestly.
- Run an active-learning study: compare uncertainty-guided sampling with random sampling on real simulation budgets.
- Add a proper multi-objective optimizer and calibrated uncertainty (e.g., conformal prediction).
- Add more geometry parameters and inverse-design models.

## 18. Conclusion

MetaOpt-AI shows a complete, tested, reproducible workflow for ML-assisted exploration of
metasurface antenna geometry. It keeps provenance, uncertainty and explanation together with
prediction and optimization. On synthetic data, the models and optimizers recovered the
known generating formula consistently, and the software labels these outputs as
demonstrations rather than physical validation. The real engineering value of the
platform will only be known once it is applied to genuine EM-simulation or measurement data.

## 19. References

See `REFERENCES.md` (Part A: established references; Part B: unverified context to be checked).
