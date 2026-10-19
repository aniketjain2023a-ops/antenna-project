# Experiment log

Record every experiment you run or report. The SQLite database stores the machine-readable
details; this log records *why* you ran each experiment and what you concluded.

## Template

```
### EXP-YYYYMMDD-NN  <short title>
- Date / time:
- Purpose / hypothesis:
- Dataset: id ___  name ___  type [SYNTHETIC | PUBLIC | SIMULATION | MEASUREMENT]
- Dataset SHA-256 (first 16 chars):
- Features:            Targets:
- Models (variant):
- Split / CV / seed:   test=__  folds=__  seed=__
- Tuning: on/off  iterations=__  search space changes:
- DB experiment IDs:
- Key metrics (test R2 / RMSE, CV R2, coverage95):
- Optimization: objective ___ constraints ___ bounds ___ algorithm(s) ___ budget ___ seed ___
- DB optimization run IDs:
- Candidate (predicted):            Uncertainty:
- Report file:
- Observations:
- Conclusion (state limits, e.g. "synthetic data - workflow demonstration only"):
- Follow-up actions (e.g. EM-simulate candidate in solver X):
```

## Recorded entries

### EXP-20261019-01  Full demonstration pipeline (synthetic)
- Purpose: verify the end-to-end workflow and produce sample results for the report.
- Dataset: id 1, "Synthetic demonstration dataset v1 (L_mm, G_mm)", SYNTHETIC, SHA-256 cd5d5ec0f182d2ff...
- Features: L_mm, G_mm. Targets: S11_dB, Gain_dBi.
- Models: Dummy, RF, ET, GB, HGB, GP (default + tuned). Split 0.2, 5-fold CV, seed 42, tuning 8 iterations.
- Key metrics: GP (tuned) S11 test R2 0.961 / RMSE 0.786 dB, CV R2 0.967; GP gain R2 0.995 / RMSE 0.156 dBi.
- Optimization: max gain s.t. S11 <= -10 dB (DE, multi-start NM, dual annealing; budget 1500; seed 42)
  -> L=10.171 mm, G=1.281 mm, predicted gain 8.02 dBi, S11 -15.5 dB (baseline 7.78 dBi).
- Report: sample_outputs/sample_report.pdf; details in sample_outputs/demo_results_summary.md.
- Conclusion: the workflow works and the three optimizers agree. Synthetic data, so no physical conclusion.
- Follow-up: replace with EM-simulation data; repeat.
