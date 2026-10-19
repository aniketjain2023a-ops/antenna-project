# Three-minute demonstration script

**Before the demo:** `source .venv/bin/activate && streamlit run app.py`. Make sure models are
trained (Dataset & Model → **Train models**, or `python scripts/run_demo_pipeline.py --quick`).
Keep the browser at http://localhost:8501. The three sections are in the top navigation bar.

| Time | Section | Do | Say |
|---|---|---|---|
| 0:00-0:25 | Dataset & Model | Point at the badges under the title and the amber notice | "MetaOpt-AI is a local platform for exploring metasurface antenna geometry with ML surrogates. The badges show which dataset is active and that it is SYNTHETIC, generated from a documented formula. The app says so on every page and in every report." |
| 0:25-0:50 | Dataset & Model | Data quality card; open *Provenance*; mention *Import a dataset* | "Every dataset has a type, provenance and a SHA-256 checksum in SQLite. Missing values are never filled in, and outliers are only flagged. Real EM-simulation or measured data can be imported here, with column mapping and a provenance confirmation." |
| 0:50-1:20 | Dataset & Model | Results: comparison table and the two actual-vs-predicted plots | "Six model types including a Dummy baseline. The data is split first, cross-validation and tuning use training data only, and the active model is chosen by cross-validated R², so the test set is never used for selection. The Gaussian Process reaches R² 0.96 for S11 and 0.995 for gain." |
| 1:20-1:45 | Predict & Optimize | Press **Predict performance**; then type L = 14 | "Enter a geometry and get predicted S11 and gain with an approximate uncertainty and the model's validation metrics. Outside the training range the status turns amber: that is extrapolation." |
| 1:45-2:20 | Predict & Optimize | Max gain, S11 ≤ −10 dB → **Run optimization** | "Optimization is a separate step. Differential Evolution calls the trained surrogate within the bounds. The candidate, L ≈ 10.17 mm and G ≈ 1.28 mm, raises predicted gain from 7.78 to 8.02 dBi while still meeting the S11 constraint. Green means better. It is an ML-selected candidate, not a validated antenna." |
| 2:20-2:45 | Results & History → Explainability | Permutation importance and the sensitivity curve | "L dominates the predictions, and the sensitivity curve shows gain peaking near resonance. These are associations the model learned, not proof of causation." |
| 2:45-3:00 | Results & History → Data & export | **Generate PDF report**, CSV downloads | "Everything is stored in SQLite and exported as a PDF and CSVs with seeds, checksums and versions, so each experiment can be reproduced. The next step is real EM-simulation data." |

**Backup if something is slow:** open `sample_outputs/sample_report.pdf` and `sample_outputs/demo_results_summary.md`.
