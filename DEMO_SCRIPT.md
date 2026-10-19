# Three-minute demonstration script

**Before the demo:** `source .venv/bin/activate && streamlit run app.py`. Make sure models are
trained (Overview → Quick start, or `python scripts/run_demo_pipeline.py --quick`).
Keep the browser at http://localhost:8501.

| Time | Page | Do | Say |
|---|---|---|---|
| 0:00-0:20 | Overview | Point at the yellow banner and the metric cards | "MetaOpt-AI is a local platform for exploring metasurface antenna geometry with ML surrogates. This dataset is SYNTHETIC, generated from a documented formula, and the app says so on every page and in every report." |
| 0:20-0:45 | Dataset Management | Scroll through provenance, the checksum, the quality table and the correlation map | "Every dataset carries a type, provenance and a SHA-256 checksum in SQLite. Missing values are never imputed, and outliers are only flagged. Uploads need column mapping and a provenance confirmation." |
| 0:45-1:15 | ML Model Lab | Show the comparison table and chart; open actual-vs-predicted for the GP | "Six models including a Dummy baseline. We split first, run CV and tuning on training data only, and choose the active model by CV R², so the test set is never used for selection. The GP reaches R² 0.96 for S11 and 0.995 for gain. Tuning did not always help, and the app reports that honestly." |
| 1:15-1:35 | Prediction | Enter L=10, G=1, click Predict; then L=14 | "Predictions come with an approximate uncertainty. Outside the training range the app warns that this is extrapolation." |
| 1:35-2:15 | Optimization Lab | Max gain s.t. S11 ≤ -10, DE + Nelder-Mead, Run | "The optimizers call the trained surrogate. Both find L ≈ 10.17 mm, G ≈ 1.28 mm with predicted gain 8.02 dBi versus 7.78 for the baseline. The trade-off plot shows the predicted Pareto front, and the oracle check shows the surrogate matches the synthetic formula. This is an ML-selected candidate, not a validated antenna." |
| 2:15-2:40 | Explainable AI | Permutation importance, then the uncertainty map | "L dominates the predictions. That is an association learned by the model, not proof of causation. The uncertainty map marks where new EM simulations would be most informative. We do not claim this improves accuracy." |
| 2:40-3:00 | Report Generator / History | Click Generate PDF, then Download | "Everything is stored in SQLite and exported as a PDF and CSVs with seeds, checksums and versions, so the experiment is reproducible. The next step is real EM-simulation data." |

**Backup if something is slow:** open `sample_outputs/sample_report.pdf` and `sample_outputs/demo_results_summary.md`.
