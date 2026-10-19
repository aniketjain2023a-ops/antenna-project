# Three-minute demonstration script

**Before the demo:** `source .venv/bin/activate`, run `python scripts/prepare_public_datasets.py`
once, then `streamlit run app.py`. Keep the browser at http://localhost:8501. The three pages are in
the top navigation bar.

| Time | Page | Do | Say |
|---|---|---|---|
| 0:00-0:25 | Home | Point at the badges under the title | "MetaOpt-AI predicts and optimizes antenna performance with ML surrogates. The active dataset is METU-SIMPL: 5,000 microstrip patch designs simulated in HFSS from 23 to 33 GHz, published with a journal paper. The badge marks it as EM simulation data, not measurements." |
| 0:25-0:50 | Home | Data quality card; open *Provenance* | "Every dataset has its source, citation and a SHA-256 checksum in SQLite. From the simulated S11 curves I derive S11 at 28 GHz, the minimum S11, the resonant frequency and the −10 dB bandwidth. Where the resonance lies outside the band, frequency and bandwidth are left missing, not invented." |
| 0:50-1:20 | Home | Results table and actual-vs-predicted plots | "Dummy, Random Forest and Gradient Boosting, evaluated on a held-out 20% test set. Random Forest wins every target, chosen by cross-validation on training data. Accuracy is moderate: R² 0.68 for resonant frequency and 0.44 for S11 at 28 GHz, and I report that honestly." |
| 1:20-1:45 | Design & Optimize | Dataset selector → METU-SIMPL; **Predict performance** | "The selector shows only datasets whose pipeline completed. Zenodo is listed as unavailable because it could not be downloaded. The form matches this dataset's 11 parameters with units and valid ranges." |
| 1:45-2:20 | Design & Optimize | Minimize S11 at 28 GHz → **Run optimization** | "Differential Evolution varies patch length, width and feed position, holding materials and thicknesses fixed. It moves the predicted resonance toward 28 GHz. The diagnostics show the nearest design that was actually simulated: a sanity reference. The candidate itself is an ML prediction until it is simulated in HFSS." |
| 2:20-2:45 | Results → Across datasets / Explainability | Comparison chart, permutation importance | "Each metric is labelled with its dataset. The datasets are not merged, because their inputs differ. Importance is physically sensible: patch length and permittivity set the resonance, and substrate height sets the bandwidth." |
| 2:45-3:00 | Results → Data & export | CSV downloads, PDF | "Everything is stored in SQLite and exportable, with seeds and checksums, so each result can be reproduced." |
