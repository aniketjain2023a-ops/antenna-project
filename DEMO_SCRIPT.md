# Three-minute demonstration script

**Before the demo:** activate the virtual environment, run `python scripts/prepare_public_datasets.py`
once, then `./start.sh`. Open http://127.0.0.1:8000. The three pages are in the top navigation bar.

| Time | Page | Do | Say |
|---|---|---|---|
| 0:00-0:25 | Home | Point at the workflow strip and the dataset cards | "MetaOpt-AI is a React and FastAPI website around a Python ML pipeline. The workflow is dataset, training, prediction, optimization and comparison. METU-SIMPL is 5,000 microstrip patch designs simulated in HFSS from 23 to 33 GHz, from a published paper. Zenodo is shown as not available because it could not be downloaded, so it is not used anywhere." |
| 0:25-0:55 | Home | Data card (missing values, provenance), then the training table | "Each dataset has its source and checksum. From the simulated S11 curves I derive S11 at 28 GHz, the minimum S11, the resonant frequency and the bandwidth. Resonances outside the band are left missing, not invented. Random Forest beats Gradient Boosting and the Dummy baseline on every target. Accuracy is moderate: R² 0.68 for resonant frequency and 0.44 for S11 at 28 GHz." |
| 0:55-1:30 | Design & Optimize | Select METU-SIMPL; change Patch length to 3.0, then 1.0, then back with Median | "The form is generated from the dataset's own 11 inputs with units and valid ranges. The prediction updates as I type. Outside the training range it says extrapolation; invalid input is caught before it reaches the model." |
| 1:30-2:10 | Design & Optimize | Minimize S11 at 28 GHz → Run optimization | "Optimization is a separate step: Differential Evolution varies patch length, width and feed position, holding the materials fixed, within the training bounds. The candidate moves the predicted resonance toward 28 GHz. Green marks predicted improvements. The nearest design that was actually simulated is shown as a reference. The candidate itself has not been simulated, and the page says so." |
| 2:10-2:45 | Results | Comparison chart, then diagnostics for a METU model | "Every metric is labelled with its dataset. The datasets are kept separate because their inputs differ. Permutation importance is physically sensible: patch length and permittivity set the resonance, substrate height sets the bandwidth." |
| 2:45-3:00 | Results | Optimization history; click a CSV download | "Every run is stored in SQLite and downloadable as CSV." |
