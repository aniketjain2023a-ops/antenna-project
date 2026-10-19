# Dataset inspection and comparison report

Prepared 2026-10-19 by running `scripts/prepare_public_datasets.py`. Every number below was produced
by that run; nothing is estimated.

## 1. Download status

| Source | URL | Download | Notes |
|---|---|---|---|
| **METU-SIMPL Microstrip-Patch** | https://github.com/METU-SIMPL/Microstrip-Patch | **Succeeded** (git clone, commit `7113818a1da9aead6a84852111deec5f075af8eb`, 2026-10-19 17:57 UTC) | 5 files copied unchanged to `data/raw/metu_simpl/`; SHA-256 of each in `MANIFEST.json`, re-verified before every preprocessing run |
| **Zenodo record 15866865** | https://zenodo.org/records/15866865 | **FAILED** | `zenodo.org` was refused by the build environment's network policy (HTTP 403 on CONNECT); a second route (web fetch) could not resolve the host. **No file, schema, unit or license information could be inspected.** Recorded in `data/raw/zenodo/DOWNLOAD_STATUS.json` |

The Zenodo dataset is therefore **not used anywhere** in the application, models or results, and it
is not offered in the dataset selector. `datasets/public/zenodo.py` can download the record (via the
Zenodo REST API, with checksum verification) once the host is reachable. It deliberately has **no
preprocessing pipeline** until the files have been inspected. Run
`python scripts/prepare_public_datasets.py zenodo` where zenodo.org is accessible, then inspect the
files before writing a pipeline.

## 2. METU-SIMPL: verified facts

| Item | Finding | How verified |
|---|---|---|
| Files | `README.md`, `input_parameters.xlsx` (5,001 × 11 incl. header), `S11_re.xlsx` and `S11_im.xlsx` (5,001 × 201 incl. index header), `antenna_model.aedt` (HFSS project) | opened and shape-checked |
| Antenna type | Probe-fed rectangular microstrip patch with solder-resist layer, pad and ground antipad | HFSS project geometry (`patch`, `substrate`, `solder_resist`, `probe`, `pad`) |
| Data kind | **Full-wave EM simulation** (Ansys HFSS, terminal S-parameters), **not measurement** | README "Generated using full-wave electromagnetic simulations using Ansys HFSS"; `.aedt` solution type `HFSS Terminal Network` |
| Frequency | 23–33 GHz, 0.05 GHz step, 201 points (columns are indexed 0–200 in the xlsx) | `.aedt` sweep `RangeStart='23GHz' RangeEnd='33GHz' RangeStep='0.05GHz'`; 201 = column count |
| S11 representation | Complex S11: separate real and imaginary parts; max \|S11\| = 0.999 (passive, plausible) | computed |
| Units | Lengths in mm; `c_probe` dimensionless (probe offset = `c_probe × l_patch`); permittivities unitless | `.aedt` `VariableProp` definitions (e.g. `l_patch='3.66mm'`, `d_probe='c_probe*l_patch'`) |
| Sampling | All 11 inputs uniform over their ranges (means at mid-range); 0 duplicate rows, 0 missing values | computed |
| License | **No license file** in the repository. README asks to cite Saçın & Durgun (2025) and acknowledge the authors | file listing of the downloaded commit |
| Citation | E. S. Saçın, A. C. Durgun, "Microstrip Patch Antenna Modeling Using Neural Networks with Knowledge-Based Regularization," *Neural Computing and Applications* 37, 3827–3837 (2025), doi:10.1007/s00521-024-10860-5 | repository README |

**Use decision.** Without a license, the data is used only locally, for non-commercial academic
analysis with citation. The raw and processed files are **not committed** to this repository
(`.gitignore`); `scripts/prepare_public_datasets.py` downloads them from the original source.

### Inputs (11) after renaming

| Original column | Name in MetaOpt-AI | Unit | Range in data |
|---|---|---|---|
| length of patch | patch_length_mm | mm | 1.80 – 5.20 |
| width of patch | patch_width_mm | mm | 1.80 – 5.20 |
| height of patch | patch_thickness_mm | mm | 0.010 – 0.040 |
| height of substrate | substrate_height_mm | mm | 0.10 – 0.80 |
| height of solder resist layer | solder_resist_height_mm | mm | 0.020 – 0.100 |
| radius of the probe | probe_radius_mm | mm | 0.015 – 0.050 |
| c_pad | pad_clearance_mm | mm | 0.000 – 0.025 |
| c_antipad | antipad_clearance_mm | mm | 0.025 – 0.100 |
| c_probe | feed_position_ratio | – | 0.050 – 0.450 |
| dielectric constant of substrate | substrate_eps_r | – | 2.0 – 5.0 |
| dielectric constant of solder resist layer | solder_resist_eps_r | – | 2.0 – 5.0 |

### Targets derived from the simulated S11 curve

| Target | Definition | Rows with a value |
|---|---|---|
| S11_28GHz_dB | 20·log10\|S11\| at exactly 28.00 GHz | 5,000 |
| S11_min_dB | deepest \|S11\| within 23–33 GHz | 5,000 |
| f_res_GHz | frequency of that minimum, **only if strictly inside the band** | 2,866 (2,134 censored: minimum on a band edge, so the resonance lies outside 23–33 GHz) |
| BW10_MHz | width of the contiguous \|S11\| ≤ −10 dB region around the minimum, **only if fully inside the band** | 1,201 |

Censored values are left missing, never filled in. Each model is trained only on rows where its
target is defined.

Sample counts: 5,000 rows before preprocessing → 5,000 after (no duplicates or missing inputs to
remove).

## 3. Compatibility assessment

| Check | METU-SIMPL | Zenodo 15866865 | Synthetic demo (bundled) |
|---|---|---|---|
| Antenna type | probe-fed microstrip patch | **unknown** (not inspected) | abstract patch/metasurface formula |
| Inputs | 11 physical parameters (mm, ratio, εr) | unknown | L_mm, G_mm |
| Targets / S11 | complex S11, 23–33 GHz sweep | unknown | S11 at one assumed 10 GHz point; gain |
| Method | HFSS full-wave simulation | unknown | analytical formula + noise |
| Overlap / duplicates | none within the dataset | cannot be checked | not applicable |

**Decision: Approach B (separate pipelines and models).**

- Zenodo could not be inspected, so it cannot be combined with anything.
- METU-SIMPL and the synthetic demo share no input definitions (11 HFSS parameters vs. two abstract
  variables). Their S11 targets refer to different frequencies and methods. Merging them would mix
  incompatible physical quantities.

**Cross-dataset evaluation (Approach C): not possible.** No two available datasets share input and
target definitions, so no model was evaluated on another dataset's data, and no generalization claim
is made.

## 4. Models and results (METU-SIMPL, held-out test set, seed 42, 80/20 split, 5-fold CV on training data)

| Target | Model | Test R² | Test MAE | Test RMSE | CV R² (mean ± sd) | Train / test rows |
|---|---|---|---|---|---|---|
| S11 at 28 GHz (dB) | **Random Forest** | **0.441** | 1.110 | 2.117 | 0.392 ± 0.023 | 4,000 / 1,000 |
| | Gradient Boosting | 0.258 | 1.307 | 2.441 | 0.289 ± 0.050 | |
| | Dummy (mean) | −0.000 | 1.741 | 2.834 | −0.001 | |
| Min S11 23–33 GHz (dB) | **Random Forest** | **0.487** | 3.862 | 5.992 | 0.484 ± 0.033 | 4,000 / 1,000 |
| | Gradient Boosting | 0.326 | 4.672 | 6.865 | 0.367 ± 0.028 | |
| | Dummy (mean) | −0.000 | 6.247 | 8.364 | −0.000 | |
| Resonant frequency (GHz) | **Random Forest** | **0.679** | 1.143 | 1.550 | 0.650 ± 0.034 | 2,292 / 574 |
| | Gradient Boosting | 0.515 | 1.501 | 1.904 | 0.467 ± 0.023 | |
| | Dummy (mean) | −0.009 | 2.367 | 2.747 | −0.001 | |
| −10 dB bandwidth (MHz) | **Random Forest** | **0.547** | 248.8 | 329.3 | 0.543 ± 0.047 | 960 / 241 |
| | Gradient Boosting | 0.534 | 255.3 | 333.7 | 0.519 ± 0.063 | |
| | Dummy (mean) | −0.001 | 387.5 | 489.2 | −0.007 | |

**Selected models.** Random Forest is selected for every target, using cross-validated R² on the
training split only. Both learned models clearly beat the Dummy baseline, but accuracy is
**moderate** (R² 0.44–0.68). S11 at a single frequency is hard to learn from 11 raw parameters with
these default models; the associated paper reports that knowledge-based regularization improves
accuracy. That technique is not reproduced here.

**Permutation importance** (drop in test R²) is physically sensible:

- **Resonant frequency:** dominated by patch length (1.17) and substrate εr (0.84), consistent with
  the half-wavelength relation f ≈ c / (2L√εeff).
- **Bandwidth:** dominated by substrate height (0.60), consistent with thicker substrates giving
  wider bandwidth.

Like all permutation importances, these show what the models rely on, not proof of causation.

Files: `results/metu_simpl/metrics.csv`, `feature_importance.csv`, `selected_models.json`,
`figures/*.png` (actual-vs-predicted, residuals, model comparison, permutation importance). Test
predictions are written locally to `results/metu_simpl/test_predictions.csv`; they are not committed
because they contain the dataset's values.

## 5. Optimization dataset

**METU-SIMPL** supports S11 optimization because its target definition and frequency are verified:

- **Objective:** minimize predicted |S11| at 28.00 GHz (inside the 23–33 GHz sweep).
- **Algorithm:** SciPy Differential Evolution only.
- **Varied parameters (default):** patch length, patch width and feed position. All others (layer
  thicknesses, probe and clearances, permittivities) are held at the baseline, because they are
  usually fixed by the fabrication process.
- **Bounds:** limited to the training range; extrapolating bounds are rejected.
- **Optional:** one constraint on another predicted target.
- **Labelling:** the result is shown as an ML-selected candidate. The page also shows the nearest
  design that was actually simulated as a sanity reference. **No candidate has been simulated in HFSS
  or measured.**
