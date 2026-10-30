# What is missing from the report — `MetaOpt-AI_Project_Report_Aniket_Jain.docx`

The report was compared against the project's code, its database and `results/metu_simpl/`.
Every number given below to add was read from those files on 30 Oct 2026; none is estimated.

Sections A and B must be fixed before submission. The rest makes the report more complete.

---

## A. Placeholders still unfilled (highlighted in the .docx)

| Where | What to put |
|---|---|
| Title page: "[Register Number]" | **23BEC0470** |
| Title page, Declaration, Certificate, Acknowledgements: "[Prof. / Dr. Guide Name]" | your guide's name (4 places) |
| Title page: "[Month, Year]" | the submission month and year |
| Certificate: "Aniket Jain & [Reg. No.]" | **Aniket Jain (23BEC0470)**. The "&" reads as if there were two students |
| Certificate: "[DD.MM.YYYY to DD.MM.YYYY]" | project start and end dates |
| Declaration: "Date :" | the signing date |
| Ch. 3, 10, 11: "[Draft section — … please review]" | read and edit these sections, then **delete the highlighted marker** |
| Table 7.1: Timeline column | dates for phases 1–5 and 7 (only phase 6, 22 Oct 2026, is filled) |
| Table 11.1: hardware cost and Total | e.g. "0 (existing laptop)" and a total of ₹0, or the actual figure |
| Ch. 12: venue / status | name the venue and status, or write "No publication or patent has been filed." |
| Contents, List of Figures, List of Tables: page numbers | regenerate after the final edit (Word → References → Update Table) |

---

## B. Errors and inconsistencies to correct

1. **§4.1.6 claims a model was used that never was.** The report says Histogram-based Gradient Boosting "is additionally used on the synthetic data". It is available in the code but was **never trained**: no HGB experiment exists in the database. §4.3 also lists it as "compared". Either remove it from both places, or say "available but not evaluated".
2. **Run numbers differ between the report and `REPORT_CONTEXT.md`.**
   - The report uses optimization **run #2** (L 2.3884, W 1.8487, feed 0.4325, S11@28 −6.562 dB, f_res 29.45 GHz, 1.34 s, distance 0.442). The context file used **run #16** (L 2.4417, W 2.6495, S11@28 −6.446 dB, f_res 27.887 GHz).
   - I re-checked the run #2 candidate with the trained models: they give S11@28 −6.567 dB, min S11 −15.79 dB, f_res 29.43 GHz, BW10 1,556 MHz. The nearest simulated design is row 1622 at distance 0.442. So the report's run #2 numbers are consistent and appear to come from a genuine run on your Mac.
   - **Use run #2 throughout** and make sure Fig. 8.3 is a screenshot of run #2. The cloud screenshot `fig04_design_optimization_result.png` shows run #16 (2.4417 / 2.6495 mm), so it does **not** match Table 8.4.
   - Likewise, Table 8.3 (median design: −0.547 dB, f_res 24.91 GHz ± 2.98) differs slightly from the context file (−0.549 dB, 24.807 GHz ± 2.91). Keep your Mac values, but say once that they come from your own machine's run.
3. **Introduction, ¶2:** "A common remedy is surrogate modelling … \[1\]". Reference \[1\] (Saçın & Durgun) is a dataset/NN paper, not a source for surrogate modelling in general. Cite Koziel & Ogurtsov (see §F) for this sentence.
4. **§4.2 cites FIPS 180-4** for SHA-256 but it is not in the reference list. Add it (see §F).
5. **§4.1.9 says Latin-hypercube initialization** but does not cite it. Add McKay et al. (1979) (see §F).
6. **Table 8.1 note:** "MAE units follow the RMSE column" is easy to miss. Put units in the MAE cells instead (1.110 dB, 3.862 dB, 1.143 GHz, 248.8 MHz).
7. **Eq. (1.1)** is the simplified cavity-model form. Add "(neglecting the fringing length extension ΔL \[2\])" so an examiner does not mark it as incomplete.
8. **Certificate:** check that the pronoun ("by him") is the one you want there.

---

## C. Content missing from the chapters (ready to paste)

### C1. Model settings — not stated anywhere (add to §4.1.6)

The METU-SIMPL models were trained with default settings ("default" variant); no hyperparameter search was run.

| Model | Settings used |
|---|---|
| Dummy | predicts the training-set mean |
| Random Forest | 150 trees, other parameters at scikit-learn defaults, random_state 42 |
| Gradient Boosting | scikit-learn defaults: 100 stages, learning rate 0.1, max depth 3, random_state 42 |
| Gaussian Process (synthetic only) | ARD RBF kernel × constant + white noise, `normalize_y=True`, 2 optimizer restarts |
| Extra Trees (synthetic only) | 150 trees |

Suggested sentence: "No hyperparameter tuning was performed; tuning on the training split is supported by the software and is left for future work."

### C2. Differential Evolution settings — λ, F and CR are used in equations but never given (add to §4.1.9)

| Setting | Value |
|---|---|
| Population | 15 × number of free parameters (45 for 3 free parameters) |
| Differential weight *F* | dithered, drawn uniformly from \[0.5, 1.0\] each generation |
| Crossover rate *CR* | 0.7 |
| Generations | budget ÷ population − 1 (32 for budget 1,500 and 3 parameters, so 45 × 33 = **1,485 evaluations**, matching the run) |
| Tolerance | 10⁻⁶ (relative); the synthetic run stopped early at 810 evaluations for this reason |
| Initialization | Latin hypercube; no gradient polishing |
| Penalty weight λ in Eq. (4.5) | **100** |
| Strategy | SciPy default `best1bin` |

Eq. (4.4) shows the `rand/1` mutation, but SciPy's default is `best1bin`: **v** = **x**_best + *F*(**x**_r1 − **x**_r2). Either change Eq. (4.4) to this form or state the strategy explicitly.

### C3. Target statistics — the reader never sees what the targets look like (new table in §4.1.2)

| Target | n | Min | Max | Mean | Std | Median |
|---|---|---|---|---|---|---|
| S11 at 28 GHz (dB) | 5,000 | −24.158 | −0.039 | −1.858 | 2.731 | −0.838 |
| Minimum S11 (dB) | 5,000 | −54.171 | −0.054 | −7.721 | 8.019 | −5.022 |
| f_res (GHz) | 2,866 | 23.05 | 32.95 | 28.629 | 2.727 | 29.000 |
| BW10 (MHz) | 1,201 | 0 | 2,900 | 837.9 | 486.6 | 800 |

Useful facts to add with this table:
- Only **129 of 5,000** designs (2.6%) have S11 ≤ −10 dB at 28 GHz.
- **1,392** designs (27.8%) reach −10 dB anywhere in the band.

This explains why S11@28 is hard to learn: the data are dominated by poorly matched designs.

### C4. Errors of the other models — Table 8.1 gives MAE/RMSE for RF only

| Target | GB MAE | GB RMSE | Dummy MAE | Dummy RMSE | GB CV R² |
|---|---|---|---|---|---|
| S11@28 (dB) | 1.307 | 2.441 | 1.741 | 2.834 | 0.289 ± 0.050 |
| S11_min (dB) | 4.672 | 6.865 | 6.247 | 8.364 | 0.367 ± 0.028 |
| f_res (GHz) | 1.501 | 1.904 | 2.367 | 2.747 | 0.467 ± 0.023 |
| BW10 (MHz) | 255.3 | 333.7 | 387.5 | 489.2 | 0.519 ± 0.063 |

Suggested sentence: "Random Forest reduces RMSE relative to the Dummy baseline by 25% (S11@28), 28% (S11_min), 44% (f_res) and 33% (BW10)."

### C5. Training time — not reported (add a column to Table 8.1)

Random Forest: 19.4 s (S11@28), 17.7 s (S11_min), 8.9 s (f_res), 3.6 s (BW10).
Gradient Boosting: 5.7, 5.6, 3.1 and 1.3 s.
Dummy: < 0.02 s.

These are single-machine timings and include the 5-fold CV. Compare them with the minutes to hours an HFSS run takes, which is the motivation of the whole project.

### C6. Uncertainty coverage — an important result that is missing (add to §4.1.7 / §8.1)

The fraction of test points whose true value lies within the prediction ± 1.96 × tree spread:

| Target | Coverage |
|---|---|
| S11@28 | 96.9% |
| S11_min | 95.4% |
| f_res | 98.3% |
| BW10 | 96.7% |

Suggested wording: "On the test set, the ±1.96σ band contained 95–98% of true values, so the tree spread is on average not over-confident. This is a marginal check, not a calibration guarantee, and the spread is still reported as an approximate ± value."

### C7. Full feature importance — Table 8.2 omits S11_min and shows only the top two features

Random Forest, permutation importance (mean ± std over 10 repeats), top four features:

| Target | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| f_res | Patch length 1.170 ± 0.052 | Substrate εr 0.837 ± 0.040 | Patch width 0.276 ± 0.022 | Substrate height 0.087 ± 0.009 |
| BW10 | Substrate height 0.598 ± 0.090 | Patch length 0.343 ± 0.024 | Feed position 0.108 ± 0.016 | Substrate εr 0.086 ± 0.021 |
| S11@28 | Patch length 0.417 ± 0.021 | Substrate εr 0.362 ± 0.015 | Patch width 0.269 ± 0.026 | Substrate height 0.219 ± 0.029 |
| S11_min | Patch length 0.462 ± 0.027 | Substrate εr 0.433 ± 0.031 | Patch width 0.356 ± 0.027 | Feed position 0.179 ± 0.019 |

Impurity importance (§4.1.8 says it is "also reported", but it is not), Random Forest, top three:

| Target | Top three |
|---|---|
| f_res | patch length 0.352, substrate εr 0.259, patch width 0.122 |
| BW10 | substrate height 0.390, patch length 0.168, feed position 0.091 |
| S11@28 | substrate εr 0.197, patch length 0.144, patch width 0.141 |
| S11_min | substrate εr 0.184, patch length 0.183, patch width 0.173 |

Points for the discussion:
- The two methods agree on the dominant features.
- Feed position matters for S11_min (matching), which is physically expected because the probe position sets the input impedance.
- Copper thickness and solder-resist parameters are near zero for every target.

### C8. Synthetic results — RMSE missing for the non-selected models (§8.5)

| Model | Gain R² | Gain RMSE (dBi) | S11 R² | S11 RMSE (dB) |
|---|---|---|---|---|
| Gaussian Process | 0.995 | 0.156 | 0.959 | 0.800 |
| Extra Trees | 0.990 | 0.229 | 0.951 | 0.877 |
| Random Forest | 0.983 | 0.303 | 0.907 | 1.208 |
| Gradient Boosting | 0.983 | 0.304 | 0.844 | 1.566 |
| Dummy | −0.004 | 2.309 | −0.012 | 3.985 |

Split: 400 train / 100 test.

Also add the synthetic data's formula, input ranges and noise level (from `datasets/` and the synthetic dataset documentation). The report says "a documented analytic formula" but never shows it.

### C9. Preprocessing details missing from §4.1.2

- The sweep has **201 frequency points** (23–33 GHz in 0.05 GHz steps). The S11 curves are stored in `data/processed/metu_simpl/s11_db_curves.npz`.
- Exact censoring rules (from `preprocessing_summary.json`):
  - f_res is missing where the minimum lies on a band edge.
  - BW10 is missing where min S11 > −10 dB **or** the −10 dB region reaches a band edge.
- Pipeline counts: 0 rows removed for missing inputs and 0 duplicates removed.
- Source commit (full hash): `7113818a1da9aead6a84852111deec5f075af8eb`. Downloaded 19 Oct 2026.

### C10. Database design — mentioned but not described (add to Ch. 5)

Tables: `datasets`, `antenna_designs` (5,500 rows), `model_experiments` (38), `prediction_records` (4), `optimization_runs` (18), `app_settings`, `schema_version`.

Each optimization run stores:
- seed, budget and evaluations used;
- objective, bounds and constraints;
- baseline and candidate parameters, predictions and uncertainty;
- feasibility, run time and the dataset checksum.

A small ER diagram or table listing would complete Ch. 5.

### C11. End-to-end tests — named but not described (§8.6)

The five browser tests are:
1. The Home page lists the datasets honestly, including the unavailable source.
2. Models can be trained from the website.
3. Live prediction works, with validation and save.
4. Optimization runs on both datasets.
5. The Results page loads and CSV download works.

### C12. UI design rationale — absent

One paragraph on the interface design:
- Three-page structure.
- Inline validation and live prediction.
- Provenance badges.
- "Not physically validated" warnings.
- Accessibility: reduced-motion, reduced-transparency and increased-contrast support; light and dark themes.

### C13. Literature review is too short (§1.1)

It covers only patch theory and one paper. A B.Tech report normally reviews **8–15** works. Add short subsections, citing only papers you have actually opened:
- **ML for antenna design:** a review paper (search "machine learning antenna design review"; REFERENCES.md Part B lists two candidates to verify).
- **Surrogate-based optimization of antennas** (Koziel & Ogurtsov).
- **Global optimization:** DE (Storn & Price) vs GA/PSO; efficient global optimization (Jones et al.).
- **Explainability** (Fisher et al.; Molnar).
- **Uncertainty** (Meinshausen; Kuleshov et al.).
- **Metasurfaces** (Holloway et al.), since the original brief mentioned them.

Finish with a short comparison table: work · data · model · optimizer · explains? · uncertainty?

### C14. Research gap wording (§1.2)

"It does not couple the surrogate to a design optimizer…" is a claim about \[1\]. Make sure you have read \[1\] and that this is true, or soften it to "The work in \[1\] focuses on…".

---

## D. Figures missing

All of these already exist in the repository and can be inserted directly.

| Suggested figure | File |
|---|---|
| Patch antenna geometry (probe feed, solder-resist layer) | not in the repo. Draw a labelled schematic yourself, or reproduce the paper's figure with permission and attribution |
| Example S11 curves: one matched, one censored (resonance outside the band) | not in the repo. Plot from `data/processed/metu_simpl/s11_db_curves.npz`; this shows censoring better than any text |
| Actual vs predicted, four targets | `results/metu_simpl/figures/actual_vs_predicted_{S11_28GHz_dB,S11_min_dB,f_res_GHz,BW10_MHz}.png` |
| Residuals, four targets | `results/metu_simpl/figures/residuals_*.png` |
| Permutation importance, four targets | `results/metu_simpl/figures/permutation_importance_*.png` |
| Home page | `docs/report/figures/fig01_home_hero.png`, `fig02_home_dataset_training.png` |
| Design page: form and live prediction | `docs/report/figures/fig03_design_prediction.png` |
| Results page | `docs/report/figures/fig05_results_comparison.png`, `fig06_results_diagnostics.png` |
| DE convergence curve for run #2 | screenshot of the convergence chart on the Design & Optimize page after running it |
| Flowchart of the method (data → targets → split → CV → selection → prediction → DE → nearest-simulated check) | draw it; Fig. 5.1 shows software layers, not the method |

At minimum, add the actual-vs-predicted plot for f_res and S11@28 and one importance chart. Results chapters are expected to show plots, not only tables.

---

## E. Tables missing

- Target statistics (C3)
- DE settings (C2) and model settings (C1)
- Full error metrics for every model (C4), with training time (C5)
- Uncertainty coverage (C6)
- Literature comparison (C13)
- Database tables (C10)

---

## F. References to add or complete

The following are already listed in `REFERENCES.md` Part A. **Open each DOI to confirm it before citing.**

- S. Koziel and S. Ogurtsov, *Antenna Design by Simulation-Driven Optimization*. Cham, Switzerland: Springer, 2014. *(for the surrogate-modelling sentence in Ch. 1)*
- M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code," *Technometrics*, vol. 21, no. 2, pp. 239–245, 1979. *(Latin hypercube)*
- National Institute of Standards and Technology, "Secure Hash Standard (SHS)," FIPS PUB 180-4, 2015. *(SHA-256, §4.2)*
- A. Fisher, C. Rudin and F. Dominici, "All models are wrong, but many are useful…," *JMLR*, vol. 20, no. 177, pp. 1–81, 2019. *(permutation importance)*
- D. M. Pozar, *Microwave Engineering*, 4th ed. Hoboken, NJ, USA: Wiley, 2012. *(S11 and return loss)*
- C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.
- D. R. Jones, M. Schonlau and W. J. Welch, "Efficient global optimization of expensive black-box functions," *J. Global Optim.*, vol. 13, pp. 455–492, 1998.
- N. Meinshausen, "Quantile regression forests," *JMLR*, vol. 7, pp. 983–999, 2006. *(future work: calibration)*
- C. L. Holloway *et al.*, "An overview of the theory and applications of metasurfaces…," *IEEE Antennas Propag. Mag.*, vol. 54, no. 2, pp. 10–35, 2012. *(if metasurfaces are mentioned)*
- Software cited as web resources with an access date: FastAPI, React, SQLAlchemy, Playwright, Ansys HFSS.

**DOIs missing from the existing references:** add them to the entries that have one.

| Ref. | DOI |
|---|---|
| \[3\] Storn & Price | 10.1023/A:1008202821328 |
| \[4\] Breiman | 10.1023/A:1010933404324 |
| \[5\] Friedman | 10.1214/aos/1013203451 |
| \[8\] Virtanen et al. | 10.1038/s41592-019-0686-2 |
| \[9\] Geurts et al. | 10.1007/s10994-006-6226-1 |

**Do not cite the Zenodo record.** It was never opened.

---

## G. Abbreviations and symbols missing

**Abbreviations used in the text but not listed:**

| Abbreviation | Meaning |
|---|---|
| XAI | Explainable Artificial Intelligence |
| ET | Extra Trees |
| HGB | Histogram-based Gradient Boosting |
| LHS | Latin Hypercube Sampling |
| UI | User Interface |
| JSON | JavaScript Object Notation |
| SDG | Sustainable Development Goal (already listed) |
| METU-SIMPL | Middle East Technical University – Signal and Image Processing Lab (**check the expansion on their GitHub page before using it**) |
| FIPS | Federal Information Processing Standard |
| ER | Entity-Relationship (if you add C10) |
| dB / dBi | decibel / decibel relative to isotropic |
| mmWave | millimetre wave |
| 5G | fifth-generation mobile networks |

**Symbols to add:**

| Symbol | Meaning |
|---|---|
| *CR* | crossover rate |
| σ | prediction spread (uncertainty) |
| *n* | number of test samples |
| *d* | normalized distance to the nearest simulated design |
| *W* | patch width |
| *h* | substrate height |
| ΔL | fringing length extension |
| **x**_best | best population member (if Eq. 4.4 is changed) |

---

## H. Appendices — none present (add at least A and B)

- **Appendix A — How to run:**
  ```
  git clone … ; python3 -m venv .venv ; source .venv/bin/activate ;
  pip install -r requirements.txt ; python scripts/prepare_public_datasets.py ; ./start.sh
  ```
- **Appendix B — Code excerpts:**
  - Target derivation with censoring (`datasets/public/metu_simpl.py`)
  - DE wrapper (`optimization/differential_evolution.py`)
  - Penalised objective (`optimization/objectives.py`)
- **Appendix C — Full API list:** all 11 routes, including health, preview, experiments and the four CSV exports. Table 5.1 shows 7.
- **Appendix D — Full metrics CSV** (`results/metu_simpl/metrics.csv`) and screenshots of all three pages.
- **Appendix E — Repository link:** `github.com/aniketjain2023a-ops/antenna-project`, branch `react-fastapi`.

---

## I. Formatting checks

- Figure and table numbers must match the List of Figures and List of Tables after you add the new ones.
- Equations: number them consistently, (1.1), (4.1)–(4.5), and refer to them by number.
- Use one style for units throughout ("dB", "GHz", "mm"), with a space between number and unit.
- Give every figure a source line where it comes from the application ("Screenshot of MetaOpt-AI, run #2").
