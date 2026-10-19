"""Page 9 - Project information, methodology and limitations."""

from __future__ import annotations

import streamlit as st

import config
from datasets import synthetic
from reporting.report_generator import LIMITATIONS


def render() -> None:
    st.title("Project Information & Limitations")
    st.markdown(f"""
**{config.PROJECT_NAME} - {config.PROJECT_TITLE}**
Student: **{config.STUDENT_NAME}** | Registration No.: **{config.REGISTRATION_NUMBER}** | Version {config.APP_VERSION}

### Engineering problem
Metasurface-loaded antennas have geometry parameters (patch/metasurface length, unit-cell gap, ...) whose
effect on matching (S11) and gain is coupled and non-linear. Each full-wave EM simulation is expensive, so
designers benefit from a cheap **surrogate model** that approximates the simulator from previously computed
samples, plus tools to search the design space, explain the surrogate and see where it is unreliable.

### Methodology
1. **Data with provenance** - every dataset is typed (SYNTHETIC / PUBLIC / SIMULATION / MEASUREMENT), checksummed
   (SHA-256) and stored in SQLite with its design rows.
2. **Validation** - type detection, missing values (never imputed for targets), duplicates, IQR outliers (flagged only).
3. **Learning** - Dummy baseline, Random Forest, Extra Trees, Gradient Boosting, Histogram GB, Gaussian Process;
   leakage-safe pipelines, seeded split, K-fold CV and RandomizedSearchCV on training data only.
4. **Prediction** - with extrapolation detection and approximate uncertainty (GP std / tree spread).
5. **Optimization** - Differential Evolution, LHS multi-start Nelder-Mead and Dual Annealing calling the surrogate;
   penalised constraints; range-normalised trade-off objective; optional risk-averse (mean -/+ kappa sigma) mode.
6. **Explainability** - impurity & permutation importance, one-at-a-time sensitivity, partial dependence,
   uncertainty maps with suggested regions for new simulations.
7. **Reproducibility** - seeds, checksums, hyper-parameters, bounds, objectives and results logged; PDF/CSV reports.

### Synthetic demonstration data (assumptions)
`datasets/synthetic.py`: half-wave resonance with assumed effective permittivity {synthetic.EPS_EFF},
design frequency {synthetic.F0_GHZ} GHz, loaded Q = {synthetic.Q_LOADED}, gap-dependent matching
(best near G = 0.8 mm), metasurface gain term (peak near G = 1.3 mm), mismatch loss, fixed
{synthetic.FIXED_LOSS_DB} dB loss and Gaussian noise (S11 sigma {synthetic.S11_NOISE_DB} dB, gain sigma
{synthetic.GAIN_NOISE_DB} dB). L in {synthetic.L_RANGE_MM} mm, G in {synthetic.G_RANGE_MM} mm.
**This is not an EM simulation and not a measurement.**

### Intended contribution
The contribution is the **implementation, integration and evaluation** of an explainable, uncertainty-aware,
provenance-tracking workflow. It does *not* claim to have invented ML-based antenna optimization, which is an
established research area.
""")
    st.subheader("Scientific limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    st.subheader("Association is not causation")
    st.markdown("Importance and dependence plots describe what the model relies on. They do not prove that changing a "
                "parameter *causes* a performance change in a physical antenna; only controlled EM simulation or "
                "measurement can establish that.")
    st.subheader("Documentation")
    st.markdown("See `README.md`, `PROJECT_REPORT.md`, `RESULTS_INTERPRETATION.md`, `REPRODUCIBILITY.md`, "
                "`VIVA_QUESTIONS.md`, `DEMO_SCRIPT.md`, `PRESENTATION_OUTLINE.md` and `REFERENCES.md` in the project folder.")
