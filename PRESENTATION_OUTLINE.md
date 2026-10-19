# Presentation outline (10 slides, about 10 minutes)

**Slide 1 - Title**
MetaOpt-AI: Explainable Machine Learning for Metasurface Antenna Optimization · Aniket Jain (23BEC0470).
One-line pitch: an explainable, uncertainty-aware, reproducible ML workflow for antenna design exploration.

**Slide 2 - Problem**
Full-wave EM simulations are slow. S11 and gain depend non-linearly, and in coupled ways, on geometry.
Black-box surrogates are hard to trust. Visual: the S11 resonance valley over the (L, G) map.

**Slide 3 - Objectives and contribution**
Seven objectives (data + provenance, leakage-safe ML, prediction + uncertainty, constrained optimization, XAI, reports, tests).
Contribution = integration and evaluation. We do not claim to have invented ML-based antenna optimization.

**Slide 4 - Architecture**
Layered diagram: React website → FastAPI → services → ML / optimization → repository → SQLite.
List the 7 tables and the provenance types.

**Slide 5 - Data and honesty**
Synthetic generator equations (resonance, Q, matching, metasurface gain term, noise).
Show the SYNTHETIC labelling in the CSV, the DB, the UI banner and the PDF footer. 58/500 samples have S11 ≤ -10 dB.

**Slide 6 - ML methodology and results**
Protocol: split → CV → tuning on training data only → single test evaluation; selection by CV R².
Table: GP S11 R² 0.961 / RMSE 0.79 dB; gain R² 0.995 / RMSE 0.16 dBi; Dummy ≈ 0.
Honest tuning result: helped GB, hurt RF/ET.

**Slide 7 - Explainability and uncertainty**
Permutation importance (L ≫ G), OAT sensitivity, PDP, uncertainty map with U1-U5.
Coverage95: GP 0.94-0.95 vs tree spread 0.83 (over-confident). Association ≠ causation.

**Slide 8 - Optimization**
Three objectives, normalized trade-off, penalty constraints, three algorithms agree.
Max gain: 7.78 → 8.02 dBi (predicted) with S11 -15.5 dB. Pareto plot. The oracle check agrees with the formula.

**Slide 9 - Limitations**
Synthetic data; no EM solver; two parameters; single frequency (no bandwidth); approximate uncertainty;
weighted sum reaches only convex front regions; benefit of uncertainty sampling not demonstrated.

**Slide 10 - Conclusion and future work**
A working, tested (50 tests), reproducible platform. Next: real openEMS/CST/measured data, frequency sweeps
for bandwidth, an active-learning study, NSGA-II, conformal calibration. Live demo / questions.
