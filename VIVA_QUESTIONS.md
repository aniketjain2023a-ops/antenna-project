# Viva questions and concise answers (40)

## Antenna and metasurface fundamentals
1. **What is S11?** The input reflection coefficient: the ratio of reflected to incident wave at port 1. In dB it is 20·log10|Γ| (= 10·log10|Γ|²). More negative means less power is reflected.
2. **Why is S11 ≤ -10 dB a common target?** At -10 dB, |Γ|² = 0.1, so about 90 % of the incident power is accepted by the antenna (VSWR ≈ 1.92). It is a convention, not a law of physics.
3. **What is gain (dBi)?** Gain is directivity times radiation efficiency, relative to an isotropic radiator. *Realized* gain also includes mismatch loss, 10·log10(1-|Γ|²).
4. **Does a deeper S11 mean better gain?** No. S11 describes matching only. In our synthetic data the best-matched design (S11 ≈ -22.8 dB predicted) has about 0.7 dB less gain than the gain-optimal design (S11 ≈ -15.5 dB).
5. **What is bandwidth here, and why isn't it modelled?** Typically the frequency range where S11 ≤ -10 dB. Our data has S11 at only one frequency, so bandwidth cannot be computed. The app never invents it.
6. **What sets a patch's resonant frequency?** Roughly f_r ≈ c / (2·L_eff·√εeff). Fringing fields make L_eff longer than the physical length L.
7. **What is a metasurface?** A thin 2-D array of sub-wavelength scatterers that controls the amplitude and phase of reflection and transmission. It is the 2-D counterpart of a metamaterial.
8. **What does the gap G represent in the model?** The spacing between metasurface elements, which changes capacitive loading. In the synthetic formula it shifts the effective length, the matching quality and the metasurface gain term.
9. **What is loaded Q and its link to bandwidth?** Stored energy over dissipated plus radiated power per cycle. Fractional bandwidth is roughly proportional to 1/Q. We assumed Q = 5.
10. **Is your dataset real?** No. It is SYNTHETIC: a documented physics-inspired formula plus noise. It is labelled that way in the CSV, in the database, in the UI and in the reports.

## Machine learning
11. **What is regression?** Learning a mapping from inputs to continuous outputs, here (L, G) → (S11, gain).
12. **Why have a Dummy baseline?** It predicts the training mean (R² ≈ 0). Any model worth using must clearly beat it.
13. **How does a Random Forest work?** It averages many decision trees, each trained on a bootstrap sample with random feature selection at each split. Averaging reduces variance.
14. **Extra Trees vs Random Forest?** Extra Trees choose split thresholds at random and (by default) use the whole sample. This adds randomness, often lowers variance, and is faster.
15. **How does Gradient Boosting differ?** It builds shallow trees one after another, each fitting the residuals of the current ensemble, scaled by a learning rate. It reduces bias step by step.
16. **What is a Gaussian Process?** A distribution over functions defined by a kernel. Conditioning on data gives a predictive mean *and* variance. Cost is O(n³), so it suits small or medium datasets.
17. **What do the ARD kernel and WhiteKernel do?** ARD learns a separate length-scale for each input (how quickly the output varies with that input). WhiteKernel models observation noise.
18. **Define R², MAE, RMSE.** R² = 1 - SS_res/SS_tot. MAE is the mean absolute error. RMSE is the square root of the mean squared error. MAE and RMSE are in the target's unit (dB or dBi).
19. **What is overfitting, and how do you detect it?** Fitting noise instead of structure. You detect it when training performance is much better than CV or test performance. We compare CV R² with test R².
20. **What is data leakage, and how did you prevent it?** Information from the evaluation data influencing training. We split first, keep scaling inside the Pipeline, run CV and tuning on training data only, and select models by CV R², never by test R².
21. **Why cross-validation?** One split gives a noisy estimate. K-fold CV averages over K splits and also gives a spread (std).
22. **What did tuning achieve?** It is measured, not assumed. Tuning helped Gradient Boosting (up to 26 % lower RMSE) but made Random Forest and Extra Trees slightly worse, and changed little for the GP.
23. **Why RandomizedSearchCV instead of GridSearchCV?** It samples a fixed number of configurations, which keeps the cost predictable on a laptop.
24. **What happens with small datasets?** Fewer than 20 rows is refused. Below 50 rows CV is disabled. Below 150 rows the app warns that estimates are unstable.

## Explainability and uncertainty
25. **Impurity vs permutation importance?** Impurity (MDI) is computed from training splits and can be biased. Permutation importance measures the drop in test R² when one feature is shuffled. It is model-agnostic and evaluated on held-out data.
26. **What is a partial dependence plot?** The average model prediction as one feature varies while the others keep their observed values. It can hide interactions.
27. **Association vs causation?** Importance shows what the *model* relies on. It does not prove that changing a parameter physically causes the effect. Only controlled EM simulation or measurement can show that.
28. **How do you estimate uncertainty?** For the GP, the predictive standard deviation. For RF and Extra Trees, the spread across individual trees. Both are approximate.
29. **Are your intervals calibrated?** Not claimed. We report empirical coverage of ±1.96σ on the test set: GP 0.94-0.95 (close to 0.95), tree spread for S11 0.83-0.84 (over-confident).
30. **What is extrapolation, and how is it handled?** Predicting outside the training range. The app flags it in red, and the optimizer rejects bounds outside the training range unless the user explicitly allows them.
31. **What is the uncertainty map for?** It shows where the model is least certain, which is where extra EM simulations could be most informative. We do not claim that this sampling improves accuracy, because we have not tested it.

## Optimization
32. **How does Differential Evolution work?** It keeps a population of candidates. A mutant vector is built from scaled differences of other members, crossed over with a target, and kept if it is better. It needs no gradients.
33. **Why derivative-free optimizers?** Tree surrogates are piecewise constant, so their gradients are zero almost everywhere.
34. **What is your second algorithm?** LHS multi-start Nelder-Mead: screen a Latin-hypercube sample, then run bounded simplex searches from the best points. Dual annealing is a third option.
35. **How are constraints handled?** With a static penalty: objective + 100 × Σ(violation / target data range). Violations are also reported separately, and each candidate is marked feasible or infeasible.
36. **How do you combine S11 (dB) with gain (dBi)?** Never directly. Each is normalized to a [0, 1] desirability over its observed range (1 = best), then a weighted sum is taken.
37. **What is a Pareto front?** The set of designs where no other design is better in every objective. We show a sampled, predicted front for S11 vs gain.
38. **Limits of a weighted sum?** It can only reach convex parts of the Pareto front. NSGA-II-type methods would be needed for full multi-objective search.
39. **What is the κ (kappa) option?** Risk-averse optimization: it uses mean - κσ for gain and mean + κσ for S11, which penalizes designs in uncertain regions.

## Software and scientific integrity
40. **Why SQLite and SQLAlchemy?** SQLite is a zero-configuration, file-based, transactional database that suits a local tool. SQLAlchemy gives ORM models, foreign keys, check constraints and a separate data-access layer.
41. **How is reproducibility ensured?** Fixed seeds, dataset SHA-256 checksums, stored hyperparameters, validation config, bounds, objectives and library versions, saved model artefacts, and a one-command demo script.
42. **What is your contribution?** The implementation and integration of an explainable, uncertainty-aware, provenance-tracking workflow, plus its evaluation. It is not the invention of ML-based antenna optimization.
43. **What is the single most important limitation?** No real EM or measured data was used, so every accuracy number describes recovery of an assumed formula, not real antenna physics.
