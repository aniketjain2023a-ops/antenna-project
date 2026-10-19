# How to interpret the metrics, plots and results

> Context: the numbers quoted come from `sample_outputs/demo_results_summary.md`, measured on the
> SYNTHETIC dataset. They show how well the models recover an assumed formula, not real antenna accuracy.

## Metrics

| Metric | Meaning | Good value | Our example |
|---|---|---|---|
| **Test R²** | Fraction of target variance explained on held-out data. 0 = as good as predicting the mean; negative = worse. | Close to 1 | GP: 0.961 (S11), 0.995 (gain). Dummy ≈ 0 |
| **Test MAE** | Typical absolute error in the target unit | Small relative to the target range | GP S11: 0.66 dB on a range of ~24 dB |
| **Test RMSE** | Like MAE, but penalizes large errors more. RMSE ≥ MAE | Small; RMSE ≈ MAE means no large outliers | GP S11: 0.79 dB |
| **CV R² mean ± std** | Average and spread over 5 folds of the *training* data, used for model selection | High mean, small std | GP S11: 0.967 ± 0.007 |
| **Coverage95** | Share of test points inside mean ± 1.96σ | ≈ 0.95. Lower = over-confident, higher = over-cautious | GP 0.94-0.95; RF/ET for S11 0.83-0.84 (over-confident); ET gain 1.00 (over-cautious) |

Noise floor: the synthetic data contains noise with σ = 0.6 dB (S11) and 0.15 dB (gain). An RMSE
close to these values means the model is near the best that is achievable. The GP's gain RMSE
(0.156 dBi) is essentially at the noise level. Its S11 RMSE (0.79 dB) is slightly above the floor
because the resonance valley is sharp.

## Plots

- **Actual vs predicted:** points on the dashed y = x line are perfect. Systematic bending
  (for example, predictions not reaching the deepest S11 values) means the model smooths the
  extremes. That is visible for S11 below about -18 dB.
- **Residual plot:** residuals should scatter around 0 with no pattern. A funnel shape means the
  error depends on the predicted value.
- **Model comparison bars:** compare models per target. Always check that every model clearly
  beats the Dummy model.
- **Distribution histograms:** S11 is skewed toward 0 dB because most random geometries are off
  resonance. Only 58/500 samples reach -10 dB. This is expected and is why outlier flags on S11
  should not be "cleaned".
- **Correlation matrix:** Pearson r measures *linear* association only. The S11 resonance is
  non-linear in L, so |r| understates how important L really is.
- **Feature importance (permutation):** the drop in test R² when a feature is shuffled. L_mm ≈ 1.8-1.9,
  G_mm 0.37-0.68: the model relies mostly on L. Values above 1 are possible because R² can fall below 0.
- **Sensitivity (OAT) plot:** the predicted target as one parameter is swept with the others fixed, with a
  ±1σ band. The gain-vs-L curve peaks near the resonance (L ≈ 10.2 mm at G = 1.1 mm).
- **Partial dependence:** the average effect of one feature. It is flatter than the OAT curve because
  it averages over all G values, so the interaction is hidden.
- **Uncertainty map:** for the GP, uncertainty is highest at the corners and edges of the domain,
  where there are fewest neighbouring samples. For tree models it is highest along the steep
  resonance valley, where trees disagree.
- **Baseline vs candidate bars:** grey = baseline, blue = ML candidates, with error bars of ±1σ (approximate).
- **Convergence plot:** best penalised objective vs number of surrogate evaluations. Early high values
  are penalties for infeasible designs.
- **Trade-off (Pareto) scatter:** each dot is a sampled design predicted by the surrogate. The green
  line is the predicted Pareto front, stars are optimizer candidates, and the diamond is the baseline.
  Moving left (deeper S11) along the front costs gain.
- **Synthetic oracle table:** compares surrogate predictions with the noise-free generating formula.
  This is only possible for synthetic data. At the gain optimum the agreement is within 0.01 dBi. At the
  S11 optimum the surrogate under-predicts the depth of the null (-22.8 vs -24.9 dB).

## Optimization results

| Objective | Result (predicted) | Interpretation |
|---|---|---|
| Max gain, S11 ≤ -10 dB | L = 10.171, G = 1.281 mm → 8.02 dBi, -15.5 dB | +0.24 dBi over the baseline. The constraint is not active (S11 margin 5.5 dB). The optimum sits where the metasurface term (peak G = 1.3) meets the resonance |
| Min S11, gain ≥ 7 dBi | L = 9.936, G = 0.853 mm → -22.8 dB, 7.29 dBi | Best matching near G = 0.8 mm, as designed into the formula. The gain constraint is satisfied with a 0.29 dB margin |
| Trade-off 0.5/0.5 | L = 9.945, G = 0.876 mm → -22.8 dB, 7.33 dBi | Normalization makes the deep S11 worth more than the small gain loss at equal weights. Raise the gain weight to move toward the gain optimum |

All three algorithms agree to within 0.01 mm. When independent optimizers agree, a search
failure is unlikely, but this says nothing about whether the surrogate is correct.

## What these results do NOT show
- That any candidate would perform as predicted when fabricated or simulated in an EM solver.
- That the uncertainty intervals are calibrated in general.
- That uncertainty-guided sampling would improve the models.
