# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.38] - 2026-10-29

### Added

- **Mergeable sufficient-statistic states for OLS / WLS / ridge: `solvers::SufficientStats` (#122, DataZooDE/anofox-statistics#266).** The state holds the row counts (all rows and rows with positive weight), `Σw`, `Σ ln w`, the weighted means and the centred co-moments `Cxx`, `Cxy`, `Cyy`. Rows go in with a weighted Welford update (`push`, `push_weighted`) or a two-pass block constructor (`from_data`). States combine with the Chan–Golub–LeVeque pairwise formula (`merge`, `merged`) in any order and grouping, and an empty state is the identity. The means are stored relative to a per-state shift (the first row), so data with a large offset (`x ≈ 1e9 + noise`) keeps the precision of the batch fit. `to_bytes` / `from_bytes` give a stable, versioned little-endian encoding that rejects a wrong magic or version, a length that does not match the feature count, and inconsistent contents.
  - **`OlsRegressor::fit_from_state`, `WlsRegressor::fit_from_state`, `RidgeRegressor::fit_from_state`** return what `fit()` returns on the concatenated rows (to 1e-12 relative). That covers coefficients, R-`lm`-style aliasing in column order (from the pivots of a sequential Cholesky factorisation of the Gram matrix, same tolerance), classical standard errors, t, p and confidence intervals, R², adjusted R², F, MSE, log-likelihood, AIC / AICc / BIC, the variance factor, and confidence / prediction intervals for new rows. For ridge the λ (with its `LambdaScaling`) is applied at finalize, so one state serves a λ grid; the sandwich SEs are included. Weights recorded in the state are honoured: OLS on a weighted state is the WLS fit.
  - Row-level quantities are not available from a state: residuals, fitted values, leverage, LOO, Cook's distance and the HC0–HC3 sandwich (`Σ eᵢ² xᵢxᵢᵀ` needs the residuals, so it is not mergeable in one pass). `residuals` / `fitted_values` are empty, and `hc_standard_errors` / `hc_inference` return an error. GLMs have no exact single-pass state; the module docs show how a state with working responses and weights makes each IRLS step distributable.
  - Validated in `tests/issue122_sufficient_stats.rs`: random splits into 1–7 parts with random merge orders and byte round trips against `fit()` (OLS and WLS with and without intercept, zero weights, ridge over λ ∈ {0, 1e-3, 0.5, 10} with both scalings, aliased duplicate / constant / linear-combination columns, intervals); large offsets against a fit on the shifted data; and R `lm(weights =)`.
- **Regression-spline bases for ordinary design matrices: `splines` (#99, RMS §2.4).**
  - **`Rcs`**: a restricted (natural) cubic spline in Harrell's truncated-power parametrisation (`Hmisc::rcspline.eval` / `rms::rcs`). With `k` knots it has `k − 1` columns, `x, x', x'', …`, and is linear beyond the outer knots. Normalisation is `RcsNorm::{None, LastInterval, Range}` (`norm = 0, 1, 2`), with `Range`, i.e. `(t_k − t₁)²`, as the default. `Rcs::new(knots)` takes user knots; `Rcs::from_data(x, nk)` places the defaults. Keep the `Rcs` value to rebuild the basis on new data. `Rcs::restate` returns the unrestricted truncated-power coefficients of a fitted spline (`rcspline.restate`, including the two implied by the linearity restriction), and `RcsRestated::eval` evaluates them.
  - **Default knots**, `rcs_default_knots(x, nk)` / `rcs_default_knots_with(x, nk, fractied)`, port `rcspline.eval`:
    - the quantiles of RMS Table 2.3 (`default_knot_quantiles`, R type-7 quantiles);
    - outer knots moved to the 5th smallest / largest value when `n < 100`;
    - separate handling of extremes tied in at least `fractied` of the rows;
    - interior distinct values when there are few distinct values;
    - the alternate algorithm when the quantiles are not distinct.
  - **`Lsp`**: a linear spline `x, (x − t₁)₊, …` (`rms::lsp`).
  - **`restricted_interaction(a, b)`**: `rms::%ia%` (RMS §2.7.2), the products of two bases without doubly nonlinear terms, in the rms order.
  - Each basis is a `Basis` with rms column names and per-column `nonlinear` flags (`nonlinear_columns()` gives the chunk for a test of linearity).
  - Validated in `tests/r_validation_rcs.rs` (generator `tests/r_scripts/generate_rcs_validation.R`) against Hmisc 5 / rms 8: default knots for 3–7 knots on six data sets (small n, large n, ties at one or both extremes, few distinct values, integers), the basis for every normalisation, `rms::rcs`, `rcspline.restate`, `rms::lsp` and `%ia%`.
- **Term-aware Wald tests and the `rms::anova.rms` table (#100, RMS §2.6, §4).** Everything works from `β̂` and `V̂` alone, so any fit qualifies, and robust or bootstrap covariances can be passed in.
  - **`inference::wald_test(beta, vcov, C, r, dist)`** tests `H₀: Cβ = r` with `W = (Cβ̂ − r)ᵀ(CV̂Cᵀ)⁻¹(Cβ̂ − r)` against `χ²(q)`, or as `F = W/q` against `F(q, df)` when the dispersion is estimated (`WaldDistribution::{ChiSquare, F}`). `wald_chunk(beta, vcov, columns, dist)` is the chunk test. `WaldTest::chi_square_minus_df` gives the RMS importance measure. `drop_intercept` splits off the intercept.
  - **`design::TermMap` / `design::Design`** record which slope columns belong to which predictor, which are nonlinear, and which are interactions. `Design` stacks `splines::Basis` blocks (`Basis::linear`, `Basis::new` for factor dummies) into the design matrix. Two-way interactions come in both rms forms: `add_interaction` is the full tensor product of `a * b` (first factor fastest); `add_restricted_interaction` is `%ia%`.
  - **`inference::anova(beta, vcov, &terms, &AnovaOptions)`** ports `anova.rms` row by row, with the same labels, the same columns tested and the same rules:
    - each predictor with its higher-order factors, `All Interactions` and `Nonlinear`;
    - each interaction with `Nonlinear`, `Nonlinear Interaction : f(A,B) vs. AB`, `f(A,B) vs. Af(B) + Bg(A)` and the single-factor nonlinear-interaction rows;
    - `TOTAL NONLINEAR`, `TOTAL INTERACTION`, `TOTAL NONLINEAR + INTERACTION` and `TOTAL`.

    With σ² and an F reference it adds partial sums of squares and the `ERROR` row (`ss = TRUE`).
  - `FittedOls::vcov`, `FittedWls::vcov` and `FittedRidge::vcov` return `σ̂² · M`, intercept first.
  - Validated in `tests/r_validation_anova.rs` (generator `tests/r_scripts/generate_anova_validation.R`) against `rms::anova.rms` for four models: OLS with an rcs × linear interaction (χ², F and partial SS), the same design as `lrm`, an rcs × rcs tensor interaction plus a second interaction, and `%ia%`. It also checks the nested-model F test (`anova(lm, lm)`), a two-row general linear hypothesis with `r ≠ 0`, and an end-to-end Rust design → OLS → `vcov` → ANOVA run that reproduces rms. Statistics agree to 1e-9; p-values agree to statrs' tail accuracy (1e-5 relative near p = 1e-11).
- `splines::Basis::new` / `Basis::linear` constructors and `SplineError::Length`.
- **Predictive accuracy indexes: `diagnostics::accuracy` (#102, RMS §4.5, §10.8).** These are free functions over `(y, prediction, [weights])`, so they apply to any model.
  - `concordance(x, y, w)`: weighted concordant / discordant / tied pair counts (`survival::concordancefit`), with `c()`, `dxy()`, `gamma()` and `tau_a()` (the rank statistics of `rms::lrm`). O(n log n), using a Fenwick tree.
  - `somers2(p, y, w)`: ROC area and Somers' Dxy (`Hmisc::somers2`, case weights).
  - `rcorr_cens(x, time, event)`: Harrell's C for right-censored data, matching `Hmisc::rcorr.cens`. It gives C, Dxy, the S.D. of Dxy (`sqrt(Σ(cᵢ − dᵢ − Dxy·rᵢ)²)/R`), and the relevant / concordant / uncertain ordered-pair counts. O(n log n).
  - `brier_score`, `gini_md` (`Hmisc::GiniMd`, the g-index) and `nagelkerke_r2`. `r2_measures(lr, p, n, ess)` is `Hmisc::R2Measures`: R²(n), R²(p,n) and the effective-sample-size variants of RMS 2nd ed. §4.5. `effective_sample_size(n, freqs)` is `n(1 − ΣP³)`.
  - `lrm_stats(y, lp, w, df)`: the binary `rms::lrm` stats vector from the linear predictor — model LR, P, C, Dxy, γ, τ-a, Nagelkerke R², R2Measures, Brier, g, gr, gp.
  - `val_prob(p, y)`: `rms::val.prob` external validation — Dxy, C, R², D, U, Q (with χ² and p), Brier, the logistic calibration intercept and slope, Emax / E90 / Eavg from `lowess(p, y, iter = 0)`, and Spiegelhalter's z.
  - Validated in `tests/r_validation_accuracy.rs` (generator `tests/r_scripts/generate_accuracy_validation.R`) against `concordancefit` (with ties and weights), `somers2` (with and without weights), the `lrm` stats (with and without weights), `rcorr.cens` (censored and uncensored, with ties), `R2Measures` and `val.prob`; values agree to 1e-8 or better.
- **Cluster-robust (Huber–White) sandwich covariance: `inference::cluster` (#105, RMS §9.5).** It computes `V = A (Σ_g s_g s_gᵀ) A`, as `sandwich::vcovCL` and `rms::robcov` do, with one or more clustering dimensions. Several dimensions are combined by inclusion–exclusion (Cameron, Gelbach & Miller), e.g. `V₁ + V₂ − V₁∩₂` for two-way clustering.
  - `ClusterOptions { kind: ClusterType::{HC0, HC1, HC2, HC3}, cadjust }` follows `vcovCL(type, cadjust)`:
    - `cadjust` multiplies each dimension by `G/(G−1)`;
    - HC1 multiplies by `(n−1)/(n−k)`;
    - HC2 is Bell–McCaffrey bias-reduced linearization and HC3 is jackknife-like, with per-cluster `(I − H_gg)^{−½}` / `(I − H_gg)^{−1}`. These are for linear models only, weighted ones included.
    - Presets: `ClusterOptions::cr1()` is Stata's CR1 and the default; `::hc0()` is `vcovCL`'s GLM default; `::robcov()` applies no correction, as `rms::robcov` and `survreg(cluster =)`.
  - `cluster_sandwich(bread, scores, clusters, opts)` works for any model from its inverse information and per-observation scores.
  - `lm_cluster_vcov`, plus `FittedOls::cluster_vcov` / `FittedWls::cluster_vcov`, for linear models (aliased columns get zero rows and columns).
  - `glm_cluster_vcov(family, x, …, mu, prior_weights, …)` for GLMs of any `GlmFamily` (scores `p(y − μ)/(V g′) x`, bread `(XᵀWX)⁻¹`).
  - `FittedAft::score_contributions` and `FittedAft::cluster_vcov` for AFT models (all parameters including `log σ`).
  - The result is a covariance matrix, intercept first, ready for `inference::wald_test` / `anova`.
  - Validated in `tests/r_validation_cluster.rs` (generator `tests/r_scripts/generate_cluster_validation.R`):
    - against `vcovCL` for OLS with HC0–HC3, with and without `cadjust`, and for two-way clustering;
    - for WLS with HC0–HC3 and for a model without an intercept;
    - against `rms::robcov`;
    - for Poisson, logit, probit and Gamma(log) GLMs, plus two-way and prior-weighted Poisson;
    - against `survreg(..., cluster = id)` for a Weibull AFT.
- `compute_hc_standard_errors` already refuses fits without residuals; `lm_cluster_vcov` does the same (e.g. fits from `SufficientStats`).
- **Nonparametric survival analysis: `survival` (#113, RMS Ch 17).**
  - **`survfit(&SurvInput, strata, &SurvfitOptions)`** is `survival::survfit`. Per stratum and at every distinct time it returns n.risk, n.event, n.censor, the Kaplan–Meier survival, the Greenwood `std.err` of `−log S`, confidence limits (`ConfType::{Log, LogLog, Plain}`, any level), and the cumulative hazard with its standard error (`HazardType::NelsonAalen`, or the tie-corrected `FlemingHarrington` of `ctype = 2`).
    - `SurvInput` takes case weights and left truncation (counting-process `(entry, stop]` data).
    - `timefix` (on by default) merges times that differ by floating-point noise, as `Surv(..., timefix = TRUE)` / `aeqSurv` does.
    - `kaplan_meier(time, event, opts)` is the one-curve shortcut.
  - **`SurvCurve` methods:**
    - `median()` — the median with its confidence limits, including the midpoint rule when S = 0.5 over an interval;
    - `restricted_mean(τ)` — the restricted mean and its standard error;
    - `summary(τ)` — the `summary(survfit)$table` row;
    - `survival_at(t)`.
  - **`survdiff(time, event, group, strata, rho)`** is `survival::survdiff`: the k-sample log-rank test (`ρ = 0`) and the G-ρ family (`ρ = 1`, Peto–Peto), optionally stratified, with observed, expected, variance matrix, χ², d.f. and p.
  - Validated in `tests/r_validation_survival.rs` (generator `tests/r_scripts/generate_survival_validation.R`) on `survival::lung`:
    - KM with every CI type and Fleming–Harrington;
    - sex strata and the summary tables (RMST, median and its limits);
    - case weights (with `robust = FALSE`; for non-integer weights R defaults to a robust variance that this does not implement);
    - left-truncated data;
    - the median midpoint rule;
    - two- and four-group log-rank, Peto–Peto and stratified log-rank.
- **Cox proportional hazards regression: `survival::cox` (#114, RMS Ch 20).** `coxph(x, &CoxInput, &CoxOptions)` maximises the partial likelihood by Newton–Raphson with step halving, as `survival::coxph`.
  - **Tie handling:** Efron (default) or Breslow.
  - **Inputs:** case weights, offsets, strata, and counting-process `(entry, stop]` data for left truncation and time-dependent covariates. Risk sets are kept in one descending sweep that adds rows at their stop time and removes them at their entry time.
  - **Options:** an optional ridge penalty `λ/2 Σβ²` (`ridge(..., theta = λ, scale = FALSE)`), and R's `timefix`.
  - **`CoxFit` returns:** coefficients, covariance, SEs, z and p-values; the log partial likelihoods at 0 and at the estimate; the LR, Wald and score tests; `concordance()` (Harrell's C within strata), `dxy()` and `r2()` (as `rms::cph`); and `linear_predictors()`.
  - **Residuals and robust variance:** `score_residuals()` (Efron-corrected, `residuals(type = "score")`) and `robust_vcov(clusters)` (`coxph(..., cluster = id)`, via `inference::cluster_sandwich`).
  - **Prediction:** `predict_survival(x0, stratum)` is `survfit(fit, newdata)` — the Efron / Breslow cumulative hazard for the covariates, its standard error including the uncertainty of β̂, and log-scale confidence limits.
  - Validated in `tests/r_validation_cox.rs` (generator `tests/r_scripts/generate_cox_validation.R`) against coxph on `lung`:
    - Efron, Breslow, weights with an offset, strata and ridge;
    - score residuals and the robust variance (Efron, Breslow, weighted);
    - predicted survival with SEs and CIs (Efron, Breslow);
    - `cph`'s LR, R² and Dxy;
    - counting-process data with a time-dependent covariate (`heart`), with Efron, Breslow and the robust variance.
- **Cox and AFT model diagnostics (#115, RMS §18.3, §20.6–20.8).**
  - **Residuals on `CoxFit`:** `martingale_residuals`, `deviance_residuals`, `schoenfeld_residuals(scaled)` (event rows by stratum and time, Efron-averaged over tied deaths; scaled = `β̂ + n_d V r`), `dfbeta` (weighted, as survival ≥ 3) and `dfbetas`, as `residuals.coxph`.
  - **`CoxFit::cox_zph(transform, terms)`** is the Grambsch–Therneau proportional-hazards test of survival ≥ 3. It is the score test at `(β̂, 0)` for adding `g(t)·x`, per term and global, with `ZphTransform::{Km, Rank, Identity, Log}`. It also returns the transformed times and the scaled Schoenfeld residuals for plotting β(t).
  - **AFT:** `FittedAft::standardized_residuals` (`(log t − η)/σ`) and `cox_snell_residuals` (`−log S(t | x)`), for checking the error distribution against a Kaplan–Meier curve.
  - Validated in `tests/r_validation_coxdiag.rs` (generator `tests/r_scripts/generate_coxdiag_validation.R`) against `residuals.coxph` and `cox.zph` (all four transforms and the plotting data), for Efron, Breslow, weighted, stratified and counting-process (`heart`) fits, and against `survreg` for the AFT residuals.
- **Resampling model validation and calibration: `validation` (#103, RMS §5.3, §10.11).**
  - **`predab_resample(resamples, names, fit, measure)`** is the generic engine of `rms:::predab.resample`. The `fit` closure replays the whole modelling process (variable selection included) on each training sample, and `measure` evaluates any vector of indexes on the training rows (apparent) and on the test rows. It reports apparent, training, test, optimism and corrected values, with the number of resamples used per index.
  - **`Resamples`:**
    - `bootstrap` — Efron–Gong optimism;
    - `crossvalidation` — B-fold, with rms's group boundaries;
    - `point632` — Efron's .632, with rms's omission weights;
    - each from explicit indices (e.g. to replay R's draws) or seeded (`*_seeded`).
  - **`validate_lrm` / `validate_ols`** are `validate.lrm` / `validate.ols`: Dxy, R², calibration intercept and slope, Emax, D, U, Q, Brier, g, gp / R², MSE, g, intercept, slope.
  - **`calibrate_lrm` / `calibrate_ols`** are `calibrate`: the lowess calibration curve on a 50-point grid, its bootstrap overfitting correction, and the mean absolute error, mean squared error and 0.9 quantile absolute error of the corrected curve.
  - Validated in `tests/r_validation_validate.rs` (generator `tests/r_scripts/generate_validate_validation.R`). R's resamples are replayed exactly, and the tests cover `validate` for `lrm` and `ols` with bootstrap, 10-fold crossvalidation and .632, plus `calibrate` for `lrm` and `ols`.
- **Likelihood-ratio and score tests for nested models, and profile-likelihood CIs: `inference::likelihood` (#101, RMS Ch 9).**
  - **`LikelihoodFit` trait** (`−2 log L`, the null value, the dispersion the difference is scaled by, and the parameter count). It is implemented for `FittedBinomial`, `FittedPoisson`, `FittedNegativeBinomial` (via `−2 log L`, since the deviance depends on θ), `FittedTweedie`, `FittedGamma`, `FittedAft` and `CoxFit`.
  - **`lr_test(reduced, full)`** is `anova(fit0, fit1, test = "LRT")`, with the difference scaled by the larger model's dispersion. **`model_lr_test(fit)`** is the model LR χ² against the null model with its d.f.
  - **`fit_glm(family, &GlmData, Dispersion)`** is a compact IRLS fit of any `GlmFamily`, with prior weights, offsets, the R-style null deviance and the fixed or Pearson-estimated dispersion.
  - **`glm_score_test(family, reduced, x_add, φ)`** is Rao's score test for adding predictors, evaluated at the reduced fit only: the joint `Uᵀ S⁻¹ U / φ` of `anova(..., test = "Rao")`, plus the per-column z of `statmod::glm.scoretest`.
  - **`profile_confint(family, data, dispersion, level)`** reproduces `confint.glm` (R ≥ 4.4 / MASS): the same profiling grid, a port of R's FMM `spline()`, and `approx` inversion.
  - Validated in `tests/r_validation_likelihood.rs` (generator `tests/r_scripts/generate_likelihood_validation.R`) against `glm`, `anova` LRT / Rao, `glm.scoretest` and `confint` (95% and 90%), for binomial, weighted Poisson with an offset, and Gamma(log) with estimated dispersion. It also checks the trait on the crate's binomial and Gamma regressors.
- **Partial effects and contrasts on the covariate scale: `effects` (#116, RMS §5.1).** A `ModelSpec` records how each predictor enters the design — `linear`, `rcs`, `lsp`, or `categorical` with treatment contrasts — and the interactions (tensor or `%ia%`). It carries `datadist`-style `Limits`: quartiles for effects; the median (or the mode for categorical predictors) to adjust to; the `10/max(n, 200)` quantiles for prediction grids; and rms's rules for few distinct values. `design(columns)` builds the design, and `row(setting)` builds a design row with the other predictors at their adjust-to values.
  - `summary(beta, vcov, …)` is `summary.rms`: inter-quartile-range effects for numeric predictors, and each level against the adjust-to level for categorical ones, with CIs (t for OLS, normal otherwise). `exp` gives odds / hazard / time ratios.
  - `contrast` and `double_contrast` are `contrast.rms`, single and double difference, with any covariance (model, robust or bootstrap).
  - `predict_grid` is `Predict`: a 200-point grid between the prediction limits, or given values, with pointwise CIs.
  - Validated in `tests/r_validation_effects.rs` (generator `tests/r_scripts/generate_effects_validation.R`) against `datadist`, `summary` (with odds ratios), `contrast` (single, double, factor levels) and `Predict` for `ols` and `lrm` with an rcs × linear interaction and a three-level factor.

- **Bootstrap covariance for any fit: `validation::bootcov` (#104, RMS §5.2, §5.4).**
  - `bootcov(samples, fit)` refits any model (a closure returning its coefficients) on each resample. It returns the replicates, their mean and the bootstrap covariance (divisor B − 1, as `rms::bootcov`), which is usable as `V̂` in `wald_test`, `anova` and `effects` contrasts.
  - `percentile_ci(level)` gives percentile intervals; `linear_combination_ci(c, level)` gives them for contrasts and predictions.
  - `bootstrap_rows` draws ordinary resamples; `cluster_bootstrap_rows` draws cluster (block) resamples of whole clusters.
  - `rank_intervals(original, replicates, level)` gives bootstrap intervals for the ranks of predictor importance, e.g. from partial χ² − d.f. (RMS §5.4).
  - Validated in `tests/r_validation_bootcov.rs` (generator `tests/r_scripts/generate_bootcov_validation.R`) against `bootcov` replicates, covariance and percentiles, with ordinary and cluster bootstrap, replaying R's resamples.

- **Fast backward step-down: `inference::fastbw` (#108, RMS §5.5.2).** `fastbw(beta, vcov, n_intercepts, &terms, rule, type)` is `rms::fastbw`.
  - It uses only `β̂` and `V̂`, deleting whole terms by Wald χ² with `StepRule::Aic { k, threshold }` or `StepRule::P { sls }`, applied to the residual or the individual test.
  - It reports each step's χ², d.f., p, residual χ², d.f., p and AIC, plus the kept terms and the approximate reduced-model coefficients `β_in − V_io V_oo⁻¹ β_out`.
  - Validated in `tests/r_validation_fastbw.rs` (generator `tests/r_scripts/generate_fastbw_validation.R`) against `fastbw` on an `lrm` with a spline and a factor, for all four rule / type combinations.

### Changed

- **Fewer allocations and less work per OLS fit (#129).** For many small per-group fits, 100K fits of 20×3 (`examples/many_small_fits.rs`) go from 58 to 32 allocations per fit (−45%) and from 6.3 to 4.4 µs without inference, and from 76 to 48 allocations and from 11.3 to 9.2 µs with inference.
  - With `compute_inference(true)`, `(XᵀX)⁻¹` is computed once and shared by the standard errors and the prediction-interval factor; it used to be computed twice.
  - With `compute_inference(false)`, the factor is deferred: the fit keeps the small Gram matrix of the kept columns and inverts it on first use (`predict_with_interval`, `variance_factor`, `vcov`). Results agree with the eager factor to 1e-12.
  - `lm_alias_columns` keeps its orthonormal basis in one flat buffer, with the same arithmetic.
  - Fit results are moved into the `RegressionResult` instead of allocated and cloned.

### Fixed

- `compute_hc_standard_errors` (and the `hc_*` methods built on it) returns an error instead of panicking when the residuals do not have one entry per row of `x`.

## [0.5.37] - 2026-10-23

### Added

- **Exact leave-one-out diagnostics for ridge regression (DataZooDE/anofox-statistics#235).** `FittedRidge::loo(x, y) -> RidgeLoo { leverage, loo_residual, loo_fitted, press }`: the diagonal of the ridge hat matrix `H = X̃(X̃'X̃ + λP)⁻¹X̃'` (intercept unpenalised, i.e. `1/n + x̃ᵢ'(Xc'Xc + λI)⁻¹x̃ᵢ` on the centred design, from one thin SVD), the exact leave-one-out residuals `(yᵢ − ŷᵢ) / (1 − hᵢ)`, and `Press` (PRESS, its RMSE, predicted R²). `λ` is the effective penalty on the sum-of-squares scale; with `LambdaScaling::Glmnet` that is `λ·n` of the full fit, held fixed for the leave-one-out fits. λ = 0 gives the OLS values of `augment` / `press`. Checked against explicit refits without each row (`tests/ridge_loo.rs`, agreement ≤ 1e-12, with and without an intercept, p > n, glmnet scaling).
- **One-step approximate leave-one-out for the GLMs.** `diagnostics::Augment` has two new fields, `approx_loo_fitted = g⁻¹(ηᵢ − hᵢ zᵢ / (1 − hᵢ))` (`zᵢ` the working residual, `hᵢ` the IRLS leverage) and `approx_loo_residual = yᵢ − approx_loo_fitted`: the Pregibon (1981) one-step approximation, R's `influence(fit)$coefficients` applied to the linear predictor. Named `approx_*` because it is not the refit; the exact `loo_residual` / `loo_fitted` stay `NaN` for GLMs, and the new fields are `NaN` for the linear models. Validated against R in `tests/r_validation_approx_loo_glm.rs` (Poisson, logistic, Gamma; dedicated regressors and the penalized-GLM engine), which also shows the approximation is 20–60× closer to the exact refits than the full-data fit.

### Changed

- `diagnostics::Augment` has the two new public fields above; code that builds an `Augment` with a struct literal must set them.

## [0.5.36] - 2026-10-23

### Added

- **`inference::anova_table(x, y, terms, &AnovaTableOptions) -> AnovaTable` (DataZooDE/anofox-statistics#231).** Per-term ANOVA tables for OLS without interactions: sequential sums of squares (`AnovaType::I`, R's `anova(lm(...))`, each term adjusted for the terms before it) and partial sums of squares (`AnovaType::II`, `car::Anova(fit, type = 2)`, each term adjusted for all other terms, i.e. the drop-one nested F test against the full model). A term is one column or a group of columns (`terms: Some(&[vec![0], vec![1, 2], ...])`, for example the dummies of a factor); `None` makes every column its own term. Each `AnovaTerm` has `df` (the rank the term adds), `sum_sq`, `mean_sq`, `f_statistic` and `p_value`, all F tests against the full model's residual mean square; the table carries `residual_df`, `residual_sum_sq`, `residual_mean_sq` and `n_observations`. A term that adds no rank (aliased) gets `df = 0`, `sum_sq = 0` and NaN F / p (R drops it from `anova()`, `car::Anova` refuses the model); no residual df gives NaN F everywhere; without an intercept the sums of squares are uncentred. Validated in `tests/r_validation_anova_table.rs` against R 4.x `anova()` and car 3.1.5 `Anova(type = 2)` to 1e-9 (generator `tests/r_scripts/generate_anova_table_validation.R`): three regressors, a different term order, no intercept, an aliased column, and `mtcars` `mpg ~ wt + factor(cyl) + hp` with a two-column factor term.

## [0.5.35] - 2026-10-23

### Added

- **Leave-one-out (PRESS) diagnostics for the least-squares fits (#88, DataZooDE/anofox-statistics#220).** `diagnostics::Augment` has two new fields: `loo_residual = (yᵢ − ŷᵢ) / (1 − hᵢ)`, the exact error `yᵢ − ŷ₍ᵢ₎` of the fit without row `i` (R's `rstandard(fit, type = "predictive")` for OLS; on the response scale for WLS), and `loo_fitted = ŷ₍ᵢ₎`. New `diagnostics::press(fit, x, y, weights) -> Press { press, press_rmse, loo_r_squared, n }`: `press = Σ wᵢ loo_residualᵢ²` (R's `sum(rstandard(fit, type = "predictive")^2)`), `press_rmse = √(press / n)` and the predicted R² `1 − press / SST`, with the SST of the fit's R² (centred with an intercept, uncentred without). No refitting: both come from the leverage. A row with `hᵢ > 1 − 10ε` (R's `lm.influence` threshold, e.g. a dummy with one observation, or zero residual degrees of freedom) has no leave-one-out prediction and gives NaN, as do `press` and its derived statistics; the GLMs give NaN (no exact closed form). Validated against R in `tests/r_validation_loo_press.rs` (generator `tests/r_scripts/generate_loo_press_validation.R`, every residual also checked against an explicit refit). Ridge (`H = X(XᵀX + λI)⁻¹Xᵀ`) is not covered yet.

### Changed

- `diagnostics::Augment` has the two new public fields above; code that builds an `Augment` with a struct literal must set them.
## [0.5.34] - 2026-10-23

### Added

- **Credible intervals for `BayesianRidge` / `ArdRegression` (#92, DataZooDE/anofox-statistics#222).** `credible_intervals(level) -> CredibleIntervals` on `FittedBayesianRidge` and `FittedArd`: the posterior SD of every coefficient (`sqrt(diag(Σ))`, 0 for features ARD pruned) and the central credible interval `mean ± z_{(1+level)/2} · sd` of the Gaussian posterior (conditional on the estimated precisions). The intercept, which is not part of `Σ` (as in scikit-learn), gets a flat prior: its posterior SD is `sqrt(1/(n·α) + x̄ᵀ Σ x̄)`, the noise uncertainty of `ȳ` plus the slope uncertainty propagated through the centring. No test statistic or p-value. Validated in `tests/r_validation_bayesian.rs` against scikit-learn 1.9.1's `sigma_` and, for the intercept, the inverted full posterior precision of `(b0, β)` (generator `validation/python/generate_bayesian_validation.py`), to 1e-8.
- **Theil–Sen coefficient intervals (#92, DataZooDE/anofox-statistics#222).** `TheilSenRegressor` previously had no inference: `std_errors`, the t / p values and the intervals of its `RegressionResult` were `None` and the F test NaN. New builder options `compute_inference` (default `false`), `confidence_level` (0.95), `ci_method` (`TheilSenCiMethod::{Auto, Exact, Bootstrap}`, default `Auto`) and `n_bootstrap` (999):
  - **Exact** (one feature with an intercept): the distribution-free interval of Sen (1968), the pairwise slopes at ranks `round((N − C)/2)` and `round((N + C)/2 + 1)` with `C = z · sqrt(Var S)` and Kendall's tie-corrected `Var S`, plus Kendall's continuity-corrected z (in `t_statistics`) and its two-sided p-value for slope = 0. Matches R's `trend::sens.slope` (and `cor.test(method = "kendall")` with ties in x) to 1e-12 (`tests/issue92_theil_sen_inference.rs`, generator `tests/r_scripts/generate_theil_sen_ci_validation.R`). No standard error and no intercept interval; a bound whose rank falls outside `1..N` (tiny n) is NaN. The order statistics use the slope-counting path above 4M pairs.
  - **Bootstrap** (any number of features, and the intercept): pairs bootstrap with `n_bootstrap` refits; `std_errors` / `intercept_std_error` are the bootstrap SDs and the intervals the percentile intervals (R `quantile` type 7). Deterministic given `random_state` and independent of the row order; degenerate resamples are dropped (`FittedTheilSen::bootstrap_replicates()`). No t / p values.
  - `Auto` uses the exact interval where it applies and the bootstrap otherwise; `Exact` on any other design is an error. `FittedTheilSen::ci_method()` reports the method used. Aliased columns get NaN inference.
- `OptionsError::Unsupported(String)` for option combinations an estimator does not support.

## [0.5.33] - 2026-10-23

### Added

- **GLMM inference (#89, DataZooDE/anofox-statistics#218).** `FittedGlmm` gains:
  - `vcov()`: the full fixed-effect covariance `σ² M⁻¹` (lme4's `vcov()`).
  - **Satterthwaite degrees of freedom** for Gaussian fits built with `GlmmRegressorBuilder::satterthwaite(true)`: `satterthwaite_df()`, and `inference(GlmmDdf, level) -> GlmmInference` with t statistics, t p-values and t intervals (`lmerTest`'s `summary(lmer(...))` / `contest1D(..., confint = TRUE)`), or the existing Wald z inference with `GlmmDdf::Wald`. The df use lmerTest's construction: Richardson-extrapolated finite-difference Hessian of the REML / ML deviance in `(θ, σ)` and Jacobian of `vcov(β)`, the Hessian inverted through its eigen decomposition.
  - **Conditional standard errors of the BLUPs**: `random_effects_se()`, `random_effects_se_matrix()` and `factor_random_effects_se_matrix(f)`, the diagonal of `σ² Λ (ΛᵀZᵀWZΛ + I)⁻¹ Λᵀ` (`lme4::ranef(condVar = TRUE)`), for every family and the crossed / nested path.
  - `random_effects_conf_int(level)` / `factor_random_effects_conf_int(f, level)`: normal intervals `b ∓ z·se` for the BLUPs, as `broom.mixed::tidy(effects = "ran_vals", conf.int = TRUE)`.
  - Per-observation results of the training data: `fitted_values()` (with the BLUPs), `fitted_population()` (random effects at zero), `residuals()` (response) and `pearson_residuals()` (lme4's `residuals(type = "pearson")`), and `df_residual()` (lme4's `df.residual()`).
  - Validated against lme4 2.0.1 / lmerTest 3.2.1 in `tests/issue89_glmm_inference.rs` (generator `tests/r_scripts/generate_glmm_satterthwaite_validation.R`): vcov, BLUP SEs and fitted values to 1e-6, Satterthwaite df to 1e-5.

### Changed

- **More precise θ for random slopes and crossed factors.** The Nelder–Mead search over a multi-parameter θ is followed by up to 8 finite-difference Newton steps (each accepted only if it lowers the deviance). Nelder–Mead stops on the spread of the simplex values, which left θ accurate to about the square root of the tolerance; the fixed effects, SEs and variance components of such fits now agree with a tightly converged `lmer` / `glmer(nAGQ = 0)` to about 1e-7 instead of 1e-5.
## [0.5.32] - 2026-10-22

### Added

- **`inference::nested_f_test(x_reduced, x_full, y, &NestedFTestOptions) -> NestedFTest` (#90, DataZooDE/anofox-statistics#221).** The nested-model F test of R's `anova(lm(reduced), lm(full))`: both OLS fits on the same rows, `F = ((RSS_r − RSS_f) / df1) / (RSS_f / df2)` with `df1 = p_full − p_reduced`, `df2 = n − p_full` and the upper-tail p-value, plus both RSS values and both adjusted R² values (`summary.lm`'s). Parameter counts are numerical ranks, as R's `df.residual`, so aliased columns add nothing; `df1 = 0` or `df2 = 0` gives a NaN statistic (R's `NA`). An empty reduced design is the intercept-only model (the overall F test). The reduced columns must lie in the span of the full design (checked by regressing each on it; `NestedFTestError::NotNested` otherwise). `nested_f_test_indices(x_full, y, &reduced, ..)` takes the reduced design as column indices of the full one. Validated in `tests/r_validation_nested_f.rs` (generator `tests/r_scripts/generate_nested_f_validation.R`), including `df1 = 1` (F = t²), the intercept-only and no-intercept cases and a rank-deficient full design.

## [0.5.31] - 2026-10-22

### Added

- **`solvers::mediation::mediate(y, m, x, covariates, &MediationOptions) -> MediationResult` (#91, DataZooDE/anofox-statistics#223).** Causal mediation analysis for linear models, the no-interaction case of R's `mediation::mediate`: the mediator model `M ~ X + C` (path `a`) and the outcome model `Y ~ X + M + C` (`b`, `c'`) are fitted by OLS on the same complete rows, giving the ACME `a·b·Δ`, the ADE `c'·Δ`, the total effect and the proportion mediated for the contrast `Δ = treat_value − control_value`. Inference by a seeded nonparametric case-resampling bootstrap (default; percentile or `mediate`'s BCa interval, bootstrap SD as `std_error`, `mediate`'s p-value `2·min(#>0, #<0)/#draws`), the Sobel / delta method, or `mediate`'s quasi-Bayesian Monte Carlo. `MediationEffect::from_draws` summarises draws as `mediate` does. Validated in `tests/r_validation_mediation.rs` (generator `tests/r_scripts/generate_mediation_validation.R`): point estimates match `mediate`, `lavaan` and `lm` to 1e-10, intervals and p-values match `mediate` exactly given its own draws and within simulation error from ours, and the Sobel z matches `bda::mediation.test`.

## [0.5.30] - 2026-10-22

### Added

- **Cross-validation strategies for the regularisation paths: `solvers::cv` (#87, DataZooDE/anofox-statistics#204).**
  - **Fold generators**, deterministic and matching scikit-learn 1.9.1 without shuffling: `group_kfold_ids(groups, k, GroupBalance, seed)` (`GroupBalance::Rows` is `GroupKFold`: largest group first into the fold with the fewest rows; `GroupBalance::Random` is `GroupKFold(shuffle=True)` with the crate's generator), `logo_ids` (`LeaveOneGroupOut`), `stratified_kfold_ids(strata, k, seed: Option<u64>)` (`StratifiedKFold`; `Some(seed)` shuffles the fold order within each stratum), `stratified_group_kfold_ids` (`StratifiedGroupKFold`), `quantile_strata(y, n_bins)` to stratify a regression CV on a continuous response, and `group_codes`. Keys only need `Ord`, so several group variables are a tuple key.
  - **Explicit splits**: `CvSplit { train, test }`, `splits_from_fold_ids`, `time_series_splits(n, &TimeSeriesSplitOptions)` (`TimeSeriesSplit` with `n_splits`, `test_size`, `gap`, `TrainWindow::Expanding | Sliding(max_train_size)` and `min_train_size`) and `group_time_series_splits(time_keys, …)` for panels where several rows share a time point.
  - **`cv_path_splits(x, y, lambdas, splits, opts, CvMetric)`**: the `cv.glmnet` curve on any splits, with `CvMetric::Mse` or `CvMetric::Mae` (`cv_se` over the splits). `path::cv_path` keeps its signature and results and now calls it with `splits_from_fold_ids`. A Gaussian deviance metric is not added: for the Gaussian family it is the MSE.
  - **`nested_cv_path(x, y, lambdas: Option<_>, outer, opts, &NestedCvOptions)`**: the inner folds (`InnerFolds::{Kfold, Group, Logo, Stratified, TimeSeries}`) are built from each outer training set only, `λ` is chosen by `LambdaRule::{Min, OneSe}`, and with `lambdas = None` each outer split computes glmnet's default grid from its own training rows. Returns the per-split `λ`, test metric and inner curve plus the weighted mean and standard error. `nested_inner_splits` exposes the inner splits as row indices.
  - Validated in `tests/cv_splits_validation.rs` against scikit-learn fixtures (generator `validation/python/generate_cv_splits_validation.py`), with property tests (no group spans two folds, stratum counts per fold differ by at most one, no outer test row in any inner split, inner selection unchanged when outer test responses change) and a hand-rolled nested loop.

## [0.5.29] - 2026-10-15

### Fixed

- **Builds on Rust 1.86 again.** `solvers::lowess::lowess` (0.5.28) used `usize::is_multiple_of`, stable only since Rust 1.87, which broke the DuckDB extension's wasm build (Rust 1.86). It uses `n % 2 == 0` as `huber` and `ransac` do.

## [0.5.28] - 2026-10-15

### Added

- **Posterior predictive intervals for `BayesianRidge` and `ArdRegression` (DataZooDE/anofox-statistics#61).** Both fits keep the posterior covariance of the coefficients (`sigma()`, sklearn's `sigma_`: `(λI + αXᵀX)⁻¹`, over the kept features for ARD) and the feature means. `predict_std(x)` returns the posterior predictive standard deviation `sqrt((x − x̄)ᵀΣ(x − x̄) + 1/α)`, as sklearn's `predict(X, return_std=True)`, and `predict_with_interval` now returns Gaussian credible bounds `ŷ ± z·sd` (`IntervalType::Prediction`) or, without the `1/α` noise term, bounds for the mean (`IntervalType::Confidence`) instead of point predictions only. New accessors: `n_iter()` on both, `log_marginal_likelihood()` on `FittedBayesianRidge` (sklearn's final `scores_` entry), `kept_features()` on `FittedArd`; `sigma_diag()` now returns a `Vec<f64>` (0 for features ARD pruned).
- **`solvers::lowess::lowess(x, y, &LowessOptions) -> LowessFit` (DataZooDE/anofox-statistics#63).** Cleveland's robust locally weighted scatterplot smoother, a port of R's `stats::lowess` (`clowess`): `frac` (R's `f`, default 2/3), `iterations` (R's `iter`, default 3) and `delta` (default 1% of the range of `x`). Returns the sorted `x`, the smoothed values and the sort permutation. Validated against R in `tests/r_validation_lowess.rs` (generator `tests/r_scripts/generate_lowess_validation.R`).

### Fixed

- **`BayesianRidge` / `ArdRegression` follow scikit-learn step for step.** Initialisation `α = 1 / (var(y) + eps)`, the convergence test from the second iteration on (ARD: after zeroing pruned coefficients), and the final refit of the posterior mean with the converged hyperparameters, so coefficients, α, λ, `sigma_` and `n_iter_` now agree with scikit-learn 1.9.1 to 1e-8 (previously within 5e-3 / 5e-2). Their `RegressionResult` no longer reports the `RegressionResult::empty` zeros for `adj_r_squared`, `rmse`, `mse`, the F test, the log-likelihood and the information criteria (a 0 log-likelihood read as a perfect fit); these have no classical definition for an empirical-Bayes fit and are NaN.
- **`LmDynamicRegressor` reproduces `greybox::lmDynamic` (normal distribution, `bruteforce = TRUE`) (DataZooDE/anofox-statistics#63).** The candidate set now includes the intercept-only model; the pointwise information criteria are greybox's `pAIC` / `pAICc` / `pBIC`, `2k − 2·n·ℓ_t` (the log-density was not multiplied by `n`); normal candidates are fitted by least squares with `σ̂² = RSS / (n − k)` as greybox's `lm` path (ALM's ML scale before); weight smoothing follows greybox (R `lowess` on the logits of each model's weights over time, mapped back and renormalised) instead of a tricube moving average of the weights; and `result().log_likelihood` is greybox's combined log-likelihood (each model's log-likelihood weighted by its mean weight) instead of 0; the F test and the information criteria of the combined model are NaN instead of 0. Weights of exactly 0 or 1 are clamped inside (0, 1) before the logit (greybox returns NaN there). Validated in `tests/r_validation_lm_dynamic.rs` (generator `tests/r_scripts/generate_lm_dynamic_validation.R`) against greybox 2.0.9. With no regressors and an intercept the intercept-only model is fitted instead of returning an error.

### Changed

- The Python validation environment pins scikit-learn 1.9.1 and numpy 2.5.4 (was 1.5.2 / 2.1.3); the Theil-Sen, RANSAC, Passive-Aggressive and LARS fixtures are unchanged apart from their headers. `tests/r_validation_passive_aggressive.rs` adds a single in-order online pass checked against sklearn's `partial_fit`.

## [0.5.27] - 2026-10-14

### Fixed

- **LARS / LassoLars log-likelihood, information criteria and F test (DataZooDE/anofox-statistics#196).** `LarsRegressor` left `log_likelihood`, `aic`, `aicc`, `bic`, `f_statistic` and `f_pvalue` at their `RegressionResult::empty` defaults (0, and 1 for the p-value), so a LARS fit looked like the best model by AIC/BIC. They are now computed with the shared linear fit statistics and the LARS degrees of freedom already used for `adj_r_squared` / `rmse` (active coefficients plus the intercept): the Gaussian log-likelihood at the ML variance `RSS / n`, `k = df + 1`, as R's `logLik.lm` / `AIC` / `BIC` / `summary.lm`. On the full LARS path they equal `lm` on the same predictors, with and without an intercept. NaN for a perfect fit (RSS = 0), as for the other linear solvers. Validated in `tests/issue65_lars_fit_stats.rs` against R `lm`.

## [0.5.26] - 2026-10-12

### Added

- **Regularisation paths and k-fold CV for ridge / elastic net (DataZooDE/anofox-statistics#154).** New module `solvers::path`: `fit_path(x, y, lambdas, &PathOptions) -> RegPath` computes the Gaussian elastic-net path with `glmnet`'s objective and conventions (`1/(2n)·RSS + λ[(1−α)/2·‖β‖² + α‖β‖₁]`, λ on the `LambdaScaling::Glmnet` scale, unpenalised intercept, optional `standardize` with the `1/n` SD, `y` scaled internally by its `1/n` SD as `glmnet` does, coefficients on the original scale). Ridge (`α = 0`) is solved exactly from one SVD for the whole path; `α > 0` uses coordinate descent with warm starts and active-set iterations. `lambda_grid` gives `glmnet`'s default grid (`λ_max` from the data, `nlambda`, `lambda.min.ratio` 1e-4 / 0.01). `cv_path(x, y, lambdas, fold_ids, &opts) -> CvPath` gives `cv.glmnet`'s aggregates (`cv_mean`, `cv_se`, `lambda_min`, `lambda_1se`, plus indices, `df` and the full-data path) for fold ids passed in; `kfold_ids(n, k, seed)` draws deterministic folds. `RegPath` holds `lambdas`, `intercepts`, `coefficients` (`p × n_lambda`), `df` and `alpha`, with `predict`. Convenience methods `RidgeRegressor::path` / `cv_path` and `ElasticNetRegressor::path` / `cv_path`, and `FittedLars::path()` returns the LARS / LassoLars path in the same `RegPath` shape. Validated in `tests/r_validation_reg_path.rs` (generator `tests/r_scripts/generate_reg_path_validation.R`) against a reference implementation of glmnet's documented objective, grid and `cv.glmnet` / `getOptcv.glmnet` aggregates in R, and against `grpreg` for the lasso (glmnet itself was not available on the generating machine).
- **Quantile-regression inference and the τ process (DataZooDE/anofox-statistics#157).** New module `solvers::quantile_inference` with `QuantileSe::{Iid, Nid, Boot { seed, r }}`, `FittedQuantile::inference(x, y, se, level) -> QuantileInference` (estimate, std_error, statistic, p_value, conf_low, conf_high per coefficient, intercept first, plus covariance, bandwidth, sparsity and residual df), `quantile_inference(x, y, tau, with_intercept, se, level)`, `quantile_process(x, y, taus, with_intercept, Option<QuantileSe>, level)` and `hall_sheather_bandwidth`. `Iid` and `Nid` follow `quantreg::summary.rq(se = "iid" | "nid")` (Hall–Sheather bandwidth), `Boot` is the seeded `(x, y)`-pair bootstrap. t statistics and p-values use Student's t with `n − k` df as `summary.rq`; confidence limits are `estimate ± t·se`. Validated against quantreg in `tests/r_validation_quantile_inference.rs` (generator `tests/r_scripts/generate_quantile_inference_validation.R`). Weighted fits are rejected. Rank-inversion intervals (`se = "rank"`) are not implemented yet.
- `FittedQuantile::has_intercept`.
- **Predictions with standard errors (#80).** New `inference::PredictWithSe` trait, `predict_se(x, &PredictSeOptions)` returning `PredictionSe { fit, se_fit, lower, upper, .. }`, with options for the scale (`link` / `response`), interval (`confidence` / `prediction`) and level. Implemented for `FittedOls`, `FittedWls`, `FittedRidge`, `FittedPoisson`, `FittedBinomial`, `FittedNegativeBinomial`, `FittedTweedie`, `FittedGamma` and `FittedPenalizedGlm` (plus `predict_se_with_offset`); `GlmResult::predict_se(x, ..)` covers the penalized-GLM family functions (offset column honoured). Linear models match `predict.lm(se.fit = TRUE, interval = ...)`; GLMs match `predict.glm(se.fit = TRUE, type = ...)`, with intervals built on the link scale (normal quantile) and mapped through the inverse link on the response scale; prediction intervals are rejected for GLMs.
- **Partial residuals (#80).** `diagnostics::partial_residuals` (linear fits, as `residuals(type = "partial")` / `car::crPlots`), `partial_residuals_glm` (working residuals, as `residuals.glm(type = "partial")`) and `added_variable` (as `car::avPlots`, optional weights).
- **Heteroskedasticity tests (#80).** `diagnostics::breusch_pagan` (Koenker studentized or classic, as `lmtest::bptest`) and `white_test` (squares and cross-products), returning `HeteroskedasticityTest { statistic, df, p_value, method }`.
- **AFT survival curves (#80).** `FittedAft::survival_curve(x_row, times)` / `quantiles(x_row, probs)` and the same on `AftResult` (plus `AftResult::distribution` / `linear_predictor`), as `survival::psurvreg` / `predict(survreg, type = "quantile")`.
- R validation in `tests/r_validation_predict_se.rs` (generator `tests/r_scripts/generate_predict_se_validation.R`).

## [0.5.25] - 2026-10-11

### Added

- **Per-observation diagnostics for the penalized GLM engine (#78).** New `diagnostics::augment_glm(&GlmResult, y, x)` / `augment_glm_with(.., ResidualType)` for the results of `penalized_glm::fit_poisson`, `fit_binomial`, `fit_logistic`, `fit_negbinomial`, `fit_gamma` and `fit_tweedie` (same `y` / column-major `x` as the fit, offset column included), and `FittedPenalizedGlm::augment` / `augment_with` for `PenalizedGlmRegressor`. They return the same `Augment` as `diagnostics::augment` on the dedicated regressors: fitted mean, deviance (default) / Pearson / response residual, `rstandard`, `rstudent`, `hatvalues` (from the final IRLS weights), `cooks.distance` and `dffits`. Dispersion conventions follow R and the dedicated regressors: binomial / logistic / Poisson use 1 throughout, the negative binomial (fixed or estimated theta) uses 1 for `rstandard` / Cook's distance and the leave-one-out scale for `rstudent` / DFFITS (as `MASS::glm.nb`), Gamma / Tweedie / Poisson with `estimate_dispersion` (R's `quasipoisson`) use the Pearson estimate. Offsets enter only through the fitted means; constant columns (`NaN` coefficients) are excluded and the rank is found by a pivoted QR, so collinear columns count once; rows dropped for non-finite values are `NaN` in every field. A penalized fit (`lambda > 0` or a non-flat prior) is rejected with `GlmEngineError::InvalidInput`, since the R diagnostics assume the maximum-likelihood fit. The engine has no prior weights, so none are taken. Validated against R 4.6.1 `glm` / `MASS::glm.nb` / `statmod::tweedie` (Poisson, Poisson with offset, quasipoisson, binomial 0/1 and proportions, logistic, negative binomial with estimated and fixed theta, Gamma(log), Tweedie p = 1.5) and against `augment` on the dedicated regressors in `tests/r_validation_augment_penalized_glm.rs` (generator `tests/r_scripts/generate_augment_penalized_glm_validation.R`).
- `GlmResult` records what the diagnostics need: `family: PenalizedGlmFamily`, `dispersion_rule: DispersionRule`, `offset_column: Option<usize>` and `penalized: bool`.

### Changed

- `GlmResult` gains the four public fields above, so code that builds a `GlmResult` with a struct literal must set them; `GlmResult::from_engine_fit` fills them with the family's defaults (no offset, not penalized), and the `fit_*` functions set them from their options.

## [0.5.24] - 2026-10-10

### Fixed

- `FittedGlmm::n_parameters` (and so `aic` / `bic`) did not count a fixed negative-binomial θ (`nb_theta`). lme4 counts it for both `glmer.nb` and `glmer(family = negative.binomial(θ))` (`df = 4` for one fixed effect, one random intercept and θ); it is now counted in both cases. Regression test against lme4 in `tests/model_summary_glm_aft_glmm.rs`.

## [0.5.23] - 2026-10-10

### Added

- **Model description on every family-function result.** `penalized_glm::GlmResult` (returned by `fit_poisson`, `fit_binomial`, `fit_logistic`, `fit_negbinomial`, `fit_gamma`, `fit_tweedie`) and `aft::AftResult` (`fit_aft`) gain a `model_info: ModelInfo` field and implement `HasModelInfo`, so callers that use the family functions instead of the regressors can label a fit. New: `PenalizedGlmFamily::model_info`.
- **`FittedGlmm::n_parameters`, `aic`, `bic` and `n_observations`.** The parameter count is the `df` of lme4's `logLik`: fixed effects, `q(q+1)/2` covariance parameters per grouping factor, and the dispersion for Gaussian, Gamma, Tweedie and an estimated negative-binomial theta. AIC / BIC match `AIC()` / `BIC()` of `glmer(nAGQ = 0)` and `lmer` (REML), including random slopes (`tests/model_summary_glm_aft_glmm.rs`).

### Changed

- The penalized GLM engine reports `model_type` `"glm"` instead of `"penalized_glm"` (family and link unchanged), matching the anofox integration contract.
- `impl From<EngineFit> for GlmResult` is replaced by `GlmResult::from_engine_fit(fit, &family)`, because the result now records the family.

## [0.5.22] - 2026-10-09

### Fixed

- **`penalized_glm::fit_negbinomial` (estimated theta) and `fit_tweedie` were 15-40x slower than `fit_poisson` on large `n` (#72).** The `MASS::glm.nb` theta alternation now follows MASS more closely: every IRLS fit is warm-started from the previous coefficients (`etastart`), and `theta.ml` in round k uses the means of round k-1's fit (MASS updates `mu` after the theta step). On data without overdispersion, where the ML theta diverges and each `theta.ml` hits its iteration limit, the alternation therefore stops after one round instead of running all 25. The intermediate fits skip the null deviance, log-likelihood and inference, and the digamma / trigamma / lgamma sums of `theta.ml` and the convergence log-likelihood are evaluated once per distinct count instead of per row. The Tweedie series density (log-likelihood / AIC / BIC) is accumulated in one streaming pass (no per-row allocation, each term evaluated once, `lgamma(j + 1)` by recurrence). 250k rows, 2 features: negative binomial with estimated theta 8.4 s -> 0.5 s (Poisson 0.2 s), Tweedie (p = 1.5) 0.60 s -> 0.42 s with the log-likelihood 0.33 s -> 0.12 s. Results are unchanged within tolerance; on overdispersed data the estimates agree with `MASS::glm.nb` to ~1e-10 (slightly closer than before). New: `irls::fit_irls_from` (warm-started IRLS). Tests: `tests/issue72_glm_nb_tweedie_perf.rs` (MASS reference, Poisson limit, and a release-only timing gate relative to `fit_poisson`).

## [0.5.21] - 2026-10-09

### Fixed

- **OLS / WLS alias rank-deficient designs like R's `lm` for every solver (#64).** Aliasing is now decided up front, independently of the solver, with R's LINPACK `dqrdc2` rule (limited pivoting, tolerance `1e-7` relative to each column's own norm): columns are taken in their given order and a column that is numerically a linear combination of the intercept and the earlier kept columns gets a `NaN` coefficient and `aliased[j] = true`. The model is then solved on the kept columns (unit-scaled), so coefficients, intercept, standard errors, sigma, R² / adjusted R² and `n_parameters` equal R's `lm` / `lm(weights = )`. Behaviour changes: the `Svd` solver (the DuckDB default) no longer returns a minimum-norm split of collinear columns with no flag (`[x1, x1]` gave `[0.744, 0.744]`) but aliases the later column; `Qr` / WLS alias the *later* column of a collinear set (`[x1, 2*x1]` aliased column 1 before); the decision no longer depends on column scaling. New: `OlsRegressor::with_alias_tolerance` / `OlsRegressorBuilder::alias_tolerance` (same on WLS), `utils::lm_alias_columns`, `utils::LM_ALIAS_TOLERANCE`. Validated against R in `tests/issue64_lm_aliasing.rs` (exact duplicates, scaled copies, combinations of 2 and 3 columns, near-collinear just below / above the tolerance, WLS, no intercept; generator `tests/r_scripts/issue64_lm_aliasing.R`).
- WLS without intercept computed standard errors from the full (singular) `X'WX` when a column was aliased; they now come from the reduced fit, as R.
- **LARS / LassoLars fill `adj_r_squared`, `mse` and `rmse` (#65).** They were left at the `0.0` placeholders. `rmse` is the residual standard error `sqrt(RSS / (n - df))` and `adj_r_squared` uses the OLS convention, with `df` = active coefficients + intercept (Efron et al. 2004); `NaN` when `n <= df`. On the full LARS path the values equal R's `summary(lm(...))` (`tests/issue65_lars_fit_stats.rs`).

## [0.5.20] - 2026-10-08

### Added

- **`ModelInfo` / `HasModelInfo` (#66).** `anofox_regression::core::ModelInfo { model_type: &'static str, family: Option<&'static str>, link: &'static str }` and the trait `HasModelInfo { fn model_info(&self) -> ModelInfo }`, implemented by every fitted model (least squares, penalised, robust, quantile, PLS, isotonic, ALM, all GLMs incl. the penalized GLM, AFT, GLMM, ...). Stable snake_case names; `family` uses R's family names and is `None` for estimators without a likelihood.
- **`diagnostics::augment` (#66).** `augment(&fit, &x, &y, weights) -> Result<Augment, RegressionError>` and `augment_with(.., ResidualType)` return per-observation `fitted, residual, resid_type, std_residual, stud_residual, leverage, cooks_d, dffits` for OLS, WLS and the GLMs (Poisson, binomial / logistic, negative binomial, Tweedie, Gamma), matching R 4.6's `rstandard`, `rstudent`, `cooks.distance`, `dffits` and `broom::augment` (deviance residuals by default for GLMs, Pearson via `ResidualType::Pearson`). Leverage is computed from the thin QR of the IRLS-weighted design, never the n x n hat matrix (O(n p) memory). New public items: `Augment`, `Augmentable`, `AugmentSpec`, `ResidualType`.
- **Intercept inference on every GLM (#66).** Binomial, Poisson, negative binomial, Tweedie and Gamma fits now fill `intercept_t_statistic`, `intercept_p_value`, `intercept_conf_interval` and the slope `conf_interval_lower/upper` (Wald, R's `confint.default`); Tweedie / Gamma additionally report statistics and p-values (t on the residual df, as `summary.glm`), which were missing. `GlmInferenceResult` (penalized GLM) gains `intercept_std_error`, `intercept_z_value`, `intercept_p_value`, `intercept_ci_lower`, `intercept_ci_upper`; `AftInference` gains the same z / p / CI fields.
- R validation of the above and of the linear `logLik` / `AIC` / `BIC` (R's `logLik.lm`, sigma counted as a parameter) for OLS, WLS and Ridge (lambda = 0): `tests/r_validation_m1a.rs`, generator `tests/r_scripts/generate_m1a_validation.R`, VALIDATION.md section 24c.

### Fixed

- **Probit link accuracy (#66).** The binomial probit link used an Abramowitz-Stegun erf (absolute error 1.5e-7) for the normal CDF and an AS 241 quantile with a mistyped coefficient, which put probit standard errors off R in the 5th digit. Both now use `statrs` `erfc` / `erfc_inv`.
- Wald p-values of the GLMs use the survival function, so very small p-values no longer underflow to 0.

## [0.5.19] - 2026-10-08

### Fixed

- **LARS / LassoLars on exactly collinear columns (#62).** Whether a duplicate column (e.g. `x2 = 2 * x1`) entered the active set depended on last-bit rounding of its residual correlation (absolute `f64::EPSILON` threshold), so the same data returned `SingularMatrix` on aarch64 but a fit on x86_64. Linearly dependent columns are now detected up front (Gram-Schmidt in column order, scale-relative tolerance `1e-7` as R's `lm`); the later column is aliased (NaN coefficient, `aliased[j] = true`, excluded from `rank`, ignored by `predict`) and the path runs on the full-rank subset. The entry threshold is relative to the initial maximum correlation. Regression test `tests/issue62_lars_collinear.rs` covers several scalings and ±1 ulp perturbations.

## [0.5.18] - 2026-10-08

### Added

- **GLMM families (#29).** `GlmmRegressor::negative_binomial()` (NB2, log link; size θ estimated by maximising the profiled Laplace log-likelihood as `lme4::glmer.nb(nAGQ = 0)`, or fixed with `GlmmRegressorBuilder::nb_theta(θ)` as `glmer(family = MASS::negative.binomial(θ))`), `GlmmRegressor::gamma()` (log link, `glmer(family = Gamma(link = "log"))`) and `GlmmRegressor::tweedie(power)` (`1 < power < 2`, log link). The criterion is lme4's `ldL2 + ‖u‖² + aic`; Gamma / Tweedie evaluate the family log-likelihood at dispersion `deviance / n` (R's `Gamma()$aic`), report `sigma() = sqrt(pwrss / n)` and scale the fixed-effect SEs by it. `FittedGlmm::nb_theta() -> Option<f64>`. All work with `fit`, `fit_crossed` and random slopes.
- **GLMM offset (#29).** `GlmmRegressorBuilder::offset(Col<f64>)` adds a per-observation offset with coefficient 1 to the linear predictor of every family (`offset()` in an lme4 formula), for `fit` and `fit_crossed`. `FittedGlmm::predict_fixed_with_offset(x, offset)`.
- **Random slopes with several grouping factors (#29).** `GlmmRegressorBuilder::random_slopes_per_factor(Vec<Vec<usize>>)` gives each factor of `fit_crossed` its own random slopes and unstructured covariance, e.g. `(1 + x | a) + (1 | b)` → `vec![vec![0], vec![]]`. `FittedGlmm::factor_random_cov(f)`, `factor_random_sd(f)` and `factor_random_effects_matrix(f)` expose each factor's `q_f × q_f` covariance and full BLUPs; `factors()[f].sd` / `.blups` remain the first component. The plain `random_slopes` still applies to a single factor and errors with several.
- Validated against lme4 (`tests/r_validation_glmm_families.rs`, generator `tests/r_scripts/generate_glmm_families_validation.R`, VALIDATION.md §24b): fixed effects, SEs, σ, random SDs / correlation, BLUPs and logLik agree to ~1e-5–5e-5; the estimated NB θ to 3e-4. Tweedie is validated against `glmer` with a `statmod::tweedie` family whose `aic` uses `tweedie::dtweedie` (no lme4 Tweedie family; glmmTMB is a different estimator).
- Binomial offset round trip against R `glm(..., offset)` (#24).

### Changed

- The crossed / nested GLMM engine now assembles `ZΛ_θ` with a block-diagonal `Λ_θ`; results of existing intercept-only crossed fits are unchanged.

## [0.5.17] - 2026-10-08

### Behaviour changes (results now match R)

- **`penalized_glm` Poisson standard errors use dispersion 1 (#54).** The covariance was scaled by `max(1, Pearson χ²/df)` (quasi-Poisson). It is now 1, as R's `glm(family = poisson)` / `summary.glm`, so on overdispersed data SEs, z, p-values and CIs shrink by `√(χ²/df)` and `dispersion()` is `1.0`. Coefficients are unchanged. Opt into R's `quasipoisson` scale (Pearson χ²/df, not floored) with `PenalizedGlmRegressorBuilder::estimate_dispersion(true)` or `PoissonOptions { estimate_dispersion: true, .. }`.
- **`penalized_glm` Gamma log-likelihood / AIC / BIC (#55)** are evaluated at the dispersion `deviance / n`, exactly R's `logLik.glm` / `AIC` (`Gamma()$aic`). The Pearson dispersion still scales the covariance (as `summary.glm`) and is still the reported `dispersion`. Coefficients and SEs are unchanged. Tweedie keeps the Pearson-dispersion log-likelihood (R's `tweedie` family has no AIC).
- **`penalized_glm` null deviance follows R's `glm` (#56)** for every family: with an intercept and an offset it is the deviance of the intercept-only model fitted *with the offset*; without an intercept it is the deviance at `μ = linkinv(offset)` (`linkinv(0)` with no offset). Previously it was always the deviance at `μ = mean(y)`. `null_deviance` and `pseudo_r_squared` change for offset and no-intercept models; intercept models without an offset are unchanged.
- The penalized IRLS start value no longer subtracts the offset twice in the first working response (converged estimates are unchanged).

### Added

- `PenalizedGlmRegressorBuilder::estimate_dispersion(bool)`, `PoissonOptions::estimate_dispersion` (#54). Adding the field means `PoissonOptions { .. }` literals without `..Default::default()` need `estimate_dispersion: false`.
- `penalized_glm::engine::null_deviance(family, y, offset, fit_intercept, &IrlsConfig)`.
- R references for overdispersed Poisson / quasipoisson, Gamma `logLik`, and null deviance with offsets (Poisson, binomial, Gamma) and without an intercept (`tests/r_scripts/penalized_glm_aft_reference.R`).
- **Moment-heuristic AID (#57), `solvers::aid::heuristic`.** `compute_aid(&[f64], &AidOptions) -> Result<AidResult, AidError>` and `compute_aid_anomalies(&[f64], &AidOptions) -> Result<Vec<AidAnomalyFlags>, AidError>` with `AidOptions { intermittent_threshold (0.3), outlier_method: OutlierMethod::{ZScore, Iqr} }`. The fit-free classification used by the DuckDB extension, reproduced bit-for-bit (parity test against a verbatim copy of the extension's code): regular/intermittent by zero share; distribution by moments (poisson / negative_binomial / geometric for counts, lognormal / gamma / rectified_normal / normal otherwise); per-observation stockout, new-product, obsolete-product and high/low-outlier flags plus their counts. `mean`/`variance` are over all finite values (zeros included). `AidClassifier` is unchanged.
- **Empirical-Bayes shrinkage (#58), `solvers::eb_shrink`.** Moved from the extension. `eb_shrink(estimates, standard_errors, &EbShrinkOptions { method: TauMethod::{DerSimonianLaird, None}, tau_squared: Option<f64> }) -> Result<EbShrinkResult, EbShrinkError>` shrinks per-group estimates toward their precision-weighted mean under `theta_g ~ N(mu, tau²)`. Returns `mu`, `mu_se`, `tau_squared`, `i_squared`, `q`, `n_groups` and per input row `ShrunkenGroup { estimate, se, shrunken, shrunken_se, weight }` (unusable rows stay aligned as `NaN`). Matches `metafor::rma(method = "DL")` and `blup()` (`tests/r_validation_eb_shrink.rs`); `shrunken_se` is the posterior SD conditional on `mu`, and `i_squared` is always `(Q - df) / Q`. Re-exported in the prelude.

## [0.5.16] - 2026-10-08

### Added

- **Family-generic penalized GLM engine (#44), `solvers::penalized_glm`.** Moved from the DuckDB extension `anofox-statistics`. One IRLS loop over `GlmFamily` with explicit per-coefficient priors (`PriorSpec::{flat, normal(loc, scale), laplace(loc, scale)}`, optional leading intercept entry; Laplace priors use proximal coordinate descent) and a legacy uniform ridge `lambda`; covariance at the mode `VcovType::{Laplace (X'WX+P)⁻¹ (default), Sandwich, Naive (X'WX)⁻¹}`; offsets; constant/aliased columns reported as `NaN`; per-family log-likelihood, AIC, BIC, deviance, null deviance, pseudo-R², `converged` flag (last iterate on non-convergence), full `vcov` and information matrix with `matrix_parameters`. Families: Poisson (log/identity/sqrt), Binomial (logit/probit/cloglog), Logistic (with training accuracy), Negative Binomial (fixed θ or θ estimated as in `MASS::glm.nb`), Gamma (log), Tweedie (1 ≤ p ≤ 2, log). The weighted least-squares step uses a thin column-pivoted QR (no n×n allocation).
  - Builder: `PenalizedGlmRegressor::builder().family(PenalizedGlmFamily::…).priors(…).lambda(…).vcov_type(…).offset(…).compute_inference(…).build().fit(&x, &y) -> FittedPenalizedGlm` (`coefficients`, `intercept`, `deviance`, `log_likelihood`, `aic`, `bic`, `dispersion`, `theta`, `converged`, `inference`, `std_errors`, `vcov`, `separation`, `accuracy`, `predict`, `predict_with_offset`).
  - Column-major entry points for SQL/FFI front ends: `fit_poisson`, `fit_binomial`, `fit_logistic`, `fit_negbinomial`, `fit_gamma`, `fit_tweedie` with `PoissonOptions`, … `LogisticOptions` → `GlmResult { core: GlmFitResult, inference: Option<GlmInferenceResult>, log_likelihood, bic, fitted_values, valid_rows }`. Building blocks (`engine::fit`, `irls`, `laplace`, `loglik`, `penalty`, `normal_eq`, `design`) are public. Errors: `GlmEngineError` (`From` for `RegressionError`).
  - Conventions (unchanged from the extension; changed to R's in 0.5.17, #54–#56): Poisson standard errors use the Pearson dispersion floored at 1; the Gamma/Tweedie log-likelihood uses the Pearson dispersion (R's `logLik.glm` uses deviance/n); the null deviance is that of the intercept-only model without the offset.
- **Accelerated failure time survival regression (#45), `solvers::aft`.** Moved from the extension. `log T = x'β + σW` with right censoring for Weibull, exponential, log-normal and log-logistic (`AftDistribution`, with `cdf_time`, `survival_time`, `quantile_time` and standardized density/survival derivatives); Newton–Raphson on `(β, log σ)` with step halving, optional priors, Laplace/sandwich/naive covariance. `AftRegressor::builder().distribution(…).compute_inference(…).build().fit(&x, &time, &event) -> FittedAft` (`coefficients`, `intercept`, `scale`, `log_likelihood`, `null_log_likelihood`, `aic`, `bic`, `converged`, `inference`, `predict_quantile`, `predict_median`, `predict_survival`, `predict_cdf`), and the column-major `fit_aft(time, x, event, &AftOptions) -> AftResult`.
- R validation (`tests/r_validation_penalized_glm_aft.rs`, generator `tests/r_scripts/penalized_glm_aft_reference.R`): `glm`, `MASS::glm.nb`, `negative.binomial(θ)`, a posterior-mode Newton fit for the priors and all three covariance types, and `survival::survreg` for the four distributions (coefficients, scale, log-likelihoods, SEs incl. log(scale), quantiles).

## [0.5.15] - 2026-10-08

### Behaviour changes (results now match R)

- **Negative binomial standard errors use dispersion 1 (#49).** The coefficient covariance was scaled by `max(1, Pearson χ²/df)`. When χ²/df > 1 this made the SEs, z, p-values, CIs and `predict_with_se` standard errors larger than `MASS::glm.nb` (e.g. by √1.97 at θ = 10 in the regression test). The NB variance `μ + μ²/θ` already models the overdispersion, so the dispersion is now 1 for both fixed and estimated θ, as in `summary(glm.nb)`, `vcov()` and `predict.glm(se.fit = TRUE)`. `FittedNegativeBinomial::dispersion` is therefore `1.0`. Coefficients are unchanged.
- **Theil–Sen on rank-deficient designs (#50).** Linearly dependent columns (e.g. `x2 = 2·x1`, or a constant column with an intercept) are now aliased like `OlsRegressor` / R `lm`. The later redundant column gets a `NaN` coefficient and `aliased = true`, the estimator is fitted on the remaining columns, and prediction ignores the aliased columns. Previously the singular subsample systems produced coefficients of about 1e14.

### Added

- **`NegativeBinomialRegressorBuilder::estimate_dispersion(bool)` (#49).** Opt-in Pearson χ²/df dispersion (not floored), which reproduces R's `summary.glm` default for `glm(family = MASS::negative.binomial(θ))`. Default `false`.
- **GLMM Wald inference (#51).** `FittedGlmm::z_values() -> Vec<f64>` (`β̂/SE`), `FittedGlmm::p_values() -> Vec<f64>` (two-sided normal) and `FittedGlmm::conf_int(level: f64) -> (Vec<f64>, Vec<f64>)` (Wald `β̂ ∓ z·SE`) cover all fixed effects, intercept first. They match lme4's `summary()` z / `Pr(>|z|)` and `confint(method = "Wald")` for `lmer` and `glmer(nAGQ = 0)` (Poisson, binomial). For a Gaussian LMM the p-value is the asymptotic normal approximation. References: `tests/r_scripts/generate_glmm_inference_validation.R`.

## [0.5.14] - 2026-10-08

### Behaviour changes (results now match R)

These are bug fixes, but they change reported numbers. Coefficients are unchanged unless noted.

- **Log-likelihood / AIC / AICc / BIC of the linear solvers (#41).** OLS, WLS, Ridge, Huber, ElasticNet, BLS/NNLS, PLS and RLS now use the Gaussian log-likelihood at the ML variance `RSS/n` (WLS adds `½ Σ ln wᵢ`, zero weights excluded), and `k = n_params + 1` (the residual variance counts as a parameter), exactly as R's `logLik.lm` / `AIC` / `BIC`. Previously `RSS/(n−p)` and `k = n_params`.
- **No-intercept R², adjusted R² and F (#38)** use the uncentred total sum of squares `Σ w y²` and `df_total = n`, as `summary.lm`. OLS/WLS no longer clamp R².
- **Ridge inference (#37).** `std_errors` / `intercept_std_error` are the sandwich SEs `σ²·A X'X A`, `A = (X'X + λP)⁻¹` (they converge to the OLS SEs as λ → 0). For λ > 0, t-statistics, p-values and coefficient confidence intervals are `None`: a t-test on the shrunken coefficient is not valid. λ = 0 still delegates to OLS.
- **Ridge prediction/confidence intervals (#40)** use the sandwich variance factor (previously `(X'X + λP)⁻¹`, too wide).
- **Constant columns without an intercept (#39).** OLS/WLS (and Ridge with λ = 0) keep a constant column when `with_intercept(false)`. That column *is* the intercept (`lm(y ~ 0 + c + x)`). Previously it was reported as aliased while the solve still used it, so fitted values, residuals, R² and predictions were inconsistent. `y ~ 0 + c` no longer errors. Only all-zero columns are dropped. The constant-column test is scale-relative (`max − min ≤ tol·max|x|`), and `(X'X)⁻¹` is equilibrated, so non-constant columns in tiny units (e.g. `x·1e-12`) are no longer dropped.
- **RANSAC statistics (#35).** The final model is the OLS fit on the inliers. `r_squared`, `adj_r_squared`, `mse`, `rmse`, `f_statistic` and the information criteria are those of that fit; they were 0 before, and R² was over all rows. `n_observations` is now the inlier count, while residuals and fitted values still cover all rows.
- **Poisson / NegativeBinomial log-likelihood (#42)** is the full family log-likelihood at the fitted mean (R `logLik(glm)`, `MASS::glm.nb`) instead of `−deviance/2`. NegBin AIC counts θ only when it is estimated (glm.nb); a fixed θ is a known family parameter (`glm(negative.binomial(θ))`). The Poisson/binomial **GLMM** `log_likelihood()` adds the saturated term (lme4 `logLik(glmer)`).
- **Tweedie / Gamma (#43).** The dispersion is the Pearson estimate (R `summary.glm`), which changes the SEs and prediction SEs. `log_likelihood` is the family log-likelihood for powers 0, 1, 2 and 3 (Gamma as R's `Gamma()$aic`). It is NaN for other powers, which have no closed-form density, instead of a Gaussian approximation.
- **RANSAC inlier test is `|r| ≤ threshold`** (sklearn's `residuals <= residual_threshold`, previously `<`). A constant response, where the default MAD threshold is 0, now fits (intercept = y, coefficients 0) instead of failing with `ConvergenceFailed`.
- **WLS with constant non-unit weights** no longer delegates to OLS, so sigma and the log-likelihood scale with the weights as in `lm(weights = c)`.
- Earlier entries below (#30–#34): one-feature **Theil–Sen** is now the classical median of pairwise slopes, **NegBin θ** is the exact ML estimate, and **quantile regression** reaches the exact LP optimum.

### Added

- **Variance-factor / leverage API (#40).** `inference::compute_variance_factor(x, weights, with_intercept, excluded, ridge_lambda)` returns `M` with `Var(x₀'β̂) = σ²·x₀'Mx₀` for OLS `(X'X)⁻¹`, WLS `(X'WX)⁻¹` and (weighted) ridge `A X'WX A`. Also added: `compute_ridge_variance_factor`, `leverage_new(m, x_new, with_intercept)` and `intervals_from_variance_factor(...)`. `FittedOls`, `FittedWls` and `FittedRidge` gain `variance_factor()` and `leverage_new(&x_new)`. Robust and sparse estimators (Huber, Theil–Sen, quantile, elastic net, NNLS) return point predictions only from `predict_with_interval`.
- **Huber inference (#36).** `HuberRegressorBuilder::compute_inference(bool)` and `confidence_level(f64)` give SEs, t-values, p-values and CIs from the asymptotic covariance in `MASS:::summary.rlm(method = "XtX")` (matches `rlm` to 1e-6). The L2 `alpha` is not part of this covariance.
- **RANSAC (#35).** `RansacRegressorBuilder::compute_inference(bool)` and `confidence_level(f64)` provide inference for the inlier OLS fit; it is optimistic because it ignores the inlier selection. `FittedRansac::inlier_fit()` exposes that fit, and `predict_with_interval` delegates to it.
- **RLS.** `RlsRegressorBuilder::initial_p_diagonal(f64)` sets `P₀ = δ·I` (default 1e6, unchanged). The forgetting factor and δ are validated. `FittedRls::update` skips observations with non-finite values and returns NaN for them.
- `utils::detect_constant_columns_relative` and `utils::detect_zero_columns`.

### Fixed

- **O(n²) memory in QR/SVD solvers — OOM on large inputs.** Every QR-based solve called `compute_Q()`, and the SVD paths `svd()`, which materialise the full `n × n` orthogonal factor (8·n² bytes: ~3 GB at n = 20 000, ~320 GB at n = 200 000) although only its first `min(n, p)` columns are used. Affected `OlsRegressor`, `WlsRegressor`, `BlsRegressor`/NNLS, the IRLS GLMs (Poisson, Binomial, NegativeBinomial, Tweedie, Gamma), `AlmRegressor`, the SVD solvers of OLS/WLS/Ridge and `BayesianRidge`, and the condition-number diagnostics. All now use the thin factors (`compute_thin_Q()` / `thin_svd()`); results are bitwise identical. OLS with inference at n = 20k: 2.8 s / 3.1 GB → 0.007 s / 12 MB; n = 200k now takes 0.04 s / 36 MB. Regression tests in `tests/large_n_tests.rs`.
- **Negative binomial with zero counts.** The unit deviance for `y = 0` had the wrong sign (`2θ·ln(θ/(μ+θ))` instead of `2θ·ln(1 + μ/θ)`), giving negative deviances on any data containing zeros, and the θ estimator used a simplified score without the digamma terms (collapsing to the 0.01 clamp). θ is now estimated exactly as `MASS::theta.ml`; the θ/IRLS alternation also stops once the coefficients are stable (non-overdispersed data, θ → ∞). Matches `MASS::glm.nb` to ≤ 1e-6.
- **Quantile regression stopped short of the optimum.** The smoothed IRLS could stall at a non-optimal point (e.g. τ = 0.62 on small data). Its result now warm-starts exact simplex pivots on the check loss, so fits match `quantreg::rq(method = "br")` (coefficients to ~1e-13 on 100 random problems; previously up to 0.18 % above the optimal objective).
- **Theil–Sen with one feature** now returns the classical estimator (median of pairwise slopes over pairs with distinct x, intercept `median(y − b·x)`), exact at any n via slope-counting selection for large inputs. sklearn's 1-D spatial-median behaviour remains available with `univariate_pairwise(false)`.
- **GLMs with a constant feature column and an intercept** (Poisson, Binomial, NegativeBinomial, Tweedie, Gamma, LogisticRegression) now drop the column like OLS / R `glm`: NaN coefficient and inference, `aliased = true`, ignored in predictions. Previously the intercept was zeroed and the constant column absorbed it, with garbage standard errors.

## [0.5.13] - 2026-08-01

### Fixed

- **Column-pivot unpermute bug — silent wrong coefficients (#26, upstream #114).** `col_piv_qr` factorises `A·P = Q·R`, so the back-substituted solution is in pivot order and must be mapped back through the pivot permutation. Five solvers undid this incorrectly, scattering coefficients into the wrong slots whenever the pivot permutation was not an involution — i.e. **whenever the design's columns are on different scales** (very common). Invisible on well-scaled designs (identity/involutive pivot), which is why the test suites and R-validations — all well-scaled — never caught it.
  - **`BlsRegressor` / NNLS**: one-shot solve, so nothing corrected it. On a randomised multi-scale sweep, 196/200 fits were wrong (RSS ~1e8 where the exact solution has RSS 0). Now 0/200.
  - **`OlsRegressor` and `WlsRegressor`**: the same class of defect via the mirror idiom (a coefficient *gather* through the forward pivot array instead of its inverse). 186/200 multi-scale fits returned scrambled coefficients (and therefore wrong predictions, RSS ~6e8). This affected the most fundamental solvers on any differently-scaled design. Now 0/200, and validated against R `lm()` (coefficients, standard errors, and fitted values match to ~1e-9 on non-trivially-pivoted designs).
  - **`PoissonRegressor` / `BinomialRegressor` / `NegativeBinomialRegressor` / `TweedieRegressor`**: the same wrong line, but self-corrected by the IRLS fixed-point iteration; fixed for correctness and consistency.
  - `AlmRegressor` already used the correct idiom and was unaffected; ridge, elastic-net, LARS, Huber, RANSAC, Theil-Sen, quantile, PLS, and isotonic do not use pivoted QR (audited).
  - New regression coverage: 3-cycle-pivot recovery tests for OLS/WLS/NNLS, and `tests/r_validation_ols_pivot.rs` (non-trivial pivots vs `lm()`), verified to fail if the bug is reintroduced.

- **Rank detection missed exact collinearity in large-magnitude columns.** OLS/WLS determined numerical rank with an **absolute** `rank_tolerance` against the `R` diagonal, so an exact dependency among high-magnitude columns (e.g. `x3 = 2·x1` with `x1 ~ 1e3`) went undetected — the column was not aliased and the fit was subtly wrong. Rank is now assessed **relative** to the largest pivot (`rank_tolerance · |R[0,0]|`), matching the convention WLS's SVD path already used and R's behaviour. Regression-tested against `lm()` (fitted values match to ~1e-10 with the collinear column correctly aliased).

## [0.5.12] - 2026-07-31

### Added

- **Generalized linear mixed models — `GlmmRegressor` / `FittedGlmm` (#25).** A random intercept and optional random slopes over one grouping factor, and crossed / nested random-intercept factors, `g(μ_ij) = x_ij'β + z_ij'b_j` with `b_j ~ N(0, Σ)` and unstructured `Σ`. This completes Phases A–C of the mixed-model work.
  - Follows the `lme4` profiled-deviance formulation (Bates et al., JSS 2015). The random effects are `b = Λ_θ u` with a `q×q` lower-triangular relative-covariance factor `T(θ)` per group, so `Σ = σ²·T Tᵀ`. For a random intercept (`q=1`) the single scale-free ratio `θ = σ_b/σ` is profiled by golden-section search; with random slopes (`q>1`) the `q(q+1)/2` entries of `T` are profiled by Nelder–Mead, matching `lme4`'s derivative-free optimization over the Cholesky factor. The single-factor random block is eliminated by a per-group Schur complement (inverting a `q×q`).
  - `GlmmRegressor::gaussian()` fits a linear mixed model by exact profiled REML (or ML via `.reml(false)`); `GlmmRegressor::poisson()` and `GlmmRegressor::binomial()` fit non-Gaussian families by Laplace-approximate PIRLS. Because the fixed effects are part of the penalized conditional mode, the GLMM path matches `glmer(nAGQ = 0)`. `random_slopes(cols)` names the `x` columns that get a random slope alongside the intercept.
  - **Crossed / nested factors** via `fit_crossed(x, y, &[&group_a, &group_b, …])`: each factor contributes a random intercept with its own variance component (`(1|a) + (1|b)`; nesting `(1|a/b)` is expressed by passing the `a:b` interaction id). The combined random block is no longer block-diagonal, so the elimination uses a dense combined solve over all random levels; one ratio `θ_f` per factor is profiled by Nelder–Mead. `FittedGlmm::factors()` returns per-factor SDs and BLUPs; `n_factors()` reports the count.
  - `fit(x, y, group)` takes a per-observation grouping-factor id (any `usize`, compacted internally). `FittedGlmm` exposes `fixed_effects` / `intercept` / `slopes` / `std_errors`, the BLUP `random_effects` (intercept) and `random_effects_matrix` (full per-group vectors), the covariance `random_cov` / `random_sd` / `random_corr`, the scalar variance components (`theta`, `sigma`, `sd_random`, `var_random`), `deviance` / `log_likelihood`, `converged`, and `predict_fixed`.
  - Validated against `lme4` across 12 scenarios (VALIDATION.md §24, `tests/r_validation_glmm.rs`): the Gaussian LMM (intercept, intercept+slope, crossed, nested, ML, unbalanced, boundary/singular, no-intercept) matches `lmer` to ~5–6 digits on fixed effects, **fixed-effect standard errors** (vs `vcov`), residual and random-effect SDs, the intercept–slope correlation, **per-group / per-factor BLUPs** (vs `ranef`), and log-likelihood; the boundary case confirms θ → 0 collapses to OLS. The Poisson and binomial GLMMs match `glmer(nAGQ = 0)` on fixed effects, SEs, the random-intercept SD, and BLUPs. Fits are deterministic (bitwise-identical across runs).

- **`converged` on the GLM fitted structs (#23).** `FittedPoisson`, `FittedBinomial`, `FittedTweedie`, and `FittedNegativeBinomial` gain a public `converged: bool` field; `FittedGamma` gains a `converged()` accessor (it wraps Tweedie). Previously the IRLS convergence flag was computed internally but never surfaced — non-convergence could only be observed as an `Err(ConvergenceFailed)`.
  - Each builder gains `error_on_non_convergence(bool)`. The default (`true`) preserves the historical behaviour of erroring on non-convergence. Setting it to `false` returns the last iterate with `converged == false`, letting callers (e.g. the DuckDB extension's GLM aggregates) report convergence rather than collapsing it into a NULL/error.

- **`FittedGamma::predict_with_offset` (#24).** Gamma gains the per-row offset prediction helper (delegating to the wrapped Tweedie), for parity with `FittedPoisson` / `FittedBinomial` / `FittedTweedie` / `FittedNegativeBinomial`. Offset support in the GLM solvers was already complete; this closes the one prediction-API gap and adds gamma offset round-trip tests.

## [0.5.11] - 2026-07-17

### Added

- **`distributions::Normal`** re-export (from `statrs`), alongside the existing `StudentsT` / `FisherSnedecor` / `ContinuousCDF`. Lets downstream consumers compute the standard-normal quantile (`inverse_cdf`) without pulling `statrs` directly — e.g. for Q–Q theoretical quantiles.

## [0.5.10] - 2026-07-16

### Added

- **Penalized B-spline (P-spline) smoother — `PSplineRegressor` / `FittedPSpline`.** A one-dimensional GAM-style smoother (Eilers & Marx, 1996): a cubic B-spline basis over the predictor range with an order-2 difference penalty, the smoothing parameter λ selected by minimizing GCV (or fixed via `with_lambda`). This is the estimator behind ggplot2's `geom_smooth(method = "gam")`.
  - Follows the standard `Regressor` / `FittedRegressor` traits. `fit(x, y)` takes the raw predictor as a single-column design matrix (`n × 1`) and builds the basis internally; builder knobs `with_n_basis` (0 = auto ≈ n/4, clamped), `with_penalty_order` (default 2), `with_lambda` (fixed λ, else GCV).
  - `predict_with_interval` returns pointwise standard errors and t-based confidence/prediction intervals from the Bayesian posterior covariance `σ²·(BᵀB + λP)⁻¹` (the `mgcv` default), using the residual effective degrees of freedom.
  - `FittedPSpline::edf()` / `sigma2()` expose the effective degrees of freedom and residual variance.
  - Numerically exact on linear signals (the 2nd-difference penalty leaves linear functions unpenalized), partition-of-unity basis, recovers smooth trends; tests cover all four properties.

## [0.5.9] - 2026-06-24

### Added

- **Streaming OLS / Ridge from accumulated moments (#22).** Both regressors now accept the rank-`p` sufficient statistics `(n, Σx, Σy, XᵀX, Xᵀy)` instead of an explicit `N × p` design matrix, so very large panels (millions of rows) can be fit without materialising the design matrix.
  - `solvers::MomentAccumulator` — `O(p²)` storage, `O(p²)` per row, with `push_row` and `merge` for parallel/chunked accumulation, plus a `clear` method to reuse the buffer.
  - `RidgeRegressor::fit_from_moments(xtx, xty, sum_x, sum_y, n)` and `RidgeRegressor::fit_from_accumulator(&acc)`. Mathematically identical to the whole-panel `fit()` when `with_intercept = true`: builds `Gc = XᵀX − Σx Σxᵀ / n` and `cc = Xᵀy − Σx Σy / n`, solves `(Gc + λ_eff · I) β = cc` via Cholesky, recovers the intercept as `ȳ − x̄ᵀβ`. `λ_eff` honors the configured `LambdaScaling`.
  - `OlsRegressor::fit_from_moments` and `OlsRegressor::fit_from_accumulator`. Same shape with `λ = 0`. **Caveat**: from moments alone the streaming path cannot reproduce the QR-with-pivoting rank-deficiency handling — only diagonal-zero singularity is detected; near-singular cases will return a particular least-squares solution. Callers needing pivoted rank handling should use the row-based `fit()`.
  - Statistics that require per-row data (residuals, R², MSE, AIC, …) are returned as `NaN` / empty since the input rows are not retained; `predict` works normally.
  - Validated to `1e-10` against the whole-panel `fit()` for Ridge across `with_intercept`, `LambdaScaling::Raw` / `Glmnet`, and `with_intercept = false`; `1e-8` for OLS (different decomposition order). Chunked-and-merged accumulators match the single-accumulator solve to `1e-12`.
  - New `examples/streaming_ridge.rs`. README gains a streaming-fit bullet under Linear Regression.

## [0.5.8] - 2026-06-24

### Defensive

- **OLS / WLS — guard against the "garbage coefficient" failure mode (#21).** Downstream FFI consumers reported that `OlsRegressor::fit` (and `WlsRegressor::fit`) occasionally returned coefficients that were `NaN` or implausibly large (e.g. `-1.2e+149`) on otherwise valid, well-conditioned input, at a rate of roughly 1–5 % per fit. The same reproducer running in pure Rust under ASan and MSan over **800 000 fits** (single- and multi-threaded) did not trigger the failure, so the root cause appears to live in the FFI call context rather than in this crate, but the silent corruption was easy to miss downstream. This release adds two checks in the shared fit path:
  - **Input guard**: `fit` now returns `RegressionError::NumericalError` immediately if any cell of `X` or `y` is non-finite, rather than letting the NaN propagate into the centering and decomposition steps.
  - **Output guard**: after the QR / SVD / Cholesky branch, any active (non-aliased) coefficient that is non-finite or `|β| > 1e120` is rejected with a `NumericalError` mentioning #21. Aliased columns intentionally remain `NaN` and continue to pass.

  These do not fix the underlying intermittent corruption — they turn the silent symptom into a loud, catchable error.

## [0.5.7] - 2026-06-06

### Fixed

- `src/solvers/huber.rs` and `src/solvers/ransac.rs` used `usize::is_multiple_of`, which is only stable from Rust 1.87. Downstream consumers that build on older toolchains (notably the `wasm32-unknown-emscripten` pipeline in [DataZooDE/anofox-statistics](https://github.com/DataZooDE/anofox-statistics), which is pinned to Rust 1.86) could not compile 0.5.6. Both call sites now use the equivalent `n % 2 == 0`, with `#[allow(clippy::manual_is_multiple_of)]` so clippy auto-fix can't re-introduce the unstable method. (#20)

The compiled output is bit-identical to 0.5.6.

## [0.5.6] - 2026-06-04

### Fixed

- Broken intra-doc link in `gamma.rs` that caused the `docs` job to fail under `RUSTDOCFLAGS=-D warnings`. As a downstream effect this had also skipped the `Publish to crates.io` job for v0.5.5, leaving crates.io at 0.5.4 while GitHub and npm were at 0.5.5.

### Documentation

- README updated to list the six new estimators introduced in 0.5.5 (TheilSen, RANSAC, BayesianRidge, ARD, LARS / LassoLars, PassiveAggressive, Gamma) and the scikit-learn validation pipeline. Validation table split into R-validated and sklearn-validated sections; test count updated to 499+.

There are no API changes vs 0.5.5.

## [0.5.5] - 2026-06-03

### Added

- **GammaRegressor** (#16): sklearn-style convenience wrapper for `TweedieRegressor` with `var_power = 2`, log link. Validated against R `glm(family = Gamma(link = "log"))` to `1e-4` on coefficients.
- **TheilSenRegressor** (#15): robust regression via spatial (L1 geometric) median of OLS-on-subsample coefficient vectors, with exhaustive enumeration when `C(n, n_subsamples) ≤ max_subpopulation`. Vardi-Zhang modified Weiszfeld iteration; matches `sklearn.linear_model.TheilSenRegressor` to `1e-10` (univariate) / `5e-3` (multivariate).
- **RansacRegressor** (#14): random-sample-consensus regression with OLS base estimator, Fischler-Bolles stop-probability bound on `max_trials`, and inlier-mask reporting. Validated against `sklearn.linear_model.RANSACRegressor` to `1e-9` on fixtures with a unique consensus set.
- **PassiveAggressiveRegressor** (#18): online learning with PA-I (`epsilon_insensitive`) and PA-II (`squared_epsilon_insensitive`) loss variants; `partial_fit` for streaming use. Validated against `sklearn.linear_model.PassiveAggressiveRegressor` with `shuffle=False` to `2e-2`.
- **LarsRegressor / LassoLars** (#17): Efron-Hastie-Johnstone-Tibshirani Least Angle Regression solver with full coefficient path; LassoLars variant with drop-on-zero-crossing and linear-interpolation termination at the requested alpha. Validated against `sklearn.linear_model.Lars` / `LassoLars` to `1e-6` / `5e-3`.
- **BayesianRidge** (#13): SVD-based evidence maximisation with closed-form α / λ updates. Validated against `sklearn.linear_model.BayesianRidge` to `5e-3` on coefficients.
- **ArdRegression** (#13): per-feature precisions with feature pruning via `threshold_lambda`. Validated against `sklearn.linear_model.ARDRegression` to `5e-2`.
- WASM bindings for all of the above are exported from `@sipemu/anofox-regression`.
- Python-based reproducible validation oracle under `validation/python/` (pinned `scikit-learn==1.5.2`, `numpy==2.1.3`) with per-estimator fixture generators in `validation/python/generate_*.py` writing into `tests/fixtures/`.
- VALIDATION.md updated with sections 17–23 covering the new estimators and a "Regenerate Python (sklearn) references" subsection.

## [0.5.4] - 2026-03-28

### Added

- HuberRegressor: robust regression with Huber loss (#12)
  - IRLS algorithm with adaptive Huber weights
  - MAD-based scale estimation
  - L2 regularization via `alpha` parameter
  - Outlier detection via `outliers()` and `n_outliers()`
  - Validated against R's `MASS::rlm(method="M", psi=psi.huber)`
- LogisticRegression: binary classifier with sklearn-like API (#11)
  - Wraps `BinomialRegressor` with logit link
  - `predict()` returns class labels (0/1), `predict_proba()` returns probabilities
  - `decision_function()` returns log-odds, `score()` returns accuracy
  - L2 regularization via `Penalty::L2(lambda)` or `.c(C)` (sklearn convention)
  - Validated against R's `glm(family=binomial(link="logit"))`

## [0.5.3] - 2026-03-28

### Changed

- AID classifier performance optimization delivering 50x–800x speedups
  - Replaced full ALM/IRLS optimizer with closed-form MLE for intercept-only models
  - Inline O(1) log-likelihood formulas using precomputed sums (eliminates per-distribution O(n) passes)
  - Conditional precomputation: `ln_gamma` sums only for count data, `ln()` sums only for continuous data
  - One-pass variance via `Σy² - (Σy)²/n`, eliminating a separate data pass
  - Lookup table for `ln(k!)` covering k=0..256, avoiding expensive `ln_gamma` calls for typical demand data
  - Zero short-circuit in Negative Binomial log-likelihood for intermittent demand patterns
  - Stack-allocated candidate array instead of heap-allocated `Vec`

### Added

- Criterion benchmark suite for AID classifier (`benches/aid_benchmark.rs`)

## [0.5.1] - 2026-01-16

### Added

- Penalized IRLS (L2 regularization) for GLMs: Poisson, Binomial, Tweedie, Negative Binomial
  - New `lambda()` builder method for all GLM regressors
  - Solves `(X'WX + λI)β = X'Wz` with unpenalized intercept
- Condition number diagnostics for regression matrices
  - `condition_number()` - compute κ(X) = σ_max / σ_min
  - `condition_diagnostic()` - comprehensive analysis with singular values and condition indices
  - `classify_condition_number()` - severity classification (WellConditioned, Moderate, High, Severe)
  - `variance_decomposition_proportions()` - Belsley-Kuh-Welsch collinearity analysis
- Quasi-separation detection for binary GLMs
  - `check_binary_separation()` - detect Complete, Quasi, MonotonicResponse separation
  - `check_count_sparsity()` - detect sparse count data issues

### Fixed

- Quasi-separation MonotonicResponse detection now requires repeated x-values
  - Prevents false positives when each x value has only one observation

### Changed

- npm publishing workflow updated to match anofox-forecast pattern
  - Uses curl installer for wasm-pack
  - Builds WASM to js/ directory with custom package.json preservation
  - Added test-js job for JavaScript tests
  - npm upgrade for OIDC provenance support

## [0.4.0] - 2025-12-12

### Added

- R validation tests for all 24 ALM extended distributions
- R validation tests for AID (Automatic Identification of Demand)
- CI/CD workflow for automatic publishing to crates.io on release

### Changed

- Elastic Net now uses argmin L-BFGS optimization instead of custom coordinate descent
- Unified optimization framework: both ALM and Elastic Net now use argmin

### Fixed

- All 24 ALM distributions validated against R greybox package:
  - AsymmetricLaplace: Fixed scale estimation using weighted absolute residuals
  - Binomial: Fixed to use proportions (0-1) as expected by likelihood function
  - NegativeBinomial: Uses size parameter for dispersion modeling
  - InverseGaussian: Validated with log link
  - FoldedNormal: Fixed scale estimation using second-moment method
  - S distribution: Added starting points with negative intercepts
  - Beta: Fixed precision parameter estimation using method-of-moments
  - BoxCoxNormal: Validated with L-BFGS optimization
  - CumulativeLogistic/CumulativeNormal: Implemented Bernoulli log-likelihood
  - Geometric: Changed to Log link, modeling mean λ = (1-p)/p
  - LogitNormal: Changed to Identity link on logit-scale location parameter
  - LogLaplace/LogGeneralisedNormal: Fixed scale estimation for log-space residuals

## [0.3.2] - 2025-12-10

### Added

- Loss functions: MAE, MSE, RMSE, MAPE, sMAPE, MASE, pinball loss
- Dynamic Linear Model (LmDynamic) with time-varying coefficients
- LOWESS (Locally Weighted Scatterplot Smoothing)
- AID (Automatic Identification of Demand) classifier for demand pattern classification
- Bounded Least Squares (BLS) regression with parameter constraints
- Prediction intervals for linear models
- Comprehensive tests for leverage, family distributions, and dynamic regression

### Fixed

- OLS example to use independent predictors

## [0.3.1] - 2025-12-10

### Changed

- Updated `faer` dependency to disable default features for WASM compatibility
  - Removes `spindle` threadpool dependency (which depends on `atomic-wait`)
  - Enables only `std` and `linalg` features
  - Linear algebra functionality remains intact; parallel operations disabled (not supported on WASM)

## [0.3.0] - 2025-12-09

### Added

- Augmented Linear Model (ALM) from greybox R package with 24+ distributions:
  - Continuous: Normal, Laplace, Student's t, Logistic, Asymmetric Laplace, Generalised Normal, S
  - Positive continuous: Log-Normal, Log-Laplace, Log-S, Log-Generalised Normal, Gamma, Inverse Gaussian, Exponential, Folded Normal, Rectified Normal
  - Bounded (0,1): Beta, Logit-Normal
  - Count data: Poisson, Negative Binomial, Binomial, Geometric
  - Ordinal: Cumulative Logistic, Cumulative Normal
  - Transformed: Box-Cox Normal
- Link functions: Identity, Log, Logit, Probit, Inverse, Sqrt, Complementary log-log
- Comprehensive R validation tests for GLM, WLS, Ridge, Elastic Net, Tweedie, and ALM

### Changed

- Updated `faer` from 0.20 to 0.23
- Updated `statrs` from 0.17 to 0.18
- Updated `getrandom` from 0.2 to 0.3 (WASM target)

### Fixed

- Prediction interval calculation for perfect fit scenarios (MSE = 0)
- Binomial deviance residuals numerical stability

## [0.2.0] - 2025-12-08

### Added

- Poisson GLM with log and identity links
- Negative Binomial GLM with theta estimation
- Binomial GLM with logit, probit, and cloglog links
- Tweedie GLM for compound Poisson-Gamma distributions
- GLM residuals: Pearson, deviance, and working residuals
- Prediction with standard errors for GLM models
- Offset support for exposure adjustment in count models

## [0.1.0] - 2025-12-08

### Added

- Ordinary Least Squares (OLS) regression with full inference
- Weighted Least Squares (WLS) regression
- Ridge Regression with L2 regularization
- Elastic Net with L1 + L2 regularization
- Recursive Least Squares (RLS) with online learning support
- Coefficient standard errors, t-statistics, and p-values
- Confidence and prediction intervals
- Model diagnostics: R², Adjusted R², RMSE, F-statistic, AIC, AICc, BIC
- Residual analysis: standardized and studentized residuals
- Leverage and influence measures: Cook's distance, DFFITS
- Variance Inflation Factor (VIF) for multicollinearity detection
- Automatic handling of collinear and constant columns
