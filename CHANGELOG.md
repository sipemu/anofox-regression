# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
