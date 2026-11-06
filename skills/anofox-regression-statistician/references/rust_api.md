# anofox-regression (Rust) — compact API reference

Crate `anofox-regression` **0.12.0** (MIT, MSRV 1.87). Every name here was checked against the 0.12.0
source. The snippets were built and run on 0.12.0 with faer 0.23.

## Contents
1. [Cargo setup, types, prelude](#1-cargo-setup-types-prelude)
2. [Common traits and patterns](#2-common-traits-and-patterns)
3. [Estimators](#3-estimators)
   - 3.1 [Linear: OLS, WLS, Ridge, ElasticNet, LARS, streaming](#31-linear-ols-wls-ridge-elasticnet-lars-streaming)
   - 3.2 [Robust / constrained / online: Huber, Theil-Sen, RANSAC, BLS, RLS, PA](#32-robust--constrained--online)
   - 3.3 [Bayesian, PLS, LmDynamic](#33-bayesian-pls-lmdynamic)
   - 3.4 [Smoothers and shape constraints: PSpline, LOWESS, isotonic](#34-smoothers-and-shape-constraints)
   - 3.5 [Quantile regression and inference](#35-quantile-regression-and-inference)
   - 3.6 [GLMs: logistic, binomial, Poisson, NB, Tweedie, Gamma](#36-glms)
   - 3.7 [ALM (greybox)](#37-alm-augmented-linear-model)
   - 3.8 [Penalized GLM engine (priors, bayesglm-like)](#38-penalized-glm-engine)
   - 3.9 [GLMM (lme4-like)](#39-glmm)
   - 3.10 [AFT survival](#310-aft-survival-survreg)
   - 3.11 [Mediation, EB shrinkage, AID](#311-mediation-eb-shrinkage-aid)
4. [Diagnostics and inference modules](#4-diagnostics-and-inference)
5. [Regularisation paths and cross-validation](#5-regularisation-paths-and-cross-validation)
6. [R / sklearn equivalence table](#6-r--sklearn-equivalence)
7. [Gotchas and limitations](#7-gotchas-and-limitations)
8. [Splines and design](#8-splines-and-design)
9. [Term-aware inference: Wald chunk tests, anova.rms, LR/score tests, fastbw, approximation, pooling, robust covariance](#9-term-aware-inference-wald-chunk-tests-anovarms-lrscore-tests-fastbw-approximation-pooling-robust-covariance)
10. [Effects: datadist, summary, contrasts, Predict, nomogram](#10-effects-datadist-summary-contrasts-predict-nomogram)
11. [Validation: optimism bootstrap, CV, .632, calibration, bootcov](#11-validation-optimism-bootstrap-cv-632-calibration-bootcov)
12. [Accuracy indexes and separation](#12-accuracy-indexes-and-separation)
13. [Penalized maximum likelihood: pentrace](#13-penalized-maximum-likelihood-pentrace)
14. [Survival: Kaplan–Meier, log-rank, Cox (`survival`, `survival::cox`)](#14-survival-kaplanmeier-log-rank-cox-survival-survivalcox)
15. [Ordinal regression (`solvers::ordinal`, `solvers::continuation_ratio`)](#15-ordinal-regression-solversordinal-solverscontinuation_ratio)
16. [GLS for serial correlation (`solvers::gls`), R `nlme::gls`](#16-gls-for-serial-correlation-solversgls-r-nlmegls)
17. [Multinomial / penalized logistic (sklearn objective) vs `LogisticRegressor`](#17-multinomial--penalized-logistic-sklearn-objective-vs-logisticregressor)
18. [Response transformations (`transform`)](#18-response-transformations-transform)
19. [Data reduction, blinded to Y (`reduction`), Hmisc equivalents](#19-data-reduction-blinded-to-y-reduction-hmisc-equivalents)
20. [Multiple imputation: `imputation::areg_impute` (Hmisc `aregImpute`)](#20-multiple-imputation-imputationareg_impute-hmisc-aregimpute)
21. [GLMM changes since 0.5.35](#21-glmm-changes-since-0535)

---

## 1. Cargo setup, types, prelude

```toml
[dependencies]
anofox-regression = "0.12"         # 0.12.0 at time of writing
faer = "0.23"                      # MUST match the crate's faer (also re-exported as anofox_regression::faer)
```
- Optional features (all off by default): `parallel` (rayon; adds `validation::predab_resample_par`
  and `bootcov_par`, bit-identical to the serial versions) and `serde` (`Serialize`/`Deserialize`
  for every fitted model, `RegressionResult`, options, the spline bases and the survival results; a
  loaded model predicts and gives inference bit-identically; NaN/±∞ are encoded as strings, so read
  JSON with serde_json's `float_roundtrip`; format policy in `STABILITY.md`). Dependencies: `faer 0.23`
  (std, linalg), `statrs 0.18` (internal only: no statrs type is in the API any more, and the old
  `distributions` module is gone), `basin 3.1` (replaced argmin) and `thiserror 2`.
- Data types: design `faer::Mat<f64>` (n × p, **no intercept column**), response `faer::Col<f64>`.
  Build them with `Mat::from_fn(n, p, |i, j| ..)` and `Col::from_fn(n, |i| ..)`, and index them as
  `x[(i, j)]` and `y[i]`. Infinite values are an error in every estimator (#181).
- `use anofox_regression::prelude::*;` brings in:
  - estimators: `OlsRegressor, WlsRegressor, RidgeRegressor, ElasticNetRegressor, HuberRegressor,
    BlsRegressor, RlsRegressor, PlsRegressor, PSplineRegressor, QuantileRegressor,
    IsotonicRegressor, OutOfBounds, LogisticRegressor, Penalty, BinomialRegressor, PoissonRegressor,
    NegativeBinomialRegressor, TweedieRegressor, GammaRegressor, AlmRegressor, AlmDistribution,
    LinkFunction, GlmmRegressor, GlmmRegressorBuilder, GlmmDdf, GlmmInference, FactorSummary,
    AftRegressor, AftDistribution, PenalizedGlmRegressor, PenalizedGlmFamily, PriorSpec, VcovType,
    eb_shrink, EbShrinkOptions, EbShrinkResult, TauMethod`;
  - fitted types: `FittedAlm, FittedBinomial, FittedGamma, FittedGlmm, FittedHuber, FittedIsotonic,
    FittedLogistic, FittedNegativeBinomial, FittedPSpline, FittedPls, FittedPoisson, FittedQuantile,
    FittedTweedie, FittedAft, FittedPenalizedGlm` (**not** `FittedOls/Wls/Ridge/ElasticNet`, which
    you rarely need to name);
  - traits: `Regressor, FittedRegressor, IterativeFit, HasModelInfo, PredictWithSe, Augmentable`;
  - core: `IntervalType, PredictionType, PredictionResult, RegressionResult, RegressionOptions,
    RegressionOptionsBuilder, LambdaScaling, SolverType, NaAction, NaError, NaHandler, NaInfo,
    ModelInfo`, families `BinomialFamily, BinomialLink, GlmFamily, NegativeBinomialFamily,
    PoissonFamily, PoissonLink, TweedieFamily`;
  - inference: `HcType, HcInference, nested_f_test, nested_f_test_indices, NestedFTest,
    NestedFTestOptions, NestedFTestError, anova_table, AnovaTable, AnovaTableOptions, AnovaTerm,
    AnovaType, AnovaTableError, PredictSeOptions`;
  - diagnostics: `augment, augment_with, Augment, ResidualType, compute_leverage, cooks_distance,
    studentized_residuals, standardized_residuals, variance_inflation_factor, condition_number,
    condition_diagnostic, classify_condition_number, variance_decomposition_proportions,
    ConditionDiagnostic, ConditionSeverity, check_binary_separation, check_count_sparsity,
    detect_separation, LinearSeparation, SeparationCheck, SeparationType, deviance_residuals,
    pearson_residuals, working_residuals, influential_cooks, high_leverage_points`.
- **Not in the prelude.** From `anofox_regression::solvers::{..}`: `TheilSenRegressor,
  RansacRegressor, LarsRegressor, LarsMethod, BayesianRidgeRegressor, ArdRegressor,
  PassiveAggressiveRegressor, PaLoss, PaState, LmDynamicRegressor, InformationCriterion,
  CandidateModel, MomentAccumulator, SufficientStats, AidClassifier, HuberScale, TheilSenCiMethod,
  AlmLoss, GroupedRegressor, SurvivalRegressor, FittedOls`, and the `*RegressorBuilder` types
  (e.g. `ElasticNetRegressorBuilder`, `QuantileRegressorBuilder`; you normally get them from
  `::builder()`). Modules: `solvers::{path, cv, lowess, quantile_inference, mediation, aft,
  penalized_glm, eb_shrink, lm_dynamic, aid}`, `diagnostics::{breusch_pagan, white_test, press,
  partial_residuals, added_variable, generalized_vif, dffits, externally_studentized_residuals,
  augment_glm, accuracy}` and `inference::{compute_hc_inference, compute_hc_standard_errors,
  PredictionSe, PredictSeError, likelihood, cluster, anova, ..}`.
- **Renamed in 0.9 (#180).** `LogisticRegression` → `LogisticRegressor`, `BayesianRidge` →
  `BayesianRidgeRegressor`, `ArdRegression` → `ArdRegressor` (old names are `#[deprecated]` aliases;
  use the new ones), `lm_dynamic::ModelSpec` → `CandidateModel`. The penalized-GLM internals
  (`engine`, `irls`, `laplace`, ...) are crate-private.
- Errors: `RegressionError` (`#[non_exhaustive]`: `DimensionMismatch{x_rows, y_len},
  InsufficientObservations{needed, got}, SingularMatrix, AllFeaturesConstant, InvalidOptions(..),
  ConvergenceFailed{iterations}, InvalidWeights, InvalidInput(String), InvalidParameter{name,
  reason}, NumericalError(String)`). The penalized GLM, AFT, `augment_glm` and partial-residual
  functions now also return `RegressionError`. Other modules have their own error enums, and the
  crate-level `anofox_regression::Error` / `anofox_regression::Result<T>` has a `From` for every one,
  so `fn f() -> anofox_regression::Result<_>` can `?` anything (or use `Box<dyn Error>`).
- **Extensible types (#177).** Public enums (families, links, methods) and all-public-field structs
  (results, **options**) are `#[non_exhaustive]`: `match` needs a `_` arm, and option structs cannot
  be built with a struct literal. Use `Default` plus the per-field fluent setters, e.g.
  `LowessOptions::default().frac(0.5)`, `MediationOptions::default().method(MediationMethod::Sobel)`,
  `PathOptions::elastic_net(0.5).with_intercept(true)`. Result fields remain readable.
- Older doctests write `use statistics::...`. That was the crate's old name: use `anofox_regression::`.

## 2. Common traits and patterns

```rust
pub trait Regressor { type Fitted: FittedRegressor;
    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError>; }
pub trait FittedRegressor {
    fn predict(&self, x: &Mat<f64>) -> Col<f64>;
    fn result(&self) -> &RegressionResult;
    fn coefficients(&self) -> &Col<f64>;  fn intercept(&self) -> Option<f64>;
    fn r_squared(&self) -> f64;           fn score(&self, x, y) -> f64;   // R² on new data
    fn predict_with_interval(&self, x, interval: Option<IntervalType>, level: f64) -> PredictionResult;
}
pub trait IterativeFit { fn converged(&self) -> bool; fn iterations(&self) -> usize;
    fn ensure_converged(self) -> Result<Self, RegressionError>; }      // Err(ConvergenceFailed)
pub trait GroupedRegressor  { fn fit_grouped(&self, x, y, groups: &[usize]) -> Result<Fitted, _>; }
pub trait SurvivalRegressor { fn fit_survival(&self, x, time: &Col<f64>, event: &[bool]) -> Result<Fitted, _>; }
```
- The usual flow is `Estimator::builder().opt(..).build().fit(&x, &y)?`. `Regressor` must be in scope
  for `.fit` (the prelude includes it).
- `Regressor` is now also implemented by `LogisticRegressor` (`FittedLogistic: FittedRegressor`;
  `predict` gives **0/1 labels**, `result()` is the binomial table). It is **not** implemented by
  `GlmmRegressor` (`fit(x, y, &group)` or `GroupedRegressor::fit_grouped`), `AftRegressor`
  (`fit(x, time, &event)` or `SurvivalRegressor::fit_survival`) or `PenalizedGlmRegressor` (inherent
  `fit`). `FittedGlmm`, `FittedAft` and `FittedPenalizedGlm` do not implement `FittedRegressor`.
- **`IterativeFit`** (prelude) replaces the old inherent `converged()`/`iterations()`/`n_iter()`
  methods. It is implemented by the IRLS GLMs, logistic, GLMM, ordinal, continuation ratio, penalized
  GLM, AFT, Huber, quantile, Bayesian ridge, ARD, passive-aggressive, ALM, GLS and Theil–Sen.
  `fit(..)?.ensure_converged()?` is the uniform strict policy.
- Builder options shared by most estimators: `with_intercept(bool)` (default true; **now the name
  everywhere**, including LARS, Bayesian ridge, ARD, `PathOptions`, `AftOptions` and the penalized
  GLM; `fit_intercept` is gone), `compute_inference(bool)`, `confidence_level(f64)` (0.95),
  `max_iterations`, `tolerance`. Theil-Sen, passive-aggressive, BayesianRidge and ARD still use
  `max_iter`. The `RegressionOptions`-based estimators (OLS, WLS, ridge, elastic net, the IRLS GLMs)
  also take `na_action(NaAction)`.
- **Accessor convention (#179):** vectors are returned as faer columns: stored values as
  `ColRef<'_, f64>`, computed ones as owned `Col<f64>` (use `.iter()`, `[i]`, `.nrows()`). Some
  older accessors still return slices or `Vec` (e.g. `FittedPenalizedGlm::std_errors() ->
  Option<&[f64]>`, `FittedGlmm::vcov() -> &[Vec<f64>]`, `FittedHuber::outliers() -> &[bool]`).
- Low-level configuration: `RegressionOptions::builder()...build()?` (or `RegressionOptions::default()`
  plus setters) and `XxxRegressor::new(options)`. Defaults are `with_intercept=true,
  compute_inference=true, confidence_level=0.95, lambda=0, alpha=0, lambda_scaling=Raw,
  forgetting_factor=1, solver=Qr, max_iterations=1000, tolerance=1e-8` (was 1e-6; R's
  `glm.control(epsilon)`, #206), `rank_tolerance=1e-10, na_action=Omit`. `SolverType::{Qr (pivoted,
  default), Svd, Cholesky}`.

### RegressionResult (exact public fields)
`coefficients: Col<f64>` (slopes only; aliased entries are NaN), `intercept: Option<f64>`,
`residuals`, `fitted_values`, `rank`, `n_parameters` (includes the intercept), `n_observations`,
`aliased: Vec<bool>`, `column_permutation: Option<Vec<usize>>`, `rank_tolerance`, `r_squared`,
`adj_r_squared`, `rmse`, `mse`, `f_statistic`, `f_pvalue`, `aic`, `aicc`, `bic`, `log_likelihood`,
**inference** (`Option`, present when `compute_inference` is on and the model supports it):
`std_errors`, `t_statistics`, `p_values`, `conf_interval_lower`, `conf_interval_upper`
(each `Option<Col<f64>>`), `intercept_std_error`, `intercept_t_statistic`, `intercept_p_value`
(`Option<f64>`), `intercept_conf_interval: Option<(f64, f64)>`, `confidence_level`,
`na_info: Option<NaInfo>`.
Methods: `residual_df()`, `model_df()`, `n_active_coefficients()`, `is_valid()`, `has_aliased()`,
`get_coefficient(i)`, `tss()`, `rss()`, `ess()`, `residuals_expanded()`, `fitted_expanded()`,
`had_na_removed()`, `n_na_removed()`, `n_original_observations()` (`std_errors_expanded()` is
deprecated and just returns `std_errors`).
For GLMs the "t_statistics" are Wald z (or t) values. `r_squared` is the deviance pseudo-R², and
`f_statistic` is a deviance-based F. p-values no longer underflow to 0 below 1e-16 (#200).

### Prediction
- `PredictionResult { fit, lower, upper, se }` (all `Col<f64>`). `IntervalType::{Confidence,
  Prediction}`. `PredictionType::{Response, Link}`.
- `PredictWithSe` (R's `predict(..., se.fit=TRUE)`) is implemented for `FittedOls, FittedWls,
  FittedRidge, FittedPoisson, FittedBinomial, FittedNegativeBinomial, FittedTweedie, FittedGamma` and
  `FittedPenalizedGlm`:
  ```rust
  let p = fit.predict_se(&x_new, &PredictSeOptions::new().confidence().level(0.9))?; // .link()/.response()/.prediction()
  // PredictionSe { fit, se_fit, lower, upper, scale, interval, level, df, residual_scale }
  ```
  A prediction interval for a GLM returns `PredictSeError::PredictionIntervalUnsupported`.
- Poisson, binomial, NB and Tweedie also have an inherent `predict_with_se(&x, PredictionType,
  Option<IntervalType>, level) -> PredictionResult` (for Gamma use `.inner().predict_with_se(..)` or
  the trait). With `PredictionType::Response` the CI is computed on the link scale and back-transformed.
- Coefficient covariance: `vcov() -> Option<Mat<f64>>`, **intercept first**, on `FittedOls`,
  `FittedWls`, `FittedRidge` (σ̂²·M) and on every IRLS GLM fit and `FittedLogistic` (φ̂·(XᵀWX)⁻¹,
  R's `vcov(glm)`; `None` without inference). It feeds `inference::wald_test`/`anova` directly.

### NaAction and NaHandler
`NaAction::{Omit (default), Exclude, Fail, Pass}` (R's na.omit/na.exclude/na.fail/na.pass). **Since
0.8 the solvers apply it** (#176, #205) for the `RegressionOptions`-based estimators: OLS, WLS,
ridge, elastic net, BLS, RLS, Poisson, NB, Tweedie/Gamma, binomial, ALM and lmDynamic (set it with
`.na_action(..)` on the builder). By default NaN rows are dropped; `Exclude` makes
`residuals_expanded()`/`fitted_expanded()` NaN-pad back to the input length; `Fail` errors. Rows
cannot be dropped when weights or an offset are set (NaN is then an error). Estimators without an
`NaAction` (Huber, quantile, Theil-Sen, LARS, ...) **error on NaN**; the penalized GLM, AFT, mediation
and LOWESS drop non-finite rows themselves. Manual pre-processing still works:
`NaHandler::process(&x, &y, NaAction::Exclude)?` gives `x_clean`, `y_clean` and `na_info: NaInfo{
n_original, n_clean, na_mask, kept_indices, n_removed, action}`; `na_info.expand(&col)` pads back.
`NaHandler::process_with_weights` also exists.

### ModelInfo
`HasModelInfo::model_info()` returns `ModelInfo { model_type: &'static str, family: Option<&str>,
link: &str }`. It is implemented by every fitted model.

### augment (broom::augment)
`augment(&fit, &x, &y, weights: Option<&Col<f64>>) -> Result<Augment, _>`, and
`augment_with(.., ResidualType::{Response, Pearson, Deviance})`. These work for any `Augmentable`:
`FittedOls, FittedWls, FittedPoisson, FittedBinomial, FittedLogistic, FittedNegativeBinomial,
FittedTweedie, FittedGamma` (weighted GLM fits use their prior weights). `Augment` fields: `fitted,
residual, resid_type, std_residual` (rstandard), `stud_residual` (rstudent), `leverage`
(hatvalues), `cooks_d`, `dffits`, `loo_residual` and `loo_fitted` (exact; NaN for GLMs), and
`approx_loo_fitted`/`approx_loo_residual` (one-step Pregibon LOO for GLMs; NaN for linear models).
`press(&fit, &x, &y, None)` returns `Press{press, press_rmse, loo_r_squared, n}`. Ridge:
`FittedRidge::loo(&x, &y) -> RidgeLoo{leverage, loo_residual, loo_fitted, press}`. For engine GLMs
use `FittedPenalizedGlm::augment(&x,&y)` or `diagnostics::augment_glm(..)`; penalized fits (λ > 0
or a non-flat prior) are rejected with `InvalidInput`.

## 3. Estimators

### 3.1 Linear: OLS, WLS, Ridge, ElasticNet, LARS, streaming

| Estimator | Builder options (beyond common) | Extras on the fitted model |
|---|---|---|
| `OlsRegressor::builder()` | `rank_tolerance, solve_method(SolverType), alias_tolerance, na_action` | `FittedOls`: `vcov()`, `hc_vcov(&x, HcType)` (0.12, `vcovHC` matrix), `hc_standard_errors(&x, HcType)`, `hc_inference(&x, HcType)`, `cluster_vcov(&x, &[&cl], &ClusterOptions)`, `variance_factor()`, `leverage_new(&x_new)`, `options()`; `OlsRegressor::detect_constant_columns(&x)`, `OlsRegressor::is_full_rank(&x)` |
| `WlsRegressor::builder()` | `weights(Col)` (required), `solve_method, alias_tolerance, na_action` | `FittedWls`: `weights()`, `vcov()`, `hc_vcov(&x, HcType)` (0.12, weighted `vcovHC`), `cluster_vcov(..)`, `variance_factor()`, `leverage_new()`; zero-weight rows do not count in the df (#195) |
| `RidgeRegressor::builder()` | `lambda(f64)`, `lambda_scaling(LambdaScaling::{Raw, Glmnet})`, `solve_method`, `conditioning_aware(lambda_small, cond_threshold)` | `FittedRidge`: `lambda()` (the λ used), `vcov()`, `variance_factor()`, `leverage_new()`, `loo(&x,&y)`; inference and intervals are computed |
| `ElasticNetRegressor::builder()` | `lambda, alpha` (1 = lasso), `lambda_scaling, max_iterations, tolerance` (glmnet's thresh) | `FittedElasticNet`: `lambda(), alpha(), n_nonzero()`; **no SEs** |
| `LarsRegressor::builder()` (`solvers`) | `method(LarsMethod::{Lar, Lasso})`, `with_intercept`, `n_nonzero_coefs(usize)`, `alpha` (sklearn 1/(2n) scale), `standardize` | `FittedLars`: `alphas() -> ColRef`, `coefs_path()`, `active()` (sklearn's active_), `method()`, `path() -> RegPath`; no SEs |

- **Elastic net is now coordinate descent** (#187; before 0.8 it was a soft-thresholded ridge fit and
  wrong). `Raw` minimises `‖y−Xβ‖² + λ(α‖β‖₁ + (1−α)‖β‖²)`, which for the lasso is glmnet with
  `λ_raw = 2n·λ_glmnet`. `LambdaScaling::Glmnet` solves glmnet's objective directly (equals
  `glmnet(x, y, alpha, lambda, standardize = FALSE)`). Ridge `Raw` is `‖y−Xβ‖² + λ‖β‖²` and `Glmnet`
  multiplies λ by n. The intercept is never penalised. Coordinate descent allows ≥100 000 passes
  whatever `max_iterations` says, and returns `ConvergenceFailed` at its pass limit.
- LassoLars (`LarsMethod::Lasso`) drops a variable whose coefficient crosses zero (fixed in 0.10,
  #305); it matches sklearn `lars_path(method="lasso")` to 1e-10.
- `RidgeRegressor`/`ElasticNetRegressor` also have `.path(&x,&y,&lambdas)` and
  `.cv_path(&x,&y,&lambdas,&fold_ids)` (see §5).
- **Streaming / distributed fits.** `MomentAccumulator::new(p)` accumulates with `push_row(&[f64],
  y)?` and `merge(&other)?`; then `OlsRegressor::builder().build().fit_from_accumulator(&acc)?` (Ridge
  too) or `fit_from_moments(xtx, xty, sum_x, sum_y, n)`. The mergeable, serialisable
  `SufficientStats` (`push`, `push_weighted`, `from_data(&x, &y, weights)`, `merge`, `to_bytes`/
  `from_bytes`) feeds `fit_from_state(&st)` on OLS, WLS and ridge and returns what `fit()` would,
  **including SEs, intervals and aliasing**. Row-level output (residuals, leverage, HC SEs) is empty
  or an error for these fits.

```rust
use anofox_regression::prelude::*;
use anofox_regression::inference::ClusterOptions;
use faer::{Col, Mat};
let fit = OlsRegressor::builder().with_intercept(true).confidence_level(0.95).build().fit(&x, &y)?;
let r = fit.result();
println!("b0={:?} se={:?}", fit.intercept(), r.intercept_std_error);
for j in 0..x.ncols() {
    println!("b{} = {:.4}  SE {:.4}  p {:.3e}", j + 1, fit.coefficients()[j],
        r.std_errors.as_ref().unwrap()[j], r.p_values.as_ref().unwrap()[j]);
}
let pi = fit.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95); // pi.lower / pi.upper
let hc = fit.hc_inference(&x, HcType::HC3)?;                                     // robust SEs
let v = fit.vcov().unwrap();                                    // σ²(XᵀX)⁻¹, intercept first
let v_cl = fit.cluster_vcov(&x, &[&cluster_ids], &ClusterOptions::cr1())?;     // vcovCL
```

### 3.2 Robust / constrained / online

| Estimator | Key options | Fitted accessors |
|---|---|---|
| `HuberRegressor::builder()` | `epsilon` (1.35), `alpha` (L2, **default 1e-4**, sklearn), `scale_method(HuberScale::{Mad (default, MASS::rlm), Joint (sklearn)})`, `max_iterations` (100), `tolerance` (1e-5), `compute_inference` (**false**), `confidence_level` | `scale()`, `outliers() -> &[bool]`, `n_outliers()`, `epsilon()`, `weights()` (= `rlm()$w`), `IterativeFit`; Wald t table when inference is on |
| `TheilSenRegressor::builder()` | `max_subpopulation` (10 000), `n_subsamples, max_iter` (300), `tolerance, random_state(u64), univariate_pairwise, compute_inference` (false), `ci_method(TheilSenCiMethod::{Auto, Exact, Bootstrap})`, `n_bootstrap` (999) | `ci_method()`, `bootstrap_replicates()`, `IterativeFit` (Weiszfeld). Exact = Sen's CI (1 feature); Bootstrap = percentile CI with no test statistic |
| `RansacRegressor::builder()` | `min_samples, residual_threshold, max_trials` (100), `stop_probability, stop_n_inliers, random_state, compute_inference` (false) | `inlier_mask()`, `n_inliers()`, `n_trials()`, `residual_threshold()`, `inlier_fit() -> &FittedOls`; ties broken by inlier R² as sklearn |
| `BlsRegressor::builder()` / `BlsRegressor::nnls()` | `lower_bounds(Vec)`, `upper_bounds(Vec)`, `lower_bound_all(f64)`, `upper_bound_all(f64)`, `max_iterations, tolerance` | BVLS solver since #191 (correct with non-zero bounds and intercepts); no SEs |
| `RlsRegressor::builder()` | `forgetting_factor` (1.0), `initial_p_diagonal` | `update(&mut self, &Col x_row, y) -> f64`, `p_matrix()`, `forgetting_factor()` |
| `PassiveAggressiveRegressor::builder()` | `c, epsilon, max_iter, tolerance, shuffle, loss(PaLoss::{EpsilonInsensitive, SquaredEpsilonInsensitive}), random_state` | `iterations()` (`IterativeFit`; was `n_iter()`); online use: `model.partial_fit(&mut PaState::new(p), &row, y)?` |

```rust
use anofox_regression::solvers::HuberScale;
let fitted = HuberRegressor::builder().epsilon(1.35).compute_inference(true).build().fit(&x, &y)?;
println!("σ(MAD)={:.4} outliers={} conv={}", fitted.scale(), fitted.n_outliers(), fitted.converged());
let sk = HuberRegressor::builder().scale_method(HuberScale::Joint).build().fit(&x, &y)?; // sklearn scale_
```

### 3.3 Bayesian, PLS, LmDynamic

- `BayesianRidgeRegressor::builder()` and `ArdRegressor::builder()` take `with_intercept, max_iter,
  tolerance, alpha_1, alpha_2, lambda_1, lambda_2`. BayesianRidge adds `alpha_init` and
  `lambda_init`; ARD adds `threshold_lambda`. The fitted models provide `alpha()` (noise precision),
  `lambda()` or ARD's `lambdas() -> ColRef`/`kept_features()`, `sigma()` (posterior covariance),
  `sigma_diag() -> Col`, `iterations()`/`converged()` (`IterativeFit`), `predict_std(&x)`
  (centred, as sklearn ≥ 1.9, #306), and `credible_intervals(level) -> CredibleIntervals{level, sd,
  lower, upper, intercept_sd, intercept_interval}`. BayesianRidge also has
  `log_marginal_likelihood()`.
- `PlsRegressor::builder()` (or `PlsRegressor::new(k)`) takes `n_components, with_intercept,
  tolerance, scale(bool)`. `FittedPls` provides `n_components(), weights(), x_loadings(),
  y_loadings(), scores(), x_means(), y_mean(), transform(&x), explained_variance_ratio()` (now
  `pls::explvar/100`; fixed in #198, as was `with_intercept(false)`).
- `LmDynamicRegressor::builder()` (greybox `lmDynamic`) takes
  `ic(InformationCriterion::{AIC, AICc, BIC})` (default AICc), `distribution(AlmDistribution)`,
  `auto_span()` (**the default**: optimised LOWESS span as `lmDynamic(f = NULL)`), `lowess_span(f)`
  (fixed span), `no_smoothing()`, `max_models(usize)` (default 4096; all 2^p subsets are enumerated
  and the fit errors beyond the cap or for p > 30), `with_intercept, confidence_level`.
  `FittedLmDynamic` provides `dynamic_coefficients() -> &Mat` (n × p), `model_weights()`,
  `smoothed_weights()`, `model_specs() -> &[CandidateModel]`, `pointwise_ic()`, `span()`,
  `span_criterion(f)`, `coefficient_at(i, j)` and `coefficients_at(i)`.

### 3.4 Smoothers and shape constraints

- **P-spline** (`mgcv` `s(x, bs = "ps")` since #210, 1 predictor): `PSplineRegressor::new()` (or
  `::builder()`, which returns `Self`), then `.with_n_basis(k)`, `.with_penalty_order(2)` or
  `.with_lambda(λ)` (default: λ by continuous GCV), then `.fit(&x_n_by_1, &y)`. `FittedPSpline`
  provides `edf()`, `sigma2()`, `lambda()` and `predict_with_interval` (Bayesian posterior SEs);
  predictions extrapolate linearly outside the range. Since 0.12 `result()` has `log_likelihood`, `aic`, `aicc`,
  `bic` with `edf + 1` d.f. (= mgcv's logLik, AIC and BIC; they were 0 before), and a stored model is
  rebuilt by `FittedPSpline::from_parts(knots, coefficients, &covariance, sigma2, edf, n)` from `knots()`,
  `coefficients()`, `covariance()` (mgcv's Bayesian Vp matrix), `sigma2()`, `edf()`, `n_observations()`. For regression-spline **design columns** (rcs,
  B-splines) see §8.
- **LOWESS** (R's `stats::lowess`): `solvers::lowess::lowess(&[f64] x, &[f64] y,
  &LowessOptions::default())` (fields/setters `frac` 2/3, `iterations` 3, `delta: Option`) returns
  `LowessFit{ x, fitted, order }` (sorted x). `lowess_smooth(&Col, span)` and
  `lowess_smooth_weights(&Mat, span)` are now true LOWESS over the index (#204).
- **Isotonic** (PAVA, R's `isoreg`): `IsotonicRegressor::builder()` with `increasing(bool)`,
  `weights(Col)`, `tolerance`, `out_of_bounds(OutOfBounds::{Clip (default), Nan, Extrapolate})`.
  Fit with `.fit_1d(&Col x, &Col y)` (or `Regressor::fit` on a single-column `Mat`). `FittedIsotonic`
  provides `predict_1d`, `predict_single`, `x_thresholds()`, `y_values()`, `fitted_values()` and
  `is_increasing()`. Predictions between knots are a **right-continuous step** (value at the nearest
  training x to the left; R `approx(method="constant")`, not sklearn's linear interpolation).

### 3.5 Quantile regression and inference

`QuantileRegressor::builder()` takes `tau` (0.5), `with_intercept`, `max_iterations` (100),
`tolerance` (1e-6), `epsilon` (1e-6, IRLS smoothing for the start), `weights(Col)` and `alpha(f64)`
(L1 penalty on the slopes, sklearn's `QuantileRegressor` objective; default 0 = `quantreg::rq`). The
IRLS start is polished by simplex pivots, so the solution is the exact LP optimum; `converged()`
reports whether it was certified. `FittedQuantile` provides `tau()`, `alpha()`, `check_loss()`,
`pseudo_r_squared()` (both weighted, R1 against `rq(y ~ 1)`), `has_intercept()` and `IterativeFit`.
Its `RegressionResult` has **no SEs**. For inference (quantreg `summary.rq`):

```rust
use anofox_regression::solvers::quantile_inference::{quantile_process, QuantileSe};
let q = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y)?;
let qi = q.inference(&x, &y, QuantileSe::Nid, 0.95)?;     // Iid | Nid | Boot{seed, r}
// QuantileInference { tau, se_type, has_intercept, estimate, std_error, statistic, p_value,
//   conf_low, conf_high, conf_level, df_residual, bandwidth, sparsity, covariance, n_boot }
// all vectors put the intercept first
let ci = qi.predict_interval(&x_new, 0.95)?;               // predict.rq(interval = "confidence")
let proc_ = quantile_process(&x, &y, &[0.1, 0.5, 0.9], true, Some(QuantileSe::Iid), 0.95)?;
```
Also available: `quantile_inference(x, y, tau, with_intercept, se, level)` and
`hall_sheather_bandwidth(tau, n)`. Inference is rejected for weighted and penalized (`alpha > 0`)
fits; `FittedQuantile::predict_with_interval` returns NaN limits (use `predict_interval`). Quantile
lines may cross (no monotonicity is enforced).

### 3.6 GLMs

All of these fit by IRLS, implement `Regressor`/`FittedRegressor` and `IterativeFit`. The builders
share `with_intercept, compute_inference, confidence_level, max_iterations, tolerance` (1e-8, R's
relative-deviance rule), `na_action`, plus: `lambda(f64)` (a ridge penalty on the slopes; SEs use
`(XᵀWX + λI)⁻¹` as `rms::lrm(penalty=)`), `offset(Col)` (on the link scale, e.g. `log(exposure)`),
and `error_on_non_convergence(bool)` (default `true` → `Err(ConvergenceFailed)`; with `false` the
last iterate is returned with `converged() == false`). Collinear columns are aliased as R's `glm`
does (NaN coefficient, #192). The null deviance respects offset, intercept and weights (#190).

| Entry point | Specific options | Public fields / accessors |
|---|---|---|
| `LogisticRegressor::builder()` | `penalty(Penalty::{None, L2(λ)})`, `l2(λ)`, `c(C)` (sklearn C = 1/λ), `threshold` | `predict` → **0/1 labels**, `predict_proba`, `decision_function`, `score` (accuracy), `coefficients()`, `intercept()`, `result()`, `vcov()`, `iterations()`, `inner() -> &FittedBinomial` |
| `BinomialRegressor::logistic()/probit()/cloglog()/builder()` | `link(BinomialLink::{Logit, Probit, Cloglog})`, `weights(Col)` (trials / case weights); y in [0,1] | fields `deviance, null_deviance, dispersion (=1), iterations, converged`; `vcov()`, `predict_probability`, `predict_linear`, `pearson_/deviance_/working_residuals()`, `prior_weights()`, `predict_with_offset` |
| `PoissonRegressor::log()/identity()/sqrt()/builder()` | `link(PoissonLink)`, `weights(Col)`, `estimate_dispersion(bool)` (quasipoisson) | same fields; `prior_weights()`, `predict_count` |
| `NegativeBinomialRegressor::builder()` / `::with_theta(θ)` | `theta` (1.0), `estimate_theta(bool)` (true), `theta_max_iter` (25), `theta_tolerance` (1e-6), `estimate_dispersion(bool)` (false); **no weights** | adds field `theta`, `overdispersion_ratio()`, `predict_count` |
| `TweedieRegressor::builder()/gaussian()/poisson()/gamma()/inverse_gaussian()` | `var_power(p)` (1.5; p ∈ (0,1) → `InvalidParameter` at fit), `link_power(q)` (0 = log; default canonical 1−p), `weights(Col)` | fields `deviance, null_deviance, dispersion` (Pearson), `iterations, converged`; `prior_weights()`, `predict_mu`, `predict_eta` |
| `GammaRegressor::builder()` | `link_power` (default log), `weights(Col)`; wraps Tweedie p = 2 | `vcov()`, `inner() -> &FittedTweedie` (dispersion, deviance), `predict_mu/eta`, `predict_with_offset` |

Wald reference distribution: **z** for binomial, Poisson and NB (as `summary.glm`/`glm.nb`); **t on
the residual df** for Tweedie, Gamma, quasi-Poisson and NB with `estimate_dispersion(true)`. AIC, BIC
and `log_likelihood` (full family likelihood, weighted) are in `result()`. **Read the deviance from
the `deviance` field.** For LR tests between fits use `inference::lr_test(&reduced, &full)` (§9).

```rust
let fitted = PoissonRegressor::log().offset(log_exposure).build().fit(&x, &counts)?;
println!("dev {:.3} / null {:.3}  disp {:.3}  AIC {:.2}  conv {}",
    fitted.deviance, fitted.null_deviance, fitted.dispersion, fitted.result().aic, fitted.converged());
let mu_ci = fitted.predict_with_se(&x_new, PredictionType::Response, Some(IntervalType::Confidence), 0.95);
let quasi = PoissonRegressor::log().estimate_dispersion(true).weights(w).build().fit(&x, &counts)?;

let nb = NegativeBinomialRegressor::builder().estimate_theta(true).build().fit(&x, &counts)?;
println!("theta = {:.3}", nb.theta);
let lr = LogisticRegressor::builder().build().fit(&x, &y01)?;
let p = lr.predict_proba(&x_new);                           // lr.result() has the coef table
let strict = PoissonRegressor::log().build().fit(&x, &counts)?.ensure_converged()?;
```

### 3.7 ALM (Augmented Linear Model)

`AlmRegressor::builder()` takes `distribution(AlmDistribution::Normal)` (or another variant), `link(LinkFunction)`,
`extra_parameter(f64)` (shape, df, or the α for AsymmetricLaplace), `loss(AlmLoss::{Likelihood, MSE,
MAE, HAM, ROLE{trim}})` (now actually minimised, #210), `role_trim`, `with_intercept,
compute_inference, confidence_level, max_iterations, tolerance` (NaN rows are dropped, `NaAction::Omit`). It supports
25 distributions: Normal, Laplace, StudentT, Logistic, AsymmetricLaplace, GeneralisedNormal, S,
LogNormal, LogLaplace, LogS, LogGeneralisedNormal, FoldedNormal, RectifiedNormal, BoxCoxNormal,
Gamma, InverseGaussian, Exponential, Beta, LogitNormal, Poisson, NegativeBinomial, Binomial,
Geometric, CumulativeLogistic and CumulativeNormal. Likelihood fits are polished to greybox's
optimum (#144); Box–Cox normal coefficients are on the transformed scale (#145).
`FittedAlm` provides `scale()`, `extra_parameter()`, `distribution()`, `link()`, `loss()`, `vcov()`
(greybox's `vcov(alm)` when inference is on), `IterativeFit`, `predict_with_interval` (`predict.alm`
limits for supported families, NaN otherwise) and `result()` (log_likelihood, AIC with greybox's
parameter count). `LinkFunction` is one of `Identity, Log, Logit, Probit, Inverse, Sqrt, Cloglog`.
```rust
let fitted = AlmRegressor::builder().distribution(AlmDistribution::Laplace).with_intercept(true)
    .build().fit(&x, &y)?;
println!("scale {:.4}  logLik {:.3}", fitted.scale(), fitted.result().log_likelihood);
```

### 3.8 Penalized GLM engine

`solvers::penalized_glm` (re-exported in the prelude) matches R `glm`/`MASS::glm.nb` when no prior is
set, and gives the posterior mode (like `arm::bayesglm` with fixed scales) when priors are set.
- `PenalizedGlmRegressor::builder()` takes:
  - `family(PenalizedGlmFamily::poisson()|binomial()|logistic()|negative_binomial()|gamma()|tweedie(p))`.
    The variants are `Poisson(PoissonLink)`, `Binomial(BinomialLink)`, `Logistic{threshold}`,
    `NegativeBinomial{theta: Option}`, `Gamma` and `Tweedie{power}` (1 ≤ p ≤ 2, log link).
  - `with_intercept`, `max_iterations(u32)` (100), `tolerance` (1e-8), `compute_inference`
    (**default false**), `confidence_level`, `lambda`.
  - `priors(Vec<PriorSpec>)` with `PriorSpec::flat()`, `normal(loc, scale)` or `laplace(loc, scale)`.
    Give p entries, or p+1 with the intercept first.
  - `vcov_type(VcovType::{Laplace (default), Sandwich, Naive})`, `offset(Col)`, and
    `estimate_dispersion(bool)` (Poisson only: quasipoisson SEs). No prior weights.
- Fit with `.fit(&x, &y)` (rows with NaN are dropped). It returns `Result<FittedPenalizedGlm,
  RegressionError>` (was `GlmEngineError`). A fit that does not converge is **not** an error, so
  check `converged()` or call `ensure_converged()`.
- `FittedPenalizedGlm` provides `coefficients()`, `intercept()`, `family()`, `deviance()`,
  `null_deviance()`, `pseudo_r_squared()`, `log_likelihood()`, `aic()`, `bic()`, `dispersion()`,
  `theta()`, `n_observations()`, `accuracy()`, `std_errors() -> Option<&[f64]>` (slopes),
  `vcov() -> Option<&Mat>` (includes φ for Gamma/Tweedie/quasi-Poisson since #210),
  `inference() -> Option<&GlmInferenceResult{std_errors, z_values, p_values, ci_lower, ci_upper,
  intercept_*}>`, `separation()`, `fitted_values() -> ColRef`, `result() -> &GlmResult`,
  `predict`, `predict_linear`, `predict_with_offset`, `augment(&x,&y)`, `predict_se` and
  `IterativeFit`.
- The column-major functional API (x is `&[Vec<f64>]` with **one Vec per column**) is
  `fit_poisson/fit_binomial/fit_logistic/fit_negbinomial/fit_gamma/fit_tweedie(y, x, &XxxOptions)`
  (build the options with `XxxOptions::default()` + setters). It returns `GlmResult{core:
  GlmFitResult, inference, log_likelihood, bic, fitted_values, valid_rows, model_info}`.
```rust
let fit = PenalizedGlmRegressor::builder()
    .family(PenalizedGlmFamily::poisson())
    .priors(vec![PriorSpec::flat(), PriorSpec::normal(0.0, 1.0)])   // intercept, x1
    .vcov_type(VcovType::Laplace).compute_inference(true)
    .build().fit(&x, &y)?;
assert!(fit.converged());
```

### 3.9 GLMM

`GlmmRegressor::gaussian()` (lmer, REML/ML) and `poisson()`, `binomial()`, `negative_binomial()`,
`gamma()` and `tweedie(p)` each return a `GlmmRegressorBuilder`. Builder options: `with_intercept`,
`random_intercept(bool)`, `random_slopes(Vec<usize>)` (column indices of x),
`random_slopes_per_factor(Vec<Vec<usize>>)`, `reml(bool)`, `nb_theta(f64)` (fixes θ), `offset(Col)`,
`max_iterations`, `tolerance`, `theta_max` (default 1000), `satterthwaite(bool)` (Gaussian only;
default false) and `n_agq(k)`: `0` (default) = PIRLS = `glmer(nAGQ=0)`, `1` = glmer's default
Laplace, `k > 1` = adaptive Gauss–Hermite (single scalar random intercept).
- Fit a single factor with `.fit(&x, &y, &group: &[usize])` (or `GroupedRegressor::fit_grouped`).
  Use `.fit_crossed(&x, &y, &[&a, &b])` for `(1|a)+(1|b)`. For nesting `(1|a/b)`, pass an `a:b`
  interaction id as a factor.
- `FittedGlmm` gives the fixed effects (**intercept first**) through `fixed_effects() -> ColRef`,
  `intercept()`, `slopes()`, `std_errors()`, `z_values() -> Col`, `p_values()` (normal),
  `conf_int(level) -> (Vec, Vec)` (Wald), `vcov() -> &[Vec<f64>]`, `satterthwaite_df()` and
  `inference(GlmmDdf::{Wald, Satterthwaite}, level) -> GlmmInference{std_errors, statistics, df,
  p_values, conf_low, conf_high}`.
- Variance components: `theta()` (σ_b/σ), `sigma()`, `sd_random()`, `var_random()`,
  `random_cov()`, `random_sd()`, `random_corr()` and `nb_theta()`.
- BLUPs: `random_effects()`, `random_effects_matrix()`, `random_effects_se()`,
  `random_effects_se_matrix()`, `random_effects_conf_int(level)`, `n_random_effects()`, and per
  factor `factors() -> &[FactorSummary{n_levels, sd, blups}]`, `n_factors()` and `factor_random_*(f)`.
- Fit statistics: `deviance()` (`−2(logLik − saturated)` for GLMMs), `log_likelihood()`, `aic()`,
  `bic()`, `n_parameters()`, `n_observations()`, `n_groups()`, `df_residual()`, `is_reml()`, and
  `converged()`/`iterations()` (`IterativeFit`).
- Predictions and residuals: `fitted_values()` (conditional), `fitted_population()`, `residuals()`,
  `pearson_residuals()`, `predict_fixed(&x)` (population level only; there is no BLUP-based predict
  for new data) and `predict_fixed_with_offset`.
```rust
let fit = GlmmRegressor::gaussian().reml(true).satterthwaite(true).build().fit(&x, &y, &group)?;
let inf = fit.inference(GlmmDdf::Satterthwaite, 0.95)?;     // lmerTest-style t tests
println!("sd_b {:.3}  sigma {:.3}  p {:?}", fit.sd_random(), fit.sigma(), inf.p_values);
```

### 3.10 AFT survival (survreg)

`AftRegressor::builder()` takes `distribution(AftDistribution::{Weibull, LogNormal, LogLogistic,
Exponential})`, `with_intercept`, `max_iterations(u32)`, `tolerance` (1e-9), `compute_inference`
(**default false**), `confidence_level`, `priors(Vec<PriorSpec>)` and `vcov_type`. Fit with
`.fit(&x, &time: &Col (>0), &event: &[bool])` (or `SurvivalRegressor::fit_survival`), where `true`
means an event and `false` means right-censored; it returns `RegressionError` on failure.
`FittedAft` provides `coefficients()`, `intercept()`, `scale()` (σ), `distribution()`,
`log_likelihood()`, `null_log_likelihood()`, `aic()`, `bic()`, `IterativeFit`, `fit_result()`
(`n_events`, `n_censored`), `result()`, and `inference() -> Option<&AftInference{std_errors,
z_values, p_values, ci_lower, ci_upper, intercept_*, log_scale_std_error, vcov, information, ..}>`
(slopes; intercept separate). Predictions: `predict_linear`, `predict_quantile(&x, p)`,
`predict_median`, `predict_survival(&x, t)`, `predict_cdf(&x, t)`, and per row `survival_curve(row,
&times)` / `quantiles(row, &probs)`. Diagnostics: `standardized_residuals(&x, &time)`,
`cox_snell_residuals(&x, &time)`, `score_contributions(..)`, `cluster_vcov(..)` (`survreg(cluster=)`).
The functional form is `fit_aft(time, x_colmajor, event_f64, &AftOptions)` (options via
`AftOptions::default()` + setters; the distribution field is `dist`). Coefficients are on the
log-time scale: exp(β) is a time ratio. For Cox PH, Kaplan–Meier and log-rank see §14.
```rust
let fit = AftRegressor::builder().distribution(AftDistribution::Weibull)
    .compute_inference(true).build().fit(&x, &t, &event)?;
let median = fit.predict_quantile(&x, 0.5);
```

### 3.11 Mediation, EB shrinkage, AID

- **Mediation** (linear, no-interaction `mediation::mediate`):
  `solvers::mediation::mediate(&y, &m, &x, &covariates: &[C], &MediationOptions::default()
  .method(MediationMethod::{Bootstrap (default), Sobel, QuasiBayesian}).boot_ci(BootCi::{Percentile,
  Bca}).n_sims(1000).seed(42))` (also `conf_level, treat_value` 1, `control_value` 0) returns
  `MediationResult{acme, ade, total, prop_mediated: MediationEffect{estimate, std_error, statistic,
  p_value, conf_low, conf_high}, a, b, c_prime, r_squared_mediator, r_squared_outcome, n_obs, ..}`.
- **EB shrinkage** (metafor DL random-effects meta-analysis plus BLUPs):
  `eb_shrink(&estimates, &ses, &EbShrinkOptions::default())` (setters `method(TauMethod::
  {DerSimonianLaird, None})`, `tau_squared(Option<f64>)`) returns `EbShrinkResult{mu, mu_se,
  tau_squared, i_squared, q, n_groups, groups: Vec<ShrunkenGroup{estimate, se, shrunken,
  shrunken_se, weight}>}`. `shrunken_se` treats μ as known, so it is slightly smaller than metafor's
  `blup()` SE.
- **AID demand classifier** (greybox): `AidClassifier::builder().anomaly_alpha(..)
  .intermittent_threshold(..).detect_anomalies(bool).ic(InformationCriterion).build().classify(&y)`
  returns `DemandClassification{demand_type: DemandType::{Regular, Intermittent}, is_fractional,
  distribution: DemandDistribution, parameters, anomalies, ic_values, zero_proportion}`. Candidates
  are fitted by ML since #203.

## 4. Diagnostics and inference

All of these take the design **without** an intercept column unless they have a `with_intercept`
flag.

| Need | Function (module) | Notes |
|---|---|---|
| VIF | `variance_inflation_factor(&x) -> Col`, `generalized_vif(&x, &group_sizes) -> Vec`, `generalized_vif_terms(&x, &group_sizes) -> Vec<GvifTerm { gvif, df, adjusted_gvif }>`, `high_vif_predictors(&vif, thr)` | GVIF is Fox–Monette `det` form since #196 (= car::vif); report `adjusted_gvif` = `GVIF^(1/(2·df))`. Since 0.11.1 aliased columns (constant, exactly collinear, n ≤ p) give `∞` for the terms involved only (car::vif errors); undefined (empty group, sizes ≠ ncol, no rows) is `NaN` |
| Condition number | `condition_number(&x, with_intercept)` (unscaled `kappa`, ∞ if singular), `condition_diagnostic(..) -> ConditionDiagnostic{condition_number, condition_number_xtx, singular_values, condition_indices, scaled_condition_number, severity, warning}`, `classify_condition_number`, `variance_decomposition_proportions` | Indices are BKW (unit-length columns, #207); severity from `scaled_condition_number`: <30 WellConditioned, <100 Moderate, <1000 High, else Severe |
| Leverage | `compute_leverage(&x, with_intercept)`, `compute_leverage_with_aliased`, `high_leverage_points(&h, n_params, thr)` | hatvalues |
| Influence | `cooks_distance(&resid, &h, mse, n_params)`, `dffits(..)`, `influential_cooks(&d, thr)`, `influential_dffits` | No DFBETAS/covratio for linear models or GLMs (Cox has `dfbeta`/`dfbetas`, §14). Prefer `augment()` |
| Residuals (linear) | `standardized_residuals(&r, mse)`, `studentized_residuals(&r, &h, mse)` (internal), `externally_studentized_residuals(&r,&h,mse,p)`, `residual_outliers` | |
| Residuals (GLM) | `response_/pearson_/deviance_/working_residuals(&y, &mu, &family)`, `standardized_pearson_/standardized_deviance_residuals(.., &h, dispersion)`, `pearson_chi_squared`, `estimate_dispersion_pearson/deviance(y, mu, family, n_params)` | the family is e.g. `PoissonFamily::log()`, `BinomialFamily::logistic()`, `NegativeBinomialFamily::new(θ)`, `TweedieFamily::gamma()`; fitted-model residual methods are `√w`-weighted as R |
| Heteroskedasticity | `breusch_pagan(&resid, &z, studentize)` (true = Koenker, the `bptest` default), `white_test(&resid, &x)` | returns `HeteroskedasticityTest{statistic, df, p_value, method}` |
| Component+residual / added-variable | `partial_residuals(&fit, &x, &y) -> Vec<PartialResiduals{term, x, component, partial_residual}>`, `partial_residuals_glm`, `added_variable(&x, &y, w, with_intercept) -> Vec<AddedVariable{term, x_residual, y_residual, slope}>` | |
| Separation | `detect_separation(&x, &y, with_intercept) -> LinearSeparation{separated, infinite, direction}` (exact LP, `detectseparation`); heuristic `check_binary_separation(&x, &y)`, `check_count_sparsity(&x, &y)` | `SeparationCheck{has_separation, separated_predictors, separation_types: SeparationType::{None, Complete, Quasi, MonotonicResponse}, warning_message}`; prefer `detect_separation` (sees combinations) |
| Robust SEs | `FittedOls::hc_inference(&x, HcType::{HC0, HC1, HC2, HC3})` returns `HcInference{hc_type, std_errors, t_statistics, p_values, conf_interval_lower/upper, intercept: Option<HcInterceptInference>}`; generic `inference::compute_hc_inference(x, coefs, intercept, resid, aliased, with_intercept, hc, level)` (now returns `RegressionError`) | sandwich::vcovHC. Cluster-robust: `cluster_vcov` on OLS/WLS/AFT/ordinal, `inference::cluster::{lm_cluster_vcov, glm_cluster_vcov, cluster_sandwich}`, `CoxFit::robust_vcov` (§9) |
| Nested F (anova) | `nested_f_test(&x_reduced, &x_full, &y, &NestedFTestOptions::default())` or `nested_f_test_indices(&x_full, &y, &reduced_cols, &opts)` | `NestedFTest{f_statistic, df1, df2, p_value, rss_reduced, rss_full, adj_r_squared_*, ..}`; Gaussian/OLS only |
| ANOVA table (lm) | `anova_table(&x, &y, terms: Option<&[Vec<usize>]>, &AnovaTableOptions) -> AnovaTable{terms: Vec<AnovaTerm{df, sum_sq, mean_sq, f_statistic, p_value}>, residual_df, ..}` | `AnovaType::I` = `anova(lm)`, `II` = `car::Anova(type=2)`; OLS without interactions. Term-aware Wald `anova.rms` table: §9 |
| Variance factor / intervals | `compute_variance_factor(x, w, with_intercept, excluded, ridge_λ)`, `compute_ridge_variance_factor`, `leverage_new(&M, &x_new, with_intercept)`, `intervals_from_variance_factor(..)` | building blocks; `(XᵀX)⁻¹` now from a QR of the design (#181) |
| Coefficient math | `CoefficientInference::{standard_errors, t_statistics, p_values, confidence_intervals, ..}` | low-level |
| broom::augment / PRESS | `augment`, `augment_with`, `press`, `augment_glm`, `FittedRidge::loo` | see §2 |
| Discrimination / calibration indexes | `diagnostics::accuracy::{concordance, somers2, rcorr_cens, lrm_stats, val_prob, brier_score, ..}` | see §12 |

Likelihood-ratio tests between nested GLM / AFT / Cox fits: `inference::lr_test(&reduced, &full) ->
ChiSqTest{statistic, df, p_value}` and `model_lr_test(&fit)`; score tests and profile-likelihood CIs:
`inference::likelihood::{glm_score_test, profile_confint}` (see §9).

## 5. Regularisation paths and cross-validation

`solvers::path` implements the Gaussian elastic net on **glmnet's scale**: the objective is
`1/(2n)RSS + λ[(1−α)/2‖β‖² + α‖β‖₁]`, and y is internally scaled to unit variance as glmnet does.
- `PathOptions::ridge()` or `PathOptions::elastic_net(alpha)`, then the setters `.with_intercept(b)`,
  `.standardize(b)` (or `.with_standardize(b)`), `.tol(..)`, `.max_iter(..)`. The fields are `alpha,
  with_intercept` (was `fit_intercept`), `standardize, tol, max_iter`.
- `lambda_grid(&x, &y, &opts, n_lambda, lambda_min_ratio: Option<f64>)` gives the default glmnet
  grid.
- `fit_path(&x, &y, &lambdas, &opts)` returns `RegPath{lambdas, intercepts, coefficients (p × L Mat),
  df, alpha}`, with `coefficients_at(l)` and `predict(&x) -> Mat`.
- `kfold_ids(n, k, seed)` gives fold ids. `cv_path(&x, &y, &lambdas, &fold_ids, &opts)` returns
  `CvPath{lambdas, cv_mean, cv_se, df, lambda_min, lambda_1se, index_min, index_1se, n_folds, path}`.

`solvers::cv` provides fold generators that match scikit-learn:
- `group_codes(&keys)`, `group_kfold_ids(&groups, k, GroupBalance::{Rows, Random}, seed)`,
  `logo_ids(&groups)`, `stratified_kfold_ids`, `quantile_strata(&y, n_bins)` and
  `stratified_group_kfold_ids`.
- `time_series_splits(n, &TimeSeriesSplitOptions)` (fields/setters `n_splits, test_size, gap,
  window: TrainWindow, min_train_size`) and `group_time_series_splits`.
- `splits_from_fold_ids(&ids) -> Vec<CvSplit{train, test}>`; `CvSplit::new(train, test)`.
- `cv_path_splits(&x, &y, &lambdas, &splits, &opts, CvMetric::{Mse, Mae})`.
- `nested_cv_path(&x, &y, lambdas: Option<&[f64]>, &outer_splits, &opts,
  &NestedCvOptions::new(InnerFolds::Kfold{k, seed}))` (also `Group/Logo/Stratified/TimeSeries`, with
  `rule: LambdaRule::{Min, OneSe}`) returns `NestedCv{folds, mean, se, metric}`;
  `nested_inner_splits` exposes the inner splits.

These paths are **Gaussian only** (there is no CV path for GLMs; for an ordinal/logistic penalty
trace use `pentrace`, §13). For bootstrap / cross-validated optimism of **any** modelling process use
`validation::{predab_resample, Resamples, validate_lrm, validate_ols, calibrate_*, bootcov}` (§11).
```rust
use anofox_regression::solvers::path::{cv_path, fit_path, kfold_ids, lambda_grid, PathOptions};
let opts = PathOptions::elastic_net(0.5).with_intercept(true);
let lambdas = lambda_grid(&x, &y, &opts, 50, None)?;
let path = fit_path(&x, &y, &lambdas, &opts)?;
let cv = cv_path(&x, &y, &lambdas, &kfold_ids(x.nrows(), 5, 42), &opts)?;
let beta_1se = cv.path.coefficients_at(cv.index_1se);
```

## 6. R / sklearn equivalence

Sources: `validation/VALIDATION.md`, README and `tests/r_validation_*.rs`. Since 0.8 the R fixtures
are regenerable (`validation/regenerate_r_fixtures.sh`, R 4.6.1) and checked monthly at 1e-8.

| Rust | Oracle |
|---|---|
| `OlsRegressor` | `lm()`, `summary.lm`, `confint`, `logLik/AIC/BIC`, `predict.lm(interval=)` |
| `WlsRegressor` | `lm(weights=)` |
| `RidgeRegressor`, `ElasticNetRegressor`, `solvers::path` | `glmnet()` / `cv.glmnet` (use `LambdaScaling::Glmnet`; `standardize = FALSE` for the estimators) |
| `HuberRegressor` | `MASS::rlm(psi=psi.huber)` (`HuberScale::Mad`); sklearn `HuberRegressor` (`HuberScale::Joint`) |
| `BlsRegressor` | `nnls::nnls()`, `bvls::bvls()`, `quadprog::solve.QP` |
| `RlsRegressor` | batch OLS (`lm`) at forgetting factor 1; exponentially weighted LS otherwise |
| `PoissonRegressor`, `BinomialRegressor`, `LogisticRegressor` | `glm(family=poisson/quasipoisson/binomial, weights=)`, `confint.default`; L2 logistic = `rms::lrm(penalty=)` |
| `NegativeBinomialRegressor` | `MASS::glm.nb()`; with `estimate_dispersion(true)`, `glm(family=negative.binomial(θ))` |
| `TweedieRegressor` | `statmod::tweedie` with `glm` |
| `GammaRegressor` | `glm(family=Gamma(link="log"))` |
| `PenalizedGlmRegressor` / `fit_*` | `glm`, `MASS::glm.nb`; with priors, a Newton posterior mode (bayesglm-like); Laplace priors vs `glmnet(alpha=1)` |
| `GlmmRegressor` | `lme4::lmer`, `glmer(nAGQ=0/1/k)`, `glmer.nb`; `lmerTest` Satterthwaite; `ranef(condVar=TRUE)` |
| `AftRegressor` | `survival::survreg`, `psurvreg`, `predict(type="quantile")` |
| `AlmRegressor` | `greybox::alm()` |
| `LmDynamicRegressor` | `greybox::lmDynamic` |
| `AidClassifier` | `greybox::aid` (candidate AICc = `AICc(alm(y ~ 1))`) |
| `QuantileRegressor` + `inference` / `quantile_process` | `quantreg::rq`, `summary.rq(se="iid"/"nid"/"boot")`, `predict.rq(interval="confidence")`; with `alpha`, sklearn `QuantileRegressor` |
| `IsotonicRegressor` | `stats::isoreg` / `Iso::pava` |
| `lowess` | `stats::lowess` |
| `PlsRegressor` | `pls::plsr(method="simpls")` |
| `PSplineRegressor` | `mgcv::gam(s(x, bs="ps"), method="GCV.Cp")` |
| `mediate` | `mediation::mediate` (exact point estimates; simulated CIs agree within MC error) |
| `eb_shrink` | `metafor::rma(method="DL")` and `blup()` |
| `augment`, `press`, influence | `hatvalues, rstandard, rstudent, cooks.distance, dffits`, `broom::augment` |
| `hc_inference` | `sandwich::vcovHC(type="HC0..HC3")`, `coeftest`, `coefci` |
| `cluster_vcov`, `inference::cluster` | `sandwich::vcovCL`, `rms::robcov` |
| `breusch_pagan`, `white_test` | `lmtest::bptest` |
| `nested_f_test`, `anova_table` | `anova(lm_reduced, lm_full)`, `anova(lm)`, `car::Anova(type=2)` |
| `partial_residuals`, `added_variable` | `residuals(type="partial")`, `car::avPlots` |
| `variance_inflation_factor`, `generalized_vif` | `car::vif` |
| `condition_diagnostic` | `kappa(exact=TRUE)`, `olsrr::ols_eigen_cindex` |
| `detect_separation` | `detectseparation` |
| `TheilSenRegressor`, `RansacRegressor`, `PassiveAggressiveRegressor`, `LarsRegressor`, `BayesianRidgeRegressor`, `ArdRegressor` | scikit-learn 1.5 to 1.9.1 equivalents (Theil-Sen, LassoLars, PA at 1e-9); LAR path also `lars::lars` |
| `solvers::cv` fold generators | sklearn `GroupKFold`, `LeaveOneGroupOut`, `StratifiedKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit` |

Tolerances: after the #187–#210 audit most models are checked at 1e-6 to 1e-10 against R (OLS,
diagnostics, HC SEs, GLMs at `glm(epsilon=1e-15)`, elastic net vs glmnet, GLMM vs lme4 2.1.0 at
~1e-6). Remaining looser spots: GLS compound-symmetry/unstructured/spatial structures (1e-4, nlme's
optimiser), ALM distributions without a direct greybox counterpart (Student t, Beta, Box–Cox, S)
and Laplace/S SEs (finite-difference Hessian not meaningful; prefer bootstrap SEs).

## 7. Gotchas and limitations

**Design matrix and intercept**
- Never add a column of ones. Control the intercept with `with_intercept(true)` (the only name now).
  `coefficients` holds the **slopes only**, and the intercept's statistics are in the separate
  `intercept_*` fields. Exceptions that put the **intercept first**: `FittedGlmm` vectors,
  `QuantileInference`, `GlmmInference` and the `vcov()` matrices of the linear and IRLS GLM fits.
- The `fit_*` functions in `penalized_glm`, plus `fit_aft` and `mediate`, take a **column-major**
  `&[Vec<f64>]`, with one inner Vec per predictor (not per row).
- There is no formula interface, but there are design builders: `splines::{Rcs, Lsp, BSpline}`,
  `design::Design`/`TermMap` and `effects::ModelSpec` build spline, dummy and interaction columns
  and record which columns form each term (§8, §10). `generalized_vif` takes column counts per term.
- Option structs are `#[non_exhaustive]`: use `Default::default()` plus setters, not struct literals.

**Rank deficiency**
- Pivoted QR detects aliasing the way R does. Aliased slopes come back as `NaN` (as R's `NA`), with
  `result.aliased[j] = true`. `rank` and `n_parameters` count only the estimable terms. Use
  `has_aliased()` and `n_active_coefficients()`.
- The IRLS GLMs alias collinear non-constant columns too (since #192). `vcov()` has zero rows and
  columns for aliased columns.

**Dispersion and inference conventions**
- **Binomial**: dispersion is 1 and tests are z tests.
- **PoissonRegressor**: dispersion is **1** (as `glm(family = poisson)`; changed in #189 — before 0.8
  it was `max(1, χ²/df)`). `estimate_dispersion(true)` gives quasi-Poisson (Pearson χ²/df, t tests).
- **NegativeBinomial**: dispersion is 1 (the glm.nb likelihood covariance at θ̂), with z tests.
  `estimate_dispersion(true)` switches to Pearson χ²/df with t tests.
- **Tweedie and Gamma**: dispersion is Pearson χ²/(n−p) and tests are **t tests** on the residual df.
- **Gaussian LMM** `p_values()` use the normal approximation (`lmer` prints none). Use
  `inference(GlmmDdf::Satterthwaite, ..)` with `satterthwaite(true)`.
- **RANSAC** inference describes the OLS refit on the inliers. It ignores the selection step, so it
  is optimistic.
- `compute_inference` defaults to **false** for: the penalized GLM, AFT, Huber, Theil-Sen and RANSAC.
- These have **no SEs**: ElasticNet, LARS, BLS, RLS, PLS, PA, PSpline coefficients, LmDynamic, and
  QuantileRegressor until you call `.inference()`. For Bayesian models use `credible_intervals`.
- GLM log-likelihood, AIC and BIC are full-likelihood values (binomial with trial weights includes
  `lchoose`, #193). Linear-model logLik uses R's `logLik.lm` (σ counts as a parameter). ALM counts
  parameters as greybox's `nparam` (#202).
- Huber's default `alpha = 1e-4` is a small L2 penalty (sklearn); set `.alpha(0.0)` for an
  unpenalised fit as `MASS::rlm`.

**Convergence**
- The dedicated IRLS GLMs return `Err(ConvergenceFailed)` by default (`error_on_non_convergence`).
  Everything else (penalized GLM, AFT, Huber, quantile, GLMM, Bayesian, PA, ALM, Theil–Sen) returns
  `Ok` with `converged() == false`: check it, or call `.ensure_converged()?` (`IterativeFit`).
- Separation in logistic models produces huge SEs. Run `detect_separation` first, then add `lambda`
  or a `PriorSpec::normal` (Firth's correction is not implemented).
- Resampling-heavy code (Theil-Sen bootstrap inference, validation, bootcov) can be very slow in a
  debug build: use `cargo run --release`.

**Missing values**
- `NaAction` is applied by the `RegressionOptions`-based estimators (default `Omit` drops NaN rows,
  `Exclude` NaN-pads `*_expanded()`); other estimators error on NaN. ±∞ is always an error.

**Now implemented** (these were gaps in 0.5.35; see the named sections)
- Regression splines in the design: `splines::{Rcs, Lsp, BSpline, rcs_default_knots,
  restricted_interaction}` → §8.
- LR / score tests and profile-likelihood CIs: `inference::{lr_test, model_lr_test, glm_score_test,
  profile_confint}`; Wald chunk tests and `anova.rms`: `inference::{wald_test, wald_chunk, anova}`;
  `fastbw` → §9.
- Cluster-robust SEs for linear models, GLMs, AFT, ordinal and Cox: `inference::cluster` → §9.
- Bootstrap / CV optimism-corrected validation and calibration for any model:
  `validation::{predab_resample, validate_lrm, validate_ols, calibrate_lrm, calibrate_ols,
  calibrate_cph, bootcov}` → §11.
- Survival: `survival::{kaplan_meier, survfit, survdiff, coxph}` (Cox with `dfbeta`/`dfbetas`,
  `cox_zph`) → §14.
- Ordinal (`OrdinalRegressor`, `rms::orm`/`polr`) and continuation-ratio models → §15; multinomial
  logistic (`MultinomialLogisticRegressor`) → §17.
- GLS for longitudinal data (`GlsRegressor`, `nlme::gls`) → §16.
- Multiple imputation (`imputation::areg_impute`) with Rubin pooling (`inference::pool::pool_rubin`)
  → §18–§20, §9.
- GLM prior weights: `weights(..)` on Poisson, binomial, Tweedie and Gamma.

**Still not implemented** (checked against the 0.11.0 `src/`)
- DFBETAS and covratio for linear models and GLMs (only Cox has dfbeta/dfbetas).
- HAC (Newey–West) SEs.
- Prior weights for `NegativeBinomialRegressor` and `PenalizedGlmRegressor`.
- Firth-penalised logistic regression.
- Zero-inflated or hurdle count models, GAMs with several smooths (only the 1-D `PSplineRegressor`;
  use spline bases in an ordinary design instead), GEE, and BLUP-based prediction for new GLMM
  groups.
- CV regularisation paths for GLMs (paths are Gaussian only).

**Misc**
- `IsotonicRegressor` predicts a right-continuous step function. Quantile fits at several τ can
  cross. `OutOfBounds::Clip` is the default.
- `FittedRls::update` changes the coefficients but not the stored `result()` statistics.
- Fitted models are `Send + Sync`; with the `serde` feature they can be stored and reloaded.

---

## 8. Splines and design

Added in 0.6.0. Module `anofox_regression::splines` (bases) and `anofox_regression::design` (term-aware
design). Every basis comes back as a **`Basis { columns: Mat<f64>, names: Vec<String>, nonlinear: Vec<bool> }`**.
`Basis::new(cols, names, nonlinear)?` builds one by hand (factor dummies), `Basis::linear(&x, "x")` gives
one linear column, and `.n_columns()` / `.nonlinear_columns()` read it. **No basis has an intercept column.**
A non-finite `x` gives a NaN row. **Keep the `Rcs` / `Lsp` / `BSpline` value**: you need it to rebuild the
same basis on new data for prediction (all three serialize with the `serde` feature).

| Entry point | Options | Result / R equivalent |
|---|---|---|
| `Rcs::from_data(&x, nk)?` / `Rcs::new(&knots)?` | `.with_norm(RcsNorm::{None, LastInterval, Range (default)})`; `nk ≥ 3` | `.basis(&x, "age")` gives `k−1` columns `age, age', age''…` (`nonlinear = [false, true, …]`). `.knots()`, `.n_columns()`. Matches `rms::rcs` / `Hmisc::rcspline.eval` (1e-12) |
| `rcs_default_knots(&x, nk)?`, `rcs_default_knots_with(&x, nk, fractied)` | `DEFAULT_FRACTIED = 0.05` | knots of `rcspline.eval(knots.only=TRUE)`: type-7 quantiles from `default_knot_quantiles(nk)` (outer 0.10 for k=3, 0.05 for k=4–6, 0.025 for k>6). With n<100 the outer knots move to the 5th smallest/largest value. Tied extremes get special handling. Needs ≥ 6 finite values; fewer than 5 distinct values (e.g. a constant column) is `SplineError::TooFewUniqueKnots` (panicked before 0.12) |
| `rcs.restate(intercept, &coef)?` | `coef` = the `k−1` basis coefficients, linear first | `RcsRestated { intercept, linear, knots, cubic }` (one cubic per knot) and `.eval(x)`. Matches `Hmisc::rcspline.restate` |
| `Lsp::new(&knots)?` | — | `.basis(&x, name)` gives `x, (x−t₁)₊, …`. The hinge columns are flagged nonlinear. Matches `rms::lsp` |
| `BSpline::from_data(&x, n_knots, degree, KnotPlacement::{Uniform, Quantile})?` / `BSpline::new(&knots, degree)?` | `.with_extrapolation(SplineExtrapolation::{Constant (default), Linear, Continue, Periodic, Error})`, `.with_bias(bool)` (default true) | `.basis(&x, name)?` (**returns a `Result`**, unlike Rcs) gives `n_knots+degree−1` columns (periodic: `n_knots−1`; one fewer without the bias column) named `x_sp_0…`. **All columns are flagged nonlinear** (no linear column, so a B-spline block has no "nonlinear" chunk test). Matches sklearn `SplineTransformer` (1e-12) |
| `restricted_interaction(&a, &b)?` | — | `rms::%ia%`: the products without doubly-nonlinear terms. `rcs(x1,4) %ia% rcs(x2,3)` gives `x1*x2, x1*x2', x1'*x2, x1''*x2` |

**`Design`** stacks blocks and records a **`TermMap`**:
`let mut d = Design::new(); let t = d.add("age", basis)?;` returns the term index.
`d.add_interaction(a, b)?` is the full tensor product (rms `a * b`, first factor varying fastest), and
`d.add_restricted_interaction(a, b)?` is `%ia%`. Both are two-way, between main-effect terms only
(`DesignError::{NotAMainEffect, SelfInteraction, RowMismatch}`). Read it back with `d.matrix()` (n × p,
**no intercept**), `d.column_names()`, `d.terms()` and `d.block(t)`.
`TermMap` has `.terms() -> &[Term{name, columns, nonlinear, interaction: Option<Interaction{factors, components, restricted}>}]`,
`.n_columns()` and `.find("age * sex")`. A `TermMap` can also be built by hand (`add_main(name, &[bool])`,
`add_interaction(a, b, restricted)`) when you have only `β̂`/`V̂`.
**The crate has no formula parser or factor coder.** For a factor, either build the treatment dummies
yourself with `Basis::new` (all `false`), or use `effects::ModelSpec::categorical`, which builds the design
for you (see Effects). For `pentrace` / `base_penalty_matrix`, pass the term indices of the factors.

```rust
use anofox_regression::{design::Design, splines::{Basis, Rcs, BSpline, KnotPlacement, SplineExtrapolation}};
let rcs = Rcs::from_data(&age, 4)?;                       // knots at .05,.35,.65,.95 quantiles
let mut des = Design::new();
let a = des.add("age", rcs.basis(&age, "age"))?;
let s = des.add("sex", Basis::linear(&sex, "sex"))?;
let dummies = Mat::from_fn(n, 2, |i, j| f64::from(u8::from(grp[i] == (j + 1) as f64)));
des.add("grp", Basis::new(dummies, vec!["grp=1".into(), "grp=2".into()], vec![false; 2])?)?;
des.add_interaction(a, s)?;                               // rcs(age,4) * sex
let x = des.matrix();                                     // 9 columns, no intercept
let bs = BSpline::from_data(&age, 5, 3, KnotPlacement::Quantile)?
    .with_extrapolation(SplineExtrapolation::Linear).with_bias(false);
let bb = bs.basis(&[20.0, 50.0, 90.0], "age")?;           // 6 cols, extrapolated linearly
```

## 9. Term-aware inference: Wald chunk tests, anova.rms, LR/score tests, fastbw, approximation, pooling, robust covariance

### Getting β̂ and V̂
`vcov()` on `FittedOls`, `FittedWls`, `FittedRidge`, `FittedBinomial`, `FittedLogistic`, `FittedPoisson`,
`FittedNegativeBinomial`, `FittedTweedie` and `FittedGamma` returns `Option<Mat<f64>>`, **intercept first**,
with zero rows and columns for aliased columns (GLMs need `compute_inference(true)`). `FittedGls::vcov()` returns `&Mat`.
`FittedOrdinal` has `vcov()` (intercepts first), `vcov_slopes()` and `vcov_sandwich()`.
`FittedPenalizedGlm::vcov()` returns `Option<&Mat>`. **The Wald / anova functions want slopes only**: use
`drop_intercept(&beta_full, &vcov) -> (slopes, V_slopes)`, where `beta_full = [b0, b1..]`.

| Function (`anofox_regression::inference::…`) | Arguments | Returns / R oracle |
|---|---|---|
| `wald_test(&beta, &vcov, &C, Option<&r>, dist)` | `dist: WaldDistribution::{ChiSquare, F{df_residual}}` | `WaldTest{chi_square, df, f: Option, df_residual, p_value}`, `.chi_square_minus_df()`. `car::linearHypothesis` |
| `wald_chunk(&beta, &vcov, &cols, dist)` | slope column indices | the same. `WaldError::{Singular, Empty, Column, Dimension}` |
| `anova(&beta, &vcov, &terms, &AnovaOptions)` | `AnovaOptions::default().distribution(..).sigma2(Some(mse))` (adds partial SS and the ERROR row) | `WaldAnovaTable{rows: Vec<AnovaRow{label, kind: AnovaRowKind, term, columns, test, partial_ss}>, error: Option<(df, SS, MS)>}`. `rms::anova.rms` (1e-9; `main.effect=FALSE, india=TRUE, indnl=TRUE`) |
| `fastbw(&beta, &vcov, n_intercepts, &terms, rule, kind)` | `StepRule::{Aic{k, threshold}, P{sls}}`, `StepType::{Residual, Individual}` | `StepDown{deleted: Vec<StepDeletion{term, chisq, df, p_value, residual_chisq, residual_df, residual_p, aic}>, kept_terms, coefficients, vcov, kept_parameters}` (approximated, zeros for the deleted; `vcov` = fastbw `var` in the full layout, `kept_parameters` = `parms.kept`, 0.12). `rms::fastbw` (1e-9). **Ignores hierarchy**, as rms does |
| `approximation::approximate_model(&x, &lp, &vcov_full, &terms, &keep)` | `lp = b0 + Xβ̂` of any fit; `vcov` **intercept first** | `ModelApproximation{kept_terms, columns, coefficients (intercept first), vcov = W V Wᵀ, r_squared}`. RMS §5.5.2 `ols(lp ~ ., sigma=1)` |
| `approximation::approximation_path(&x, &lp, &terms)` | — | `Vec<ApproximationStep{term, r_squared}>` in fastbw deletion order. RMS Ch 19 |
| `approximation::heuristic_shrinkage(lr_chisq, df)`, `shrink_slopes(&beta, n_intercepts, gamma)` | — | `γ = (LR−p)/LR`. **Intercepts are not re-estimated**: refit them yourself |
| `anova_table(&x, &y, Option<&[Vec<usize>]>, &AnovaTableOptions)` | `.anova_type(AnovaType::{I, II})` | OLS without interactions only: `anova(lm)` / `car::Anova(type=2)` |
| `nested_f_test(&x_red, &x_full, &y, &opts)` | — | OLS `anova(lm0, lm1)` (see the old §4) |

**Gotchas of the anova table.** `table.row(label)` returns the *first* row whose trimmed label matches
exactly. Sub-row labels such as `" All Interactions"` and `" Nonlinear"` repeat under several predictors, so
select rows with `row.term` and `row.kind` instead (`AnovaRowKind::{MainEffect, CombinedEffect, Interaction,
CombinedInteractions, Nonlinear, NonlinearInteraction, …, TotalNonlinear, TotalInteraction,
TotalNonlinearInteraction, Total}`). As in rms, `TOTAL INTERACTION` is left out when the model has only one
interaction. Pass `F{df_residual}` for OLS to get rms's F rows; use `ChiSquare` for GLM, Cox and AFT fits.
**Any `V̂` works here**: model-based, cluster (below), `BootCov.vcov` or penalized.

```rust
let fit = OlsRegressor::builder().build().fit(&x, &col(&y))?;
let r = fit.result();
let (beta, v) = drop_intercept(&full_beta(r.intercept.unwrap(), &r.coefficients), &fit.vcov().unwrap());
let opts = AnovaOptions::default()
    .distribution(WaldDistribution::F { df_residual: r.residual_df() as f64 }).sigma2(Some(r.mse));
let tab = anova(&beta, &v, des.terms(), &opts)?;          // rows "age  (Factor+Higher Order Factors)", " Nonlinear …", …, "TOTAL"
let cols: Vec<usize> = des.terms().terms().iter()         // every nonlinear column of the design
    .flat_map(|t| t.columns.iter().zip(&t.nonlinear).filter(|(_, &nl)| nl).map(|(&c, _)| c)).collect();
let w = wald_chunk(&beta, &v, &cols, WaldDistribution::ChiSquare)?;
let sd = fastbw(&beta, &v, 0, des.terms(), StepRule::Aic { k: 2.0, threshold: 0.0 }, StepType::Residual);
let ap = approximate_model(&x, &lp, &fit.vcov().unwrap(), des.terms(), &[a, s])?; // ap.r_squared
// helper used above: fn full_beta(b0: f64, s: &Col<f64>) -> Col<f64> { Col::from_fn(s.nrows()+1, |j| if j==0 {b0} else {s[j-1]}) }
```

### Likelihood tests (`inference::likelihood`, R oracles `anova(..., test="LRT"/"Rao")`, `statmod::glm.scoretest`, `confint.glm`; 1e-6)
- `lr_test(&reduced, &full) -> ChiSqTest{statistic, df, p_value}`. The statistic is `(−2ℓ_red + 2ℓ_full)/φ_full`
  on `k_full − k_red` d.f. `model_lr_test(&fit) -> Option<ChiSqTest>` is rms's "Model L.R.".
  Both take any `LikelihoodFit`: `FittedBinomial, FittedPoisson, FittedTweedie, FittedNegativeBinomial` (uses
  −2logL because θ differs), `FittedGamma, FittedAft, CoxFit, GlmFit`. **`FittedLogistic` does not implement
  it**: fit with `BinomialRegressor::logistic()` or use `.inner()`. The two fits must be nested and fitted to
  the same rows; nothing checks this.
- `fit_glm(&family, &GlmData::new(&x, &y).with_intercept(true).weights(Some(&w)).offset(Some(&o)), Dispersion::{Fixed(φ), Estimated})`
  is a compact IRLS fit that returns `GlmFit{coefficients (intercept first), vcov, deviance, null_deviance, dispersion, df_residual, mu, iterations}`.
- `glm_score_test(&family, &reduced_data, &x_add, dispersion) -> ScoreTest{test, z}` is the Rao test, evaluated only at the reduced fit.
- `profile_confint(&family, &data, dispersion, level) -> Vec<(lo, hi)>` (intercept first) reproduces R ≥ 4.4
  `confint(glm)`, including its grid and spline. A bound the grid does not reach is NaN (R's `NA`).
  `LikelihoodError::NotConverged` means profiling found a better optimum than the fit.
- The family types are `BinomialFamily::logistic()/probit()/cloglog()`, `PoissonFamily`, and the others in `core`.

```rust
let fam = BinomialFamily::logistic();
let (full, red) = (BinomialRegressor::logistic().build().fit(&x_full, &y)?, BinomialRegressor::logistic().build().fit(&x_red, &y)?);
let lr = lr_test(&red, &full);                            // lr.statistic, lr.df, lr.p_value
let m = model_lr_test(&full).unwrap();
let sc = glm_score_test(&fam, &GlmData::new(&x_red, &y_vec), &x_add, 1.0)?;
let ci = profile_confint(&fam, &GlmData::new(&x_full, &y_vec), Dispersion::Fixed(1.0), 0.95)?;
let g = heuristic_shrinkage(m.statistic, m.df as f64);    // 0.75 → expect ~25% overfitting
```

### Robust covariance (`inference::cluster`; oracles `sandwich::vcovCL` 1e-10, `rms::robcov`, `survreg(cluster=)` 1e-5)
| Model | Call | Notes |
|---|---|---|
| OLS / WLS | `fit.cluster_vcov(&x, &[&codes], &ClusterOptions::cr1())?`, or `lm_cluster_vcov(&x, &resid, Option<&w>, with_intercept, &aliased, clusters, &opts)` | `ClusterType::{HC0, HC1, HC2 (Bell–McCaffrey CR2), HC3}`. HC2 and HC3 are for linear models only |
| any GLM family | `glm_cluster_vcov(&family, &x, with_intercept, &y, &mu, Option<&prior_w>, clusters, &opts)` | `mu` = fitted means (`result().fitted_values`) |
| AFT | `fitted_aft.cluster_vcov(&x, &time, &event, clusters, &opts)`; `FittedAft::from_result(fit_aft(..)?)` (0.12) turns a `fit_aft` result into the fitted model | `survreg(cluster=)` with `robcov()` |
| Cox | `cox_fit.robust_vcov(&clusters)` | `coxph(cluster=)` |
| ordinal (lrm/orm) | `fitted_ordinal.cluster_vcov(&x, &y, clusters, &opts)` | `rms::robcov(orm)` with `robcov()` |
| anything | `cluster_sandwich(&bread, &scores, clusters, &opts)` | bread = inverse information (k×k); scores n×k |

- `ClusterOptions` presets: `::cr1()` (the default; `vcovCL` lm default, Stata CR1), `::hc0()` (G/(G−1) only;
  the `vcovCL` GLM default) and `::robcov()` (no correction; `rms::robcov`). You can also set
  `ClusterOptions{kind, cadjust}` with the setters. `clusters: &[&[usize]]` takes one integer-code vector per
  dimension. Two-way clustering is inclusion–exclusion. Code arbitrary keys with `solvers::cv::group_codes(&keys) -> (codes, G)`.
- **The result is intercept first.** Pass it through `drop_intercept` before `anova`.
- **GLM / AFT entry points (0.12).** `glm_result.cluster_vcov(&y, &x_cols, &[&cl], &ClusterOptions::hc0())` on
  the `GlmResult` of the `penalized_glm` fit functions (`fit_poisson`, `fit_logistic`, …; pass the same `y` / column-major `x`, offset column included; rows the
  fit dropped are dropped; result in fitted order, intercept first) = `vcovCL(glm)`; `aft_result.cluster_vcov(&time,
  &x_cols, &event, &[&cl], &ClusterOptions::robcov())` on `fit_aft`'s result = `survreg(cluster=)`. With priors /
  ridge both use the **penalized bread** `(XᵀWX + P)⁻¹` (`rms::robcov` of a penalized `lrm`), whatever `vcov_type`
  the fit reports; `FittedAft::cluster_vcov` does the same since 0.12.
- **HC matrix.** `FittedOls::hc_vcov` / `FittedWls::hc_vcov` (or `inference::compute_hc_vcov(&x, &resid,
  Option<&w>, &aliased, with_intercept, HcType)`) return the full `vcovHC` matrix (intercept first, zero rows for
  aliased columns). For a GLM HC0 sandwich, pass singleton
  clusters `(0..n)` with `ClusterOptions::robcov()` to `glm_cluster_vcov`. By construction this is
  `sandwich::sandwich(glm)`, but it is not separately R-tested. The penalized GLM has its own `VcovType`.

### Multiple imputation (`inference::pool`; oracles `Hmisc::fit.mult.impute`, `mice::pool`, `mice::D1`; 1e-10)
`pool_rubin(&[Col], &[Mat], df_complete: Option<f64>)? -> PooledFit{m, coefficients, vcov (T = Ū+(1+1/m)B),
within, between, variance_inflation, riv, lambda, df_rubin, df_barnard_rubin, fmi, df_complete}` and
`.std_errors()`. It needs `m ≥ 2`. `pooled_wald_d1(&pooled, &idx)? -> PooledWald{f, df1, df2, p_value, riv}`.
With `df_complete` set, df2 is Reiter's, as `mice::D1` computes it (changed in 0.8). Pass the fits in the same
layout throughout (e.g. intercept-first `vcov()`). The indices refer to that layout.
```rust
let p = pool_rubin(&coefs, &vcovs, Some(197.0))?;          // coefs[i] = [b0, b..], vcovs[i] = fit.vcov().unwrap()
let w = pooled_wald_d1(&p, &[1, 2])?;                    // w.f, w.df1, w.df2, w.p_value
```

## 10. Effects: datadist, summary, contrasts, Predict, nomogram

Module `anofox_regression::effects` (oracles: `rms::datadist/summary.rms/contrast.rms/Predict` 1e-9,
`nomogram` 1e-10). A **`ModelSpec`** records how each predictor enters the model, and it builds the design:
`spec.linear(name, &x)`, `spec.rcs(name, Rcs, &x)`, `spec.lsp(name, Lsp, &x)`, `spec.categorical(name, &codes)`
(levels sorted, treatment dummies for every level but the first, columns named `grp=1`…).
`spec.interaction(a, b)` and `spec.restricted_interaction(a, b)` add interactions. Then
`spec.design(&[&x1, &x2, …])? -> Design` puts the main effects in predictor order, then the interactions.
Fit on `design.matrix()`. **Every effect method takes the full `beta` and `vcov`, intercept first when
`intercept = true`** (exactly what `fit.vcov()` returns).
- `Limits::numeric(&x)` (datadist) gives `{low_effect, adjust_to, high_effect, low_prediction, high_prediction, low, high}`:
  the quartiles, the median and the `10/max(n,200)` quantiles, with rms's rules for fewer than 4 distinct
  values. A categorical predictor adjusts to its mode, and its effect limits are NaN. Read them with
  `spec.predictors()[k].limits`.
- `Critical::{Normal, StudentT(df)}`. Use `StudentT(residual_df)` for OLS and `Normal` for GLM, Cox and AFT.

| Method | Returns |
|---|---|
| `spec.summary(&beta, &v, intercept, crit, level)` | `Vec<EffectRow{label, predictor, low, high, effect: ContrastEstimate}>`: the IQR effect of each numeric predictor, and every level vs the adjust-to level of a factor (label `"grp - 0:2"`). Take `exp()` of the estimate and limits for OR/HR. `summary.rms` |
| `spec.contrast(&beta, &v, intercept, &[(k, va)], &[(k, vb)], crit, level)?` | `Result<ContrastEstimate{estimate, se, lower, upper, statistic, p_value}>` (0.12: `row`, `contrast`, `double_contrast`, `predict_grid*` return `Result`; a value that is not a level of a categorical predictor is `InvalidInput`, it used to be coded as the reference). Unlisted predictors are held at adjust-to |
| `spec.double_contrast(.., a, b, a2, b2, ..)` | `(a−b)−(a2−b2)`: an interaction effect |
| `spec.predict_grid(&beta, &v, intercept, k, Option<Vec<f64>>, crit, level)` | `(values, Vec<ContrastEstimate>)` on the **linear-predictor scale** (with the intercept); the default grid is 200 points between the prediction limits (the levels for a categorical predictor, 0.12). Apply `plogis`/`exp` yourself |
| `spec.predict_grid_simultaneous(..)` | `(values, est, crit_value)`, a simultaneous band (`Predict(conf.type="simultaneous")`) |
| `effects::simultaneous_critical(&rows, &v, crit, level)` | the multcomp single-step max-\|t\| critical value for any set of contrast rows. It is deterministic (Halton): within ~2e-4 of a tight mvtnorm average (R itself is ±1e-3). It costs about 10⁸ ops for 50 rows of rank 15 |
| `spec.row(&[(k, v)], intercept)?` | the design row for a setting, for your own `cᵀβ` |
| `spec.add_predictor(name, kind, Limits::new(low_effect, adjust_to, high_effect, low_prediction, high_prediction, low, high), distinct)?`, `spec.categorical_with_levels(name, &levels)?` (0.12) | rebuild a stored spec without its data (kind / limits / distinct from `predictors()`), or fix a factor's levels independently of the data subset (adjusted to the first level) |
| `spec.nomogram(&beta, intercept, &lp, &[Option<Vec<f64>>], maxscale, nint)?` | `Nomogram{axes: Vec<NomogramAxis{name, values, xbeta, points}>, scale, intercept, total_points, lp_at, lp_points}`, `.points_to_lp(t)`, `.lp_to_points(lp)`, `.function_axis(\|l\| plogis, nint) -> (ticks, positions)`. **Main effects only** (`DesignError::Invalid` otherwise). `lp` = the fit's linear predictors |

```rust
let mut spec = ModelSpec::new();
let age = spec.rcs("age", Rcs::from_data(&age_v, 4)?, &age_v);
let sex = spec.linear("sex", &sex_v);
let _grp = spec.categorical("grp", &grp_v);               // codes 0,1,2 → dummies grp=1, grp=2
let des = spec.design(&[&age_v, &sex_v, &grp_v])?;
let x = des.matrix();
let fit = BinomialRegressor::logistic().compute_inference(true).build().fit(&x, &col(&y_bin))?;
let r = fit.result();
let beta = full_beta(r.intercept.unwrap(), &r.coefficients);
let v = fit.vcov().unwrap();                               // intercept first, as ModelSpec expects
for e in spec.summary(&beta, &v, true, Critical::Normal, 0.95) { println!("{} OR {:.3}", e.label, e.effect.estimate.exp()); }
let c = spec.contrast(&beta, &v, true, &[(age, 70.0)], &[(age, 40.0)], Critical::Normal, 0.95)?;
let (_, band, crit) = spec.predict_grid_simultaneous(&beta, &v, true, age, Some(vec![35.0, 45.0, 55.0, 65.0, 75.0]), Critical::Normal, 0.95)?;
let nom = spec.nomogram(&beta, true, &lp, &[None, None, None], 100.0, 5)?;
let (risk, pos) = nom.function_axis(|l| 1.0 / (1.0 + (-l).exp()), 5);
```

## 11. Validation: optimism bootstrap, CV, .632, calibration, bootcov

Module `anofox_regression::validation` (oracles: `rms::validate` and `calibrate` 1e-5/1e-6, `bootcov` 1e-5,
`calibrate.cph` 1e-9, all with R's resamples replayed). **The crate's RNG is not R's.** Seeded resamples
differ from R; to reproduce R exactly, pass R's samples (0-based) to `Resamples::bootstrap(n, samples)`.

| Entry point | Arguments | Result |
|---|---|---|
| `Resamples::bootstrap_seeded(n, B, seed)?` / `::bootstrap(n, samples)?` | — | Efron–Gong optimism. Test = all rows |
| `Resamples::crossvalidation_seeded(n, B, seed)?` / `::crossvalidation(&perm, B)?` | — | B-fold with rms's group boundaries. Neighbouring groups can share or skip a row when n/B is not an integer, as in rms |
| `Resamples::point632_seeded(n, B, seed)?` / `::point632(n, samples)?` | — | Efron .632 |
| `validate_lrm(&x, &y01, &res)?` | x without intercept | `ValidationTable`, indexes `LRM_INDEXES = [Dxy, R2, Intercept, Slope, Emax, D, U, Q, B, g, gp]`. `validate.lrm` |
| `validate_ols(&x, &y, &res)?` | — | `OLS_INDEXES = ["R-square", "MSE", "g", "Intercept", "Slope"]`. `validate.ols` |
| `calibrate_lrm(&x, &y01, &res, Option<predy>)?`, `calibrate_ols(..)` | default grid: 50 points between the 5th smallest and largest prediction | `Calibration{predy, calibrated_orig, calibrated_corrected, optimism, n, mean_abs_error, mean_squared_error, quantile90_abs_error}`. `calibrate` (lowess) |
| `calibrate_cph(&x, &time, &event, u, &res, m, Option<cuts>, &CoxOptions)?` | `m` = subjects per group | `SurvivalCalibration{predicted, cuts, index_orig, training, test, optimism, corrected, n, mean_predicted, km, km_corrected, std_err}`. `calibrate.cph(cmethod="KM")`. Refits with fewer than 5 events are skipped. Use rms's `eps=1e-4`, 10 iterations to match `cph` |
| `predab_resample(&res, &names, fit: FnMut(&[usize]) -> Option<M>, measure: FnMut(&M, &[usize], apparent: bool) -> Vec<f64>)?` | **replays the whole modelling process** (selection, tuning) inside `fit` | `ValidationTable` (`rms:::predab.resample`). A `None` fit is skipped and counted in `failed` |
| `bootcov(&samples, fit: FnMut(&[usize]) -> Option<Vec<f64>>)?` | samples from `bootstrap_rows(n, B, seed)` or `cluster_bootstrap_rows(&codes, B, seed)` | `BootCov{replicates (B×k), mean, vcov (÷(B−1)), failed}`, `.percentile_ci(level)`, `.linear_combination_ci(&c, level)`. It needs ≥ 2 successful replicates |
| `rank_intervals(&orig_importance, &replicates_B×m, level)` | e.g. χ²−df per predictor | `Vec<(rank, lo, hi)>`, where rank 1 is the least important (RMS §5.4) |
| `predab_resample_par`, `bootcov_par` | feature `parallel` (rayon); closures must be `Fn + Sync` | identical to the sequential results |

`ValidationTable{names, index_orig, training, test, optimism, corrected, n, failed}`, and `.get(name) ->
Option<(orig, training, test, optimism, corrected, n)>`. **`validate_*` / `calibrate_*` refit a fixed design**
(logistic: `tol 1e-12`, 100 iterations). Any data-driven step (fastbw, knot choice, pentrace) has to go
through `predab_resample` to be counted in the optimism.

```rust
let boot = Resamples::bootstrap_seeded(n, 200, 42)?;
let v = validate_lrm(&x, &y_bin, &boot)?;
let (_, _, _, _, slope, _) = v.get("Slope").unwrap();      // corrected calibration slope
let cal = calibrate_lrm(&x, &y_bin, &boot, None)?;          // cal.mean_abs_error, cal.quantile90_abs_error
let tab = predab_resample(&boot, &["C"],
    |rows| { let xr = Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]);
             let yr = Col::from_fn(rows.len(), |i| y_bin[rows[i]]);
             BinomialRegressor::logistic().build().fit(&xr, &yr).ok() },
    |fit, rows, _apparent| { let xr = Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]);
             let p = fit.predict(&xr); let ps: Vec<f64> = (0..rows.len()).map(|i| p[i]).collect();
             let yr: Vec<f64> = rows.iter().map(|&i| y_bin[i]).collect();
             vec![somers2(&ps, &yr, None).map_or(f64::NAN, |s| s.c)] })?;
let bc = bootcov(&cluster_bootstrap_rows(&cluster, 200, 7), |rows| {
    let xr = Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]);
    let yr = Col::from_fn(rows.len(), |i| y_bin[rows[i]]);
    let f = BinomialRegressor::logistic().build().fit(&xr, &yr).ok()?; let r = f.result();
    Some(std::iter::once(r.intercept?).chain(r.coefficients.iter().copied()).collect()) })?;
let (beta, vb) = drop_intercept(&full_beta(b0, &slopes), &bc.vcov);   // bootstrap anova / contrasts
```

## 12. Accuracy indexes and separation

`anofox_regression::diagnostics::accuracy` contains free functions over plain slices, so they work with any
model (oracles: Hmisc, rms and survival, 1e-12). **The argument order is not consistent across them. Check each one:**

| Function | Signature | Returns / R |
|---|---|---|
| `concordance` | `(x_pred, y, Option<w>)` | `Concordance{concordant, discordant, tied_x, tied_y, tied_xy}`, `.c() .dxy() .gamma() .tau_a()`. `survival::concordancefit` |
| `somers2` | `(p, y01, Option<w>)` | `Somers2{c, dxy, n}`. `Hmisc::somers2` |
| `rcorr_cens` | `(x, &time, &event: &[bool])` — **larger x = longer survival** (pass `-lp` for a hazard score) | `CensoredConcordance{c, dxy, sd, n, missing, uncensored, relevant_pairs, concordant, uncertain}`. `Hmisc::rcorr.cens` |
| `brier_score` | `(y, p, Option<w>)` → `f64` | |
| `gini_md` | `(&lp)` → the g-index | `Hmisc::GiniMd` |
| `lrm_stats` | `(y01, lp_with_intercept, Option<w>, df)` | `LrmStats{obs, sum_weights, model_lr, df, p_value, c, dxy, gamma, tau_a, r2 (Nagelkerke), r2_measures, effective_sample_size, brier, g, gr, gp}`. the `rms::lrm` stats vector (`effective_sample_size` = the ess of rms's `R2(ess)` labels, since 0.11.1) |
| `val_prob` | `(p, y01)` | `ValProb{dxy, c, r2, d, d_chisq, d_p, u, u_chisq, u_p, q, brier, intercept, slope, emax, e90, eavg, s_z, s_p}`. `rms::val.prob` (external validation); p of exactly 0 / 1 handled as rms (p = 0 with y = 1 gives `u = ∞`, `q = −∞`) since 0.11.1 |
| `r2_measures` | `(lr, p, n, Option<ess>)` | `R2Measures{r2, r2_adj, r2_ess, r2_adj_ess}`. `Hmisc::R2Measures`. `effective_sample_size(n, &freqs)` |
| `nagelkerke_r2` | `(lr, null_deviance, n)` | |

`diagnostics::detect_separation(&x, &y01, with_intercept) -> LinearSeparation{separated, infinite (±inf/0,
intercept first), direction}` (in the prelude). It solves the exact Konis linear program (`detectseparation`), so it
also catches separation by a *combination* of predictors, which `check_binary_separation` misses. Run it
before any logistic fit that has sparse cells.

```rust
let s = lrm_stats(&y_bin, &lp, None, 2.0)?;               // s.model_lr, s.c, s.dxy, s.r2, s.brier, s.g
let c = concordance(&lp, &y_bin, None)?.c();
let vp = val_prob(&p, &y_bin)?;                            // vp.intercept, vp.slope, vp.emax, vp.s_z
let cc = rcorr_cens(&lp.iter().map(|v| -v).collect::<Vec<_>>(), &time, &event)?;
let sep = detect_separation(&xs, &ys, true);               // sep.separated, sep.infinite = [-inf, inf, inf]
```

## 13. Penalized maximum likelihood: pentrace

`anofox_regression::solvers::{…}` (oracle `rms::lrm/ols(penalty=)`, `pentrace`; 1e-5 lrm, 1e-10 ols). The
penalty is `½βᵀPβ` with `P = A P₀ A`. `P₀` puts the column variances on the diagonal (the coefficients are
standardized), with an `I − 1/(L+1)` block for the dummies of a factor. `A = diag(√λ)` sets λ by column type.

| Function | Arguments | Result |
|---|---|---|
| `PenaltySpec::default().simple(λ).nonlinear(Some(..)).interaction(Some(..)).nonlinear_interaction(Some(..))`, `PenaltySpec::uniform(λ)` | unset components fall back: nonlinear → simple, interaction → nonlinear, nl-interaction → interaction | `.resolved() -> [f64; 4]` |
| `penalty_multipliers(&terms, &spec)` | — | λ per slope column (`Penalty.setup`) |
| `base_penalty_matrix(&x, &terms, &categorical)?`, `penalty_matrix(&x, &terms, &categorical, &spec)?` | `categorical` = **term indices** of the factor terms | p×p (slopes) |
| `penalized_ols(&x, &y, &P)?` | the intercept is unpenalized | `PenalizedOls{coefficients, vcov, vcov_sandwich, residuals, sigma, r_squared, lr_chisq, effective_df_diagonal, effective_df}`, **all intercept first** |
| `OrdinalRegressor::builder().penalty_matrix(P).build().fit(&x, &y)?` | binary or ordinal; `.link(OrdinalLink::{Logistic, Probit, LogLog, CLogLog, Cauchit})` | `FittedOrdinal`: `intercepts()`, `coefficients()`, `vcov()`, `vcov_sandwich()`, `effective_df_diagonal()` (**slopes only**), `effective_df()`, `lr_chisq()` |
| `penalty_grid(&simple, &nonlinear, &interaction, &nl_inter)` | an empty slice leaves that component unset | `Vec<PenaltySpec>` (expand.grid with `complex.more=TRUE`) |
| `pentrace(&x, &y, &terms, &categorical, &grid, link, criterion)?` / `pentrace_ols(&x, &y, &terms, &categorical, &grid, criterion)?` | `PentraceCriterion::{CorrectedAic (default), Aic, Bic}` | `Pentrace{rows: Vec<PentraceRow{penalty, df, lr_chisq, aic, bic, aic_c}>, best, best_fit}` |
| `effective_df_by_term(&terms, &diag)` | `diag` must be **slopes only** | Vec of effective d.f. per term |

**`PenalizedOls.effective_df_diagonal` is intercept first and `FittedOrdinal::effective_df_diagonal()` is
slopes only.** Strip the first entry from the former before you call `effective_df_by_term`. For Wald tests
or anova on a penalized fit, use `vcov_sandwich` (rms `var.penalty="sandwich"`) or the simple `vcov`, and
report the effective d.f.

```rust
let cat = [g];                                             // term index of the factor
let spec = PenaltySpec::default().simple(1.0).nonlinear(Some(4.0)).interaction(Some(4.0));
let pm = penalty_matrix(&x, des.terms(), &cat, &spec)?;
let po = penalized_ols(&x, &col(&y), &pm)?;
let diag = Col::from_fn(x.ncols(), |j| po.effective_df_diagonal[j + 1]);   // drop the intercept
let by_term = effective_df_by_term(des.terms(), &diag);
let grid = penalty_grid(&[0.0, 2.0, 8.0, 32.0], &[], &[], &[]);
let tl = pentrace(&x, &col(&y_bin), des.terms(), &cat, &grid, OrdinalLink::Logistic, PentraceCriterion::Aic)?;
let best = &tl.rows[tl.best];                              // best.penalty.resolved(), best.df; tl.best_fit is a FittedOrdinal
```

## 14. Survival: Kaplan–Meier, log-rank, Cox (`survival`, `survival::cox`)

Module `anofox_regression::survival` (not in the prelude). Data are **slices**: `time: &[f64]`,
`event: &[bool]` (true = event), strata/groups/clusters `&[usize]`. Errors are
`SurvivalError::{Length, Invalid(&str), Empty}` (they convert into `anofox_regression::Error` and work with `?` into
`Box<dyn Error>`). Oracles: `tests/r_validation_survival.rs`, `r_validation_survfit_robust.rs`,
`r_validation_cox.rs`, `r_validation_coxdiag.rs` and `issue208_cox_concordance.rs` (data `survival::lung`/`heart`).

| Entry point | Options / inputs | Returns |
|---|---|---|
| `kaplan_meier(time, event, &SurvfitOptions)` (R `survfit(Surv(t,d)~1)`) | no strata, no weights | `SurvCurve` |
| `survfit(&SurvInput, strata: Option<&[usize]>, &SurvfitOptions)` (R `survfit(Surv(...)~strata)`) | `SurvInput::new(time, event).entry(&e).weights(&w).cluster(&id)`: `entry` gives left truncation / `(entry, stop]`, `cluster` is `survfit(id=)` | `Vec<SurvCurve>`, one per stratum code in increasing order |
| `SurvfitOptions::default()` + setters `.conf_level(0.95) .conf_type(ConfType::{Log (default), LogLog, Plain}) .hazard(HazardType::{NelsonAalen (default), FlemingHarrington}) .timefix(true) .robust(None)` | `robust: None` follows R: the IJ robust SE is used when the weights are not all integers **or a cluster is given**, Greenwood otherwise. `Some(false)` forces Greenwood | |
| `survdiff(time, event, group, strata: Option<&[usize]>, rho)` (R `survdiff`) | `rho = 0.0` log-rank, `1.0` Peto–Peto; stratified if `strata` | `SurvDiff{groups, n, observed, expected, variance: Mat, chisq, df, p_value}` |

`SurvCurve` fields (all `Vec<f64>` over the distinct times): `stratum: Option<usize>, records, time, n_risk, n_event,
n_censor, surv, std_err, lower, upper, cumhaz, std_chaz`. Methods: `median() -> SurvQuantile{estimate, lower, upper}`
(NaN when the curve/limit never reaches 0.5), `restricted_mean(τ) -> (rmean, se)`, `summary(τ) -> SurvSummary{records,
n_max, n_start, events, rmean, se_rmean, median, median_lower, median_upper}` (= `summary(survfit)$table`; R's default τ
is the last time of the curve, or with strata the largest last time over the strata: pass that τ yourself), and `survival_at(t)` (right-continuous step).
**`std_err` is the SE of −log S (= SE(S)/S), not SE(S)**. With robust SEs R prints SE(S) itself.

```rust
use anofox_regression::survival::{kaplan_meier, survdiff, survfit, ConfType, HazardType, SurvInput, SurvfitOptions};
let km = kaplan_meier(&time, &event, &SurvfitOptions::default())?;
let m = km.median();                                   // m.estimate, m.lower, m.upper
let (rm, se) = km.restricted_mean(5.0);                // RMST to t = 5
let opts = SurvfitOptions::default().conf_type(ConfType::LogLog).hazard(HazardType::FlemingHarrington);
let curves = survfit(&SurvInput::new(&time, &event), Some(&grp), &opts)?;
for c in &curves { let s = c.summary(*c.time.last().unwrap()); println!("{:?} {} {}", c.stratum, s.events, s.median); }
let lr = survdiff(&time, &event, &grp, None, 0.0)?;    // lr.chisq, lr.df, lr.p_value, lr.observed, lr.expected
let inp = SurvInput::new(&time, &event).weights(&w).entry(&entry).cluster(&cl);   // weighted, truncated, IJ SEs
let c = survfit(&inp, None, &SurvfitOptions::default())?.remove(0);
```

### Cox PH: `survival::cox::coxph(x, &CoxInput, &CoxOptions) -> Result<CoxFit, SurvivalError>`
R `survival::coxph` / `rms::cph`. `x: &Mat<f64>` has **no intercept column** (at least one covariate). Newton–Raphson with
step halving, as coxph.

| Builder / option | Meaning (default) |
|---|---|
| `CoxInput::new(&time, &event)` `.weights(&w)` `.offset(&o)` `.strata(&s)` `.entry(&start)` | case weights; offset on the log-hazard scale; separate baseline hazards per stratum; counting-process `(start, stop]` data (left truncation, time-dependent covariates via split rows) |
| `CoxOptions::default()` `.ties(Ties::{Efron, Breslow})` | Efron (coxph default) |
| `.max_iter(20) .eps(1e-9) .conf_level(0.95) .timefix(true)` | coxph defaults (`eps` is the relative change in the log-likelihood) |
| `.ridge(λ)` | penalty λ/2·Σβ², = `coxph(... ridge(x, theta=λ, scale=FALSE))`. Unscaled: **standardise x first** |

`CoxFit` public fields: `coefficients: Vec<f64>` (**log hazard ratios**; exp them for HRs), `var: Mat` (inverse
(penalised) information = `vcov`), `std_errors, z, p_values` (Wald z), `loglik: (ℓ₀, ℓ̂)`, `lr_test, wald_test,
score_test` (χ² on `df`), `df, n, n_events, iterations, converged`. Methods:

| Method | R equivalent |
|---|---|
| `test_p_value(chisq)` | p on `df` d.f. (for `lr_test` etc.) |
| `concordance()`, `dxy()`, `r2()` | `coxph` `concordance` (timewt "n", weights and entry times respected, within strata), `rms::cph` Dxy and R² `(1−e^{−LR/n})/(1−e^{2ℓ₀/n})` |
| `linear_predictors()` | `xβ + offset` of the fitted rows, **uncentred** (R centres them) |
| `martingale_residuals()`, `deviance_residuals()` → `Vec<f64>`; `score_residuals()`, `dfbeta()`, `dfbetas()` → `Mat n×p` | `residuals.coxph(type=…)`; dfbeta is weighted (survival ≥ 3) |
| `schoenfeld_residuals(scaled) -> (rows, times, Mat events×p)` | `type="schoenfeld"/"scaledsch"`, event rows ordered by stratum then time |
| `robust_vcov(&clusters) -> Result<Mat, ClusterError>` | `coxph(..., cluster=id)`. Pass `0..n` for `robust=TRUE` |
| `cox_zph(ZphTransform::{Km (default), Rank, Identity, Log}, terms: Option<&[Vec<usize>]>) -> CoxZph{terms: Vec<ZphTest{chisq, df, p_value}>, global, x, time, y}` | `cox.zph` (survival ≥ 3 score test). `terms` groups columns, e.g. all spline columns of one predictor. `x`/`y` are the transformed time and the scaled Schoenfeld residuals for plotting β(t) |
| `predict_survival(&x0, stratum: Option<usize>) -> CoxSurvCurve{time, n_risk, cumhaz, surv, std_err, lower, upper}` | `survfit(fit, newdata)`: one **uncentred** covariate row. `std_err` is the SE of the cumulative hazard and includes the uncertainty of β̂; the limits are log-type |

```rust
use anofox_regression::survival::cox::{coxph, CoxInput, CoxOptions, Ties, ZphTransform};
let fit = coxph(&x, &CoxInput::new(&time, &event), &CoxOptions::default())?;
let hr: Vec<f64> = fit.coefficients.iter().map(|b| b.exp()).collect();
println!("LR {:.2} p {:.2e} C {:.3} Dxy {:.3} R2 {:.3}", fit.lr_test, fit.test_p_value(fit.lr_test),
         fit.concordance(), fit.dxy(), fit.r2());
let zph = fit.cox_zph(ZphTransform::Km, None)?;        // zph.terms[j].p_value, zph.global.p_value
let rv = fit.robust_vcov(&cluster_ids)?;              // sandwich, coxph(cluster = id)
let (rows, t, sch) = fit.schoenfeld_residuals(true);
let curve = fit.predict_survival(&[1.0, 2.0], None)?; // S(t | x0) with CI
let k = curve.time.partition_point(|&s| s <= 3.0) - 1; // S(3): curve.surv[k], curve.lower[k], curve.upper[k]
let input = CoxInput::new(&time, &event).strata(&st).weights(&w).entry(&entry);
let f2 = coxph(&x, &input, &CoxOptions::default().ties(Ties::Breslow).ridge(1.0))?;
let c2 = f2.predict_survival(&[0.0, 1.0], Some(st[0]))?;   // a stratified fit needs Some(stratum)
```
Also related: `validation::calibrate_cph(&x, &time, &event, u, &resamples, m, cuts: Option<Vec<f64>>, &CoxOptions)` (`rms::calibrate.cph`,
KM method) and `CoxFit` implements `LikelihoodFit` (for LR tests). AFT diagnostics added with #115:
`FittedAft::standardized_residuals(&x, &time)` (`(log t − η)/σ`) and `cox_snell_residuals(&x, &time)` (both `time: &Col`) (`−log S(t|x)`; compare their KM
against Exp(1)). **Limitations:** no `predict` of linear predictors for new data (compute `x_new·β` yourself; the
coefficient vector is uncentred); no frailty/`tt()` terms; `predict_survival` takes one row at a time and has no offset argument.

## 15. Ordinal regression (`solvers::ordinal`, `solvers::continuation_ratio`)

`OrdinalRegressor` = **`rms::orm` / `lrm`** (and `MASS::polr`). Model `P(Y ≥ y_{j+1}|x) = F(α_j + xᵀβ)`, with the intercepts
`α₁ > … > α_{k−1}` decreasing. **polr's ζ_j = −α_j**, with the same β (positive β gives higher Y). A continuous Y gets one
intercept per distinct value (orm's semiparametric model, tridiagonal Newton, cheap). Imports come from
`anofox_regression::solvers::{OrdinalRegressor, OrdinalLink, OrdinalQuantileMethod, xmean_ordinaly, …}` (not the prelude).
Oracles: `r_validation_ordinal.rs`, `r_validation_ordinal_penalty.rs`, `r_validation_quantile_orm.rs`, `r_validation_210_part10.rs`.

| Builder (`OrdinalRegressor::builder()`) | Default / meaning |
|---|---|
| `link(OrdinalLink::{Logistic, Probit, LogLog, CLogLog, Cauchit})` | Logistic (proportional odds). Names as orm `family`; CLogLog = discrete PH |
| `weights(Col)`, `offset(Col)` | frequency weights; offset added to the linear predictor |
| `penalty(λ)` | `λ·diag(var(x_j))` = orm/lrm `penalty=λ` for simple numeric predictors (**scaled by the sample variances**) |
| `penalty_matrix(P)` | `½βᵀPβ` (`penalty.matrix`), e.g. from `solvers::pentrace` |
| `max_iterations(100)`, `tolerance(1e-10)` | Newton. The fit is `.fit(&x, &y)` with x without an intercept and y the ordered values (any reals) |

| `FittedOrdinal` method | Notes / R |
|---|---|
| `levels()`, `counts()`, `intercepts() -> &Col` (α, length k−1), `coefficients() -> &Col` (β) | orm `coef` = (α, β) |
| `log_likelihood()`, `penalized_log_likelihood()`, `null_log_likelihood()`, `lr_chisq()`, `lr_df()`, `lr_p_value()`, `aic()`, `effective_df()`, `effective_df_diagonal()`, `n_observations()`, `converged()`, `iterations()` | the LR test uses the effective d.f. when penalized |
| `vcov_slopes()` (O(kp²)), `std_errors()` (slopes), `vcov()` (full, **intercepts first**, dense k²), `vcov_sandwich()` | penalized: `var.penalty="simple"` vs `"sandwich"` |
| `score_contributions(&x, &y)`, `cluster_vcov(&x, &y, &[&cl], &ClusterOptions::robcov())` | `rms::robcov(fit, cluster)`. **Pass the fit's own x, y**. `ClusterOptions` is in `anofox_regression::inference` |
| `linear_predictor(&x)` (xβ, no α, no offset), `exceedance_probabilities(&x)` (n×(k−1)), `category_probabilities(&x)` (n×k) | `predict(orm, type="fitted"/"fitted.ind")` |
| `predict_mean(&x)`, `predict_mean_with_se(&x) -> (mean, se)` | `Mean.orm` (delta method) |
| `predict_quantile(&x, q, method)`, `predict_quantile_ci(&x, q, method, level) -> (est, lo, hi)` | `Quantile.orm`. `OrdinalQuantileMethod::{Interpolated (default), Discrete}` |
| `z_values()`, `p_values()` (0.12) | Wald z = β/SE of the slopes and normal p-values (orm `Wald Z`) |
| `FittedOrdinal::from_parts(link, levels, intercepts, coefficients, &vcov)`, `from_information(.., information())` (0.12) | rebuild a stored model: predictions exact, SE-based results to rounding (`from_parts` inverts the dense vcov, O(k³)); `information() -> OrdinalInformation{intercept_diagonal, intercept_off_diagonal, cross, slopes}` is the compact O(kp) form for continuous Y. Likelihood summaries of a rebuilt model are NaN |
| `proportional_odds_score_test(&x, &y, &cols) -> ChiSqTest{statistic, df, p_value}` | score test against partial PO for `cols` (empty = all). d.f. = \|cols\|·(k−2). Dense in k: only for few categories |

`xmean_ordinaly(&x_col, &y, cr: bool) -> XmeanOrdinaly{levels, observed, expected_po, expected_cr: Option}` gives the
numbers of `plot.xmean.ordinaly` (RMS §13.3.3), a univariable check of the PO/CR assumption.

`ContinuationRatioRegressor` = **`cr.setup` + `lrm`**: logit `P(Y=y_j | Y≥y_j, x) = θ_j + xᵀγ`. The builder takes
`cohort_specific(&[cols])` (cohort×x interactions, i.e. non-equal slopes) and `weights(&[f64])` (per subject). `.fit(&x, &y)` needs ≥ 3
levels. `FittedContinuationRatio`: `levels()`, `coefficients()` (**intercept θ₁, then the cohort effects θ_j−θ₁ for
j=2..k−1, the slopes γ, then the cohort-specific differences, cohort fastest**), `vcov()`, `log_likelihood()`,
`conditional_probabilities(&x)`, `category_probabilities(&x)`, `n_features()`, `logistic() -> &FittedBinomial` (the
full binomial fit on the expanded data), `cohort_specific()`. `FittedContinuationRatio::from_parts(levels, n_features,
&cohort_specific, &coefficients, Some(&vcov))` (0.12) rebuilds a stored model. `cr_setup(&y) -> ContinuationRatioData{y, cohort, subs, reps, levels}` is the
raw expansion (cluster on `subs` for bootstrap/robust SEs). Oracle: `r_validation_continuation_ratio.rs`.

```rust
use anofox_regression::inference::ClusterOptions;
use anofox_regression::solvers::{cr_setup, xmean_ordinaly, ContinuationRatioRegressor, OrdinalLink,
    OrdinalQuantileMethod, OrdinalRegressor};
let fit = OrdinalRegressor::builder().link(OrdinalLink::Logistic).build().fit(&x, &y)?;
println!("alpha {:?} beta {:?} LR {:.2} p {:.2e}", fit.intercepts(), fit.coefficients(), fit.lr_chisq(), fit.lr_p_value());
let pc = fit.category_probabilities(&x0);                         // 1 × k
let (mu, se) = fit.predict_mean_with_se(&x0);
let (q, lo, hi) = fit.predict_quantile_ci(&x0, 0.5, OrdinalQuantileMethod::Interpolated, 0.95);
let po = fit.proportional_odds_score_test(&x, &y, &[])?;          // po.statistic, po.df, po.p_value
let v = fit.cluster_vcov(&x, &y, &[&cluster_ids], &ClusterOptions::robcov())?;   // (k-1+p)², intercepts first
let pen = OrdinalRegressor::builder().penalty(2.0).build().fit(&x, &y)?;          // pen.effective_df()
let xm = xmean_ordinaly(&Col::from_fn(n, |i| x[(i, 0)]), &y, true)?;
let cr = ContinuationRatioRegressor::builder().cohort_specific(&[1]).build().fit(&x, &y)?;
let d = cr_setup(&y)?;                                            // d.y, d.cohort, d.subs
```

## 16. GLS for serial correlation (`solvers::gls`), R `nlme::gls`

`GlsRegressor::builder()` sets `.correlation(GlsCorrelation)`, `.variance(GlsVariance)`, `.method(GlsMethod::{Reml (default),
Ml})`, `.with_intercept(true)`, `.confidence_level(0.95)` and `.max_iterations(500)` (L-BFGS on the var–cov parameters). Fit with
**`.fit(&x, &y, &group: &[usize], time: Option<&[f64]>)`**; `None` uses the position within the group (`form = ~1|id`).
`GroupedRegressor::fit_grouped(x, y, groups)` is the same with `time = None`. Oracles: `r_validation_gls.rs`, `r_validation_gls_210.rs`.

| `GlsCorrelation` | nlme | `correlation_params()` |
|---|---|---|
| `Independent` (default) | — | empty |
| `CompoundSymmetry` | `corCompSymm` | ρ |
| `Ar1` (integer times / position) · `ContinuousAr1` | `corAR1` · `corCAR1` | φ |
| `Exponential{nugget}` · `Gaussian{nugget}` · `Linear{nugget}` · `Spherical{nugget}` | `corExp/corGaus/corLin/corSpher` | range (, nugget) |
| `Unstructured` | `corSymm` (positions = distinct sorted times) | r₁₂, r₁₃, …, r₂₃, … |

`GlsVariance::{Constant (default), Ident(Vec<usize>) (varIdent; the stratum of the first obs is the reference), Power(Vec<f64>)
(varPower ~v), Exponential(Vec<f64>) (varExp ~v)}`, and `variance_params()` returns the SD ratios or δ.
`FittedGls`: `coefficients()` (**intercept first**), `vcov()`, `std_errors()`, `t_values()`, `p_values()` and
`conf_int() -> (lo, hi)` (t on N−p, as `summary.gls`); `sigma()`, `log_likelihood()`, `aic()`, `bic()` (REML uses N−p),
`n_params()`, `df_residual()`; `var_cov_intervals() -> Option<&VarCovIntervals{correlation, variance, sigma}>` with each
an `Interval{lower, estimate, upper}` (`intervals.gls`; `None` if the Hessian is not negative definite);
`fitted_values()`, `residuals(GlsResidualType::{Response, Pearson, Normalized (default)})`, `predict(&x)`,
`predict_with_se(&x) -> (mean, se)` (x without the intercept), `variogram(&time, kind, nint) -> Vec<VariogramBin{variog,
dist, n_pairs}>` (`Variogram(fit, form=~t|id)`, quantile collapse), and `converged()`/`iterations()`.
**Compare fixed effects by LR only with `GlsMethod::Ml`**; REML likelihoods of different X are not comparable.
Data that cannot carry the structure are `RegressionError::Gls(GlsError::{NoRepeatedMeasures, TooFewTimes,
NonIntegerTimes, InvalidVarianceCovariate, OptimisationFailed})` since 0.12 (were `InvalidInput` strings).

```rust
use anofox_regression::solvers::{GlsCorrelation, GlsMethod, GlsRegressor, GlsResidualType, GlsVariance};
let fit = GlsRegressor::builder()
    .correlation(GlsCorrelation::Ar1).variance(GlsVariance::Ident(arm.clone())).method(GlsMethod::Reml)
    .build().fit(&x, &y, &subject, Some(&time))?;
let (lo, hi) = fit.conf_int();
println!("phi {:?} sd-ratio {:?} sigma {:.3} AIC {:.2}", fit.correlation_params(), fit.variance_params(), fit.sigma(), fit.aic());
if let Some(iv) = fit.var_cov_intervals() { println!("phi [{:.3}, {:.3}]", iv.correlation[0].lower, iv.correlation[0].upper); }
let (m, se) = fit.predict_with_se(&x_new);
let vg = fit.variogram(&time, GlsResidualType::Pearson, 10)?;   // check the AR(1) fit: variog should be flat for normalized residuals
```

## 17. Multinomial / penalized logistic (sklearn objective) vs `LogisticRegressor`

| | `LogisticRegressor` (prelude) | `solvers::MultinomialLogisticRegressor` (new 0.10.0) |
|---|---|---|
| Target | binary 0/1 only | K ≥ 2 classes, any distinct finite labels (`classes()` sorted) |
| Fit | IRLS of `BinomialRegressor` (logit) = `glm(binomial)` | proximal Newton = sklearn `LogisticRegression` |
| Penalty | **default none**. `l2(λ)`, `c(C)` (L2 only) | **default `lambda = 1` (sklearn C = 1)**. `.lambda(λ)` / `.c(C)`, `.l1_ratio(α)` (0 ridge, 1 lasso, exact zeros); intercepts unpenalized |
| Weights | none | `.weights(Col)` (sample_weight; fold class weights in yourself) |
| Inference | SEs, Wald z, CIs via `inner().result()`, `vcov()` | **none** (no SEs, no vcov): use it for prediction only |
| Output | `coefficients() -> &Col`, `intercept()`, `predict` (labels), `predict_proba -> Col` | `coefficients() -> &Mat` (1×p for 2 classes: log-odds of the 2nd class; K×p for K≥3), `intercepts() -> &Col` (sum to 0 for K≥3), `decision_function`, `predict_proba -> Mat n×K`, `predict`, `score` (accuracy), `log_likelihood()`, `objective()`, `converged()`; `FittedMultinomialLogistic::from_parts(classes, coefficients, intercepts)` (0.12) rebuilds a stored model |

Objective: `Σ sᵢℓᵢ + λ(α‖B‖₁ + (1−α)/2‖B‖²)`. The loss is **not averaged over n**, unlike glmnet. For K ≥ 3 the softmax
parameters are centred to sum to zero across classes (sklearn's symmetric parameterisation, not a baseline category
like `nnet::multinom`; subtract the reference row to get baseline log-odds). Other builder options: `with_intercept(true)`,
`max_iterations(100)`, `tolerance(1e-8)`, `error_on_non_convergence(true)`. Oracle: `tests/issue307_multinomial_logistic.rs`
(sklearn). Verified: with `lambda(0.0)` and 2 classes the coefficients equal `LogisticRegressor` to 1e-11.
```rust
use anofox_regression::solvers::MultinomialLogisticRegressor;
let fit = MultinomialLogisticRegressor::builder().c(10.0).l1_ratio(0.5).build().fit(&x, &y)?;  // y in {0,1,2}
let p = fit.predict_proba(&x);                     // n × 3, columns in fit.classes() order
let b = fit.coefficients();                         // 3 × p (centred)
let mb = MultinomialLogisticRegressor::builder().lambda(0.0).build().fit(&x, &y01)?;  // == glm logit point estimates
```
For an RMS-style analysis of a nominal outcome there is no inference here. For an **ordered** outcome use `OrdinalRegressor`.

## 18. Response transformations (`transform`)

| Function | R | Notes |
|---|---|---|
| `box_cox(&x, &y, &BoxCoxOptions) -> BoxCoxProfile{lambdas, log_likelihood, lambda_hat, max_log_likelihood, ci_lower: Option, ci_upper: Option}` | `MASS::boxcox(lm(y~x))` | y > 0. Options `.lambdas(grid)` (default `seq(-2,2,0.1)`), `.eps(1/50)`, `.with_intercept(true)`, `.confidence_level(0.95)`. `lambda_hat` is searched within the grid range; a CI limit is `None` if the profile does not cross the cut within the grid |
| `box_cox_transform(y, λ)`, `box_cox_inverse(z, λ)` | — | scalar; `λ = 0` gives log/exp |
| `smearing_estimate(&lp, &residuals, inverse_fn, Smearing::{Mean, Median, Quantile(q), Fitted}) -> Result<Vec<f64>>` | `Hmisc::smearingEst` | Duan: `Mean` = mean of g⁻¹(lp+rᵢ); `Fitted` = naive g⁻¹(lp) (a median when the residuals are symmetric) |
| `areg(&x, &y: &[f64], &AregOptions) -> Areg` | `Hmisc::areg` | options `.nk(4)` (0 = linear), `.x_types(Some(vec![AregType::{Spline, Linear, Categorical}]))`, `.y_type(AregType::Spline)` |

`Areg` fields: `x_coefficients, y_coefficients` (intercept first), `r_squared, x_df, y_df, transformed_y,
transformed_x` (centred), `linear_predictors, residuals, mad: Option, med: Option`. Methods: `y_knots()`,
`transform_y(&y)`, `predict(&x_new) -> Vec` (on the **transformed-y** scale), `inverse_y(&t) -> Option<Vec>` (back to
the y scale; `None` if the transform is not monotone), `bootstrap(&indices: &[Vec<usize>]) -> AregBootstrap{r2_optimism,
r2_corrected, mad_optimism, mad_corrected, med_optimism, med_corrected}` (you supply the resample row indices).
Store / rebuild (0.12): `Areg::from_parts(fit.x_bases(), fit.x_coefficients.clone(), fit.y_basis(),
fit.y_coefficients.clone(), fit.y_range())?` with `AregBasis::{Linear, Spline{knots}, Categorical{levels}}`; the rebuilt
model transforms, predicts and back-transforms new rows as the fit. `box_cox_inverse` is `NaN` where `λz + 1 ≤ 0`
(no y > 0 maps there; it returned a value before 0.12).
Oracles: `r_validation_transform.rs`, `r_validation_areg.rs`.
```rust
use anofox_regression::transform::{areg, box_cox, box_cox_inverse, box_cox_transform, smearing_estimate,
    AregOptions, AregType, BoxCoxOptions, Smearing};
let prof = box_cox(&x, &y, &BoxCoxOptions::default())?;      // prof.lambda_hat, prof.ci_lower, prof.ci_upper
let z = Col::from_fn(n, |i| box_cox_transform(y[i], 0.0));
let ols = OlsRegressor::builder().build().fit(&x, &z)?;
let lp: Vec<f64> = (0..n).map(|i| ols.result().fitted_values[i]).collect();
let res: Vec<f64> = (0..n).map(|i| ols.result().residuals[i]).collect();
let mean_y = smearing_estimate(&lp, &res, |v| box_cox_inverse(v, 0.0), Smearing::Mean)?;   // E[Y|x]
let a = areg(&x, &yv, &AregOptions::default().nk(4).y_type(AregType::Spline))?;           // yv: Vec<f64>
let yhat = a.inverse_y(&a.predict(&x_new));                                               // Option<Vec<f64>>
let b = a.bootstrap(&boot_indices)?;                                                       // b.r2_corrected
```

## 19. Data reduction, blinded to Y (`reduction`), Hmisc equivalents

| Function | R | Returns / notes |
|---|---|---|
| `spearman2(&x, &y, p)` (p = 1 or 2), `spearman2_categorical(&codes, &y)` | `Hmisc::spearman2` | `Spearman2{rho2, f, df1, df2, p_value, n, adjusted_rho2}`. p=2 adds rank² (non-monotone); x with < 3 distinct values falls back to p=1. **Uses Y**: screening for d.f. allocation only, never for selection |
| `hoeffding_d(&x: &Mat)` | `Hmisc::hoeffd` | `HoeffdingD{d, p_value}`, d = 30·D. O(p²n²) |
| `hclust(&diss, Linkage::{WardD, Single, Complete, Average, Mcquitty, Median, Centroid})` | `stats::hclust` | `Hclust{merge: Vec<(Node::Leaf/Cluster, ..)>, height, order}`, `cut_height(h) -> Vec<usize>` labels |
| `varclus(&x, Similarity::{Spearman, Pearson, Hoeffding}, SimilarityTransform::{Square, Abs, None}, Linkage)` | `Hmisc::varclus` | `Varclus{sim, hclust}`; cluster on 1 − sim. Read clusters off with `vc.hclust.cut_height(1 − ρ²)` |
| `redun(&x, &[RedunVariable::{Numeric, Linear, Categorical}], &RedunOptions)` | `Hmisc::redun` | options `.r2(0.9) .adjusted(false) .nk(3) .tlinear(true) .rank .qrank .allcat`; `Redun{kept, removed, r_squared, r_squared_later, r_squared_first, too_few}` |
| `principal_components(&x, scale: bool)` | `prcomp` | `PrincipalComponents{center, scale, sdev, rotation, scores}`, `proportion_of_variance()`, `transform(&x_new, k)`. Signs fixed so the largest loading is positive |
| `cluster_scores(&x, &[Vec<usize>], k)` | RMS §8 summary scores | first k PCs of each standardised cluster, side by side |
| `transcan(&x, &categorical: &[bool], &TranscanOptions)` | `Hmisc::transcan(method="canonical")` | options `nk: Option` (None: 3/4/5 by n), `iter_max(50)`, `eps(0.1)`, `curtail(true)`; NaN = missing (single imputation on the transformed scale). `Transcan{transformed, r_squared, r_squared_adj, iterations}`. **Sign of each transformation is arbitrary** |

Oracles: `r_validation_reduction.rs`, `r_validation_redun.rs`, `r_validation_transcan.rs`.
```rust
use anofox_regression::reduction::{cluster_scores, hoeffding_d, principal_components, redun, spearman2, transcan,
    varclus, Linkage, RedunOptions, RedunVariable, Similarity, SimilarityTransform, TranscanOptions};
let s = spearman2(&x0, &y, 2)?;                                    // s.rho2, s.adjusted_rho2, s.p_value
let vc = varclus(&x, Similarity::Spearman, SimilarityTransform::Square, Linkage::Complete)?;
let labels = vc.hclust.cut_height(0.5);                            // merges with 1-rho^2 <= 0.5 kept together
let rd = redun(&x, &[RedunVariable::Numeric; 5], &RedunOptions::default().r2(0.8))?;   // rd.kept, rd.removed
let pc = principal_components(&x, true)?;                          // pc.proportion_of_variance()
let sc = cluster_scores(&x, &[vec![0, 1], vec![2], vec![3, 4]], 1)?;   // n × 3 summary scores
let tc = transcan(&x_with_nan, &[false; 5], &TranscanOptions::default())?;
```

## 20. Multiple imputation: `imputation::areg_impute` (Hmisc `aregImpute`)

`areg_impute(&x: &Mat (NaN = missing, ≥ 2 cols), &[ImputeVariable::{Numeric, Linear, Categorical}], &AregImputeOptions)
-> AregImpute{missing: Vec<Vec<usize>>, imputed: [col][row][m], r_squared}`. Options: `.n_impute(5) .nk(3) .tlinear(true)
.burnin(None)` (None: 3, or 0 when only one variable has NA), `.matching(DonorMatch::{Weighted(0.2) (default), Closest})`,
`.seed(1)` (R's RNG: replays `set.seed(seed); aregImpute(...)` exactly). PMM on bootstrap areg fits (chained equations).
`imp.complete(&x, m)` returns completed data set m. **Put Y (and every analysis variable) in the imputation matrix**, as
RMS §3.7 recommends. Then fit per data set and pool with `inference::pool::pool_rubin(&estimates: &[Col], &vcovs: &[Mat],
df_complete: Option<f64>) -> PooledFit` (Rubin; also `pooled_wald_d1` for chunk tests; documented elsewhere). Oracle:
`r_validation_aregimpute.rs`.
```rust
use anofox_regression::imputation::{areg_impute, AregImputeOptions, ImputeVariable};
use anofox_regression::inference::pool::pool_rubin;
let imp = areg_impute(&data, &[ImputeVariable::Numeric; 4], &AregImputeOptions::default().n_impute(10).seed(42))?;
let (mut est, mut vc) = (vec![], vec![]);
for m in 0..10 {
    let d = imp.complete(&data, m);                      // y is column 3
    let xm = Mat::from_fn(n, 3, |i, j| d[(i, j)]);
    let ym = Col::from_fn(n, |i| d[(i, 3)]);
    let f = OlsRegressor::builder().build().fit(&xm, &ym)?;
    let (b, b0) = (f.coefficients(), f.intercept().unwrap());
    est.push(Col::from_fn(4, |k| if k == 0 { b0 } else { b[k - 1] }));   // intercept first, matching vcov()
    vc.push(f.vcov().ok_or("no vcov")?);                 // FittedOls::vcov() -> Option<Mat>, intercept first
}
let pooled = pool_rubin(&est, &vc, Some((n - 4) as f64))?;     // pooled.coefficients, pooled.vcov, pooled.fmi
```

## 21. GLMM changes since 0.5.35

- **`GlmmRegressorBuilder::n_agq(k)`** (#67), lme4's `nAGQ`: `0` = **still the crate default** (Laplace with β from
  PIRLS, = `glmer(nAGQ=0)`); `1` = Laplace with β optimised jointly with θ (**`glmer`'s default**; SEs from the deviance
  Hessian in (θ, β)). Set `.n_agq(1)` to reproduce a default `glmer` fit. `k > 1` is adaptive Gauss–Hermite, only for a
  **single scalar random intercept**. `fit_crossed` with several factors and a non-Gaussian family
  accepts only `n_agq = 0`; anything else returns `InvalidParameter{name: "n_agq"}`, as does `k > 1` with slopes. Ignored by the Gaussian family. Oracle `r_validation_glmm_nagq.rs` (lme4 2.1.0; ~1e-6).
- Accessors return faer columns (#179): `fixed_effects()`, `std_errors()`, `slopes()`, `random_effects()`,
  `fitted_values()`, `fitted_population()`, `pearson_residuals()` return `ColRef`; `z_values()`, `p_values()`,
  `random_effects_se()`, `random_sd()` and `residuals()` return `Col`. `vcov()` is still `&[Vec<f64>]`, and `conf_int(level)` is still `(Vec, Vec)`.
- `GroupedRegressor::fit_grouped(x, y, groups)` (trait in `solvers`) is implemented by GLMM and GLS; `IterativeFit`
  (`converged()`, `iterations()`, `ensure_converged()?`) is in the prelude.
- `deviance()` is documented as `−2(logLik − saturated)` for GLMMs; `n_parameters`/AIC count a fixed NB θ as lme4 does.
- Optional `serde` feature (0.11): every fitted model above (`FittedGlmm`, `FittedGls`, `FittedOrdinal`,
  `FittedContinuationRatio`, `FittedMultinomialLogistic`, `CoxFit`, `CoxZph`, `CoxSurvCurve`, `SurvCurve`) serialises.
```rust
for k in [0usize, 1, 7] {        // binomial random intercept: nAGQ = 0 / Laplace (glmer default) / AGQ-7
    let f = GlmmRegressor::binomial().n_agq(k).build().fit(&x, &y, &group)?;
    println!("nAGQ={k}: {:?} se {:?} sd_b {:.3} ll {:.3}", f.fixed_effects(), f.std_errors(), f.sd_random(), f.log_likelihood());
}
```
