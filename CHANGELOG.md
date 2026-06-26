# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
