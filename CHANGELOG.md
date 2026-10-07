# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.14] - 2026-10-07

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
