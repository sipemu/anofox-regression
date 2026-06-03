# Validation Report

This document describes how the Rust `anofox-regression` library is validated against established statistical software to ensure numerical accuracy and correctness.

## Overview

The validation suite uses two oracle ecosystems, chosen per estimator:

- **R** (`base stats`, `MASS`, `glmnet`, `quantreg`, `greybox`, …) for the GLM-style and classic regression methods. R scripts live under `tests/r_scripts/`.
- **scikit-learn** (Python) for the estimators whose canonical reference is sklearn — `TheilSenRegressor`, `RANSACRegressor`, `PassiveAggressiveRegressor`, `Lars`, `LassoLars`, `BayesianRidge`, `ARDRegression`. The Python oracle scripts and a pinned, reproducible `uv`-managed venv live under `validation/python/`.

Both pipelines emit a `tests/fixtures/*_validation.rs` file of `const` arrays that Rust integration tests `include!(...)` and assert agreement against, with per-estimator tolerances.

## Core Architecture

The validation system uses a three-component structure:

1. **Reference Generation**: R scripts generate expected values using `set.seed(42)` for reproducibility
2. **Test Data Storage**: Generated data is stored in the `validation/` folder and embedded in Rust test files
3. **Rust Test Suites**: Integration tests load reference data and verify numerical agreement between implementations

## Key Components

### Reference Generation Scripts

#### R-based oracles

| Script | Purpose |
|--------|---------|
| `validation/generate_validation_data.R` | Generates OLS, Ridge, Elastic Net, WLS, and diagnostic test cases |
| `tests/r_scripts/generate_regression_validation.R` | Validates WLS, Ridge, Elastic Net, and Tweedie regressors |
| `tests/r_scripts/generate_glm_validation.R` | Validates GLM implementations (Poisson, Binomial, Gamma, etc.) |
| `tests/r_scripts/generate_gamma_validation.R` | Validates `GammaRegressor` against `glm(family = Gamma(link="log"))` |
| `tests/r_scripts/generate_alm_validation.R` | Validates ALM implementations (7 core distributions) |
| `tests/r_scripts/generate_alm_validation_extended.R` | Validates ALM extended distributions (14 additional) |
| `tests/r_scripts/generate_alm_loss_validation.R` | Validates ALM loss function implementations |
| `tests/r_scripts/generate_aid_validation.R` | Validates AID demand classification |
| `tests/r_scripts/generate_quantile_validation.R` | Validates Quantile Regression using `quantreg` package |
| `tests/r_scripts/generate_quantile_extended_validation.R` | Extended Quantile validation with Engel/Stackloss datasets |
| `tests/r_scripts/generate_isotonic_validation.R` | Validates Isotonic Regression using base R `isoreg()` |
| `tests/r_scripts/generate_huber_validation.R` | Validates `HuberRegressor` against `MASS::rlm` |
| `tests/r_scripts/generate_logistic_validation.R` | Validates `LogisticRegression` against `glm(family = binomial)` |
| `tests/r_scripts/generate_pls_validation.R` | Validates PLS regression |

#### scikit-learn-based oracles (Python, `validation/python/`)

| Script | Purpose |
|--------|---------|
| `validation/python/generate_theil_sen_validation.py` | `TheilSenRegressor` — univariate exact + multivariate exhaustive branch |
| `validation/python/generate_ransac_validation.py` | `RANSACRegressor` with `LinearRegression` base estimator |
| `validation/python/generate_pa_validation.py` | `PassiveAggressiveRegressor` PA-I and PA-II variants |
| `validation/python/generate_lars_validation.py` | `Lars`, `LassoLars`, plus path truncation by `n_nonzero_coefs` |
| `validation/python/generate_bayesian_validation.py` | `BayesianRidge` and `ARDRegression` evidence-maximisation fits |

The Python environment is reproducible: `validation/python/requirements.txt` pins `scikit-learn==1.5.2` and `numpy==2.1.3`, and `validation/python/README.md` documents the exact `uv` setup commands. Re-running each script regenerates the corresponding `tests/fixtures/<name>_validation.rs` file bit-for-bit on the pinned versions.

### R Packages Used

- **base stats**: `lm()`, `glm()`, `isoreg()` for core regression, GLM, and isotonic regression
- **MASS**: `glm.nb()` for negative binomial regression, `rlm(method="M", psi=psi.huber)` for Huber
- **glmnet**: Ridge regression, Elastic Net, Lasso with coordinate descent
- **statmod**: Tweedie family distributions, inverse Gaussian
- **greybox**: `alm()` for augmented linear models, `aid()` for demand identification
- **quantreg**: `rq()` for quantile regression

### Python Packages Used

- **scikit-learn 1.5.2**: `TheilSenRegressor`, `RANSACRegressor`, `PassiveAggressiveRegressor`, `Lars`, `LassoLars`, `BayesianRidge`, `ARDRegression`
- **numpy 2.1.3**: deterministic `default_rng(seed)` for fixture sampling

## Validation Categories

### 1. Ordinary Least Squares (OLS)

Validates coefficients, standard errors, t-statistics, p-values, R², adjusted R², F-statistic, AIC, BIC, log-likelihood, and residuals.

**Tolerance**: `1e-8` for coefficients and standard errors

**Test cases**:
- Simple linear regression (n=20, p=1)
- Multiple regression (n=50, p=2)

### 2. Weighted Least Squares (WLS)

Validates coefficient estimation with observation weights for heteroscedastic data.

**Tolerance**: `0.01` for coefficients

**Test cases**:
- Weights inversely proportional to variance (1/x²)
- Comparison with OLS on same data

### 3. Ridge Regression

Validates L2-penalized regression with closed-form solution: β = (X'X + λI)⁻¹X'y

**Tolerance**: `0.01` for coefficients

**Test cases**:
- λ = 0 (should match OLS)
- λ = 0.1, 1.0, 10.0 (increasing shrinkage)
- Collinear data (VIF > 16000)

### 4. Elastic Net

Validates combined L1+L2 penalty using coordinate descent optimization.

**Objective**: ||y - Xβ||² + λ(1-α)||β||² + λα||β||₁

**Tolerance**: `0.2` for coefficients (coordinate descent may produce slight differences)

**Test cases**:
- α = 0.5, λ = 0.3 (balanced penalty)
- α = 1.0 (Lasso)
- Sparse coefficient recovery

### 5. Poisson GLM

Validates count data regression with log, identity, and sqrt link functions.

**Tolerance**: `0.01` for coefficients, `0.1` for deviance

**Test cases**:
- Log link (standard Poisson)
- Identity link
- With offset (exposure modeling)

### 6. Binomial GLM

Validates binary/proportion response regression with multiple link functions.

**Tolerance**: `0.1` for coefficients, `0.5` for deviance

**Test cases**:
- Logit link (logistic regression)
- Probit link
- Complementary log-log link

### 7. Tweedie GLM

Validates exponential dispersion models covering Poisson, Gamma, and Inverse Gaussian.

**Tolerance**: `0.05` - `0.2` depending on variance power

**Test cases**:
- Gamma (var.power = 2)
- Poisson (var.power = 1)
- Inverse Gaussian (var.power = 3)
- Compound Poisson-Gamma (var.power = 1.5)

### 8. Negative Binomial GLM

Validates overdispersed count data regression with theta parameter.

**Tolerance**: `0.05` for coefficients

**Test cases**:
- Fixed theta (from R's `glm.nb()`)
- Theta estimation
- High theta approaching Poisson

### 9. RLS (Recursive Least Squares)

Validates online/streaming regression that updates coefficients incrementally.

**Tolerance**: `0.01` - `0.1` (compared to batch OLS)

**Test cases**:
- Convergence to OLS with forgetting factor = 1.0
- Forgetting factor behavior (exponential weighting of recent data)
- Online updates

### 10. BLS/NNLS (Bounded/Non-Negative Least Squares)

Validates constrained regression with coefficient bounds.

**Tolerance**: `0.1` for coefficients

**Test cases**:
- NNLS (all coefficients ≥ 0) vs R's `nnls` package
- Box constraints with arbitrary lower/upper bounds
- Comparison with OLS when OLS solution satisfies constraints

### 11. Regression Diagnostics

Validates model diagnostic statistics against R's `lm()` influence measures.

**Tolerance**: `1e-6` to `1e-8`

**Metrics validated**:
- Leverage (hat values)
- Cook's distance
- Studentized residuals
- Variance Inflation Factor (VIF)

### 12. ALM (Augmented Linear Model)

Validates maximum likelihood estimation for 21 distribution families against R's `greybox::alm()` function.

**Tolerance**: `0.15` for coefficients, `0.20` for log-likelihood

**Distributions validated** (grouped by data type):

| Category | Distributions | Link Function | Optimizer | Status |
|----------|---------------|---------------|-----------|--------|
| Symmetric Continuous | Normal, Laplace, Logistic, StudentT | Identity | IRLS | ✓ All validated |
| Robust | GeneralisedNormal | Identity | IRLS | ✓ Validated |
| Robust | AsymmetricLaplace | Identity | IRLS | ✓ Validated |
| Robust | S | Identity | L-BFGS | ✓ Validated (better LL than R) |
| Log-domain | LogNormal, LogLaplace, LogGeneralisedNormal | Log | IRLS | ✓ All validated |
| Log-domain | LogS | Log | L-BFGS | ✓ Implemented |
| Positive Continuous | Gamma, Exponential | Log | IRLS | ✓ Validated |
| Positive Continuous | InverseGaussian | Log | IRLS | ✓ Validated |
| Unit Interval (0,1) | LogitNormal | Identity* | IRLS | ✓ Validated |
| Unit Interval (0,1) | Beta | Logit | L-BFGS | ✓ Validated |
| Zero-inflated | FoldedNormal | Identity | L-BFGS | ✓ Validated (better LL than R) |
| Zero-inflated | RectifiedNormal | Identity | L-BFGS | ✓ Validated |
| Transform | BoxCoxNormal | Identity (transformed) | L-BFGS | ✓ Validated |
| Count | Poisson, Geometric | Log | IRLS | ✓ Validated |
| Count | NegativeBinomial | Log | IRLS | ✓ Validated |
| Count | Binomial | Logit | IRLS | ✓ Validated |
| Cumulative | CumulativeLogistic, CumulativeNormal | Logit/Probit | L-BFGS | ✓ Validated |

*LogitNormal uses Identity link on logit-scale location parameter (R greybox parameterization)

**Optimizer Notes**:
- **IRLS**: Iteratively Reweighted Least Squares - fast, reliable for standard GLM-like distributions
- **L-BFGS**: Limited-memory BFGS optimization via argmin crate - used for distributions with complex likelihood surfaces (FoldedNormal, S, Beta, BoxCoxNormal, RectifiedNormal)

**Test cases**:
- Each distribution with n=50 observations
- Simple linear regression (1 predictor + intercept)
- Validates intercept, coefficient, scale parameter, and log-likelihood

### 13. AID (Automatic Identification of Demand)

Validates demand classification against R's `greybox::aid()` function.

**Components validated**:
- Demand type classification (Regular vs Intermittent)
- Fractional vs count data detection
- New product detection (leading zeros)
- Obsolete product detection (trailing zeros)
- Stockout detection (unexpected zeros)
- Information criteria (AIC, BIC, AICc)

**Test cases**:
- Regular count demand (Poisson-like, 0% zeros)
- Regular fractional demand (Normal-like)
- Intermittent count demand (65% zeros)
- Intermittent fractional demand (39% zeros)
- New product (30 leading zeros)
- Obsolete product (30 trailing zeros)
- Stockouts (3 unexpected zeros in middle)
- Overdispersed count (Negative Binomial)
- Skewed positive (Gamma/LogNormal)
- IC comparison (AIC vs BIC vs AICc)

### 14. Quantile Regression

Validates quantile regression against R's `quantreg::rq()` function.

**Algorithm**: Iteratively Reweighted Least Squares (IRLS) with asymmetric weights based on the check function ρ_τ(u) = u(τ - I(u < 0)).

**Tolerance**: `0.01` for coefficients

**Test files**:
- `tests/r_validation_quantile.rs`: Core R validation (6 tests)
- `tests/r_validation_quantile_extended.rs`: Extended R validation with classic datasets (11 tests)
- `tests/quantile_edge_cases.rs`: Edge cases and input validation (22 tests)

**Test cases**:
- Median regression (τ = 0.5) - robust alternative to OLS
- Multiple quantiles (τ = 0.1, 0.25, 0.5, 0.75, 0.9) on same dataset
- Weighted quantile regression (heteroscedastic data)
- No-intercept model
- Real-world heteroscedastic data (variance increasing with x)
- Upper quantile regression (τ = 0.9)

**Classic R datasets validated**:
- Engel food expenditure dataset (n=235, Koenker & Bassett 1982)
- Stackloss multivariate dataset (n=21, 3 predictors)

**Edge cases**:
- Input validation (dimension mismatch, invalid tau, negative weights)
- Collinearity handling (constant features, perfect collinearity)
- Convergence stability (large/small coefficients, extreme tau values)
- Special data patterns (constant y, alternating values)
- Minimum sample sizes (2-3 observations)

**Features validated**:
- Coefficient estimation for different quantile levels
- Fitted values at specified quantiles
- Weighted observations support
- Intercept/no-intercept modes
- Outlier robustness vs OLS
- Heavy-tailed distribution handling

### 15. Isotonic Regression

Validates isotonic regression against R's base `isoreg()` function.

**Algorithm**: Pool Adjacent Violators Algorithm (PAVA) for monotonic constraint fitting.

**Tolerance**: `1e-6` for fitted values (closed-form solution)

**Test files**:
- `tests/r_validation_isotonic.rs`: Core R validation (7 tests)
- `tests/isotonic_edge_cases.rs`: Edge cases and input validation (27 tests)

**Test cases**:
- Simple increasing constraint (n=10)
- Decreasing constraint (via reflection)
- Weighted observations
- Data with ties (multiple observations at same x)
- Step function output (perfect monotonic data)
- Two observations edge case
- All equal values (constant output)

**Edge cases**:
- Input validation (dimension mismatch, negative weights, zero weights)
- PAVA ties handling (averaging, many ties, mixed with non-ties)
- Weighted PAVA with extreme weight ratios
- Out-of-bounds prediction modes (Clip, Nan)
- Monotonicity preservation verification
- Large dataset performance (n=10,000)
- Floating point edge cases (very small differences, large values)
- Special data patterns (step functions, mixed signs)
- R² bounds validation

**Features validated**:
- Fitted values under monotonic constraint
- Increasing vs decreasing modes
- Weighted PAVA algorithm
- Out-of-bounds handling (clip, nan, error)
- Prediction/interpolation for new x values
- Matrix interface (multi-column support)
- R² calculation and bounds

### 16. Stress Tests

Numerical stress tests and API edge cases beyond standard R validation.

**Test file**: `tests/stress_tests.rs` (19 tests)

#### Scale Invariance (Quantile)

Tests IRLS algorithm behavior relative to the smoothing parameter (epsilon = 1e-6).

| Test | Scale Factor | Expected Behavior |
|------|--------------|-------------------|
| Baseline | 1x | Coefficients match true values |
| Macro-scale | 1e8 | Coefficients scale proportionally |
| Micro-scale | 1e-8 | Requires smaller epsilon to avoid OLS degradation |
| Micro-default | 1e-8 (default ε) | Documents epsilon dominance effect |

#### Sawtooth PAVA Stress (Isotonic)

Worst-case input for PAVA block-merging with alternating pattern [10, 0, 10, 0, ...].

| Test | n | Constraint | Expected Result |
|------|---|------------|-----------------|
| Standard | 100 | Increasing | Flat line at mean (5.0) |
| Decreasing | 100 | Decreasing | Monotonic non-increasing |
| Large | 1000 | Increasing | Flat line at mean (50.0) |

#### Unsorted Input Handling (Isotonic)

Verifies API sorts X internally and returns results in original order.

| Test | Input | Verification |
|------|-------|--------------|
| Simple | x=[3,1,2], y=[30,10,20] | Predictions match original order |
| Matches sorted | Random permutation | Same predictions as pre-sorted |
| With violations | Unsorted + PAVA needed | Correct merging after sort |

#### Singular Matrix Safety (Quantile)

Ensures graceful handling of rank-deficient design matrices.

| Test | Deficiency Type | Expected Behavior |
|------|-----------------|-------------------|
| Zero-variance feature | Column of zeros | Error or finite coefficients |
| Collinear with intercept | Column of ones | Error or finite coefficients |
| Perfect collinearity | x₂ = 2x₁ | Error or finite predictions |
| Near-singular | x₂ ≈ x₁ + ε | Finite predictions |

#### f32/f64 Precision (Both)

Documents that the crate uses f64 exclusively; tests f32-range values work correctly.

| Test | Description | Tolerance |
|------|-------------|-----------|
| Quantile f32-range | Values cast from f32 | 1e-2 |
| Isotonic f32-range | Values cast from f32 | 1e-5 |
| Documentation | Compile-time API check | N/A |

#### Additional Stress Tests

| Test | Description |
|------|-------------|
| Difficult convergence | Heavy outliers (±500, ±1000) |
| Many ties | 10 unique x with 50 reps each (n=500) |

### 17. Gamma Regression

Validates `GammaRegressor` (a sklearn-style convenience wrapper around `TweedieRegressor` with `var_power = 2`) against R's `glm(family = Gamma(link = "log"))`.

**Oracle**: base R `glm`.
**Generator**: `tests/r_scripts/generate_gamma_validation.R`.
**Test file**: `tests/r_validation_gamma.rs`.
**Tolerance**: `1e-4` for coefficients, `1e-3` for deviance.

| Test | Description |
|------|-------------|
| Univariate Gamma | `n = 60`, simulated `y ~ rgamma(shape=2, rate=2/μ)` with `μ = exp(0.5 + 0.4 x)` |
| Multivariate Gamma | `n = 80`, two predictors, `μ = exp(0.2 + 0.6 x₁ − 0.3 x₂)` |

### 18. Theil-Sen Regression

Validates `TheilSenRegressor` against `sklearn.linear_model.TheilSenRegressor`.

**Algorithm**: For each subsample of `n_features + 1` observations, solve OLS exactly; take the Vardi-Zhang modified Weiszfeld spatial median of the resulting coefficient vectors. Exhaustive enumeration when `C(n, n_subsamples) ≤ max_subpopulation` matches sklearn's deterministic branch.
**Oracle**: scikit-learn 1.5.2.
**Generator**: `validation/python/generate_theil_sen_validation.py`.
**Test file**: `tests/r_validation_theil_sen.rs`.
**Tolerance**: `1e-10` (univariate, exact); `5e-3` (multivariate spatial median, covers Weiszfeld termination).

| Test | Description |
|------|-------------|
| Univariate exhaustive | `n = 30`, 1 feature, outliers injected; identical spatial median to sklearn |
| Multivariate exhaustive | `n = 14`, 2 features, `C(14, 3) = 364 < 10 000` ⇒ deterministic enumeration |

### 19. RANSAC Regression

Validates `RansacRegressor` against `sklearn.linear_model.RANSACRegressor` with `LinearRegression` base estimator.

**Algorithm**: Random subsample → OLS fit → inliers by `|residual| < threshold`; keep subset with most inliers, refit OLS on the consensus set. Uses Fischler–Bolles stop-probability bound to cap `max_trials` dynamically.
**Oracle**: scikit-learn 1.5.2 with a deliberately clean inlier/outlier gap so the consensus set is unique. Both implementations refit OLS on identical inlier sets, so coefficients match to machine precision.
**Generator**: `validation/python/generate_ransac_validation.py`.
**Test file**: `tests/r_validation_ransac.rs`.
**Tolerance**: `1e-9` for coefficients and intercept, exact match on inlier mask.

| Test | Description |
|------|-------------|
| Univariate | `n = 65` (50 inliers + 15 outliers), residual_threshold = 0.5 |
| Multivariate | `n = 100`, `p = 2`, 80 inliers / 20 outliers, residual_threshold = 0.3 |

### 20. Passive-Aggressive Regression

Validates `PassiveAggressiveRegressor` (online learning, PA-I and PA-II) against `sklearn.linear_model.PassiveAggressiveRegressor`.

**Algorithm**: For each sample, leave weights unchanged if epsilon-insensitive loss is zero; otherwise step `τ` such that the constraint is exactly satisfied (PA-I clamps to `C`; PA-II uses squared loss). `||x||²` is the bare feature norm (not augmented with `1` for the bias), matching sklearn's `_sgd_fast.pyx`. Multi-epoch `fit` uses sklearn's `best_loss` / `no_improvement_count` early-stopping criterion (5 epochs without `tol*n` improvement).
**Oracle**: scikit-learn 1.5.2 with `shuffle=False` so the per-epoch traversal order is identical and the algorithm is fully deterministic.
**Generator**: `validation/python/generate_pa_validation.py`.
**Test file**: `tests/r_validation_passive_aggressive.rs`.
**Tolerance**: `2e-2` for coefficients and intercept.

| Test | Description |
|------|-------------|
| PA-I univariate | `n = 100`, epsilon-insensitive loss, C = 1.0, ε = 0.1 |
| PA-I multivariate | `n = 80`, `p = 3`, epsilon-insensitive |
| PA-II multivariate | `n = 100`, `p = 2`, squared epsilon-insensitive, C = 0.5 |

### 21. LARS and LassoLars

Validates `LarsRegressor` (plain LARS) and the Lasso variant against `sklearn.linear_model.Lars` and `LassoLars`.

**Algorithm**: Efron–Hastie–Johnstone–Tibshirani LARS (2004). At each step adds the most-correlated predictor, walks the equiangular direction, and chooses step size by the joining condition (LARS) or by the joining/dropping condition (LassoLars). For `LassoLars` the returned coefficients are linearly interpolated between path knots so the result corresponds to exactly the requested `alpha` (matches sklearn's `lasso_path` convention).
**Oracle**: scikit-learn 1.5.2.
**Generator**: `validation/python/generate_lars_validation.py`.
**Test file**: `tests/r_validation_lars.rs`.
**Tolerance**: `1e-6` for plain LARS (deterministic linear algebra); `5e-3` for `LassoLars` (path-interpolation precision).

| Test | Description |
|------|-------------|
| LARS full path | `n = 60`, `p = 4`, all coefficients reach OLS (= unregularised OLS) |
| LARS truncated | `n = 80`, `p = 6`, `n_nonzero_coefs = 2` (two-step path) |
| LassoLars | `n = 100`, `p = 8`, `alpha = 0.1` with sklearn-style `(1/2n)` parametrisation |

### 22. BayesianRidge

Validates `BayesianRidge` against `sklearn.linear_model.BayesianRidge`.

**Algorithm**: SVD-based evidence maximisation (MacKay 1992 / Tipping 2001). Jointly updates noise precision α and weight precision λ via closed-form gamma-prior updates of `γ = Σ s² / (αs² + λ)`. Convergence on `Σ_j |β_new − β_old| < tol` matches sklearn.
**Oracle**: scikit-learn 1.5.2 with default gamma-prior shapes `α₁ = α₂ = λ₁ = λ₂ = 1e-6`.
**Generator**: `validation/python/generate_bayesian_validation.py`.
**Test file**: `tests/r_validation_bayesian.rs`.
**Tolerance**: `5e-3` for coefficients/intercept; `0.5` on α and `0.05` on λ.

| Test | Description |
|------|-------------|
| BayesianRidge defaults | `n = 100`, `p = 4`, `α ≈ 25.3`, `λ ≈ 0.6` |

### 23. ARD Regression

Validates `ArdRegression` against `sklearn.linear_model.ARDRegression`.

**Algorithm**: ARD prior with per-feature precisions `λ_j`. Per-iteration: Cholesky on `αX'X + diag(λ)` for the subset of un-pruned features; update γ_j and λ_j; prune features with `λ_j > threshold_lambda`.
**Oracle**: scikit-learn 1.5.2.
**Generator**: `validation/python/generate_bayesian_validation.py`.
**Test file**: `tests/r_validation_bayesian.rs`.
**Tolerance**: `5e-2` for active coefficients; pruning verified structurally (irrelevant features either have `λ > 10` or `|β| < 0.05`).

| Test | Description |
|------|-------------|
| Sparse recovery | `n = 120`, `p = 6`, two features pruned (true β = 0) |

## Test Coverage

| Category | Tests | Tolerance |
|----------|-------|-----------|
| OLS | 15+ | 1e-8 |
| WLS | 5+ | 0.01 |
| Ridge | 10+ | 0.01 |
| Elastic Net | 5+ | 0.2 |
| RLS | 10+ | 0.01-0.1 |
| BLS/NNLS | 5+ | 0.1 |
| Poisson GLM | 15+ | 0.01 |
| Binomial GLM | 10+ | 0.1 |
| Tweedie GLM | 10+ | 0.05-0.2 |
| Negative Binomial | 8+ | 0.05 |
| Diagnostics | 10+ | 1e-6 |
| ALM | 21+ | 0.15 |
| AID | 12+ | - |
| Quantile | 39+ | 0.01 |
| Isotonic | 34+ | 1e-6 |
| Stress Tests | 19 | Varies |
| Gamma | 2 | 1e-4 |
| Theil-Sen | 2 | 1e-10 / 5e-3 |
| RANSAC | 2 | 1e-9 |
| Passive-Aggressive | 3 | 2e-2 |
| LARS / LassoLars | 3 | 1e-6 / 5e-3 |
| Bayesian Ridge / ARD | 2 | 5e-3 / 5e-2 |
| **Total** | **470+** | - |

## Reproducibility

All validation is reproducible through:

1. **Fixed random seeds**: All R scripts use `set.seed(42)`; all Python scripts use `np.random.default_rng(42)` (or another small constant)
2. **Pinned versions**: R packages from CRAN are loaded at the version installed in CI; the Python venv pins `scikit-learn==1.5.2` and `numpy==2.1.3` via `validation/python/requirements.txt`
3. **Version-controlled data**: Reference output stored as Rust `const` arrays in `tests/fixtures/*_validation.rs`
4. **CI/CD verification**: Tests run automatically on every commit
5. **Transparent documentation**: R/Python code embedded in Rust test comments

## Running Validation

### Regenerate R References

```bash
cd validation
Rscript generate_validation_data.R > validation_output.txt
```

For per-estimator R fixtures (Huber, Logistic, Gamma, …):

```bash
Rscript tests/r_scripts/generate_gamma_validation.R    > tests/fixtures/gamma_validation.rs
Rscript tests/r_scripts/generate_huber_validation.R    > tests/fixtures/huber_validation.rs
Rscript tests/r_scripts/generate_logistic_validation.R > tests/fixtures/logistic_validation.rs
```

### Regenerate Python (sklearn) references

One-time environment bootstrap (requires [`uv`](https://github.com/astral-sh/uv)):

```bash
cd validation/python
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Then regenerate any of the sklearn-based fixtures:

```bash
.venv/bin/python generate_theil_sen_validation.py > ../../tests/fixtures/theil_sen_validation.rs
.venv/bin/python generate_ransac_validation.py    > ../../tests/fixtures/ransac_validation.rs
.venv/bin/python generate_pa_validation.py        > ../../tests/fixtures/pa_validation.rs
.venv/bin/python generate_lars_validation.py      > ../../tests/fixtures/lars_validation.rs
.venv/bin/python generate_bayesian_validation.py  > ../../tests/fixtures/bayesian_validation.rs
```

### Run All Rust Tests

```bash
cargo test
```

### Run Specific Validation Tests

```bash
# OLS validation
cargo test test_ols_simple_vs_r

# GLM validation
cargo test r_validation

# Diagnostic validation
cargo test test_leverage
cargo test test_cooks_distance

# New-in-0.5.5 sklearn-backed validations
cargo test --test r_validation_theil_sen
cargo test --test r_validation_ransac
cargo test --test r_validation_passive_aggressive
cargo test --test r_validation_lars
cargo test --test r_validation_bayesian
```

## Implementation Notes

### Tolerance Choices Explained

The tolerance levels used in validation tests vary significantly across different regression methods. This section explains why certain methods require looser tolerances than others.

#### OLS (Tolerance: 1e-8)

OLS uses a **closed-form solution** via QR decomposition: β = (X'X)⁻¹X'y. Both R and this library use numerically stable QR factorization, producing nearly identical results down to floating-point precision. The only differences arise from minor variations in LAPACK implementations.

#### WLS (Tolerance: 0.01)

While WLS also has a closed-form solution (weighted normal equations), the tolerance is slightly looser because:
- Extreme weight ratios (e.g., 1/x² weights spanning 1.0 to 0.001) can amplify small numerical differences
- R and Rust may handle near-zero weights differently in edge cases
- Standard error calculations involve the weighted residual variance, which accumulates small rounding errors

#### Ridge Regression (Tolerance: 0.01)

Ridge has a closed-form solution: β = (X'X + λI)⁻¹X'y. The moderate tolerance accounts for:
- Different implementations of the regularization term (some add λ to diagonal before inversion, others use SVD)
- **Lambda scaling conventions**: R's `glmnet` uses λ/n scaling by default, while this library uses raw λ. Tests must account for this difference
- Intercept handling: Whether the intercept is penalized affects final coefficients

#### Elastic Net (Tolerance: 0.2)

Elastic Net requires the **largest tolerance** because it uses **coordinate descent optimization**, an iterative algorithm with no closed-form solution:

1. **Algorithm differences**: R's `glmnet` uses a highly optimized coordinate descent with warm starts and active set strategies. This library implements a standard coordinate descent that may converge to slightly different local optima
2. **Convergence criteria**: Different stopping rules (relative vs. absolute tolerance, coefficient change vs. objective function change) lead to different final solutions
3. **Soft-thresholding**: The L1 penalty creates non-smooth optimization landscapes where multiple solutions may be equally valid within numerical precision
4. **Cycling order**: The order in which coordinates are updated can affect the final solution
5. **Initialization**: Different starting points can lead to different convergence paths

Despite these differences, both implementations produce statistically equivalent models with similar predictive performance.

#### Poisson GLM (Tolerance: 0.01 coefficients, 0.1 deviance)

Poisson GLM uses **Iteratively Reweighted Least Squares (IRLS)**, which introduces several sources of numerical variation:

1. **Convergence criteria**: R's `glm()` and this library use different stopping rules
2. **Starting values**: Initial coefficient estimates affect convergence path
3. **Step halving**: Different line search strategies when IRLS overshoots
4. **Link function derivatives**: Small differences in computing the working weights

The deviance tolerance is larger because it accumulates differences across all observations.

#### Binomial GLM (Tolerance: 0.1 coefficients, 0.5 deviance)

Binomial GLM has **larger tolerances** than Poisson because:

1. **Boundary issues**: Probabilities near 0 or 1 require careful numerical handling to avoid log(0)
2. **Separation**: Near-complete separation in data can cause coefficient inflation, handled differently across implementations
3. **Link-specific sensitivity**:
   - **Logit**: Most stable, tolerance ~0.1
   - **Probit**: Involves normal CDF, tolerance ~0.1
   - **Cloglog**: Most numerically sensitive (involves exp(exp(x))), tolerance ~0.5

#### Tweedie GLM (Tolerance: 0.05-0.2)

Tweedie models span multiple distributions with varying numerical stability:

| Variance Power | Distribution | Tolerance | Reason |
|----------------|--------------|-----------|--------|
| p = 1 | Poisson | 0.01 | Well-conditioned |
| p = 2 | Gamma | 0.05 | Log link, moderate sensitivity |
| p = 3 | Inverse Gaussian | 0.1-0.2 | Highly sensitive to outliers |
| 1 < p < 2 | Compound Poisson-Gamma | 0.2 | Complex density, deviance approximations differ |

The deviance calculation for Tweedie distributions involves special functions that may be implemented differently.

#### Negative Binomial GLM (Tolerance: 0.05 coefficients)

Negative binomial requires moderate tolerance due to:

1. **Theta estimation**: The dispersion parameter θ is estimated jointly with coefficients using alternating optimization. R's `glm.nb()` uses profile likelihood while this library uses moment estimation, leading to different θ values
2. **Theta sensitivity**: Small differences in θ propagate to coefficient estimates
3. **Variance function**: V(μ) = μ + μ²/θ means the working weights depend heavily on θ

#### RLS (Tolerance: 0.01-0.1)

Recursive Least Squares is an **online algorithm** that processes data sequentially:

1. **P matrix initialization**: RLS starts with an initial covariance matrix P₀ = δI. The choice of δ affects early estimates and creates differences from batch OLS
2. **Forgetting factor**: With λ < 1, RLS exponentially downweights older observations, intentionally diverging from OLS
3. **Numerical accumulation**: Sequential updates accumulate small rounding errors over many iterations
4. **Convergence behavior**: Even with λ = 1, RLS converges to OLS asymptotically but may not exactly match for finite samples

#### BLS/NNLS (Tolerance: 0.1)

Bounded Least Squares uses **active set methods** to handle inequality constraints:

1. **Active set identification**: Different algorithms may identify slightly different active constraint sets
2. **Degeneracy**: When multiple constraints are nearly active, small numerical differences can change which constraints are binding
3. **R's `nnls` package**: Uses the Lawson-Hanson algorithm; this library uses a similar but not identical implementation
4. **Constraint boundaries**: Solutions on constraint boundaries are sensitive to numerical precision

#### Diagnostics (Tolerance: 1e-6)

Diagnostic statistics use **direct matrix computations** without iteration:
- Leverage: H = X(X'X)⁻¹X' diagonal elements
- Cook's distance: Closed-form using leverage and residuals
- Studentized residuals: Direct formula using MSE and leverage

The tight tolerance reflects that these are deterministic calculations.

#### ALM (Tolerance: 0.15 coefficients, 0.20 log-likelihood)

ALM (Augmented Linear Model) uses a **hybrid optimization approach**:
- **IRLS** (Iteratively Reweighted Least Squares): For standard GLM-like distributions (Normal, Laplace, Poisson, etc.)
- **L-BFGS** (via argmin crate): For distributions with complex likelihood surfaces (FoldedNormal, RectifiedNormal, S, Beta, BoxCoxNormal)

1. **Distribution diversity**: 24 distributions with varying complexity (Normal to BoxCoxNormal)
2. **Optimizer choices**: R's `greybox::alm()` uses `nloptr` (BOBYQA); this library uses IRLS or L-BFGS depending on distribution
3. **Scale estimation**: Different methods for estimating scale parameters (MLE vs method of moments)
4. **Link function handling**: Some distributions use non-canonical links to match R greybox
5. **Extra parameters**: Distributions like GeneralisedNormal and BoxCoxNormal have shape/lambda parameters

**All 24 distributions validated**:
- Core (9): Normal, Laplace, StudentT, Logistic, LogNormal, Poisson, Gamma, Exponential, GeneralisedNormal
- Extended (15): Geometric, LogitNormal, LogLaplace, LogGeneralisedNormal, RectifiedNormal, FoldedNormal, S, Beta, BoxCoxNormal, CumulativeLogistic, CumulativeNormal, LogS, NegativeBinomial, Binomial, InverseGaussian, AsymmetricLaplace

**Key fixes implemented for R compatibility**:
- **Geometric**: Changed from Logit link to Log link, modeling mean λ = (1-p)/p instead of probability p
- **LogitNormal**: Changed from Logit link to Identity link, modeling logit-scale location parameter directly
- **LogLaplace**: Fixed scale estimation and IRLS weights to use log-space residuals
- **LogGeneralisedNormal**: Fixed scale estimation to use log-space residuals, corrected likelihood coefficient
- **RectifiedNormal**: Added L-BFGS optimization for direct likelihood maximization
- **FoldedNormal**: Fixed scale estimation using second-moment method (σ² = E[Y²] - μ²), added multi-start optimization
- **S distribution**: Added starting points with negative intercepts, achieves better LL than R
- **Beta**: Fixed precision parameter estimation using method-of-moments, uses scale for φ in likelihood
- **BoxCoxNormal**: Validated with L-BFGS optimization, tests verify finite LL and correct coefficient signs
- **CumulativeLogistic/CumulativeNormal**: Implemented Bernoulli log-likelihood for binary classification
- **AsymmetricLaplace**: Fixed scale estimation using weighted absolute residuals based on alpha (quantile) parameter
- **Binomial**: Tests use proportions (0-1) as expected by the likelihood function
- **NegativeBinomial**: Uses size parameter for dispersion modeling
- **InverseGaussian**: Validated with log link, produces reasonable coefficient estimates

**Architectural differences with R greybox**:
- **Optimization method**: R uses `nloptr` (BOBYQA algorithm); this library uses IRLS or L-BFGS with multi-start
- **Beta**: R models both α and β shape parameters; this library uses precision φ parameterization
- **FoldedNormal/S**: Our optimizer often finds better optima than R (higher log-likelihood)

The relatively large tolerance (15%) allows for optimizer implementation differences while ensuring statistical equivalence.

#### AID (No numeric tolerance - classification based)

AID (Automatic Identification of Demand) is primarily a **classification algorithm**:

1. **Demand type**: Binary classification (Regular vs Intermittent) based on zero proportion threshold
2. **Distribution selection**: Best distribution chosen by information criterion
3. **Anomaly detection**: Boolean flags for new product, obsolete, stockouts

Tests validate classification correctness rather than numeric precision. IC values are compared with 10% tolerance.

#### Quantile Regression (Tolerance: 0.01)

Quantile regression uses **Iteratively Reweighted Least Squares (IRLS)** with asymmetric weights:

1. **Check function**: The objective function ρ_τ(u) = u(τ - I(u < 0)) is non-differentiable at zero, requiring iterative approximation
2. **Algorithm differences**: R's `quantreg::rq()` uses the Barrodale-Roberts simplex algorithm by default, while this library uses IRLS with smooth approximation
3. **Epsilon smoothing**: Small epsilon (1e-4) added to avoid division by zero in weight calculation
4. **Convergence criteria**: Different stopping rules may lead to slightly different final solutions
5. **Quantile sensitivity**: Extreme quantiles (τ near 0 or 1) are more sensitive to algorithm differences

Despite algorithmic differences, both implementations produce statistically equivalent quantile estimates.

#### Isotonic Regression (Tolerance: 1e-6)

Isotonic regression uses the **Pool Adjacent Violators Algorithm (PAVA)**, a **deterministic algorithm** with no iteration variability:

1. **Closed-form solution**: PAVA is guaranteed to find the exact solution to the isotonic optimization problem
2. **Numerical stability**: The algorithm only involves averaging, which is numerically stable
3. **R equivalence**: Both R's `isoreg()` and this library implement identical PAVA algorithms
4. **Order handling**: Data is sorted by x before processing; identical sorting produces identical results

The tight tolerance (1e-6) reflects that PAVA is deterministic and numerically stable.

#### Stress Tests (Tolerance: Varies)

Stress tests verify numerical robustness and API behavior under extreme conditions:

1. **Scale invariance**: Tests IRLS behavior when data scale approaches or falls below the smoothing parameter (epsilon). Macro-scale (1e8) should scale proportionally; micro-scale (1e-8) may require smaller epsilon
2. **Sawtooth PAVA**: Worst-case alternating pattern [10, 0, 10, 0, ...] forces maximum block merging. Result must be monotonic
3. **Unsorted input**: API sorts X internally; results must match sorted input and preserve original order
4. **Singular matrices**: Rank-deficient designs should return Error or finite values, never panic
5. **f32 precision**: Crate uses f64 only; f32-range values must work correctly with appropriate tolerance (1e-2)

These tests verify robustness rather than exact numerical agreement with a reference implementation.

### Summary Table

| Method | Solution Type | Key Challenge | Tolerance |
|--------|---------------|---------------|-----------|
| OLS | Closed-form (QR) | Floating-point precision | 1e-8 |
| WLS | Closed-form (weighted QR) | Extreme weight ratios | 0.01 |
| Ridge | Closed-form (regularized) | Lambda scaling conventions | 0.01 |
| Elastic Net | Iterative (coordinate descent) | Non-convex, algorithm differences | 0.2 |
| RLS | Sequential (online) | P matrix initialization, accumulation | 0.01-0.1 |
| BLS/NNLS | Iterative (active set) | Constraint boundary sensitivity | 0.1 |
| Poisson GLM | Iterative (IRLS) | Convergence criteria | 0.01-0.1 |
| Binomial GLM | Iterative (IRLS) | Boundary handling, link sensitivity | 0.1-0.5 |
| Tweedie GLM | Iterative (IRLS) | Variance power sensitivity | 0.05-0.2 |
| Negative Binomial | Iterative (IRLS + theta) | Joint estimation | 0.05 |
| Diagnostics | Closed-form (matrix) | Direct computation | 1e-6 |
| ALM | Hybrid (IRLS or L-BFGS) | Distribution diversity, optimizer choice | 0.15 |
| AID | Classification | Zero proportion threshold | Classification |
| Quantile | Iterative (IRLS) | Check function non-differentiability | 0.01 |
| Isotonic | Deterministic (PAVA) | Exact monotonic solution | 1e-6 |
| Stress Tests | Mixed | Numerical robustness, API edge cases | Varies |

### Known Differences from R

1. **Log-likelihood formula**: R uses `RSS/n` in the log-likelihood calculation while this library uses `RSS/(n-p)` (MSE), causing small AIC/BIC differences
2. **Lambda scaling**: `glmnet` uses λ/n scaling by default; tests adjust accordingly
3. **Coordinate descent**: Elastic Net convergence may differ slightly from `glmnet`

## Reference

For the R code used to generate validation data, see:
- `validation/generate_validation_data.R`
- `tests/r_scripts/*.R`

## Validation Enhancements (January 2026)

Additional validation tests added in `tests/validation_enhancements.rs` address gaps identified during external review.

### Quantile Regression Enhancements

| Test Category | Tests | Description |
|---------------|-------|-------------|
| **Weighted observations** | 4 | Survey weights, heteroscedastic data, extreme weight ratios (1000:1) |
| **Quantile crossing** | 2 | Multiple τ values, documents that IRLS does not enforce monotonicity across quantiles |
| **High-dimensional** | 2 | p=50 with n=200; p=80 with n=100 (near-singular designs) |
| **Extrapolation** | 1 | Predictions outside training range, verifies linear extrapolation |
| **Sparse X regions** | 1 | Gaps in predictor space (clusters at extremes) |

### Isotonic Regression Enhancements

| Test Category | Tests | Description |
|---------------|-------|-------------|
| **Tie-breaking** | 2 | Documents weighted and unweighted averaging for duplicate X values |
| **Interpolation method** | 2 | Documents step function (not linear) interpolation between knots |
| **Extrapolation modes** | 1 | Tests Clip and NaN out-of-bounds behavior |
| **Sparse X regions** | 1 | Step function behavior in gaps |
| **Strict vs non-decreasing** | 1 | Documents that output allows ties (non-decreasing, not strictly increasing) |

### General Robustness Enhancements

| Test Category | Tests | Description |
|---------------|-------|-------------|
| **NaN/Inf handling** | 5 | NaN in X, NaN in y, Inf in y for both methods |
| **NaN propagation** | 1 | Single NaN in y vector behavior |
| **All-zero weights** | 3 | Edge case for weighted regression (both methods) |
| **Determinism** | 3 | Bitwise identical output across runs (IRLS and PAVA) |

### Key Documented Behaviors

1. **Quantile crossing**: IRLS algorithm does not enforce monotonicity across τ values. Fitted quantile lines may cross.

2. **Isotonic interpolation**: Uses step function, NOT linear interpolation. `predict(15)` between knots at x=10 and x=20 returns the value at x=10.

3. **Isotonic tie-breaking**: Duplicate X values with different Y values are averaged (weighted average if weights provided) before PAVA.

4. **All-zero weights**: Quantile regression produces degenerate solution (coef=0); Isotonic regression produces NaN. Neither crashes.

5. **Determinism**: Both methods are deterministic - same input produces bitwise identical output. PAVA is exact; IRLS converges to same point given same initialization.

### Total Test Count

| File | Tests | Purpose |
|------|-------|---------|
| `validation_enhancements.rs` | 29 | Validation gaps identified in external review |

This brings the total quantile + isotonic test count to **102+ tests** across all test files.
