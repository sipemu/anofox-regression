# anofox-regression

[![CI](https://github.com/sipemu/anofox-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-regression/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/anofox-regression.svg)](https://crates.io/crates/anofox-regression)
[![Documentation](https://docs.rs/anofox-regression/badge.svg)](https://docs.rs/anofox-regression)
[![codecov](https://codecov.io/gh/sipemu/anofox-regression/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/anofox-regression)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A robust statistics library for regression analysis in Rust, validated against R ([VALIDATION](validation/VALIDATION.md)).

This library provides sklearn-style regression estimators with full statistical inference support including standard errors, t-statistics, p-values, confidence intervals, and prediction intervals.

## Features

- **Linear Regression**
  - Ordinary Least Squares (OLS) with full inference
  - Weighted Least Squares (WLS)
  - Ridge Regression (L2 regularization)
  - Elastic Net (L1 + L2 regularization via L-BFGS)
  - Recursive Least Squares (RLS) with online learning
  - Bounded Least Squares (BLS/NNLS) with box constraints
  - Dynamic Linear Model (LmDynamic) with time-varying coefficients

- **Generalized Linear Models**
  - Poisson GLM (Log, Identity, Sqrt links) with offset support
  - Negative Binomial GLM (overdispersed count data with theta estimation)
  - Binomial GLM (Logistic, Probit, Complementary log-log)
  - Tweedie GLM (Gaussian, Poisson, Gamma, Inverse-Gaussian, Compound Poisson-Gamma)

- **Augmented Linear Model (ALM)**
  - 24 distributions: Normal, Laplace, Student-t, Gamma, Beta, Log-Normal, and more
  - Based on the [greybox R package](https://github.com/config-i1/greybox)

- **Quantile & Monotonic Regression**
  - Quantile Regression (IRLS with asymmetric weights, any τ ∈ (0,1))
  - Isotonic Regression (PAVA algorithm for monotonic constraints)

- **Smoothing & Classification**
  - LOWESS (Locally Weighted Scatterplot Smoothing)
  - AID (Automatic Identification of Demand) classifier

- **Loss Functions**
  - MAE, MSE, RMSE, MAPE, sMAPE, MASE, pinball loss

- **Model Diagnostics**
  - R², Adjusted R², RMSE, F-statistic, AIC, AICc, BIC
  - Leverage, Cook's distance, VIF, studentized residuals

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
anofox-regression = "0.5"
```

## Examples

The library includes runnable examples demonstrating each major feature:

```bash
cargo run --example ols              # Ordinary Least Squares
cargo run --example wls              # Weighted Least Squares
cargo run --example ridge            # Ridge regression
cargo run --example elastic_net      # Elastic Net
cargo run --example rls              # Recursive Least Squares
cargo run --example bls              # Bounded/Non-negative LS
cargo run --example poisson          # Poisson GLM
cargo run --example negative_binomial # Negative Binomial GLM
cargo run --example binomial         # Logistic regression
cargo run --example tweedie          # Tweedie GLM
cargo run --example alm              # Augmented Linear Model
cargo run --example lm_dynamic       # Dynamic Linear Model
cargo run --example lowess           # LOWESS smoothing
cargo run --example aid              # Demand classification
cargo run --example quantile         # Quantile regression
cargo run --example isotonic         # Isotonic regression
```

## Quick Start

### OLS Regression

```rust
use anofox_regression::prelude::*;
use faer::{Mat, Col};

let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64 * 0.1);
let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64 * 0.1);

let fitted = OlsRegressor::builder()
    .with_intercept(true)
    .build()
    .fit(&x, &y)?;

println!("R² = {:.4}", fitted.r_squared());
println!("Coefficients: {:?}", fitted.coefficients());
```

### Prediction Intervals

```rust
let result = fitted.predict_with_interval(
    &x_new,
    Some(IntervalType::Prediction),
    0.95,
);
println!("Fit: {:?}", result.fit);
println!("Lower: {:?}", result.lower);
println!("Upper: {:?}", result.upper);
```

### Poisson GLM

```rust
let fitted = PoissonRegressor::log()
    .with_intercept(true)
    .build()
    .fit(&x, &y)?;

println!("Deviance: {}", fitted.deviance);
let counts = fitted.predict_count(&x_new);
```

### Logistic Regression

```rust
let fitted = BinomialRegressor::logistic()
    .with_intercept(true)
    .build()
    .fit(&x, &y)?;

let probs = fitted.predict_probability(&x_new);
```

### Augmented Linear Model

```rust
// Laplace regression (robust to outliers)
let fitted = AlmRegressor::builder()
    .distribution(AlmDistribution::Laplace)
    .with_intercept(true)
    .build()
    .fit(&x, &y)?;

println!("Log-likelihood: {}", fitted.log_likelihood);
```

### Quantile Regression

```rust
// Median regression (tau = 0.5)
let fitted = QuantileRegressor::builder()
    .tau(0.5)
    .build()
    .fit(&x, &y)?;

println!("Median coefficients: {:?}", fitted.coefficients());

// 90th percentile regression
let fitted_90 = QuantileRegressor::builder()
    .tau(0.9)
    .build()
    .fit(&x, &y)?;
```

### Isotonic Regression

```rust
// Fit monotonically increasing function
let fitted = IsotonicRegressor::builder()
    .increasing(true)
    .build()
    .fit_1d(&x, &y)?;

println!("R² = {:.4}", fitted.result().r_squared);
let predictions = fitted.predict_1d(&x_new);
```

## Validation

This library is developed using Test-Driven Development (TDD) with R as the oracle (ground truth). All implementations are validated against R's statistical functions:

| Rust | R Equivalent | Package |
|------|--------------|---------|
| `OlsRegressor` | `lm()` | stats |
| `WlsRegressor` | `lm()` with weights | stats |
| `RidgeRegressor`, `ElasticNetRegressor` | `glmnet()` | glmnet |
| `BlsRegressor` | `nnls()` | nnls |
| `PoissonRegressor` | `glm(..., family=poisson)` | stats |
| `BinomialRegressor` | `glm(..., family=binomial)` | stats |
| `NegativeBinomialRegressor` | `glm.nb()` | MASS |
| `TweedieRegressor` | `tweedie()` | statmod |
| `AlmRegressor` | `alm()` | greybox |
| `QuantileRegressor` | `rq()` | quantreg |
| `IsotonicRegressor` | `isoreg()` | stats |
| Diagnostics | `cooks.distance()`, `hatvalues()`, `vif()` | stats, car |

All 485+ test cases ensure numerical agreement with R within appropriate tolerances.

**For complete transparency on the validation process, see [`validation/VALIDATION.md`](validation/VALIDATION.md)**, which documents tolerance rationale for each method and reproduction instructions.

## Dependencies

- [faer](https://crates.io/crates/faer) - High-performance linear algebra
- [statrs](https://crates.io/crates/statrs) - Statistical distributions
- [argmin](https://crates.io/crates/argmin) - Numerical optimization (L-BFGS)

## Attribution

This library includes Rust implementations of algorithms from several open-source projects. See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) for complete attribution and license information.

Key attributions:
- **greybox** - ALM distributions and AID classifier methodology (independent implementation)
- **argmin** (MIT/Apache-2.0) - L-BFGS optimization
- **faer** (MIT) - Linear algebra operations
- **statrs** (MIT) - Statistical distributions

## License

MIT License
