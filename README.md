# regress-rs

[![CI](https://github.com/sipemu/statistics/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/statistics/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/regress-rs.svg)](https://crates.io/crates/regress-rs)
[![Documentation](https://docs.rs/regress-rs/badge.svg)](https://docs.rs/regress-rs)
[![codecov](https://codecov.io/gh/sipemu/statistics/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/regress-rs)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A robust statistics library for regression analysis in Rust.

This library provides sklearn-style regression estimators with full statistical inference support including standard errors, t-statistics, p-values, confidence intervals, and prediction intervals.

## Features

- **Multiple Regression Methods**
  - Ordinary Least Squares (OLS)
  - Weighted Least Squares (WLS)
  - Ridge Regression (L2 regularization)
  - Elastic Net (L1 + L2 regularization)
  - Recursive Least Squares (RLS) with online learning

- **Statistical Inference**
  - Coefficient standard errors, t-statistics, and p-values
  - Confidence intervals for coefficients
  - Prediction intervals (R-style `predict(..., interval="prediction")`)
  - Confidence intervals for mean response

- **Model Diagnostics**
  - R², Adjusted R², RMSE
  - F-statistic and p-value
  - AIC, AICc, BIC, Log-likelihood
  - Residual analysis (standardized, studentized)
  - Leverage and influence measures (Cook's distance, DFFITS)
  - Variance Inflation Factor (VIF) for multicollinearity detection

- **Robust Handling**
  - Automatic detection of collinear/constant columns
  - Rank-deficient matrix handling
  - Edge cases (extreme weights, near-singular matrices)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
regress-rs = "0.1"
```

## Quick Start

### Basic OLS Regression

```rust
use statistics::prelude::*;
use faer::{Mat, Col};

// Create sample data
let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64 * 0.1);
let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64 * 0.1);

// Fit OLS model
let model = OlsRegressor::builder()
    .with_intercept(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

// Access results
println!("R² = {:.4}", fitted.r_squared());
println!("Coefficients: {:?}", fitted.coefficients());
println!("Intercept: {:?}", fitted.intercept());

// Make predictions
let x_new = Mat::from_fn(5, 2, |i, j| (i + j + 100) as f64 * 0.1);
let predictions = fitted.predict(&x_new);
```

### Prediction Intervals

```rust
use statistics::prelude::*;

let fitted = OlsRegressor::builder()
    .with_intercept(true)
    .build()
    .fit(&x, &y)
    .unwrap();

// 95% prediction intervals for new observations
let result = fitted.predict_with_interval(
    &x_new,
    Some(IntervalType::Prediction),
    0.95,
);
println!("Fit: {:?}", result.fit);
println!("Lower: {:?}", result.lower);
println!("Upper: {:?}", result.upper);

// 95% confidence intervals for mean response
let result = fitted.predict_with_interval(
    &x_new,
    Some(IntervalType::Confidence),
    0.95,
);
```

### Weighted Least Squares

```rust
use statistics::prelude::*;

let weights = Col::from_fn(100, |i| 1.0 / (i + 1) as f64);

let model = WlsRegressor::builder()
    .with_intercept(true)
    .weights(weights)
    .build();
let fitted = model.fit(&x, &y).unwrap();
```

### Ridge Regression

```rust
use statistics::prelude::*;

let model = RidgeRegressor::builder()
    .with_intercept(true)
    .lambda(0.1)
    .compute_inference(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

// Access inference statistics
let result = fitted.result();
if let Some(se) = &result.std_errors {
    println!("Standard errors: {:?}", se);
}
```

### Elastic Net

```rust
use statistics::prelude::*;

let model = ElasticNetRegressor::builder()
    .with_intercept(true)
    .lambda(1.0)
    .alpha(0.5)  // 0 = Ridge, 1 = Lasso
    .max_iterations(1000)
    .tolerance(1e-6)
    .build();
let fitted = model.fit(&x, &y).unwrap();

println!("Non-zero coefficients: {}", fitted.n_nonzero());
```

### Recursive Least Squares (Online Learning)

```rust
use statistics::prelude::*;

let model = RlsRegressor::builder()
    .with_intercept(true)
    .forgetting_factor(0.99)  // Recent data weighted more
    .build();
let mut fitted = model.fit(&x, &y).unwrap();

// Online update with new observation
let x_new = Col::from_fn(2, |j| j as f64);
let y_new = 5.0;
let prediction = fitted.update(&x_new, y_new);
```

### Model Diagnostics

```rust
use statistics::prelude::*;

let fitted = OlsRegressor::builder()
    .with_intercept(true)
    .build()
    .fit(&x, &y)
    .unwrap();

let result = fitted.result();

// Goodness of fit
println!("R² = {:.4}", result.r_squared);
println!("Adjusted R² = {:.4}", result.adj_r_squared);
println!("RMSE = {:.4}", result.rmse);

// F-test
println!("F-statistic = {:.4}", result.f_statistic);
println!("F p-value = {:.4}", result.f_pvalue);

// Information criteria
println!("AIC = {:.4}", result.aic);
println!("BIC = {:.4}", result.bic);

// Residual diagnostics
let std_resid = standardized_residuals(&result.residuals, result.mse);
let leverage = compute_leverage(&x);
let cooks_d = cooks_distance(&result.residuals, &leverage, result.mse, result.n_parameters);

// Detect influential points
let influential = influential_cooks(&cooks_d, result.n_observations);

// Variance Inflation Factor for multicollinearity
let vif = variance_inflation_factor(&x);
```

## API Reference

### Regression Result Fields

| Field | Description |
|-------|-------------|
| `coefficients` | Estimated regression coefficients |
| `intercept` | Intercept term (if fitted) |
| `std_errors` | Standard errors of coefficients |
| `t_statistics` | t-statistics for coefficients |
| `p_values` | Two-tailed p-values |
| `conf_interval_lower/upper` | Confidence intervals for coefficients |
| `r_squared` | Coefficient of determination |
| `adj_r_squared` | Adjusted R² |
| `mse` | Mean squared error |
| `rmse` | Root mean squared error |
| `f_statistic` | F-statistic for overall model |
| `f_pvalue` | p-value for F-test |
| `aic` | Akaike Information Criterion |
| `aicc` | Corrected AIC |
| `bic` | Bayesian Information Criterion |
| `log_likelihood` | Log-likelihood |
| `residuals` | Model residuals |
| `fitted_values` | Predicted values on training data |

### Interval Types

- `IntervalType::Prediction` - Prediction interval for new observations (wider)
- `IntervalType::Confidence` - Confidence interval for mean response (narrower)

### Lambda Scaling

For Ridge and Elastic Net, use `LambdaScaling::Glmnet` to match R's glmnet package:

```rust
let model = RidgeRegressor::builder()
    .lambda(0.1)
    .lambda_scaling(LambdaScaling::Glmnet)  // lambda * n
    .build();
```

## Validation

This library is validated against R's statistical functions:

- `lm()` for OLS
- `lm()` with weights for WLS
- `glmnet::glmnet()` for Ridge and Elastic Net
- `predict(..., interval="prediction")` for prediction intervals
- `cooks.distance()`, `hatvalues()`, `rstandard()` for diagnostics
- `car::vif()` for variance inflation factors

All tests ensure numerical agreement with R within appropriate tolerances.

## Dependencies

- [faer](https://crates.io/crates/faer) - High-performance linear algebra
- [statrs](https://crates.io/crates/statrs) - Statistical distributions

## License

MIT License
