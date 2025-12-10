# anofox-regression

[![CI](https://github.com/sipemu/anofox-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-regression/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/anofox-regression.svg)](https://crates.io/crates/anofox-regression)
[![Documentation](https://docs.rs/anofox-regression/badge.svg)](https://docs.rs/anofox-regression)
[![codecov](https://codecov.io/gh/sipemu/anofox-regression/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/anofox-regression)
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
  - Bounded Least Squares (BLS/NNLS) with box constraints
  - Tweedie GLM (Gaussian, Poisson, Gamma, Inverse-Gaussian, Compound Poisson-Gamma)
  - Poisson GLM (Log, Identity, Sqrt links) with offset support
  - Negative Binomial GLM (overdispersed count data with theta estimation)
  - Binomial GLM (Logistic, Probit, Complementary log-log)
  - Augmented Linear Model (ALM) with 24+ distributions (Normal, Laplace, Student-t, Gamma, Beta, etc.)

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
  - GLM residuals (Pearson, deviance, working)
  - Leverage and influence measures (Cook's distance, DFFITS)
  - Variance Inflation Factor (VIF) for multicollinearity detection

- **Robust Handling**
  - Automatic detection of collinear/constant columns
  - Rank-deficient matrix handling
  - Edge cases (extreme weights, near-singular matrices)
  - R-compatible NA handling (na.omit, na.exclude, na.fail, na.pass)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
anofox-regression = "0.3"
```

## Quick Start

### Basic OLS Regression

```rust
use anofox_regression::prelude::*;
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
use anofox_regression::prelude::*;

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
use anofox_regression::prelude::*;

let weights = Col::from_fn(100, |i| 1.0 / (i + 1) as f64);

let model = WlsRegressor::builder()
    .with_intercept(true)
    .weights(weights)
    .build();
let fitted = model.fit(&x, &y).unwrap();
```

### Ridge Regression

```rust
use anofox_regression::prelude::*;

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
use anofox_regression::prelude::*;

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
use anofox_regression::prelude::*;

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

### Bounded Least Squares (NNLS)

```rust
use anofox_regression::prelude::*;

// Non-negative least squares (all coefficients >= 0)
let model = BlsRegressor::nnls().build();
let fitted = model.fit(&x, &y).unwrap();

// Custom box constraints: lower <= coefficients <= upper
let model = BlsRegressor::builder()
    .lower_bounds(vec![0.0, -1.0, 0.0])
    .upper_bounds(vec![1.0, 1.0, f64::INFINITY])
    .build();
let fitted = model.fit(&x, &y).unwrap();
```

### Tweedie GLM

```rust
use anofox_regression::prelude::*;

// Gamma regression with log link (insurance claims, positive continuous data)
let model = TweedieRegressor::gamma()
    .with_intercept(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

println!("Deviance: {}", fitted.deviance);
println!("Null deviance: {}", fitted.null_deviance);

// Poisson regression (count data)
let model = TweedieRegressor::poisson()
    .with_intercept(true)
    .build();

// Compound Poisson-Gamma (zero-inflated continuous data)
let model = TweedieRegressor::builder()
    .var_power(1.5)  // Between 1 (Poisson) and 2 (Gamma)
    .link_power(0.0) // Log link
    .with_intercept(true)
    .build();
```

### Poisson GLM (Count Data)

```rust
use anofox_regression::prelude::*;

// Poisson regression with log link (count data)
let model = PoissonRegressor::log()
    .with_intercept(true)
    .compute_inference(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

println!("Deviance: {}", fitted.deviance);
println!("Dispersion: {}", fitted.dispersion);

// Predict counts
let counts = fitted.predict_count(&x_new);

// Poisson with identity link
let model = PoissonRegressor::identity()
    .with_intercept(true)
    .build();

// Poisson with sqrt link
let model = PoissonRegressor::sqrt()
    .with_intercept(true)
    .build();

// Rate modeling with offset (for exposure)
// y_i ~ Poisson(exposure_i * rate), log(E[y]) = log(exposure) + Xβ
let exposures = Col::from_fn(100, |i| (i + 1) as f64);
let offset = Col::from_fn(100, |i| exposures[i].ln());
let model = PoissonRegressor::log()
    .with_intercept(true)
    .offset(offset)
    .build();
let fitted = model.fit(&x, &y).unwrap();

// Predict with new offset (exposure = 2 for all new observations)
let x_new = Mat::from_fn(5, 2, |i, j| (i + j) as f64);
let new_offset = Col::from_fn(5, |_| 2.0_f64.ln());
let rates = fitted.predict_with_offset(&x_new, &new_offset);
```

### Negative Binomial GLM (Overdispersed Count Data)

```rust
use anofox_regression::prelude::*;

// Negative binomial with automatic theta estimation (like MASS::glm.nb)
let model = NegativeBinomialRegressor::builder()
    .with_intercept(true)
    .estimate_theta(true)  // Estimate dispersion parameter
    .build();
let fitted = model.fit(&x, &y).unwrap();

println!("Estimated theta: {}", fitted.theta);
println!("Overdispersion ratio: {}", fitted.overdispersion_ratio());
println!("Deviance: {}", fitted.deviance);

// Negative binomial with fixed theta
let model = NegativeBinomialRegressor::with_theta(2.0)
    .with_intercept(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

// Rate modeling with offset
let exposures = Col::from_fn(100, |i| (i + 1) as f64);
let offset = Col::from_fn(100, |i| exposures[i].ln());
let model = NegativeBinomialRegressor::builder()
    .with_intercept(true)
    .offset(offset)
    .build();
let fitted = model.fit(&x, &y).unwrap();

// GLM residuals
let pearson = fitted.pearson_residuals();
let deviance = fitted.deviance_residuals();
```

### Binomial GLM (Logistic Regression)

```rust
use anofox_regression::prelude::*;

// Logistic regression (binary classification)
let model = BinomialRegressor::logistic()
    .with_intercept(true)
    .compute_inference(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

// Predict probabilities
let probs = fitted.predict_probability(&x_new);

// Predict with standard errors and confidence intervals
let pred = fitted.predict_with_se(
    &x_new,
    PredictionType::Response,  // Probability scale
    Some(IntervalType::Confidence),
    0.95,
);
println!("Predicted probability: {:?}", pred.fit);
println!("Standard error: {:?}", pred.se);
println!("95% CI: [{:?}, {:?}]", pred.lower, pred.upper);

// Probit regression
let model = BinomialRegressor::probit()
    .with_intercept(true)
    .build();

// Complementary log-log regression
let model = BinomialRegressor::cloglog()
    .with_intercept(true)
    .build();

// GLM residuals
let pearson = fitted.pearson_residuals();
let deviance = fitted.deviance_residuals();
let working = fitted.working_residuals();
```

### Augmented Linear Model (ALM)

The ALM supports 24+ distributions for maximum likelihood regression, based on the [greybox R package](https://github.com/config-i1/greybox).

```rust
use anofox_regression::prelude::*;

// Laplace (LAD) regression - robust to outliers
let model = AlmRegressor::builder()
    .distribution(AlmDistribution::Laplace)
    .with_intercept(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

println!("Log-likelihood: {}", fitted.log_likelihood);
println!("AIC: {}", fitted.result().aic);

// Student-t regression - heavy-tailed errors
let model = AlmRegressor::builder()
    .distribution(AlmDistribution::StudentT)
    .with_intercept(true)
    .build();
let fitted = model.fit(&x, &y).unwrap();

// Log-Normal for positive skewed data
let model = AlmRegressor::builder()
    .distribution(AlmDistribution::LogNormal)
    .with_intercept(true)
    .build();

// Beta regression for proportions in (0, 1)
let model = AlmRegressor::builder()
    .distribution(AlmDistribution::Beta)
    .with_intercept(true)
    .build();

// Gamma for positive continuous data
let model = AlmRegressor::builder()
    .distribution(AlmDistribution::Gamma)
    .with_intercept(true)
    .build();

// Custom link function
let model = AlmRegressor::builder()
    .distribution(AlmDistribution::Normal)
    .link(LinkFunction::Log)
    .with_intercept(true)
    .build();
```

### NA Handling

```rust
use anofox_regression::{NaAction, NaHandler};

// Process data with NA values (represented as f64::NAN)
let result = NaHandler::process(&x, &y, NaAction::Omit).unwrap();
println!("Rows removed: {}", result.na_info.n_removed);

// na.exclude: Remove NA but expand output back to original length
let result = NaHandler::process(&x, &y, NaAction::Exclude).unwrap();
let residuals_expanded = result.na_info.expand(&fitted_residuals);

// na.fail: Error if any NA present
let result = NaHandler::process(&x, &y, NaAction::Fail);
assert!(result.is_err());

// na.pass: Keep NA values (solver must handle them)
let result = NaHandler::process(&x, &y, NaAction::Pass).unwrap();
```

### Model Diagnostics

```rust
use anofox_regression::prelude::*;

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

### Tweedie GLM Result Fields

| Field | Description |
|-------|-------------|
| `deviance` | Total deviance of fitted model |
| `null_deviance` | Deviance of intercept-only model |
| `dispersion` | Estimated dispersion parameter |
| `iterations` | Number of IRLS iterations |

### Tweedie Family (var_power)

| var_power | Distribution | Use Case |
|-----------|--------------|----------|
| 0 | Gaussian | Standard linear regression |
| 1 | Poisson | Count data |
| 1-2 | Compound Poisson-Gamma | Zero-inflated continuous (insurance, rainfall) |
| 2 | Gamma | Positive continuous |
| 3 | Inverse-Gaussian | Positive, right-skewed |

### Poisson GLM Result Fields

| Field | Description |
|-------|-------------|
| `deviance` | Total deviance of fitted model |
| `null_deviance` | Deviance of intercept-only model |
| `dispersion` | Estimated dispersion parameter |
| `iterations` | Number of IRLS iterations |

### Poisson Link Functions

| Link | Function | Inverse | Use Case |
|------|----------|---------|----------|
| Log | ln(μ) | exp(η) | Canonical, most common |
| Identity | μ | η | Linear relationships |
| Sqrt | √μ | η² | Alternative for count data |

### Negative Binomial GLM Result Fields

| Field | Description |
|-------|-------------|
| `deviance` | Total deviance of fitted model |
| `null_deviance` | Deviance of intercept-only model |
| `theta` | Estimated or fixed dispersion parameter |
| `dispersion` | Estimated dispersion parameter |
| `iterations` | Number of iterations |

### Negative Binomial Parameters

| Parameter | Description |
|-----------|-------------|
| `theta` | Dispersion parameter (size). Larger = less overdispersion. |
| `estimate_theta` | If true, estimate theta via alternating ML. Default: true. |

**When to use Negative Binomial vs Poisson:**
- Use **Poisson** when Var(Y) ≈ E[Y]
- Use **Negative Binomial** when Var(Y) > E[Y] (overdispersion)
- NB variance: V(μ) = μ + μ²/θ (approaches Poisson as θ → ∞)

### Binomial GLM Result Fields

| Field | Description |
|-------|-------------|
| `deviance` | Total deviance of fitted model |
| `null_deviance` | Deviance of intercept-only model |
| `iterations` | Number of IRLS iterations |

### Binomial Link Functions

| Link | Function | Inverse | Use Case |
|------|----------|---------|----------|
| Logit | log(p/(1-p)) | 1/(1+exp(-η)) | Standard logistic regression |
| Probit | Φ⁻¹(p) | Φ(η) | Dose-response, bioassay |
| Cloglog | log(-log(1-p)) | 1-exp(-exp(η)) | Asymmetric, extreme events |

### GLM Residual Types

| Type | Formula | Use Case |
|------|---------|----------|
| Pearson | (y - μ) / √V(μ) | Outlier detection, overdispersion |
| Deviance | sign(y - μ) × √d_i | Model fit assessment |
| Working | (y - μ) × (dη/dμ) | IRLS algorithm diagnostics |

### ALM (Augmented Linear Model) Result Fields

| Field | Description |
|-------|-------------|
| `log_likelihood` | Maximized log-likelihood |
| `scale` | Estimated scale parameter |
| `iterations` | Number of IRLS iterations |

### ALM Distributions

| Category | Distributions |
|----------|---------------|
| **Continuous** | Normal, Laplace, Student-t, Logistic, Asymmetric Laplace, Generalised Normal, S |
| **Log-transformed** | Log-Normal, Log-Laplace, Log-S, Log-Generalised Normal |
| **Positive** | Gamma, Inverse Gaussian, Exponential, Folded Normal |
| **Box-Cox** | Box-Cox Normal |
| **Proportions** | Beta, Logit-Normal |
| **Count data** | Poisson, Negative Binomial, Binomial, Geometric |
| **Ordinal** | Cumulative Logistic, Cumulative Normal |

### ALM Link Functions

| Link | Function | Inverse | Typical Use |
|------|----------|---------|-------------|
| Identity | μ = η | η | Normal, Laplace, Student-t |
| Log | μ = exp(η) | ln(μ) | Gamma, Poisson, Log-Normal |
| Logit | μ = 1/(1+e⁻ᶯ) | ln(μ/(1-μ)) | Beta, Binomial |
| Probit | μ = Φ(η) | Φ⁻¹(μ) | Ordinal models |
| Inverse | μ = 1/η | 1/μ | Inverse Gaussian |
| Sqrt | μ = η² | √μ | Count data (alternative) |
| Cloglog | μ = 1-e⁻ᵉˣᵖ⁽ᶯ⁾ | ln(-ln(1-μ)) | Asymmetric binary |

### Prediction Types (GLM)

| Type | Description |
|------|-------------|
| `PredictionType::Response` | Predictions on response scale (probabilities for binomial) |
| `PredictionType::Link` | Predictions on link scale (log-odds for logit) |

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
- `nnls::nnls()` for Non-negative Least Squares
- `statmod::tweedie()` for Tweedie GLM
- `glm(..., family=poisson)` for Poisson GLM (log, identity, sqrt links)
- `MASS::glm.nb()` for Negative Binomial GLM with theta estimation
- `glm(..., family=binomial)` for Binomial GLM (logit, probit, cloglog)
- `glm(..., family=Gamma)` for Gamma GLM
- `glm(..., family=inverse.gaussian)` for Inverse Gaussian GLM
- `glm(..., offset=...)` for rate modeling with offset terms
- `greybox::alm()` for Augmented Linear Model (Normal, Laplace, Student-t, Log-Normal, Gamma, etc.)
- `residuals(..., type="pearson/deviance/working")` for GLM residuals
- `predict(..., se.fit=TRUE)` for GLM predictions with standard errors
- `na.omit()`, `na.exclude()`, `na.fail()`, `na.pass()` for NA handling
- `predict(..., interval="prediction")` for prediction intervals
- `cooks.distance()`, `hatvalues()`, `rstandard()` for diagnostics
- `car::vif()` for variance inflation factors

All tests ensure numerical agreement with R within appropriate tolerances (708 tests total).

## Dependencies

- [faer](https://crates.io/crates/faer) - High-performance linear algebra
- [statrs](https://crates.io/crates/statrs) - Statistical distributions

## License

MIT License
