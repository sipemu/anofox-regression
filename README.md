# anofox-regression

[![CI](https://github.com/sipemu/anofox-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-regression/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/anofox-regression.svg)](https://crates.io/crates/anofox-regression)
[![Documentation](https://docs.rs/anofox-regression/badge.svg)](https://docs.rs/anofox-regression)
[![codecov](https://codecov.io/gh/sipemu/anofox-regression/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/anofox-regression)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

> **Need to run this on 10GB of data?** Use our [DuckDB extension](https://github.com/DataZooDE/anofox-statistics)
>
> **Need to use this in a React Dashboard?** Use our [npm package](https://www.npmjs.com/package/@sipemu/anofox-regression)

A robust statistics library for regression analysis in Rust, validated against R and scikit-learn ([VALIDATION](validation/VALIDATION.md)).

This library provides sklearn-style regression estimators with full statistical inference support including standard errors, t-statistics, p-values, confidence intervals, and prediction intervals.

## Features

- **Linear Regression**
  - Ordinary Least Squares (OLS) with full inference
  - Weighted Least Squares (WLS)
  - Ridge Regression (L2 regularization)
  - Elastic Net (L1 + L2 regularization, coordinate descent as `glmnet`)
  - Huber Regression (robust to outliers, IRLS with Huber loss)
  - Recursive Least Squares (RLS) with online learning
  - Bounded Least Squares (BLS/NNLS) with box constraints
  - Dynamic Linear Model (LmDynamic) with time-varying coefficients
  - Least Angle Regression (LARS) and LassoLars with full coefficient path
  - Ridge / elastic-net regularisation paths and `cv.glmnet`-style CV (`solvers::path`),
    with group, leave-one-group-out, stratified, stratified-group and time-series
    (rolling-origin) folds, CV on explicit train / test splits and nested CV (`solvers::cv`)
  - **Regression splines** (`splines`): restricted cubic splines with Harrell's
    default knots (`rms::rcs`), linear splines (`rms::lsp`), restricted
    interactions (`%ia%`) and the algebraic form of a fitted spline
    (`rcspline.restate`), as design-matrix columns for any model
  - **Multi-d.f. Wald tests and the `rms` ANOVA table** (`inference::anova`,
    `wald_test`): chunk tests of predictors, nonlinearity and interactions from
    one fit with any (model, robust or bootstrap) covariance, on a `design::Design`
    that records its terms
  - **Predictive accuracy indexes** (`diagnostics::accuracy`): C / Dxy / γ / τ-a,
    Harrell's C for censored data, Brier, Nagelkerke and adjusted R², g-indexes,
    the `lrm` stats vector and `val.prob` calibration (intercept, slope, Emax, …)
  - **Cluster-robust covariance** (`inference::cluster`): one- and multi-way
    clustered sandwich (CR0 / CR1 / CR2 / CR3 as `sandwich::vcovCL`, `rms::robcov`)
    for OLS / WLS, any GLM family and AFT
  - **Nonparametric survival** (`survival`): Kaplan–Meier with Greenwood CIs,
    Nelson–Aalen / Fleming–Harrington hazards, median and restricted mean,
    strata, weights, left truncation, and log-rank / G-ρ tests (`survfit`, `survdiff`)
  - **Cox proportional hazards** (`survival::cox`): Efron / Breslow ties, weights,
    offsets, strata, time-dependent covariates, ridge, robust variance and predicted
    survival curves (`coxph`, `survfit.coxph`, `cph`)
  - **Model validation** (`validation`): optimism bootstrap, cross-validation and
    .632 validation of any modelling process, and overfitting-corrected calibration
    curves (`rms::validate`, `rms::calibrate`)
  - **Incremental and distributed fits** via `SufficientStats` — a mergeable,
    serializable (versioned bytes) state of weighted means and centred
    co-moments. `OlsRegressor`, `WlsRegressor` and `RidgeRegressor::fit_from_state`
    return the same coefficients, aliasing, standard errors, R², F,
    log-likelihood and intervals as `fit()` on the concatenated rows; ridge
    applies λ at finalize, so one state serves a λ grid. Stays accurate for
    data with large offsets. The older `MomentAccumulator` (raw moments,
    coefficients only) is kept (see `cargo run --example streaming_ridge`)

- **Robust & Bayesian Regression**
  - Theil-Sen (Vardi-Zhang spatial-median, ~29.3% breakdown point)
  - RANSAC (consensus-set refit with Fischler-Bolles stop-probability)
  - Bayesian Ridge (SVD evidence maximisation)
  - ARD Regression (per-feature precisions with relevance pruning)

- **Generalized Linear Models**
  - Logistic Regression (binary classification with sklearn-like API)
  - Poisson GLM (Log, Identity, Sqrt links) with offset support
  - Negative Binomial GLM (overdispersed count data with theta estimation)
  - Binomial GLM (Logistic, Probit, Complementary log-log)
  - Tweedie GLM (Gaussian, Poisson, Gamma, Inverse-Gaussian, Compound Poisson-Gamma)
  - Gamma Regression (sklearn-style wrapper for Tweedie(power=2), log link)

- **Generalized Linear Mixed Models**
  - GLMM with a random intercept, optional random slopes, and crossed / nested factors (`GlmmRegressor`): Gaussian (profiled REML/ML, matches `lme4::lmer`), Poisson and Binomial (Laplace PIRLS, matches `glmer(nAGQ=0)`). Unstructured random-effects covariance profiled by golden-section (intercept) or Nelder–Mead (slopes / multiple factors); `fit_crossed` handles `(1|a) + (1|b)` and `(1|a/b)`.

- **Correlated, ordinal and survival outcomes**
  - Generalized least squares (`GlsRegressor`, `nlme::gls`): AR(1), continuous AR(1),
    compound symmetry, spatial structures with nugget, `varIdent` / `varPower` / `varExp`
    variance functions, REML / ML
  - Ordinal regression (`OrdinalRegressor`, `rms::orm` / `MASS::polr`) with logit, probit,
    cloglog and loglog links, and the continuation-ratio model (`ContinuationRatioRegressor`,
    `rms::cr.setup`)
  - Accelerated failure time models (`AftRegressor`, `survival::survreg`): Weibull,
    exponential, log-normal, log-logistic, with censoring

- **Penalized and smooth models**
  - Penalized GLMs with Normal / Laplace priors and Laplace-approximation inference
    (`PenalizedGlmRegressor`), penalty selection by effective AIC (`pentrace`, `rms::pentrace`)
  - P-splines (`PSplineRegressor`, `mgcv` `bs = "ps"`) and partial least squares (`PlsRegressor`, `pls::plsr`)

- **Regression modeling strategies** (Harrell, *RMS*)
  - Effects, contrasts and prediction grids on the covariate scale, simultaneous bands and
    nomograms (`effects`: `summary.rms`, `contrast`, `Predict`, `nomogram`)
  - Multiple imputation (`imputation::areg_impute`, `Hmisc::aregImpute`) and Rubin's rules
    with D1 Wald tests (`inference::pool_rubin`, `mice::pool`)
  - Data reduction before modelling (`reduction`): `varclus`, `redun`, `spearman2`,
    Hoeffding's D, principal components, `transcan`
  - Response transformations (`transform`): Box–Cox profiles, `areg` optimal transformations,
    smearing estimates
  - Likelihood-ratio and score tests, profile-likelihood intervals (`inference::lr_test`,
    `glm_score_test`, `profile_confint`), fast backward elimination (`fastbw`), model
    approximation, bootstrap covariance (`bootcov`), causal mediation (`mediate`) and
    empirical-Bayes shrinkage (`eb_shrink`)

- **Online Learning**
  - Passive-Aggressive Regressor (PA-I and PA-II) with `partial_fit`

- **Augmented Linear Model (ALM)**
  - 24 distributions: Normal, Laplace, Student-t, Gamma, Beta, Log-Normal, and more
  - Based on the [greybox R package](https://github.com/config-i1/greybox)

- **Quantile & Monotonic Regression**
  - Quantile Regression (any τ ∈ (0,1); IRLS start polished to the exact LP optimum as `quantreg::rq`, with iid / nid / bootstrap inference)
  - Isotonic Regression (PAVA algorithm for monotonic constraints)

- **Smoothing & Classification**
  - LOWESS (Locally Weighted Scatterplot Smoothing)
  - AID (Automatic Identification of Demand) classifier with optimized closed-form fitting

- **Loss Functions**
  - MAE, MSE, RMSE, MAPE, sMAPE, MASE, pinball loss

- **Model Diagnostics**
  - R², Adjusted R², RMSE, F-statistic, AIC, AICc, BIC
  - Leverage, Cook's distance, VIF, studentized residuals

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
anofox-regression = "0.7"
```

The minimum supported Rust version is 1.87. The optional `parallel` feature runs
bootstrap and validation resamples on rayon.

## Examples

The library includes runnable examples demonstrating each major feature:

```bash
# Linear
cargo run --example ols                 # Ordinary Least Squares
cargo run --example wls                 # Weighted Least Squares
cargo run --example ridge               # Ridge regression
cargo run --example elastic_net         # Elastic Net
cargo run --example rls                 # Recursive Least Squares
cargo run --example bls                 # Bounded/Non-negative LS
cargo run --example pls                 # Partial Least Squares
cargo run --example streaming_ridge     # Streaming fit via MomentAccumulator
cargo run --example lars                # LARS and LassoLars
cargo run --example lm_dynamic          # Dynamic Linear Model
cargo run --example collinearity_intervals  # Intervals with aliased columns
cargo run --release --example many_small_fits  # 100K small OLS fits (allocation benchmark)

# Robust & Bayesian
cargo run --example huber               # Huber regression (robust)
cargo run --example theil_sen           # Theil–Sen estimator
cargo run --example ransac              # RANSAC regression
cargo run --example bayesian            # Bayesian Ridge + ARD

# Online
cargo run --example passive_aggressive  # PA-I / PA-II + partial_fit

# GLMs
cargo run --example poisson             # Poisson GLM
cargo run --example negative_binomial   # Negative Binomial GLM
cargo run --example binomial            # Binomial GLM (logit/probit/cloglog)
cargo run --example logistic            # LogisticRegression (sklearn-style)
cargo run --example tweedie             # Tweedie GLM
cargo run --example gamma               # Gamma GLM (log link)

# Other
cargo run --example alm                 # Augmented Linear Model
cargo run --example lowess              # LOWESS smoothing
cargo run --example aid                 # Demand classification
cargo run --example quantile            # Quantile regression
cargo run --example isotonic            # Isotonic regression
```

## Quick Start

Every block below is a complete program (they run as doctests).

### OLS Regression with Inference and Prediction Intervals

```rust
use anofox_regression::prelude::*;
use faer::{Col, Mat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(50, 2, |i, j| if j == 0 { i as f64 } else { (i as f64).sin() });
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * x[(i, 0)] - 0.5 * x[(i, 1)] + 0.1 * (i % 3) as f64);

    let fitted = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)?;

    let r = fitted.result();
    println!("R² = {:.4}, coefficients = {:?}", r.r_squared, r.coefficients);
    println!("standard errors = {:?}, p-values = {:?}", r.std_errors, r.p_values);

    let x_new = Mat::from_fn(2, 2, |i, j| if j == 0 { 50.0 + i as f64 } else { 0.0 });
    let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    println!("fit {:?}, lower {:?}, upper {:?}", pred.fit, pred.lower, pred.upper);
    Ok(())
}
```

### Poisson and Logistic GLMs

```rust
use anofox_regression::prelude::*;
use faer::{Col, Mat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(60, 1, |i, _| i as f64 / 20.0);
    let counts = Col::from_fn(60, |i| ((0.5 + 0.8 * x[(i, 0)]).exp() + (i % 3) as f64).floor());
    let poisson = PoissonRegressor::log().build().fit(&x, &counts)?;
    println!("deviance = {}", poisson.deviance);
    println!("expected counts = {:?}", poisson.predict_count(&x));

    let events = Col::from_fn(60, |i| if (i * 7) % 10 < 6 * i / 60 + 2 { 1.0 } else { 0.0 });
    let logistic = BinomialRegressor::logistic().build().fit(&x, &events)?;
    println!("probabilities = {:?}", logistic.predict_probability(&x));
    Ok(())
}
```

### Robust and Quantile Regression

```rust
use anofox_regression::prelude::*;
use faer::{Col, Mat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Mat::from_fn(40, 1, |i, _| i as f64);
    // a linear trend with two gross outliers
    let y = Col::from_fn(40, |i| {
        2.0 + 0.5 * i as f64 + if i == 5 || i == 30 { 25.0 } else { (i % 4) as f64 * 0.2 }
    });

    // Laplace (least absolute deviation) ALM
    let laplace = AlmRegressor::builder()
        .distribution(AlmDistribution::Laplace)
        .build()
        .fit(&x, &y)?;
    println!("ALM log-likelihood = {}", laplace.result().log_likelihood);

    // median and 90th-percentile regression
    for tau in [0.5, 0.9] {
        let q = QuantileRegressor::builder().tau(tau).build().fit(&x, &y)?;
        println!("τ = {tau}: slope = {:.3}", q.result().coefficients[0]);
    }
    Ok(())
}
```

### Isotonic Regression

```rust
use anofox_regression::prelude::*;
use faer::Col;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Col::from_fn(20, |i| i as f64);
    let y = Col::from_fn(20, |i| (i as f64).sqrt() + if i % 5 == 0 { -0.8 } else { 0.0 });
    let fitted = IsotonicRegressor::builder().increasing(true).build().fit_1d(&x, &y)?;
    println!("R² = {:.4}", fitted.result().r_squared);
    println!("monotone fit at new points = {:?}", fitted.predict_1d(&Col::from_fn(3, |i| 2.5 + 5.0 * i as f64)));
    Ok(())
}
```

## Validation

This library is developed using Test-Driven Development (TDD) against established statistical oracles. Most estimators are validated against R; the sklearn-style estimators are validated against scikit-learn 1.9.1 via a pinned reproducible Python environment under [`validation/python/`](validation/python/).

### R-validated estimators

| Rust | R Equivalent | Package |
|------|--------------|---------|
| `OlsRegressor` | `lm()` | stats |
| `WlsRegressor` | `lm()` with weights | stats |
| `RidgeRegressor`, `ElasticNetRegressor` | `glmnet()` | glmnet |
| `HuberRegressor` | `rlm(psi=psi.huber)` | MASS |
| `LogisticRegression` | `glm(..., family=binomial)` | stats |
| `BlsRegressor` | `nnls()` | nnls |
| `PoissonRegressor` | `glm(..., family=poisson)` | stats |
| `BinomialRegressor` | `glm(..., family=binomial)` | stats |
| `NegativeBinomialRegressor` | `glm.nb()` | MASS |
| `TweedieRegressor` | `tweedie()` | statmod |
| `GammaRegressor` | `glm(family=Gamma(link="log"))` | stats |
| `GlmmRegressor` | `lmer()` / `glmer(nAGQ=0)` | lme4 |
| `AlmRegressor` | `alm()` | greybox |
| `QuantileRegressor` | `rq()` | quantreg |
| `IsotonicRegressor` | `isoreg()` | stats |
| `GlsRegressor` | `gls()` | nlme |
| `OrdinalRegressor`, `ContinuationRatioRegressor` | `orm()`, `polr()`, `lrm(cr.setup)` | rms, MASS |
| `AftRegressor` | `survreg()` | survival |
| `survival::cox` | `coxph()`, `survfit()` | survival |
| `PenalizedGlmRegressor`, `pentrace` | `glmnet()` with penalty factors, `pentrace()` | glmnet, rms |
| `PSplineRegressor` | `gam(s(x, bs = "ps"))` | mgcv |
| `imputation`, `inference::pool_rubin` | `aregImpute()`, `pool()` | Hmisc, mice |
| `effects`, `validation`, `reduction` | `summary.rms`, `validate`, `calibrate`, `varclus`, `redun` | rms, Hmisc |
| Diagnostics | `cooks.distance()`, `hatvalues()`, `vif()` | stats, car |

### scikit-learn-validated estimators

| Rust | scikit-learn equivalent |
|------|-------------------------|
| `TheilSenRegressor` | `linear_model.TheilSenRegressor` |
| `RansacRegressor` | `linear_model.RANSACRegressor` |
| `PassiveAggressiveRegressor` | `linear_model.PassiveAggressiveRegressor` |
| `LarsRegressor`, LassoLars variant | `linear_model.Lars`, `linear_model.LassoLars` |
| `BayesianRidge` | `linear_model.BayesianRidge` |
| `ArdRegression` | `linear_model.ARDRegression` |

745 reference comparisons check numerical agreement with the chosen oracle within documented tolerances; [`validation/VALIDATION.md`](validation/VALIDATION.md) maps every reference script to its tests.

**For complete transparency on the validation process, see [`validation/VALIDATION.md`](validation/VALIDATION.md)**, which documents the per-estimator tolerance rationale and reproduction instructions for both the R and Python pipelines.

## Dependencies

- [faer](https://crates.io/crates/faer) - High-performance linear algebra
- [statrs](https://crates.io/crates/statrs) - Statistical distributions
- [basin](https://crates.io/crates/basin) - Numerical optimization (L-BFGS, Nelder–Mead)
- [rayon](https://crates.io/crates/rayon) - Parallel resampling (optional `parallel` feature)

## Attribution

This library includes Rust implementations of algorithms from several open-source projects. See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) for complete attribution and license information.

Key attributions:
- **greybox** - ALM distributions and AID classifier methodology (independent implementation)
- **basin** (MIT/Apache-2.0) - L-BFGS optimization
- **faer** (MIT) - Linear algebra operations
- **statrs** (MIT) - Statistical distributions

## License

MIT License
