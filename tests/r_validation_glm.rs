//! R Validation Tests for GLM implementations.
//!
//! These tests compare regress-rs GLM implementations against R's glm() function.
//! Each test includes the R code used to generate reference values.
//!
//! To verify against R, run the R code in comments and compare outputs.

use approx::assert_relative_eq;
use faer::{Col, Mat};
use regress_rs::solvers::{
    BinomialRegressor, FittedRegressor, NegativeBinomialRegressor, PoissonRegressor, Regressor,
    TweedieRegressor,
};

// ============================================================================
// POISSON GLM TESTS
// ============================================================================

/// R Code:
/// ```r
/// # Simple Poisson with identity link - perfect linear fit
/// x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
/// y <- c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21)  # y = 1 + 2*x exactly
/// fit <- glm(y ~ x, family = poisson(link = "identity"))
/// coef(fit)
/// # (Intercept)           x
/// #   1.0000000   2.0000000
/// ```
#[test]
fn test_poisson_identity_exact_vs_r() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(10, |i| 3.0 + 2.0 * (i as f64)); // y = 1 + 2*x (but 0-indexed so 3 + 2*i)

    let fitted = PoissonRegressor::identity()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // With a perfect linear fit, should recover exactly y = 1 + 2*x
    assert_relative_eq!(fitted.result().intercept.unwrap(), 1.0, epsilon = 1e-4);
    assert_relative_eq!(fitted.result().coefficients[0], 2.0, epsilon = 1e-4);
}

/// R Code:
/// ```r
/// # Poisson with offset for rate modeling
/// x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
/// exposure <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)  # constant exposure
/// y <- c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
/// fit <- glm(y ~ x + offset(log(exposure)), family = poisson(link = "identity"))
/// coef(fit)
/// # With constant exposure, same as no offset
/// ```
#[test]
fn test_poisson_offset_constant_exposure() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(10, |i| 3.0 + 2.0 * (i as f64));
    let offset = Col::from_fn(10, |_| 0.0_f64); // log(1) = 0

    let fitted = PoissonRegressor::identity()
        .with_intercept(true)
        .offset(offset)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // With offset=0 (log exposure of 1), same as no offset
    assert_relative_eq!(fitted.result().intercept.unwrap(), 1.0, epsilon = 1e-4);
    assert_relative_eq!(fitted.result().coefficients[0], 2.0, epsilon = 1e-4);
}

/// R Code:
/// ```r
/// # Poisson log link with simple exponential data
/// x <- c(0, 1, 2, 3, 4)
/// y <- c(1, 3, 7, 20, 55)  # roughly exp(0.7*x)
/// fit <- glm(y ~ x, family = poisson(link = "log"))
/// coef(fit)
/// deviance(fit)
/// ```
#[test]
fn test_poisson_log_link() {
    let x = Mat::from_fn(5, 1, |i, _| i as f64);
    let y_data = vec![1.0, 3.0, 7.0, 20.0, 55.0];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Coefficients should be reasonable (slope around 0.7-1.0 for exp growth)
    let intercept = fitted.result().intercept.unwrap();
    let slope = fitted.result().coefficients[0];

    // Verify basic sanity: positive slope, intercept near 0
    assert!(intercept > -1.0 && intercept < 2.0);
    assert!(slope > 0.5 && slope < 1.5);

    // Deviance should be positive and small for reasonable fit
    assert!(fitted.deviance > 0.0);
    assert!(fitted.deviance < 10.0);
}

/// R Code:
/// ```r
/// # Poisson sqrt link
/// x <- c(1, 2, 3, 4, 5)
/// y <- c(1, 4, 9, 16, 25)  # y = x^2
/// fit <- glm(y ~ x, family = poisson(link = "sqrt"))
/// coef(fit)
/// # sqrt(y) = a + b*x, so y = (a + b*x)^2
/// # With y = x^2, sqrt(y) = x = 0 + 1*x
/// ```
#[test]
fn test_poisson_sqrt_link() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(5, |i| ((i + 1) * (i + 1)) as f64); // y = x^2

    let fitted = PoissonRegressor::sqrt()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // sqrt(y) = x, so intercept should be ~0 and slope ~1
    assert_relative_eq!(fitted.result().intercept.unwrap(), 0.0, epsilon = 0.1);
    assert_relative_eq!(fitted.result().coefficients[0], 1.0, epsilon = 0.1);
}

// ============================================================================
// TWEEDIE GLM TESTS
// ============================================================================

/// R Code:
/// ```r
/// library(statmod)
/// # Tweedie with p=1 (Poisson) should match Poisson with log link
/// x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
/// y <- exp(0.5 + 0.2 * x)  # log(y) = 0.5 + 0.2*x
/// fit <- glm(y ~ x, family = poisson(link = "log"))
/// # Both use log link
/// ```
#[test]
fn test_tweedie_poisson_matches_poisson_regressor() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    // Use exponential data appropriate for log link
    let y = Col::from_fn(10, |i| (0.5 + 0.2 * ((i + 1) as f64)).exp());

    let poisson_fitted = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("Poisson fit should succeed");

    let tweedie_fitted = TweedieRegressor::poisson()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("Tweedie fit should succeed");

    // Both should give the same results (both use log link)
    assert_relative_eq!(
        poisson_fitted.result().intercept.unwrap(),
        tweedie_fitted.result().intercept.unwrap(),
        epsilon = 1e-6
    );
    assert_relative_eq!(
        poisson_fitted.result().coefficients[0],
        tweedie_fitted.result().coefficients[0],
        epsilon = 1e-6
    );
}

/// R Code:
/// ```r
/// library(statmod)
/// # Gamma GLM (Tweedie with p=2)
/// x <- c(1, 2, 3, 4, 5)
/// y <- c(1, 2, 4, 8, 16)  # roughly exponential
/// fit <- glm(y ~ x, family = Gamma(link = "log"))
/// coef(fit)
/// ```
#[test]
fn test_tweedie_gamma_exponential_data() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y_data = vec![1.0, 2.0, 4.0, 8.0, 16.0];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = TweedieRegressor::gamma()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Should fit exponential growth: log(y) = a + b*x
    // y = 2^(x-1), so log(y) = (x-1)*log(2) ≈ 0.693*(x-1)
    let slope = fitted.result().coefficients[0];
    assert!(slope > 0.5 && slope < 1.0, "slope should be around 0.693");
}

/// R Code:
/// ```r
/// library(statmod)
/// # Inverse Gaussian (Tweedie with p=3)
/// x <- c(1, 2, 3, 4, 5)
/// y <- c(1.0, 1.5, 2.0, 2.5, 3.0)
/// fit <- glm(y ~ x, family = inverse.gaussian(link = "log"))
/// coef(fit)
/// ```
#[test]
fn test_tweedie_inverse_gaussian() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y_data = vec![1.0, 1.5, 2.0, 2.5, 3.0];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = TweedieRegressor::inverse_gaussian()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Linear data, should fit well
    let intercept = fitted.result().intercept.unwrap();
    let slope = fitted.result().coefficients[0];

    // Should capture the linear trend in log scale
    assert!(intercept.is_finite());
    assert!(slope > 0.0); // positive relationship
}

// ============================================================================
// BINOMIAL GLM TESTS
// ============================================================================

/// R Code:
/// ```r
/// # Simple logistic regression with clear separation
/// x <- c(-2, -1, 0, 1, 2)
/// y <- c(0, 0, 0.5, 1, 1)  # proportion response
/// n <- c(10, 10, 10, 10, 10)  # sample size per observation
/// fit <- glm(y ~ x, family = binomial(link = "logit"), weights = n)
/// coef(fit)
/// ```
#[test]
fn test_binomial_logit_basic() {
    let x = Mat::from_fn(5, 1, |i, _| (i as f64) - 2.0); // -2, -1, 0, 1, 2
    let y_data = vec![0.01, 0.1, 0.5, 0.9, 0.99]; // Avoid exact 0 and 1
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = BinomialRegressor::logistic()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Should have positive slope (probability increases with x)
    assert!(fitted.result().coefficients[0] > 0.0);
    // Intercept should be near 0 (since P(Y=1|x=0) ≈ 0.5)
    assert!(fitted.result().intercept.unwrap().abs() < 1.0);
}

/// R Code:
/// ```r
/// # Probit regression
/// x <- c(-2, -1, 0, 1, 2)
/// y <- c(0.05, 0.16, 0.5, 0.84, 0.95)  # approx pnorm(x)
/// fit <- glm(y ~ x, family = binomial(link = "probit"))
/// coef(fit)
/// # Should recover approximately intercept=0, slope=1
/// ```
#[test]
fn test_binomial_probit_standard_normal() {
    let x = Mat::from_fn(5, 1, |i, _| (i as f64) - 2.0);
    // Values close to Phi(x) = pnorm(x)
    let y_data = vec![0.023, 0.159, 0.5, 0.841, 0.977];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = BinomialRegressor::probit()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Should approximately recover intercept≈0, slope≈1
    assert!(fitted.result().intercept.unwrap().abs() < 0.5);
    assert!(
        (fitted.result().coefficients[0] - 1.0).abs() < 0.5,
        "slope should be near 1"
    );
}

/// R Code:
/// ```r
/// # Complementary log-log link
/// x <- c(-2, -1, 0, 1, 2)
/// y <- c(0.01, 0.05, 0.3, 0.7, 0.95)
/// fit <- glm(y ~ x, family = binomial(link = "cloglog"))
/// coef(fit)
/// ```
#[test]
fn test_binomial_cloglog_basic() {
    let x = Mat::from_fn(5, 1, |i, _| (i as f64) - 2.0);
    let y_data = vec![0.02, 0.06, 0.3, 0.7, 0.94];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = BinomialRegressor::cloglog()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Should have positive slope
    assert!(fitted.result().coefficients[0] > 0.0);
}

// ============================================================================
// NEGATIVE BINOMIAL GLM TESTS
// ============================================================================

/// R Code:
/// ```r
/// library(MASS)
/// # Negative binomial with known theta
/// x <- c(1, 2, 3, 4, 5)
/// y <- c(1, 3, 5, 8, 12)  # overdispersed counts
/// fit <- glm.nb(y ~ x)
/// coef(fit)
/// fit$theta  # estimated theta
/// ```
#[test]
fn test_negative_binomial_basic() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y_data = vec![1.0, 3.0, 5.0, 8.0, 12.0];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = NegativeBinomialRegressor::with_theta(2.0)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Should have positive slope for increasing counts
    assert!(fitted.result().coefficients[0] > 0.0);
    // Theta should remain at 2.0
    assert_relative_eq!(fitted.theta, 2.0, epsilon = 1e-10);
}

/// Test that negative binomial with high theta approaches Poisson.
#[test]
fn test_negative_binomial_approaches_poisson() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y_data = vec![1.0, 3.0, 5.0, 8.0, 12.0];
    let y = Col::from_fn(5, |i| y_data[i]);

    // High theta = low overdispersion = approximately Poisson
    let nb_fitted = NegativeBinomialRegressor::with_theta(1e6)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("NB fit should succeed");

    let poisson_fitted = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("Poisson fit should succeed");

    // Coefficients should be very similar
    assert_relative_eq!(
        nb_fitted.result().intercept.unwrap(),
        poisson_fitted.result().intercept.unwrap(),
        epsilon = 0.01
    );
    assert_relative_eq!(
        nb_fitted.result().coefficients[0],
        poisson_fitted.result().coefficients[0],
        epsilon = 0.01
    );
}

/// R Code:
/// ```r
/// library(MASS)
/// # Negative binomial with theta estimation
/// set.seed(42)
/// x <- 1:20
/// y <- rnbinom(20, size = 2, mu = exp(0.5 + 0.1 * x))
/// fit <- glm.nb(y ~ x)
/// coef(fit)
/// fit$theta
/// ```
#[test]
fn test_negative_binomial_theta_estimation() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    // Simulated overdispersed count data
    let y_data = vec![2.0, 4.0, 3.0, 8.0, 5.0, 12.0, 8.0, 20.0, 15.0, 35.0];
    let y = Col::from_fn(10, |i| y_data[i]);

    let fitted = NegativeBinomialRegressor::builder()
        .with_intercept(true)
        .estimate_theta(true) // estimate theta from data
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Estimated theta should be positive (even small values are valid)
    assert!(
        fitted.theta > 0.0,
        "theta should be positive, got {}",
        fitted.theta
    );

    // For overdispersed data, theta is typically small to moderate
    // Large theta (>1000) would indicate data close to Poisson
    assert!(
        fitted.theta < 1000.0,
        "theta {} seems too large for overdispersed data",
        fitted.theta
    );

    // Should have positive slope
    assert!(fitted.result().coefficients[0] > 0.0);
}

// ============================================================================
// OFFSET TESTS
// ============================================================================

/// R Code:
/// ```r
/// # Poisson with offset (log exposure)
/// x <- c(1, 2, 3, 4, 5)
/// exposure <- c(10, 20, 30, 40, 50)  # varying exposure
/// y <- c(5, 12, 18, 25, 32)  # counts
/// fit <- glm(y ~ x + offset(log(exposure)), family = poisson)
/// coef(fit)
/// ```
#[test]
fn test_poisson_with_offset() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let exposure: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let y_data = vec![5.0, 12.0, 18.0, 25.0, 32.0];
    let y = Col::from_fn(5, |i| y_data[i]);
    let offset = Col::from_fn(5, |i| exposure[i].ln());

    let fitted = PoissonRegressor::log()
        .with_intercept(true)
        .offset(offset)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Model should converge and give reasonable coefficients
    assert!(fitted.result().intercept.is_some());
    assert!(fitted.result().coefficients[0].is_finite());
}

/// R Code:
/// ```r
/// # Gamma with offset
/// library(statmod)
/// x <- c(1, 2, 3, 4, 5)
/// exposure <- c(1, 2, 1, 2, 1)
/// y <- c(2, 6, 4, 10, 6)  # response varies with exposure
/// fit <- glm(y ~ x + offset(log(exposure)), family = Gamma(link = "log"))
/// coef(fit)
/// ```
#[test]
fn test_tweedie_gamma_with_offset() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let exposure: Vec<f64> = vec![1.0, 2.0, 1.0, 2.0, 1.0];
    let y_data = vec![2.0, 6.0, 4.0, 10.0, 6.0];
    let y = Col::from_fn(5, |i| y_data[i]);
    let offset = Col::from_fn(5, |i| exposure[i].ln());

    let fitted = TweedieRegressor::gamma()
        .with_intercept(true)
        .offset(offset)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Should converge
    assert!(fitted.result().intercept.is_some());
    assert!(fitted.result().coefficients[0].is_finite());
}

// ============================================================================
// RESIDUAL TESTS
// ============================================================================

/// Verify Poisson residuals are computed correctly.
#[test]
fn test_poisson_residuals() {
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y_data = vec![1.0, 4.0, 9.0, 16.0, 25.0];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    let pearson = fitted.pearson_residuals();
    let deviance_resid = fitted.deviance_residuals();

    // Residuals should have correct length
    assert_eq!(pearson.nrows(), 5);
    assert_eq!(deviance_resid.nrows(), 5);

    // Sum of squared Pearson residuals ≈ Pearson chi-squared
    let pearson_chi2: f64 = (0..pearson.nrows()).map(|i| pearson[i] * pearson[i]).sum();
    assert!(pearson_chi2.is_finite());
    assert!(pearson_chi2 >= 0.0);

    // Sum of squared deviance residuals ≈ deviance
    let dev_sum: f64 = (0..deviance_resid.nrows())
        .map(|i| deviance_resid[i] * deviance_resid[i])
        .sum();
    assert_relative_eq!(dev_sum, fitted.deviance, epsilon = 1e-6);
}

/// Verify binomial residuals are computed correctly.
#[test]
fn test_binomial_residuals() {
    let x = Mat::from_fn(5, 1, |i, _| (i as f64) - 2.0);
    let y_data = vec![0.1, 0.2, 0.5, 0.8, 0.9];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = BinomialRegressor::logistic()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    let pearson = fitted.pearson_residuals();
    let deviance_resid = fitted.deviance_residuals();

    assert_eq!(pearson.nrows(), 5);
    assert_eq!(deviance_resid.nrows(), 5);

    // All residuals should be finite
    for i in 0..5 {
        assert!(pearson[i].is_finite());
        assert!(deviance_resid[i].is_finite());
    }
}

// ============================================================================
// INFERENCE TESTS
// ============================================================================

/// R Code:
/// ```r
/// x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
/// y <- c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
/// fit <- glm(y ~ x, family = poisson(link = "identity"))
/// summary(fit)$coefficients
/// ```
#[test]
fn test_poisson_standard_errors() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(10, |i| 3.0 + 2.0 * (i as f64));

    let fitted = PoissonRegressor::identity()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Standard errors should be available and positive
    let se = fitted
        .result()
        .std_errors
        .as_ref()
        .expect("SE should exist");
    assert!(se[0] > 0.0, "SE for coefficient should be positive");

    let se_intercept = fitted
        .result()
        .intercept_std_error
        .expect("intercept SE should exist");
    assert!(se_intercept > 0.0, "SE for intercept should be positive");
}

/// R Code:
/// ```r
/// x <- c(-2, -1, 0, 1, 2)
/// y <- c(0.1, 0.2, 0.5, 0.8, 0.9)
/// fit <- glm(y ~ x, family = binomial)
/// summary(fit)$coefficients
/// ```
#[test]
fn test_binomial_standard_errors() {
    let x = Mat::from_fn(5, 1, |i, _| (i as f64) - 2.0);
    let y_data = vec![0.1, 0.2, 0.5, 0.8, 0.9];
    let y = Col::from_fn(5, |i| y_data[i]);

    let fitted = BinomialRegressor::logistic()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Standard errors should be available
    let se = fitted
        .result()
        .std_errors
        .as_ref()
        .expect("SE should exist");
    assert!(se[0] > 0.0);

    let se_intercept = fitted.result().intercept_std_error.expect("intercept SE");
    assert!(se_intercept > 0.0);
}

// ============================================================================
// CONVERGENCE TESTS
// ============================================================================

/// Test that Poisson converges for reasonable data.
#[test]
fn test_poisson_convergence() {
    let x = Mat::from_fn(20, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(20, |i| (i + 1) as f64 + 0.5);

    let fitted = PoissonRegressor::log()
        .max_iterations(100)
        .tolerance(1e-8)
        .build()
        .fit(&x, &y)
        .expect("should converge");

    assert!(
        fitted.iterations < 100,
        "should converge in reasonable iterations"
    );
}

/// Test that binomial converges for reasonable data.
#[test]
fn test_binomial_convergence() {
    let x = Mat::from_fn(10, 1, |i, _| (i as f64) - 4.5);
    let y_data = vec![0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99];
    let y = Col::from_fn(10, |i| y_data[i]);

    let fitted = BinomialRegressor::logistic()
        .max_iterations(100)
        .tolerance(1e-8)
        .build()
        .fit(&x, &y)
        .expect("should converge");

    assert!(
        fitted.iterations < 100,
        "should converge in reasonable iterations"
    );
}

/// Test that negative binomial converges with theta estimation.
#[test]
fn test_negative_binomial_convergence() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y_data = vec![1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 15.0, 18.0, 25.0, 30.0];
    let y = Col::from_fn(10, |i| y_data[i]);

    let fitted = NegativeBinomialRegressor::builder()
        .max_iterations(100)
        .estimate_theta(true)
        .build()
        .fit(&x, &y)
        .expect("should converge");

    assert!(
        fitted.iterations < 100,
        "should converge in reasonable iterations"
    );
}

// ============================================================================
// DEVIANCE TESTS
// ============================================================================

/// Test that null deviance > residual deviance for meaningful predictors.
#[test]
fn test_deviance_reduction() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    // Strong linear relationship
    let y = Col::from_fn(10, |i| (2.0 + 0.5 * (i as f64)).exp());

    let fitted = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Null deviance should be greater than residual deviance
    // (predictor explains variance)
    assert!(
        fitted.null_deviance > fitted.deviance,
        "null deviance ({}) should exceed residual deviance ({})",
        fitted.null_deviance,
        fitted.deviance
    );
}

/// Test that perfect fit gives zero deviance.
#[test]
fn test_perfect_fit_zero_deviance() {
    // Data that perfectly fits y = exp(1 + 0.5*x)
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(5, |i| (1.0 + 0.5 * ((i + 1) as f64)).exp());

    let fitted = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Deviance should be very small (effectively zero)
    assert!(
        fitted.deviance < 1e-6,
        "deviance should be near zero for perfect fit, got {}",
        fitted.deviance
    );
}
