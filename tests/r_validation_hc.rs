//! Integration tests for HC (Heteroskedasticity-Consistent) standard errors.
//!
//! Reference values computed analytically from the sandwich estimator formulas,
//! matching R's `sandwich::vcovHC()` output.
//!
//! R validation code:
//! ```r
//! library(sandwich)
//! library(lmtest)
//!
//! # Simple heteroskedastic data
//! x <- c(1, 2, 3, 4, 5)
//! y <- c(2.1, 4.3, 5.8, 8.2, 9.9)
//! m <- lm(y ~ x)
//!
//! sqrt(diag(vcovHC(m, type="HC0")))  # [1] 0.12237 0.03178
//! sqrt(diag(vcovHC(m, type="HC1")))  # [1] 0.15796 0.04103
//! sqrt(diag(vcovHC(m, type="HC2")))  # [1] 0.15489 0.04185
//! sqrt(diag(vcovHC(m, type="HC3")))  # [1] 0.20343 0.05722
//!
//! # No-intercept model
//! m0 <- lm(y ~ x - 1)
//! sqrt(diag(vcovHC(m0, type="HC0")))  # [1] 0.02377
//! sqrt(diag(vcovHC(m0, type="HC1")))  # [1] 0.02658
//! sqrt(diag(vcovHC(m0, type="HC2")))  # [1] 0.02817
//! sqrt(diag(vcovHC(m0, type="HC3")))  # [1] 0.03408
//!
//! # Multiple predictors
//! set.seed(42)
//! n <- 20
//! x1 <- 1:n
//! x2 <- rnorm(n)
//! y <- 1 + 0.5*x1 + 2*x2 + rnorm(n) * x1 * 0.3
//! m2 <- lm(y ~ x1 + x2)
//! coeftest(m2, vcov = vcovHC(m2, type="HC1"))
//! ```

use anofox_regression::inference::{compute_hc_standard_errors, HcType};
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use approx::assert_relative_eq;
use faer::{Col, Mat};

// --------------------------------------------------------------------------
// Test data: simple heteroskedastic case (5 obs, 1 predictor)
// --------------------------------------------------------------------------

const X_SIMPLE: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
const Y_SIMPLE: [f64; 5] = [2.1, 4.3, 5.8, 8.2, 9.9];

// Reference values computed analytically from sandwich formulas.
// OLS: y = 0.21 + 1.95*x
// Residuals: [-0.06, 0.19, -0.26, 0.19, -0.06]
// Leverage:  [0.60, 0.30, 0.20, 0.30, 0.60]

// HC0 SE (intercept, slope)
const HC0_SE_INTERCEPT: f64 = 0.12237;
const HC0_SE_SLOPE: f64 = 0.03178;

// HC1 SE = sqrt(n/(n-p)) * HC0 SE, with n=5, p=2
const HC1_SE_INTERCEPT: f64 = 0.15796;
const HC1_SE_SLOPE: f64 = 0.04103;

// HC2 SE (leverage-corrected)
const HC2_SE_INTERCEPT: f64 = 0.15489;
const HC2_SE_SLOPE: f64 = 0.04185;

// HC3 SE (jackknife-like)
const HC3_SE_INTERCEPT: f64 = 0.20343;
const HC3_SE_SLOPE: f64 = 0.05722;

fn fit_simple_model() -> anofox_regression::solvers::FittedOls {
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(5, |i| Y_SIMPLE[i]);

    OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("simple model should fit")
}

#[test]
fn test_hc0_simple_regression() {
    let fitted = fit_simple_model();
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);

    let hc = fitted.hc_standard_errors(&x, HcType::HC0).unwrap();

    assert_relative_eq!(
        hc.intercept_std_error.unwrap(),
        HC0_SE_INTERCEPT,
        epsilon = 0.01
    );
    assert_relative_eq!(hc.std_errors[0], HC0_SE_SLOPE, epsilon = 0.01);
}

#[test]
fn test_hc1_simple_regression() {
    let fitted = fit_simple_model();
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);

    let hc = fitted.hc_standard_errors(&x, HcType::HC1).unwrap();

    assert_relative_eq!(
        hc.intercept_std_error.unwrap(),
        HC1_SE_INTERCEPT,
        epsilon = 0.01
    );
    assert_relative_eq!(hc.std_errors[0], HC1_SE_SLOPE, epsilon = 0.01);
}

#[test]
fn test_hc2_simple_regression() {
    let fitted = fit_simple_model();
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);

    let hc = fitted.hc_standard_errors(&x, HcType::HC2).unwrap();

    assert_relative_eq!(
        hc.intercept_std_error.unwrap(),
        HC2_SE_INTERCEPT,
        epsilon = 0.01
    );
    assert_relative_eq!(hc.std_errors[0], HC2_SE_SLOPE, epsilon = 0.01);
}

#[test]
fn test_hc3_simple_regression() {
    let fitted = fit_simple_model();
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);

    let hc = fitted.hc_standard_errors(&x, HcType::HC3).unwrap();

    assert_relative_eq!(
        hc.intercept_std_error.unwrap(),
        HC3_SE_INTERCEPT,
        epsilon = 0.01
    );
    assert_relative_eq!(hc.std_errors[0], HC3_SE_SLOPE, epsilon = 0.01);
}

// --------------------------------------------------------------------------
// Test: HC ordering property (HC3 >= HC2 >= HC0 for slope SE)
// --------------------------------------------------------------------------

#[test]
fn test_hc_ordering() {
    let fitted = fit_simple_model();
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);

    let hc0 = fitted.hc_standard_errors(&x, HcType::HC0).unwrap();
    let hc2 = fitted.hc_standard_errors(&x, HcType::HC2).unwrap();
    let hc3 = fitted.hc_standard_errors(&x, HcType::HC3).unwrap();

    assert!(
        hc3.std_errors[0] >= hc2.std_errors[0] - 1e-10,
        "HC3 ({}) should be >= HC2 ({})",
        hc3.std_errors[0],
        hc2.std_errors[0]
    );
    assert!(
        hc2.std_errors[0] >= hc0.std_errors[0] - 1e-10,
        "HC2 ({}) should be >= HC0 ({})",
        hc2.std_errors[0],
        hc0.std_errors[0]
    );
}

// --------------------------------------------------------------------------
// Test: Full HC inference (t-stats, p-values, confidence intervals)
// --------------------------------------------------------------------------

#[test]
fn test_hc1_full_inference() {
    let fitted = fit_simple_model();
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);

    let hc = fitted.hc_inference(&x, HcType::HC1).unwrap();

    assert_eq!(hc.hc_type, HcType::HC1);
    assert_eq!(hc.confidence_level, 0.95);

    // SE should match standalone computation
    assert_relative_eq!(hc.std_errors[0], HC1_SE_SLOPE, epsilon = 0.01);

    // t-statistic = coef / se
    let coef = fitted.coefficients()[0];
    let expected_t = coef / hc.std_errors[0];
    assert_relative_eq!(hc.t_statistics[0], expected_t, epsilon = 0.001);

    // p-value should be in [0, 1]
    assert!(hc.p_values[0] >= 0.0 && hc.p_values[0] <= 1.0);

    // Confidence interval should contain the coefficient
    assert!(hc.conf_interval_lower[0] < coef);
    assert!(hc.conf_interval_upper[0] > coef);

    // Intercept inference
    let int_inf = hc
        .intercept
        .as_ref()
        .expect("should have intercept inference");
    assert_relative_eq!(int_inf.std_error, HC1_SE_INTERCEPT, epsilon = 0.01);
    assert!(int_inf.p_value >= 0.0 && int_inf.p_value <= 1.0);
    assert!(int_inf.conf_interval.0 < int_inf.conf_interval.1);
}

// --------------------------------------------------------------------------
// Test: No-intercept model
// --------------------------------------------------------------------------

// Reference: no-intercept OLS y = β₁x
// β₁ = 110.4/55 = 2.007273
// (X'X)^-1 = 1/55 = 0.018182
// Leverage: h_i = x_i²/55

const HC0_SE_NOINT: f64 = 0.02377;
const HC1_SE_NOINT: f64 = 0.02658;
const HC2_SE_NOINT: f64 = 0.02817;
const HC3_SE_NOINT: f64 = 0.03408;

#[test]
fn test_hc_no_intercept() {
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(5, |i| Y_SIMPLE[i]);

    let fitted = OlsRegressor::builder()
        .with_intercept(false)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("no-intercept model should fit");

    let hc0 = fitted.hc_standard_errors(&x, HcType::HC0).unwrap();
    assert!(hc0.intercept_std_error.is_none());
    assert_relative_eq!(hc0.std_errors[0], HC0_SE_NOINT, epsilon = 0.01);

    let hc1 = fitted.hc_standard_errors(&x, HcType::HC1).unwrap();
    assert_relative_eq!(hc1.std_errors[0], HC1_SE_NOINT, epsilon = 0.01);

    let hc2 = fitted.hc_standard_errors(&x, HcType::HC2).unwrap();
    assert_relative_eq!(hc2.std_errors[0], HC2_SE_NOINT, epsilon = 0.01);

    let hc3 = fitted.hc_standard_errors(&x, HcType::HC3).unwrap();
    assert_relative_eq!(hc3.std_errors[0], HC3_SE_NOINT, epsilon = 0.01);
}

// --------------------------------------------------------------------------
// Test: Multiple predictors
// --------------------------------------------------------------------------

#[test]
fn test_hc_multiple_predictors() {
    // 10 observations, 2 predictors with heteroskedastic errors
    let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2 = [0.5, -1.2, 0.8, -0.3, 1.5, -0.7, 0.2, 1.1, -0.9, 0.6];
    let y_data = [3.2, 3.5, 6.1, 7.0, 10.8, 11.0, 14.5, 17.2, 17.0, 21.5];

    let n = 10;
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { x1[i] } else { x2[i] });
    let y = Col::from_fn(n, |i| y_data[i]);

    let fitted = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("multi-predictor model should fit");

    // All 4 HC types should produce valid results
    for hc_type in [HcType::HC0, HcType::HC1, HcType::HC2, HcType::HC3] {
        let hc = fitted.hc_standard_errors(&x, hc_type).unwrap();

        // Both coefficient SEs should be positive and finite
        assert!(
            hc.std_errors[0] > 0.0 && hc.std_errors[0].is_finite(),
            "{:?}: x1 SE should be positive finite, got {}",
            hc_type,
            hc.std_errors[0]
        );
        assert!(
            hc.std_errors[1] > 0.0 && hc.std_errors[1].is_finite(),
            "{:?}: x2 SE should be positive finite, got {}",
            hc_type,
            hc.std_errors[1]
        );

        // Intercept SE should be positive and finite
        let int_se = hc.intercept_std_error.unwrap();
        assert!(
            int_se > 0.0 && int_se.is_finite(),
            "{:?}: intercept SE should be positive finite, got {}",
            hc_type,
            int_se
        );
    }

    // HC ordering: HC3 >= HC2 >= HC0 for each coefficient
    let hc0 = fitted.hc_standard_errors(&x, HcType::HC0).unwrap();
    let hc2 = fitted.hc_standard_errors(&x, HcType::HC2).unwrap();
    let hc3 = fitted.hc_standard_errors(&x, HcType::HC3).unwrap();

    for j in 0..2 {
        assert!(
            hc3.std_errors[j] >= hc2.std_errors[j] - 1e-10,
            "HC3[{}] ({}) should be >= HC2[{}] ({})",
            j,
            hc3.std_errors[j],
            j,
            hc2.std_errors[j]
        );
        assert!(
            hc2.std_errors[j] >= hc0.std_errors[j] - 1e-10,
            "HC2[{}] ({}) should be >= HC0[{}] ({})",
            j,
            hc2.std_errors[j],
            j,
            hc0.std_errors[j]
        );
    }
}

// --------------------------------------------------------------------------
// Test: Full inference with multiple predictors
// --------------------------------------------------------------------------

#[test]
fn test_hc_inference_multiple_predictors() {
    let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let x2 = [0.5, -1.2, 0.8, -0.3, 1.5, -0.7, 0.2, 1.1, -0.9, 0.6];
    let y_data = [3.2, 3.5, 6.1, 7.0, 10.8, 11.0, 14.5, 17.2, 17.0, 21.5];

    let n = 10;
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { x1[i] } else { x2[i] });
    let y = Col::from_fn(n, |i| y_data[i]);

    let fitted = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .confidence_level(0.95)
        .build()
        .fit(&x, &y)
        .expect("model should fit");

    let hc = fitted.hc_inference(&x, HcType::HC3).unwrap();

    // Check all inference fields are populated
    assert_eq!(hc.hc_type, HcType::HC3);
    assert_eq!(hc.confidence_level, 0.95);

    for j in 0..2 {
        assert!(hc.std_errors[j] > 0.0 && hc.std_errors[j].is_finite());
        assert!(hc.t_statistics[j].is_finite());
        assert!(hc.p_values[j] >= 0.0 && hc.p_values[j] <= 1.0);
        assert!(hc.conf_interval_lower[j] < hc.conf_interval_upper[j]);

        // t-stat should equal coef / SE
        let expected_t = fitted.coefficients()[j] / hc.std_errors[j];
        assert_relative_eq!(hc.t_statistics[j], expected_t, epsilon = 0.001);
    }

    // Intercept inference
    let int_inf = hc
        .intercept
        .as_ref()
        .expect("should have intercept inference");
    assert!(int_inf.std_error > 0.0);
    assert!(int_inf.t_statistic.is_finite());
    assert!(int_inf.p_value >= 0.0 && int_inf.p_value <= 1.0);
    assert!(int_inf.conf_interval.0 < int_inf.conf_interval.1);
}

// --------------------------------------------------------------------------
// Test: HC SEs are generally larger than classical SEs for heteroskedastic data
// --------------------------------------------------------------------------

#[test]
fn test_hc_vs_classical_heteroskedastic() {
    // Data with clear heteroskedasticity (variance increases with x)
    let n = 20;
    let x_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    // y = 1 + 2*x + noise * x (heteroskedastic)
    let y_data = [
        2.5, 5.8, 6.2, 10.1, 10.3, 14.8, 12.9, 18.5, 16.2, 23.7, 20.1, 27.9, 24.3, 31.6, 28.4,
        35.8, 33.1, 39.2, 36.5, 43.1,
    ];

    let x = Mat::from_fn(n, 1, |i, _| x_data[i]);
    let y = Col::from_fn(n, |i| y_data[i]);

    let fitted = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("model should fit");

    let classical_se = fitted
        .result()
        .std_errors
        .as_ref()
        .expect("classical SE should exist");

    // HC1 should generally differ from classical SE under heteroskedasticity
    let hc1 = fitted.hc_standard_errors(&x, HcType::HC1).unwrap();

    // Both should be positive and finite
    assert!(hc1.std_errors[0] > 0.0 && hc1.std_errors[0].is_finite());
    assert!(classical_se[0] > 0.0 && classical_se[0].is_finite());

    // They should differ (exact direction depends on heteroskedasticity pattern)
    let ratio = hc1.std_errors[0] / classical_se[0];
    assert!(
        (ratio - 1.0).abs() > 1e-6,
        "HC1 and classical SE should differ for heteroskedastic data, ratio = {}",
        ratio
    );
}

// --------------------------------------------------------------------------
// Test: Direct compute_hc_standard_errors function
// --------------------------------------------------------------------------

#[test]
fn test_compute_hc_standard_errors_directly() {
    // Verify the low-level function works with known residuals
    let x = Mat::from_fn(5, 1, |i, _| X_SIMPLE[i]);
    let residuals = Col::from_fn(5, |i| [-0.06, 0.19, -0.26, 0.19, -0.06][i]);
    let aliased = vec![false];

    let result = compute_hc_standard_errors(&x, &residuals, &aliased, true, HcType::HC0).unwrap();

    assert_relative_eq!(
        result.intercept_std_error.unwrap(),
        HC0_SE_INTERCEPT,
        epsilon = 0.01
    );
    assert_relative_eq!(result.std_errors[0], HC0_SE_SLOPE, epsilon = 0.01);
}
