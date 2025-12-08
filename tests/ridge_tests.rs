//! Ridge regression tests.

mod common;

use approx::assert_relative_eq;
use faer::{Col, Mat};
use regress_rs::solvers::{FittedRegressor, Regressor, RidgeRegressor};

// ============================================================================
// Basic Ridge Regression Tests
// ============================================================================

#[test]
fn test_basic_ridge_regression() {
    // Simple linear regression with small regularization
    let x = Mat::from_fn(10, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y = Col::zeros(10);
    for i in 0..10 {
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)];
    }

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.01)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // With small lambda, should be close to OLS solution
    assert!(fitted.r_squared() > 0.99);
}

#[test]
fn test_lambda_zero_equals_ols() {
    // When lambda = 0, Ridge should equal OLS
    let (x, y, _) = common::generate_linear_data(50, 3, 2.0, 0.1, 42);

    let ridge = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.0)
        .build();

    let ols = regress_rs::solvers::OlsRegressor::builder()
        .with_intercept(true)
        .build();

    let ridge_fit = ridge.fit(&x, &y).expect("ridge fit should succeed");
    let ols_fit = ols.fit(&x, &y).expect("ols fit should succeed");

    // Coefficients should be nearly identical
    for i in 0..3 {
        assert_relative_eq!(
            ridge_fit.coefficients()[i],
            ols_fit.coefficients()[i],
            epsilon = 1e-8
        );
    }

    assert_relative_eq!(
        ridge_fit.intercept().unwrap(),
        ols_fit.intercept().unwrap(),
        epsilon = 1e-8
    );
}

#[test]
fn test_increasing_lambda_shrinks_coefficients() {
    // As lambda increases, coefficients should shrink toward zero
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let lambdas = [0.0, 0.1, 1.0, 10.0, 100.0];
    let mut prev_coef_norm = f64::INFINITY;

    for lambda in lambdas {
        let model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(lambda)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");

        // Compute L2 norm of coefficients
        let coef_norm: f64 = fitted
            .coefficients()
            .iter()
            .map(|&c| c.powi(2))
            .sum::<f64>()
            .sqrt();

        // Coefficient norm should decrease (or stay same) as lambda increases
        assert!(
            coef_norm <= prev_coef_norm + 1e-10,
            "lambda={}: coef_norm={} should be <= prev={}",
            lambda,
            coef_norm,
            prev_coef_norm
        );

        prev_coef_norm = coef_norm;
    }
}

#[test]
fn test_intercept_not_penalized() {
    // Intercept should not shrink toward zero with increasing lambda
    let mut x = Mat::zeros(20, 1);
    let mut y = Col::zeros(20);

    // y = 100 + x with large intercept
    for i in 0..20 {
        x[(i, 0)] = i as f64 * 0.1;
        y[i] = 100.0 + x[(i, 0)];
    }

    let fitted_low = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    let fitted_high = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(10.0)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Intercept should remain relatively stable
    let intercept_diff = (fitted_low.intercept().unwrap() - fitted_high.intercept().unwrap()).abs();
    assert!(
        intercept_diff < 10.0,
        "Intercept should be relatively stable, diff={}",
        intercept_diff
    );
}

#[test]
fn test_ridge_without_intercept() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(10, |i| 3.0 * (i + 1) as f64);

    let model = RidgeRegressor::builder()
        .with_intercept(false)
        .lambda(0.01)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.intercept().is_none());
    assert!(fitted.r_squared() > 0.99);
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_ridge_predict() {
    let x_train = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y_train = Col::zeros(20);
    for i in 0..20 {
        y_train[i] = 1.0 + 2.0 * x_train[(i, 0)] + 3.0 * x_train[(i, 1)];
    }

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.01)
        .build();

    let fitted = model.fit(&x_train, &y_train).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(5, 2, |i, j| ((i + j + 20) as f64) * 0.1);
    let predictions = fitted.predict(&x_new);

    assert_eq!(predictions.nrows(), 5);

    // Predictions should be reasonable
    for p in predictions.iter() {
        assert!(p.is_finite());
    }
}

#[test]
fn test_ridge_score() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.1, 42);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let score = fitted.score(&x, &y);

    // Score should be close to r_squared on training data
    assert_relative_eq!(score, fitted.r_squared(), epsilon = 1e-10);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_ridge_handles_collinearity() {
    // Ridge should handle collinear features better than OLS
    let (x, y) = common::generate_collinear_data(20);

    // OLS will have aliased coefficients
    let ols = regress_rs::solvers::OlsRegressor::builder()
        .with_intercept(true)
        .build();
    let ols_fit = ols.fit(&x, &y).expect("ols fit should succeed");
    assert!(ols_fit.result().has_aliased());

    // Ridge with lambda > 0 should have all finite coefficients
    let ridge = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();
    let ridge_fit = ridge.fit(&x, &y).expect("ridge fit should succeed");

    // All coefficients should be finite (no NaN)
    for coef in ridge_fit.coefficients().iter() {
        assert!(coef.is_finite(), "Ridge coefficient should be finite");
    }
}

#[test]
fn test_ridge_r_squared_bounds() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(1.0)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R² should be in [0, 1]
    assert!(fitted.r_squared() >= 0.0);
    assert!(fitted.r_squared() <= 1.0);
}

#[test]
fn test_ridge_dimension_mismatch() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(5); // Wrong size

    let model = RidgeRegressor::builder().lambda(0.1).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

// ============================================================================
// Inference Tests
// ============================================================================

#[test]
fn test_ridge_inference() {
    let (x, y, _) = common::generate_linear_data(100, 2, 1.0, 0.5, 42);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    // Should have inference statistics
    assert!(result.std_errors.is_some());
    assert!(result.t_statistics.is_some());
    assert!(result.p_values.is_some());
    assert!(result.conf_interval_lower.is_some());
    assert!(result.conf_interval_upper.is_some());

    // Standard errors should be positive
    if let Some(ref se) = result.std_errors {
        for i in 0..se.nrows() {
            assert!(se[i] > 0.0, "SE[{}] should be positive", i);
        }
    }
}

#[test]
fn test_ridge_prediction_intervals() {
    use regress_rs::core::IntervalType;

    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let x_new = Mat::from_fn(5, 2, |i, j| ((i + j + 50) as f64) * 0.1);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    assert_eq!(result.fit.nrows(), 5);
    assert_eq!(result.lower.nrows(), 5);
    assert_eq!(result.upper.nrows(), 5);

    // Lower should be less than fit, fit less than upper
    for i in 0..5 {
        if result.lower[i].is_finite() && result.upper[i].is_finite() {
            assert!(result.lower[i] <= result.fit[i]);
            assert!(result.fit[i] <= result.upper[i]);
        }
    }
}

#[test]
fn test_ridge_lambda_scaling_glmnet() {
    use regress_rs::core::LambdaScaling;

    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);

    // Compare raw vs glmnet scaling
    let model_raw = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .lambda_scaling(LambdaScaling::Raw)
        .build();

    let model_glmnet = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .lambda_scaling(LambdaScaling::Glmnet)
        .build();

    let fit_raw = model_raw.fit(&x, &y).expect("fit should succeed");
    let fit_glmnet = model_glmnet.fit(&x, &y).expect("fit should succeed");

    // Both should fit, but coefficients will differ due to scaling
    assert!(fit_raw.r_squared() > 0.9);
    assert!(fit_glmnet.r_squared() > 0.9);
}

#[test]
fn test_ridge_getters() {
    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.5)
        .build();

    let x = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(20, |i| i as f64);

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Test getters
    assert_relative_eq!(fitted.lambda(), 0.5, epsilon = 1e-10);
    assert!(fitted.options().with_intercept);
}

#[test]
fn test_ridge_confidence_level() {
    let (x, y, _) = common::generate_linear_data(100, 2, 1.0, 0.5, 42);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .compute_inference(true)
        .confidence_level(0.99)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    assert_relative_eq!(result.confidence_level, 0.99, epsilon = 1e-10);
}

#[test]
fn test_ridge_no_inference() {
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.5, 42);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .compute_inference(false)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    // Should not have inference statistics
    assert!(result.std_errors.is_none());
    assert!(result.t_statistics.is_none());
    assert!(result.p_values.is_none());
}

#[test]
fn test_ridge_very_high_regularization() {
    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(10000.0)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // With extreme regularization, coefficients should be very small
    for coef in fitted.coefficients().iter() {
        assert!(coef.abs() < 1.0, "Coefficient should be shrunk toward zero");
    }
}

#[test]
fn test_ridge_single_observation() {
    let x = Mat::from_fn(1, 2, |_, j| j as f64);
    let y = Col::from_fn(1, |_| 5.0);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();

    let result = model.fit(&x, &y);
    // Should error due to insufficient observations
    assert!(result.is_err());
}
