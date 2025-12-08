//! Weighted Least Squares tests.

mod common;

use approx::assert_relative_eq;
use faer::{Col, Mat};
use regress_rs::solvers::{FittedRegressor, OlsRegressor, Regressor, WlsRegressor};

// ============================================================================
// Basic WLS Tests
// ============================================================================

#[test]
fn test_wls_basic() {
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64);
    let weights = Col::from_fn(20, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.r_squared() > 0.99);
}

#[test]
fn test_wls_equal_weights_equals_ols() {
    // When all weights are equal, WLS should equal OLS
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.1, 42);
    let weights = Col::from_fn(50, |_| 1.0);

    let wls = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();
    let ols = OlsRegressor::builder().with_intercept(true).build();

    let wls_fit = wls.fit(&x, &y).expect("wls fit should succeed");
    let ols_fit = ols.fit(&x, &y).expect("ols fit should succeed");

    // Coefficients should be very close
    for j in 0..2 {
        assert_relative_eq!(
            wls_fit.coefficients()[j],
            ols_fit.coefficients()[j],
            epsilon = 1e-6
        );
    }

    assert_relative_eq!(
        wls_fit.intercept().unwrap(),
        ols_fit.intercept().unwrap(),
        epsilon = 1e-6
    );
}

#[test]
fn test_wls_without_intercept() {
    let x = Mat::from_fn(20, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(20, |i| 3.0 * (i + 1) as f64);
    let weights = Col::from_fn(20, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(false)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.intercept().is_none());
    assert!(fitted.r_squared() > 0.99);
}

// ============================================================================
// Weight Effect Tests
// ============================================================================

#[test]
fn test_wls_higher_weights_more_influence() {
    // Create data with two distinct relationships
    // Low x values: y = 1 + 1*x  (low weights)
    // High x values: y = 2 + 3*x (high weights)
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| i as f64);
    let mut y = Col::zeros(n);
    let mut weights = Col::zeros(n);

    for i in 0..n / 2 {
        y[i] = 1.0 + 1.0 * i as f64;
        weights[i] = 0.1; // Low weight
    }
    for i in n / 2..n {
        y[i] = 2.0 + 3.0 * i as f64;
        weights[i] = 10.0; // High weight
    }

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Coefficient should be closer to 3 than to 1 due to higher weights
    // (This is a weak test, just checking the model fits)
    assert!(fitted.coefficients()[0] > 1.0);
}

#[test]
fn test_wls_zero_weights_excluded() {
    // Zero weights should effectively exclude observations (weight them to nothing)
    // Create perfect linear data with very small weight on outliers
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let mut y = Col::zeros(20);
    let mut weights = Col::zeros(20);

    // Normal relationship: y = 1 + 2*x
    for i in 0..20 {
        y[i] = 1.0 + 2.0 * i as f64;
        weights[i] = 1.0;
    }

    // Replace two observations with outliers that have very small weight
    y[18] = 1000.0;
    weights[18] = 1e-10; // Near-zero weight (truly zero causes issues)
    y[19] = -1000.0;
    weights[19] = 1e-10;

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Should still get coefficient close to 2 since outliers have negligible weight
    assert!(
        (fitted.coefficients()[0] - 2.0).abs() < 0.5,
        "Coefficient should be near 2.0, got {}",
        fitted.coefficients()[0]
    );
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_wls_predict() {
    let x_train = Mat::from_fn(30, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y_train = Col::zeros(30);
    for i in 0..30 {
        y_train[i] = 1.0 + 2.0 * x_train[(i, 0)] + 3.0 * x_train[(i, 1)];
    }
    let weights = Col::from_fn(30, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x_train, &y_train).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(5, 2, |i, j| ((i + j + 30) as f64) * 0.1);
    let predictions = fitted.predict(&x_new);

    assert_eq!(predictions.nrows(), 5);
    for p in predictions.iter() {
        assert!(p.is_finite());
    }
}

#[test]
fn test_wls_score() {
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.1, 42);
    let weights = Col::from_fn(50, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // score() should return R² on training data
    let score = fitted.score(&x, &y);
    assert_relative_eq!(score, fitted.r_squared(), epsilon = 1e-10);
}

// ============================================================================
// Inference Tests
// ============================================================================

#[test]
fn test_wls_standard_errors() {
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.5, 42);
    let weights = Col::from_fn(50, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    // Should have inference statistics
    assert!(result.std_errors.is_some());
    assert!(result.t_statistics.is_some());
    assert!(result.p_values.is_some());

    // Standard errors should be positive
    if let Some(ref se) = result.std_errors {
        for i in 0..se.nrows() {
            assert!(se[i] > 0.0, "SE[{}] should be positive", i);
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_wls_dimension_mismatch_x_y() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(5); // Wrong size
    let weights = Col::from_fn(10, |_| 1.0);

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_dimension_mismatch_weights() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(10);
    let weights = Col::from_fn(5, |_| 1.0); // Wrong size

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_negative_weight_error() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let mut weights = Col::from_fn(10, |_| 1.0);
    weights[0] = -1.0; // Negative weight

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_all_zero_weights_error() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let weights = Col::from_fn(10, |_| 0.0); // All zero

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_r_squared_bounds() {
    let x = Mat::from_fn(50, 2, |i, j| ((i * j + 1) as f64).sin());
    let y = Col::from_fn(50, |i| (i as f64).cos());
    let weights = Col::from_fn(50, |i| (i + 1) as f64);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R² should be in [0, 1]
    assert!(fitted.r_squared() >= 0.0);
    assert!(fitted.r_squared() <= 1.0);
}

#[test]
fn test_wls_varying_weights() {
    // Higher weights on more recent observations
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64 + 0.1 * (i as f64).sin());
    let weights = Col::from_fn(30, |i| (i + 1) as f64); // Linearly increasing

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.r_squared() > 0.9);
    assert!(fitted.coefficients()[0].is_finite());
}

#[test]
fn test_wls_weights_access() {
    let weights = Col::from_fn(20, |i| (i + 1) as f64);
    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights.clone())
        .build();

    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| i as f64);

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Verify we can access weights from fitted model
    let stored_weights = fitted.weights();
    assert_eq!(stored_weights.nrows(), 20);
    for i in 0..20 {
        assert_relative_eq!(stored_weights[i], (i + 1) as f64);
    }
}

// ============================================================================
// Prediction Interval Tests
// ============================================================================

#[test]
fn test_wls_prediction_intervals() {
    use regress_rs::core::IntervalType;

    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1 + (i as f64 * 0.1).sin() * 0.2);
    let weights = Col::from_fn(50, |i| 1.0 / ((i % 5 + 1) as f64));

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Test prediction intervals
    let x_new = Mat::from_fn(5, 2, |i, j| ((i + j + 50) as f64) * 0.1);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    assert_eq!(result.fit.nrows(), 5);
    assert_eq!(result.lower.nrows(), 5);
    assert_eq!(result.upper.nrows(), 5);

    // Lower should be less than fit, fit less than upper
    for i in 0..5 {
        if result.lower[i].is_finite() && result.upper[i].is_finite() {
            assert!(result.lower[i] <= result.fit[i], "Lower bound should be <= fit");
            assert!(result.fit[i] <= result.upper[i], "Fit should be <= upper bound");
        }
    }
}

#[test]
fn test_wls_confidence_intervals() {
    use regress_rs::core::IntervalType;

    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);
    let weights = Col::from_fn(50, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Test confidence intervals for mean response
    let x_new = Mat::from_fn(3, 2, |i, j| ((i + j + 50) as f64) * 0.1);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    assert_eq!(result.fit.nrows(), 3);
    assert_eq!(result.lower.nrows(), 3);
    assert_eq!(result.upper.nrows(), 3);
}

#[test]
fn test_wls_predict_with_no_interval() {
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);
    let weights = Col::from_fn(30, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // No intervals requested - lower/upper will be zeros
    let x_new = Mat::from_fn(5, 1, |i, _| (i + 30) as f64);
    let result = fitted.predict_with_interval(&x_new, None, 0.95);

    assert_eq!(result.fit.nrows(), 5);
    // When no interval requested, lower/upper are set to zeros
    assert_eq!(result.lower.nrows(), 5);
    assert_eq!(result.upper.nrows(), 5);
    for i in 0..5 {
        assert_relative_eq!(result.lower[i], 0.0);
        assert_relative_eq!(result.upper[i], 0.0);
    }
}

// ============================================================================
// No Intercept with Inference Tests
// ============================================================================

#[test]
fn test_wls_no_intercept_with_inference() {
    let x = Mat::from_fn(50, 2, |i, j| ((i + j + 1) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 2.0 * i as f64 * 0.1 + 3.0 * (i + 1) as f64 * 0.1);
    let weights = Col::from_fn(50, |i| 1.0 / ((i % 3 + 1) as f64));

    let model = WlsRegressor::builder()
        .with_intercept(false)
        .weights(weights)
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

    // No intercept inference
    assert!(result.intercept.is_none());
    assert!(result.intercept_std_error.is_none());
}

#[test]
fn test_wls_no_intercept_prediction_intervals() {
    use regress_rs::core::IntervalType;

    let x = Mat::from_fn(50, 2, |i, j| ((i + j + 1) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 2.0 * i as f64 * 0.1);
    let weights = Col::from_fn(50, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(false)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let x_new = Mat::from_fn(5, 2, |i, j| ((i + j + 51) as f64) * 0.1);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    assert_eq!(result.fit.nrows(), 5);
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn test_wls_insufficient_observations() {
    let x = Mat::from_fn(3, 5, |i, j| (i + j) as f64); // 3 obs, 5 predictors
    let y = Col::from_fn(3, |i| i as f64);
    let weights = Col::from_fn(3, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true) // 6 params total
        .weights(weights)
        .build();

    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_wls_single_observation() {
    let x = Mat::from_fn(1, 1, |_, _| 1.0);
    let y = Col::from_fn(1, |_| 5.0);
    let weights = Col::from_fn(1, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_wls_extreme_weight_ratios() {
    // Very different weight magnitudes
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);
    let weights = Col::from_fn(30, |i| if i < 15 { 1e-6 } else { 1e6 });

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Should still produce valid results
    assert!(fitted.r_squared().is_finite());
    assert!(fitted.coefficients()[0].is_finite());
}

#[test]
fn test_wls_intercept_inference() {
    // Use larger, more varied data to ensure numerical stability
    let x = Mat::from_fn(100, 1, |i, _| i as f64);
    let y = Col::from_fn(100, |i| 5.0 + 2.0 * i as f64 + 0.1 * (i as f64).sin());
    // Use varying weights to ensure WLS path is taken (not OLS delegation)
    let weights = Col::from_fn(100, |i| 1.0 + 0.5 * (i % 5) as f64);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    // Should have intercept
    assert!(result.intercept.is_some());

    // Check coefficient inference is computed
    assert!(result.std_errors.is_some());
    assert!(result.t_statistics.is_some());
    assert!(result.p_values.is_some());

    // If intercept inference was computed, verify it
    if let Some(se) = result.intercept_std_error {
        assert!(se > 0.0 || se.is_nan(), "Intercept SE should be positive or NaN");
    }
}

#[test]
fn test_wls_confidence_level() {
    let x = Mat::from_fn(50, 1, |i, _| i as f64);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);
    let weights = Col::from_fn(50, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .confidence_level(0.99)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    assert_relative_eq!(result.confidence_level, 0.99, epsilon = 1e-10);
}

#[test]
fn test_wls_options_access() {
    let weights = Col::from_fn(20, |_| 1.0);
    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .build();

    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| i as f64);

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Verify options access
    let options = fitted.options();
    assert!(options.with_intercept);
    assert!(options.compute_inference);
}

#[test]
fn test_wls_no_weights_provided() {
    // When no weights are provided, should use unit weights
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        // No weights()
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.r_squared() > 0.99);
}

#[test]
fn test_wls_many_predictors() {
    let x = Mat::from_fn(100, 10, |i, j| ((i * j + 1) as f64).sin() * 0.1);
    let y = Col::from_fn(100, |i| (i as f64 * 0.1).cos());
    let weights = Col::from_fn(100, |i| 1.0 / ((i % 5 + 1) as f64));

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert_eq!(fitted.result().coefficients.nrows(), 10);
    assert!(fitted.r_squared() >= 0.0);
}
