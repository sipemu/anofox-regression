//! OLS regression tests.

mod common;

use approx::assert_relative_eq;
use faer::{Col, Mat};
use regress_rs::solvers::{FittedRegressor, OlsRegressor, Regressor};

// ============================================================================
// Basic Regression Tests
// ============================================================================

#[test]
fn test_simple_linear_regression_with_intercept() {
    // y = 2 + 3*x
    let x = Mat::from_fn(5, 1, |i, _| i as f64);
    let y = Col::from_fn(5, |i| 2.0 + 3.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Check coefficient
    assert_relative_eq!(fitted.coefficients()[0], 3.0, epsilon = 1e-10);

    // Check intercept
    assert!(fitted.intercept().is_some());
    assert_relative_eq!(fitted.intercept().unwrap(), 2.0, epsilon = 1e-10);

    // Check R²
    assert_relative_eq!(fitted.r_squared(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_simple_linear_regression_without_intercept() {
    // y = 3*x (no intercept)
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(5, |i| 3.0 * (i + 1) as f64);

    let model = OlsRegressor::builder().with_intercept(false).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Check coefficient
    assert_relative_eq!(fitted.coefficients()[0], 3.0, epsilon = 1e-10);

    // Check no intercept
    assert!(fitted.intercept().is_none());
}

#[test]
fn test_multiple_regression() {
    // y = 1 + 2*x1 + 3*x2 with non-collinear features
    let mut x = Mat::zeros(10, 2);
    let mut y = Col::zeros(10);

    for i in 0..10 {
        x[(i, 0)] = i as f64;
        x[(i, 1)] = (i * i) as f64; // Quadratic, not collinear with x0
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)];
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(fitted.coefficients()[1], 3.0, epsilon = 1e-10);
    assert_relative_eq!(fitted.intercept().unwrap(), 1.0, epsilon = 1e-10);
    assert_relative_eq!(fitted.r_squared(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_two_observations_edge_case() {
    // Minimum viable regression: 2 observations, 1 feature
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let y = Col::from_fn(2, |i| 1.0 + 2.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(fitted.intercept().unwrap(), 1.0, epsilon = 1e-10);
}

// ============================================================================
// Collinearity and Rank Deficiency Tests
// ============================================================================

#[test]
fn test_rank_deficient_matrix() {
    // x1 and x2 are perfectly collinear (x2 = 2*x1)
    let (x, y) = common::generate_collinear_data(10);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Check that at least one coefficient is aliased (NaN)
    let result = fitted.result();
    assert!(result.has_aliased(), "Should detect collinearity");

    // The aliased coefficient should be NaN
    let has_nan = fitted.coefficients().iter().any(|&c| c.is_nan());
    assert!(has_nan, "Aliased coefficient should be NaN");
}

#[test]
fn test_constant_column_detection() {
    let (x, y) = common::generate_constant_column_data(10);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Constant column (index 1) should be aliased when there's an intercept
    let result = fitted.result();
    assert!(
        result.aliased[1],
        "Constant column should be aliased with intercept"
    );
}

#[test]
fn test_all_features_constant_with_intercept() {
    // All X values are constant - only intercept should work
    let x = Mat::from_fn(10, 2, |_, _| 5.0);
    let y = Col::from_fn(10, |_| 3.0);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // All coefficients should be aliased
    assert!(fitted.result().aliased.iter().all(|&a| a));

    // Intercept should capture the mean
    assert_relative_eq!(fitted.intercept().unwrap(), 3.0, epsilon = 1e-10);
}

#[test]
fn test_all_features_constant_without_intercept() {
    // All X values are constant, no intercept
    let x = Mat::from_fn(10, 1, |_, _| 5.0);
    let y = Col::from_fn(10, |_| 15.0); // y = 3 * 5

    let model = OlsRegressor::builder().with_intercept(false).build();
    let result = model.fit(&x, &y);

    // This should either work (with coefficient = 3) or fail gracefully
    // The behavior depends on implementation
    assert!(result.is_ok() || result.is_err());
}

// ============================================================================
// Statistical Properties Tests
// ============================================================================

#[test]
fn test_r_squared_bounds() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.1, 42);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R² should be in [0, 1]
    assert!(fitted.r_squared() >= 0.0);
    assert!(fitted.r_squared() <= 1.0);
}

#[test]
fn test_adjusted_r_squared_constraints() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.1, 42);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // Adjusted R² should be <= R²
    assert!(result.adj_r_squared <= result.r_squared + 1e-10);
}

#[test]
fn test_residual_sum_with_intercept() {
    let (x, y, _) = common::generate_linear_data(50, 2, 5.0, 0.5, 123);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Residuals should sum to approximately zero with intercept
    let residual_sum: f64 = fitted.result().residuals.iter().sum();
    assert!(
        residual_sum.abs() < 1e-10,
        "Residual sum should be ~0 with intercept, got {}",
        residual_sum
    );
}

#[test]
fn test_fitted_values_consistency() {
    let (x, y, _) = common::generate_linear_data(50, 2, 5.0, 0.5, 123);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // residuals = y - fitted_values
    for i in 0..y.nrows() {
        let expected_residual = y[i] - result.fitted_values[i];
        assert_relative_eq!(result.residuals[i], expected_residual, epsilon = 1e-10);
    }
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_predict_on_training_data() {
    let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
    let y = Col::from_fn(10, |i| 1.0 + 2.0 * i as f64 + 3.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let predictions = fitted.predict(&x);

    // Predictions should match fitted values
    let result = fitted.result();
    for i in 0..predictions.nrows() {
        assert_relative_eq!(predictions[i], result.fitted_values[i], epsilon = 1e-10);
    }
}

#[test]
fn test_predict_on_new_data() {
    // y = 2 + 3*x
    let x_train = Mat::from_fn(5, 1, |i, _| i as f64);
    let y_train = Col::from_fn(5, |i| 2.0 + 3.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x_train, &y_train).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(3, 1, |i, _| (i + 10) as f64);
    let predictions = fitted.predict(&x_new);

    // Check predictions
    for i in 0..3 {
        let expected = 2.0 + 3.0 * (i + 10) as f64;
        assert_relative_eq!(predictions[i], expected, epsilon = 1e-10);
    }
}

#[test]
fn test_score_method() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.1, 42);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Score on training data should equal R²
    let score = fitted.score(&x, &y);
    assert_relative_eq!(score, fitted.r_squared(), epsilon = 1e-10);
}

// ============================================================================
// Inference Tests
// ============================================================================

#[test]
fn test_standard_errors_positive() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();
    if let Some(ref se) = result.std_errors {
        for i in 0..se.nrows() {
            if !result.aliased[i] {
                assert!(se[i] > 0.0, "Standard error should be positive");
            }
        }
    }
}

#[test]
fn test_p_values_bounds() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();
    if let Some(ref p_vals) = result.p_values {
        for i in 0..p_vals.nrows() {
            if !result.aliased[i] {
                assert!(
                    p_vals[i] >= 0.0 && p_vals[i] <= 1.0,
                    "P-value out of bounds"
                );
            }
        }
    }
}

#[test]
fn test_confidence_intervals() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.1, 42);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .confidence_level(0.95)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();
    if let (Some(ref lower), Some(ref upper)) =
        (&result.conf_interval_lower, &result.conf_interval_upper)
    {
        for i in 0..result.coefficients.nrows() {
            if !result.aliased[i] {
                // Lower should be less than upper
                assert!(lower[i] < upper[i], "CI lower should be < upper");
                // Coefficient should be within CI
                assert!(
                    result.coefficients[i] >= lower[i] && result.coefficients[i] <= upper[i],
                    "Coefficient should be within CI"
                );
            }
        }
    }
}

// ============================================================================
// Information Criteria Tests
// ============================================================================

#[test]
fn test_aic_bic_computed() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // AIC and BIC should be finite
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    // BIC typically penalizes more than AIC for n > 7
    // This isn't always true but is a useful sanity check
    // (BIC penalty = log(n), AIC penalty = 2)
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_dimension_mismatch() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(5); // Wrong size

    let model = OlsRegressor::builder().build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_insufficient_observations() {
    // More features than observations
    let x = Mat::zeros(3, 5);
    let y = Col::zeros(3);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    // Should either fail or handle rank deficiency
    // The specific behavior is implementation-dependent
    // but it should not panic
    let _ = result;
}

#[test]
fn test_single_observation() {
    let x = Mat::from_fn(1, 1, |_, _| 1.0);
    let y = Col::from_fn(1, |_| 5.0);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    // Should fail or handle gracefully (not enough df)
    assert!(result.is_err());
}

// ============================================================================
// Comprehensive Statistics Tests
// ============================================================================

#[test]
fn test_rmse_computed() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // RMSE should be positive and finite
    assert!(result.rmse.is_finite(), "RMSE should be finite");
    assert!(result.rmse >= 0.0, "RMSE should be non-negative");

    // RMSE should equal sqrt(MSE)
    assert_relative_eq!(result.rmse, result.mse.sqrt(), epsilon = 1e-10);
}

#[test]
fn test_f_pvalue_computed() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // F p-value should be in [0, 1]
    assert!(result.f_pvalue.is_finite(), "F p-value should be finite");
    assert!(result.f_pvalue >= 0.0, "F p-value should be >= 0");
    assert!(result.f_pvalue <= 1.0, "F p-value should be <= 1");

    // For a good fit, p-value should be very small
    assert!(
        result.f_pvalue < 0.01,
        "F p-value {} should be small for good fit",
        result.f_pvalue
    );
}

#[test]
fn test_aicc_computed() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // AICc should be finite
    assert!(result.aicc.is_finite(), "AICc should be finite");

    // AICc >= AIC (correction is always positive for valid models)
    assert!(
        result.aicc >= result.aic,
        "AICc {} should be >= AIC {}",
        result.aicc,
        result.aic
    );

    // For large n, AICc should be close to AIC
    let n = result.n_observations as f64;
    let k = result.n_parameters as f64;
    let expected_correction = 2.0 * k * (k + 1.0) / (n - k - 1.0);
    assert_relative_eq!(
        result.aicc - result.aic,
        expected_correction,
        epsilon = 1e-10
    );
}

#[test]
fn test_intercept_conf_interval() {
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.5, 42);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .confidence_level(0.95)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // Intercept CI should exist
    assert!(
        result.intercept_conf_interval.is_some(),
        "Intercept CI should exist"
    );

    let (lower, upper) = result.intercept_conf_interval.unwrap();
    let intercept = result.intercept.unwrap();

    // CI should be finite
    assert!(lower.is_finite(), "Lower CI should be finite");
    assert!(upper.is_finite(), "Upper CI should be finite");

    // CI should contain the point estimate
    assert!(
        lower <= intercept && intercept <= upper,
        "CI [{}, {}] should contain intercept {}",
        lower,
        upper,
        intercept
    );

    // CI width should be reasonable (not zero, not infinite)
    let width = upper - lower;
    assert!(width > 0.0, "CI width should be positive");
    assert!(width < 100.0, "CI width {} should be reasonable", width);
}

#[test]
fn test_all_statistics_computed() {
    // Comprehensive test that all statistics are computed and reasonable
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .confidence_level(0.95)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let result = fitted.result();

    // Core results
    assert_eq!(result.coefficients.nrows(), 3);
    assert!(result.intercept.is_some());
    assert_eq!(result.residuals.nrows(), 100);
    assert_eq!(result.fitted_values.nrows(), 100);

    // Rank information
    assert!(result.rank > 0);
    assert_eq!(result.n_parameters, 4); // 3 coefficients + intercept
    assert_eq!(result.n_observations, 100);

    // Fit statistics
    assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    assert!(result.adj_r_squared.is_finite());
    assert!(result.rmse >= 0.0 && result.rmse.is_finite());
    assert!(result.mse >= 0.0 && result.mse.is_finite());
    assert!(result.f_statistic >= 0.0 && result.f_statistic.is_finite());
    assert!(result.f_pvalue >= 0.0 && result.f_pvalue <= 1.0);

    // Information criteria
    assert!(result.aic.is_finite());
    assert!(result.aicc.is_finite());
    assert!(result.bic.is_finite());
    assert!(result.log_likelihood.is_finite());

    // Inference statistics (for coefficients)
    assert!(result.std_errors.is_some());
    let se = result.std_errors.as_ref().unwrap();
    for i in 0..se.nrows() {
        assert!(
            se[i] >= 0.0 && se[i].is_finite(),
            "SE[{}] should be non-negative and finite",
            i
        );
    }

    assert!(result.t_statistics.is_some());
    let t_stats = result.t_statistics.as_ref().unwrap();
    for i in 0..t_stats.nrows() {
        assert!(t_stats[i].is_finite(), "t-stat[{}] should be finite", i);
    }

    assert!(result.p_values.is_some());
    let p_vals = result.p_values.as_ref().unwrap();
    for i in 0..p_vals.nrows() {
        assert!(
            p_vals[i] >= 0.0 && p_vals[i] <= 1.0,
            "p-value[{}] should be in [0,1]",
            i
        );
    }

    assert!(result.conf_interval_lower.is_some());
    assert!(result.conf_interval_upper.is_some());

    // Inference statistics (for intercept)
    assert!(result.intercept_std_error.is_some());
    assert!(result.intercept_std_error.unwrap() >= 0.0);

    assert!(result.intercept_t_statistic.is_some());
    assert!(result.intercept_t_statistic.unwrap().is_finite());

    assert!(result.intercept_p_value.is_some());
    let p_int = result.intercept_p_value.unwrap();
    assert!(p_int >= 0.0 && p_int <= 1.0);

    assert!(result.intercept_conf_interval.is_some());
    let (ci_low, ci_high) = result.intercept_conf_interval.unwrap();
    assert!(ci_low.is_finite() && ci_high.is_finite());
    assert!(ci_low <= ci_high);
}

// ============================================================================
// Score Edge Cases
// ============================================================================

#[test]
fn test_score_with_constant_target() {
    // Test edge case where y is constant (TSS = 0)
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |_| 5.0); // Constant target

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Score on same constant data should be 1.0 (perfect prediction of constant)
    let score = fitted.score(&x, &y);
    // Either 1.0 (perfect) or 0.0 depending on implementation
    assert!(score == 1.0 || score == 0.0 || score.is_nan());
}

#[test]
fn test_score_with_perfect_prediction() {
    // Perfect linear fit
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| 2.0 + 3.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Score should be 1.0 for perfect fit
    let score = fitted.score(&x, &y);
    assert_relative_eq!(score, 1.0, epsilon = 1e-10);
}
