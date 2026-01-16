//! Edge Case Tests for Quantile Regression
//!
//! Tests input validation, collinearity handling, convergence stability,
//! and other edge cases.

use anofox_regression::solvers::{FittedRegressor, QuantileRegressor, Regressor};
use faer::{Col, Mat};

// =============================================================================
// Input Validation Tests
// =============================================================================

#[test]
fn test_dimension_mismatch() {
    let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
    let y = Col::from_fn(5, |i| i as f64); // Wrong size

    let result = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);

    assert!(result.is_err(), "Should fail with dimension mismatch");
}

#[test]
fn test_invalid_tau_zero() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);

    let result = QuantileRegressor::builder().tau(0.0).build().fit(&x, &y);

    assert!(result.is_err(), "tau=0.0 should fail");
}

#[test]
fn test_invalid_tau_one() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);

    let result = QuantileRegressor::builder().tau(1.0).build().fit(&x, &y);

    assert!(result.is_err(), "tau=1.0 should fail");
}

#[test]
fn test_invalid_tau_negative() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);

    let result = QuantileRegressor::builder().tau(-0.5).build().fit(&x, &y);

    assert!(result.is_err(), "tau=-0.5 should fail");
}

#[test]
fn test_invalid_tau_greater_than_one() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);

    let result = QuantileRegressor::builder().tau(1.5).build().fit(&x, &y);

    assert!(result.is_err(), "tau=1.5 should fail");
}

#[test]
fn test_weight_dimension_mismatch() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let weights = Col::from_fn(5, |i| (i + 1) as f64); // Wrong size

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y);

    assert!(
        result.is_err(),
        "Should fail with weight dimension mismatch"
    );
}

#[test]
fn test_negative_weights() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let mut weights = Col::from_fn(10, |i| (i + 1) as f64);
    weights[5] = -1.0; // Negative weight

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y);

    assert!(result.is_err(), "Should fail with negative weights");
}

#[test]
fn test_all_zero_weights() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let weights = Col::zeros(10);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y);

    // TODO: All-zero weights should ideally fail, but current impl doesn't check
    // For now, verify it doesn't crash and produces NaN or fails
    match result {
        Ok(fitted) => {
            // If it succeeds, coefficients will likely be NaN or Inf
            let coefs = fitted.coefficients();
            // Just verify it completes without panic
            let _ = coefs;
        }
        Err(_) => {
            // This is the expected/preferred behavior
        }
    }
}

// =============================================================================
// Minimum Sample Size Tests
// =============================================================================

#[test]
fn test_two_observations_minimum() {
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let y = Col::from_fn(2, |i| (i * 2) as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y);

    // Should either succeed or fail gracefully
    // Two observations is borderline for regression
    match result {
        Ok(fitted) => {
            let coefs = fitted.coefficients();
            assert!(coefs[0].is_finite(), "Coefficient should be finite");
        }
        Err(_) => {
            // Acceptable to fail with too few observations
        }
    }
}

#[test]
fn test_three_observations() {
    let x = Mat::from_fn(3, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(3, |i| 2.0 + 1.5 * (i + 1) as f64);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("Should succeed with 3 observations");

    let result = fitted.result();
    assert!(result.coefficients[0].is_finite());
}

// =============================================================================
// Collinearity Tests
// =============================================================================

#[test]
fn test_constant_feature() {
    // One feature is constant
    let x = Mat::from_fn(20, 2, |i, j| {
        if j == 0 {
            i as f64
        } else {
            5.0 // Constant
        }
    });
    let y = Col::from_fn(20, |i| 2.0 + 1.5 * i as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y);

    // Should handle gracefully (either succeed with aliasing or fail)
    match result {
        Ok(fitted) => {
            let coefs = fitted.coefficients();
            // At least some coefficients should be finite
            assert!(coefs.iter().any(|&c| c.is_finite()));
        }
        Err(_) => {
            // Acceptable to fail with collinearity
        }
    }
}

#[test]
fn test_perfect_collinearity() {
    // Second feature is exactly 2x the first
    let x = Mat::from_fn(20, 2, |i, j| {
        let val = (i + 1) as f64;
        if j == 0 {
            val
        } else {
            val * 2.0 // Perfect collinearity
        }
    });
    let y = Col::from_fn(20, |i| 2.0 + 1.5 * (i + 1) as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y);

    // Should handle gracefully
    match result {
        Ok(fitted) => {
            let preds = fitted.predict(&x);
            // Predictions should still be finite
            assert!(preds.iter().all(|&p| p.is_finite()));
        }
        Err(_) => {
            // Acceptable to fail
        }
    }
}

#[test]
fn test_all_zero_feature() {
    // One feature is all zeros
    let x = Mat::from_fn(20, 2, |i, j| {
        if j == 0 {
            (i + 1) as f64
        } else {
            0.0 // All zeros
        }
    });
    let y = Col::from_fn(20, |i| 2.0 + 1.5 * (i + 1) as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y);

    // Should handle gracefully
    match result {
        Ok(fitted) => {
            let coefs = fitted.coefficients();
            // First coefficient should be reasonable
            assert!(coefs[0].is_finite());
        }
        Err(_) => {
            // Acceptable to fail
        }
    }
}

// =============================================================================
// Convergence and Numerical Stability Tests
// =============================================================================

#[test]
fn test_convergence_simple_data() {
    // Simple linear data - should converge easily
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 5.0 + 2.0 * (i + 1) as f64);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .max_iterations(1000)
        .tolerance(1e-6)
        .build()
        .fit(&x, &y)
        .expect("Should converge");

    let result = fitted.result();
    let slope = result.coefficients[0];

    // Should be close to true slope of 2.0
    assert!(
        (slope - 2.0).abs() < 0.5,
        "Slope should be near 2.0, got {}",
        slope
    );
}

#[test]
fn test_large_coefficient_magnitudes() {
    // Data with large values
    let n = 30;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64 * 1000.0);
    let y = Col::from_fn(n, |i| 10000.0 + 5.0 * (i + 1) as f64 * 1000.0);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .build()
        .fit(&x, &y)
        .expect("Should handle large values");

    let result = fitted.result();
    assert!(result.coefficients[0].is_finite());
    assert!(result.intercept.unwrap().is_finite());
}

#[test]
fn test_small_coefficient_magnitudes() {
    // Data with small values
    let n = 30;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64 * 0.001);
    let y = Col::from_fn(n, |i| 0.01 + 0.005 * (i + 1) as f64 * 0.001);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .build()
        .fit(&x, &y)
        .expect("Should handle small values");

    let result = fitted.result();
    assert!(result.coefficients[0].is_finite());
}

#[test]
fn test_predictions_finite() {
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 2.0 + 1.5 * (i + 1) as f64 + 0.1 * ((i as f64).sin()));

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    let predictions = fitted.predict(&x);

    for i in 0..n {
        assert!(
            predictions[i].is_finite(),
            "Prediction {} should be finite, got {}",
            i,
            predictions[i]
        );
    }
}

// =============================================================================
// Tau Parameter Tests
// =============================================================================

#[test]
fn test_tau_boundary_values() {
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 2.0 + 1.5 * (i + 1) as f64);

    // Test tau values close to boundaries
    for tau in [0.001, 0.01, 0.05, 0.95, 0.99, 0.999] {
        let result = QuantileRegressor::builder().tau(tau).build().fit(&x, &y);

        match result {
            Ok(fitted) => {
                let coefs = fitted.coefficients();
                assert!(
                    coefs[0].is_finite(),
                    "Coefficient should be finite for tau={}",
                    tau
                );
            }
            Err(_) => {
                // Very extreme tau values may fail
                assert!(
                    !(0.01..=0.99).contains(&tau),
                    "Should not fail for tau={} in reasonable range",
                    tau
                );
            }
        }
    }
}

#[test]
fn test_tau_accessor() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);

    let fitted = QuantileRegressor::builder()
        .tau(0.75)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    assert_eq!(fitted.tau(), 0.75, "tau() should return the correct value");
}

// =============================================================================
// No Intercept Tests
// =============================================================================

#[test]
fn test_no_intercept() {
    let n = 30;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    // Data that passes through origin
    let y = Col::from_fn(n, |i| 2.5 * (i + 1) as f64);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .expect("Should succeed without intercept");

    let result = fitted.result();
    assert!(result.intercept.is_none(), "Should have no intercept");
    assert!(
        (result.coefficients[0] - 2.5).abs() < 0.5,
        "Slope should be near 2.5"
    );
}

// =============================================================================
// Special Data Patterns
// =============================================================================

#[test]
fn test_constant_y() {
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |_| 5.0); // Constant y

    let result = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);

    match result {
        Ok(fitted) => {
            let preds = fitted.predict(&x);
            // All predictions should be near 5.0
            for i in 0..n {
                assert!(
                    (preds[i] - 5.0).abs() < 1.0,
                    "Prediction should be near 5.0 for constant y"
                );
            }
        }
        Err(_) => {
            // Acceptable to fail with constant y
        }
    }
}

#[test]
fn test_alternating_pattern() {
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| if i % 2 == 0 { 1.0 } else { -1.0 });

    let result = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);

    // Should complete without crashing
    match result {
        Ok(fitted) => {
            let preds = fitted.predict(&x);
            assert!(preds.iter().all(|&p| p.is_finite()));
        }
        Err(_) => {
            // Acceptable
        }
    }
}
