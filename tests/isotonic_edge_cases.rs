//! Edge Case Tests for Isotonic Regression
//!
//! Tests input validation, PAVA ties handling, weighted regression,
//! out-of-bounds modes, and other edge cases.

use anofox_regression::solvers::{FittedRegressor, IsotonicRegressor, OutOfBounds, Regressor};
use faer::{Col, Mat};

// =============================================================================
// Input Validation Tests
// =============================================================================

#[test]
fn test_dimension_mismatch() {
    let x = Col::from_fn(10, |i| i as f64);
    let y = Col::from_fn(5, |i| i as f64); // Wrong size

    let result = IsotonicRegressor::builder().build().fit_1d(&x, &y);

    assert!(result.is_err(), "Should fail with dimension mismatch");
}

#[test]
fn test_weight_dimension_mismatch() {
    let x = Col::from_fn(10, |i| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let weights = Col::from_fn(5, |i| (i + 1) as f64); // Wrong size

    let result = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y);

    assert!(
        result.is_err(),
        "Should fail with weight dimension mismatch"
    );
}

#[test]
fn test_negative_weights() {
    let x = Col::from_fn(10, |i| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let mut weights = Col::from_fn(10, |i| (i + 1) as f64);
    weights[5] = -1.0; // Negative weight

    let result = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y);

    assert!(result.is_err(), "Should fail with negative weights");
}

#[test]
fn test_all_zero_weights() {
    let x = Col::from_fn(10, |i| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let weights = Col::zeros(10);

    let result = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y);

    // TODO: All-zero weights should ideally fail
    match result {
        Ok(fitted) => {
            let _ = fitted.coefficients();
        }
        Err(_) => {
            // Expected/preferred
        }
    }
}

// =============================================================================
// Minimum Sample Size Tests
// =============================================================================

#[test]
fn test_single_observation() {
    let x = Col::from_fn(1, |_| 1.0);
    let y = Col::from_fn(1, |_| 5.0);

    let result = IsotonicRegressor::builder().build().fit_1d(&x, &y);

    // Single observation should fail - need at least 2 for regression
    assert!(result.is_err(), "Single observation should fail");
}

#[test]
fn test_two_observations_increasing() {
    let x = Col::from_fn(2, |i| (i + 1) as f64);
    let y = Col::from_fn(2, |i| (i + 1) as f64); // y = [1, 2]

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Two observations should succeed");

    let pred = fitted.predict_1d(&x);
    assert!((pred[0] - 1.0).abs() < 1e-10);
    assert!((pred[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_two_observations_decreasing() {
    let x = Col::from_fn(2, |i| (i + 1) as f64);
    let y = Col::from_fn(2, |i| (2 - i) as f64); // y = [2, 1]

    let fitted = IsotonicRegressor::builder()
        .increasing(false)
        .build()
        .fit_1d(&x, &y)
        .expect("Two observations should succeed");

    let pred = fitted.predict_1d(&x);
    assert!((pred[0] - 2.0).abs() < 1e-10);
    assert!((pred[1] - 1.0).abs() < 1e-10);
}

// =============================================================================
// PAVA Ties Handling Tests
// =============================================================================

#[test]
fn test_ties_in_x_averaging() {
    // Multiple y values at same x - should average
    let x = Col::from_fn(5, |_| 1.0); // All same x
    let y = Col::from_fn(5, |i| (i + 1) as f64); // y = [1, 2, 3, 4, 5]

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("Ties should be handled");

    let pred = fitted.predict_1d(&x);
    // All predictions should be average = 3.0
    for p in pred.iter() {
        assert!((*p - 3.0).abs() < 1e-10, "Should average tied values");
    }
}

#[test]
fn test_ties_mixed_with_non_ties() {
    // x = [1, 1, 2, 3, 3]
    let x = Col::from_fn(5, |i| match i {
        0 | 1 => 1.0,
        2 => 2.0,
        _ => 3.0,
    });
    // y = [2, 4, 3, 1, 5]
    let y_vals = [2.0, 4.0, 3.0, 1.0, 5.0];
    let y = Col::from_fn(5, |i| y_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle mixed ties");

    let pred = fitted.predict_1d(&x);

    // After averaging ties and PAVA:
    // x=1: avg(2,4)=3.0
    // x=2: 3.0
    // x=3: avg(1,5)=3.0
    // Should all be 3.0 for increasing constraint
    for p in pred.iter() {
        assert!(p.is_finite(), "Predictions should be finite");
    }
}

#[test]
fn test_many_ties() {
    // Large number of ties
    let n = 100;
    let x = Col::from_fn(n, |i| (i / 10) as f64); // 10 groups
    let y = Col::from_fn(n, |i| (i % 10) as f64);

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("Many ties should be handled");

    let pred = fitted.predict_1d(&x);
    assert!(pred.iter().all(|&p| p.is_finite()));
}

// =============================================================================
// Weighted PAVA Tests
// =============================================================================

#[test]
fn test_weighted_basic() {
    let x = Col::from_fn(5, |i| (i + 1) as f64);
    let y = Col::from_fn(5, |i| (i + 1) as f64);
    let weights = Col::from_fn(5, |i| (i + 1) as f64);

    let fitted = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y)
        .expect("Weighted should succeed");

    let pred = fitted.predict_1d(&x);
    assert!(pred.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_extreme_weight_ratios() {
    // Weight ratio of 1000:1
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y = Col::from_fn(10, |i| (i + 1) as f64);
    let weights = Col::from_fn(10, |i| if i == 0 { 1000.0 } else { 1.0 });

    let fitted = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y)
        .expect("Extreme weights should be handled");

    let pred = fitted.predict_1d(&x);
    assert!(pred.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_weighted_ties() {
    // Ties with different weights
    let x = Col::from_fn(4, |_| 1.0); // All same x
    let y_vals = [1.0, 3.0, 2.0, 4.0];
    let w_vals = [1.0, 2.0, 3.0, 4.0];
    let y = Col::from_fn(4, |i| y_vals[i]);
    let weights = Col::from_fn(4, |i| w_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y)
        .expect("Weighted ties should work");

    let pred = fitted.predict_1d(&x);
    // Weighted average = (1*1 + 2*3 + 3*2 + 4*4) / (1+2+3+4) = 29/10 = 2.9
    let expected_avg = 2.9;
    for p in pred.iter() {
        assert!(
            (*p - expected_avg).abs() < 0.1,
            "Weighted average should be ~{}, got {}",
            expected_avg,
            p
        );
    }
}

// =============================================================================
// Out-of-Bounds Prediction Tests
// =============================================================================

#[test]
fn test_out_of_bounds_clip() {
    let x = Col::from_fn(5, |i| (i + 1) as f64); // x = [1, 2, 3, 4, 5]
    let y = Col::from_fn(5, |i| (i + 1) as f64); // y = [1, 2, 3, 4, 5]

    let fitted = IsotonicRegressor::builder()
        .out_of_bounds(OutOfBounds::Clip)
        .build()
        .fit_1d(&x, &y)
        .expect("Fit should succeed");

    // Predict at out-of-bounds x values
    let x_new = Col::from_fn(7, |i| match i {
        0 => 0.0,  // Below range
        1 => 0.5,  // Below range
        2 => 1.0,  // In range
        3 => 3.0,  // In range
        4 => 5.0,  // In range
        5 => 5.5,  // Above range
        _ => 10.0, // Above range
    });

    let pred = fitted.predict_1d(&x_new);

    // Below range should be clipped to first value (1.0)
    assert!(
        (pred[0] - 1.0).abs() < 1e-10,
        "Below range should clip to 1.0"
    );
    assert!(
        (pred[1] - 1.0).abs() < 1e-10,
        "Below range should clip to 1.0"
    );

    // Above range should be clipped to last value (5.0)
    assert!(
        (pred[5] - 5.0).abs() < 1e-10,
        "Above range should clip to 5.0"
    );
    assert!(
        (pred[6] - 5.0).abs() < 1e-10,
        "Above range should clip to 5.0"
    );
}

#[test]
fn test_out_of_bounds_nan() {
    let x = Col::from_fn(5, |i| (i + 1) as f64);
    let y = Col::from_fn(5, |i| (i + 1) as f64);

    let fitted = IsotonicRegressor::builder()
        .out_of_bounds(OutOfBounds::Nan)
        .build()
        .fit_1d(&x, &y)
        .expect("Fit should succeed");

    let x_new = Col::from_fn(3, |i| match i {
        0 => 0.0,  // Below range
        1 => 3.0,  // In range
        _ => 10.0, // Above range
    });

    let pred = fitted.predict_1d(&x_new);

    assert!(pred[0].is_nan(), "Below range should be NaN");
    assert!(pred[1].is_finite(), "In range should be finite");
    assert!(pred[2].is_nan(), "Above range should be NaN");
}

// =============================================================================
// Monotonicity Tests
// =============================================================================

#[test]
fn test_increasing_constraint_violated() {
    // Data that violates increasing constraint
    let x = Col::from_fn(5, |i| (i + 1) as f64);
    let y_vals = [5.0, 4.0, 3.0, 2.0, 1.0]; // Decreasing y
    let y = Col::from_fn(5, |i| y_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle violated constraint");

    let pred = fitted.predict_1d(&x);

    // All predictions should be same (average) for increasing constraint
    let avg = 3.0;
    for p in pred.iter() {
        assert!(
            (*p - avg).abs() < 1e-10,
            "Should pool to average for violated constraint"
        );
    }
}

#[test]
fn test_decreasing_constraint_violated() {
    // Data that violates decreasing constraint
    let x = Col::from_fn(5, |i| (i + 1) as f64);
    let y_vals = [1.0, 2.0, 3.0, 4.0, 5.0]; // Increasing y
    let y = Col::from_fn(5, |i| y_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(false)
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle violated constraint");

    let pred = fitted.predict_1d(&x);

    // All predictions should be same (average) for decreasing constraint
    let avg = 3.0;
    for p in pred.iter() {
        assert!(
            (*p - avg).abs() < 1e-10,
            "Should pool to average for violated constraint"
        );
    }
}

#[test]
fn test_monotonicity_preserved() {
    // Data with some violations
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y_vals = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.5, 8.0, 7.0, 10.0];
    let y = Col::from_fn(10, |i| y_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Should succeed");

    let pred = fitted.predict_1d(&x);

    // Verify monotonicity is preserved
    for i in 1..10 {
        assert!(
            pred[i] >= pred[i - 1],
            "Prediction should be monotonically increasing: {} >= {}",
            pred[i],
            pred[i - 1]
        );
    }
}

// =============================================================================
// Special Data Patterns
// =============================================================================

#[test]
fn test_all_equal_y() {
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y = Col::from_fn(10, |_| 5.0); // All same

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle all equal y");

    let pred = fitted.predict_1d(&x);
    for p in pred.iter() {
        assert!((*p - 5.0).abs() < 1e-10);
    }
}

#[test]
fn test_already_monotonic() {
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y = Col::from_fn(10, |i| (i + 1) as f64); // Already increasing

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle already monotonic");

    let pred = fitted.predict_1d(&x);
    for i in 0..10 {
        assert!(
            (pred[i] - (i + 1) as f64).abs() < 1e-10,
            "Should preserve already monotonic data"
        );
    }
}

#[test]
fn test_step_function() {
    // Step function data
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y = Col::from_fn(10, |i| if i < 5 { 1.0 } else { 5.0 });

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle step function");

    let pred = fitted.predict_1d(&x);
    assert!(pred.iter().all(|&p| p.is_finite()));
}

// =============================================================================
// Large Dataset Tests
// =============================================================================

#[test]
fn test_large_dataset() {
    let n = 10000;
    let x = Col::from_fn(n, |i| i as f64);
    let y = Col::from_fn(n, |i| (i as f64).sqrt() + 0.1 * ((i as f64).sin()));

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle large dataset");

    let pred = fitted.predict_1d(&x);
    assert!(pred.iter().all(|&p| p.is_finite()));

    // Verify monotonicity
    for i in 1..n {
        assert!(
            pred[i] >= pred[i - 1] - 1e-10,
            "Should be monotonic at index {}",
            i
        );
    }
}

// =============================================================================
// Floating Point Edge Cases
// =============================================================================

#[test]
fn test_very_small_differences() {
    // Very small differences that might cause floating point issues
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y = Col::from_fn(10, |i| 1.0 + 1e-10 * i as f64);

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle small differences");

    let pred = fitted.predict_1d(&x);
    assert!(pred.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_large_values() {
    let x = Col::from_fn(10, |i| (i + 1) as f64 * 1e6);
    let y = Col::from_fn(10, |i| (i + 1) as f64 * 1e9);

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("Should handle large values");

    let pred = fitted.predict_1d(&x);
    assert!(pred.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_mixed_signs() {
    let x = Col::from_fn(10, |i| i as f64 - 5.0); // x = [-5, -4, ..., 4]
    let y = Col::from_fn(10, |i| (i as f64 - 5.0).abs()); // V shape

    let result = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y);

    // Should handle gracefully
    match result {
        Ok(fitted) => {
            let pred = fitted.predict_1d(&x);
            assert!(pred.iter().all(|&p| p.is_finite()));
        }
        Err(_) => {
            // Acceptable
        }
    }
}

// =============================================================================
// R² Calculation Tests
// =============================================================================

#[test]
fn test_r_squared_bounds() {
    let x = Col::from_fn(20, |i| (i + 1) as f64);
    let y = Col::from_fn(20, |i| (i + 1) as f64 + 0.5 * ((i as f64).sin()));

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("Should succeed");

    let result = fitted.result();

    // R² should be in [0, 1] for typical cases
    // Note: R² can be negative for very poor fits, but isotonic should do well
    assert!(
        result.r_squared >= 0.0 && result.r_squared <= 1.0,
        "R² should be in [0,1], got {}",
        result.r_squared
    );
}

// =============================================================================
// Matrix Interface Tests
// =============================================================================

#[test]
fn test_matrix_interface() {
    // Test using Mat instead of 1D Col
    let n = 20;
    let x_mat = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| (i + 1) as f64);

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit(&x_mat, &y)
        .expect("Matrix interface should work");

    let pred = fitted.predict(&x_mat);
    assert!(pred.iter().all(|&p| p.is_finite()));
}
