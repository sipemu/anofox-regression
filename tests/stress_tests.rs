//! Stress Tests for Quantile and Isotonic Regression
//!
//! These tests verify numerical robustness, edge cases, and API behavior
//! beyond standard R validation.

use anofox_regression::solvers::{
    FittedRegressor, IsotonicRegressor, QuantileRegressor, Regressor,
};
use faer::{Col, Mat};

// =============================================================================
// 1. Quantile Regression: Scale Invariance (The "Epsilon" Test)
// =============================================================================
//
// Goal: Verify how the IRLS algorithm handles data scale relative to its
// internal smoothing parameter (epsilon = 1e-6 by default).

#[test]
fn test_quantile_scale_invariance_baseline() {
    // Generate simple linear dataset: y = 2 + 3*x
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 2.0 + 3.0 * (i + 1) as f64);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("Baseline fit should succeed");

    let result = fitted.result();
    let baseline_intercept = result.intercept.unwrap();
    let baseline_slope = result.coefficients[0];

    // Verify baseline is reasonable
    assert!(
        (baseline_intercept - 2.0).abs() < 1.0,
        "Baseline intercept {} should be near 2.0",
        baseline_intercept
    );
    assert!(
        (baseline_slope - 3.0).abs() < 0.5,
        "Baseline slope {} should be near 3.0",
        baseline_slope
    );
}

#[test]
fn test_quantile_scale_invariance_macro() {
    // Macro-scale: y_macro = y * 1e8
    let n = 50;
    let scale = 1e8;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y_macro = Col::from_fn(n, |i| (2.0 + 3.0 * (i + 1) as f64) * scale);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y_macro)
        .expect("Macro-scale fit should succeed");

    let result = fitted.result();
    let intercept = result.intercept.unwrap();
    let slope = result.coefficients[0];

    // Coefficients should scale proportionally
    assert!(
        (intercept - 2.0 * scale).abs() / scale < 1.0,
        "Macro intercept {} should be near {} (2.0 * scale)",
        intercept,
        2.0 * scale
    );
    assert!(
        (slope - 3.0 * scale).abs() / scale < 0.5,
        "Macro slope {} should be near {} (3.0 * scale)",
        slope,
        3.0 * scale
    );
}

#[test]
fn test_quantile_scale_invariance_micro() {
    // Micro-scale: y_micro = y * 1e-8
    // Note: With epsilon=1e-6, residuals of order 1e-8 may be dominated by epsilon,
    // potentially causing degradation toward OLS behavior.
    let n = 50;
    let scale = 1e-8;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y_micro = Col::from_fn(n, |i| (2.0 + 3.0 * (i + 1) as f64) * scale);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .epsilon(1e-12) // Use smaller epsilon for micro-scale
        .build()
        .fit(&x, &y_micro)
        .expect("Micro-scale fit should succeed");

    let result = fitted.result();
    let intercept = result.intercept.unwrap();
    let slope = result.coefficients[0];

    // Coefficients should scale proportionally
    // Using relative tolerance due to small absolute values
    let expected_intercept = 2.0 * scale;
    let expected_slope = 3.0 * scale;

    assert!(
        intercept.is_finite(),
        "Micro intercept should be finite, got {}",
        intercept
    );
    assert!(
        slope.is_finite(),
        "Micro slope should be finite, got {}",
        slope
    );

    // Allow larger relative tolerance for micro-scale due to numerical precision
    let rel_tol = 0.5; // 50% relative tolerance
    assert!(
        (intercept - expected_intercept).abs() / expected_intercept.abs().max(1e-15) < rel_tol,
        "Micro intercept {} should be near {} (relative tol {})",
        intercept,
        expected_intercept,
        rel_tol
    );
    assert!(
        (slope - expected_slope).abs() / expected_slope.abs().max(1e-15) < rel_tol,
        "Micro slope {} should be near {} (relative tol {})",
        slope,
        expected_slope,
        rel_tol
    );
}

#[test]
fn test_quantile_scale_invariance_micro_default_epsilon() {
    // Test with default epsilon - may show degradation
    let n = 50;
    let scale = 1e-8;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y_micro = Col::from_fn(n, |i| (2.0 + 3.0 * (i + 1) as f64) * scale);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        // Using default epsilon = 1e-6
        .build()
        .fit(&x, &y_micro);

    // Should at least complete without panic
    match result {
        Ok(fitted) => {
            let res = fitted.result();
            // Document behavior: with default epsilon >> residual scale,
            // weights become nearly uniform, approaching OLS
            assert!(
                res.coefficients[0].is_finite(),
                "Coefficient should be finite even with epsilon dominance"
            );
        }
        Err(_) => {
            // Acceptable if it fails gracefully
        }
    }
}

// =============================================================================
// 2. Isotonic Regression: The "Sawtooth" Stress Test
// =============================================================================
//
// Goal: Stress test the PAVA block-merging and backtracking logic with
// worst-case alternating pattern requiring constant merging.

#[test]
fn test_isotonic_sawtooth_stress() {
    // Construct alternating pattern: [10, 0, 10, 0, ...]
    // This is worst-case for PAVA, requiring constant merging
    let n = 100;
    let x = Col::from_fn(n, |i| i as f64);
    let y = Col::from_fn(n, |i| if i % 2 == 0 { 10.0 } else { 0.0 });

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Sawtooth fit should succeed");

    let predictions = fitted.predict_1d(&x);

    // Result should be monotonic (non-decreasing)
    for i in 1..n {
        assert!(
            predictions[i] >= predictions[i - 1] - 1e-10,
            "Monotonicity violated at index {}: {} < {}",
            i,
            predictions[i],
            predictions[i - 1]
        );
    }

    // For pure alternating pattern with increasing constraint,
    // all values should converge to the mean (5.0)
    let mean: f64 = y.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        assert!(
            (predictions[i] - mean).abs() < 1e-6,
            "Sawtooth result at {} should be {} (mean), got {}",
            i,
            mean,
            predictions[i]
        );
    }
}

#[test]
fn test_isotonic_sawtooth_decreasing() {
    // Test decreasing constraint with sawtooth
    // Note: For decreasing constraint, the first pair (10, 0) satisfies 10 >= 0,
    // but subsequent pairs alternate violations. The result is NOT necessarily
    // a flat line at the mean, but must be monotonically non-increasing.
    let n = 100;
    let x = Col::from_fn(n, |i| i as f64);
    let y = Col::from_fn(n, |i| if i % 2 == 0 { 10.0 } else { 0.0 });

    let fitted = IsotonicRegressor::builder()
        .increasing(false) // Decreasing constraint
        .build()
        .fit_1d(&x, &y)
        .expect("Decreasing sawtooth fit should succeed");

    let predictions = fitted.predict_1d(&x);

    // Result should be monotonic (non-increasing)
    for i in 1..n {
        assert!(
            predictions[i] <= predictions[i - 1] + 1e-10,
            "Decreasing monotonicity violated at index {}: {} > {}",
            i,
            predictions[i],
            predictions[i - 1]
        );
    }

    // All predictions should be finite and within data range
    for i in 0..n {
        assert!(
            predictions[i].is_finite() && predictions[i] >= 0.0 && predictions[i] <= 10.0,
            "Prediction at {} should be in [0, 10], got {}",
            i,
            predictions[i]
        );
    }
}

#[test]
fn test_isotonic_sawtooth_large() {
    // Larger sawtooth to stress test performance
    let n = 1000;
    let x = Col::from_fn(n, |i| i as f64);
    let y = Col::from_fn(n, |i| if i % 2 == 0 { 100.0 } else { 0.0 });

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Large sawtooth fit should succeed");

    let predictions = fitted.predict_1d(&x);

    // Verify monotonicity
    for i in 1..n {
        assert!(
            predictions[i] >= predictions[i - 1] - 1e-10,
            "Large sawtooth monotonicity violated at {}",
            i
        );
    }

    // Verify convergence to mean
    let mean = 50.0;
    assert!(
        (predictions[0] - mean).abs() < 1e-6,
        "Large sawtooth should converge to mean"
    );
}

// =============================================================================
// 3. Isotonic Regression: Unsorted Input Handling
// =============================================================================
//
// Goal: Define and verify API behavior for unsorted X input.
// The API sorts internally and returns results in original order.

#[test]
fn test_isotonic_unsorted_input_simple() {
    // Unsorted input: x = [3, 1, 2], y = [30, 10, 20]
    // After sorting by x: (1, 10), (2, 20), (3, 30) -> already monotonic
    let x = Col::from_fn(3, |i| [3.0, 1.0, 2.0][i]);
    let y = Col::from_fn(3, |i| [30.0, 10.0, 20.0][i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Unsorted input should succeed");

    let predictions = fitted.predict_1d(&x);

    // Predictions should be in ORIGINAL order (matching input x order)
    // x[0]=3 -> y should be 30
    // x[1]=1 -> y should be 10
    // x[2]=2 -> y should be 20
    assert!(
        (predictions[0] - 30.0).abs() < 1e-6,
        "Prediction for x=3 should be 30, got {}",
        predictions[0]
    );
    assert!(
        (predictions[1] - 10.0).abs() < 1e-6,
        "Prediction for x=1 should be 10, got {}",
        predictions[1]
    );
    assert!(
        (predictions[2] - 20.0).abs() < 1e-6,
        "Prediction for x=2 should be 20, got {}",
        predictions[2]
    );
}

#[test]
fn test_isotonic_unsorted_matches_sorted() {
    // Compare unsorted vs pre-sorted input
    let n = 20;

    // Create random-ish unsorted indices
    let unsorted_order: Vec<usize> = vec![
        15, 3, 8, 19, 1, 12, 6, 17, 4, 10, 0, 14, 7, 2, 18, 11, 5, 16, 9, 13,
    ];

    // Sorted data: x = 0..20, y = x + noise
    let x_sorted = Col::from_fn(n, |i| i as f64);
    let y_sorted = Col::from_fn(n, |i| (i as f64) + ((i % 3) as f64 - 1.0));

    // Unsorted data (same values, different order)
    let x_unsorted = Col::from_fn(n, |i| unsorted_order[i] as f64);
    let y_unsorted = Col::from_fn(n, |i| {
        let orig_i = unsorted_order[i];
        (orig_i as f64) + ((orig_i % 3) as f64 - 1.0)
    });

    // Fit both
    let fitted_sorted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x_sorted, &y_sorted)
        .expect("Sorted fit should succeed");

    let fitted_unsorted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x_unsorted, &y_unsorted)
        .expect("Unsorted fit should succeed");

    // Get predictions at the same x values
    let test_x = Col::from_fn(5, |i| (i * 4) as f64); // [0, 4, 8, 12, 16]

    let pred_sorted = fitted_sorted.predict_1d(&test_x);
    let pred_unsorted = fitted_unsorted.predict_1d(&test_x);

    // Predictions should match
    for i in 0..5 {
        assert!(
            (pred_sorted[i] - pred_unsorted[i]).abs() < 1e-6,
            "Sorted vs unsorted prediction mismatch at x={}: {} vs {}",
            i * 4,
            pred_sorted[i],
            pred_unsorted[i]
        );
    }
}

#[test]
fn test_isotonic_unsorted_with_violations() {
    // Unsorted input where sorted result needs PAVA merging
    // x = [5, 1, 3, 2, 4], y = [1, 5, 3, 4, 2]
    // Sorted by x: (1,5), (2,4), (3,3), (4,2), (5,1) - decreasing trend
    // With increasing constraint, PAVA should merge all to mean = 3.0
    let x = Col::from_fn(5, |i| [5.0, 1.0, 3.0, 2.0, 4.0][i]);
    let y = Col::from_fn(5, |i| [1.0, 5.0, 3.0, 4.0, 2.0][i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Unsorted with violations should succeed");

    let predictions = fitted.predict_1d(&x);

    // All predictions should be the mean (3.0) due to complete violation
    let mean = 3.0;
    for i in 0..5 {
        assert!(
            (predictions[i] - mean).abs() < 1e-6,
            "Prediction at index {} should be {} (mean), got {}",
            i,
            mean,
            predictions[i]
        );
    }
}

// =============================================================================
// 4. Quantile Regression: Singular Matrix Safety
// =============================================================================
//
// Goal: Ensure the linear algebra solver handles zero-variance features
// gracefully without panicking or infinite loops.

#[test]
fn test_quantile_singular_zero_variance_feature() {
    // Dataset with intercept + valid feature + zero-variance feature (all 0s)
    // The zero column is collinear with... nothing directly, but causes rank deficiency
    let n = 20;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i + 1) as f64 // Valid feature
        } else {
            0.0 // Zero-variance feature
        }
    });
    let y = Col::from_fn(n, |i| 2.0 + 3.0 * (i + 1) as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y);

    // Should either succeed (ignoring zero column) or return error
    // Must NOT panic or loop infinitely
    match result {
        Ok(fitted) => {
            let res = fitted.result();
            // First coefficient (valid feature) should be reasonable
            assert!(
                res.coefficients[0].is_finite(),
                "Valid coefficient should be finite"
            );
            // Zero-variance coefficient might be 0, NaN, or any finite value
            // The key is it didn't crash
        }
        Err(_) => {
            // Acceptable: graceful error for rank deficiency
        }
    }
}

#[test]
fn test_quantile_singular_collinear_with_intercept() {
    // Feature that is constant (all 1s) - collinear with intercept
    let n = 20;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i + 1) as f64 // Valid feature
        } else {
            1.0 // Constant = collinear with intercept
        }
    });
    let y = Col::from_fn(n, |i| 2.0 + 3.0 * (i + 1) as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y);

    // Should handle gracefully
    match result {
        Ok(fitted) => {
            let res = fitted.result();
            // Valid feature coefficient should still be estimable
            assert!(
                res.coefficients[0].is_finite(),
                "Valid feature coefficient should be finite"
            );
        }
        Err(_) => {
            // Acceptable: error for collinearity
        }
    }
}

#[test]
fn test_quantile_singular_perfect_collinearity() {
    // x2 = 2 * x1 (perfect collinearity)
    let n = 20;
    let x = Mat::from_fn(n, 2, |i, j| {
        let val = (i + 1) as f64;
        if j == 0 {
            val
        } else {
            val * 2.0 // Perfect collinearity
        }
    });
    let y = Col::from_fn(n, |i| 2.0 + 3.0 * (i + 1) as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y);

    // Must not panic
    match result {
        Ok(fitted) => {
            // Predictions should still be reasonable
            let preds = fitted.predict(&x);
            assert!(
                preds.iter().all(|&p| p.is_finite()),
                "Predictions should be finite"
            );
        }
        Err(_) => {
            // Acceptable: error for singularity
        }
    }
}

#[test]
fn test_quantile_near_singular() {
    // Nearly singular: x2 = x1 + small noise
    let n = 50;
    let x = Mat::from_fn(n, 2, |i, j| {
        let val = (i + 1) as f64;
        if j == 0 {
            val
        } else {
            val + 1e-10 * (i as f64).sin() // Near-collinear
        }
    });
    let y = Col::from_fn(n, |i| 2.0 + 3.0 * (i + 1) as f64);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .max_iterations(200)
        .build()
        .fit(&x, &y);

    // Should handle gracefully (may have large coefficients but shouldn't crash)
    match result {
        Ok(fitted) => {
            let preds = fitted.predict(&x);
            // At minimum, predictions should be finite
            let all_finite = preds.iter().all(|&p| p.is_finite());
            assert!(all_finite, "Predictions should be finite for near-singular");
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

// =============================================================================
// 5. Generics Check: f32 Compilation
// =============================================================================
//
// Note: The anofox-regression crate currently uses f64 exclusively.
// This section documents this limitation and tests that the API works
// correctly with f64 data that could conceptually be f32.

#[test]
fn test_f64_precision_sufficient_for_f32_range() {
    // Test that f64 implementation handles f32-range values correctly
    // f32 range: ~1e-38 to ~1e38, precision: ~7 decimal digits

    let n = 30;
    // Use values in f32 range with f32-like precision
    let x = Mat::from_fn(n, 1, |i, _| {
        let val = (i + 1) as f32;
        val as f64 // Convert to f64 for API
    });
    let y = Col::from_fn(n, |i| {
        let val = (2.5_f32 + 1.5_f32 * (i + 1) as f32) as f64;
        val
    });

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("f32-range values should work");

    let result = fitted.result();
    let intercept = result.intercept.unwrap();
    let slope = result.coefficients[0];

    // Use f32-appropriate tolerance (about 1e-3 relative)
    let tol = 1e-2;
    assert!(
        (intercept - 2.5).abs() < tol * 10.0,
        "Intercept {} should be near 2.5 with f32 tolerance",
        intercept
    );
    assert!(
        (slope - 1.5).abs() < tol,
        "Slope {} should be near 1.5 with f32 tolerance",
        slope
    );
}

#[test]
fn test_isotonic_f32_range_values() {
    // Test isotonic regression with f32-range values
    let n = 20;
    let x = Col::from_fn(n, |i| (i as f32) as f64);
    let y = Col::from_fn(n, |i| {
        // Values that would be typical f32
        ((i as f32) * 1.5_f32 + ((i % 3) as f32 - 1.0_f32)) as f64
    });

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("f32-range isotonic should work");

    let predictions = fitted.predict_1d(&x);

    // Verify monotonicity
    for i in 1..n {
        assert!(
            predictions[i] >= predictions[i - 1] - 1e-5,
            "Monotonicity should hold for f32-range values"
        );
    }
}

#[test]
fn test_documentation_f64_only() {
    // This test documents that the crate is f64-only
    // If generics are added in the future, this test should be updated

    // Verify that the API signature requires f64
    let x: Mat<f64> = Mat::from_fn(10, 1, |i, _| i as f64);
    let y: Col<f64> = Col::from_fn(10, |i| i as f64);

    // These compile because the API uses f64
    let _ = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);
    let _ = IsotonicRegressor::builder().build().fit_1d(
        &Col::from_fn(10, |i| i as f64),
        &Col::from_fn(10, |i| i as f64),
    );

    // Note: If you need f32 support, values must be converted to f64 first
    // let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
    // let x_f64: Col<f64> = Col::from_fn(3, |i| x_f32[i] as f64);
}

// =============================================================================
// Additional Stress Tests
// =============================================================================

#[test]
fn test_quantile_convergence_difficult_data() {
    // Data that may be difficult for IRLS to converge
    // Heavy outliers in one direction
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let mut y_vals: Vec<f64> = (0..n).map(|i| 2.0 + 1.0 * (i + 1) as f64).collect();

    // Add extreme outliers
    y_vals[45] = 1000.0;
    y_vals[46] = 1000.0;
    y_vals[47] = 1000.0;
    y_vals[48] = -500.0;
    y_vals[49] = -500.0;

    let y = Col::from_fn(n, |i| y_vals[i]);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .with_intercept(true)
        .max_iterations(500)
        .build()
        .fit(&x, &y);

    // Should converge (median is robust to outliers)
    match result {
        Ok(fitted) => {
            let res = fitted.result();
            // Median regression should largely ignore outliers
            // Slope should still be approximately 1.0
            assert!(
                res.coefficients[0].is_finite(),
                "Coefficient should be finite"
            );
            assert!(
                (res.coefficients[0] - 1.0).abs() < 2.0,
                "Slope {} should be near 1.0 despite outliers",
                res.coefficients[0]
            );
        }
        Err(e) => {
            panic!("Should converge for median regression: {:?}", e);
        }
    }
}

#[test]
fn test_isotonic_many_ties_stress() {
    // Stress test with many ties at each x value
    let n_unique = 10;
    let reps = 50;
    let n = n_unique * reps;

    let x = Col::from_fn(n, |i| (i / reps) as f64);
    let y = Col::from_fn(n, |i| {
        let base = (i / reps) as f64;
        // Add variation within each tie group
        base * 2.0 + ((i % reps) as f64 - (reps / 2) as f64) * 0.1
    });

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("Many ties should succeed");

    let predictions = fitted.predict_1d(&x);

    // Verify monotonicity
    let mut prev_x = f64::NEG_INFINITY;
    let mut prev_pred = f64::NEG_INFINITY;

    for i in 0..n {
        if x[i] > prev_x + 1e-10 {
            // New x value - prediction should be >= previous
            assert!(
                predictions[i] >= prev_pred - 1e-10,
                "Monotonicity violated between x={} and x={}",
                prev_x,
                x[i]
            );
            prev_x = x[i];
            prev_pred = predictions[i];
        }
    }
}
