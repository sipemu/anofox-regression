//! Validation Enhancement Tests for Quantile and Isotonic Regression
//!
//! This module addresses validation gaps identified during external review:
//! - Weighted quantile regression
//! - Quantile crossing behavior
//! - High-dimensional regression (p close to n)
//! - Prediction extrapolation
//! - NaN/Inf input handling
//! - Sparse X regions
//! - Isotonic tie-breaking documentation
//! - Isotonic interpolation method documentation
//! - NaN propagation
//! - All-zero weights edge cases
//! - Determinism/reproducibility

use anofox_regression::solvers::{
    FittedRegressor, IsotonicRegressor, OutOfBounds, QuantileRegressor, Regressor,
};
use faer::{Col, Mat};

// =============================================================================
// 1. WEIGHTED QUANTILE REGRESSION TESTS
// =============================================================================

#[test]
fn test_weighted_quantile_basic() {
    // Basic weighted quantile regression
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 2.0 + 1.5 * (i + 1) as f64 + 0.1 * ((i as f64).sin()));

    // Uniform weights should match unweighted
    let weights_uniform = Col::from_fn(n, |_| 1.0);

    let fitted_unweighted = QuantileRegressor::builder()
        .tau(0.5)
        .build()
        .fit(&x, &y)
        .expect("unweighted fit should succeed");

    let fitted_weighted = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights_uniform)
        .build()
        .fit(&x, &y)
        .expect("weighted fit should succeed");

    let coef_unweighted = fitted_unweighted.coefficients()[0];
    let coef_weighted = fitted_weighted.coefficients()[0];

    assert!(
        (coef_unweighted - coef_weighted).abs() < 0.01,
        "Uniform weights should match unweighted: {} vs {}",
        coef_unweighted,
        coef_weighted
    );
}

#[test]
fn test_weighted_quantile_heteroscedastic() {
    // Weighted quantile regression for heteroscedastic data
    // Common in survey data where weights represent sampling probabilities
    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) / 10.0);

    // Heteroscedastic: variance increases with x
    // y = 5 + 2*x + noise, noise ~ N(0, 0.1*x)
    let y = Col::from_fn(n, |i| {
        let xi = (i as f64) / 10.0;
        5.0 + 2.0 * xi + 0.3 * xi * ((i as f64 * 0.1).sin())
    });

    // Weights inversely proportional to variance (optimal for heteroscedasticity)
    let weights = Col::from_fn(n, |i| {
        let xi = ((i as f64) / 10.0).max(0.1);
        1.0 / xi // Higher weight for lower variance observations
    });

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y)
        .expect("heteroscedastic weighted fit should succeed");

    let slope = fitted.coefficients()[0];
    let intercept = fitted.intercept().unwrap();

    // Slope should be close to 2.0
    assert!(
        (slope - 2.0).abs() < 0.5,
        "Weighted slope {} should be near 2.0",
        slope
    );
    assert!(
        intercept.is_finite(),
        "Intercept should be finite: {}",
        intercept
    );
}

#[test]
fn test_weighted_quantile_survey_weights() {
    // Simulate survey weights (some observations represent more of the population)
    let n = 80;
    let x = Mat::from_fn(n, 1, |i, _| (i % 20) as f64);
    let y = Col::from_fn(n, |i| 1.0 + 0.5 * ((i % 20) as f64));

    // Survey weights: first 20 obs have weight 5, rest have weight 1
    let weights = Col::from_fn(n, |i| if i < 20 { 5.0 } else { 1.0 });

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y)
        .expect("survey weights fit should succeed");

    assert!(fitted.coefficients()[0].is_finite());
    assert!(fitted.intercept().unwrap().is_finite());
}

#[test]
fn test_weighted_quantile_extreme_weights() {
    // Test with extreme weight ratios (1000:1)
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 2.0 + 1.5 * (i + 1) as f64);

    let weights = Col::from_fn(n, |i| if i == 25 { 1000.0 } else { 1.0 });

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y);

    match result {
        Ok(fitted) => {
            assert!(
                fitted.coefficients()[0].is_finite(),
                "Coefficient should be finite with extreme weights"
            );
        }
        Err(_) => {
            // Acceptable to fail gracefully
        }
    }
}

// =============================================================================
// 2. QUANTILE CROSSING TESTS
// =============================================================================

#[test]
fn test_quantile_crossing_detection() {
    // Fit multiple quantiles and check if they cross
    // In theory, Q(tau1) <= Q(tau2) for tau1 < tau2 at all x
    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) / 10.0);
    let y = Col::from_fn(n, |i| {
        let xi = (i as f64) / 10.0;
        2.0 + 1.5 * xi + 0.5 * ((i as f64 * 0.3).sin())
    });

    let taus = [0.1, 0.25, 0.5, 0.75, 0.9];
    let mut intercepts = Vec::new();
    let mut slopes = Vec::new();

    for &tau in &taus {
        let fitted = QuantileRegressor::builder()
            .tau(tau)
            .build()
            .fit(&x, &y)
            .unwrap_or_else(|_| panic!("fit should succeed for tau={}", tau));

        intercepts.push(fitted.intercept().unwrap());
        slopes.push(fitted.coefficients()[0]);
    }

    // Check predictions at several x values for crossing
    let test_points = [0.0, 2.5, 5.0, 7.5, 10.0];
    let mut crossings_found = 0;

    for &x_val in &test_points {
        let mut predictions: Vec<f64> = Vec::new();
        for i in 0..taus.len() {
            predictions.push(intercepts[i] + slopes[i] * x_val);
        }

        // Check if predictions are monotonically increasing with tau
        for i in 1..predictions.len() {
            if predictions[i] < predictions[i - 1] - 1e-6 {
                crossings_found += 1;
            }
        }
    }

    // Document behavior: crossing may occur with IRLS algorithm
    // This is expected and documented - no monotonicity constraint across quantiles
    println!(
        "Quantile crossings detected at {} of {} test points (expected: algorithm does not enforce monotonicity)",
        crossings_found,
        test_points.len() * (taus.len() - 1)
    );
}

#[test]
fn test_quantile_ordering_homoscedastic() {
    // For homoscedastic data, quantile lines should be parallel (same slope)
    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    // Homoscedastic: constant variance
    let y = Col::from_fn(n, |i| 5.0 + 2.0 * (i + 1) as f64);

    let fit_25 = QuantileRegressor::builder()
        .tau(0.25)
        .build()
        .fit(&x, &y)
        .unwrap();

    let fit_75 = QuantileRegressor::builder()
        .tau(0.75)
        .build()
        .fit(&x, &y)
        .unwrap();

    let slope_25 = fit_25.coefficients()[0];
    let slope_75 = fit_75.coefficients()[0];

    // For homoscedastic data, slopes should be similar
    assert!(
        (slope_25 - slope_75).abs() < 0.5,
        "Homoscedastic data: slopes should be similar: {} vs {}",
        slope_25,
        slope_75
    );

    // Intercepts should differ (Q75 > Q25)
    let intercept_25 = fit_25.intercept().unwrap();
    let intercept_75 = fit_75.intercept().unwrap();
    // Note: for linear data without noise, intercepts might be equal
    println!(
        "Homoscedastic intercepts: Q25={}, Q75={}",
        intercept_25, intercept_75
    );
}

// =============================================================================
// 3. HIGH-DIMENSIONAL TESTS (p close to n)
// =============================================================================

#[test]
fn test_quantile_high_dimensional_p50_n200() {
    // p=50 predictors, n=200 observations
    let n = 200;
    let p = 50;

    let x = Mat::from_fn(n, p, |i, j| {
        // Generate pseudo-random features
        let seed = (i * 31 + j * 17) as f64;
        (seed * 0.1).sin() + 0.01 * (i + j) as f64
    });

    // y = sum of first 5 features + noise
    let y = Col::from_fn(n, |i| {
        let mut sum = 2.0; // intercept
        for j in 0..5 {
            sum += 0.5 * x[(i, j)];
        }
        sum + 0.1 * ((i as f64 * 0.1).sin())
    });

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .max_iterations(200)
        .build()
        .fit(&x, &y);

    match result {
        Ok(fitted) => {
            assert_eq!(fitted.coefficients().nrows(), p);
            assert!(
                fitted.coefficients().iter().all(|&c| c.is_finite()),
                "All coefficients should be finite in high-dim"
            );

            // Predictions should be finite
            let preds = fitted.predict(&x);
            assert!(
                preds.iter().all(|&p| p.is_finite()),
                "All predictions should be finite"
            );
        }
        Err(e) => {
            panic!("High-dimensional fit failed: {:?}", e);
        }
    }
}

#[test]
fn test_quantile_high_dimensional_p80_n100() {
    // More challenging: p=80 with n=100 (p close to n)
    let n = 100;
    let p = 80;

    let x = Mat::from_fn(n, p, |i, j| ((i * 13 + j * 7) as f64 * 0.01).sin());

    let y = Col::from_fn(n, |i| {
        1.0 + 0.1 * x[(i, 0)] + 0.1 * x[(i, 1)] + 0.05 * ((i as f64).sin())
    });

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .max_iterations(300)
        .tolerance(1e-4)
        .build()
        .fit(&x, &y);

    // May succeed or fail gracefully - document behavior
    match result {
        Ok(fitted) => {
            println!("p={}, n={}: Fit succeeded", p, n);
            assert!(fitted.coefficients().iter().all(|&c| c.is_finite()));
        }
        Err(e) => {
            println!("p={}, n={}: Fit failed as expected: {:?}", p, n, e);
            // This is acceptable for near-singular designs
        }
    }
}

// =============================================================================
// 4. PREDICTION EXTRAPOLATION TESTS
// =============================================================================

#[test]
fn test_quantile_extrapolation_beyond_range() {
    // Train on x in [1, 10], predict on x in [-5, 20]
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| 1.0 + (i as f64) * 9.0 / 49.0);
    let y = Col::from_fn(n, |i| 2.0 + 1.5 * x[(i, 0)]);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Extrapolate to values outside training range
    let x_extrap = Mat::from_fn(5, 1, |i, _| [-5.0, 0.0, 5.0, 15.0, 20.0][i]);
    let preds = fitted.predict(&x_extrap);

    // All predictions should be finite (linear extrapolation)
    for i in 0..5 {
        assert!(
            preds[i].is_finite(),
            "Extrapolation at x={} should be finite, got {}",
            x_extrap[(i, 0)],
            preds[i]
        );
    }

    // Verify linear extrapolation behavior
    let slope = fitted.coefficients()[0];
    let intercept = fitted.intercept().unwrap();

    for i in 0..5 {
        let expected = intercept + slope * x_extrap[(i, 0)];
        assert!(
            (preds[i] - expected).abs() < 1e-6,
            "Extrapolation should be linear: expected {}, got {}",
            expected,
            preds[i]
        );
    }
}

#[test]
fn test_isotonic_extrapolation_modes() {
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y = Col::from_fn(10, |i| (i + 1) as f64 * 2.0);

    // Test Clip mode
    let fitted_clip = IsotonicRegressor::builder()
        .out_of_bounds(OutOfBounds::Clip)
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    let x_below = Col::from_fn(1, |_| 0.0);
    let x_above = Col::from_fn(1, |_| 20.0);

    let pred_below = fitted_clip.predict_1d(&x_below);
    let pred_above = fitted_clip.predict_1d(&x_above);

    // Clip should return boundary values
    assert!(
        (pred_below[0] - 2.0).abs() < 1e-6,
        "Below range should clip to first value"
    );
    assert!(
        (pred_above[0] - 20.0).abs() < 1e-6,
        "Above range should clip to last value"
    );

    // Test Nan mode
    let fitted_nan = IsotonicRegressor::builder()
        .out_of_bounds(OutOfBounds::Nan)
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    let pred_below_nan = fitted_nan.predict_1d(&x_below);
    let pred_above_nan = fitted_nan.predict_1d(&x_above);

    assert!(pred_below_nan[0].is_nan(), "Below range should return NaN");
    assert!(pred_above_nan[0].is_nan(), "Above range should return NaN");
}

// =============================================================================
// 5. NaN/Inf INPUT HANDLING TESTS
// =============================================================================

#[test]
fn test_quantile_nan_in_y() {
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let mut y = Col::from_fn(n, |i| (i + 1) as f64);
    y[10] = f64::NAN; // Single NaN

    let result = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);

    // Document behavior: should either error or propagate NaN
    match result {
        Ok(fitted) => {
            // If it succeeds, check if NaN propagates to predictions
            let preds = fitted.predict(&x);
            let has_nan = preds.iter().any(|&p| p.is_nan());
            println!(
                "NaN in y: fit succeeded, predictions contain NaN: {}",
                has_nan
            );
        }
        Err(e) => {
            println!("NaN in y: fit failed with error: {:?}", e);
            // This is acceptable behavior
        }
    }
}

#[test]
fn test_quantile_nan_in_x() {
    let n = 20;
    let mut x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    x[(10, 0)] = f64::NAN; // Single NaN
    let y = Col::from_fn(n, |i| (i + 1) as f64);

    let result = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);

    match result {
        Ok(fitted) => {
            let coef = fitted.coefficients()[0];
            println!(
                "NaN in X: fit succeeded, coefficient is_nan: {}",
                coef.is_nan()
            );
        }
        Err(e) => {
            println!("NaN in X: fit failed with error: {:?}", e);
        }
    }
}

#[test]
fn test_quantile_inf_in_y() {
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let mut y = Col::from_fn(n, |i| (i + 1) as f64);
    y[10] = f64::INFINITY;

    let result = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);

    match result {
        Ok(fitted) => {
            let coef = fitted.coefficients()[0];
            let is_inf = coef.is_infinite();
            println!(
                "Inf in y: fit succeeded, coefficient is_infinite: {}",
                is_inf
            );
        }
        Err(e) => {
            println!("Inf in y: fit failed with error: {:?}", e);
        }
    }
}

#[test]
fn test_isotonic_nan_in_y() {
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let mut y = Col::from_fn(10, |i| (i + 1) as f64);
    y[5] = f64::NAN;

    let result = IsotonicRegressor::builder().build().fit_1d(&x, &y);

    match result {
        Ok(fitted) => {
            let has_nan = fitted.fitted_values().iter().any(|&v| v.is_nan());
            println!(
                "Isotonic NaN in y: fit succeeded, fitted values contain NaN: {}",
                has_nan
            );
        }
        Err(e) => {
            println!("Isotonic NaN in y: fit failed with error: {:?}", e);
        }
    }
}

#[test]
fn test_isotonic_inf_in_y() {
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let mut y = Col::from_fn(10, |i| (i + 1) as f64);
    y[5] = f64::INFINITY;

    let result = IsotonicRegressor::builder().build().fit_1d(&x, &y);

    match result {
        Ok(fitted) => {
            let has_inf = fitted.fitted_values().iter().any(|&v| v.is_infinite());
            println!(
                "Isotonic Inf in y: fit succeeded, fitted values contain Inf: {}",
                has_inf
            );
        }
        Err(e) => {
            println!("Isotonic Inf in y: fit failed with error: {:?}", e);
        }
    }
}

// =============================================================================
// 6. SPARSE X REGION TESTS
// =============================================================================

#[test]
fn test_quantile_sparse_x_gaps() {
    // X with gaps: clusters at extremes, nothing in middle
    let n = 40;
    let x = Mat::from_fn(n, 1, |i, _| {
        if i < 20 {
            (i as f64) * 0.1 // Cluster at [0, 2]
        } else {
            8.0 + ((i - 20) as f64) * 0.1 // Cluster at [8, 10]
        }
    });

    let y = Col::from_fn(n, |i| 2.0 + 1.5 * x[(i, 0)]);

    let fitted = QuantileRegressor::builder()
        .tau(0.5)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Predict in the gap region [3, 7]
    let x_gap = Mat::from_fn(5, 1, |i, _| 3.0 + (i as f64));
    let preds_gap = fitted.predict(&x_gap);

    // All predictions in gap should be finite (linear interpolation)
    for i in 0..5 {
        assert!(
            preds_gap[i].is_finite(),
            "Prediction in gap at x={} should be finite",
            3.0 + i as f64
        );
    }

    // Predictions should follow the linear pattern
    let slope = fitted.coefficients()[0];
    assert!(
        (slope - 1.5).abs() < 0.3,
        "Slope should be recovered despite gap: {}",
        slope
    );
}

#[test]
fn test_isotonic_sparse_x_gaps() {
    // Isotonic regression with gap in X
    let x = Col::from_fn(20, |i| {
        if i < 10 {
            i as f64 // [0, 9]
        } else {
            20.0 + ((i - 10) as f64) // [20, 29]
        }
    });

    let y = Col::from_fn(20, |i| {
        if i < 10 {
            (i as f64) + 1.0 // [1, 10]
        } else {
            15.0 + ((i - 10) as f64) // [15, 24]
        }
    });

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    // Predict in the gap [10, 19]
    let x_gap = Col::from_fn(5, |i| 12.0 + (i as f64) * 2.0);
    let preds_gap = fitted.predict_1d(&x_gap);

    // Predictions should be monotonic and use step function
    for i in 1..5 {
        assert!(
            preds_gap[i] >= preds_gap[i - 1],
            "Isotonic predictions in gap should be monotonic"
        );
    }

    // In the gap, predictions should be constant (step function behavior)
    // Based on the last value before the gap
    println!(
        "Isotonic predictions in gap: {:?}",
        preds_gap.iter().collect::<Vec<_>>()
    );
}

// =============================================================================
// 7. ISOTONIC TIE-BREAKING DOCUMENTATION TESTS
// =============================================================================

#[test]
fn test_isotonic_ties_weighted_averaging() {
    // Document that ties are handled by weighted averaging
    // x = [1, 1, 1], y = [2, 4, 6], weights = [1, 2, 3]
    // Weighted average = (1*2 + 2*4 + 3*6) / (1+2+3) = (2+8+18)/6 = 28/6 = 4.667

    let x = Col::from_fn(3, |_| 1.0);
    let y_vals = [2.0, 4.0, 6.0];
    let y = Col::from_fn(3, |i| y_vals[i]);
    let w_vals = [1.0, 2.0, 3.0];
    let weights = Col::from_fn(3, |i| w_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    let expected_avg = (1.0 * 2.0 + 2.0 * 4.0 + 3.0 * 6.0) / (1.0 + 2.0 + 3.0);
    let fv = fitted.fitted_values();

    for i in 0..3 {
        assert!(
            (fv[i] - expected_avg).abs() < 1e-10,
            "Weighted average at tie should be {}, got {}",
            expected_avg,
            fv[i]
        );
    }
}

#[test]
fn test_isotonic_ties_unweighted_averaging() {
    // Unweighted: simple average
    // x = [1, 1, 1], y = [2, 4, 6]
    // Average = (2+4+6)/3 = 4.0

    let x = Col::from_fn(3, |_| 1.0);
    let y_vals = [2.0, 4.0, 6.0];
    let y = Col::from_fn(3, |i| y_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    let expected_avg = (2.0 + 4.0 + 6.0) / 3.0;
    let fv = fitted.fitted_values();

    for i in 0..3 {
        assert!(
            (fv[i] - expected_avg).abs() < 1e-10,
            "Unweighted average at tie should be {}, got {}",
            expected_avg,
            fv[i]
        );
    }
}

// =============================================================================
// 8. ISOTONIC INTERPOLATION METHOD DOCUMENTATION TESTS
// =============================================================================

#[test]
fn test_isotonic_uses_step_function_not_linear() {
    // Document: isotonic regression uses step function interpolation,
    // NOT linear interpolation between knots

    let x = Col::from_fn(3, |i| (i + 1) as f64 * 10.0); // [10, 20, 30]
    let y = Col::from_fn(3, |i| (i + 1) as f64 * 10.0); // [10, 20, 30]

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    // Predict at x = 15 (between 10 and 20)
    let x_mid = Col::from_fn(1, |_| 15.0);
    let pred_mid = fitted.predict_1d(&x_mid);

    // Step function: should return value at lower bound (10.0)
    // If it were linear interpolation, it would be 15.0
    assert!(
        (pred_mid[0] - 10.0).abs() < 1e-6,
        "Step function at x=15 should return 10.0 (lower bound), got {}",
        pred_mid[0]
    );

    // Predict at x = 25 (between 20 and 30)
    let x_mid2 = Col::from_fn(1, |_| 25.0);
    let pred_mid2 = fitted.predict_1d(&x_mid2);

    assert!(
        (pred_mid2[0] - 20.0).abs() < 1e-6,
        "Step function at x=25 should return 20.0 (lower bound), got {}",
        pred_mid2[0]
    );
}

#[test]
fn test_isotonic_prediction_at_exact_knots() {
    // Predictions at exact knot points should return the fitted value

    let x = Col::from_fn(5, |i| (i + 1) as f64);
    let y = Col::from_fn(5, |i| (i + 1) as f64 * 2.0);

    let fitted = IsotonicRegressor::builder()
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    let preds = fitted.predict_1d(&x);
    let fv = fitted.fitted_values();

    for i in 0..5 {
        assert!(
            (preds[i] - fv[i]).abs() < 1e-10,
            "Prediction at knot x={} should equal fitted value",
            (i + 1) as f64
        );
    }
}

// =============================================================================
// 9. NaN PROPAGATION TESTS
// =============================================================================

#[test]
fn test_nan_propagation_single_nan_in_y() {
    // Document behavior when single NaN appears in y vector
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let mut y = Col::from_fn(n, |i| (i + 1) as f64);
    y[10] = f64::NAN;

    let result = QuantileRegressor::builder().tau(0.5).build().fit(&x, &y);

    // Document the actual behavior
    match result {
        Ok(fitted) => {
            let coef_is_nan = fitted.coefficients()[0].is_nan();
            let intercept_is_nan = fitted.intercept().map(|i| i.is_nan()).unwrap_or(true);
            println!(
                "Single NaN in y: coef_is_nan={}, intercept_is_nan={}",
                coef_is_nan, intercept_is_nan
            );
        }
        Err(_) => {
            println!("Single NaN in y: fit returned error (preferred behavior)");
        }
    }
}

// =============================================================================
// 10. ALL-ZERO WEIGHTS EDGE CASE TESTS
// =============================================================================

#[test]
fn test_quantile_all_zero_weights() {
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| (i + 1) as f64);
    let weights = Col::zeros(n);

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y);

    // Document behavior: all-zero weights may succeed with degenerate solution
    // or return error. Both are acceptable - just document what happens.
    match result {
        Ok(fitted) => {
            let coef = fitted.coefficients()[0];
            let intercept = fitted.intercept();
            // With all-zero weights, the OLS initialization dominates
            // and the WLS step produces zero coefficients (0/0 -> 0 via regularization)
            println!(
                "All-zero weights: coef={}, intercept={:?} (degenerate but handled)",
                coef, intercept
            );
            // Just verify it doesn't crash and produces some result
            assert!(coef.is_finite() || coef.is_nan());
        }
        Err(e) => {
            println!("All-zero weights returned error (preferred): {:?}", e);
        }
    }
}

#[test]
fn test_isotonic_all_zero_weights() {
    let x = Col::from_fn(10, |i| (i + 1) as f64);
    let y = Col::from_fn(10, |i| (i + 1) as f64);
    let weights = Col::zeros(10);

    let result = IsotonicRegressor::builder()
        .weights(weights)
        .build()
        .fit_1d(&x, &y);

    match result {
        Ok(fitted) => {
            let has_nan = fitted.fitted_values().iter().any(|&v| v.is_nan());
            println!(
                "Isotonic all-zero weights: fit succeeded, contains NaN: {}",
                has_nan
            );
        }
        Err(e) => {
            println!(
                "Isotonic all-zero weights correctly returned error: {:?}",
                e
            );
        }
    }
}

#[test]
fn test_quantile_some_zero_weights() {
    // Some zero weights (should work - just ignores those observations)
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 2.0 + 1.5 * (i + 1) as f64);
    let weights = Col::from_fn(n, |i| if i < 5 { 0.0 } else { 1.0 });

    let result = QuantileRegressor::builder()
        .tau(0.5)
        .weights(weights)
        .build()
        .fit(&x, &y);

    match result {
        Ok(fitted) => {
            assert!(
                fitted.coefficients()[0].is_finite(),
                "Some zero weights should still produce finite result"
            );
        }
        Err(e) => {
            panic!("Some zero weights should not fail: {:?}", e);
        }
    }
}

// =============================================================================
// 11. DETERMINISM/REPRODUCIBILITY TESTS
// =============================================================================

#[test]
fn test_quantile_determinism() {
    // Same input should produce bitwise identical output
    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |i| 2.0 + 1.5 * (i + 1) as f64 + 0.1 * ((i as f64).sin()));

    let mut results = Vec::new();

    for _ in 0..5 {
        let fitted = QuantileRegressor::builder()
            .tau(0.5)
            .max_iterations(100)
            .tolerance(1e-6)
            .build()
            .fit(&x, &y)
            .expect("fit should succeed");

        results.push((
            fitted.intercept().unwrap(),
            fitted.coefficients()[0],
            fitted.pseudo_r_squared(),
        ));
    }

    // All runs should produce identical results
    for i in 1..5 {
        assert!(
            (results[i].0 - results[0].0).abs() < 1e-15,
            "Intercept not deterministic: {} vs {}",
            results[i].0,
            results[0].0
        );
        assert!(
            (results[i].1 - results[0].1).abs() < 1e-15,
            "Coefficient not deterministic: {} vs {}",
            results[i].1,
            results[0].1
        );
        assert!(
            (results[i].2 - results[0].2).abs() < 1e-15,
            "R² not deterministic: {} vs {}",
            results[i].2,
            results[0].2
        );
    }
}

#[test]
fn test_isotonic_determinism() {
    // PAVA should be perfectly deterministic
    let x = Col::from_fn(30, |i| (i + 1) as f64);
    let y = Col::from_fn(30, |i| (i + 1) as f64 + 2.0 * ((i as f64 * 0.5).sin()));

    let mut results: Vec<Vec<f64>> = Vec::new();

    for _ in 0..5 {
        let fitted = IsotonicRegressor::builder()
            .increasing(true)
            .build()
            .fit_1d(&x, &y)
            .expect("fit should succeed");

        let fv: Vec<f64> = fitted.fitted_values().iter().copied().collect();
        results.push(fv);
    }

    // All runs should produce bitwise identical results
    for (i, result) in results.iter().enumerate().skip(1) {
        for (j, &val) in result.iter().enumerate() {
            assert!(
                (val - results[0][j]).abs() < 1e-15,
                "Isotonic fitted value at {} not deterministic (run {}): {} vs {}",
                j,
                i,
                val,
                results[0][j]
            );
        }
    }
}

#[test]
fn test_isotonic_determinism_with_ties() {
    // Determinism with ties (which require averaging)
    let x = Col::from_fn(20, |i| (i / 4) as f64); // Groups of 4
    let y = Col::from_fn(20, |i| (i % 4) as f64 + (i / 4) as f64);

    let mut results: Vec<Vec<f64>> = Vec::new();

    for _ in 0..5 {
        let fitted = IsotonicRegressor::builder()
            .build()
            .fit_1d(&x, &y)
            .expect("fit should succeed");

        let fv: Vec<f64> = fitted.fitted_values().iter().copied().collect();
        results.push(fv);
    }

    for (i, result) in results.iter().enumerate().skip(1) {
        for (j, &val) in result.iter().enumerate() {
            assert!(
                val == results[0][j], // Bitwise equality
                "Isotonic with ties not deterministic at {} (run {}): {} vs {}",
                j,
                i,
                val,
                results[0][j]
            );
        }
    }
}

// =============================================================================
// 12. STRICT MONOTONICITY CONSIDERATION (DOCUMENTATION)
// =============================================================================

#[test]
fn test_isotonic_non_decreasing_allows_ties() {
    // Document: current implementation is non-decreasing (y[i] <= y[i+1]),
    // NOT strictly increasing (y[i] < y[i+1])

    // Data where PAVA produces ties in output
    let x = Col::from_fn(5, |i| (i + 1) as f64);
    let y_vals = [5.0, 3.0, 4.0, 2.0, 6.0]; // Forces pooling
    let y = Col::from_fn(5, |i| y_vals[i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .build()
        .fit_1d(&x, &y)
        .expect("fit should succeed");

    let fv = fitted.fitted_values();

    // Check if there are ties (equal consecutive values)
    let mut has_ties = false;
    for i in 1..5 {
        if (fv[i] - fv[i - 1]).abs() < 1e-10 {
            has_ties = true;
            break;
        }
    }

    println!(
        "Non-decreasing isotonic has ties in output: {} (expected: true)",
        has_ties
    );

    // Verify monotonicity is non-decreasing (not strict)
    for i in 1..5 {
        assert!(
            fv[i] >= fv[i - 1] - 1e-10,
            "Should be non-decreasing: {} >= {}",
            fv[i],
            fv[i - 1]
        );
    }
}
