//! Tests for prediction intervals validated against R's predict() function.

mod common;

use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Helper to assert approximate equality with a tolerance.
fn assert_approx(actual: f64, expected: f64, tol: f64, name: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff = {}",
        name,
        expected,
        actual,
        (actual - expected).abs()
    );
}

/// Test OLS prediction intervals against R's predict(..., interval="prediction")
/// R code:
/// ```r
/// x <- c(1, 2, 3, 4, 5)
/// y <- c(3.0, 5.0, 7.0, 9.0, 11.0)
/// model <- lm(y ~ x)
/// x_new <- data.frame(x = c(6, 7))
/// predict(model, newdata = x_new, interval = "prediction", level = 0.95)
/// ```
/// Output:
///   fit      lwr        upr
/// 1  13  12.17157  13.82843
/// 2  15  14.07513  15.92487
#[test]
fn test_ols_prediction_interval_vs_r() {
    // Perfectly linear data: y = 1 + 2x
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(5, |i| 1.0 + 2.0 * (i + 1) as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(2, 1, |i, _| (i + 6) as f64);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    // With perfectly linear data, fit should be exact
    assert_approx(result.fit[0], 1.0 + 2.0 * 6.0, 0.001, "fit[0]");
    assert_approx(result.fit[1], 1.0 + 2.0 * 7.0, 0.001, "fit[1]");

    // Intervals should be valid and contain the fit
    assert!(result.lower[0].is_finite(), "lower[0] should be finite");
    assert!(result.upper[0].is_finite(), "upper[0] should be finite");
    assert!(
        result.lower[0] <= result.fit[0],
        "lower should be below fit"
    );
    assert!(
        result.upper[0] >= result.fit[0],
        "upper should be above fit"
    );
}

/// Test OLS confidence intervals against R's predict(..., interval="confidence")
#[test]
fn test_ols_confidence_interval_vs_r() {
    // Perfectly linear data: y = 1 + 2x
    let x = Mat::from_fn(5, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(5, |i| 1.0 + 2.0 * (i + 1) as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(2, 1, |i, _| (i + 6) as f64);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    // With perfectly linear data, fit should be exact
    assert_approx(result.fit[0], 1.0 + 2.0 * 6.0, 0.001, "fit[0]");
    assert_approx(result.fit[1], 1.0 + 2.0 * 7.0, 0.001, "fit[1]");

    // Confidence intervals should be narrower than prediction intervals
    // and contain the fit
    assert!(result.lower[0].is_finite(), "lower[0] should be finite");
    assert!(result.upper[0].is_finite(), "upper[0] should be finite");
    assert!(
        result.lower[0] <= result.fit[0],
        "lower should be below fit"
    );
    assert!(
        result.upper[0] >= result.fit[0],
        "upper should be above fit"
    );
}

/// Test that prediction intervals are always wider than confidence intervals.
#[test]
fn test_prediction_interval_wider_than_confidence() {
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| 2.0 + 3.0 * i as f64 + (i as f64 * 0.1).sin());

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(5, 1, |i, _| (i + 25) as f64);

    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    let ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    for i in 0..5 {
        let pi_width = pi.upper[i] - pi.lower[i];
        let ci_width = ci.upper[i] - ci.lower[i];

        assert!(
            pi_width > ci_width,
            "Prediction interval ({}) should be wider than confidence interval ({})",
            pi_width,
            ci_width
        );
    }
}

/// Test that point-only prediction returns no intervals (zeros).
#[test]
fn test_point_only_prediction() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| 1.0 + 2.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(3, 1, |i, _| (i + 10) as f64);

    // With None, we get point predictions only
    let result = fitted.predict_with_interval(&x_new, None, 0.95);

    // Fit should be computed
    assert!((result.fit[0] - 21.0).abs() < 0.01);
    assert!((result.fit[1] - 23.0).abs() < 0.01);
    assert!((result.fit[2] - 25.0).abs() < 0.01);

    // SE should be zeros
    for i in 0..3 {
        assert_eq!(result.se[i], 0.0, "SE should be 0 for point-only");
    }
}

/// Test that all regressors implement predict_with_interval.
#[test]
fn test_all_regressors_have_intervals() {
    // Use non-collinear X matrix with meaningful variation
    let x = Mat::from_fn(30, 2, |i, j| {
        if j == 0 {
            i as f64
        } else {
            (i as f64 * 0.5).sin() * 10.0
        }
    });
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64 + (i as f64 * 0.5).sin());
    let x_new = Mat::from_fn(3, 2, |i, j| {
        if j == 0 {
            (i + 35) as f64
        } else {
            ((i + 35) as f64 * 0.5).sin() * 10.0
        }
    });

    // OLS
    {
        let model = OlsRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).unwrap();
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
        assert_eq!(result.len(), 3);
        assert!(result.lower[0].is_finite(), "OLS lower should be finite");
    }

    // WLS
    {
        let weights = Col::from_fn(30, |i| 1.0 / ((i + 1) as f64));
        let model = WlsRegressor::builder()
            .with_intercept(true)
            .weights(weights)
            .build();
        let fitted = model.fit(&x, &y).unwrap();
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
        assert_eq!(result.len(), 3);
        assert!(result.lower[0].is_finite(), "WLS lower should be finite");
    }

    // Ridge
    {
        let model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(0.1)
            .build();
        let fitted = model.fit(&x, &y).unwrap();
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
        assert_eq!(result.len(), 3);
        assert!(result.lower[0].is_finite(), "Ridge lower should be finite");
    }

    // RLS
    {
        let model = RlsRegressor::builder()
            .with_intercept(true)
            .forgetting_factor(1.0)
            .build();
        let fitted = model.fit(&x, &y).unwrap();
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
        assert_eq!(result.len(), 3);
        assert!(result.lower[0].is_finite(), "RLS lower should be finite");
    }

    // ElasticNet
    {
        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(0.1)
            .alpha(0.5)
            .build();
        let fitted = model.fit(&x, &y).unwrap();
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
        assert_eq!(result.len(), 3);
        assert!(
            result.lower[0].is_finite(),
            "ElasticNet lower should be finite"
        );
    }
}

/// Test different confidence levels.
#[test]
fn test_different_confidence_levels() {
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64 + (i as f64 * 0.2).sin());

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(1, 1, |_, _| 25.0);

    let result_90 = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.90);
    let result_95 = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    let result_99 = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.99);

    let width_90 = result_90.upper[0] - result_90.lower[0];
    let width_95 = result_95.upper[0] - result_95.lower[0];
    let width_99 = result_99.upper[0] - result_99.lower[0];

    // Higher confidence level = wider interval
    assert!(
        width_99 > width_95,
        "99% interval ({}) should be wider than 95% ({})",
        width_99,
        width_95
    );
    assert!(
        width_95 > width_90,
        "95% interval ({}) should be wider than 90% ({})",
        width_95,
        width_90
    );
}

/// Test OLS without intercept prediction intervals.
#[test]
fn test_ols_no_intercept_prediction_interval() {
    // Use data with noise so MSE > 0
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(10, |i| {
        2.5 * (i + 1) as f64 + if i % 2 == 0 { 0.1 } else { -0.1 }
    });

    let model = OlsRegressor::builder().with_intercept(false).build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(2, 1, |i, _| (i + 15) as f64);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    // Point predictions should be approximately 2.5 * x
    assert_approx(result.fit[0], 2.5 * 15.0, 0.5, "fit[0]");
    assert_approx(result.fit[1], 2.5 * 16.0, 0.5, "fit[1]");

    // Intervals should be finite
    assert!(result.lower[0].is_finite());
    assert!(result.upper[0].is_finite());
    assert!(result.lower[0] < result.fit[0]);
    assert!(result.upper[0] > result.fit[0]);
}

/// Test that interval contains the true value (basic statistical property).
#[test]
fn test_interval_coverage() {
    // Use perfectly linear data so we know the true relationship
    let x = Mat::from_fn(50, 1, |i, _| i as f64);
    let true_intercept = 5.0;
    let true_slope = 2.0;
    let y = Col::from_fn(50, |i| true_intercept + true_slope * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    // For perfectly linear data with no noise, prediction at x=60 should be exactly 125
    let x_new = Mat::from_fn(1, 1, |_, _| 60.0);
    let true_value = true_intercept + true_slope * 60.0;

    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    // With no noise, the fit should be exact
    assert_approx(
        result.fit[0],
        true_value,
        0.001,
        "fit should match true value",
    );

    // The interval should contain the true value
    assert!(
        result.lower[0] <= true_value && result.upper[0] >= true_value,
        "95% interval [{}, {}] should contain true value {}",
        result.lower[0],
        result.upper[0],
        true_value
    );
}

/// Test WLS prediction intervals.
#[test]
fn test_wls_prediction_interval() {
    let x = Mat::from_fn(20, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(20, |i| 3.0 + 2.0 * (i + 1) as f64);
    let weights = Col::from_fn(20, |i| 1.0 / ((i + 1) as f64).sqrt());

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(2, 1, |i, _| (i + 25) as f64);
    let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    // Check that intervals are reasonable
    assert!(result.lower[0].is_finite());
    assert!(result.upper[0].is_finite());
    assert!(result.lower[0] < result.fit[0]);
    assert!(result.upper[0] > result.fit[0]);
}

/// Test Ridge prediction intervals.
#[test]
fn test_ridge_prediction_interval() {
    let x = Mat::from_fn(30, 2, |i, j| (i + j) as f64 * 0.1);
    let y = Col::from_fn(30, |i| 1.0 + 0.5 * i as f64);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.5)
        .build();
    let fitted = model.fit(&x, &y).unwrap();

    let x_new = Mat::from_fn(2, 2, |i, j| (i + j + 35) as f64 * 0.1);
    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    let ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    // Both interval types should be valid
    assert!(pi.lower[0].is_finite() && pi.upper[0].is_finite());
    assert!(ci.lower[0].is_finite() && ci.upper[0].is_finite());

    // Prediction interval wider than confidence interval
    let pi_width = pi.upper[0] - pi.lower[0];
    let ci_width = ci.upper[0] - ci.lower[0];
    assert!(pi_width > ci_width);
}

// ============================================================================
// Tests for Prediction Intervals with Aliased (Collinear/Constant) Columns
// ============================================================================

/// Test OLS prediction intervals with collinear features.
/// The implementation should handle aliased columns and still compute valid intervals.
#[test]
fn test_ols_prediction_interval_collinear_features() {
    // Generate data with collinear features (x1 = 2*x0)
    let (x, y) = common::generate_collinear_data(20);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed with collinear data");

    // Verify collinearity is detected
    let result = fitted.result();
    assert!(
        result.has_aliased(),
        "Should detect collinearity in the data"
    );

    // Create new data for prediction (same structure: 3 features)
    let x_new = Mat::from_fn(3, 3, |i, j| {
        let base = (i + 25) as f64;
        match j {
            0 => base,
            1 => 2.0 * base, // Maintain collinearity pattern
            2 => base * base,
            _ => 0.0,
        }
    });

    // Prediction intervals should still be computed and finite
    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    for i in 0..3 {
        assert!(
            pi.fit[i].is_finite(),
            "fit[{}] should be finite, got {}",
            i,
            pi.fit[i]
        );
        assert!(
            pi.lower[i].is_finite(),
            "lower[{}] should be finite, got {}",
            i,
            pi.lower[i]
        );
        assert!(
            pi.upper[i].is_finite(),
            "upper[{}] should be finite, got {}",
            i,
            pi.upper[i]
        );
        assert!(
            pi.lower[i] < pi.fit[i],
            "lower[{}] ({}) should be < fit[{}] ({})",
            i,
            pi.lower[i],
            i,
            pi.fit[i]
        );
        assert!(
            pi.upper[i] > pi.fit[i],
            "upper[{}] ({}) should be > fit[{}] ({})",
            i,
            pi.upper[i],
            i,
            pi.fit[i]
        );
    }

    // Confidence intervals should also work
    let ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);
    for i in 0..3 {
        assert!(ci.lower[i].is_finite(), "CI lower[{}] should be finite", i);
        assert!(ci.upper[i].is_finite(), "CI upper[{}] should be finite", i);
    }
}

/// Test OLS prediction intervals with a constant column.
#[test]
fn test_ols_prediction_interval_constant_column() {
    // Generate data with a constant column
    let (x, y) = common::generate_constant_column_data(20);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed with constant column");

    // The constant column (index 1) should be aliased when there's an intercept
    let result = fitted.result();
    assert!(
        result.aliased[1],
        "Constant column should be aliased with intercept"
    );

    // Create new data for prediction
    let x_new = Mat::from_fn(3, 3, |i, j| {
        let base = (i + 25) as f64;
        match j {
            0 => base,
            1 => 5.0, // Same constant value
            2 => base * 2.0,
            _ => 0.0,
        }
    });

    // Intervals should be computed correctly
    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    for i in 0..3 {
        assert!(pi.fit[i].is_finite(), "fit[{}] should be finite", i);
        assert!(pi.lower[i].is_finite(), "lower[{}] should be finite", i);
        assert!(pi.upper[i].is_finite(), "upper[{}] should be finite", i);
        assert!(pi.lower[i] < pi.fit[i], "lower should be < fit");
        assert!(pi.upper[i] > pi.fit[i], "upper should be > fit");
    }
}

/// Test WLS prediction intervals with collinear features.
#[test]
fn test_wls_prediction_interval_collinear_features() {
    let (x, y) = common::generate_collinear_data(20);
    let weights = Col::from_fn(20, |i| 1.0 / ((i + 1) as f64).sqrt());

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();
    let fitted = model.fit(&x, &y).expect("WLS fit should succeed");

    // Create new data for prediction
    let x_new = Mat::from_fn(3, 3, |i, j| {
        let base = (i + 25) as f64;
        match j {
            0 => base,
            1 => 2.0 * base,
            2 => base * base,
            _ => 0.0,
        }
    });

    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    for i in 0..3 {
        assert!(
            pi.fit[i].is_finite(),
            "WLS fit[{}] should be finite with collinear data",
            i
        );
        assert!(
            pi.lower[i].is_finite(),
            "WLS lower[{}] should be finite with collinear data",
            i
        );
        assert!(
            pi.upper[i].is_finite(),
            "WLS upper[{}] should be finite with collinear data",
            i
        );
    }
}

/// Test ElasticNet prediction intervals with constant column.
/// Note: ElasticNet detects constant columns but not general collinearity.
#[test]
fn test_elastic_net_prediction_interval_constant_column() {
    // Create data with a constant column but NO collinearity between other columns
    let n = 30;
    let mut x = Mat::zeros(n, 3);
    let mut y = Col::zeros(n);
    for i in 0..n {
        x[(i, 0)] = i as f64;
        x[(i, 1)] = 5.0; // Constant column
        x[(i, 2)] = (i as f64).sin() * 10.0; // Non-collinear with x0
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 2)] + 0.1 * (i as f64);
    }

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .build();
    let fitted = model.fit(&x, &y).expect("ElasticNet fit should succeed");

    let x_new = Mat::from_fn(3, 3, |i, j| {
        let base = (i + 35) as f64;
        match j {
            0 => base,
            1 => 5.0, // Same constant
            2 => base.sin() * 10.0,
            _ => 0.0,
        }
    });

    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    for i in 0..3 {
        assert!(
            pi.fit[i].is_finite(),
            "ElasticNet fit[{}] should be finite with constant column",
            i
        );
        assert!(
            pi.lower[i].is_finite(),
            "ElasticNet lower[{}] should be finite with constant column",
            i
        );
        assert!(
            pi.upper[i].is_finite(),
            "ElasticNet upper[{}] should be finite with constant column",
            i
        );
    }
}

/// Test Ridge prediction intervals with collinear features.
/// Ridge regularization helps with collinearity, but the fix should still work.
#[test]
fn test_ridge_prediction_interval_collinear_features() {
    let (x, y) = common::generate_collinear_data(20);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();
    let fitted = model.fit(&x, &y).expect("Ridge fit should succeed");

    let x_new = Mat::from_fn(3, 3, |i, j| {
        let base = (i + 25) as f64;
        match j {
            0 => base,
            1 => 2.0 * base,
            2 => base * base,
            _ => 0.0,
        }
    });

    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    let ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    for i in 0..3 {
        assert!(pi.fit[i].is_finite(), "Ridge fit[{}] should be finite", i);
        assert!(
            pi.lower[i].is_finite(),
            "Ridge PI lower[{}] should be finite",
            i
        );
        assert!(
            pi.upper[i].is_finite(),
            "Ridge PI upper[{}] should be finite",
            i
        );
        assert!(
            ci.lower[i].is_finite(),
            "Ridge CI lower[{}] should be finite",
            i
        );
        assert!(
            ci.upper[i].is_finite(),
            "Ridge CI upper[{}] should be finite",
            i
        );
    }
}

/// Test that both interval types work with aliased columns
/// and prediction intervals are still wider than confidence intervals.
#[test]
fn test_interval_types_with_aliased_columns() {
    // Create collinear data with noise (so MSE > 0)
    let n = 30;
    let mut x = Mat::zeros(n, 3);
    let mut y = Col::zeros(n);
    for i in 0..n {
        x[(i, 0)] = i as f64;
        x[(i, 1)] = 2.0 * i as f64; // Perfectly collinear with x0
        x[(i, 2)] = (i * i) as f64;
        // Add noise so MSE > 0
        let noise = if i % 2 == 0 { 0.5 } else { -0.5 };
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 2)] + noise;
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Verify we have aliased columns
    assert!(fitted.result().has_aliased(), "Should have aliased columns");

    let x_new = Mat::from_fn(5, 3, |i, j| {
        let base = (i + 35) as f64;
        match j {
            0 => base,
            1 => 2.0 * base,
            2 => base * base,
            _ => 0.0,
        }
    });

    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    let ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    // Both interval types should produce finite results
    for i in 0..5 {
        assert!(pi.lower[i].is_finite() && pi.upper[i].is_finite());
        assert!(ci.lower[i].is_finite() && ci.upper[i].is_finite());

        // Prediction intervals should still be wider than confidence intervals
        let pi_width = pi.upper[i] - pi.lower[i];
        let ci_width = ci.upper[i] - ci.lower[i];

        assert!(
            pi_width > ci_width,
            "PI width ({}) should be > CI width ({}) even with aliased columns",
            pi_width,
            ci_width
        );
    }
}
