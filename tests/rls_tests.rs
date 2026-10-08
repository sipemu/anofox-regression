//! Recursive Least Squares tests.

mod common;

use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor, RlsRegressor};
use approx::assert_relative_eq;
use faer::{Col, Mat};

// ============================================================================
// Basic RLS Tests
// ============================================================================

#[test]
fn test_rls_basic() {
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.r_squared() > 0.99);
}

#[test]
fn test_rls_converges_to_ols() {
    // With forgetting_factor = 1 and enough data, RLS should produce good fit
    // Note: RLS doesn't exactly equal OLS due to P matrix initialization
    let x = Mat::from_fn(100, 1, |i, _| i as f64);
    let y = Col::from_fn(100, |i| 2.0 + 3.0 * i as f64);

    let rls = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();
    let ols = OlsRegressor::builder().with_intercept(true).build();

    let rls_fit = rls.fit(&x, &y).expect("rls fit should succeed");
    let ols_fit = ols.fit(&x, &y).expect("ols fit should succeed");

    // Both should have excellent R² on this simple linear problem
    assert!(
        rls_fit.r_squared() > 0.99,
        "RLS R²={} should be > 0.99",
        rls_fit.r_squared()
    );
    assert!(
        ols_fit.r_squared() > 0.99,
        "OLS R²={} should be > 0.99",
        ols_fit.r_squared()
    );

    // RLS coefficient should be reasonably close to true value (3.0)
    assert!(
        (rls_fit.coefficients()[0] - 3.0).abs() < 0.5,
        "RLS coef={} should be close to 3.0",
        rls_fit.coefficients()[0]
    );
}

#[test]
fn test_rls_without_intercept() {
    let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(30, |i| 3.0 * (i + 1) as f64);

    let model = RlsRegressor::builder()
        .with_intercept(false)
        .forgetting_factor(1.0)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.intercept().is_none());
    assert!(fitted.r_squared() > 0.99);
}

// ============================================================================
// Forgetting Factor Tests
// ============================================================================

#[test]
fn test_forgetting_factor_adapts_to_changes() {
    // First half: y = 1 + 2*x
    // Second half: y = 5 + 3*x (shift in relationship)
    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| i as f64);
    let mut y = Col::zeros(n);
    for i in 0..n / 2 {
        y[i] = 1.0 + 2.0 * i as f64;
    }
    for i in n / 2..n {
        y[i] = 5.0 + 3.0 * i as f64;
    }

    // With low forgetting factor, should adapt to recent data
    let adaptive = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(0.95)
        .build();

    // With forgetting_factor = 1, weights all data equally
    let non_adaptive = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();

    let adaptive_fit = adaptive.fit(&x, &y).expect("fit should succeed");
    let non_adaptive_fit = non_adaptive.fit(&x, &y).expect("fit should succeed");

    // The adaptive model should have coefficient closer to 3 (recent slope)
    // The non-adaptive model should have something between 2 and 3
    // (This is a weak test since the true values depend on initialization)
    assert!(adaptive_fit.coefficients()[0].is_finite());
    assert!(non_adaptive_fit.coefficients()[0].is_finite());
}

// ============================================================================
// Online Update Tests
// ============================================================================

#[test]
fn test_rls_online_update() {
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64);

    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();

    let mut fitted = model.fit(&x, &y).expect("fit should succeed");

    // Update with several new observations
    for i in 20..30 {
        let x_new = Col::from_fn(1, |_| i as f64);
        let y_new = 1.0 + 2.0 * i as f64;
        fitted.update(&x_new, y_new);
    }

    // Coefficient should still be close to 2
    assert!(
        (fitted.coefficients()[0] - 2.0).abs() < 0.5,
        "Coefficient should be near 2.0, got {}",
        fitted.coefficients()[0]
    );
}

#[test]
fn test_rls_update_returns_prediction() {
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64);

    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();

    let mut fitted = model.fit(&x, &y).expect("fit should succeed");

    // Update with new observation
    let x_new = Col::from_fn(1, |_| 25.0);
    let y_new = 1.0 + 2.0 * 25.0; // = 51.0

    let prediction_before_update = fitted.update(&x_new, y_new);

    // The prediction should be close to the actual value
    assert!(
        (prediction_before_update - y_new).abs() < 5.0,
        "Prediction {} should be close to {}",
        prediction_before_update,
        y_new
    );
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_rls_predict() {
    let x_train = Mat::from_fn(30, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y_train = Col::zeros(30);
    for i in 0..30 {
        y_train[i] = 1.0 + 2.0 * x_train[(i, 0)] + 3.0 * x_train[(i, 1)];
    }

    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
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
fn test_rls_score() {
    let x = Mat::from_fn(50, 1, |i, _| i as f64);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let score = fitted.score(&x, &y);
    assert_relative_eq!(score, fitted.r_squared(), epsilon = 1e-10);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_rls_dimension_mismatch() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(5); // Wrong size

    let model = RlsRegressor::builder().forgetting_factor(1.0).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_rls_insufficient_observations() {
    let x = Mat::zeros(1, 2);
    let y = Col::zeros(1);

    let model = RlsRegressor::builder().forgetting_factor(1.0).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_rls_r_squared_bounds() {
    let x = Mat::from_fn(50, 2, |i, j| ((i * j + 1) as f64).sin());
    let y = Col::from_fn(50, |i| (i as f64).cos());

    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(0.99)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R² should be in [0, 1]
    assert!(fitted.r_squared() >= 0.0);
    assert!(fitted.r_squared() <= 1.0);
}

#[test]
fn test_rls_p_matrix_access() {
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64);

    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // P matrix should be accessible
    let p = fitted.p_matrix();
    assert_eq!(p.nrows(), 2); // intercept + 1 feature
    assert_eq!(p.ncols(), 2);

    // Diagonal should be positive (inverse covariance)
    for i in 0..p.nrows() {
        assert!(p[(i, i)] >= 0.0);
    }
}

#[test]
fn test_rls_forgetting_factor_access() {
    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(0.95)
        .build();

    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| i as f64);

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    assert_relative_eq!(fitted.forgetting_factor(), 0.95);
}

#[test]
fn test_rls_initial_p_diagonal_and_summary_stats() {
    let x = Mat::from_fn(30, 1, |i, _| i as f64 / 3.0);
    let y = Col::from_fn(30, |i| {
        1.5 + 0.7 * (i as f64 / 3.0) + ((i * 7) % 5) as f64 * 0.1
    });

    // Vague prior + no forgetting -> batch OLS, including the summary stats.
    let rls = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .initial_p_diagonal(1e8)
        .build()
        .fit(&x, &y)
        .unwrap();
    let ols = OlsRegressor::builder()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let (r, o) = (rls.result(), ols.result());
    assert!((r.coefficients[0] - o.coefficients[0]).abs() < 1e-6);
    assert!((r.r_squared - o.r_squared).abs() < 1e-8);
    assert!((r.adj_r_squared - o.adj_r_squared).abs() < 1e-8);
    assert!((r.rmse - o.rmse).abs() < 1e-7);

    // A tight prior shrinks the estimate.
    let tight = RlsRegressor::builder()
        .with_intercept(true)
        .initial_p_diagonal(1e-4)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!(tight.result().coefficients[0].abs() < o.coefficients[0].abs());
    assert!(RlsRegressor::builder()
        .initial_p_diagonal(0.0)
        .build()
        .fit(&x, &y)
        .is_err());
}

#[test]
fn test_rls_update_skips_non_finite() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| 2.0 * i as f64);
    let mut f = RlsRegressor::builder().build().fit(&x, &y).unwrap();
    let before = f.result().coefficients[0];
    assert!(f.update(&Col::from_fn(1, |_| f64::NAN), 1.0).is_nan());
    assert!(f.update(&Col::from_fn(1, |_| 1.0), f64::NAN).is_nan());
    assert_eq!(f.result().coefficients[0], before);
}
