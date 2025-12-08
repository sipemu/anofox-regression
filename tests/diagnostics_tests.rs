//! Diagnostics integration tests.

mod common;

use approx::assert_relative_eq;
use faer::{Col, Mat};
use statistics::diagnostics::{
    compute_leverage, cooks_distance, high_leverage_points, high_vif_predictors, influential_cooks,
    standardized_residuals, studentized_residuals, variance_inflation_factor,
};
use statistics::solvers::{FittedRegressor, OlsRegressor, Regressor};

// ============================================================================
// Leverage Tests
// ============================================================================

#[test]
fn test_leverage_with_ols() {
    let x = Mat::from_fn(30, 2, |i, j| {
        if j == 0 {
            i as f64
        } else {
            (i as f64 * 0.5).sin()
        }
    });
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let leverage = compute_leverage(&x, true);
    let high = high_leverage_points(&leverage, fitted.result().n_parameters, None);

    // All leverage values should be in [0, 1]
    for i in 0..leverage.nrows() {
        assert!(leverage[i] >= 0.0 && leverage[i] <= 1.0);
    }

    // High leverage points list should be valid
    for &idx in &high {
        assert!(idx < 30);
    }
}

// ============================================================================
// Residual Diagnostics Tests
// ============================================================================

#[test]
fn test_residual_diagnostics_with_ols() {
    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y = Col::zeros(50);
    for i in 0..50 {
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)] + (i as f64 * 0.01).sin();
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let leverage = compute_leverage(&x, true);
    let residuals = &fitted.result().residuals;
    let mse = fitted.result().mse;

    let std_resid = standardized_residuals(residuals, mse);
    let stud_resid = studentized_residuals(residuals, &leverage, mse);

    // Standardized residuals should have reasonable values
    for i in 0..std_resid.nrows() {
        assert!(
            std_resid[i].abs() < 10.0,
            "Standardized residual {} too large",
            i
        );
    }

    // Studentized residuals should have reasonable values
    for i in 0..stud_resid.nrows() {
        assert!(
            stud_resid[i].abs() < 10.0,
            "Studentized residual {} too large",
            i
        );
    }
}

// ============================================================================
// Influence Diagnostics Tests
// ============================================================================

#[test]
fn test_cooks_distance_with_ols() {
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let leverage = compute_leverage(&x, true);
    let residuals = &fitted.result().residuals;
    let mse = fitted.result().mse;
    let n_params = fitted.result().n_parameters;

    let cooks = cooks_distance(residuals, &leverage, mse, n_params);

    // All Cook's distances should be non-negative
    for i in 0..cooks.nrows() {
        assert!(
            cooks[i] >= 0.0 || cooks[i].is_nan(),
            "Cook's distance[{}] = {} should be >= 0",
            i,
            cooks[i]
        );
    }

    // Get influential points
    let influential = influential_cooks(&cooks, None);
    for &idx in &influential {
        assert!(idx < 30);
    }
}

#[test]
fn test_influential_point_detection() {
    // Create data with one outlier
    let mut x = Mat::from_fn(30, 1, |i, _| i as f64);
    let mut y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

    // Add an outlier with high leverage
    x[(29, 0)] = 100.0; // Extreme x value
    y[29] = 300.0; // Doesn't follow pattern

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let leverage = compute_leverage(&x, true);
    let residuals = &fitted.result().residuals;
    let mse = fitted.result().mse;
    let n_params = fitted.result().n_parameters;

    let cooks = cooks_distance(residuals, &leverage, mse, n_params);

    // Point 29 should have high Cook's distance
    let max_cooks_idx = (0..30)
        .filter(|&i| cooks[i].is_finite())
        .max_by(|&a, &b| cooks[a].partial_cmp(&cooks[b]).unwrap())
        .unwrap();

    assert_eq!(
        max_cooks_idx, 29,
        "Point 29 should have highest Cook's distance"
    );
}

// ============================================================================
// VIF Tests
// ============================================================================

#[test]
fn test_vif_with_independent_predictors() {
    // Create predictors that are approximately orthogonal
    let x = Mat::from_fn(100, 2, |i, j| {
        if j == 0 {
            (i as f64 * 0.1).sin()
        } else {
            (i as f64 * 0.1).cos()
        }
    });

    let vif = variance_inflation_factor(&x);

    // VIF should be close to 1 for orthogonal predictors
    for j in 0..vif.nrows() {
        assert!(
            vif[j] < 2.0,
            "VIF[{}] = {} should be < 2 for independent predictors",
            j,
            vif[j]
        );
    }
}

#[test]
fn test_vif_detects_collinearity() {
    // Create predictors with high collinearity
    let x = Mat::from_fn(100, 2, |i, j| {
        if j == 0 {
            i as f64
        } else {
            i as f64 * 1.01 + 0.1 // Almost perfectly correlated
        }
    });

    let vif = variance_inflation_factor(&x);
    let high = high_vif_predictors(&vif, 5.0);

    // At least one predictor should have high VIF
    assert!(
        !high.is_empty(),
        "Collinear predictors should have high VIF: {:?}",
        vif.iter().collect::<Vec<_>>()
    );
}

#[test]
fn test_vif_with_multiple_predictors() {
    let x = Mat::from_fn(100, 3, |i, j| match j {
        0 => i as f64,
        1 => (i as f64 * 0.5).sin(),
        2 => i as f64 * 2.0 + (i as f64 * 0.3).cos(), // Somewhat correlated with x0
        _ => 0.0,
    });

    let vif = variance_inflation_factor(&x);

    // All VIF values should be >= 1
    for j in 0..vif.nrows() {
        assert!(vif[j] >= 1.0, "VIF[{}] = {} should be >= 1", j, vif[j]);
    }
}

// ============================================================================
// Full Workflow Test
// ============================================================================

#[test]
fn test_full_diagnostic_workflow() {
    // Generate data
    let x = Mat::from_fn(50, 2, |i, j| {
        if j == 0 {
            i as f64
        } else {
            (i as f64 * 0.2).sin() * 10.0
        }
    });
    let y = Col::from_fn(50, |i| {
        5.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)] + (i as f64 * 0.1).cos()
    });

    // Fit model
    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Run diagnostics
    let leverage = compute_leverage(&x, true);
    let residuals = &fitted.result().residuals;
    let mse = fitted.result().mse;
    let n_params = fitted.result().n_parameters;

    let std_resid = standardized_residuals(residuals, mse);
    let stud_resid = studentized_residuals(residuals, &leverage, mse);
    let cooks = cooks_distance(residuals, &leverage, mse, n_params);
    let vif = variance_inflation_factor(&x);

    // Verify all diagnostics are computed
    assert_eq!(leverage.nrows(), 50);
    assert_eq!(std_resid.nrows(), 50);
    assert_eq!(stud_resid.nrows(), 50);
    assert_eq!(cooks.nrows(), 50);
    assert_eq!(vif.nrows(), 2);

    // Check basic properties
    let leverage_sum: f64 = leverage.iter().sum();
    assert!(
        (leverage_sum - n_params as f64).abs() < 1.0,
        "Sum of leverage {} should be close to n_params {}",
        leverage_sum,
        n_params
    );

    // VIF should be >= 1
    for j in 0..vif.nrows() {
        assert!(vif[j] >= 1.0);
    }
}
