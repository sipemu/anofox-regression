//! DuckDB Scenario Tests
//!
//! These tests reproduce the exact scenarios from DuckDB integration failures
//! to determine whether issues originate in the Rust implementation or the
//! DuckDB integration layer.
//!
//! Run with: cargo test duckdb_scenarios -- --nocapture

mod common;

use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor, RidgeRegressor};
use faer::{Col, Mat};

// ============================================================================
// Test 1: Alternating Dummy Variables (DuckDB Test 1)
// ============================================================================
// y = [1,2,3,4,5,6], X = [[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]]
// With intercept: X becomes rank-deficient (col1 + col2 = 1 = intercept column)
// Expected: One column aliased with NaN coefficient

#[test]
fn test_alternating_dummy_basic_ols() {
    let (x, y) = common::generate_alternating_dummies(6);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    // Fit should succeed (not return error)
    assert!(result.is_ok(), "Fit should succeed, got: {:?}", result);

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 1: Alternating Dummies");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", result.aliased);

    // With intercept, one column should be aliased
    let nan_count = coefficients.iter().filter(|c| c.is_nan()).count();
    assert!(
        nan_count >= 1,
        "Expected at least one NaN coefficient due to rank deficiency, got {} NaNs",
        nan_count
    );

    // Intercept should be valid
    assert!(
        fitted.intercept().is_some(),
        "Intercept should be present"
    );
    assert!(
        !fitted.intercept().unwrap().is_nan(),
        "Intercept should not be NaN"
    );
}

// ============================================================================
// Test 2: Constant Feature (DuckDB Test 2)
// ============================================================================
// X = [[1,5],[2,5],[3,5],[4,5],[5,5]] - second column constant
// Expected: Second coefficient is NaN (aliased with intercept)

#[test]
fn test_constant_feature_aliased() {
    let mut x = Mat::zeros(5, 2);
    let mut y = Col::zeros(5);

    for i in 0..5 {
        x[(i, 0)] = (i + 1) as f64;
        x[(i, 1)] = 5.0; // Constant column
        y[i] = (i + 1) as f64;
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Fit should succeed with constant column");

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 2: Constant Feature");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", result.aliased);

    // Second column (constant) should be aliased
    assert!(
        result.aliased[1],
        "Constant column (index 1) should be aliased"
    );
    assert!(
        coefficients[1].is_nan(),
        "Constant column coefficient should be NaN"
    );

    // First column should have valid coefficient
    assert!(
        !coefficients[0].is_nan(),
        "Non-constant column coefficient should not be NaN"
    );
}

// ============================================================================
// Test 3: All-Zero Feature (DuckDB Test 3)
// ============================================================================
// X = [[1,0],[2,0],[3,0],[4,0],[5,0]] - second column all zeros
// Expected: Second coefficient is NaN (zero variance)

#[test]
fn test_all_zero_feature_aliased() {
    let mut x = Mat::zeros(5, 2);
    let mut y = Col::zeros(5);

    for i in 0..5 {
        x[(i, 0)] = (i + 1) as f64;
        x[(i, 1)] = 0.0; // All zeros
        y[i] = (i + 1) as f64;
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Fit should succeed with zero column");

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 3: All-Zero Feature");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", result.aliased);

    // Zero column should be aliased
    assert!(
        result.aliased[1],
        "Zero column (index 1) should be aliased"
    );
    assert!(
        coefficients[1].is_nan(),
        "Zero column coefficient should be NaN"
    );

    // First column should have valid coefficient ~1.0
    assert!(
        !coefficients[0].is_nan(),
        "Non-zero column coefficient should not be NaN"
    );
    assert!(
        (coefficients[0] - 1.0).abs() < 0.001,
        "First coefficient should be ~1.0, got {}",
        coefficients[0]
    );
}

// ============================================================================
// Test 4: Perfect Multicollinearity (DuckDB Test 4)
// ============================================================================
// X = [[1,2],[2,4],[3,6],[4,8],[5,10]] - x2 = 2*x1
// Expected: One coefficient is NaN (detected by QR pivoting)

#[test]
fn test_perfect_multicollinearity() {
    let mut x = Mat::zeros(5, 2);
    let mut y = Col::zeros(5);

    for i in 0..5 {
        x[(i, 0)] = (i + 1) as f64;
        x[(i, 1)] = 2.0 * (i + 1) as f64; // x2 = 2*x1
        y[i] = (i + 1) as f64;
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Fit should succeed with multicollinearity");

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 4: Perfect Multicollinearity");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", result.aliased);
    println!("  Rank: {}", result.rank);

    // Exactly one coefficient should be NaN (one column is aliased)
    let nan_count = coefficients.iter().filter(|c| c.is_nan()).count();
    assert_eq!(
        nan_count, 1,
        "Expected exactly one NaN coefficient due to multicollinearity, got {}",
        nan_count
    );

    // Should detect rank deficiency
    assert!(
        result.has_aliased(),
        "Should detect collinearity"
    );
}

// ============================================================================
// Test 5: Multiple Zero-Variance Columns (DuckDB Test 5, 9)
// ============================================================================
// 4 features, 3 are all zeros
// Expected: 3 coefficients are NaN, 1 is valid, fit succeeds (not NULL!)

#[test]
fn test_multiple_zero_variance_columns() {
    let zero_cols = vec![1, 2, 3]; // Columns 1, 2, 3 are all zeros
    let (x, y) = common::generate_zero_variance_columns(10, 4, &zero_cols);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    // CRITICAL: Fit must NOT return error (this is what fails in DuckDB as NULL)
    assert!(
        result.is_ok(),
        "Fit should succeed with multiple zero columns, got: {:?}",
        result
    );

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 5: Multiple Zero-Variance Columns");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", result.aliased);

    // Zero columns should be aliased
    for &col in &zero_cols {
        assert!(
            result.aliased[col],
            "Zero column {} should be aliased",
            col
        );
        assert!(
            coefficients[col].is_nan(),
            "Zero column {} coefficient should be NaN",
            col
        );
    }

    // First column (non-zero) should have valid coefficient
    assert!(
        !coefficients[0].is_nan(),
        "Non-zero column coefficient should not be NaN"
    );
}

// ============================================================================
// Test 6: Prediction with Aliased Coefficients (DuckDB Test 6)
// ============================================================================
// After fitting with aliased columns, predict should work
// NaN coefficients are skipped in prediction (treated as 0)

#[test]
fn test_prediction_with_aliased_coefficients() {
    // Use data with one zero column
    let mut x = Mat::zeros(5, 2);
    let mut y = Col::zeros(5);

    for i in 0..5 {
        x[(i, 0)] = (i + 1) as f64;
        x[(i, 1)] = 0.0; // All zeros
        y[i] = 2.0 + (i + 1) as f64; // y = 2 + x1
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("Fit should succeed");

    // Make predictions
    let predictions = fitted.predict(&x);

    println!("Test 6: Prediction with Aliased Coefficients");
    println!("  Predictions: {:?}", predictions.iter().collect::<Vec<_>>());

    // All predictions should be valid (not NaN)
    for (i, &pred) in predictions.iter().enumerate() {
        assert!(
            !pred.is_nan(),
            "Prediction at index {} should not be NaN, got {}",
            i,
            pred
        );
    }

    // Predictions should be close to actual y values
    for i in 0..5 {
        assert!(
            (predictions[i] - y[i]).abs() < 0.01,
            "Prediction {} should be close to {}, got {}",
            i,
            y[i],
            predictions[i]
        );
    }
}

// ============================================================================
// Test 7: Prediction with Unseen Feature Values (DuckDB Test 7)
// ============================================================================
// Train: feature 2 always 0 (aliased, coefficient=NaN)
// Test: feature 2 = 1
// Expected: Prediction works (NaN * 1 treated as 0)

#[test]
fn test_unseen_feature_values_in_prediction() {
    // Training data: second feature is always 0
    let mut x_train = Mat::zeros(4, 2);
    let mut y_train = Col::zeros(4);

    for i in 0..4 {
        x_train[(i, 0)] = (i + 1) as f64;
        x_train[(i, 1)] = 0.0;
        y_train[i] = 2.0 + (i + 1) as f64;
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x_train, &y_train).expect("Fit should succeed");

    // Test data: second feature is 1 (unseen in training)
    let mut x_test = Mat::zeros(2, 2);
    x_test[(0, 0)] = 5.0;
    x_test[(0, 1)] = 1.0; // Unseen value!
    x_test[(1, 0)] = 6.0;
    x_test[(1, 1)] = 1.0; // Unseen value!

    let predictions = fitted.predict(&x_test);

    println!("Test 7: Unseen Feature Values in Prediction");
    println!("  Coefficients: {:?}", fitted.coefficients().iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", fitted.result().aliased);
    println!("  Test predictions: {:?}", predictions.iter().collect::<Vec<_>>());

    // Predictions should NOT be NaN
    for (i, &pred) in predictions.iter().enumerate() {
        assert!(
            !pred.is_nan(),
            "Test prediction at index {} should not be NaN, got {}",
            i,
            pred
        );
    }

    // Predictions should be reasonable (y = 2 + x1, so y(5) ~ 7, y(6) ~ 8)
    assert!(
        (predictions[0] - 7.0).abs() < 0.5,
        "Prediction for x1=5 should be ~7, got {}",
        predictions[0]
    );
    assert!(
        (predictions[1] - 8.0).abs() < 0.5,
        "Prediction for x1=6 should be ~8, got {}",
        predictions[1]
    );
}

// ============================================================================
// Test 8: Ridge Regression with Multicollinearity (DuckDB Test 10)
// ============================================================================
// Ridge with x2 = 2*x1
// Expected: NO NaN coefficients (Ridge always full rank)

#[test]
fn test_ridge_multicollinearity_no_nan() {
    let mut x = Mat::zeros(5, 2);
    let mut y = Col::zeros(5);

    for i in 0..5 {
        x[(i, 0)] = (i + 1) as f64;
        x[(i, 1)] = 2.0 * (i + 1) as f64; // x2 = 2*x1
        y[i] = (i + 1) as f64;
    }

    // Ridge with small lambda
    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Ridge fit should succeed");

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();

    println!("Test 8: Ridge Multicollinearity");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());

    // NO coefficient should be NaN with Ridge regression
    let nan_count = coefficients.iter().filter(|c| c.is_nan()).count();
    assert_eq!(
        nan_count, 0,
        "Ridge should have NO NaN coefficients, got {}",
        nan_count
    );

    // Predictions should vary (not constant mean)
    let predictions = fitted.predict(&x);
    let pred_values: Vec<f64> = predictions.iter().cloned().collect();

    println!("  Predictions: {:?}", pred_values);

    // Check predictions are not all the same
    let first = pred_values[0];
    let all_same = pred_values.iter().all(|&p| (p - first).abs() < 0.001);
    assert!(
        !all_same,
        "Ridge predictions should vary, not be constant mean"
    );
}

// ============================================================================
// Test 9: Many Multicollinear Variables
// ============================================================================
// x1 is the base, x2=2*x1, x3=3*x1, x4=4*x1, etc.
// Expected: Only one coefficient is non-NaN, rest are aliased

#[test]
fn test_many_multicollinear_variables() {
    let n_samples = 10;
    let n_features = 8; // x1, x2=2*x1, x3=3*x1, ..., x8=8*x1

    let mut x = Mat::zeros(n_samples, n_features);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        let base = (i + 1) as f64;
        for j in 0..n_features {
            x[(i, j)] = base * ((j + 1) as f64); // x_j = (j+1) * base
        }
        y[i] = base * 10.0; // y = 10 * base
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Fit should succeed with many multicollinear columns");

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 9a: Many Multicollinear Variables");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", result.aliased);
    println!("  Rank: {}", result.rank);

    // Should have rank 1 (all columns are linearly dependent)
    assert_eq!(result.rank, 1, "Rank should be 1 with all multicollinear columns");

    // Only one coefficient should be non-NaN
    let non_nan_count = coefficients.iter().filter(|c| !c.is_nan()).count();
    assert_eq!(
        non_nan_count, 1,
        "Expected exactly 1 non-NaN coefficient, got {}",
        non_nan_count
    );

    // Predictions should still work
    let predictions = fitted.predict(&x);
    let nan_predictions = predictions.iter().filter(|p| p.is_nan()).count();
    assert_eq!(nan_predictions, 0, "All predictions should be valid");

    // Predictions should be close to actual y
    for i in 0..n_samples {
        assert!(
            (predictions[i] - y[i]).abs() < 0.01,
            "Prediction {} should match y {}, got {}",
            i, y[i], predictions[i]
        );
    }
}

// ============================================================================
// Test 10: Many Constant Variables (different constant values)
// ============================================================================
// Each column has a different constant value: col1=1, col2=2, col3=3, etc.
// With intercept, ALL should be aliased

#[test]
fn test_many_constant_variables() {
    let n_samples = 10;
    let n_features = 6;

    let mut x = Mat::zeros(n_samples, n_features);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            x[(i, j)] = (j + 1) as f64; // Each column is constant but different value
        }
        y[i] = 42.0; // Constant y
    }

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    assert!(result.is_ok(), "Fit should succeed with many constant columns");

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 10: Many Constant Variables");
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Coefficients: {:?}", coefficients.iter().collect::<Vec<_>>());
    println!("  Aliased: {:?}", result.aliased);

    // ALL coefficients should be aliased (constant columns with intercept)
    let aliased_count = result.aliased.iter().filter(|&&a| a).count();
    assert_eq!(
        aliased_count, n_features,
        "All {} columns should be aliased, got {}",
        n_features, aliased_count
    );

    // Intercept should capture the mean
    assert!(
        (fitted.intercept().unwrap() - 42.0).abs() < 0.01,
        "Intercept should be 42.0, got {:?}",
        fitted.intercept()
    );

    // Predictions should all be 42.0
    let predictions = fitted.predict(&x);
    for (i, &pred) in predictions.iter().enumerate() {
        assert!(
            (pred - 42.0).abs() < 0.01,
            "Prediction {} should be 42.0, got {}",
            i, pred
        );
    }
}

// ============================================================================
// Test 11: Changepoint Production Scenario (Critical Production Failure)
// ============================================================================
// 22 features: 11 month dummies + 11 changepoint segments
// 9 changepoint segments are all zeros
// Expected: Fit succeeds, 9 coefficients are NaN, predictions work

#[test]
fn test_changepoint_production_scenario() {
    let (x, y) = common::generate_changepoint_data(36); // 36 samples (3 years)

    let model = OlsRegressor::builder().with_intercept(true).build();
    let result = model.fit(&x, &y);

    // CRITICAL: This must NOT fail (production scenario)
    assert!(
        result.is_ok(),
        "Changepoint production scenario should succeed, got: {:?}",
        result
    );

    let fitted = result.unwrap();
    let coefficients = fitted.coefficients();
    let result = fitted.result();

    println!("Test 9: Changepoint Production Scenario");
    println!("  n_features: {}", coefficients.nrows());
    println!("  Intercept: {:?}", fitted.intercept());
    println!("  Aliased count: {}", result.aliased.iter().filter(|&&a| a).count());
    println!("  Rank: {}", result.rank);

    // Count NaN coefficients (should be ~9 for zero changepoint segments)
    let nan_count = coefficients.iter().filter(|c| c.is_nan()).count();
    println!("  NaN coefficient count: {}", nan_count);

    // Should have some aliased coefficients (the zero changepoint segments)
    assert!(
        nan_count >= 9,
        "Expected at least 9 NaN coefficients for zero changepoint segments, got {}",
        nan_count
    );

    // Predictions should work
    let predictions = fitted.predict(&x);
    let nan_predictions = predictions.iter().filter(|p| p.is_nan()).count();

    println!("  NaN prediction count: {}", nan_predictions);

    assert_eq!(
        nan_predictions, 0,
        "All predictions should be valid (not NaN), got {} NaN predictions",
        nan_predictions
    );
}
