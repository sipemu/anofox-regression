//! Integration tests for solver selection API.
//!
//! Tests that OLS, Ridge, and WLS produce consistent results across
//! QR, SVD, and Cholesky solvers.

use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Helper: create well-conditioned test data (y = 2 + 3*x1 + 0.5*x2 + noise).
fn well_conditioned_data() -> (Mat<f64>, Col<f64>) {
    let n = 50;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            i as f64 / 10.0
        } else {
            (i as f64 * 0.7 + 3.0).sin() * 5.0
        }
    });
    let y = Col::from_fn(n, |i| {
        2.0 + 3.0 * x[(i, 0)] + 0.5 * x[(i, 1)] + (i as f64 * 0.1).sin() * 0.01
    });
    (x, y)
}

/// Helper: check coefficients match within tolerance.
fn coefficients_match(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(ai, bi)| {
        if ai.is_nan() && bi.is_nan() {
            true
        } else {
            (ai - bi).abs() < tol
        }
    })
}

// ==============================
// OLS solver selection tests
// ==============================

#[test]
fn test_ols_qr_svd_match() {
    let (x, y) = well_conditioned_data();

    let qr_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
    let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&qr_coefs, &svd_coefs, 1e-8),
        "OLS QR and SVD coefficients differ:\n  QR:  {:?}\n  SVD: {:?}",
        qr_coefs,
        svd_coefs
    );

    assert!(
        (qr_fit.r_squared() - svd_fit.r_squared()).abs() < 1e-10,
        "R² values differ: QR={}, SVD={}",
        qr_fit.r_squared(),
        svd_fit.r_squared()
    );
}

#[test]
fn test_ols_qr_cholesky_match() {
    let (x, y) = well_conditioned_data();

    let qr_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
    let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&qr_coefs, &chol_coefs, 1e-8),
        "OLS QR and Cholesky coefficients differ:\n  QR:   {:?}\n  Chol: {:?}",
        qr_coefs,
        chol_coefs
    );
}

#[test]
fn test_ols_svd_cholesky_match() {
    let (x, y) = well_conditioned_data();

    let svd_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();
    let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&svd_coefs, &chol_coefs, 1e-8),
        "OLS SVD and Cholesky coefficients differ"
    );
}

#[test]
fn test_ols_no_intercept_all_solvers() {
    let (x, y) = well_conditioned_data();

    let qr_fit = OlsRegressor::builder()
        .with_intercept(false)
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_fit = OlsRegressor::builder()
        .with_intercept(false)
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = OlsRegressor::builder()
        .with_intercept(false)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
    let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();
    let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&qr_coefs, &svd_coefs, 1e-8),
        "No-intercept OLS: QR vs SVD differ"
    );
    assert!(
        coefficients_match(&qr_coefs, &chol_coefs, 1e-8),
        "No-intercept OLS: QR vs Cholesky differ"
    );
}

#[test]
fn test_ols_cholesky_fallback_on_rank_deficient() {
    // Create rank-deficient data: col 2 = col 0
    let n = 20;
    let x = Mat::from_fn(n, 3, |i, j| {
        if j == 0 || j == 2 {
            i as f64
        } else {
            (i as f64 * 0.3).sin()
        }
    });
    let y = Col::from_fn(n, |i| 1.0 + 2.0 * i as f64);

    // Cholesky should fall back to QR for rank-deficient data
    let chol_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y);

    // Should succeed (falls back to QR)
    assert!(chol_fit.is_ok(), "Cholesky fallback should succeed");

    let qr_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    // Both should produce valid R²
    let chol_result = chol_fit.unwrap();
    assert!(chol_result.r_squared() > 0.9);
    assert!(qr_fit.r_squared() > 0.9);
}

#[test]
fn test_ols_svd_rank_deficient() {
    // Rank-deficient: col 2 = col 0
    let n = 20;
    let x = Mat::from_fn(n, 3, |i, j| {
        if j == 0 || j == 2 {
            i as f64
        } else {
            (i as f64 * 0.3).sin()
        }
    });
    let y = Col::from_fn(n, |i| 1.0 + 2.0 * i as f64);

    let svd_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y);

    assert!(svd_fit.is_ok(), "SVD should handle rank-deficient data");
    let result = svd_fit.unwrap();
    assert!(result.r_squared() > 0.9);
}

// ==============================
// Ridge solver selection tests
// ==============================

#[test]
fn test_ridge_all_solvers_match() {
    let (x, y) = well_conditioned_data();
    let lambda = 0.5;

    let qr_fit = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(lambda)
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_fit = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(lambda)
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(lambda)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
    let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();
    let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&qr_coefs, &svd_coefs, 1e-8),
        "Ridge QR vs SVD differ:\n  QR:  {:?}\n  SVD: {:?}",
        qr_coefs,
        svd_coefs
    );
    assert!(
        coefficients_match(&qr_coefs, &chol_coefs, 1e-8),
        "Ridge QR vs Cholesky differ:\n  QR:   {:?}\n  Chol: {:?}",
        qr_coefs,
        chol_coefs
    );
}

#[test]
fn test_ridge_no_intercept_all_solvers() {
    let (x, y) = well_conditioned_data();
    let lambda = 1.0;

    let qr_fit = RidgeRegressor::builder()
        .with_intercept(false)
        .lambda(lambda)
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_fit = RidgeRegressor::builder()
        .with_intercept(false)
        .lambda(lambda)
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = RidgeRegressor::builder()
        .with_intercept(false)
        .lambda(lambda)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
    let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();
    let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&qr_coefs, &svd_coefs, 1e-8),
        "Ridge no-intercept QR vs SVD differ"
    );
    assert!(
        coefficients_match(&qr_coefs, &chol_coefs, 1e-8),
        "Ridge no-intercept QR vs Cholesky differ"
    );
}

#[test]
fn test_ridge_cholesky_always_works_with_lambda() {
    // Even ill-conditioned data should work with Cholesky when λ > 0
    let n = 20;
    let x = Mat::from_fn(n, 3, |i, j| {
        if j == 0 || j == 2 {
            i as f64 // Collinear columns
        } else {
            (i as f64 * 0.3).sin()
        }
    });
    let y = Col::from_fn(n, |i| 1.0 + 2.0 * i as f64);

    let chol_fit = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(1.0)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y);

    assert!(
        chol_fit.is_ok(),
        "Ridge Cholesky should always work with λ > 0"
    );
}

#[test]
fn test_ridge_various_lambdas() {
    let (x, y) = well_conditioned_data();

    for lambda in &[0.01, 0.1, 1.0, 10.0] {
        let qr_fit = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(*lambda)
            .solve_method(SolverType::Qr)
            .build()
            .fit(&x, &y)
            .unwrap();

        let svd_fit = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(*lambda)
            .solve_method(SolverType::Svd)
            .build()
            .fit(&x, &y)
            .unwrap();

        let chol_fit = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(*lambda)
            .solve_method(SolverType::Cholesky)
            .build()
            .fit(&x, &y)
            .unwrap();

        let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
        let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();
        let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

        assert!(
            coefficients_match(&qr_coefs, &svd_coefs, 1e-7),
            "Ridge λ={}: QR vs SVD differ",
            lambda
        );
        assert!(
            coefficients_match(&qr_coefs, &chol_coefs, 1e-7),
            "Ridge λ={}: QR vs Cholesky differ",
            lambda
        );
    }
}

// ==============================
// WLS solver selection tests
// ==============================

#[test]
fn test_wls_all_solvers_match() {
    let (x, y) = well_conditioned_data();
    let n = x.nrows();
    let weights = Col::from_fn(n, |i| 1.0 / ((i + 1) as f64));

    let qr_fit = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights.clone())
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_fit = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights.clone())
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights.clone())
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
    let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();
    let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&qr_coefs, &svd_coefs, 1e-8),
        "WLS QR vs SVD differ:\n  QR:  {:?}\n  SVD: {:?}",
        qr_coefs,
        svd_coefs
    );
    assert!(
        coefficients_match(&qr_coefs, &chol_coefs, 1e-8),
        "WLS QR vs Cholesky differ:\n  QR:   {:?}\n  Chol: {:?}",
        qr_coefs,
        chol_coefs
    );
}

#[test]
fn test_wls_no_intercept_all_solvers() {
    let (x, y) = well_conditioned_data();
    let n = x.nrows();
    let weights = Col::from_fn(n, |i| 1.0 + (i as f64 * 0.1));

    let qr_fit = WlsRegressor::builder()
        .with_intercept(false)
        .weights(weights.clone())
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_fit = WlsRegressor::builder()
        .with_intercept(false)
        .weights(weights.clone())
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = WlsRegressor::builder()
        .with_intercept(false)
        .weights(weights.clone())
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    let qr_coefs: Vec<f64> = qr_fit.coefficients().iter().copied().collect();
    let svd_coefs: Vec<f64> = svd_fit.coefficients().iter().copied().collect();
    let chol_coefs: Vec<f64> = chol_fit.coefficients().iter().copied().collect();

    assert!(
        coefficients_match(&qr_coefs, &svd_coefs, 1e-8),
        "WLS no-intercept QR vs SVD differ"
    );
    assert!(
        coefficients_match(&qr_coefs, &chol_coefs, 1e-8),
        "WLS no-intercept QR vs Cholesky differ"
    );
}

// ==============================
// Default solver is QR
// ==============================

#[test]
fn test_default_solver_is_qr() {
    let opts = RegressionOptions::default();
    assert_eq!(opts.solver, SolverType::Qr);
}

#[test]
fn test_solver_type_in_builder() {
    let opts = RegressionOptions::builder()
        .solver(SolverType::Svd)
        .build_unchecked();
    assert_eq!(opts.solver, SolverType::Svd);

    let opts = RegressionOptions::builder()
        .solver(SolverType::Cholesky)
        .build_unchecked();
    assert_eq!(opts.solver, SolverType::Cholesky);
}

// ==============================
// Predictions match across solvers
// ==============================

#[test]
fn test_ols_predictions_match_across_solvers() {
    let (x, y) = well_conditioned_data();

    let qr_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Qr)
        .build()
        .fit(&x, &y)
        .unwrap();

    let svd_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Svd)
        .build()
        .fit(&x, &y)
        .unwrap();

    let chol_fit = OlsRegressor::builder()
        .with_intercept(true)
        .solve_method(SolverType::Cholesky)
        .build()
        .fit(&x, &y)
        .unwrap();

    // Predict on new data
    let x_new = Mat::from_fn(5, 2, |i, j| {
        if j == 0 {
            (i as f64 + 10.0) / 10.0
        } else {
            ((i as f64 + 10.0) * 0.7 + 3.0).sin() * 5.0
        }
    });

    let qr_preds = qr_fit.predict(&x_new);
    let svd_preds = svd_fit.predict(&x_new);
    let chol_preds = chol_fit.predict(&x_new);

    for i in 0..5 {
        assert!(
            (qr_preds[i] - svd_preds[i]).abs() < 1e-8,
            "Prediction[{}] QR={} SVD={}",
            i,
            qr_preds[i],
            svd_preds[i]
        );
        assert!(
            (qr_preds[i] - chol_preds[i]).abs() < 1e-8,
            "Prediction[{}] QR={} Chol={}",
            i,
            qr_preds[i],
            chol_preds[i]
        );
    }
}
