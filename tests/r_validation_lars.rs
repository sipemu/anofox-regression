//! LARS / LassoLars validation against scikit-learn 1.5.2.
//!
//! Reference values come from
//! `validation/python/generate_lars_validation.py`. Tolerance settings
//! reflect the LARS path being computed from O(n*p²) linear algebra
//! identically on both sides.

#![allow(dead_code)]

use anofox_regression::solvers::{FittedRegressor, LarsMethod, LarsRegressor, Regressor};
use faer::{Col, Mat};

include!("fixtures/lars_validation.rs");

const COEF_TOL: f64 = 1e-6;

#[test]
fn lars_full_path_matches_sklearn() {
    let n = N_LARS_FULL;
    let p = P_LARS_FULL;
    let x = Mat::from_fn(n, p, |i, j| X_LARS_FULL_FLAT[i * p + j]);
    let y = Col::from_fn(n, |i| Y_LARS_FULL[i]);

    let fitted = LarsRegressor::builder()
        .method(LarsMethod::Lar)
        .fit_intercept(true)
        .n_nonzero_coefs(p)
        .build()
        .fit(&x, &y)
        .expect("fit failed");

    let intercept = fitted.result().intercept.unwrap();
    let coefs = &fitted.result().coefficients;

    assert!(
        (intercept - EXPECTED_INTERCEPT_LARS_FULL).abs() < COEF_TOL,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_LARS_FULL
    );
    for j in 0..p {
        assert!(
            (coefs[j] - EXPECTED_COEFS_LARS_FULL[j]).abs() < COEF_TOL,
            "coef[{}] {} vs sklearn {}",
            j,
            coefs[j],
            EXPECTED_COEFS_LARS_FULL[j]
        );
    }
}

#[test]
fn lars_truncated_matches_sklearn() {
    let n = N_LARS_TRUNC;
    let p = P_LARS_TRUNC;
    let x = Mat::from_fn(n, p, |i, j| X_LARS_TRUNC_FLAT[i * p + j]);
    let y = Col::from_fn(n, |i| Y_LARS_TRUNC[i]);

    let fitted = LarsRegressor::builder()
        .method(LarsMethod::Lar)
        .fit_intercept(true)
        .n_nonzero_coefs(2)
        .build()
        .fit(&x, &y)
        .expect("fit failed");

    let intercept = fitted.result().intercept.unwrap();
    let coefs = &fitted.result().coefficients;

    assert!(
        (intercept - EXPECTED_INTERCEPT_LARS_TRUNC).abs() < COEF_TOL,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_LARS_TRUNC
    );
    for j in 0..p {
        assert!(
            (coefs[j] - EXPECTED_COEFS_LARS_TRUNC[j]).abs() < COEF_TOL,
            "coef[{}] {} vs sklearn {}",
            j,
            coefs[j],
            EXPECTED_COEFS_LARS_TRUNC[j]
        );
    }
}

#[test]
fn lasso_lars_matches_sklearn() {
    let n = N_LASSO_LARS;
    let p = P_LASSO_LARS;
    let x = Mat::from_fn(n, p, |i, j| X_LASSO_LARS_FLAT[i * p + j]);
    let y = Col::from_fn(n, |i| Y_LASSO_LARS[i]);

    // sklearn's LassoLars uses the parameter `alpha` in the
    // `(1/(2n)) ||y - Xβ||² + α ||β||₁` form. Our path solver uses the
    // canonical LARS scaling where alpha is the max-correlation
    // termination point and is exactly `n * sklearn_alpha`.
    let target_alpha = 0.1 * n as f64;

    let fitted = LarsRegressor::builder()
        .method(LarsMethod::Lasso)
        .fit_intercept(true)
        .alpha(target_alpha)
        .build()
        .fit(&x, &y)
        .expect("fit failed");

    let intercept = fitted.result().intercept.unwrap();
    let coefs = &fitted.result().coefficients;

    let tol = 5e-3; // LassoLars termination point is interpolated;
                    // we hit the same alpha within numerical precision.
    assert!(
        (intercept - EXPECTED_INTERCEPT_LASSO_LARS).abs() < tol,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_LASSO_LARS
    );
    for j in 0..p {
        assert!(
            (coefs[j] - EXPECTED_COEFS_LASSO_LARS[j]).abs() < tol,
            "coef[{}] {} vs sklearn {}",
            j,
            coefs[j],
            EXPECTED_COEFS_LASSO_LARS[j]
        );
    }
}
