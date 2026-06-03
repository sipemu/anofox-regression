//! Theil-Sen validation against scikit-learn 1.5.2.
//!
//! Reference values are produced by
//! `validation/python/generate_theil_sen_validation.py`. Regenerate with:
//!
//! ```bash
//! validation/python/.venv/bin/python \
//!     validation/python/generate_theil_sen_validation.py \
//!     > tests/fixtures/theil_sen_validation.rs
//! ```

#![allow(dead_code)]

use anofox_regression::solvers::{FittedRegressor, Regressor, TheilSenRegressor};
use faer::{Col, Mat};

include!("fixtures/theil_sen_validation.rs");

#[test]
fn univariate_matches_sklearn() {
    let n = N_THEIL_UNI;
    let x = Mat::from_fn(n, 1, |i, _| X_THEIL_UNI[i]);
    let y = Col::from_fn(n, |i| Y_THEIL_UNI[i]);

    let fitted = TheilSenRegressor::builder()
        .with_intercept(true)
        .random_state(42)
        .build()
        .fit(&x, &y)
        .expect("fit failed");

    let intercept = fitted.result().intercept.unwrap();
    let coef = fitted.result().coefficients[0];

    // The univariate Theil-Sen is exact (median of pairwise slopes is a
    // closed-form computation) — sklearn and we should agree to machine
    // precision modulo floating-point summation order.
    assert!(
        (intercept - EXPECTED_INTERCEPT_THEIL_UNI).abs() < 1e-10,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_THEIL_UNI
    );
    assert!(
        (coef - EXPECTED_COEF_THEIL_UNI).abs() < 1e-10,
        "coef {} vs sklearn {}",
        coef,
        EXPECTED_COEF_THEIL_UNI
    );
}

#[test]
fn multivariate_matches_sklearn_exhaustive() {
    let n = N_THEIL_MULTI;
    let p = P_THEIL_MULTI;
    let x = Mat::from_fn(n, p, |i, j| X_THEIL_MULTI_FLAT[i * p + j]);
    let y = Col::from_fn(n, |i| Y_THEIL_MULTI[i]);

    let fitted = TheilSenRegressor::builder()
        .with_intercept(true)
        .max_subpopulation(10_000)
        .max_iter(500)
        .tolerance(1e-6)
        .random_state(42)
        .build()
        .fit(&x, &y)
        .expect("fit failed");

    let intercept = fitted.result().intercept.unwrap();
    let coefs = &fitted.result().coefficients;

    // The multivariate case takes the Vardi-Zhang spatial median of all
    // OLS-on-subsample coefficient vectors. sklearn does the same. The
    // residual tolerance below covers Weiszfeld convergence differences.
    let tol = 5e-3;
    assert!(
        (intercept - EXPECTED_INTERCEPT_THEIL_MULTI).abs() < tol,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_THEIL_MULTI
    );
    for j in 0..p {
        assert!(
            (coefs[j] - EXPECTED_COEFS_THEIL_MULTI[j]).abs() < tol,
            "coef[{}] {} vs sklearn {}",
            j,
            coefs[j],
            EXPECTED_COEFS_THEIL_MULTI[j]
        );
    }
}
