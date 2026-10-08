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

    // sklearn's 1-D path is its spatial-median algorithm, not the classical
    // pairwise-slope median; opt into it explicitly.
    let fitted = TheilSenRegressor::builder()
        .with_intercept(true)
        .univariate_pairwise(false)
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

/// Classical univariate Theil–Sen (default): slope = median of pairwise slopes
/// over pairs with distinct x, intercept = median(y − slope·x).
///
/// R Code:
/// ```r
/// ts <- function(x, y) { s <- c(); n <- length(x)
///   for (i in 1:(n-1)) for (j in (i+1):n) if (x[i] != x[j]) s <- c(s, (y[j]-y[i])/(x[j]-x[i]))
///   b <- median(s); c(median(y - b*x), b) }
/// ts(seq(0, 10, length.out = 30), Y_THEIL_UNI)   # 1.5269819301694925 0.7069232167262927
/// ts(c(1,2,2,3,4,4,4,5,7,8), c(2,1,5,3,8,6,6,9,10,7))  # 0.666666666666667 1.333333333333333
/// ```
#[test]
fn univariate_classic_pairwise_median_vs_r() {
    let n = N_THEIL_UNI;
    let x = Mat::from_fn(n, 1, |i, _| X_THEIL_UNI[i]);
    let y = Col::from_fn(n, |i| Y_THEIL_UNI[i]);
    let fitted = TheilSenRegressor::builder()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("fit failed");
    assert!((fitted.result().coefficients[0] - 0.7069232167262927).abs() < 1e-12);
    assert!((fitted.result().intercept.unwrap() - 1.5269819301694925).abs() < 1e-12);

    // Ties in x (excluded pairs), duplicated points, even number of slopes.
    let xv = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 7.0, 8.0];
    let yv = [2.0, 1.0, 5.0, 3.0, 8.0, 6.0, 6.0, 9.0, 10.0, 7.0];
    let x = Mat::from_fn(10, 1, |i, _| xv[i]);
    let y = Col::from_fn(10, |i| yv[i]);
    let fitted = TheilSenRegressor::builder()
        .build()
        .fit(&x, &y)
        .expect("fit failed");
    assert!((fitted.result().coefficients[0] - 4.0 / 3.0).abs() < 1e-12);
    assert!((fitted.result().intercept.unwrap() - 2.0 / 3.0).abs() < 1e-12);
}
