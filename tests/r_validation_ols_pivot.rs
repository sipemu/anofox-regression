//! OLS validation against R `lm()` on designs whose column norms force a
//! **non-trivial column-pivot** in faer's `col_piv_qr` — the case that exposed
//! the unpermute bug (#26; the pre-fix OLS returned a scrambled coefficient
//! vector here). The existing OLS R-validation used only well-scaled designs
//! that pivot trivially, which is why the bug slipped through.
//!
//! Reference values from `tests/r_scripts/generate_ols_pivot_validation.R`.

#![allow(dead_code)]
#![allow(clippy::approx_constant)]

use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

include!("fixtures/ols_pivot_validation.rs");

const TOL: f64 = 1e-8;

fn design(xflat: &[f64], y: &[f64], n: usize, p: usize) -> (Mat<f64>, Col<f64>) {
    (
        Mat::from_fn(n, p, |i, j| xflat[i * p + j]),
        Col::from_fn(n, |i| y[i]),
    )
}

#[test]
fn ols_with_intercept_matches_r_under_nontrivial_pivot() {
    let (x, y) = design(&A_X, &A_Y, A_N, A_P);
    let fit = OlsRegressor::builder()
        .with_intercept(true)
        .confidence_level(0.95)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();

    // R coefficient order: [intercept, v1..vp].
    assert!((r.intercept.unwrap() - A_COEF[0]).abs() < TOL, "intercept");
    for j in 0..A_P {
        assert!(
            (r.coefficients[j] - A_COEF[j + 1]).abs() < TOL,
            "coef[{j}] = {} vs {}",
            r.coefficients[j],
            A_COEF[j + 1]
        );
    }
    // Standard errors.
    let se = r.std_errors.as_ref().unwrap();
    assert!(
        (r.intercept_std_error.unwrap() - A_SE[0]).abs() < TOL,
        "se intercept"
    );
    for j in 0..A_P {
        assert!((se[j] - A_SE[j + 1]).abs() < TOL, "se[{j}]");
    }
    // Fitted values.
    let pred = fit.predict(&x);
    for i in 0..A_N {
        assert!((pred[i] - A_FITTED[i]).abs() < 1e-7, "fitted[{i}]");
    }
}

#[test]
fn ols_no_intercept_matches_r_under_nontrivial_pivot() {
    let (x, y) = design(&B_X, &B_Y, B_N, B_P);
    let fit = OlsRegressor::builder()
        .with_intercept(false)
        .confidence_level(0.95)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();

    for j in 0..B_P {
        assert!(
            (r.coefficients[j] - B_COEF[j]).abs() < TOL,
            "coef[{j}] = {} vs {}",
            r.coefficients[j],
            B_COEF[j]
        );
    }
    let se = r.std_errors.as_ref().unwrap();
    for j in 0..B_P {
        assert!((se[j] - B_SE[j]).abs() < TOL, "se[{j}]");
    }
    let pred = fit.predict(&x);
    for i in 0..B_N {
        assert!((pred[i] - B_FITTED[i]).abs() < 1e-7, "fitted[{i}]");
    }
}

#[test]
fn ols_rank_deficient_large_scale_matches_r() {
    // Case C: exact collinearity v3 = 2*v1 at column scale ~1e1–1e3. An absolute
    // rank tolerance misses this; the relative threshold detects it. Which
    // collinear column is aliased may differ from R (a non-unique choice), so we
    // check the (unique) fitted values and that exactly one column is aliased.
    let (x, y) = design(&C_X, &C_Y, C_N, C_P);
    let fit = OlsRegressor::builder()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();

    let aliased = fit.result().aliased.iter().filter(|a| **a).count();
    assert_eq!(aliased, 1, "exactly one collinear column should be aliased");

    let pred = fit.predict(&x);
    for i in 0..C_N {
        assert!(
            (pred[i] - C_FITTED[i]).abs() < 1e-7,
            "fitted[{i}] = {} vs {}",
            pred[i],
            C_FITTED[i]
        );
    }
}
