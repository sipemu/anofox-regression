//! #210: Theil–Sen without intercept and with `n_subsamples` below `n`,
//! against scikit-learn 1.9.1 (exhaustive-enumeration branch, so the
//! reference is deterministic).
//!
//! Reference values: `validation/python/generate_theil_sen_210_validation.py`
//! (`tests/fixtures/theil_sen_210_validation.rs`).

#![allow(clippy::excessive_precision)]

use anofox_regression::solvers::{FittedRegressor, Regressor, TheilSenRegressor};
use faer::{Col, Mat};

include!("fixtures/theil_sen_210_validation.rs");

const N: usize = 12;

fn x(p: usize) -> Mat<f64> {
    Mat::from_fn(N, p, |i, j| X_FLAT[i * 2 + j])
}

fn coefs(fit: &impl FittedRegressor) -> Vec<f64> {
    let r = fit.result();
    r.intercept
        .into_iter()
        .chain(r.coefficients.iter().copied())
        .collect()
}

fn check(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
            "{what}[{k}]: {g} vs sklearn {w}"
        );
    }
}

fn builder() -> anofox_regression::solvers::TheilSenRegressorBuilder {
    TheilSenRegressor::builder()
        .max_subpopulation(10_000)
        .max_iter(1000)
        .tolerance(1e-12)
}

#[test]
fn without_intercept_matches_sklearn() {
    let y = Col::from_fn(N, |i| Y[i]);
    let f = builder()
        .with_intercept(false)
        .build()
        .fit(&x(1), &y)
        .unwrap();
    check(&coefs(&f), &NOINT_UNI, "univariate, no intercept");
    let f = builder()
        .with_intercept(false)
        .build()
        .fit(&x(2), &y)
        .unwrap();
    check(&coefs(&f), &NOINT_MULTI, "bivariate, no intercept");
}

#[test]
fn n_subsamples_above_the_minimum_matches_sklearn() {
    let y = Col::from_fn(N, |i| Y[i]);
    let f = builder().n_subsamples(5).build().fit(&x(2), &y).unwrap();
    check(&coefs(&f), &SUBSAMPLES5, "n_subsamples = 5");
    let f = builder()
        .with_intercept(false)
        .n_subsamples(4)
        .build()
        .fit(&x(2), &y)
        .unwrap();
    check(
        &coefs(&f),
        &NOINT_SUBSAMPLES4,
        "n_subsamples = 4, no intercept",
    );
}
