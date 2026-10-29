#![allow(clippy::excessive_precision)]

//! #135: `vcov()` of the GLM fits against R's `vcov(glm)` and
//! `MASS::glm.nb`. Reference values come from
//! `tests/r_scripts/generate_glm_vcov_validation.R`.

use anofox_regression::solvers::{
    BinomialRegressor, GammaRegressor, LogisticRegression, NegativeBinomialRegressor,
    PoissonRegressor, Regressor, TweedieRegressor,
};
use faer::{Col, Mat};

include!("fixtures/glm_vcov_validation.rs");

const N: usize = 80;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| X[i * 2 + j])
}

fn y(v: &[f64]) -> Col<f64> {
    Col::from_fn(N, |i| v[i])
}

fn assert_vcov(name: &str, got: Option<Mat<f64>>, expected: &[f64], tol: f64) {
    let got = got.unwrap_or_else(|| panic!("{name}: no vcov"));
    assert_eq!(got.nrows(), 3, "{name}");
    for a in 0..3 {
        for b in 0..3 {
            let e = expected[a * 3 + b];
            assert!(
                (got[(a, b)] - e).abs() <= tol * e.abs().max(1e-3),
                "{name} [{a},{b}]: {} vs {e}",
                got[(a, b)]
            );
        }
    }
}

#[test]
fn glm_vcov_matches_r() {
    let x = x();
    let pois = PoissonRegressor::log()
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build();
    assert_vcov(
        "poisson",
        pois.fit(&x, &y(&Y_POIS)).unwrap().vcov(),
        &VCOV_POIS,
        1e-6,
    );
    let bin = BinomialRegressor::logistic()
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build();
    assert_vcov(
        "binomial",
        bin.fit(&x, &y(&Y_BIN)).unwrap().vcov(),
        &VCOV_BIN,
        1e-6,
    );
    let tw = TweedieRegressor::gamma()
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build();
    assert_vcov(
        "gamma",
        tw.fit(&x, &y(&Y_GAMMA)).unwrap().vcov(),
        &VCOV_GAMMA,
        1e-6,
    );
    let nb = NegativeBinomialRegressor::builder()
        .with_intercept(true)
        .estimate_theta(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build();
    assert_vcov(
        "negative binomial",
        nb.fit(&x, &y(&Y_NB)).unwrap().vcov(),
        &VCOV_NB,
        1e-5,
    );
    let tw15 = TweedieRegressor::builder()
        .var_power(1.5)
        .link_power(0.0) // log, as tweedie(link.power = 0)
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build();
    assert_vcov(
        "tweedie 1.5",
        tw15.fit(&x, &y(&Y_TW)).unwrap().vcov(),
        &VCOV_TW,
        1e-6,
    );

    // The wrappers delegate to their GLM
    let gamma = GammaRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build();
    assert_vcov(
        "GammaRegressor",
        gamma.fit(&x, &y(&Y_GAMMA)).unwrap().vcov(),
        &VCOV_GAMMA,
        1e-6,
    );
    let logistic = LogisticRegression::builder()
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build();
    assert_vcov(
        "LogisticRegression",
        logistic.fit(&x, &y(&Y_BIN)).unwrap().vcov(),
        &VCOV_BIN,
        1e-6,
    );
}

/// A constant column is dropped (aliased): its row and column of `vcov` are
/// zero and the rest is the covariance of the fit without it.
#[test]
fn glm_vcov_with_an_aliased_column() {
    let x3 = Mat::from_fn(N, 3, |i, j| if j == 1 { 2.0 } else { X[i * 2 + j / 2] });
    let fit = PoissonRegressor::log()
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&x3, &y(&Y_POIS))
        .unwrap();
    let v = fit.vcov().expect("vcov");
    assert_eq!((v.nrows(), v.ncols()), (4, 4));
    let index = [0, 1, 3]; // intercept, x1, x2
    for (a, &i) in index.iter().enumerate() {
        assert_eq!(v[(2, i)], 0.0);
        assert_eq!(v[(i, 2)], 0.0);
        for (b, &j) in index.iter().enumerate() {
            let e = VCOV_POIS[a * 3 + b];
            assert!(
                (v[(i, j)] - e).abs() <= 1e-6 * e.abs().max(1e-3),
                "[{i},{j}]"
            );
        }
    }
}

#[test]
fn glm_vcov_needs_inference() {
    let fit = PoissonRegressor::log()
        .with_intercept(true)
        .compute_inference(false)
        .build()
        .fit(&x(), &y(&Y_POIS))
        .unwrap();
    assert!(fit.vcov().is_none());
}
