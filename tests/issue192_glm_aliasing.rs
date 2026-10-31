//! #192: the IRLS GLMs alias columns that are linear combinations of the
//! intercept and earlier columns (here x3 = a + b and x4 = 2 − a/2), with
//! `NaN` coefficients and standard errors, as R's `glm`.
//!
//! Reference values: `tests/r_scripts/generate_glm_aliasing_validation.R`
//! (R 4.x, MASS, statmod).

use anofox_regression::solvers::{
    BinomialRegressor, FittedRegressor, NegativeBinomialRegressor, PoissonRegressor, Regressor,
    TweedieRegressor,
};
use faer::{Col, Mat};

const A: [f64; N] = [
    0.68500000000000005,
    -0.62,
    0.106,
    -0.51600000000000001,
    0.36799999999999999,
    -0.51400000000000001,
    -0.32900000000000001,
    0.214,
    -0.71299999999999997,
    0.55700000000000005,
    -0.28499999999999998,
    -0.27500000000000002,
    -0.31,
    -0.021000000000000001,
    0.079000000000000001,
    -0.48699999999999999,
    -0.189,
    -0.14799999999999999,
    0.154,
    -0.94699999999999995,
    0.17199999999999999,
    0.29999999999999999,
    -0.59499999999999997,
    0.90200000000000002,
    -0.221,
    -0.17899999999999999,
    0.071999999999999995,
    -0.40699999999999997,
    0.85699999999999998,
    0.91300000000000003,
    0.96699999999999997,
    0.090999999999999998,
    -0.38,
    0.41599999999999998,
    -0.080000000000000002,
    0.32300000000000001,
    0.94399999999999995,
    0.084000000000000005,
    -0.39800000000000002,
    0.17000000000000001,
];
const B: [f64; N] = [
    0.64300000000000002,
    0.54400000000000004,
    0.025999999999999999,
    1.952,
    1.3700000000000001,
    1.958,
    0.252,
    0.95299999999999996,
    1.331,
    0.80200000000000005,
    0.35099999999999998,
    1.3640000000000001,
    1.8740000000000001,
    0.40000000000000002,
    0.72699999999999998,
    0.47699999999999998,
    1.1259999999999999,
    0.20300000000000001,
    1.427,
    1.085,
    1.032,
    0.090999999999999998,
    1.5369999999999999,
    0.10199999999999999,
    0.032000000000000001,
    0.29399999999999998,
    0.75700000000000001,
    0.89700000000000002,
    1.76,
    1.5980000000000001,
    1.982,
    1.5189999999999999,
    1.2430000000000001,
    0.182,
    1.5720000000000001,
    1.827,
    1.26,
    1.113,
    0.77900000000000003,
    1.446,
];
const YC: [f64; N] = [
    2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0, 3.0, 1.0,
    1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 8.0, 4.0, 2.0, 2.0, 4.0, 3.0, 3.0, 1.0, 0.0,
    2.0, 0.0,
];
const YB: [f64; N] = [
    1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 1.0,
];
const YG: [f64; N] = [
    2.2050000000000001,
    0.18909999999999999,
    1.2417,
    0.106,
    1.7499,
    0.28820000000000001,
    1.7509999999999999,
    0.98160000000000003,
    0.22209999999999999,
    1.8076000000000001,
    1.8405,
    0.6734,
    0.2157,
    1.1768000000000001,
    1.5289999999999999,
    0.88600000000000001,
    1.2908999999999999,
    1.4198,
    1.0639000000000001,
    0.74370000000000003,
    0.72519999999999996,
    0.58279999999999998,
    1.0431999999999999,
    2.2115999999999998,
    1.157,
    0.80659999999999998,
    1.4257,
    0.40560000000000002,
    1.6325000000000001,
    0.54820000000000002,
    0.76280000000000003,
    1.4577,
    0.39169999999999999,
    0.75580000000000003,
    1.3030999999999999,
    0.82310000000000005,
    1.7994000000000001,
    0.89710000000000001,
    0.18990000000000001,
    1.0126999999999999,
];
const POIS_COEF: [f64; 5] = [
    0.47944526042609198,
    0.49325426594316074,
    0.08113407682115914,
    f64::NAN,
    f64::NAN,
];
const POIS_SE: [f64; 5] = [
    0.23147222567302236,
    0.23243199415094221,
    0.19330214454698164,
    f64::NAN,
    f64::NAN,
];
const POIS_DEV: f64 = 51.587392671438884;
const BIN_COEF: [f64; 5] = [
    -1.3333637638563227,
    1.6092787425370678,
    0.60359854812616986,
    f64::NAN,
    f64::NAN,
];
const BIN_SE: [f64; 5] = [
    0.72597094915030613,
    0.7779802134141719,
    0.60421773820329217,
    f64::NAN,
    f64::NAN,
];
const BIN_DEV: f64 = 45.666831203192991;
const GAMMA_COEF: [f64; 5] = [
    0.34216459715975411,
    0.67903554699902702,
    -0.39218271645078828,
    f64::NAN,
    f64::NAN,
];
const GAMMA_SE: [f64; 5] = [
    0.15563418121767647,
    0.16488194558430957,
    0.13372513422354237,
    f64::NAN,
    f64::NAN,
];
const GAMMA_DEV: f64 = 11.924946936291573;
const NB_COEF: [f64; 5] = [
    0.52711038557042778,
    0.48853952660534156,
    0.034584908252591973,
    f64::NAN,
    f64::NAN,
];
const NB_SE: [f64; 5] = [
    0.31636848009184271,
    0.32980115588439624,
    0.26991995533373486,
    f64::NAN,
    f64::NAN,
];
const NB_DEV: f64 = 30.73743275230747;
const TW_COEF: [f64; 5] = [
    0.30785954370993507,
    0.63760888096904289,
    -0.35434648212998759,
    f64::NAN,
    f64::NAN,
];
const TW_SE: [f64; 5] = [
    0.14040675689229487,
    0.15564327397816069,
    0.12584475306194204,
    f64::NAN,
    f64::NAN,
];
const TW_DEV: f64 = 9.8986618695510167;
const POIS_NOINT_COEF: [f64; 4] = [
    0.61311558104968311,
    0.08113407682115914,
    f64::NAN,
    0.23972263021304591,
];
const POIS_NOINT_SE: [f64; 4] = [
    0.23706072518527277,
    0.19330214454698164,
    f64::NAN,
    0.11573611283651114,
];
const POIS_NOINT_DEV: f64 = 51.587392671438892;

const N: usize = 40;

/// Columns a, b, a + b, 2 − a/2.
fn design(n_cols: usize) -> Mat<f64> {
    Mat::from_fn(N, n_cols, |i, j| match j {
        0 => A[i],
        1 => B[i],
        2 => A[i] + B[i],
        _ => 2.0 - 0.5 * A[i],
    })
}

fn y(v: &[f64; N]) -> Col<f64> {
    Col::from_fn(N, |i| v[i])
}

/// `coef` / `se` in R order: intercept (if any) then the four columns.
fn check(fit: &impl FittedRegressor, coef: &[f64], se: &[f64], what: &str) {
    let r = fit.result();
    let off = usize::from(r.intercept.is_some());
    let mut got_coef: Vec<f64> = r.intercept.into_iter().collect();
    got_coef.extend(r.coefficients.iter());
    let mut got_se: Vec<f64> = r.intercept_std_error.into_iter().collect();
    got_se.extend(r.std_errors.as_ref().unwrap().iter());
    for k in 0..coef.len() {
        if coef[k].is_nan() {
            assert!(got_coef[k].is_nan(), "{what}: coef {k} should be aliased");
            assert!(got_se[k].is_nan(), "{what}: se {k} should be NaN");
            assert!(r.aliased[k - off], "{what}: aliased flag {k}");
        } else {
            let tol = 1e-7 * (1.0 + coef[k].abs());
            assert!(
                (got_coef[k] - coef[k]).abs() < tol,
                "{what}: coef {k} {} vs R {}",
                got_coef[k],
                coef[k]
            );
            let tol = 1e-7 * (1.0 + se[k].abs());
            assert!(
                (got_se[k] - se[k]).abs() < tol,
                "{what}: se {k} {} vs R {}",
                got_se[k],
                se[k]
            );
            if k >= off {
                assert!(!r.aliased[k - off], "{what}: aliased flag {k}");
            }
        }
    }
}

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() < 1e-8 * (1.0 + b.abs()),
        "{what}: {a} vs R {b}"
    );
}

#[test]
fn poisson_aliases_linear_combinations() {
    let fit = PoissonRegressor::log()
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&design(4), &y(&YC))
        .unwrap();
    check(&fit, &POIS_COEF, &POIS_SE, "poisson");
    close(fit.deviance, POIS_DEV, "poisson deviance");
    // Prediction ignores the aliased columns.
    let pred = fit.predict(&design(4));
    close(pred[0], fit.result().fitted_values[0], "prediction");
}

#[test]
fn poisson_without_intercept_aliases_duplicate_sum() {
    let fit = PoissonRegressor::log()
        .with_intercept(false)
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&design(4), &y(&YC))
        .unwrap();
    check(
        &fit,
        &POIS_NOINT_COEF,
        &POIS_NOINT_SE,
        "poisson, no intercept",
    );
    close(fit.deviance, POIS_NOINT_DEV, "deviance");
}

#[test]
fn binomial_aliases_linear_combinations() {
    let fit = BinomialRegressor::logistic()
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&design(4), &y(&YB))
        .unwrap();
    check(&fit, &BIN_COEF, &BIN_SE, "binomial");
    close(fit.deviance, BIN_DEV, "binomial deviance");
}

#[test]
fn gamma_and_tweedie_alias_linear_combinations() {
    let fit = TweedieRegressor::gamma()
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&design(4), &y(&YG))
        .unwrap();
    check(&fit, &GAMMA_COEF, &GAMMA_SE, "gamma");
    close(fit.deviance, GAMMA_DEV, "gamma deviance");
    let fit = TweedieRegressor::builder()
        .var_power(1.5)
        .link_power(0.0)
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&design(4), &y(&YG))
        .unwrap();
    check(&fit, &TW_COEF, &TW_SE, "tweedie");
    close(fit.deviance, TW_DEV, "tweedie deviance");
}

#[test]
fn negative_binomial_aliases_linear_combinations() {
    let fit = NegativeBinomialRegressor::with_theta(2.0)
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&design(4), &y(&YC))
        .unwrap();
    check(&fit, &NB_COEF, &NB_SE, "negative binomial");
    close(fit.deviance, NB_DEV, "nb deviance");
}
