//! #191: bounded least squares with non-zero bounds, unbounded coefficients
//! and an intercept.
//!
//! Reference values: `tests/r_scripts/generate_bls_validation.R`
//! (`nnls::nnls`, `bvls::bvls`, `quadprog::solve.QP`).

use anofox_regression::solvers::{BlsRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

const X0: [f64; N] = [
    0.878,
    0.183,
    1.526,
    0.13500000000000001,
    1.7110000000000001,
    0.83899999999999997,
    0.98499999999999999,
    0.031,
    1.246,
    0.64400000000000002,
    1.978,
    0.28599999999999998,
    1.103,
    1.1799999999999999,
    0.68500000000000005,
    1.0669999999999999,
    1.8839999999999999,
    0.050000000000000003,
    0.49399999999999999,
    0.036999999999999998,
    0.71299999999999997,
    1.286,
    1.6719999999999999,
    1.075,
    1.851,
    0.51400000000000001,
    1.1200000000000001,
    1.944,
    1.79,
    1.522,
];
const X1: [f64; N] = [
    -0.48999999999999999,
    -0.062,
    -1.3149999999999999,
    -0.63200000000000001,
    1.0269999999999999,
    0.51700000000000002,
    -0.098000000000000004,
    0.34399999999999997,
    0.31,
    -0.31,
    0.35699999999999998,
    -1.044,
    -1.4750000000000001,
    -1.0129999999999999,
    -2.3969999999999998,
    1.631,
    -0.13700000000000001,
    -1.6100000000000001,
    -0.48699999999999999,
    -0.031,
    -0.56000000000000005,
    -2.4399999999999999,
    -0.081000000000000003,
    -0.91200000000000003,
    -0.84799999999999998,
    -0.64900000000000002,
    -0.27100000000000002,
    1.448,
    -1.264,
    -0.107,
];
const X2: [f64; N] = [
    -0.78800000000000003,
    0.034000000000000002,
    -0.45900000000000002,
    -0.22500000000000001,
    0.38300000000000001,
    0.56499999999999995,
    0.14000000000000001,
    -0.95099999999999996,
    0.16,
    0.84699999999999998,
    0.38200000000000001,
    0.371,
    0.72799999999999998,
    0.60399999999999998,
    0.92700000000000005,
    0.12,
    0.53000000000000003,
    -0.53700000000000003,
    0.13400000000000001,
    0.67000000000000004,
    -0.019,
    -0.71899999999999997,
    0.76700000000000002,
    -0.54600000000000004,
    0.70499999999999996,
    0.217,
    0.83599999999999997,
    0.878,
    0.76200000000000001,
    -0.61499999999999999,
];
const Y: [f64; N] = [
    -1.1080000000000001,
    1.659,
    -1.6679999999999999,
    0.62,
    1.272,
    1.8979999999999999,
    0.77500000000000002,
    -0.38800000000000001,
    0.72899999999999998,
    2.5259999999999998,
    0.38900000000000001,
    1.3799999999999999,
    1.1120000000000001,
    1.0149999999999999,
    1.2749999999999999,
    1.901,
    0.59699999999999998,
    -0.99199999999999999,
    1.02,
    2.423,
    0.316,
    -2.6070000000000002,
    1.5109999999999999,
    -0.879,
    0.95099999999999996,
    0.95199999999999996,
    2.3090000000000002,
    2.7349999999999999,
    0.875,
    -1.4319999999999999,
];
const NNLS: [f64; 3] = [0.11788699104983859, 0.33164183100184086, 2.0303353097064596];
const BOX: [f64; 3] = [0.21283303517211041, 0.25, 1.5];
const BOX_QP: [f64; 3] = [0.2128330351721105, 0.25, 1.5];
const NNLS_INT: [f64; 4] = [
    0.56488533344421521,
    0.0,
    0.51778360238407062,
    1.8205381466040436,
];
const MIXED_INT: [f64; 4] = [
    1.2358440501880386,
    -0.88343755975027638,
    0.29999999999999999,
    2.5,
];
const XS: [f64; 21] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
    9.5, 10.0,
];
const YS: [f64; 21] = [
    -4.9915852901519209,
    -4.7409070257317429,
    -4.4985887999194016,
    -4.2575680249530796,
    -4.0095892427466318,
    -3.7527941549819892,
    -3.493430134012812,
    -3.2401064175337662,
    -2.9958788151475826,
    -2.7554402111088936,
    -2.5099999020655068,
    -2.2553657291800042,
    -1.9957983296317336,
    -1.7400939264430513,
    -1.4934971215984287,
    -1.2528790331666506,
    -1.0096139749187956,
    -0.75750987246771673,
    -0.49850122790337048,
    -0.24087054749272371,
    0.0083665563853605601,
];
const LINE: [f64; 2] = [-4.9991312696425378, 0.5000010041687889];

const N: usize = 30;
const TOL: f64 = 1e-9;

fn data() -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(N, 3, |i, j| [X0[i], X1[i], X2[i]][j]);
    (x, Col::from_fn(N, |i| Y[i]))
}

fn assert_coefs(fit: &impl FittedRegressor, intercept: Option<f64>, coefs: &[f64], what: &str) {
    let r = fit.result();
    match intercept {
        Some(b0) => assert!((r.intercept.unwrap() - b0).abs() < TOL, "{what}: intercept"),
        None => assert!(r.intercept.is_none()),
    }
    for (j, &b) in coefs.iter().enumerate() {
        assert!(
            (r.coefficients[j] - b).abs() < TOL,
            "{what}: coef {j} = {} vs R {b}",
            r.coefficients[j]
        );
    }
}

#[test]
fn nnls_without_intercept_matches_nnls() {
    let (x, y) = data();
    let fit = BlsRegressor::nnls()
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_coefs(&fit, None, &NNLS, "nnls");
}

#[test]
fn box_bounds_at_nonzero_values_match_bvls() {
    let (x, y) = data();
    let fit = BlsRegressor::builder()
        .with_intercept(false)
        .lower_bounds(vec![0.2, -1.0, 0.0])
        .upper_bounds(vec![0.5, 0.25, 1.5])
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_coefs(&fit, None, &BOX, "box (bvls)");
    assert_coefs(&fit, None, &BOX_QP, "box (quadprog)");
}

#[test]
fn nnls_with_intercept_matches_quadprog() {
    let (x, y) = data();
    let fit = BlsRegressor::nnls().build().fit(&x, &y).unwrap();
    assert_coefs(&fit, Some(NNLS_INT[0]), &NNLS_INT[1..], "nnls + intercept");
}

#[test]
fn unbounded_and_mixed_bounds_with_intercept_match_quadprog() {
    let (x, y) = data();
    let fit = BlsRegressor::builder()
        .lower_bounds(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.5])
        .upper_bounds(vec![f64::INFINITY, 0.3, 4.0])
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_coefs(&fit, Some(MIXED_INT[0]), &MIXED_INT[1..], "mixed");
}

#[test]
fn nnls_intercept_can_be_negative() {
    // y = -5 + 0.5 x: the unconstrained fit is feasible, so NNLS = OLS.
    let x = Mat::from_fn(XS.len(), 1, |i, _| XS[i]);
    let y = Col::from_fn(YS.len(), |i| YS[i]);
    let fit = BlsRegressor::nnls().build().fit(&x, &y).unwrap();
    assert_coefs(&fit, Some(LINE[0]), &LINE[1..], "line");
}
