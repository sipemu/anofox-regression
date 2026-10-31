//! #209: the GLM datasets of `r_validation_glm.rs`, whose goldens came from
//! R's default `glm.control` and were checked at 0.01 – 2.0, against fully
//! converged R (`epsilon = 1e-15`). Each reference row is intercept,
//! slope, their standard errors, deviance.
//!
//! Reference values: `tests/r_scripts/generate_glm_tight_validation.R`.

#![allow(clippy::excessive_precision)]

use anofox_regression::solvers::{
    BinomialRegressor, FittedRegressor, NegativeBinomialRegressor, PoissonRegressor, Regressor,
    TweedieRegressor,
};
use faer::{Col, Mat};

const X_POISSON_LOG: [f64; 30] = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
];
const Y_POISSON_LOG: [f64; 30] = [
    4.0, 4.0, 1.0, 4.0, 3.0, 3.0, 4.0, 1.0, 4.0, 5.0, 4.0, 5.0, 8.0, 3.0, 5.0, 10.0, 12.0, 4.0,
    7.0, 8.0, 13.0, 6.0, 17.0, 6.0, 16.0, 16.0, 17.0, 17.0, 20.0, 14.0,
];
const X_POISSON_IDENTITY_R: [f64; 20] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0,
];
const Y_POISSON_IDENTITY_R: [f64; 20] = [
    6.0, 10.0, 11.0, 19.0, 19.0, 25.0, 27.0, 28.0, 34.0, 34.0, 36.0, 39.0, 42.0, 47.0, 47.0, 53.0,
    56.0, 57.0, 60.0, 64.0,
];
const X_BINOMIAL_LOGIT: [f64; 30] = [
    -3.0, -2.793103, -2.586207, -2.37931, -2.172414, -1.965517, -1.758621, -1.551724, -1.344828,
    -1.137931, -0.931034, -0.724138, -0.517241, -0.310345, -0.103448, 0.103448, 0.310345, 0.517241,
    0.724138, 0.931034, 1.137931, 1.344828, 1.551724, 1.758621, 1.965517, 2.172414, 2.37931,
    2.586207, 2.793103, 3.0,
];
const Y_BINOMIAL_LOGIT: [f64; 30] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const Y_BINOMIAL_PROBIT: [f64; 30] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
    1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const Y_BINOMIAL_CLOGLOG: [f64; 30] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const X_NEGBINOM: [f64; 30] = [
    0.5, 0.586207, 0.672414, 0.758621, 0.844828, 0.931034, 1.017241, 1.103448, 1.189655, 1.275862,
    1.362069, 1.448276, 1.534483, 1.62069, 1.706897, 1.793103, 1.87931, 1.965517, 2.051724,
    2.137931, 2.224138, 2.310345, 2.396552, 2.482759, 2.568966, 2.655172, 2.741379, 2.827586,
    2.913793, 3.0,
];
const Y_NEGBINOM: [f64; 30] = [
    2.0, 3.0, 3.0, 3.0, 4.0, 2.0, 0.0, 4.0, 4.0, 3.0, 3.0, 6.0, 5.0, 0.0, 5.0, 7.0, 2.0, 1.0, 0.0,
    3.0, 2.0, 2.0, 0.0, 3.0, 4.0, 4.0, 5.0, 2.0, 9.0, 0.0,
];
const X_GAMMA_LOG: [f64; 30] = [
    0.5, 0.586207, 0.672414, 0.758621, 0.844828, 0.931034, 1.017241, 1.103448, 1.189655, 1.275862,
    1.362069, 1.448276, 1.534483, 1.62069, 1.706897, 1.793103, 1.87931, 1.965517, 2.051724,
    2.137931, 2.224138, 2.310345, 2.396552, 2.482759, 2.568966, 2.655172, 2.741379, 2.827586,
    2.913793, 3.0,
];
const Y_GAMMA_LOG: [f64; 30] = [
    3.674048, 0.92559, 1.682034, 0.504913, 1.586715, 0.960427, 6.179875, 1.825446, 0.616511,
    0.386211, 5.035883, 4.321588, 4.225195, 3.387172, 4.757937, 1.128609, 2.006831, 8.698692,
    8.152989, 3.821331, 3.799281, 2.193752, 0.253188, 1.252328, 4.859668, 5.930911, 7.491828,
    2.16352, 2.012775, 1.896947,
];
const X_INVGAUSS_LOG: [f64; 30] = [
    0.5, 0.586207, 0.672414, 0.758621, 0.844828, 0.931034, 1.017241, 1.103448, 1.189655, 1.275862,
    1.362069, 1.448276, 1.534483, 1.62069, 1.706897, 1.793103, 1.87931, 1.965517, 2.051724,
    2.137931, 2.224138, 2.310345, 2.396552, 2.482759, 2.568966, 2.655172, 2.741379, 2.827586,
    2.913793, 3.0,
];
const Y_INVGAUSS_LOG: [f64; 30] = [
    0.520746, 3.077116, 2.684075, 1.062602, 3.095093, 1.926178, 0.518001, 2.596461, 0.357832,
    2.379919, 0.663142, 0.303635, 0.633964, 2.156652, 2.680012, 7.339561, 2.382664, 0.246063,
    0.286626, 0.751036, 2.654481, 0.493492, 3.462012, 0.888767, 0.457298, 2.594986, 3.505298,
    0.526974, 2.697976, 2.090898,
];
const X_POISSON_OFFSET_R: [f64; 20] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0,
];
const Y_POISSON_OFFSET_R: [f64; 20] = [
    18.0, 26.0, 49.0, 63.0, 19.0, 54.0, 72.0, 129.0, 29.0, 77.0, 134.0, 144.0, 53.0, 108.0, 175.0,
    249.0, 74.0, 137.0, 240.0, 397.0,
];
const EXPOSURE_POISSON_OFFSET_R: [f64; 20] = [
    10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0,
    10.0, 20.0, 30.0, 40.0,
];
const POISSON_LOG: [f64; 5] = [
    0.7992280823927338,
    0.71083901901576352,
    0.18125859102512781,
    0.082806242714249842,
    28.754845444381562,
];
const POISSON_IDENTITY: [f64; 5] = [
    4.0249970992800854,
    3.0166669429257054,
    1.7568515296520901,
    0.20133404485419348,
    2.0630466947338739,
];
const BINOMIAL_LOGIT: [f64; 5] = [
    4.5876837731828072e-16,
    0.84825701354759164,
    0.45017761174144211,
    0.30622699688600657,
    30.109861546446044,
];
const BINOMIAL_PROBIT: [f64; 5] = [
    0.34277615327098138,
    0.91989456210831366,
    0.34077804913470777,
    0.28240999342094641,
    18.955526746799656,
];
const BINOMIAL_CLOGLOG: [f64; 5] = [
    -0.52848440355450954,
    2.8596441126510306,
    0.62777471101868376,
    1.3891705705555082,
    8.1493585068843348,
];
const NEGBINOM: [f64; 5] = [
    1.0157127616094308,
    0.053231540975373429,
    0.33541699508442308,
    0.17469483662673657,
    37.612452369788159,
];
const NEGBINOM_THETA: f64 = 5.5678350860327743;
const GAMMA_LOG: [f64; 5] = [
    0.63348662445648163,
    0.28992090136017529,
    0.34330745727669848,
    0.18045701737354791,
    19.152862994827647,
];
const INVGAUSS_LOG: [f64; 5] = [
    0.54254978891643868,
    0.036449741611857558,
    0.38933590519520306,
    0.20668461727362272,
    19.642176167129172,
];
const POISSON_OFFSET: [f64; 5] = [
    0.23816931498485278,
    0.10034721538731751,
    0.060850797493693508,
    0.0040469526266145601,
    21.570715859719495,
];

fn xm(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

fn yc(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

/// Compare intercept, slope, their SEs and the deviance.
fn check(fit: &impl FittedRegressor, deviance: f64, want: &[f64; 5], tol: f64, what: &str) {
    let r = fit.result();
    let got = [
        r.intercept.unwrap(),
        r.coefficients[0],
        r.intercept_std_error.unwrap(),
        r.std_errors.as_ref().unwrap()[0],
        deviance,
    ];
    let names = ["intercept", "slope", "se intercept", "se slope", "deviance"];
    for k in 0..5 {
        assert!(
            (got[k] - want[k]).abs() <= tol * (1.0 + want[k].abs()),
            "{what} {}: {} vs R {}",
            names[k],
            got[k],
            want[k]
        );
    }
}

const TOL: f64 = 1e-7;

#[test]
fn poisson_fits_match_converged_r() {
    let f = PoissonRegressor::log()
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&xm(&X_POISSON_LOG), &yc(&Y_POISSON_LOG))
        .unwrap();
    check(&f, f.deviance, &POISSON_LOG, TOL, "poisson log");
    let off = Col::from_fn(EXPOSURE_POISSON_OFFSET_R.len(), |i| {
        EXPOSURE_POISSON_OFFSET_R[i].ln()
    });
    let f = PoissonRegressor::log()
        .offset(off)
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&xm(&X_POISSON_OFFSET_R), &yc(&Y_POISSON_OFFSET_R))
        .unwrap();
    check(&f, f.deviance, &POISSON_OFFSET, TOL, "poisson offset");
    // R stops at maxit on this identity-link fit (an observation sits at μ ≈ 0),
    // so its reference is only good to ~1e-5.
    let f = PoissonRegressor::identity()
        .compute_inference(true)
        .tolerance(1e-10)
        .build()
        .fit(&xm(&X_POISSON_IDENTITY_R), &yc(&Y_POISSON_IDENTITY_R))
        .unwrap();
    check(&f, f.deviance, &POISSON_IDENTITY, 1e-5, "poisson identity");
}

#[test]
fn binomial_fits_match_converged_r() {
    let x = xm(&X_BINOMIAL_LOGIT);
    let cases = [
        (
            BinomialRegressor::logistic(),
            &Y_BINOMIAL_LOGIT,
            &BINOMIAL_LOGIT,
            "logit",
        ),
        (
            BinomialRegressor::probit(),
            &Y_BINOMIAL_PROBIT,
            &BINOMIAL_PROBIT,
            "probit",
        ),
        (
            BinomialRegressor::cloglog(),
            &Y_BINOMIAL_CLOGLOG,
            &BINOMIAL_CLOGLOG,
            "cloglog",
        ),
    ];
    for (b, y, want, what) in cases {
        let f = b
            .compute_inference(true)
            .tolerance(1e-13)
            .build()
            .fit(&x, &yc(y))
            .unwrap();
        // The cloglog data are close to separation (R warns about fitted
        // probabilities of 0 / 1): its SEs are only stable to ~1e-6.
        let tol = if what == "cloglog" { 1e-5 } else { TOL };
        check(&f, f.deviance, want, tol, what);
    }
}

#[test]
fn negative_binomial_gamma_and_inverse_gaussian_match_converged_r() {
    let f = NegativeBinomialRegressor::with_theta(NEGBINOM_THETA)
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&xm(&X_NEGBINOM), &yc(&Y_NEGBINOM))
        .unwrap();
    check(&f, f.deviance, &NEGBINOM, TOL, "negative binomial");
    let est = NegativeBinomialRegressor::builder()
        .estimate_theta(true)
        .tolerance(1e-13)
        .theta_tolerance(1e-14)
        .build()
        .fit(&xm(&X_NEGBINOM), &yc(&Y_NEGBINOM))
        .unwrap();
    assert!(
        (est.theta - NEGBINOM_THETA).abs() < 1e-6 * NEGBINOM_THETA,
        "theta {} vs {}",
        est.theta,
        NEGBINOM_THETA
    );
    let f = TweedieRegressor::gamma()
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&xm(&X_GAMMA_LOG), &yc(&Y_GAMMA_LOG))
        .unwrap();
    check(&f, f.deviance, &GAMMA_LOG, TOL, "gamma log");
    let f = TweedieRegressor::inverse_gaussian()
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&xm(&X_INVGAUSS_LOG), &yc(&Y_INVGAUSS_LOG))
        .unwrap();
    check(&f, f.deviance, &INVGAUSS_LOG, TOL, "inverse gaussian log");
}
