//! #195: WLS with zero weights counts only the positive-weight rows for the
//! number of observations and the residual degrees of freedom, as `lm`.
//!
//! Reference values: `tests/r_scripts/generate_wls_zero_weights_validation.R`.

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{FittedRegressor, Regressor, WlsRegressor};
use faer::{Col, Mat};

const X1: [f64; N] = [
    0.505,
    0.0030000000000000001,
    2.9689999999999999,
    4.2050000000000001,
    2.7090000000000001,
    1.8029999999999999,
    0.55800000000000005,
    1.1919999999999999,
    2.3959999999999999,
    3.5390000000000001,
    4.343,
    4.3970000000000002,
];
const X2: [f64; N] = [
    1.139,
    -0.94899999999999995,
    0.32000000000000001,
    -1.77,
    0.17199999999999999,
    0.46400000000000002,
    0.53300000000000003,
    -0.66500000000000004,
    -0.88400000000000001,
    0.45700000000000002,
    0.47199999999999998,
    -0.057000000000000002,
];
const W: [f64; N] = [
    1.9330000000000001,
    1.4319999999999999,
    0.0,
    1.7949999999999999,
    1.7390000000000001,
    1.0580000000000001,
    0.53000000000000003,
    1.7529999999999999,
    0.0,
    1.4119999999999999,
    1.468,
    1.879,
];
const Y: [f64; N] = [
    0.84999999999999998,
    1.071,
    3.4710000000000001,
    5.2930000000000001,
    3.0670000000000002,
    1.9119999999999999,
    0.90400000000000003,
    2.7149999999999999,
    2.9969999999999999,
    3.375,
    3.673,
    5.0069999999999997,
];
const COEF: [f64; 3] = [
    0.9304832934189583,
    0.81231775855685573,
    -0.57714170789621788,
];
const SE: [f64; 3] = [
    0.22021026404396782,
    0.07492116116821422,
    0.14307399324418615,
];
const T: [f64; 3] = [4.225431078149815, 10.842300705044154, -4.0338687333008378];
const P: [f64; 3] = [
    0.0039101958564312965,
    1.2523570307018635e-05,
    0.0049713487270954537,
];
const CI_LO: [f64; 3] = [
    0.40976876261094142,
    0.63515736390100486,
    -0.91545794209364262,
];
const CI_HI: [f64; 3] = [1.4511978242269752, 0.9894781532127066, -0.23882547369879309];
const SIGMA: f64 = 0.46942738414902607;
const R2: f64 = 0.95698831172766929;
const F: f64 = 77.873229942511657;
const NEW_X1: [f64; 3] = [1.0, 2.5, 4.0];
const NEW_X2: [f64; 3] = [-0.5, 0.0, 1.0];
const CONF_LO: [f64; 3] = [1.6013738201104748, 2.673299115356695, 3.0322600174742611];
const CONF_HI: [f64; 3] = [2.4613699917373713, 3.2492562642655001, 4.1729652220260656];
const PRED_LO: [f64; 3] = [0.84097643762680541, 1.8145106356610952, 2.3546362144996564];
const PRED_HI: [f64; 3] = [3.2217673742210406, 4.1080447439611003, 4.8505890250006702];
const NOBS: usize = 10;
const DF_RESID: usize = 7;

const N: usize = 12;

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() < 1e-9 * (1.0 + b.abs()),
        "{what}: {a} vs R {b}"
    );
}

#[test]
fn zero_weights_do_not_count_as_observations() {
    let x = Mat::from_fn(N, 2, |i, j| if j == 0 { X1[i] } else { X2[i] });
    let y = Col::from_fn(N, |i| Y[i]);
    let fit = WlsRegressor::builder()
        .weights(Col::from_fn(N, |i| W[i]))
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    assert_eq!(r.n_observations, NOBS);
    assert_eq!(r.residual_df(), DF_RESID);
    close(r.mse.sqrt(), SIGMA, "sigma");
    close(r.r_squared, R2, "R²");
    close(r.f_statistic, F, "F");

    let mut coef = vec![r.intercept.unwrap()];
    coef.extend(r.coefficients.iter());
    let mut se = vec![r.intercept_std_error.unwrap()];
    se.extend(r.std_errors.as_ref().unwrap().iter());
    let mut t = vec![r.intercept_t_statistic.unwrap()];
    t.extend(r.t_statistics.as_ref().unwrap().iter());
    let mut p = vec![r.intercept_p_value.unwrap()];
    p.extend(r.p_values.as_ref().unwrap().iter());
    let mut lo = vec![r.intercept_conf_interval.unwrap().0];
    lo.extend(r.conf_interval_lower.as_ref().unwrap().iter());
    let mut hi = vec![r.intercept_conf_interval.unwrap().1];
    hi.extend(r.conf_interval_upper.as_ref().unwrap().iter());
    for k in 0..3 {
        close(coef[k], COEF[k], "coef");
        close(se[k], SE[k], "se");
        close(t[k], T[k], "t");
        close(p[k], P[k], "p");
        close(lo[k], CI_LO[k], "ci lower");
        close(hi[k], CI_HI[k], "ci upper");
    }

    let nd = Mat::from_fn(3, 2, |i, j| if j == 0 { NEW_X1[i] } else { NEW_X2[i] });
    let conf = fit.predict_with_interval(&nd, Some(IntervalType::Confidence), 0.95);
    let pred = fit.predict_with_interval(&nd, Some(IntervalType::Prediction), 0.95);
    for i in 0..3 {
        close(conf.lower[i], CONF_LO[i], "confidence lower");
        close(conf.upper[i], CONF_HI[i], "confidence upper");
        close(pred.lower[i], PRED_LO[i], "prediction lower");
        close(pred.upper[i], PRED_HI[i], "prediction upper");
    }
}
