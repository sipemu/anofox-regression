#![allow(clippy::excessive_precision)]

//! #112: Box–Cox profile likelihood (`MASS::boxcox`) and Duan's smearing
//! estimator (`Hmisc::smearingEst`), RMS Ch 16. Reference values come from
//! `tests/r_scripts/generate_transform_validation.R`: `y = (2 + 0.6 x1 +
//! 1.2 x2 + ε)²`, so `λ ≈ 0.5`.

use anofox_regression::solvers::FittedRegressor;
use anofox_regression::solvers::{OlsRegressor, Regressor};
use anofox_regression::transform::{
    box_cox, box_cox_inverse, box_cox_transform, smearing_estimate, BoxCoxOptions, Smearing,
};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/transform_validation.rs");
}
use fixture::*;

const N: usize = 120;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *u, *v, tol);
    }
}

fn data() -> (Mat<f64>, Col<f64>) {
    (
        Mat::from_fn(N, 2, |i, j| X[i * 2 + j]),
        Col::from_fn(N, |i| Y[i]),
    )
}

#[test]
fn box_cox_profile_matches_mass() {
    let (x, y) = data();
    let prof = box_cox(&x, &y, &BoxCoxOptions::default()).unwrap();
    all_close("grid", &prof.log_likelihood, &BC_GRID_LOGLIK, 1e-12);
    close("lambda_hat", prof.lambda_hat, BC_MLE[0], 1e-7);
    close("max loglik", prof.max_log_likelihood, BC_MLE[1], 1e-12);
    close("ci lower", prof.ci_lower.unwrap(), BC_MLE[2], 1e-9);
    close("ci upper", prof.ci_upper.unwrap(), BC_MLE[3], 1e-9);
}

#[test]
fn box_cox_series_near_zero_and_without_intercept() {
    let (x, y) = data();
    let opts = BoxCoxOptions {
        lambdas: vec![-0.015, 0.0, 0.01],
        ..Default::default()
    };
    let prof = box_cox(&x, &y, &opts).unwrap();
    all_close("small lambda", &prof.log_likelihood, &BC_SMALL, 1e-12);

    let x1 = Mat::from_fn(N, 3, |i, j| if j < 2 { x[(i, j)] } else { 1.0 });
    let opts = BoxCoxOptions {
        lambdas: vec![0.3, 0.5],
        with_intercept: false,
        ..Default::default()
    };
    let prof = box_cox(&x1, &y, &opts).unwrap();
    all_close("no intercept", &prof.log_likelihood, &BC_NOINT_CHECK, 1e-12);
}

#[test]
fn box_cox_interval_open_at_grid_end() {
    let (x, y) = data();
    let opts = BoxCoxOptions {
        lambdas: vec![0.45, 0.5, 0.55],
        ..Default::default()
    };
    let prof = box_cox(&x, &y, &opts).unwrap();
    assert!(prof.ci_lower.is_none() && prof.ci_upper.is_none());
    close(
        "lambda_hat in a narrow grid",
        prof.lambda_hat,
        BC_MLE[0],
        1e-7,
    );
}

/// Fit `g(y) ~ x1 + x2` and return the first six linear predictors and all
/// residuals.
fn transformed_fit(g: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let (x, y) = data();
    let gy = Col::from_fn(N, |i| g(y[i]));
    let fit = OlsRegressor::builder()
        .with_intercept(true)
        .build()
        .fit(&x, &gy)
        .unwrap();
    let lp = fit.predict(&x);
    let lp6: Vec<f64> = (0..6).map(|i| lp[i]).collect();
    let r: Vec<f64> = (0..N).map(|i| gy[i] - lp[i]).collect();
    (lp6, r)
}

#[test]
fn smearing_matches_hmisc() {
    let (lp, r) = transformed_fit(f64::sqrt);
    all_close("lp", &lp, &SM_LP, 1e-10);
    let sq = |z: f64| z * z;
    let est = |s| smearing_estimate(&lp, &r, sq, s).unwrap();
    all_close("mean", &est(Smearing::Mean), &SM_MEAN, 1e-10);
    all_close("median", &est(Smearing::Median), &SM_MEDIAN, 1e-10);
    all_close("q90", &est(Smearing::Quantile(0.9)), &SM_Q90, 1e-10);
    all_close("fitted", &est(Smearing::Fitted), &SM_FITTED, 1e-10);

    let (lp, r) = transformed_fit(f64::ln);
    let m = smearing_estimate(&lp, &r, f64::exp, Smearing::Mean).unwrap();
    all_close("log mean", &m, &SM_LOG_MEAN, 1e-10);

    let lam = 0.3;
    let (lp, r) = transformed_fit(|v| box_cox_transform(v, lam));
    let m = smearing_estimate(&lp, &r, |z| box_cox_inverse(z, lam), Smearing::Mean).unwrap();
    all_close("box-cox mean", &m, &SM_BC_MEAN, 1e-10);
}

#[test]
fn transforms_round_trip_and_reject_bad_input() {
    for lam in [-1.0, 0.0, 0.3, 2.0] {
        close(
            "round trip",
            box_cox_inverse(box_cox_transform(3.7, lam), lam),
            3.7,
            1e-12,
        );
    }
    let (x, mut y) = data();
    assert!(smearing_estimate(&[1.0], &[], f64::exp, Smearing::Mean).is_err());
    assert!(smearing_estimate(&[1.0], &[0.1], f64::exp, Smearing::Quantile(1.5)).is_err());
    let empty = BoxCoxOptions {
        lambdas: vec![],
        ..Default::default()
    };
    assert!(box_cox(&x, &y, &empty).is_err());
    let bad_level = BoxCoxOptions {
        confidence_level: 1.0,
        ..Default::default()
    };
    assert!(box_cox(&x, &y, &bad_level).is_err());
    let short = Col::from_fn(N - 1, |i| y[i]);
    assert!(box_cox(&x, &short, &BoxCoxOptions::default()).is_err());
    let tiny = Mat::from_fn(3, 2, |i, j| x[(i, j)]);
    let y3 = Col::from_fn(3, |i| y[i]);
    assert!(box_cox(&tiny, &y3, &BoxCoxOptions::default()).is_err());
    y[0] = -1.0;
    assert!(box_cox(&x, &y, &BoxCoxOptions::default()).is_err());
}
