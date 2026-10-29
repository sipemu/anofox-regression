#![allow(clippy::excessive_precision, dead_code)]

//! #103: optimism-corrected validation and calibration against
//! `rms::validate` and `rms::calibrate`, replaying R's resamples.
//!
//! Reference values come from `tests/r_scripts/generate_validate_validation.R`.

use anofox_regression::validation::{
    calibrate_lrm, calibrate_ols, validate_lrm, validate_ols, Resamples, ValidationTable,
};
use faer::Mat;

include!("fixtures/validate_validation.rs");

const B: usize = 30;

fn x() -> Mat<f64> {
    Mat::from_fn(X1.len(), 3, |i, j| [X1[i], X2[i], X3[i]][j])
}

fn samples(idx: &[usize]) -> Vec<Vec<usize>> {
    let n = X1.len();
    idx.chunks(n).map(|c| c.to_vec()).collect()
}

fn close(name: &str, a: f64, b: f64, tol: f64) {
    let ok = (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs());
    assert!(ok, "{name}: {a} vs {b}");
}

/// R's table is (index × [orig, training, test, optimism, corrected, n]),
/// emitted row-major.
fn check(id: &str, t: &ValidationTable, want: &[f64], tol: f64) {
    for (k, name) in t.names.iter().enumerate() {
        let row = &want[k * 6..k * 6 + 6];
        let (o, tr, te, op, co, n) = t.get(name).unwrap();
        close(&format!("{id} {name} orig"), o, row[0], tol);
        close(&format!("{id} {name} training"), tr, row[1], tol);
        close(&format!("{id} {name} test"), te, row[2], tol);
        close(&format!("{id} {name} optimism"), op, row[3], tol * 10.0);
        close(&format!("{id} {name} corrected"), co, row[4], tol);
        assert_eq!(n as f64, row[5], "{id} {name} n");
    }
}

#[test]
fn bootstrap_validation_matches_rms() {
    let n = X1.len();
    let r = Resamples::bootstrap(n, samples(&BOOT_IDX)).unwrap();
    check(
        "lrm boot",
        &validate_lrm(&x(), &YB, &r).unwrap(),
        &LRM_BOOT,
        1e-5,
    );
    check(
        "ols boot",
        &validate_ols(&x(), &YC, &r).unwrap(),
        &OLS_BOOT,
        1e-9,
    );
}

#[test]
fn crossvalidation_matches_rms() {
    let r = Resamples::crossvalidation(&CV_PERM, 10).unwrap();
    check(
        "lrm cv",
        &validate_lrm(&x(), &YB, &r).unwrap(),
        &LRM_CV,
        1e-5,
    );
    check(
        "ols cv",
        &validate_ols(&x(), &YC, &r).unwrap(),
        &OLS_CV,
        1e-9,
    );
}

#[test]
fn point632_matches_rms() {
    let n = X1.len();
    let r = Resamples::point632(n, samples(&P632_IDX)).unwrap();
    check(
        "lrm .632",
        &validate_lrm(&x(), &YB, &r).unwrap(),
        &LRM_632,
        1e-5,
    );
}

#[test]
fn calibration_matches_rms() {
    let n = X1.len();
    let r = Resamples::bootstrap(n, samples(&BOOT_IDX)).unwrap();
    for (id, cal, predy, orig, corr, opt, nn, err) in [
        (
            "lrm",
            calibrate_lrm(&x(), &YB, &r, None).unwrap(),
            &CAL_LRM_PREDY,
            &CAL_LRM_ORIG,
            &CAL_LRM_CORR,
            &CAL_LRM_OPT,
            &CAL_LRM_N,
            &CAL_LRM_ERR,
        ),
        (
            "ols",
            calibrate_ols(&x(), &YC, &r, None).unwrap(),
            &CAL_OLS_PREDY,
            &CAL_OLS_ORIG,
            &CAL_OLS_CORR,
            &CAL_OLS_OPT,
            &CAL_OLS_N,
            &CAL_OLS_ERR,
        ),
    ] {
        for k in 0..predy.len() {
            close(&format!("{id} predy[{k}]"), cal.predy[k], predy[k], 1e-8);
            close(
                &format!("{id} orig[{k}]"),
                cal.calibrated_orig[k],
                orig[k],
                1e-7,
            );
            close(
                &format!("{id} corrected[{k}]"),
                cal.calibrated_corrected[k],
                corr[k],
                1e-6,
            );
            close(
                &format!("{id} optimism[{k}]"),
                cal.optimism[k],
                opt[k],
                1e-5,
            );
            assert_eq!(cal.n[k] as f64, nn[k], "{id} n[{k}]");
        }
        close(
            &format!("{id} mean abs error"),
            cal.mean_abs_error,
            err[0],
            1e-6,
        );
        close(&format!("{id} mse"), cal.mean_squared_error, err[1], 1e-6);
        close(&format!("{id} q90"), cal.quantile90_abs_error, err[2], 1e-6);
    }
}

#[test]
fn seeded_resamples_run_end_to_end() {
    let n = X1.len();
    let r = Resamples::bootstrap_seeded(n, 20, 42).unwrap();
    let t = validate_lrm(&x(), &YB, &r).unwrap();
    let (orig, _, _, opt, corr, used) = t.get("Dxy").unwrap();
    assert_eq!(used, 20);
    assert!(opt > 0.0 && (orig - opt - corr).abs() < 1e-12);
    let r = Resamples::crossvalidation_seeded(n, 5, 1).unwrap();
    assert!(validate_ols(&x(), &YC, &r).is_ok());
}
