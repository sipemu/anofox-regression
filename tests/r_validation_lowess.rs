//! `solvers::lowess::lowess` validation against R's `stats::lowess`.
//!
//! Reference values come from `tests/r_scripts/generate_lowess_validation.R`.

use anofox_regression::solvers::lowess::{lowess, LowessOptions};

include!("fixtures/lowess_validation.rs");

fn check(
    x: &[f64],
    y: &[f64],
    frac: f64,
    iterations: usize,
    delta: Option<f64>,
    ex: &[f64],
    ey: &[f64],
) {
    let fit = lowess(
        x,
        y,
        &LowessOptions {
            frac,
            iterations,
            delta,
        },
    )
    .unwrap();
    assert_eq!(fit.x.len(), ex.len());
    for k in 0..ex.len() {
        assert_eq!(fit.x[k], ex[k], "sorted x[{k}]");
        assert!(
            (fit.fitted[k] - ey[k]).abs() <= 1e-10 * ey[k].abs().max(1.0),
            "fitted[{k}] = {} vs R {}",
            fit.fitted[k],
            ey[k]
        );
        assert_eq!(x[fit.order[k]], fit.x[k], "order[{k}]");
    }
}

#[test]
fn lowess_matches_r_defaults() {
    check(
        &X_SINE,
        &Y_SINE,
        F_SINE,
        ITER_SINE,
        DELTA_SINE,
        &EXPECTED_X_SINE,
        &EXPECTED_Y_SINE,
    );
}

#[test]
fn lowess_matches_r_small_span_no_robustness() {
    check(
        &X_SINE_F02,
        &Y_SINE_F02,
        F_SINE_F02,
        ITER_SINE_F02,
        DELTA_SINE_F02,
        &EXPECTED_X_SINE_F02,
        &EXPECTED_Y_SINE_F02,
    );
}

#[test]
fn lowess_matches_r_with_outliers() {
    check(
        &X_OUTLIER,
        &Y_OUTLIER,
        F_OUTLIER,
        ITER_OUTLIER,
        DELTA_OUTLIER,
        &EXPECTED_X_OUTLIER,
        &EXPECTED_Y_OUTLIER,
    );
}

#[test]
fn lowess_matches_r_with_ties() {
    check(
        &X_TIES,
        &Y_TIES,
        F_TIES,
        ITER_TIES,
        DELTA_TIES,
        &EXPECTED_X_TIES,
        &EXPECTED_Y_TIES,
    );
}

#[test]
fn lowess_matches_r_with_delta_interpolation() {
    check(
        &X_DELTA,
        &Y_DELTA,
        F_DELTA,
        ITER_DELTA,
        DELTA_DELTA,
        &EXPECTED_X_DELTA,
        &EXPECTED_Y_DELTA,
    );
}

#[test]
fn lowess_matches_r_on_three_points() {
    check(
        &X_TINY,
        &Y_TINY,
        F_TINY,
        ITER_TINY,
        DELTA_TINY,
        &EXPECTED_X_TINY,
        &EXPECTED_Y_TINY,
    );
}

#[test]
fn lowess_single_point_and_errors() {
    let fit = lowess(&[2.0], &[5.0], &LowessOptions::default()).unwrap();
    assert_eq!(fit.fitted, vec![5.0]);
    assert!(lowess(&[], &[], &LowessOptions::default()).is_err());
    assert!(lowess(&[1.0, 2.0], &[1.0], &LowessOptions::default()).is_err());
    assert!(lowess(&[1.0, f64::NAN], &[1.0, 2.0], &LowessOptions::default()).is_err());
    let bad_frac = LowessOptions {
        frac: 0.0,
        ..LowessOptions::default()
    };
    assert!(lowess(&[1.0, 2.0], &[1.0, 2.0], &bad_frac).is_err());
    let bad_delta = LowessOptions {
        delta: Some(-1.0),
        ..LowessOptions::default()
    };
    assert!(lowess(&[1.0, 2.0], &[1.0, 2.0], &bad_delta).is_err());
}
