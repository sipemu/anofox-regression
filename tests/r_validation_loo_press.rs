//! R validation of the leave-one-out (PRESS) diagnostics: `Augment::loo_residual`,
//! `Augment::loo_fitted` and `diagnostics::press` (anofox-statistics#220).
//!
//! Reference values: `tests/r_scripts/generate_loo_press_validation.R`
//! (`residuals(fit) / (1 - hatvalues(fit))`, `rstandard(fit, type = "predictive")`,
//! each checked against an explicit refit without the row).

#[path = "fixtures/loo_press_validation.rs"]
mod refs;

use anofox_regression::diagnostics::{augment, press};
use anofox_regression::prelude::*;
use faer::{Col, Mat};
use refs::*;

const TOL: f64 = 1e-8;

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn design(cols: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(cols[0].len(), cols.len(), |i, j| cols[j][i])
}

fn close(label: &str, got: f64, want: f64) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() <= TOL * scale,
        "{label}: got {got:.12e}, R {want:.12e}"
    );
}

fn close_all(label: &str, got: &Col<f64>, want: &[f64]) {
    assert_eq!(got.nrows(), want.len(), "{label}: length");
    for (i, &w) in want.iter().enumerate() {
        close(&format!("{label}[{i}]"), got[i], w);
    }
}

fn mtcars() -> (Mat<f64>, Col<f64>) {
    (design(&[&MT_WT, &MT_HP]), col(&MT_MPG))
}

#[test]
fn ols_loo_matches_r() {
    let (x, y) = mtcars();
    let fit = OlsRegressor::builder().build().fit(&x, &y).unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    close_all("OLS loo_residual", &a.loo_residual, &OLS_LOO_RESID);
    close_all("OLS loo_fitted", &a.loo_fitted, &OLS_LOO_FITTED);
    let p = press(&fit, &x, &y, None).unwrap();
    assert_eq!(p.n, 32);
    close("OLS press", p.press, OLS_PRESS);
    close("OLS press_rmse", p.press_rmse, OLS_PRESS_RMSE);
    close("OLS loo_r_squared", p.loo_r_squared, OLS_LOO_R2);
    assert!(p.loo_r_squared < fit.result().r_squared);
}

#[test]
fn ols_without_intercept_uses_uncentred_sst() {
    let (x, y) = mtcars();
    let fit = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    close_all("OLS0 loo_residual", &a.loo_residual, &OLS0_LOO_RESID);
    close_all("OLS0 loo_fitted", &a.loo_fitted, &OLS0_LOO_FITTED);
    let p = press(&fit, &x, &y, None).unwrap();
    close("OLS0 press", p.press, OLS0_PRESS);
    close("OLS0 press_rmse", p.press_rmse, OLS0_PRESS_RMSE);
    close("OLS0 loo_r_squared", p.loo_r_squared, OLS0_LOO_R2);
}

#[test]
fn wls_loo_is_response_scale_and_press_is_weighted() {
    let (x, y) = mtcars();
    let w = col(&MT_W);
    let fit = WlsRegressor::builder()
        .weights(w.clone())
        .build()
        .fit(&x, &y)
        .unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    close_all("WLS loo_residual", &a.loo_residual, &WLS_LOO_RESID);
    close_all("WLS loo_fitted", &a.loo_fitted, &WLS_LOO_FITTED);
    // The fit's own weights are used when none are passed.
    let p = press(&fit, &x, &y, None).unwrap();
    close("WLS press", p.press, WLS_PRESS);
    close("WLS press_rmse", p.press_rmse, WLS_PRESS_RMSE);
    close("WLS loo_r_squared", p.loo_r_squared, WLS_LOO_R2);
    assert_eq!(p, press(&fit, &x, &y, Some(&w)).unwrap());
}

#[test]
fn leverage_one_row_has_no_loo_prediction() {
    let x = design(&[&H1_X, &H1_D]);
    let y = col(&H1_Y);
    let fit = OlsRegressor::builder().build().fit(&x, &y).unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    for (i, &want) in H1_LOO_RESID.iter().enumerate().take(7) {
        close(&format!("H1 loo_residual[{i}]"), a.loo_residual[i], want);
    }
    assert!(a.loo_residual[7].is_nan() && a.loo_fitted[7].is_nan());
    let p = press(&fit, &x, &y, None).unwrap();
    assert!(p.press.is_nan() && p.press_rmse.is_nan() && p.loo_r_squared.is_nan());
}

#[test]
fn zero_residual_df_gives_nan() {
    // Three rows, three parameters: every row has leverage 1.
    let x = design(&[&[1.0, 2.0, 4.0], &[3.0, 1.0, 2.0]]);
    let y = col(&[1.0, 5.0, 2.0]);
    let fit = OlsRegressor::builder().build().fit(&x, &y).unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    assert!(a.loo_residual.iter().all(|v| v.is_nan()));
    assert!(press(&fit, &x, &y, None).unwrap().press.is_nan());
}

#[test]
fn glm_has_no_loo_statistics() {
    let x = design(&[&MT_WT]);
    let y = col(&MT_MPG);
    let fit = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    assert!(a.loo_residual.iter().all(|v| v.is_nan()));
    assert!(press(&fit, &x, &y, None).unwrap().press.is_nan());
}
