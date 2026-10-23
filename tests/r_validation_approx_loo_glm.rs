//! R validation of the one-step approximate leave-one-out fields of the GLM
//! augment, `Augment::approx_loo_fitted` / `approx_loo_residual`
//! (anofox-statistics#235).
//!
//! Reference values: `tests/r_scripts/generate_approx_loo_glm_validation.R`
//! (`influence(fit)$coefficients`, the one-step coefficient change, applied to
//! the linear predictor), with the exact refits for comparison.

#[path = "fixtures/approx_loo_glm_validation.rs"]
mod refs;

use anofox_regression::diagnostics::{augment, augment_glm, Augment};
use anofox_regression::prelude::*;
use anofox_regression::solvers::penalized_glm::{fit_poisson, PoissonOptions};
use faer::{Col, Mat};
use refs::*;

const TOL: f64 = 1e-6;

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn design(cols: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(cols[0].len(), cols.len(), |i, j| cols[j][i])
}

fn check(label: &str, a: &Augment, fitted: &[f64], approx: &[f64], resid: &[f64], refit: &[f64]) {
    for i in 0..fitted.len() {
        let close = |got: f64, want: f64| (got - want).abs() <= TOL * want.abs().max(1.0);
        assert!(close(a.fitted[i], fitted[i]), "{label} fitted[{i}]");
        assert!(
            close(a.approx_loo_fitted[i], approx[i]),
            "{label} approx_loo_fitted[{i}]: got {:.12e}, R {:.12e}",
            a.approx_loo_fitted[i],
            approx[i]
        );
        // The residual is a difference of two numbers on the scale of y.
        assert!(
            (a.approx_loo_residual[i] - resid[i]).abs() <= TOL * approx[i].abs().max(1.0),
            "{label} approx_loo_residual[{i}]"
        );
        assert!(
            a.loo_residual[i].is_nan(),
            "{label}: no exact LOO for a GLM"
        );
    }
    // An approximation of the refit: much closer to it than the full fit is.
    let err = |v: &dyn Fn(usize) -> f64| -> f64 {
        (0..refit.len())
            .map(|i| (v(i) - refit[i]).abs())
            .sum::<f64>()
    };
    let approx_err = err(&|i| a.approx_loo_fitted[i]);
    let full_err = err(&|i| a.fitted[i]);
    eprintln!("{label}: Σ|approx - refit| = {approx_err:.3e}, Σ|full - refit| = {full_err:.3e}");
    assert!(
        approx_err < 0.25 * full_err,
        "{label}: approximation too far from the refits"
    );
}

#[test]
fn poisson_approx_loo_matches_r() {
    let x = design(&[&MT_WT, &MT_HP]);
    let y = col(&MT_CARB);
    let fit = PoissonRegressor::log().build().fit(&x, &y).unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    check(
        "Poisson",
        &a,
        &POIS_FITTED,
        &POIS_APPROX_LOO_FITTED,
        &POIS_APPROX_LOO_RESID,
        &POIS_REFIT_LOO_FITTED,
    );
}

#[test]
fn penalized_engine_poisson_approx_loo_matches_r() {
    let x = vec![MT_WT.to_vec(), MT_HP.to_vec()];
    let opts = PoissonOptions {
        max_iterations: 200,
        tolerance: 1e-15,
        ..Default::default()
    };
    let fit = fit_poisson(&MT_CARB, &x, &opts).unwrap();
    let a = augment_glm(&fit, &MT_CARB, &x).unwrap();
    check(
        "engine Poisson",
        &a,
        &POIS_FITTED,
        &POIS_APPROX_LOO_FITTED,
        &POIS_APPROX_LOO_RESID,
        &POIS_REFIT_LOO_FITTED,
    );
}

#[test]
fn logistic_approx_loo_matches_r() {
    let x = design(&[&MT_MPG, &MT_WT]);
    let y = col(&MT_VS);
    let fit = BinomialRegressor::logistic().build().fit(&x, &y).unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    check(
        "Logistic",
        &a,
        &LOGIT_FITTED,
        &LOGIT_APPROX_LOO_FITTED,
        &LOGIT_APPROX_LOO_RESID,
        &LOGIT_REFIT_LOO_FITTED,
    );
}

#[test]
fn gamma_approx_loo_matches_r() {
    let x = design(&[&MT_WT, &MT_HP]);
    let y = col(&MT_MPG);
    let fit = GammaRegressor::builder().build().fit(&x, &y).unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    check(
        "Gamma",
        &a,
        &GAMMA_FITTED,
        &GAMMA_APPROX_LOO_FITTED,
        &GAMMA_APPROX_LOO_RESID,
        &GAMMA_REFIT_LOO_FITTED,
    );
}

#[test]
fn linear_models_have_no_approx_loo() {
    let x = design(&[&MT_WT, &MT_HP]);
    let y = col(&MT_MPG);
    let fit = OlsRegressor::builder().build().fit(&x, &y).unwrap();
    let a = augment(&fit, &x, &y, None).unwrap();
    assert!(a.approx_loo_fitted.iter().all(|v| v.is_nan()));
    assert!(a.approx_loo_residual.iter().all(|v| v.is_nan()));
    assert!(a.loo_residual.iter().all(|v| v.is_finite()));
}
