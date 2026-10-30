#![allow(clippy::excessive_precision)]

//! #67: GLMMs with lme4's default Laplace approximation (`nAGQ = 1`: fixed
//! effects optimised jointly with θ, Hessian-based standard errors) and
//! adaptive Gauss–Hermite quadrature (`nAGQ = 7`), against `lme4::glmer`.
//! Reference values come from
//! `tests/r_scripts/generate_glmm_nagq_validation.R`.
//!
//! lme4 evaluates `log|L|²` with the Cholesky factor of the previous PIRLS
//! step, not at the final conditional mode (at lme4's own optimum of the
//! Poisson fit its `ldL2` is 45.97187 while the exact value from its fitted
//! means is 45.97122). Our criterion is the exact Laplace approximation, so
//! its optimum moves by a fraction of a standard error: fixed effects are
//! compared within 1% of their SE, and our maximised log-likelihood must be
//! at least lme4's and within 1e-5 of it. lme4's standard errors come from
//! a finite-difference Hessian of that same criterion, which the stale
//! factor also perturbs: for the Poisson Laplace fits they differ from ours
//! by about 3% (ours agree with lme4's own AGQ Hessian, 0.05349 vs 0.05348
//! for `x2`), so those are compared at 5%; binomial and AGQ at 1e-3.
//!
//! lme4 2.0.1 reports `logLik = −deviance/2` for `nAGQ > 1` without the
//! saturated term it includes for `nAGQ = 1`; our AGQ `deviance` equals
//! lme4's criterion and our `log_likelihood` adds the saturated term, as
//! lme4's development version does.

use anofox_regression::solvers::{FittedGlmm, GlmmRegressor, GlmmRegressorBuilder};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/glmm_nagq_validation.rs");
}
use fixture::*;

const N: usize = 200;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *u, *v, tol);
    }
}

fn fit(builder: GlmmRegressorBuilder, y: &[f64]) -> FittedGlmm {
    let x = Mat::from_fn(N, 2, |i, j| X[i * 2 + j]);
    let g: Vec<usize> = G.iter().map(|&v| v as usize).collect();
    builder
        .build()
        .fit(&x, &Col::from_fn(N, |i| y[i]), &g)
        .unwrap()
}

fn poisson_saturated() -> f64 {
    YP.iter()
        .map(|&y| {
            if y > 0.0 {
                y * y.ln() - y - (1..=y as u64).map(|k| (k as f64).ln()).sum::<f64>()
            } else {
                0.0
            }
        })
        .sum()
}

/// Fixed effects (within 1% of their SE), vcov (at `tol_v`), θ (√ of the
/// random-intercept variance) and the conditional modes.
fn check(
    tag: &str,
    f: &FittedGlmm,
    fixef: &[f64],
    vcov: &[f64],
    theta: &[f64],
    ranef: &[f64],
    tol_v: f64,
) {
    let p = fixef.len();
    for (j, (a, b)) in f.fixed_effects().iter().zip(fixef).enumerate() {
        let se = vcov[j * p + j].sqrt();
        assert!(
            (a - b).abs() <= 0.01 * se,
            "{tag} fixef[{j}]: {a} vs {b} (se {se})"
        );
    }
    let v: Vec<f64> = (0..p * p).map(|k| f.vcov()[k % p][k / p]).collect();
    all_close(&format!("{tag} vcov"), &v, vcov, tol_v);
    if theta.len() == 1 {
        close(&format!("{tag} theta"), f.random_sd()[0], theta[0], 1e-3);
    }
    let re: Vec<f64> = f
        .random_effects_matrix()
        .iter()
        .flatten()
        .copied()
        .collect();
    all_close(&format!("{tag} ranef"), &re, ranef, 1e-3);
}

/// Our maximised log-likelihood is at least lme4's, and close to it.
fn check_loglik(tag: &str, ours: f64, lme4: f64) {
    assert!(
        ours >= lme4 - 1e-6,
        "{tag} logLik: {ours} below lme4's {lme4}"
    );
    close(&format!("{tag} logLik"), ours, lme4, 1e-5);
}

#[test]
fn laplace_poisson_matches_glmer() {
    let f = fit(GlmmRegressor::poisson().n_agq(1), &YP);
    check("P1", &f, &P1_FIXEF, &P1_VCOV, &P1_THETA, &P1_RANEF, 5e-2);
    check_loglik("P1", f.log_likelihood(), P1_LOGLIK[0]);
}

#[test]
fn laplace_binomial_matches_glmer() {
    let f = fit(GlmmRegressor::binomial().n_agq(1), &YB);
    check("B1", &f, &B1_FIXEF, &B1_VCOV, &B1_THETA, &B1_RANEF, 1e-3);
    check_loglik("B1", f.log_likelihood(), B1_LOGLIK[0]);
}

#[test]
fn laplace_random_slope_matches_glmer() {
    let f = fit(
        GlmmRegressor::poisson().random_slopes(vec![0]).n_agq(1),
        &YP,
    );
    check(
        "PS1", &f, &PS1_FIXEF, &PS1_VCOV, &PS1_THETA, &PS1_RANEF, 5e-2,
    );
    check_loglik("PS1", f.log_likelihood(), PS1_LOGLIK[0]);
}

#[test]
fn adaptive_quadrature_matches_glmer() {
    let f = fit(GlmmRegressor::poisson().n_agq(7), &YP);
    check("P7", &f, &P7_FIXEF, &P7_VCOV, &P7_THETA, &P7_RANEF, 1e-3);
    close("P7 deviance", f.deviance(), -2.0 * P7_LOGLIK[0], 1e-8);
    check_loglik("P7", f.log_likelihood(), P7_LOGLIK[0] + poisson_saturated());

    let f = fit(GlmmRegressor::binomial().n_agq(7), &YB);
    check("B7", &f, &B7_FIXEF, &B7_VCOV, &B7_THETA, &B7_RANEF, 1e-3);
    check_loglik("B7", f.log_likelihood(), B7_LOGLIK[0]);
}

#[test]
fn quadrature_needs_a_scalar_random_intercept() {
    let x = Mat::from_fn(N, 2, |i, j| X[i * 2 + j]);
    let g: Vec<usize> = G.iter().map(|&v| v as usize).collect();
    let y = Col::from_fn(N, |i| YP[i]);
    let slopes = GlmmRegressor::poisson()
        .random_slopes(vec![0])
        .n_agq(5)
        .build();
    assert!(slopes.fit(&x, &y, &g).is_err());
    let g2: Vec<usize> = (0..N).map(|i| i % 7).collect();
    let crossed = GlmmRegressor::poisson().n_agq(1).build();
    assert!(crossed.fit_crossed(&x, &y, &[&g, &g2]).is_err());
}
