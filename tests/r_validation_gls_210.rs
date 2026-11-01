#![allow(clippy::excessive_precision)]

//! #210: `nlme::gls` fits not covered by `r_validation_gls.rs`, on the same
//! data — AR(1) REML, continuous AR(1) ML, `corExp` without nugget,
//! `corGaus` / `corLin` / `corSpher` with nugget, `varIdent` alone, no
//! intercept, and `predict_with_se`.
//!
//! Reference values: `tests/r_scripts/generate_gls_validation_210.R`
//! (`glsControl(tolerance = 1e-12, msTol = 1e-14)`).

use anofox_regression::solvers::{FittedGls, GlsCorrelation, GlsMethod, GlsRegressor, GlsVariance};
use faer::{Col, Mat};

// Shares the data of the original GLS validation; its references are
// checked in r_validation_gls.rs.
#[allow(dead_code)]
mod fixture {
    include!("fixtures/gls_validation.rs");
}
mod fixture_210 {
    include!("fixtures/gls_validation_210.rs");
}
use fixture::{ID, TIME, VISIT, X, Y};
use fixture_210::*;

const N: usize = 164;
const P: usize = 4;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *u, *v, tol);
    }
}

fn fit(
    cor: GlsCorrelation,
    var: GlsVariance,
    method: GlsMethod,
    time: Option<&[f64]>,
    intercept: bool,
) -> FittedGls {
    let x = Mat::from_fn(N, P, |i, j| X[i * P + j]);
    let y = Col::from_fn(N, |i| Y[i]);
    let id: Vec<usize> = ID.iter().map(|&v| v as usize).collect();
    GlsRegressor::builder()
        .correlation(cor)
        .variance(var)
        .method(method)
        .with_intercept(intercept)
        .build()
        .fit(&x, &y, &id, time)
        .unwrap()
}

/// Coefficients, vcov, σ, logLik / AIC / BIC and the natural parameters.
#[allow(clippy::too_many_arguments)]
fn check(
    tag: &str,
    f: &FittedGls,
    coef: &[f64],
    vcov: &[f64],
    sigma: &[f64],
    ll: &[f64],
    cor: &[f64],
    var: &[f64],
    tol: f64,
) {
    let b: Vec<f64> = f.coefficients().iter().copied().collect();
    let p = b.len();
    all_close(&format!("{tag} coef"), &b, coef, tol);
    let v: Vec<f64> = (0..p * p).map(|k| f.vcov()[(k / p, k % p)]).collect();
    all_close(&format!("{tag} vcov"), &v, vcov, tol);
    close(&format!("{tag} sigma"), f.sigma(), sigma[0], tol);
    close(&format!("{tag} logLik"), f.log_likelihood(), ll[0], 1e-7);
    close(&format!("{tag} AIC"), f.aic(), ll[1], 1e-7);
    close(&format!("{tag} BIC"), f.bic(), ll[2], 1e-7);
    all_close(
        &format!("{tag} cor"),
        f.correlation_params(),
        cor,
        10.0 * tol,
    );
    all_close(&format!("{tag} var"), f.variance_params(), var, 10.0 * tol);
}

#[test]
fn ar1_reml_and_continuous_ar1_ml_match_nlme() {
    let visit: Vec<f64> = VISIT.to_vec();
    let f = fit(
        GlsCorrelation::Ar1,
        GlsVariance::Constant,
        GlsMethod::Reml,
        Some(&visit),
        true,
    );
    check(
        "AR1 REML",
        &f,
        &AR1_REML_COEF,
        &AR1_REML_VCOV,
        &AR1_REML_SIGMA,
        &AR1_REML_LOGLIK,
        &AR1_REML_COR,
        &[],
        1e-7,
    );
    // The ML optimum is flat: the crate's logLik is 5e-11 above nlme's while
    // φ differs by 7e-7, hence 1e-6.
    let f = fit(
        GlsCorrelation::ContinuousAr1,
        GlsVariance::Constant,
        GlsMethod::Ml,
        Some(&TIME),
        true,
    );
    check(
        "CAR1 ML",
        &f,
        &CAR1_ML_COEF,
        &CAR1_ML_VCOV,
        &CAR1_ML_SIGMA,
        &CAR1_ML_LOGLIK,
        &CAR1_ML_COR,
        &[],
        1e-6,
    );
}

#[test]
fn exponential_without_nugget_matches_nlme() {
    let f = fit(
        GlsCorrelation::Exponential { nugget: false },
        GlsVariance::Constant,
        GlsMethod::Reml,
        Some(&TIME),
        true,
    );
    check(
        "EXP",
        &f,
        &EXP_COEF,
        &EXP_VCOV,
        &EXP_SIGMA,
        &EXP_LOGLIK,
        &EXP_COR,
        &[],
        1e-6,
    );
}

#[test]
fn spatial_structures_with_nugget_match_nlme() {
    // The range is weakly identified (≈ 36 – 54 for times within 0.7 – 5.3):
    // compare the likelihood tightly, estimates and parameters loosely. For
    // corGaus L-BFGS alone stepped onto the range → ∞ plateau (logLik 1.4e-3
    // below nlme's maximum); `minimise` now also tries Nelder–Mead.
    let cases = [
        (
            GlsCorrelation::Gaussian { nugget: true },
            &GAUS_NUG_COEF,
            &GAUS_NUG_VCOV,
            &GAUS_NUG_SIGMA,
            &GAUS_NUG_LOGLIK,
            &GAUS_NUG_COR,
        ),
        (
            GlsCorrelation::Linear { nugget: true },
            &LIN_NUG_COEF,
            &LIN_NUG_VCOV,
            &LIN_NUG_SIGMA,
            &LIN_NUG_LOGLIK,
            &LIN_NUG_COR,
        ),
        (
            GlsCorrelation::Spherical { nugget: true },
            &SPHER_NUG_COEF,
            &SPHER_NUG_VCOV,
            &SPHER_NUG_SIGMA,
            &SPHER_NUG_LOGLIK,
            &SPHER_NUG_COR,
        ),
    ];
    for (cor, coef, vcov, sigma, ll, par) in cases {
        let tag = format!("{cor:?}");
        let f = fit(
            cor,
            GlsVariance::Constant,
            GlsMethod::Reml,
            Some(&TIME),
            true,
        );
        close(&format!("{tag} logLik"), f.log_likelihood(), ll[0], 1e-7);
        let b: Vec<f64> = f.coefficients().iter().copied().collect();
        all_close(&format!("{tag} coef"), &b, coef, 1e-3);
        let p = b.len();
        let v: Vec<f64> = (0..p * p).map(|k| f.vcov()[(k / p, k % p)]).collect();
        all_close(&format!("{tag} vcov"), &v, vcov, 1e-3);
        close(&format!("{tag} sigma"), f.sigma(), sigma[0], 1e-3);
        all_close(&format!("{tag} cor"), f.correlation_params(), par, 5e-2);
    }
}

#[test]
fn var_ident_alone_matches_nlme() {
    let strata: Vec<usize> = VISIT.iter().map(|&v| v as usize).collect();
    let f = fit(
        GlsCorrelation::Independent,
        GlsVariance::Ident(strata),
        GlsMethod::Reml,
        None,
        true,
    );
    check(
        "IDENT",
        &f,
        &IDENT_COEF,
        &IDENT_VCOV,
        &IDENT_SIGMA,
        &IDENT_LOGLIK,
        &[],
        &IDENT_VAR,
        // nlme stops ~1e-5 short in the variance ratios (the crate's logLik
        // is 5e-9 higher), as for compound symmetry in r_validation_gls.rs
        1e-4,
    );
}

fn new_x() -> Mat<f64> {
    Mat::from_fn(3, P, |i, j| NEW_X[i * P + j])
}

#[test]
fn no_intercept_matches_nlme() {
    let f = fit(
        GlsCorrelation::ContinuousAr1,
        GlsVariance::Constant,
        GlsMethod::Reml,
        Some(&TIME),
        false,
    );
    check(
        "NOINT",
        &f,
        &NOINT_COEF,
        &NOINT_VCOV,
        &NOINT_SIGMA,
        &NOINT_LOGLIK,
        &NOINT_COR,
        &[],
        1e-7,
    );
    let (fit_, se) = f.predict_with_se(&new_x());
    let fit_: Vec<f64> = fit_.iter().copied().collect();
    let se: Vec<f64> = se.iter().copied().collect();
    all_close("NOINT predict", &fit_, &NOINT_PREDICT_FIT, 1e-7);
    all_close("NOINT predict se", &se, &NOINT_PREDICT_SE, 1e-7);
}

#[test]
fn predict_with_se_matches_vcov() {
    let f = fit(
        GlsCorrelation::ContinuousAr1,
        GlsVariance::Constant,
        GlsMethod::Reml,
        Some(&TIME),
        true,
    );
    let (fit_, se) = f.predict_with_se(&new_x());
    let fit_: Vec<f64> = fit_.iter().copied().collect();
    let se: Vec<f64> = se.iter().copied().collect();
    all_close("predict", &fit_, &PREDICT_FIT, 1e-7);
    all_close("predict se", &se, &PREDICT_SE, 1e-7);
}
