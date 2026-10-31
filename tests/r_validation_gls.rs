#![allow(clippy::excessive_precision)]

//! #111: generalized least squares with serial-correlation structures and
//! variance functions (RMS Ch 7), against nlme::gls. Reference values come
//! from `tests/r_scripts/generate_gls_validation.R`: 40 subjects with 3–5
//! irregular visits, `y ~ time * trt + x`.
//!
//! Estimates are compared at 1e-7 (correlation / variance parameters 1e-6),
//! except for the structures where nlme's optimiser stops ~1e-6 short
//! (compound symmetry, unstructured, spatial: 1e-4 / 1e-3); `intervals.gls`
//! uses a finite-difference Hessian, so interval limits are compared at 1e-3
//! (the log-likelihood, flat at the optimum, at 1e-7).

use anofox_regression::solvers::{
    FittedGls, GlsCorrelation, GlsMethod, GlsRegressor, GlsResidualType, GlsVariance,
};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/gls_validation.rs");
}
use fixture::*;

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

fn data() -> (Mat<f64>, Col<f64>, Vec<usize>) {
    let x = Mat::from_fn(N, P, |i, j| X[i * P + j]);
    let y = Col::from_fn(N, |i| Y[i]);
    let id = ID.iter().map(|&v| v as usize).collect();
    (x, y, id)
}

fn visits() -> Vec<usize> {
    VISIT.iter().map(|&v| v as usize).collect()
}

fn fit(
    cor: GlsCorrelation,
    var: GlsVariance,
    method: GlsMethod,
    time: Option<&[f64]>,
) -> FittedGls {
    let (x, y, id) = data();
    GlsRegressor::builder()
        .correlation(cor)
        .variance(var)
        .method(method)
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
) {
    // nlme's optimiser (even with a tight `glsControl`) stops ~1e-6 short
    // for compound symmetry, the unstructured and the spatial structures, so
    // those are compared at 1e-4 (parameters 1e-3); the others agree to 1e-7.
    let loose = tag.starts_with("CS") || tag.starts_with("SYMM") || tag.contains("nugget");
    let (tol, tol_par) = if loose { (1e-4, 1e-3) } else { (1e-7, 1e-6) };
    let p = P + 1;
    let b: Vec<f64> = f.coefficients().iter().copied().collect();
    all_close(&format!("{tag} coef"), &b, coef, tol);
    let v: Vec<f64> = (0..p * p).map(|k| f.vcov()[(k / p, k % p)]).collect();
    all_close(&format!("{tag} vcov"), &v, vcov, tol);
    close(&format!("{tag} sigma"), f.sigma(), sigma[0], tol);
    close(&format!("{tag} logLik"), f.log_likelihood(), ll[0], 1e-7);
    close(&format!("{tag} AIC"), f.aic(), ll[1], 1e-7);
    close(&format!("{tag} BIC"), f.bic(), ll[2], 1e-7);
    all_close(&format!("{tag} cor"), f.correlation_params(), cor, tol_par);
    all_close(&format!("{tag} var"), f.variance_params(), var, tol_par);
}

/// `intervals(fit, which = "var-cov")` flattened as (lower, est, upper) per
/// correlation, variance and σ parameter.
fn check_intervals(tag: &str, f: &FittedGls, expected: &[f64]) {
    check_intervals_tol(tag, f, expected, 2e-3);
}

fn check_intervals_tol(tag: &str, f: &FittedGls, expected: &[f64], tol: f64) {
    let iv = f.var_cov_intervals().expect("intervals");
    let got: Vec<f64> = iv
        .correlation
        .iter()
        .chain(&iv.variance)
        .chain(std::iter::once(&iv.sigma))
        .flat_map(|i| [i.lower, i.estimate, i.upper])
        .collect();
    all_close(&format!("{tag} intervals"), &got, expected, tol);
}

#[test]
fn continuous_ar1_matches_nlme() {
    let f = fit(
        GlsCorrelation::ContinuousAr1,
        GlsVariance::Constant,
        GlsMethod::Reml,
        Some(&TIME),
    );
    check(
        "CAR1",
        &f,
        &CAR1_COEF,
        &CAR1_VCOV,
        &CAR1_SIGMA,
        &CAR1_LOGLIK,
        &CAR1_COR,
        &[],
    );
    check_intervals("CAR1", &f, &CAR1_INTERVALS);
    let r = |k| -> Vec<f64> { f.residuals(k).iter().copied().collect() };
    all_close(
        "CAR1 pearson",
        &r(GlsResidualType::Pearson),
        &CAR1_PEARSON,
        1e-4,
    );
    all_close(
        "CAR1 normalized",
        &r(GlsResidualType::Normalized),
        &CAR1_NORMALIZED,
        1e-4,
    );

    let nd = Mat::from_fn(3, P, |i, j| {
        let (t, trt, x) = [(1.0, 0.0, 0.0), (3.0, 1.0, 0.5), (5.0, 1.0, -1.0)][i];
        [t, trt, x, t * trt][j]
    });
    let (mean, se) = f.predict_with_se(&nd);
    let mean: Vec<f64> = mean.iter().copied().collect();
    all_close("CAR1 predict", &mean, &CAR1_PREDICT, 1e-4);
    assert!(se.iter().all(|s| *s > 0.0));
}

#[test]
fn variogram_matches_nlme() {
    let f = fit(
        GlsCorrelation::ContinuousAr1,
        GlsVariance::Constant,
        GlsMethod::Reml,
        Some(&TIME),
    );
    // quantile classes of the continuous times
    let v = f.variogram(&TIME, GlsResidualType::Normalized, 20).unwrap();
    let got: Vec<f64> = v
        .iter()
        .flat_map(|b| [b.variog, b.dist, b.n_pairs as f64])
        .collect();
    all_close("variogram", &got, &CAR1_VARIOGRAM, 1e-4);
    // one class per distinct visit lag
    let v = f.variogram(&VISIT, GlsResidualType::Pearson, 20).unwrap();
    let got: Vec<f64> = v
        .iter()
        .flat_map(|b| [b.variog, b.dist, b.n_pairs as f64])
        .collect();
    all_close("variogram visit", &got, &CAR1_VARIOGRAM_VISIT, 1e-4);
}

#[test]
fn ar1_ml_matches_nlme() {
    let f = fit(
        GlsCorrelation::Ar1,
        GlsVariance::Constant,
        GlsMethod::Ml,
        Some(&VISIT),
    );
    check(
        "AR1",
        &f,
        &AR1_ML_COEF,
        &AR1_ML_VCOV,
        &AR1_ML_SIGMA,
        &AR1_ML_LOGLIK,
        &AR1_ML_COR,
        &[],
    );
    check_intervals("AR1", &f, &AR1_ML_INTERVALS);
}

#[test]
fn compound_symmetry_with_var_ident_matches_nlme() {
    let f = fit(
        GlsCorrelation::CompoundSymmetry,
        GlsVariance::Ident(visits()),
        GlsMethod::Reml,
        None,
    );
    check(
        "CS",
        &f,
        &CS_IDENT_COEF,
        &CS_IDENT_VCOV,
        &CS_IDENT_SIGMA,
        &CS_IDENT_LOGLIK,
        &CS_IDENT_COR,
        &CS_IDENT_VAR,
    );
    check_intervals("CS", &f, &CS_IDENT_INTERVALS);
}

#[test]
fn spatial_structures_match_nlme() {
    let f = fit(
        GlsCorrelation::Exponential { nugget: true },
        GlsVariance::Constant,
        GlsMethod::Reml,
        Some(&TIME),
    );
    // the range is weakly identified (≈ 29 with a CI to 670): compare the
    // likelihood tightly, the parameters loosely
    close("EXP logLik", f.log_likelihood(), EXP_NUG_LOGLIK[0], 1e-7);
    all_close("EXP cor", f.correlation_params(), &EXP_NUG_COR, 2e-2);
    let b: Vec<f64> = f.coefficients().iter().copied().collect();
    all_close("EXP coef", &b, &EXP_NUG_COEF, 1e-3);
    close("EXP sigma", f.sigma(), EXP_NUG_SIGMA[0], 1e-3);
    let v: Vec<f64> = (0..25).map(|k| f.vcov()[(k / 5, k % 5)]).collect();
    all_close("EXP vcov", &v, &EXP_NUG_VCOV, 1e-3);
    check_intervals_tol("EXP", &f, &EXP_NUG_INTERVALS, 5e-2);

    let cases = [
        (
            GlsCorrelation::Gaussian { nugget: false },
            &GAUS_COEF,
            &GAUS_VCOV,
            &GAUS_SIGMA,
            &GAUS_LOGLIK,
            &GAUS_COR,
        ),
        (
            GlsCorrelation::Linear { nugget: false },
            &LIN_COEF,
            &LIN_VCOV,
            &LIN_SIGMA,
            &LIN_LOGLIK,
            &LIN_COR,
        ),
        (
            GlsCorrelation::Spherical { nugget: false },
            &SPHER_COEF,
            &SPHER_VCOV,
            &SPHER_SIGMA,
            &SPHER_LOGLIK,
            &SPHER_COR,
        ),
    ];
    for (cor, coef, vcov, sigma, ll, par) in cases {
        let f = fit(cor, GlsVariance::Constant, GlsMethod::Reml, Some(&TIME));
        check(&format!("{cor:?}"), &f, coef, vcov, sigma, ll, par, &[]);
    }
}

#[test]
fn unstructured_with_var_ident_matches_nlme() {
    let f = fit(
        GlsCorrelation::Unstructured,
        GlsVariance::Ident(visits()),
        GlsMethod::Reml,
        Some(&VISIT),
    );
    check(
        "SYMM",
        &f,
        &SYMM_COEF,
        &SYMM_VCOV,
        &SYMM_SIGMA,
        &SYMM_LOGLIK,
        &SYMM_COR,
        &SYMM_VAR,
    );
    // 16 parameters: nlme's fdHess is ~1% off here (its SE of the first
    // varIdent ratio is 0.1290; a Richardson-extrapolated Hessian of nlme's
    // own objective, numDeriv::hessian, gives 0.1273 as we do)
    check_intervals_tol("SYMM", &f, &SYMM_INTERVALS, 1e-2);
}

#[test]
fn variance_functions_match_nlme() {
    let f = fit(
        GlsCorrelation::Independent,
        GlsVariance::Power(AX.to_vec()),
        GlsMethod::Reml,
        None,
    );
    check(
        "POWER",
        &f,
        &POWER_COEF,
        &POWER_VCOV,
        &POWER_SIGMA,
        &POWER_LOGLIK,
        &[],
        &POWER_VAR,
    );
    check_intervals("POWER", &f, &POWER_INTERVALS);

    let xcol: Vec<f64> = (0..N).map(|i| X[i * P + 2]).collect();
    let f = fit(
        GlsCorrelation::Independent,
        GlsVariance::Exponential(xcol),
        GlsMethod::Ml,
        None,
    );
    check(
        "VEXP",
        &f,
        &VEXP_ML_COEF,
        &VEXP_ML_VCOV,
        &VEXP_ML_SIGMA,
        &VEXP_ML_LOGLIK,
        &[],
        &VEXP_ML_VAR,
    );
}

#[test]
fn independent_errors_are_ols() {
    let f = fit(
        GlsCorrelation::Independent,
        GlsVariance::Constant,
        GlsMethod::Reml,
        None,
    );
    check(
        "INDEP",
        &f,
        &INDEP_COEF,
        &INDEP_VCOV,
        &INDEP_SIGMA,
        &INDEP_LOGLIK,
        &[],
        &[],
    );
    check_intervals("INDEP", &f, &INDEP_INTERVALS);
    assert_eq!(f.n_params(), P + 2);
    let (lo, hi) = f.conf_int();
    assert!(lo[1] < f.coefficients()[1] && f.coefficients()[1] < hi[1]);
    assert!(f.p_values().iter().all(|p| (0.0..=1.0).contains(p)));
}

#[test]
fn rejects_bad_input() {
    let (x, y, id) = data();
    let short = vec![0usize; N - 1];
    assert!(GlsRegressor::builder()
        .build()
        .fit(&x, &y, &short, None)
        .is_err());
    let frac: Vec<f64> = TIME.to_vec();
    assert!(GlsRegressor::builder()
        .correlation(GlsCorrelation::Ar1)
        .build()
        .fit(&x, &y, &id, Some(&frac))
        .is_err());
    let singletons: Vec<usize> = (0..N).collect();
    assert!(GlsRegressor::builder()
        .correlation(GlsCorrelation::CompoundSymmetry)
        .build()
        .fit(&x, &y, &singletons, None)
        .is_err());
    assert!(GlsRegressor::builder()
        .variance(GlsVariance::Power(vec![0.0; N]))
        .build()
        .fit(&x, &y, &id, None)
        .is_err());
}
