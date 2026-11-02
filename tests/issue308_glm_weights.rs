//! Issue #308: prior weights for `PoissonRegressor`, `GammaRegressor` and
//! `TweedieRegressor`, against R's `glm(weights = )` (one row has weight 0).
//!
//! Both sides are converged tightly (R `glm.control(epsilon = 1e-14)`,
//! `tolerance(1e-12)` here), so they agree to rounding.
//!
//! Reference values: `tests/r_scripts/generate_glm_weights_validation.R`.

#![allow(dead_code)]

use anofox_regression::solvers::{
    FittedRegressor, GammaRegressor, PoissonRegressor, Regressor, TweedieRegressor,
};
use faer::{Col, Mat};

include!("fixtures/glm_weights_validation.rs");

const N: usize = 40;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| if j == 0 { X1[i] } else { X2[i] })
}
fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}
fn rel(what: &str, got: f64, want: f64, tol: f64) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: {got} vs R NA");
        return;
    }
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: {got} vs R {want}"
    );
}
fn all(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        rel(&format!("{what}[{i}]"), *g, *w, tol);
    }
}

/// What every weighted GLM fit is checked on.
struct Expected<'a> {
    coef: &'a [f64],
    se: &'a [f64],
    dev: f64,
    null_dev: f64,
    disp: f64,
    loglik: f64,
    aic: f64,
    bic: f64,
    pearson: &'a [f64],
    devres: &'a [f64],
}

#[allow(clippy::too_many_arguments)]
fn check<F: FittedRegressor>(
    what: &str,
    fit: &F,
    dev: f64,
    null_dev: f64,
    disp: f64,
    pearson: Col<f64>,
    devres: Col<f64>,
    e: &Expected,
    check_ll: bool,
) {
    let r = fit.result();
    let coef: Vec<f64> = std::iter::once(r.intercept.unwrap())
        .chain(r.coefficients.iter().copied())
        .collect();
    // R's deviance criterion pins the deviance, which is quadratic in the
    // coefficient error: with non-canonical links (Tweedie 1.5) both fits
    // agree to ~√tolerance in the coefficients.
    all(&format!("{what} coef"), &coef, e.coef, 1e-6);
    let se: Vec<f64> = std::iter::once(r.intercept_std_error.unwrap())
        .chain(r.std_errors.as_ref().unwrap().iter().copied())
        .collect();
    all(&format!("{what} se"), &se, e.se, 1e-6);
    rel(&format!("{what} deviance"), dev, e.dev, 1e-9);
    rel(&format!("{what} null deviance"), null_dev, e.null_dev, 1e-9);
    rel(&format!("{what} dispersion"), disp, e.disp, 1e-7);
    if check_ll {
        rel(&format!("{what} logLik"), r.log_likelihood, e.loglik, 1e-9);
        rel(&format!("{what} AIC"), r.aic, e.aic, 1e-9);
        rel(&format!("{what} BIC"), r.bic, e.bic, 1e-9);
    }
    let p: Vec<f64> = pearson.iter().copied().collect();
    all(&format!("{what} pearson"), &p, e.pearson, 1e-6);
    let d: Vec<f64> = devres.iter().copied().collect();
    all(&format!("{what} deviance residuals"), &d, e.devres, 1e-6);
}

macro_rules! expected {
    ($p:ident) => {
        paste_expected!($p)
    };
}

macro_rules! paste_expected {
    (POIS) => {
        Expected {
            coef: &POIS_COEF,
            se: &POIS_SE,
            dev: POIS_DEV,
            null_dev: POIS_NULL_DEV,
            disp: POIS_DISP,
            loglik: POIS_LOGLIK,
            aic: POIS_AIC,
            bic: POIS_BIC,
            pearson: &POIS_PEARSON,
            devres: &POIS_DEVRES,
        }
    };
    (QPOIS) => {
        Expected {
            coef: &QPOIS_COEF,
            se: &QPOIS_SE,
            dev: QPOIS_DEV,
            null_dev: QPOIS_NULL_DEV,
            disp: QPOIS_DISP,
            loglik: QPOIS_LOGLIK,
            aic: QPOIS_AIC,
            bic: QPOIS_BIC,
            pearson: &QPOIS_PEARSON,
            devres: &QPOIS_DEVRES,
        }
    };
    (GAMMA) => {
        Expected {
            coef: &GAMMA_COEF,
            se: &GAMMA_SE,
            dev: GAMMA_DEV,
            null_dev: GAMMA_NULL_DEV,
            disp: GAMMA_DISP,
            loglik: GAMMA_LOGLIK,
            aic: GAMMA_AIC,
            bic: GAMMA_BIC,
            pearson: &GAMMA_PEARSON,
            devres: &GAMMA_DEVRES,
        }
    };
    (TW) => {
        Expected {
            coef: &TW_COEF,
            se: &TW_SE,
            dev: TW_DEV,
            null_dev: TW_NULL_DEV,
            disp: TW_DISP,
            loglik: TW_LOGLIK,
            aic: TW_AIC,
            bic: TW_BIC,
            pearson: &TW_PEARSON,
            devres: &TW_DEVRES,
        }
    };
    (GAUSS) => {
        Expected {
            coef: &GAUSS_COEF,
            se: &GAUSS_SE,
            dev: GAUSS_DEV,
            null_dev: GAUSS_NULL_DEV,
            disp: GAUSS_DISP,
            loglik: GAUSS_LOGLIK,
            aic: GAUSS_AIC,
            bic: GAUSS_BIC,
            pearson: &GAUSS_PEARSON,
            devres: &GAUSS_DEVRES,
        }
    };
    (GAUSSP) => {
        Expected {
            coef: &GAUSSP_COEF,
            se: &GAUSSP_SE,
            dev: GAUSSP_DEV,
            null_dev: GAUSSP_NULL_DEV,
            disp: GAUSSP_DISP,
            loglik: GAUSSP_LOGLIK,
            aic: GAUSSP_AIC,
            bic: GAUSSP_BIC,
            pearson: &GAUSSP_PEARSON,
            devres: &GAUSSP_DEVRES,
        }
    };
    (IG) => {
        Expected {
            coef: &IG_COEF,
            se: &IG_SE,
            dev: IG_DEV,
            null_dev: IG_NULL_DEV,
            disp: IG_DISP,
            loglik: IG_LOGLIK,
            aic: IG_AIC,
            bic: IG_BIC,
            pearson: &IG_PEARSON,
            devres: &IG_DEVRES,
        }
    };
}

#[test]
fn weighted_poisson_matches_glm() {
    let f = PoissonRegressor::log()
        .weights(col(&W))
        .tolerance(1e-12)
        .build()
        .fit(&x(), &col(&Y_POIS))
        .unwrap();
    check(
        "poisson",
        &f,
        f.deviance,
        f.null_deviance,
        f.dispersion,
        f.pearson_residuals(),
        f.deviance_residuals(),
        &expected!(POIS),
        true,
    );
    assert_eq!(f.prior_weights().unwrap().nrows(), N);
}

#[test]
fn weighted_quasipoisson_matches_glm() {
    let f = PoissonRegressor::log()
        .weights(col(&W))
        .tolerance(1e-12)
        .estimate_dispersion(true)
        .build()
        .fit(&x(), &col(&Y_POIS))
        .unwrap();
    check(
        "quasipoisson",
        &f,
        f.deviance,
        f.null_deviance,
        f.dispersion,
        f.pearson_residuals(),
        f.deviance_residuals(),
        &expected!(QPOIS),
        false,
    );
}

#[test]
fn weighted_gamma_matches_glm() {
    let f = GammaRegressor::builder()
        .weights(col(&W))
        .tolerance(1e-12)
        .build()
        .fit(&x(), &col(&Y_GAMMA))
        .unwrap();
    let t = f.inner();
    check(
        "gamma",
        &f,
        t.deviance,
        t.null_deviance,
        t.dispersion,
        t.pearson_residuals(),
        t.deviance_residuals(),
        &expected!(GAMMA),
        true,
    );
}

fn tweedie(
    var_power: f64,
    link_power: f64,
    w: &[f64],
    y: &[f64],
) -> anofox_regression::solvers::FittedTweedie {
    TweedieRegressor::builder()
        .var_power(var_power)
        .link_power(link_power)
        .weights(col(w))
        .tolerance(1e-12)
        .max_iterations(200)
        .build()
        .fit(&x(), &col(y))
        .unwrap()
}

#[test]
fn weighted_tweedie_matches_glm() {
    for (what, f, e, ll) in [
        (
            "tweedie 1.5",
            tweedie(1.5, 0.0, &W, &Y_TW),
            expected!(TW),
            true,
        ),
        (
            "inverse gaussian",
            tweedie(3.0, 0.0, &W, &Y_GAMMA),
            expected!(IG),
            true,
        ),
        // R's gaussian logLik is -Inf with a zero weight (log 0); ours
        // leaves the row out, so only the fit is compared here.
        (
            "gaussian",
            tweedie(0.0, 1.0, &W, &Y_GAUSS),
            expected!(GAUSS),
            false,
        ),
        (
            "gaussian, positive weights",
            tweedie(0.0, 1.0, &WP, &Y_GAUSS),
            expected!(GAUSSP),
            true,
        ),
    ] {
        check(
            what,
            &f,
            f.deviance,
            f.null_deviance,
            f.dispersion,
            f.pearson_residuals(),
            f.deviance_residuals(),
            &e,
            ll,
        );
    }
}

#[test]
fn unit_weights_change_nothing() {
    let y = col(&Y_POIS);
    let plain = PoissonRegressor::log().build().fit(&x(), &y).unwrap();
    let ones = PoissonRegressor::log()
        .weights(Col::from_fn(N, |_| 1.0))
        .build()
        .fit(&x(), &y)
        .unwrap();
    assert_eq!(plain.deviance, ones.deviance);
    assert_eq!(plain.result().log_likelihood, ones.result().log_likelihood);
}

#[test]
fn invalid_weights_are_rejected() {
    let y = col(&Y_POIS);
    for w in [
        Col::from_fn(N, |i| if i == 3 { -1.0 } else { 1.0 }),
        Col::from_fn(N - 1, |_| 1.0),
    ] {
        assert!(PoissonRegressor::log()
            .weights(w.clone())
            .build()
            .fit(&x(), &y)
            .is_err());
        assert!(TweedieRegressor::builder()
            .weights(w)
            .build()
            .fit(&x(), &col(&Y_GAMMA))
            .is_err());
    }
}

#[test]
fn augment_uses_prior_weights() {
    use anofox_regression::diagnostics::augment;
    let f = PoissonRegressor::log()
        .weights(col(&W))
        .tolerance(1e-12)
        .build()
        .fit(&x(), &col(&Y_POIS))
        .unwrap();
    let a = augment(&f, &x(), &col(&Y_POIS), None).unwrap();
    // The zero-weight row has no leverage.
    assert_eq!(a.leverage[6], 0.0);
}
