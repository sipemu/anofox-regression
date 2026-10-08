//! R validation for issue #66 (M1-A): intercept inference on linear models and
//! GLMs, `logLik`/`AIC`/`BIC` of the linear models, and `diagnostics::augment`.
//!
//! Reference values: `tests/r_scripts/generate_m1a_validation.R`
//! (`summary.lm/glm`, `confint`, `confint.default`, `logLik`, `AIC`, `BIC`,
//! `hatvalues`, `rstandard`, `rstudent`, `cooks.distance`, `dffits`, checked
//! against `influence.measures` and `broom::augment/glance` in the script).

#[path = "fixtures/m1a_validation.rs"]
mod refs;

use anofox_regression::diagnostics::{augment, augment_with, Augment, ResidualType};
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

fn close(label: &str, got: f64, want: f64, tol: f64) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol * scale,
        "{label}: got {got:.12e}, R {want:.12e}"
    );
}

fn close_all(label: &str, got: &Col<f64>, want: &[f64], tol: f64) {
    assert_eq!(got.nrows(), want.len(), "{label}: length");
    for (i, &w) in want.iter().enumerate() {
        close(&format!("{label}[{i}]"), got[i], w, tol);
    }
}

/// Coefficient table, intercept first, against R's (est, se, stat, p, lo, hi).
#[allow(clippy::too_many_arguments)]
fn check_table(
    label: &str,
    r: &RegressionResult,
    est: &[f64],
    se: &[f64],
    stat: &[f64],
    p: &[f64],
    lo: &[f64],
    hi: &[f64],
    tol: f64,
) {
    close(
        &format!("{label} intercept"),
        r.intercept.unwrap(),
        est[0],
        tol,
    );
    close(
        &format!("{label} intercept se"),
        r.intercept_std_error.expect("intercept se"),
        se[0],
        tol,
    );
    close(
        &format!("{label} intercept stat"),
        r.intercept_t_statistic.expect("intercept stat"),
        stat[0],
        tol,
    );
    close(
        &format!("{label} intercept p"),
        r.intercept_p_value.expect("intercept p"),
        p[0],
        tol,
    );
    let (ilo, ihi) = r.intercept_conf_interval.expect("intercept ci");
    close(&format!("{label} intercept ci lo"), ilo, lo[0], tol);
    close(&format!("{label} intercept ci hi"), ihi, hi[0], tol);

    close_all(&format!("{label} coef"), &r.coefficients, &est[1..], tol);
    close_all(
        &format!("{label} se"),
        r.std_errors.as_ref().unwrap(),
        &se[1..],
        tol,
    );
    close_all(
        &format!("{label} stat"),
        r.t_statistics.as_ref().unwrap(),
        &stat[1..],
        tol,
    );
    close_all(
        &format!("{label} p"),
        r.p_values.as_ref().unwrap(),
        &p[1..],
        tol,
    );
    close_all(
        &format!("{label} ci lo"),
        r.conf_interval_lower.as_ref().unwrap(),
        &lo[1..],
        tol,
    );
    close_all(
        &format!("{label} ci hi"),
        r.conf_interval_upper.as_ref().unwrap(),
        &hi[1..],
        tol,
    );
}

struct AugRef<'a> {
    fitted: &'a [f64],
    resid: &'a [f64],
    rstd: &'a [f64],
    rstud: &'a [f64],
    hat: &'a [f64],
    cook: &'a [f64],
    dffits: &'a [f64],
}

fn check_augment(label: &str, a: &Augment, r: &AugRef, tol: f64) {
    close_all(&format!("{label} fitted"), &a.fitted, r.fitted, tol);
    close_all(&format!("{label} residual"), &a.residual, r.resid, tol);
    close_all(&format!("{label} rstandard"), &a.std_residual, r.rstd, tol);
    close_all(&format!("{label} rstudent"), &a.stud_residual, r.rstud, tol);
    close_all(&format!("{label} hat"), &a.leverage, r.hat, tol);
    close_all(&format!("{label} cooks"), &a.cooks_d, r.cook, tol);
    close_all(&format!("{label} dffits"), &a.dffits, r.dffits, tol);
}

fn mtcars_x() -> Mat<f64> {
    design(&[&MT_WT, &MT_HP])
}

fn warpbreaks_x() -> Mat<f64> {
    design(&[&WB_WOOLB, &WB_TM, &WB_TH])
}

// ---------------------------------------------------------------- linear

#[test]
fn ols_intercept_inference_loglik_and_augment() {
    let x = mtcars_x();
    let y = col(&MT_MPG);
    let fit = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    check_table(
        "OLS", r, &OLS_EST, &OLS_SE, &OLS_STAT, &OLS_P, &OLS_CI_LO, &OLS_CI_HI, TOL,
    );
    close("OLS logLik", r.log_likelihood, OLS_LOGLIK, TOL);
    close("OLS AIC", r.aic, OLS_AIC, TOL);
    close("OLS BIC", r.bic, OLS_BIC, TOL);

    let a = augment(&fit, &x, &y, None).unwrap();
    assert_eq!(a.resid_type, ResidualType::Response);
    check_augment(
        "OLS",
        &a,
        &AugRef {
            fitted: &OLS_FITTED,
            resid: &OLS_RESID,
            rstd: &OLS_RSTD,
            rstud: &OLS_RSTUD,
            hat: &OLS_HAT,
            cook: &OLS_COOK,
            dffits: &OLS_DFFITS,
        },
        TOL,
    );
}

#[test]
fn ols_without_intercept_information_criteria() {
    let x = mtcars_x();
    let y = col(&MT_MPG);
    let fit = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    close("OLS0 logLik", r.log_likelihood, OLS0_LOGLIK, TOL);
    close("OLS0 AIC", r.aic, OLS0_AIC, TOL);
    close("OLS0 BIC", r.bic, OLS0_BIC, TOL);
}

#[test]
fn ridge_lambda_zero_matches_lm_information_criteria() {
    let x = mtcars_x();
    let y = col(&MT_MPG);
    let fit = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.0)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    close("Ridge(0) logLik", r.log_likelihood, OLS_LOGLIK, TOL);
    close("Ridge(0) AIC", r.aic, OLS_AIC, TOL);
    close("Ridge(0) BIC", r.bic, OLS_BIC, TOL);
    close(
        "Ridge(0) intercept se",
        r.intercept_std_error.unwrap(),
        OLS_SE[0],
        TOL,
    );
}

#[test]
fn wls_intercept_inference_loglik_and_augment() {
    let x = mtcars_x();
    let y = col(&MT_MPG);
    let w = col(&MT_W);
    let fit = WlsRegressor::builder()
        .with_intercept(true)
        .weights(w.clone())
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    check_table(
        "WLS", r, &WLS_EST, &WLS_SE, &WLS_STAT, &WLS_P, &WLS_CI_LO, &WLS_CI_HI, TOL,
    );
    close("WLS logLik", r.log_likelihood, WLS_LOGLIK, TOL);
    close("WLS AIC", r.aic, WLS_AIC, TOL);
    close("WLS BIC", r.bic, WLS_BIC, TOL);

    let refs = AugRef {
        fitted: &WLS_FITTED,
        resid: &WLS_RESID,
        rstd: &WLS_RSTD,
        rstud: &WLS_RSTUD,
        hat: &WLS_HAT,
        cook: &WLS_COOK,
        dffits: &WLS_DFFITS,
    };
    // The fit's own weights are used by default, explicit weights agree.
    check_augment("WLS", &augment(&fit, &x, &y, None).unwrap(), &refs, TOL);
    check_augment(
        "WLS(w)",
        &augment(&fit, &x, &y, Some(&w)).unwrap(),
        &refs,
        TOL,
    );
}

// ---------------------------------------------------------------- GLMs

#[test]
fn binomial_logit_intercept_inference_and_augment() {
    let x = design(&[&MT_MPG]);
    let y = col(&MT_VS);
    let fit = BinomialRegressor::logistic()
        .with_intercept(true)
        .tolerance(1e-12)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    check_table(
        "BIN",
        fit.result(),
        &BIN_EST,
        &BIN_SE,
        &BIN_STAT,
        &BIN_P,
        &BIN_CI_LO,
        &BIN_CI_HI,
        1e-6,
    );

    let a = augment(&fit, &x, &y, None).unwrap();
    assert_eq!(a.resid_type, ResidualType::Deviance);
    check_augment(
        "BIN",
        &a,
        &AugRef {
            fitted: &BIN_FITTED,
            resid: &BIN_RESID_DEV,
            rstd: &BIN_RSTD_DEV,
            rstud: &BIN_RSTUD,
            hat: &BIN_HAT,
            cook: &BIN_COOK,
            dffits: &BIN_DFFITS,
        },
        1e-6,
    );
    let ap = augment_with(&fit, &x, &y, None, ResidualType::Pearson).unwrap();
    close_all("BIN pearson", &ap.residual, &BIN_RESID_PEAR, 1e-6);
    close_all(
        "BIN rstandard(pearson)",
        &ap.std_residual,
        &BIN_RSTD_PEAR,
        1e-6,
    );
}

#[test]
fn binomial_probit_intercept_inference() {
    let x = design(&[&MT_MPG]);
    let y = col(&MT_VS);
    let fit = BinomialRegressor::probit()
        .with_intercept(true)
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&x, &y)
        .unwrap();
    check_table(
        "PROBIT",
        fit.result(),
        &PROBIT_EST,
        &PROBIT_SE,
        &PROBIT_STAT,
        &PROBIT_P,
        &PROBIT_CI_LO,
        &PROBIT_CI_HI,
        1e-6,
    );
}

#[test]
fn poisson_intercept_inference_and_augment() {
    let x = mtcars_x();
    let y = col(&MT_CARB);
    let fit = PoissonRegressor::log()
        .with_intercept(true)
        .tolerance(1e-12)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    check_table(
        "POIS",
        fit.result(),
        &POIS_EST,
        &POIS_SE,
        &POIS_STAT,
        &POIS_P,
        &POIS_CI_LO,
        &POIS_CI_HI,
        1e-6,
    );
    check_augment(
        "POIS",
        &augment(&fit, &x, &y, None).unwrap(),
        &AugRef {
            fitted: &POIS_FITTED,
            resid: &POIS_RESID_DEV,
            rstd: &POIS_RSTD_DEV,
            rstud: &POIS_RSTUD,
            hat: &POIS_HAT,
            cook: &POIS_COOK,
            dffits: &POIS_DFFITS,
        },
        1e-6,
    );
    let ap = augment_with(&fit, &x, &y, None, ResidualType::Pearson).unwrap();
    close_all("POIS pearson", &ap.residual, &POIS_RESID_PEAR, 1e-6);
    close_all(
        "POIS rstandard(pearson)",
        &ap.std_residual,
        &POIS_RSTD_PEAR,
        1e-6,
    );
}

#[test]
fn gamma_intercept_inference_and_augment() {
    let x = mtcars_x();
    let y = col(&MT_MPG);
    let fit = GammaRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .max_iterations(100)
        .tolerance(1e-12)
        .build()
        .fit(&x, &y)
        .unwrap();
    check_table(
        "GAMMA",
        fit.result(),
        &GAMMA_EST,
        &GAMMA_SE,
        &GAMMA_STAT,
        &GAMMA_P,
        &GAMMA_CI_LO,
        &GAMMA_CI_HI,
        1e-6,
    );
    check_augment(
        "GAMMA",
        &augment(&fit, &x, &y, None).unwrap(),
        &AugRef {
            fitted: &GAMMA_FITTED,
            resid: &GAMMA_RESID_DEV,
            rstd: &GAMMA_RSTD_DEV,
            rstud: &GAMMA_RSTUD,
            hat: &GAMMA_HAT,
            cook: &GAMMA_COOK,
            dffits: &GAMMA_DFFITS,
        },
        1e-6,
    );
}

#[test]
fn negative_binomial_intercept_inference_and_augment() {
    let x = warpbreaks_x();
    let y = col(&WB_BREAKS);
    let fit = NegativeBinomialRegressor::builder()
        .with_intercept(true)
        .tolerance(1e-12)
        .estimate_theta(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    close("NB theta", fit.theta, NB_THETA, 1e-4);
    check_table(
        "NB",
        fit.result(),
        &NB_EST,
        &NB_SE,
        &NB_STAT,
        &NB_P,
        &NB_CI_LO,
        &NB_CI_HI,
        1e-6,
    );
    check_augment(
        "NB",
        &augment(&fit, &x, &y, None).unwrap(),
        &AugRef {
            fitted: &NB_FITTED,
            resid: &NB_RESID_DEV,
            rstd: &NB_RSTD_DEV,
            rstud: &NB_RSTUD,
            hat: &NB_HAT,
            cook: &NB_COOK,
            dffits: &NB_DFFITS,
        },
        1e-6,
    );
}

#[test]
fn augment_rejects_mismatched_lengths() {
    let x = mtcars_x();
    let y = col(&MT_MPG);
    let fit = OlsRegressor::builder().build().fit(&x, &y).unwrap();
    let short = col(&MT_MPG[..10]);
    assert!(augment(&fit, &x, &short, None).is_err());
    assert!(augment(&fit, &x, &y, Some(&short)).is_err());
}
