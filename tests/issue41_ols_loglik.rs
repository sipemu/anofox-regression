//! Issue #41 (R reference values from R 4.6.1 / MASS / lme4): OLS/WLS/Ridge/Huber log-likelihood uses MSE = RSS/(n-p) instead of the
//! ML variance RSS/n, so log_likelihood / AIC / AICc / BIC differ from R logLik(lm).
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

#[test]
fn ols_loglik_matches_r() {
    let n = 15;
    let x = Mat::from_fn(n, 2, |k, j| {
        let i = k + 1;
        if j == 0 {
            i as f64 / 3.0
        } else {
            ((i * 5) % 7) as f64 / 2.0
        }
    });
    let y = Col::from_fn(n, |k| {
        let i = k + 1;
        2.0 + 1.5 * x[(k, 0)] - 0.8 * x[(k, 1)] + (((i * 3) % 5) as f64 - 2.0) * 0.5
    });
    let f = OlsRegressor::builder()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    println!(
        "logLik {} AIC {} BIC {}  (R: -15.875039497906823 39.750078995813645 42.582279800222487)",
        r.log_likelihood, r.aic, r.bic
    );
    assert!(
        (r.log_likelihood + 15.875039497906823).abs() < 1e-9,
        "log_likelihood {}",
        r.log_likelihood
    );
}
