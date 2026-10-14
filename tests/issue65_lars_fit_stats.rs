//! Issue #65: LARS fills `adj_r_squared` and `rmse` (residual standard
//! error) with OLS conventions, df = active coefficients + intercept.
//! DataZooDE/anofox-statistics#196: the same for the log-likelihood,
//! AIC / AICc / BIC and the F test (previously left at 0).

use anofox_regression::solvers::{
    FittedRegressor, LarsMethod, LarsRegressor, OlsRegressor, Regressor,
};
use faer::{Col, Mat};

fn data() -> (Mat<f64>, Col<f64>) {
    let n = 60;
    let x = Mat::from_fn(n, 2, |i, j| {
        let t = i as f64;
        if j == 0 {
            t / 10.0
        } else {
            t.sin()
        }
    });
    let y = Col::from_fn(n, |i| {
        let t = i as f64;
        2.0 + 1.5 * (t / 10.0) - 0.8 * t.sin() + 0.3 * (7.0 * t).cos()
    });
    (x, y)
}

#[test]
fn lars_full_path_matches_r_lm_summary() {
    let (x, y) = data();
    let f = LarsRegressor::builder()
        .method(LarsMethod::Lar)
        .fit_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    // R: s <- summary(lm(y ~ x1 + x2)); c(s$r.squared, s$adj.r.squared, s$sigma)
    assert!((r.r_squared - 0.993640840379574).abs() < 1e-9);
    assert!((r.adj_r_squared - 0.993417711971839).abs() < 1e-9);
    assert!((r.rmse - 0.216257642550628).abs() < 1e-9);
    assert!((r.mse - 0.216257642550628_f64.powi(2)).abs() < 1e-9);
}

#[test]
fn lasso_lars_uses_active_set_df() {
    let (x, y) = data();
    let f = LarsRegressor::builder()
        .method(LarsMethod::Lasso)
        .alpha(0.05)
        .fit_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    let n = y.nrows() as f64;
    let active = r.coefficients.iter().filter(|c| **c != 0.0).count() as f64;
    let rss: f64 = r.residuals.iter().map(|e| e * e).sum();
    let df_resid = n - active - 1.0;
    assert!(r.rmse > 0.0 && r.adj_r_squared > 0.0);
    assert!((r.rmse - (rss / df_resid).sqrt()).abs() < 1e-12);
    let adj = 1.0 - (1.0 - r.r_squared) * (n - 1.0) / df_resid;
    assert!((r.adj_r_squared - adj).abs() < 1e-12);
}

#[test]
fn lars_matches_ols_on_active_set_without_intercept() {
    let (x, y) = data();
    let f = LarsRegressor::builder()
        .method(LarsMethod::Lar)
        .fit_intercept(false)
        .build()
        .fit(&x, &y)
        .unwrap();
    let o = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!((f.result().rmse - o.result().rmse).abs() < 1e-8);
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

#[test]
fn lars_full_path_loglik_and_ic_match_r() {
    let (x, y) = data();
    // R: m <- lm(y ~ x1 + x2); c(logLik(m), AIC(m), BIC(m), summary(m)$fstatistic[1])
    for (intercept, ll, aic, bic, f) in [
        (
            true,
            8.2795743905495,
            -8.55914878109901,
            -0.181770532210606,
            4453.22426879395,
        ),
        // R: lm(y ~ 0 + x1 + x2)
        (
            false,
            -87.8129102551657,
            181.625820510331,
            187.908854196998,
            1240.97629700477,
        ),
    ] {
        let r = LarsRegressor::builder()
            .method(LarsMethod::Lar)
            .fit_intercept(intercept)
            .build()
            .fit(&x, &y)
            .unwrap()
            .result()
            .clone();
        assert!(
            close(r.log_likelihood, ll, 1e-9),
            "{} {}",
            intercept,
            r.log_likelihood
        );
        assert!(close(r.aic, aic, 1e-9), "{} {}", intercept, r.aic);
        assert!(close(r.bic, bic, 1e-9), "{} {}", intercept, r.bic);
        assert!(
            close(r.f_statistic, f, 1e-9),
            "{} {}",
            intercept,
            r.f_statistic
        );
        assert!(r.f_pvalue < 1e-30);
        let o = OlsRegressor::builder()
            .with_intercept(intercept)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert!(close(r.aicc, o.result().aicc, 1e-9));
    }
}

#[test]
fn lasso_lars_loglik_counts_active_set() {
    let (x, y) = data();
    let r = LarsRegressor::builder()
        .method(LarsMethod::Lasso)
        .alpha(0.05)
        .fit_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap()
        .result()
        .clone();
    let n = y.nrows() as f64;
    let active = r.coefficients.iter().filter(|c| **c != 0.0).count() as f64;
    let rss: f64 = r.residuals.iter().map(|e| e * e).sum();
    // Gaussian log-likelihood at the ML variance RSS / n; k = active + intercept + sigma.
    let ll = -n / 2.0 * ((2.0 * std::f64::consts::PI).ln() + 1.0 + (rss / n).ln());
    let k = active + 2.0;
    assert!(close(r.log_likelihood, ll, 1e-12));
    assert!(close(r.aic, 2.0 * k - 2.0 * ll, 1e-12));
    assert!(close(r.bic, k * n.ln() - 2.0 * ll, 1e-12));
    assert!(r.log_likelihood != 0.0 && r.aic != 0.0);
}
