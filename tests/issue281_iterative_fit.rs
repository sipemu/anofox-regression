//! #281: `IterativeFit` for ALM, GLS and Theil–Sen — an iteration budget of
//! one reports `converged() == false` (and `ensure_converged` fails), the
//! default budget converges.

use anofox_regression::solvers::{
    AlmDistribution, AlmRegressor, GlsCorrelation, GlsRegressor, IterativeFit, Regressor,
    TheilSenRegressor,
};
use faer::{Col, Mat};

/// `n` rows of two predictors with a smooth, non-trivial response.
fn data(n: usize) -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(n, 2, |i, j| {
        let t = i as f64;
        if j == 0 {
            (t * 0.37).sin() * 2.0
        } else {
            ((i * 7) % 11) as f64 / 5.0 - 1.0
        }
    });
    let y = Col::from_fn(n, |i| {
        let eta = 0.4 + 0.6 * x[(i, 0)] - 0.3 * x[(i, 1)];
        (eta.exp() + ((i * 3) % 5) as f64 * 0.3).max(0.05)
    });
    (x, y)
}

#[test]
fn alm_irls_and_numerical_report_convergence() {
    let (x, y) = data(80);
    for dist in [AlmDistribution::Gamma, AlmDistribution::FoldedNormal] {
        let fit = AlmRegressor::builder()
            .distribution(dist)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert!(fit.converged(), "{dist:?} converges by default");
        assert!(fit.iterations() >= 1, "{dist:?} iterations");
        let short = AlmRegressor::builder()
            .distribution(dist)
            .max_iterations(1)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert!(!short.converged(), "{dist:?} with one iteration");
        assert!(short.iterations() <= 1);
        assert!(short.ensure_converged().is_err());
    }
}

#[test]
fn gls_reports_convergence() {
    let (x, y) = data(80);
    let group: Vec<usize> = (0..80).map(|i| i / 8).collect();
    let fit = GlsRegressor::builder()
        .correlation(GlsCorrelation::Ar1)
        .build()
        .fit(&x, &y, &group, None)
        .unwrap();
    assert!(fit.converged());
    assert!(fit.iterations() >= 1);
    let short = GlsRegressor::builder()
        .correlation(GlsCorrelation::Ar1)
        .max_iterations(1)
        .build()
        .fit(&x, &y, &group, None)
        .unwrap();
    assert!(!short.converged());
    assert!(short.clone().ensure_converged().is_err());
    // no variance–covariance parameters: closed-form
    let ols = GlsRegressor::builder()
        .build()
        .fit(&x, &y, &group, None)
        .unwrap();
    assert!(ols.converged());
    assert_eq!(ols.iterations(), 0);
}

#[test]
fn theil_sen_reports_convergence() {
    let (x, y) = data(40);
    let fit = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();
    assert!(fit.converged());
    assert!(fit.iterations() >= 1);
    let short = TheilSenRegressor::builder()
        .max_iter(1)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!(!short.converged());
    assert_eq!(short.iterations(), 1);
    // one predictor without intercept: the ordinary median, exact
    let x1 = Mat::from_fn(40, 1, |i, _| x[(i, 0)]);
    let one = TheilSenRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x1, &y)
        .unwrap();
    assert!(one.converged());
    assert_eq!(one.iterations(), 0);
}
