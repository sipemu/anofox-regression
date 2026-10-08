//! Issue #37 (R reference values from R 4.6.1 / MASS / lme4): RidgeRegressor inference uses sqrt(MSE * diag((X'X + lambda I)^-1)) on the
//! *uncentred* X (although the estimator centres X), intercept SE sqrt(MSE / n), and
//! reports classical t-tests on the shrunken coefficients.
use anofox_regression::solvers::{FittedRegressor, Regressor, RidgeRegressor};
use faer::{Col, Mat};

fn data() -> (Mat<f64>, Col<f64>) {
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
    (x, y)
}

#[test]
fn ridge_with_tiny_lambda_has_ols_standard_errors() {
    // lambda -> 0 must reproduce lm()'s standard errors (continuity at OLS).
    let (x, y) = data();
    let f = RidgeRegressor::builder()
        .lambda(1e-8)
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    let se = r.std_errors.as_ref().unwrap();
    println!("lambda=1e-8: coef {:?} {:?} SE [{:?}, {}, {}]  (lm: SE 0.55492086380506511 0.14058832696496731 0.20292177104535755)",
        r.intercept, r.coefficients.iter().collect::<Vec<_>>(), r.intercept_std_error, se[0], se[1]);
    assert!(
        (se[0] - 0.14058832696496731).abs() < 1e-6,
        "xa SE {}",
        se[0]
    );
    assert!(
        (se[1] - 0.20292177104535755).abs() < 1e-6,
        "xb SE {}",
        se[1]
    );
    assert!(
        (r.intercept_std_error.unwrap() - 0.55492086380506511).abs() < 1e-6,
        "intercept SE"
    );
}

#[test]
fn ridge_standard_errors_are_the_sandwich_variance() {
    // Var(beta_ridge) = s^2 A X'X A, A = (X'X + lambda P)^-1, P = diag(0, 1, 1) (R script).
    let (x, y) = data();
    let f = RidgeRegressor::builder()
        .lambda(1.0)
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    let se = r.std_errors.as_ref().unwrap();
    println!("lambda=1: coef {:?} {:?} SE [{:?}, {}, {}] p {:?}  (sandwich SE 0.53418431237438180 0.13688648231466696 0.19112708153110064)",
        r.intercept, r.coefficients.iter().collect::<Vec<_>>(), r.intercept_std_error, se[0], se[1], r.p_values);
    assert!(
        (r.coefficients[0] - 1.38008516740810094).abs() < 1e-9,
        "same estimator as the R reference"
    );
    assert!(
        (se[0] - 0.13688648231466696).abs() < 1e-6,
        "xa SE {}",
        se[0]
    );
    assert!(
        (se[1] - 0.19112708153110064).abs() < 1e-6,
        "xb SE {}",
        se[1]
    );
    assert!(
        (r.intercept_std_error.unwrap() - 0.53418431237438180).abs() < 1e-6,
        "intercept SE"
    );
}
