//! Issue #38 (R reference values from R 4.6.1 / MASS / lme4): without an intercept R^2 / adj R^2 / F use the centred TSS (R uses the uncentred one).
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor, WlsRegressor};
use faer::{Col, Mat};

fn data() -> (Mat<f64>, Col<f64>) {
    let yv = [3.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.2, 19.9];
    (
        Mat::from_fn(10, 1, |i, _| (i + 1) as f64),
        Col::from_fn(10, |i| yv[i] + 5.0),
    )
}

#[test]
fn ols_no_intercept_matches_summary_lm() {
    let (x, y) = data();
    let f = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    println!("OLS y~0+x: r2 {} adj {} F {}  (R: 0.97794411910884449 0.97549346567649386 399.05443429870200589)",
             r.r_squared, r.adj_r_squared, r.f_statistic);
    assert!((r.r_squared - 0.97794411910884449).abs() < 1e-9);
    assert!((r.adj_r_squared - 0.97549346567649386).abs() < 1e-9);
    assert!((r.f_statistic - 399.05443429870200589).abs() < 1e-6);
}

#[test]
fn wls_no_intercept_matches_summary_lm() {
    let (x, y) = data();
    let w = Col::from_fn(10, |i| if i % 2 == 0 { 1.0 } else { 2.0 });
    let f = WlsRegressor::builder()
        .with_intercept(false)
        .weights(w)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    println!("WLS y~0+x: r2 {} adj {} F {}  (R: 0.98089334384990512 0.97877038205545008 462.04003595915855840)",
             r.r_squared, r.adj_r_squared, r.f_statistic);
    assert!((r.r_squared - 0.98089334384990512).abs() < 1e-9);
    assert!((r.adj_r_squared - 0.97877038205545008).abs() < 1e-9);
    assert!((r.f_statistic - 462.04003595915855840).abs() < 1e-6);
}
