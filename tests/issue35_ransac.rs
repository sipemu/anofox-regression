//! Issue #35 (R reference values from R 4.6.1 / MASS / lme4): RANSAC result leaves adj_r_squared / rmse / mse at 0 and no inference.
use anofox_regression::solvers::{FittedRegressor, RansacRegressor, Regressor};
use faer::{Col, Mat};

#[test]
fn ransac_reports_fit_statistics_of_the_inlier_ols() {
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(n, |k| {
        let i = k + 1;
        let base = 1.0 + 2.0 * i as f64 + (((i * 7) % 5) as f64 - 2.0) * 0.3;
        if i == 4 || i == 11 || i == 17 {
            base + 25.0
        } else {
            base
        }
    });
    let f = RansacRegressor::builder()
        .with_intercept(true)
        .residual_threshold(2.0)
        .random_state(42)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    println!("n_inliers {} intercept {:?} slope {} r2 {} adj_r2 {} rmse {} mse {} f {} f_p {} std_errors {:?}",
        f.n_inliers(), r.intercept, r.coefficients[0], r.r_squared, r.adj_r_squared, r.rmse, r.mse,
        r.f_statistic, r.f_pvalue, r.std_errors);
    assert_eq!(f.n_inliers(), 17);
    // coefficients are already the inlier OLS (R: lm on the 17 inlier rows)
    assert!((r.intercept.unwrap() - 1.0909975669099783).abs() < 1e-9);
    assert!((r.coefficients[0] - 1.9862530413625301).abs() < 1e-9);
    // R: summary(lm(y[inl] ~ x[inl]))
    assert!(
        (r.rmse - 0.44666920869378707).abs() < 1e-9,
        "rmse {} (R sigma 0.44666920869378707)",
        r.rmse
    );
    assert!(
        (r.adj_r_squared - 0.99860732006386344).abs() < 1e-9,
        "adj_r_squared {}",
        r.adj_r_squared
    );
    let se = r
        .std_errors
        .as_ref()
        .expect("std_errors should be available");
    assert!((se[0] - 0.018543168475545296).abs() < 1e-9);
}
