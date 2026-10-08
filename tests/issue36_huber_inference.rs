//! Issue #36 (R reference values from R 4.6.1 / MASS / lme4): HuberRegressor computes no standard errors / inference.
use anofox_regression::solvers::{FittedRegressor, HuberRegressor, Regressor};
use faer::{Col, Mat};

#[test]
fn huber_standard_errors_match_mass_rlm_xtx() {
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
    let mut y = Col::from_fn(n, |k| {
        let xv = (k + 1) as i64;
        1.0 + 2.0 * xv as f64 + (((xv * 7) % 5) as f64 - 2.0) * 0.4
    });
    y[5] = 40.0;
    y[14] = 5.0;
    let f = HuberRegressor::builder()
        .epsilon(1.35)
        .alpha(0.0)
        .with_intercept(true)
        .max_iterations(500)
        .tolerance(1e-12)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    println!(
        "coef {:?} {} scale {} std_errors {:?} intercept_std_error {:?}",
        r.intercept,
        r.coefficients[0],
        f.scale(),
        r.std_errors,
        r.intercept_std_error
    );
    // R: MASS::rlm(y ~ x, psi = psi.huber, k = 1.35, scale.est = "MAD")
    assert!((r.coefficients[0] - 1.9769296831089778).abs() < 1e-6);
    assert!((r.intercept.unwrap() - 1.2866827718001757).abs() < 1e-5);
    // summary(fit, method = "XtX") standard errors
    let se = r
        .std_errors
        .as_ref()
        .expect("Huber fit exposes no std_errors");
    assert!((se[0] - 0.029647934601485383).abs() < 1e-6);
    assert!((r.intercept_std_error.unwrap() - 0.355157012812956197).abs() < 1e-5);
}
