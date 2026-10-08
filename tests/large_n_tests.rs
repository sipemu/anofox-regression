//! Regression tests for O(n²) memory blow-ups on tall design matrices.
//!
//! The QR / SVD based solvers used to materialise the full `n × n` Q (or U)
//! factor, which needs `8·n²` bytes: ~320 GB at n = 200 000. They now use the
//! thin factors (`n × p`). These tests would abort with an out-of-memory error
//! (or take minutes) on the old code path.

use anofox_regression::solvers::{
    BinomialRegressor, FittedRegressor, OlsRegressor, PoissonRegressor, Regressor, WlsRegressor,
};
use faer::{Col, Mat};
use std::time::{Duration, Instant};

const N: usize = 200_000;

fn design(n: usize) -> Mat<f64> {
    Mat::from_fn(n, 3, |i, j| {
        let t = (i + 1) as f64 / n as f64;
        match j {
            0 => t,
            1 => (t * 37.0).sin(),
            _ => (t * 11.0).cos(),
        }
    })
}

#[test]
fn ols_with_inference_large_n() {
    let x = design(N);
    let y = Col::from_fn(N, |i| {
        1.0 + 2.0 * x[(i, 0)] - 0.5 * x[(i, 1)] + 0.25 * x[(i, 2)] + ((i % 13) as f64 - 6.0) * 1e-3
    });

    let start = Instant::now();
    let fitted = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("OLS fit on n = 200k must succeed");
    let elapsed = start.elapsed();

    let r = fitted.result();
    assert!((r.coefficients[0] - 2.0).abs() < 1e-3);
    assert!((r.coefficients[1] + 0.5).abs() < 1e-3);
    assert!((r.coefficients[2] - 0.25).abs() < 1e-3);
    assert!((r.intercept.unwrap() - 1.0).abs() < 1e-3);
    assert!(r.r_squared > 0.999);
    assert!(r.std_errors.as_ref().unwrap().iter().all(|s| s.is_finite()));
    // Generous bound (debug build): the old full-Q path took minutes / OOMed.
    assert!(elapsed < Duration::from_secs(60), "OLS took {elapsed:?}");
}

#[test]
fn wls_with_inference_large_n() {
    let x = design(N);
    let y = Col::from_fn(N, |i| 3.0 - x[(i, 0)] + 0.1 * x[(i, 2)]);
    let w = Col::from_fn(N, |i| 1.0 + (i % 3) as f64);
    let fitted = WlsRegressor::builder()
        .with_intercept(true)
        .weights(w)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .expect("WLS fit on n = 200k must succeed");
    assert!((fitted.result().coefficients[0] + 1.0).abs() < 1e-8);
}

#[test]
#[ignore = "~30 s in debug; run with `cargo test --release -- --ignored`"]
fn glm_irls_large_n() {
    let x = design(N);
    let yc = Col::from_fn(N, |i| (i % 7) as f64);
    let yb = Col::from_fn(N, |i| (i % 2) as f64);
    let p = PoissonRegressor::log()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &yc)
        .expect("Poisson fit on n = 200k must succeed");
    assert!(p.result().intercept.unwrap().is_finite());
    let b = BinomialRegressor::logistic()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &yb)
        .expect("Binomial fit on n = 200k must succeed");
    assert!(b.result().intercept.unwrap().is_finite());
}
