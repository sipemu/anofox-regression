//! Exact leave-one-out diagnostics of ridge regression, `FittedRidge::loo`
//! (anofox-statistics#235), checked against explicit refits without each row.

use anofox_regression::diagnostics::{augment, press};
use anofox_regression::prelude::*;
use anofox_regression::LambdaScaling;
use faer::prelude::Solve;
use faer::{Col, Mat};

const TOL: f64 = 1e-10;

/// A deterministic, mildly collinear design with noise.
fn data(n: usize, p: usize) -> (Mat<f64>, Col<f64>) {
    let noise = |i: usize, k: usize| ((i * 7919 + k * 104_729) % 1000) as f64 / 1000.0 - 0.5;
    let x = Mat::from_fn(n, p, |i, j| {
        let t = i as f64 / n as f64;
        (j as f64 + 1.0) * t + 0.3 * noise(i, j) + if j > 0 { 0.5 * t * t } else { 0.0 }
    });
    let y = Col::from_fn(n, |i| {
        2.0 + (0..p).map(|j| (j as f64 - 1.0) * x[(i, j)]).sum::<f64>() + noise(i, 99)
    });
    (x, y)
}

fn drop_row(x: &Mat<f64>, y: &Col<f64>, i: usize) -> (Mat<f64>, Col<f64>) {
    let n = x.nrows();
    let idx = |r: usize| if r < i { r } else { r + 1 };
    (
        Mat::from_fn(n - 1, x.ncols(), |r, j| x[(idx(r), j)]),
        Col::from_fn(n - 1, |r| y[idx(r)]),
    )
}

fn ridge(lambda: f64, intercept: bool) -> RidgeRegressor {
    RidgeRegressor::builder()
        .with_intercept(intercept)
        .lambda(lambda)
        .build()
}

/// `y_i - ŷ_(i)` by refitting without row `i` with the raw penalty `lambda`.
fn refit_loo(x: &Mat<f64>, y: &Col<f64>, lambda: f64, intercept: bool) -> Vec<f64> {
    (0..x.nrows())
        .map(|i| {
            let (xi, yi) = drop_row(x, y, i);
            let fit = ridge(lambda, intercept).fit(&xi, &yi).unwrap();
            let row = Mat::from_fn(1, x.ncols(), |_, j| x[(i, j)]);
            y[i] - fit.predict(&row)[0]
        })
        .collect()
}

fn check(label: &str, x: &Mat<f64>, y: &Col<f64>, lambda: f64, intercept: bool) {
    let fit = ridge(lambda, intercept).fit(x, y).unwrap();
    let loo = fit.loo(x, y).unwrap();
    let want = refit_loo(x, y, lambda, intercept);
    let mut max = 0.0_f64;
    for (i, &w) in want.iter().enumerate() {
        let d = (loo.loo_residual[i] - w).abs();
        assert!(
            d <= TOL * w.abs().max(1.0),
            "{label}[{i}]: closed form {:.15e}, refit {w:.15e}",
            loo.loo_residual[i]
        );
        assert!((loo.loo_fitted[i] - (y[i] - loo.loo_residual[i])).abs() < 1e-12);
        max = max.max(d);
    }
    eprintln!("{label}: max |closed form - refit| = {max:.3e}");
    let press: f64 = want.iter().map(|e| e * e).sum();
    assert!((loo.press.press - press).abs() <= TOL * press);
    assert_eq!(loo.press.n, x.nrows());
    assert!((loo.press.press_rmse - (press / x.nrows() as f64).sqrt()).abs() < 1e-10);
    let ybar = if intercept {
        y.iter().sum::<f64>() / y.nrows() as f64
    } else {
        0.0
    };
    let sst: f64 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum();
    assert!((loo.press.loo_r_squared - (1.0 - press / sst)).abs() < 1e-10);
    assert!(loo.press.loo_r_squared < fit.result().r_squared);
}

#[test]
fn ridge_loo_matches_refits_with_intercept() {
    let (x, y) = data(40, 3);
    for lambda in [0.01, 0.5, 3.0, 50.0] {
        check(&format!("ridge lambda={lambda}"), &x, &y, lambda, true);
    }
}

#[test]
fn ridge_loo_matches_refits_without_intercept() {
    let (x, y) = data(40, 3);
    for lambda in [0.1, 5.0] {
        check(&format!("ridge0 lambda={lambda}"), &x, &y, lambda, false);
    }
}

#[test]
fn ridge_loo_matches_refits_with_more_features_than_rows() {
    let (x, y) = data(8, 12);
    check("ridge p>n", &x, &y, 1.0, true);
    check("ridge0 p>n", &x, &y, 1.0, false);
}

#[test]
fn intercept_is_unpenalised_in_the_hat_matrix() {
    // h = X̃ (X̃'X̃ + λP)⁻¹ X̃', P = diag(0, I): built explicitly.
    let (x, y) = data(25, 2);
    let lambda = 2.0;
    let loo = ridge(lambda, true)
        .fit(&x, &y)
        .unwrap()
        .loo(&x, &y)
        .unwrap();
    let n = x.nrows();
    let xt = Mat::from_fn(n, 3, |i, j| if j == 0 { 1.0 } else { x[(i, j - 1)] });
    let mut g = xt.transpose() * &xt;
    for j in 1..3 {
        g[(j, j)] += lambda;
    }
    let ginv = g.partial_piv_lu().solve(Mat::<f64>::identity(3, 3));
    let h = &xt * &ginv * xt.transpose();
    for i in 0..n {
        assert!((loo.leverage[i] - h[(i, i)]).abs() < 1e-12, "h[{i}]");
    }
    // Trace = effective degrees of freedom, between 1 and 1 + p.
    let tr: f64 = loo.leverage.iter().sum();
    assert!(tr > 1.0 && tr < 3.0);
}

#[test]
fn glmnet_scaling_holds_the_effective_penalty_fixed() {
    let (x, y) = data(30, 3);
    let lambda = 0.05;
    let fit = RidgeRegressor::builder()
        .lambda(lambda)
        .lambda_scaling(LambdaScaling::Glmnet)
        .build()
        .fit(&x, &y)
        .unwrap();
    let loo = fit.loo(&x, &y).unwrap();
    // Same as a raw-scale fit with lambda * n.
    let want = refit_loo(&x, &y, lambda * 30.0, true);
    for (i, &w) in want.iter().enumerate() {
        assert!((loo.loo_residual[i] - w).abs() <= TOL * w.abs().max(1.0));
    }
}

#[test]
fn lambda_zero_is_ols() {
    let (x, y) = data(20, 2);
    let fit = ridge(0.0, true).fit(&x, &y).unwrap();
    let loo = fit.loo(&x, &y).unwrap();
    let ols = OlsRegressor::builder().build().fit(&x, &y).unwrap();
    let a = augment(&ols, &x, &y, None).unwrap();
    let p = press(&ols, &x, &y, None).unwrap();
    for i in 0..20 {
        assert!((loo.leverage[i] - a.leverage[i]).abs() < 1e-12);
        assert!((loo.loo_residual[i] - a.loo_residual[i]).abs() < 1e-10);
    }
    assert!((loo.press.press - p.press).abs() < 1e-10 * p.press);
}

#[test]
fn loo_needs_the_training_rows() {
    let (x, y) = data(20, 2);
    let fit = ridge(1.0, true).fit(&x, &y).unwrap();
    let short = Col::from_fn(19, |i| y[i]);
    assert!(fit.loo(&x, &short).is_err());
    let x3 = Mat::from_fn(20, 3, |i, j| x[(i, j.min(1))]);
    assert!(fit.loo(&x3, &y).is_err());
}
