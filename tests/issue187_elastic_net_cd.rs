//! #187: `ElasticNetRegressor::fit` solves the elastic net exactly (coordinate
//! descent) instead of soft-thresholding a ridge solution once.
//!
//! Golden values from R 4.x, glmnet 5.2 on the data below (written from Rust):
//!
//! ```r
//! # A: one feature, no intercept; B: three features with intercept
//! glmnet(cbind(x, 0), y, alpha = 1, lambda = lam, standardize = FALSE,
//!        intercept = FALSE, thresh = 1e-14)
//! glmnet(x, y, alpha = alpha, lambda = lam, standardize = FALSE, thresh = 1e-14)
//! ```

use anofox_regression::solvers::{ElasticNetRegressor, FittedRegressor, Regressor};
use anofox_regression::LambdaScaling;
use faer::{Col, Mat};

const N: usize = 60;

fn noise(i: usize) -> f64 {
    (((i * 7919 + 13) % 101) as f64 / 101.0 - 0.5) * 0.6
}

fn data_a(scale: f64) -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(N, 1, |i, _| scale * (0.5 + i as f64 / N as f64));
    let y = Col::from_fn(N, |i| 2.0 * x[(i, 0)] / scale + noise(i));
    (x, y)
}

fn data_b() -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(N, 3, |i, j| {
        (((i * (j + 3)) % 11) as f64 + 0.1 * j as f64) / 3.0
    });
    let y = Col::from_fn(N, |i| 1.0 + 2.0 * x[(i, 0)] - x[(i, 2)] + noise(i) * 3.0);
    (x, y)
}

fn enet(alpha: f64, lambda: f64, scaling: LambdaScaling, intercept: bool) -> ElasticNetRegressor {
    ElasticNetRegressor::builder()
        .with_intercept(intercept)
        .alpha(alpha)
        .lambda(lambda)
        .lambda_scaling(scaling)
        .tolerance(1e-14)
        .build()
}

/// One feature without intercept: glmnet and the closed-form lasso
/// `sign(z)·max(|z|/n − λ, 0) / (‖x‖²/n)`, for X and 100·X.
#[test]
fn lasso_one_feature_matches_glmnet_and_closed_form() {
    // (scale, glmnet lambda, glmnet coefficient)
    let golden = [
        (1.0, 0.01, 1.9828194338),
        (1.0, 0.1, 1.8984480958),
        (100.0, 0.01, 0.0199210028),
        (100.0, 0.1, 0.0199125657),
    ];
    for (scale, lam, expected) in golden {
        let (x, y) = data_a(scale);
        let z: f64 = (0..N).map(|i| x[(i, 0)] * y[i]).sum::<f64>() / N as f64;
        let v: f64 = (0..N).map(|i| x[(i, 0)].powi(2)).sum::<f64>() / N as f64;
        let closed = z.signum() * (z.abs() - lam).max(0.0) / v;
        assert!((closed - expected).abs() < 1e-9);

        let glmnet = enet(1.0, lam, LambdaScaling::Glmnet, false)
            .fit(&x, &y)
            .unwrap();
        // Raw objective ||y − Xβ||² + λ||β||₁ is glmnet's with λ_raw = 2nλ.
        let raw = enet(1.0, 2.0 * N as f64 * lam, LambdaScaling::Raw, false)
            .fit(&x, &y)
            .unwrap();
        for fit in [glmnet, raw] {
            let b = fit.result().coefficients[0];
            assert!(
                (b - expected).abs() < 1e-9 * expected.abs().max(1.0),
                "scale {scale}, lambda {lam}: {b} vs {expected}"
            );
        }
    }
}

/// Lasso and elastic net with intercept against glmnet.
#[test]
fn elastic_net_with_intercept_matches_glmnet() {
    // (alpha, lambda, intercept, coefficients)
    let golden = [
        (1.0, 0.05, 0.87151505, [2.01578529, 0.0, -0.95438228]),
        (1.0, 0.3, 0.84873195, [1.79123047, 0.0, -0.73177886]),
        (
            0.5,
            0.05,
            0.90203170,
            [2.02031874, -0.01574327, -0.96110797],
        ),
        (0.5, 0.3, 0.94428854, [1.82749592, 0.0, -0.82023038]),
    ];
    let (x, y) = data_b();
    for (alpha, lam, b0, b) in golden {
        let fit = enet(alpha, lam, LambdaScaling::Glmnet, true)
            .fit(&x, &y)
            .unwrap();
        let r = fit.result();
        assert!(
            (r.intercept.unwrap() - b0).abs() < 1e-7,
            "alpha {alpha}, lambda {lam}"
        );
        for (j, &want) in b.iter().enumerate() {
            assert!(
                (r.coefficients[j] - want).abs() < 1e-7,
                "alpha {alpha}, lambda {lam}, coef {j}: {} vs {}",
                r.coefficients[j],
                want
            );
        }
        // Exact zeros of the lasso are kept.
        assert_eq!(fit.n_nonzero(), b.iter().filter(|c| **c != 0.0).count());
    }
}

/// With glmnet scaling, `fit` is the last column of `path` at that `λ`.
#[test]
fn fit_agrees_with_path() {
    let (x, y) = data_b();
    for alpha in [0.2, 0.5, 1.0] {
        let lambdas = [1.0, 0.3, 0.1, 0.03];
        let model = enet(alpha, 0.03, LambdaScaling::Glmnet, true);
        let path = model.path(&x, &y, &lambdas).unwrap();
        let fit = model.fit(&x, &y).unwrap();
        let r = fit.result();
        assert!((r.intercept.unwrap() - path.intercepts[3]).abs() < 1e-8);
        for j in 0..3 {
            assert!((r.coefficients[j] - path.coefficients[(j, 3)]).abs() < 1e-8);
        }
    }
}

/// Raw objective `||y − Xβ||² + λ(α||β||₁ + (1−α)||β||²)`: KKT conditions
/// hold at the solution.
#[test]
fn raw_objective_satisfies_kkt() {
    let (x, y) = data_b();
    let (alpha, lambda) = (0.4, 25.0);
    let fit = enet(alpha, lambda, LambdaScaling::Raw, true)
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    let res = Col::from_fn(N, |i| {
        y[i] - r.intercept.unwrap() - (0..3).map(|j| x[(i, j)] * r.coefficients[j]).sum::<f64>()
    });
    assert!(res.iter().sum::<f64>().abs() < 1e-8);
    for j in 0..3 {
        let b = r.coefficients[j];
        // ∂/∂β_j of RSS + λ(1−α)β² is −2x'r + 2λ(1−α)β; L1 subgradient λα·sign.
        let g = -2.0 * (0..N).map(|i| x[(i, j)] * res[i]).sum::<f64>()
            + 2.0 * lambda * (1.0 - alpha) * b;
        if b != 0.0 {
            assert!(
                (g + lambda * alpha * b.signum()).abs() < 1e-6,
                "coef {j}: {g}"
            );
        } else {
            assert!(g.abs() <= lambda * alpha + 1e-6, "coef {j}: {g}");
        }
    }
}
