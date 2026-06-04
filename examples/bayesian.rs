//! # Bayesian Ridge and ARD Regression
//!
//! `BayesianRidge` places a Gaussian prior on the coefficients and tunes
//! the noise precision α and weight precision λ jointly via type-II
//! maximum likelihood (evidence approximation).
//!
//! `ArdRegression` extends this with a *per-feature* precision λ_j: large
//! λ_j shrinks the corresponding β_j to zero, giving automatic feature
//! relevance pruning.
//!
//! Run with: `cargo run --example bayesian`

use anofox_regression::solvers::{ArdRegression, BayesianRidge, FittedRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Bayesian Ridge ===\n");
    bayesian_ridge();
    println!("\n=== ARD Regression ===\n");
    ard();
}

fn bayesian_ridge() {
    let n = 80;
    let p = 3;
    let x = Mat::from_fn(n, p, |i, j| {
        let t = i as f64 / 8.0;
        match j {
            0 => t.sin(),
            1 => (t * 0.5).cos(),
            _ => (t * 0.21).exp().ln(),
        }
    });
    let true_beta = [1.5, -0.7, 2.0];
    let y = Col::from_fn(n, |i| {
        let mut v = 0.3;
        for j in 0..p {
            v += x[(i, j)] * true_beta[j];
        }
        v
    });

    let fitted = BayesianRidge::builder()
        .fit_intercept(true)
        .max_iter(300)
        .tolerance(1e-3)
        .build()
        .fit(&x, &y)
        .expect("Bayesian Ridge fit failed");

    let r = fitted.result();
    println!("Intercept: {:.4}   (true: 0.3)", r.intercept.unwrap());
    for j in 0..p {
        println!(
            "  β[{}] = {:.4}   (true: {})",
            j, r.coefficients[j], true_beta[j]
        );
    }
    println!("Noise precision α  = {:.4}", fitted.alpha());
    println!("Weight precision λ = {:.4}", fitted.lambda());
}

fn ard() {
    // 6 features; only 0, 2, 5 carry signal — the others should be pruned.
    let n = 100;
    let p = 6;
    let x = Mat::from_fn(n, p, |i, j| {
        let t = i as f64 / 10.0;
        match j {
            0 => t.sin(),
            1 => (t * 0.13).cos(),
            2 => (t * 0.7).exp().ln(),
            3 => (t * 0.21).sin(),
            4 => (t * 1.1).cos(),
            _ => t * 0.05,
        }
    });
    let true_beta = [2.0, 0.0, -1.5, 0.0, 0.0, 0.7];
    let y = Col::from_fn(n, |i| {
        let mut v = 0.0;
        for j in 0..p {
            v += x[(i, j)] * true_beta[j];
        }
        v
    });

    let fitted = ArdRegression::builder()
        .fit_intercept(true)
        .max_iter(300)
        .tolerance(1e-3)
        .threshold_lambda(10_000.0)
        .build()
        .fit(&x, &y)
        .expect("ARD fit failed");

    let r = fitted.result();
    println!("Intercept: {:.4}", r.intercept.unwrap());
    for j in 0..p {
        let pruned = if fitted.lambdas()[j] > 10_000.0 {
            "  (pruned)"
        } else {
            ""
        };
        println!(
            "  β[{}] = {:.4}   (true: {:>5})   λ_{} = {:.2e}{}",
            j,
            r.coefficients[j],
            true_beta[j],
            j,
            fitted.lambdas()[j],
            pruned
        );
    }
    println!("Noise precision α = {:.4}", fitted.alpha());
}
