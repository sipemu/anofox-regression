//! # Partial Least Squares (PLS) Regression
//!
//! Builds orthogonal latent components that simultaneously explain
//! variance in `X` and covariance with `y`. Particularly useful when
//! predictors are highly correlated or when `p > n`.
//!
//! Run with: `cargo run --example pls`

use anofox_regression::solvers::{FittedRegressor, PlsRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Partial Least Squares Regression ===\n");

    // 6 highly-correlated predictors (sharing a single latent factor).
    let n = 60;
    let p = 6;
    let x = Mat::from_fn(n, p, |i, j| {
        let latent = (i as f64 / 5.0).sin();
        // Each column = latent factor + small column-specific noise.
        latent + 0.05 * (i + j) as f64 * 0.1
    });
    let y = Col::from_fn(n, |i| {
        // y depends mainly on the shared latent factor.
        2.0 + 1.5 * (i as f64 / 5.0).sin()
    });

    // Fit with 2 components.
    let fitted = PlsRegressor::builder()
        .n_components(2)
        .with_intercept(true)
        .scale(false)
        .build()
        .fit(&x, &y)
        .expect("PLS fit failed");

    let r = fitted.result();
    println!("Intercept:   {:.4}", r.intercept.unwrap());
    println!("Components:  {}", fitted.n_components());
    println!("R²:          {:.4}", r.r_squared);

    let evr = fitted.explained_variance_ratio();
    println!("\nExplained-variance ratio per component:");
    for k in 0..evr.nrows() {
        println!("  comp {}: {:.4}", k, evr[k]);
    }

    // Project new data into latent space.
    let x_new = Mat::from_fn(3, p, |i, j| {
        let latent = (i as f64).sin();
        latent + 0.05 * (i + j) as f64 * 0.1
    });
    let scores = fitted.transform(&x_new);
    println!(
        "\nLatent scores for 3 new points (shape {}×{}):",
        scores.nrows(),
        scores.ncols()
    );
    for i in 0..scores.nrows() {
        print!("  ");
        for j in 0..scores.ncols() {
            print!("{:>8.4} ", scores[(i, j)]);
        }
        println!();
    }
}
