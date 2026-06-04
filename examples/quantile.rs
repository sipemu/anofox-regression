//! # Quantile Regression
//!
//! Estimates conditional quantiles of the response via IRLS with
//! asymmetric weights based on the check function `ρ_τ(u) = u (τ − 1[u<0])`.
//! Choose τ to model the median (τ=0.5), the 90th percentile (τ=0.9),
//! or any other level in (0, 1).
//!
//! Run with: `cargo run --example quantile`

use anofox_regression::solvers::{FittedRegressor, QuantileRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Quantile Regression ===\n");

    // Heteroscedastic data: y = 1 + 2x + ε, with ε spread growing in x.
    let n = 80;
    let x = Mat::from_fn(n, 1, |i, _| i as f64 * 0.1);
    let noise_pattern = [
        0.4, -0.3, 0.8, -0.7, 0.2, -0.1, 0.5, -0.4, 0.3, -0.2, 0.6, -0.5, 0.1, -0.3, 0.4, -0.6,
    ];
    let y = Col::from_fn(n, |i| {
        let xi = i as f64 * 0.1;
        let spread = 0.3 + 0.5 * xi; // grows with x
        1.0 + 2.0 * xi + noise_pattern[i % noise_pattern.len()] * spread
    });

    // Fit median and the 10/90 % quantiles.
    for &tau in &[0.1_f64, 0.5, 0.9] {
        let fitted = QuantileRegressor::builder()
            .tau(tau)
            .with_intercept(true)
            .max_iterations(200)
            .tolerance(1e-8)
            .build()
            .fit(&x, &y)
            .expect("Quantile fit failed");

        let r = fitted.result();
        println!(
            "τ = {:.1}:   intercept = {:.4},   slope = {:.4},   pseudo-R² = {:.4}",
            tau,
            r.intercept.unwrap(),
            r.coefficients[0],
            fitted.pseudo_r_squared()
        );
    }

    println!("\nSpread between τ=0.9 and τ=0.1 slopes reflects increasing variance with x.");
}
