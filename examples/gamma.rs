//! # Gamma Regression
//!
//! GLM with Gamma family and log link. Convenience wrapper around
//! `TweedieRegressor` with `var_power = 2`. Use this for positive
//! continuous responses where the variance scales as the mean squared
//! (insurance claim sizes, durations, costs).
//!
//! Run with: `cargo run --example gamma`

use anofox_regression::solvers::{FittedRegressor, GammaRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Gamma Regression (log link) ===\n");

    // Mean structure: μ = exp(0.5 + 0.4 x). Approximate Gamma draws.
    let xs: Vec<f64> = (0..40).map(|i| 0.25 + i as f64 * 0.1).collect();
    let n = xs.len();
    let x = Mat::from_fn(n, 1, |i, _| xs[i]);

    // Hand-picked positive samples with mean ≈ exp(0.5 + 0.4 x)
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let mu = (0.5 + 0.4 * xi).exp();
            // alternate above/below the mean to introduce dispersion
            if i % 2 == 0 {
                mu * 1.3
            } else {
                mu * 0.7
            }
        })
        .collect();
    let y = Col::from_fn(n, |i| ys[i]);

    let fitted = GammaRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .max_iterations(50)
        .build()
        .fit(&x, &y)
        .expect("Gamma fit failed");

    let r = fitted.result();
    println!(
        "Intercept (log scale):  {:.4}   (true: 0.5)",
        r.intercept.unwrap()
    );
    println!(
        "Coefficient (log scale): {:.4}   (true: 0.4)",
        r.coefficients[0]
    );
    println!("Deviance:                {:.4}", fitted.inner().deviance);

    // Predictions at three new x values
    let x_new = Mat::from_fn(3, 1, |i, _| match i {
        0 => 1.0,
        1 => 2.5,
        _ => 4.0,
    });
    let mu_hat = fitted.predict_mu(&x_new);
    println!(
        "\nμ̂(x=1.0) = {:.3}   μ̂(x=2.5) = {:.3}   μ̂(x=4.0) = {:.3}",
        mu_hat[0], mu_hat[1], mu_hat[2]
    );
}
