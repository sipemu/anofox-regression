//! # RANSAC Regression
//!
//! Random sample consensus: repeatedly sample a minimal subset, fit OLS on
//! it, classify all observations as inliers if their residual is below a
//! threshold, and keep the subset with the largest inlier set. Final fit
//! is OLS on the best consensus set.
//!
//! Run with: `cargo run --example ransac`

use anofox_regression::solvers::{FittedRegressor, RansacRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== RANSAC Regression ===\n");

    // 30 inliers near y = 0.5 + 0.7 x; 10 gross outliers.
    let n_in = 30;
    let n_out = 10;
    let n = n_in + n_out;

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    // inliers
    for i in 0..n_in {
        let xi = i as f64 * 0.2;
        xs.push(xi);
        ys.push(0.5 + 0.7 * xi);
    }
    // outliers
    for i in 0..n_out {
        xs.push(i as f64 * 0.4);
        ys.push(20.0 + i as f64 * 2.0);
    }

    let x = Mat::from_fn(n, 1, |i, _| xs[i]);
    let y = Col::from_fn(n, |i| ys[i]);

    let fitted = RansacRegressor::builder()
        .min_samples(2)
        .residual_threshold(0.5)
        .max_trials(200)
        .random_state(42)
        .build()
        .fit(&x, &y)
        .expect("RANSAC fit failed");

    let r = fitted.result();
    println!("Intercept: {:.4}   (true: 0.5)", r.intercept.unwrap());
    println!("Slope:     {:.4}   (true: 0.7)", r.coefficients[0]);
    println!("Inliers:   {} / {}", fitted.n_inliers(), n);
    println!("Trials:    {}", fitted.n_trials());

    let detected_outliers: Vec<usize> = fitted
        .inlier_mask()
        .iter()
        .enumerate()
        .filter(|(_, &b)| !b)
        .map(|(i, _)| i)
        .collect();
    println!("Detected outliers (indices): {:?}", detected_outliers);
    println!("(Planted outliers were the last {} observations.)", n_out);
}
