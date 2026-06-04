//! # Huber Regression
//!
//! Robust linear regression that downweights observations whose residuals
//! exceed `epsilon * scale`. Behaves like OLS on the inlier set and like an
//! L1 estimator on the outliers, giving stable estimates when the data is
//! contaminated by gross errors.
//!
//! Run with: `cargo run --example huber`

use anofox_regression::solvers::{FittedRegressor, HuberRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Huber Regression ===\n");

    // True model: y = 2 + 1.5 x. Five samples are gross outliers.
    let n = 30;
    let x = Mat::from_fn(n, 1, |i, _| i as f64 * 0.5);
    let mut y_data: Vec<f64> = (0..n).map(|i| 2.0 + 1.5 * i as f64 * 0.5).collect();
    for &i in &[5usize, 10, 15, 20, 25] {
        y_data[i] += 30.0;
    }
    let y = Col::from_fn(n, |i| y_data[i]);

    let fitted = HuberRegressor::builder()
        .epsilon(1.35) // sklearn default
        .alpha(0.0001)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .expect("Huber fit failed");

    let r = fitted.result();
    println!(
        "Coefficients:        intercept = {:.4}, slope = {:.4}",
        r.intercept.unwrap(),
        r.coefficients[0]
    );
    println!("MAD-based scale (σ): {:.4}", fitted.scale());
    println!("Outliers flagged:    {} / {}", fitted.n_outliers(), n);

    // Show which observations were flagged.
    let flagged: Vec<usize> = fitted
        .outliers()
        .iter()
        .enumerate()
        .filter(|(_, &is_out)| is_out)
        .map(|(i, _)| i)
        .collect();
    println!("Outlier indices:     {:?}", flagged);
    println!("(Planted outliers were at indices [5, 10, 15, 20, 25].)");
}
