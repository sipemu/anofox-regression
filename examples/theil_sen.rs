//! # Theil-Sen Regression
//!
//! Robust linear regression by the spatial median of OLS-on-subsample
//! coefficient vectors. ~29.3% breakdown point — the highest of any simple
//! single-fit robust estimator.
//!
//! Run with: `cargo run --example theil_sen`

use anofox_regression::solvers::{FittedRegressor, Regressor, TheilSenRegressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Theil-Sen Regression ===\n");

    // True line y = 1 + 2x, plus 6 large outliers.
    let n = 25;
    let x = Mat::from_fn(n, 1, |i, _| i as f64 * 0.4);
    let mut y_data: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * i as f64 * 0.4).collect();
    for &i in &[3usize, 7, 11, 15, 19, 23] {
        y_data[i] += 50.0; // gross outliers
    }
    let y = Col::from_fn(n, |i| y_data[i]);

    let fitted = TheilSenRegressor::builder()
        .with_intercept(true)
        .max_subpopulation(10_000) // exhaustive when C(n, p+1) ≤ this
        .random_state(42)
        .build()
        .fit(&x, &y)
        .expect("Theil-Sen fit failed");

    let r = fitted.result();
    println!(
        "Theil-Sen: intercept = {:.4}, slope = {:.4}   (true: 1.0, 2.0)",
        r.intercept.unwrap(),
        r.coefficients[0]
    );

    // Compare against plain OLS for context.
    use anofox_regression::solvers::OlsRegressor;
    let ols = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(false)
        .build()
        .fit(&x, &y)
        .expect("OLS fit failed");
    println!(
        "OLS (for context):    intercept = {:.4}, slope = {:.4}",
        ols.result().intercept.unwrap(),
        ols.result().coefficients[0]
    );
    println!("\nOLS slope is pulled toward the outliers; Theil-Sen ignores them.");
}
