//! # Isotonic Regression
//!
//! Fits a monotonically increasing (or decreasing) step function to (x, y)
//! using the Pool Adjacent Violators Algorithm (PAVA). Useful when the
//! relationship is known to be monotone but you don't want to assume a
//! parametric form.
//!
//! Run with: `cargo run --example isotonic`

use anofox_regression::solvers::{IsotonicRegressor, OutOfBounds};
use faer::Col;

fn main() {
    println!("=== Isotonic Regression ===\n");

    // Noisy, mostly-increasing data with two PAVA violations.
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let ys = [0.5, 1.2, 0.9, 2.1, 2.0, 3.5, 3.2, 4.6, 5.1, 6.0];

    let x = Col::from_fn(xs.len(), |i| xs[i]);
    let y = Col::from_fn(ys.len(), |i| ys[i]);

    let fitted = IsotonicRegressor::builder()
        .increasing(true)
        .out_of_bounds(OutOfBounds::Clip)
        .build()
        .fit_1d(&x, &y)
        .expect("Isotonic fit failed");

    println!("Fitted values (PAVA-pooled, non-decreasing):");
    for i in 0..xs.len() {
        println!(
            "  x = {:.1}   y = {:.2}   ŷ = {:.3}",
            xs[i],
            ys[i],
            fitted.fitted_values()[i]
        );
    }
    println!("\nKnot count: {}", fitted.x_thresholds().nrows());

    // Predict at new x values (out-of-bounds → Clip to fitted range).
    let x_new = Col::from_fn(3, |i| match i {
        0 => -2.0,
        1 => 4.5,
        _ => 11.0,
    });
    let preds = fitted.predict_1d(&x_new);
    println!("\nPredictions:");
    println!("  ŷ(-2.0) = {:.3}  (clipped to start of range)", preds[0]);
    println!("  ŷ( 4.5) = {:.3}  (between knots → step value)", preds[1]);
    println!("  ŷ(11.0) = {:.3}  (clipped to end of range)", preds[2]);
}
