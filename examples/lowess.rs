//! # LOWESS (Locally Weighted Scatterplot Smoothing)
//!
//! LOWESS is a non-parametric regression method that fits local weighted
//! polynomials to smooth noisy data while preserving local structure.
//!
//! ## When to Use
//! - Exploratory data analysis
//! - Smoothing noisy time series
//! - Non-linear relationship visualization
//! - Trend extraction
//!
//! ## Key Features
//! - Local polynomial regression
//! - Tricube weighting function
//! - Span parameter controls smoothness
//! - Preserves local patterns
//!
//! Run with: `cargo run --example lowess`

use anofox_regression::solvers::lowess::lowess_smooth_weights;
use faer::{Col, Mat};

fn main() {
    println!("=== LOWESS Smoothing ===\n");

    basic_smoothing();
    span_comparison();
    weight_smoothing();
}

/// Basic LOWESS smoothing
fn basic_smoothing() {
    println!("--- Basic LOWESS Smoothing ---\n");

    let n = 50;

    // Generate noisy data with underlying trend
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y_true: Vec<f64> = x.iter().map(|&xi| xi.sin() * 2.0 + xi * 0.5).collect();
    let y_noisy = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 1.7).sin()) * 0.5;
        y_true[i] + noise
    });

    // Create identity weight matrix (uniform weights)
    let weights = Mat::from_fn(n, 1, |_, _| 1.0);

    // Smooth with different spans
    let smoothed_03 = lowess_smooth_weights(&weights, 0.3);
    let smoothed_05 = lowess_smooth_weights(&weights, 0.5);

    println!("Smoothing noisy sine wave + linear trend\n");
    println!(
        "{:>6} {:>10} {:>10} {:>12} {:>12}",
        "x", "True", "Noisy", "LOWESS 0.3", "LOWESS 0.5"
    );
    println!("{}", "-".repeat(55));

    for i in [0, 10, 20, 30, 40, 49] {
        println!(
            "{:>6.2} {:>10.4} {:>10.4} {:>12.4} {:>12.4}",
            x[i],
            y_true[i],
            y_noisy[i],
            smoothed_03[(i, 0)],
            smoothed_05[(i, 0)]
        );
    }

    println!("\nNote: LOWESS smooths while preserving local structure.");
    println!("      Smaller span = more local detail, larger span = smoother.");
    println!();
}

/// Compare different span values
fn span_comparison() {
    println!("--- Span Parameter Comparison ---\n");

    let n = 40;

    // Create weight matrix
    let weights = Mat::from_fn(n, 1, |i, _| {
        // Some variation in weights
        1.0 + ((i as f64 * 0.5).sin()) * 0.3
    });

    let spans = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9];

    println!("Effect of span parameter on smoothness:\n");

    for &span in &spans {
        let smoothed = lowess_smooth_weights(&weights, span);

        // Calculate "roughness" (sum of squared second differences)
        let roughness: f64 = (1..n - 1)
            .map(|i| {
                let d2 = smoothed[(i + 1, 0)] - 2.0 * smoothed[(i, 0)] + smoothed[(i - 1, 0)];
                d2.powi(2)
            })
            .sum();

        println!("Span = {:.1}: roughness = {:.6}", span, roughness);
    }

    println!("\nNote: Larger span produces smoother output (lower roughness).");
    println!("      But too much smoothing may obscure important patterns.");
    println!();
}

/// Smoothing weight matrices for dynamic models
fn weight_smoothing() {
    println!("--- Weight Matrix Smoothing ---\n");

    // This demonstrates how LOWESS is used in dynamic linear models
    // to smooth model weights over time

    let n = 30; // Time points
    let n_models = 3; // Number of candidate models

    // Create noisy model weights (from IC-based model selection)
    let raw_weights = Mat::from_fn(n, n_models, |i, j| {
        // Base weight pattern
        let base = match j {
            0 => 0.5 - (i as f64 / n as f64) * 0.3, // Decreasing
            1 => 0.3 + (i as f64 / n as f64) * 0.2, // Increasing
            2 => 0.2,                               // Constant
            _ => 0.0,
        };
        // Add noise
        let noise = ((i as f64 * (j as f64 + 1.0) * 0.7).sin()) * 0.1;
        (base + noise).max(0.0)
    });

    // Smooth the weights
    let smoothed_weights = lowess_smooth_weights(&raw_weights, 0.4);

    println!("Smoothing model weights over time (3 candidate models):\n");
    println!(
        "{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "t", "Raw_1", "Raw_2", "Raw_3", "Sm_1", "Sm_2", "Sm_3"
    );
    println!("{}", "-".repeat(58));

    for i in [0, 5, 10, 15, 20, 25, 29] {
        println!(
            "{:>4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            i,
            raw_weights[(i, 0)],
            raw_weights[(i, 1)],
            raw_weights[(i, 2)],
            smoothed_weights[(i, 0)],
            smoothed_weights[(i, 1)],
            smoothed_weights[(i, 2)]
        );
    }

    println!("\nNote: Smoothing removes noise while preserving trends.");
    println!("      This is used in LmDynamic for stable time-varying coefficients.");
}
