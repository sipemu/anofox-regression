//! # Passive-Aggressive Regression (Online)
//!
//! Online learning algorithm (Crammer et al. 2006). On each sample the
//! model is *passive* if the epsilon-insensitive loss is zero, and
//! *aggressive* otherwise — taking the smallest update that brings the
//! constraint back to satisfaction. Includes batch `fit` and a streaming
//! `partial_fit` API.
//!
//! Run with: `cargo run --example passive_aggressive`

use anofox_regression::solvers::{
    FittedRegressor, PaLoss, PaState, PassiveAggressiveRegressor, Regressor,
};
use faer::{Col, Mat};

fn main() {
    println!("=== Passive-Aggressive Regression ===\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| -3.0 + i as f64 * 0.06);
    let y = Col::from_fn(n, |i| 1.0 + 0.8 * (-3.0 + i as f64 * 0.06));

    // Batch fit (PA-I, shuffle disabled for determinism).
    let pa = PassiveAggressiveRegressor::builder()
        .c(1.0)
        .epsilon(0.1)
        .with_intercept(true)
        .max_iter(50)
        .tolerance(1e-6)
        .shuffle(false)
        .loss(PaLoss::EpsilonInsensitive)
        .build();

    let fitted = pa.fit(&x, &y).expect("PA fit failed");
    let r = fitted.result();
    println!("Batch PA-I fit:");
    println!(
        "  intercept = {:.4}, slope = {:.4}   (true: 1.0, 0.8)",
        r.intercept.unwrap(),
        r.coefficients[0]
    );
    println!("  iterations until early-stop: {}", fitted.n_iter());

    // Streaming use: same updates, one sample at a time.
    let mut state = PaState::new(1);
    for i in 0..n {
        let row = [x[(i, 0)]];
        pa.partial_fit(&mut state, &row, y[i])
            .expect("partial_fit failed");
    }
    println!(
        "\nStreaming (1 epoch via partial_fit):  intercept = {:.4}, slope = {:.4}",
        state.intercept, state.weights[0]
    );
}
