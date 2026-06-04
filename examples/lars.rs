//! # Least Angle Regression (LARS) and LassoLars
//!
//! LARS computes the full regularization path in O(n p²) time — the same
//! cost as a single OLS fit. The Lasso variant additionally drops
//! predictors from the active set when their coefficient passes through
//! zero, reproducing the exact Lasso path.
//!
//! Run with: `cargo run --example lars`

use anofox_regression::solvers::{FittedRegressor, LarsMethod, LarsRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== LARS / LassoLars ===\n");

    // 4 features, only features 0 and 2 carry signal.
    let n = 60;
    let p = 4;
    let x = Mat::from_fn(n, p, |i, j| {
        let t = i as f64 / 5.0;
        match j {
            0 => t.sin(),
            1 => (t * 0.13).cos(), // noise direction
            2 => (t * 0.7).exp().ln(),
            _ => (t * 0.9).sin() * 0.3, // noise direction
        }
    });
    let y = Col::from_fn(n, |i| {
        let mut v = 0.5;
        v += 1.5 * x[(i, 0)];
        v += 0.0 * x[(i, 1)]; // explicitly zero
        v += -2.0 * x[(i, 2)];
        v += 0.0 * x[(i, 3)]; // explicitly zero
        v
    });

    // --- Plain LARS, full path → equals OLS solution -----------------------
    let lars = LarsRegressor::builder()
        .method(LarsMethod::Lar)
        .fit_intercept(true)
        .n_nonzero_coefs(p)
        .build()
        .fit(&x, &y)
        .expect("LARS fit failed");

    let r = lars.result();
    println!("Plain LARS (full path):");
    println!("  intercept = {:.4}   (true: 0.5)", r.intercept.unwrap());
    for j in 0..p {
        println!(
            "  β[{}] = {:.4}   (true: {})",
            j,
            r.coefficients[j],
            [1.5, 0.0, -2.0, 0.0][j]
        );
    }

    // --- LARS truncated to two predictors ----------------------------------
    let lars2 = LarsRegressor::builder()
        .method(LarsMethod::Lar)
        .fit_intercept(true)
        .n_nonzero_coefs(2)
        .build()
        .fit(&x, &y)
        .expect("LARS truncated fit failed");
    let r2 = lars2.result();
    println!("\nLARS truncated at 2 predictors:");
    for j in 0..p {
        println!("  β[{}] = {:.4}", j, r2.coefficients[j]);
    }

    // --- LassoLars at α corresponding to a partial-shrinkage point ----------
    // Our α is the max-correlation termination level; equivalent to
    // sklearn's α via α_ours = n * α_sklearn.
    let lasso = LarsRegressor::builder()
        .method(LarsMethod::Lasso)
        .fit_intercept(true)
        .alpha(0.05 * n as f64)
        .build()
        .fit(&x, &y)
        .expect("LassoLars fit failed");
    let rl = lasso.result();
    println!("\nLassoLars at α = 0.05 (sklearn-scale):");
    for j in 0..p {
        println!("  β[{}] = {:.4}", j, rl.coefficients[j]);
    }
    println!("  alphas along the path: {:?}", lasso.alphas());
}
