//! # Streaming Ridge / OLS via `MomentAccumulator`
//!
//! Use this when the full `N × p` design matrix is too large to keep in
//! memory (e.g. a 569k-series panel × 17M rows × 20 features). The fit
//! depends only on the rank-`p` sufficient statistics
//! `(n, Σx, Σy, XᵀX, Xᵀy)`, which are additive over row chunks — so you
//! can accumulate them across files, threads, or DB cursors and solve
//! once at the end.
//!
//! Run with: `cargo run --example streaming_ridge`

use anofox_regression::solvers::{
    FittedRegressor, MomentAccumulator, OlsRegressor, Regressor, RidgeRegressor,
};
use faer::{Col, Mat};

fn main() {
    println!("=== Streaming Ridge via MomentAccumulator ===\n");

    // Build a small synthetic panel: 1000 rows × 3 features.
    let n = 1000;
    let p = 3;
    let true_beta = [1.5_f64, -0.7, 0.4];
    let intercept = 0.5;
    let x = Mat::from_fn(n, p, |i, j| match j {
        0 => (i as f64 / 7.0).sin(),
        1 => (i as f64 / 11.0).cos(),
        _ => (i as f64 / 13.0).tanh(),
    });
    let y = Col::from_fn(n, |i| {
        let mut v = intercept;
        for j in 0..p {
            v += x[(i, j)] * true_beta[j];
        }
        v + ((i as f64 * 1.7).sin()) * 0.05
    });

    // --- Whole-panel reference -------------------------------------------
    let model = RidgeRegressor::builder()
        .lambda(0.1)
        .with_intercept(true)
        .compute_inference(false)
        .build();

    let whole = model.fit(&x, &y).expect("whole-panel fit");
    println!("Whole-panel fit (Ridge λ=0.1):");
    print_fit(&whole);

    // --- Streamed fit ----------------------------------------------------
    // Simulate row chunks arriving from disk / DB / a parallel worker.
    let mut acc = MomentAccumulator::new(p);
    let mut row = vec![0.0_f64; p];
    for i in 0..n {
        for j in 0..p {
            row[j] = x[(i, j)];
        }
        acc.push_row(&row, y[i]).unwrap();
    }
    println!("\nMomentAccumulator state after {} rows pushed:", acc.n());
    println!("  Σx  = {:?}", to_vec(acc.sum_x()));
    println!("  Σy  = {:.4}", acc.sum_y());
    println!("  XᵀX shape: {} × {}", acc.xtx().nrows(), acc.xtx().ncols());

    let streamed = model.fit_from_accumulator(&acc).expect("streamed fit");
    println!("\nStreamed fit (Ridge λ=0.1, identical model):");
    print_fit(&streamed);

    // --- Chunk-and-merge -------------------------------------------------
    // Real callers will usually accumulate in parallel; show the merge API.
    let chunks = [(0_usize, 250), (250, 500), (500, 750), (750, 1000)];
    let mut merged = MomentAccumulator::new(p);
    for &(lo, hi) in &chunks {
        let mut part = MomentAccumulator::new(p);
        for i in lo..hi {
            for j in 0..p {
                row[j] = x[(i, j)];
            }
            part.push_row(&row, y[i]).unwrap();
        }
        merged.merge(&part).unwrap();
    }
    let merged_fit = model.fit_from_accumulator(&merged).expect("merged fit");
    println!("\n4-chunk merged fit (must match streamed to machine precision):");
    print_fit(&merged_fit);

    // --- OLS variant ----------------------------------------------------
    let ols = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(false)
        .build();
    let ols_streamed = ols.fit_from_accumulator(&acc).expect("OLS from moments");
    println!("\nStreamed OLS (λ=0):");
    print_fit(&ols_streamed);
}

fn print_fit<F: FittedRegressor>(f: &F) {
    let r = f.result();
    println!("  intercept = {:>9.6}", r.intercept.unwrap_or(f64::NAN));
    for j in 0..r.coefficients.nrows() {
        println!("  β[{}]      = {:>9.6}", j, r.coefficients[j]);
    }
}

fn to_vec(c: &Col<f64>) -> Vec<f64> {
    (0..c.nrows()).map(|i| c[i]).collect()
}
