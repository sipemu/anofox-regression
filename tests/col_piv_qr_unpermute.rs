//! Regression tests for the column-pivoted QR unpermute step.
//!
//! `col_piv_qr` factorises `A * P = Q * R`, so a back-substitution result belongs to
//! original column `perm[i]` — the *forward* permutation. Writing it through
//! `perm.inverse()` scatters the coefficients into the wrong slots.
//!
//! The bug this guards against was invisible to the rest of the suite because the
//! pivot permutation of a well-scaled design is usually the identity, and an identity
//! (or any involution) is its own inverse. Every design below therefore has columns on
//! **deliberately different scales**, which is what forces a non-trivial pivot.
//!
//! Each design has an exact, strictly-positive solution, so a correct non-negative
//! least squares must reproduce it with zero residual.

use anofox_regression::solvers::{BlsRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

/// Deterministic, dependency-free PRNG so the tests are reproducible.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f64) / ((1u64 << 31) as f64)
}

/// Build a design whose columns span three orders of magnitude, together with the
/// response implied by `truth`.
fn scaled_design(n: usize, truth: &[f64], seed: &mut u64) -> (Mat<f64>, Col<f64>) {
    let k = truth.len();
    let scales: Vec<f64> = (0..k).map(|j| 10f64.powi((j % 3) as i32)).collect();
    let cells: Vec<f64> = (0..n * k).map(|_| lcg(seed)).collect();
    let x = Mat::from_fn(n, k, |i, j| (0.1 + 20.0 * cells[i * k + j]) * scales[j]);
    let y = Col::from_fn(n, |i| (0..k).map(|j| truth[j] * x[(i, j)]).sum());
    (x, y)
}

#[test]
fn nnls_recovers_an_exact_non_negative_solution_under_column_pivoting() {
    let truth = [2.941, 1.227, 2.164, 3.252, 3.501, 0.231];
    let mut seed = 42u64;
    let (x, y) = scaled_design(40, &truth, &mut seed);

    let fit = BlsRegressor::nnls()
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed on a well-conditioned design");

    for (j, expected) in truth.iter().enumerate() {
        assert!(
            (fit.coefficients()[j] - expected).abs() < 1e-6,
            "coefficient {j}: got {}, expected {expected}. A permuted-looking result \
             here means the QR unpermute is writing through the inverse permutation.",
            fit.coefficients()[j]
        );
    }
}

#[test]
fn nnls_residual_is_zero_whenever_a_feasible_exact_solution_exists() {
    // Mechanism-independent: whatever the solver does internally, a design with an
    // exact non-negative solution must come back with no residual.
    let mut seed = 7u64;
    let mut worst_rss = 0.0f64;

    for _ in 0..50 {
        let truth: Vec<f64> = (0..6).map(|_| 0.1 + 5.0 * lcg(&mut seed)).collect();
        let (x, y) = scaled_design(40, &truth, &mut seed);

        let fit = BlsRegressor::nnls()
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .expect("fit");

        let rss: f64 = (0..x.nrows())
            .map(|i| {
                let pred: f64 = (0..truth.len())
                    .map(|j| fit.coefficients()[j] * x[(i, j)])
                    .sum();
                (y[i] - pred).powi(2)
            })
            .sum();
        worst_rss = worst_rss.max(rss);
    }

    assert!(
        worst_rss < 1e-12,
        "worst residual sum of squares across 50 exactly-solvable designs was \
         {worst_rss:.3e}; it must be ~0"
    );
}

#[test]
fn bounded_least_squares_recovers_an_interior_solution_under_column_pivoting() {
    // The explicit-bounds path shares the same solve, so it needs its own guard.
    let truth = [1.5, 0.75, 2.25, 0.5];
    let mut seed = 99u64;
    let (x, y) = scaled_design(30, &truth, &mut seed);

    let fit = BlsRegressor::builder()
        .with_intercept(false)
        .lower_bound_all(0.0)
        .upper_bound_all(10.0)
        .build()
        .fit(&x, &y)
        .expect("fit");

    for (j, expected) in truth.iter().enumerate() {
        assert!(
            (fit.coefficients()[j] - expected).abs() < 1e-6,
            "coefficient {j}: got {}, expected {expected}",
            fit.coefficients()[j]
        );
    }
}
