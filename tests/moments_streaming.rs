//! Validates [`MomentAccumulator`] + `fit_from_moments` against the
//! whole-panel `fit()` path for both OLS and Ridge.
//!
//! Per #22 the streaming flow must be numerically equivalent to the
//! whole-panel solve to ~1e-10 on real data, since downstream callers will
//! rely on it for large-panel global regression fits where the explicit
//! design matrix is multi-GB.

use anofox_regression::core::{LambdaScaling, SolverType};
use anofox_regression::solvers::{
    FittedRegressor, MomentAccumulator, OlsRegressor, Regressor, RidgeRegressor,
};
use faer::{Col, Mat};

/// Synthesize a deterministic, well-conditioned panel.
fn build_panel(n: usize, p: usize, true_beta: &[f64], intercept: f64) -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(n, p, |i, j| {
        let t = (i as f64 + 1.0) / 7.0;
        let k = j as f64;
        match j % 4 {
            0 => (t + 0.2 * k).sin(),
            1 => (t * (0.3 + 0.1 * k)).cos(),
            2 => ((t + 0.5 * k).abs()).ln() - 0.5,
            _ => (t * 0.5).tanh() + 0.1 * k,
        }
    });
    let y = Col::from_fn(n, |i| {
        let mut v = intercept;
        for j in 0..p {
            v += x[(i, j)] * true_beta[j];
        }
        // Small deterministic perturbation so the panel isn't exactly
        // on the line — exercises the LASR-shaped centered Gram path.
        v + ((i as f64 * 1.7).sin()) * 0.05
    });
    (x, y)
}

fn accumulate(acc: &mut MomentAccumulator, x: &Mat<f64>, y: &Col<f64>) {
    let p = x.ncols();
    let n = x.nrows();
    let mut row = vec![0.0_f64; p];
    for i in 0..n {
        for j in 0..p {
            row[j] = x[(i, j)];
        }
        acc.push_row(&row, y[i]).expect("push_row");
    }
}

#[test]
fn ridge_with_intercept_streamed_matches_whole_panel() {
    let n = 800;
    let p = 5;
    let true_beta = [1.5_f64, -0.7, 0.3, 2.0, -0.4];
    let (x, y) = build_panel(n, p, &true_beta, 0.7);

    let model = RidgeRegressor::builder()
        .lambda(0.5)
        .lambda_scaling(LambdaScaling::Raw)
        .with_intercept(true)
        .solve_method(SolverType::Cholesky)
        .compute_inference(false)
        .build();

    let whole = model.fit(&x, &y).unwrap();

    let mut acc = MomentAccumulator::new(p);
    accumulate(&mut acc, &x, &y);
    let streamed = model.fit_from_accumulator(&acc).unwrap();

    let tol = 1e-10;
    assert!(
        (whole.result().intercept.unwrap() - streamed.result().intercept.unwrap()).abs() < tol,
        "intercept differs: {} vs {}",
        whole.result().intercept.unwrap(),
        streamed.result().intercept.unwrap()
    );
    for j in 0..p {
        let a = whole.result().coefficients[j];
        let b = streamed.result().coefficients[j];
        assert!(
            (a - b).abs() < tol,
            "coef[{}] differs: whole={} streamed={}",
            j,
            a,
            b
        );
    }
}

#[test]
fn ridge_no_intercept_streamed_matches_whole_panel() {
    let n = 600;
    let p = 4;
    let true_beta = [0.9_f64, -1.2, 0.0, 0.5];
    let (x, y) = build_panel(n, p, &true_beta, 0.0);

    let model = RidgeRegressor::builder()
        .lambda(0.2)
        .lambda_scaling(LambdaScaling::Raw)
        .with_intercept(false)
        .solve_method(SolverType::Cholesky)
        .compute_inference(false)
        .build();

    let whole = model.fit(&x, &y).unwrap();

    let mut acc = MomentAccumulator::new(p);
    accumulate(&mut acc, &x, &y);
    let streamed = model.fit_from_accumulator(&acc).unwrap();

    let tol = 1e-10;
    assert!(whole.result().intercept.is_none());
    assert!(streamed.result().intercept.is_none());
    for j in 0..p {
        let a = whole.result().coefficients[j];
        let b = streamed.result().coefficients[j];
        assert!((a - b).abs() < tol, "coef[{}]: {} vs {}", j, a, b);
    }
}

#[test]
fn ridge_glmnet_scaling_streamed_matches_whole_panel() {
    let n = 500;
    let p = 3;
    let true_beta = [1.0_f64, -0.5, 0.7];
    let (x, y) = build_panel(n, p, &true_beta, 0.2);

    let model = RidgeRegressor::builder()
        .lambda(0.05)
        .lambda_scaling(LambdaScaling::Glmnet) // λ_eff = λ * n
        .with_intercept(true)
        .solve_method(SolverType::Cholesky)
        .compute_inference(false)
        .build();

    let whole = model.fit(&x, &y).unwrap();

    let mut acc = MomentAccumulator::new(p);
    accumulate(&mut acc, &x, &y);
    let streamed = model.fit_from_accumulator(&acc).unwrap();

    let tol = 1e-10;
    assert!((whole.result().intercept.unwrap() - streamed.result().intercept.unwrap()).abs() < tol);
    for j in 0..p {
        let a = whole.result().coefficients[j];
        let b = streamed.result().coefficients[j];
        assert!((a - b).abs() < tol);
    }
}

#[test]
fn ols_streamed_matches_whole_panel() {
    let n = 1000;
    let p = 4;
    let true_beta = [0.8_f64, -1.1, 0.4, 1.6];
    let (x, y) = build_panel(n, p, &true_beta, -0.2);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(false)
        .build();

    let whole = model.fit(&x, &y).unwrap();

    let mut acc = MomentAccumulator::new(p);
    accumulate(&mut acc, &x, &y);
    let streamed = model.fit_from_accumulator(&acc).unwrap();

    // OLS via different decompositions (QR-with-pivot whole-panel vs
    // Cholesky on Gc streamed) — agreement is slightly looser than Ridge
    // because the two routes accumulate floating-point ops in different
    // order, but well below any signal threshold.
    let tol = 1e-8;
    assert!(
        (whole.result().intercept.unwrap() - streamed.result().intercept.unwrap()).abs() < tol,
        "intercept: {} vs {}",
        whole.result().intercept.unwrap(),
        streamed.result().intercept.unwrap()
    );
    for j in 0..p {
        let a = whole.result().coefficients[j];
        let b = streamed.result().coefficients[j];
        assert!(
            (a - b).abs() < tol,
            "coef[{}]: whole={} streamed={}",
            j,
            a,
            b
        );
    }
}

#[test]
fn merging_chunked_accumulators_matches_single() {
    // Split a 600-row panel into 4 chunks, fit each into its own
    // accumulator, merge, then solve — must match the single-accumulator
    // streamed solve to machine precision.
    let n = 600;
    let p = 3;
    let true_beta = [1.2_f64, -0.6, 0.4];
    let (x, y) = build_panel(n, p, &true_beta, 0.1);

    let model = RidgeRegressor::builder()
        .lambda(0.1)
        .with_intercept(true)
        .compute_inference(false)
        .build();

    // Single accumulator over the whole panel.
    let mut single = MomentAccumulator::new(p);
    accumulate(&mut single, &x, &y);
    let single_fit = model.fit_from_accumulator(&single).unwrap();

    // Four chunks, merged.
    let chunks = [(0_usize, 200), (200, 400), (400, 500), (500, 600)];
    let mut merged = MomentAccumulator::new(p);
    for &(lo, hi) in &chunks {
        let mut part = MomentAccumulator::new(p);
        let mut row = vec![0.0_f64; p];
        for i in lo..hi {
            for j in 0..p {
                row[j] = x[(i, j)];
            }
            part.push_row(&row, y[i]).unwrap();
        }
        merged.merge(&part).unwrap();
    }
    let merged_fit = model.fit_from_accumulator(&merged).unwrap();

    let tol = 1e-12;
    assert_eq!(merged.n(), single.n());
    assert!(
        (single_fit.result().intercept.unwrap() - merged_fit.result().intercept.unwrap()).abs()
            < tol
    );
    for j in 0..p {
        let a = single_fit.result().coefficients[j];
        let b = merged_fit.result().coefficients[j];
        assert!((a - b).abs() < tol, "coef[{}]: {} vs {}", j, a, b);
    }
}

#[test]
fn predict_works_after_fit_from_moments() {
    // Round-trip: streamed fit → predict on new data → must equal predict
    // from whole-panel fit at the same coefficients.
    let n = 400;
    let p = 3;
    let true_beta = [1.0_f64, -0.5, 0.7];
    let (x, y) = build_panel(n, p, &true_beta, 0.5);

    let model = RidgeRegressor::builder()
        .lambda(0.1)
        .with_intercept(true)
        .compute_inference(false)
        .build();
    let whole = model.fit(&x, &y).unwrap();

    let mut acc = MomentAccumulator::new(p);
    accumulate(&mut acc, &x, &y);
    let streamed = model.fit_from_accumulator(&acc).unwrap();

    let x_new = Mat::from_fn(5, p, |i, j| (i as f64 - 2.0) * 0.3 + j as f64 * 0.1);
    let p_whole = whole.predict(&x_new);
    let p_streamed = streamed.predict(&x_new);
    for i in 0..p_whole.nrows() {
        assert!(
            (p_whole[i] - p_streamed[i]).abs() < 1e-10,
            "pred[{}]: {} vs {}",
            i,
            p_whole[i],
            p_streamed[i]
        );
    }
}

#[test]
fn ols_rejects_zero_diagonal_moments() {
    // A column that is identically zero produces a zero diagonal in the
    // centered Gram. This is the only kind of rank deficiency the
    // moments-only path can reliably detect (full pivoting information
    // is lost going from per-row data to (X'X, X'y) — see the doc
    // comment on `fit_from_moments`). Cases like duplicate columns
    // still succeed with a particular least-squares solution.
    let n = 50;
    let mut acc = MomentAccumulator::new(2);
    for i in 0..n {
        let t = i as f64 / 10.0;
        acc.push_row(&[t, 0.0], 2.0 * t).unwrap(); // column 1 ≡ 0
    }
    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(false)
        .build();
    let err = model.fit_from_accumulator(&acc).unwrap_err();
    use anofox_regression::solvers::RegressionError;
    assert!(matches!(err, RegressionError::SingularMatrix));
}
