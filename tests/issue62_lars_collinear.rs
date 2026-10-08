#![allow(clippy::needless_range_loop)]

//! Issue #62: LARS on exactly collinear columns must give the same,
//! platform-independent answer: the later dependent column is aliased (NaN
//! coefficient, `aliased[j] = true`) and the fit equals the fit without it.
//! Previously the duplicate column entered the active set or not depending
//! on last-bit rounding of its residual correlation (absolute `eps`), which
//! produced `SingularMatrix` on aarch64 but a fit on x86_64.

use anofox_regression::solvers::{FittedRegressor, LarsMethod, LarsRegressor, Regressor};
use faer::{Col, Mat};

/// Data of the DuckDB extension's `regression_degenerate.test` (`collin`).
fn base(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 1..=n as i64 {
        let a = (i % 7) as f64 + 0.5;
        let b = ((i * 3) % 11) as f64 / 2.0;
        x1.push(a);
        y.push(3.0 + 0.8 * a - 0.3 * b + ((i * 37) % 13) as f64 / 13.0 - 0.5);
    }
    (x1, y)
}

fn nudge(v: f64, ulps: i64) -> f64 {
    f64::from_bits((v.to_bits() as i64 + ulps) as u64)
}

fn fit(
    cols: &[Vec<f64>],
    y: &[f64],
    method: LarsMethod,
    standardize: bool,
) -> (Vec<f64>, Vec<bool>, f64) {
    let n = y.len();
    let x = Mat::from_fn(n, cols.len(), |i, j| cols[j][i]);
    let yc = Col::from_fn(n, |i| y[i]);
    let f = LarsRegressor::builder()
        .method(method)
        .standardize(standardize)
        .build()
        .fit(&x, &yc)
        .expect("collinear LARS fit must succeed");
    let r = f.result();
    (
        r.coefficients.iter().copied().collect(),
        r.aliased.clone(),
        r.intercept.unwrap(),
    )
}

#[test]
fn duplicate_column_is_aliased_for_all_scalings_and_ulp_perturbations() {
    let (x1, y) = base(40);
    for &method in &[LarsMethod::Lar, LarsMethod::Lasso] {
        for &standardize in &[true, false] {
            let (b1, _, i1) = fit(std::slice::from_ref(&x1), &y, method, standardize);
            for &k in &[2.0, -2.0, 3.0, 0.1, 1e-3, 1e6, std::f64::consts::PI] {
                for &ulps in &[0_i64, 1, -1] {
                    // Perturb alternating entries by ±1 ulp so the rounding
                    // of the residual correlation differs case to case.
                    let x2: Vec<f64> = x1
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| nudge(k * v, if i % 2 == 0 { ulps } else { -ulps }))
                        .collect();
                    let (b, aliased, ic) = fit(&[x1.clone(), x2], &y, method, standardize);
                    let ctx = format!("method={method:?} std={standardize} k={k} ulps={ulps}");
                    assert_eq!(aliased, vec![false, true], "{ctx}");
                    assert!(b[1].is_nan(), "{ctx}");
                    assert!(
                        (b[0] - b1[0]).abs() < 1e-10 * b1[0].abs().max(1.0),
                        "{ctx}: {} vs {}",
                        b[0],
                        b1[0]
                    );
                    assert!((ic - i1).abs() < 1e-10 * i1.abs().max(1.0), "{ctx}");
                }
            }
        }
    }
}

#[test]
fn later_column_of_a_linear_combination_is_aliased_and_predict_ignores_it() {
    let (x1, y) = base(40);
    let x2: Vec<f64> = (1..=40).map(|i| ((i * 3) % 11) as f64 / 2.0).collect();
    let x3: Vec<f64> = x1.iter().zip(&x2).map(|(a, b)| 0.5 * a - 4.0 * b).collect();
    let (b_ref, _, i_ref) = fit(&[x1.clone(), x2.clone()], &y, LarsMethod::Lar, true);
    let n = y.len();
    let x = Mat::from_fn(n, 3, |i, j| [&x1, &x2, &x3][j][i]);
    let f = LarsRegressor::builder()
        .build()
        .fit(&x, &Col::from_fn(n, |i| y[i]))
        .unwrap();
    let r = f.result();
    assert_eq!(r.aliased, vec![false, false, true]);
    assert!(r.coefficients[2].is_nan());
    for j in 0..2 {
        assert!((r.coefficients[j] - b_ref[j]).abs() < 1e-9);
    }
    assert!((r.intercept.unwrap() - i_ref).abs() < 1e-9);
    let pred = f.predict(&x);
    assert!(pred.iter().all(|v| v.is_finite()));
    for i in 0..n {
        assert!((pred[i] - r.fitted_values[i]).abs() < 1e-9);
    }
}
