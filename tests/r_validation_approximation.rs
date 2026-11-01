#![allow(clippy::large_const_arrays, clippy::excessive_precision)]

//! #108: heuristic shrinkage and model approximation (RMS §4.5, §5.5.2,
//! Ch 19) against rms. Reference values come from
//! `tests/r_scripts/generate_approximation_validation.R`.

use anofox_regression::design::TermMap;
use anofox_regression::inference::{
    approximate_model, approximation_path, heuristic_shrinkage, shrink_slopes, ApproximationError,
};
use faer::{Col, Mat};

include!("fixtures/approximation_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

/// lrm(y ~ rcs(x1, 4) + x2 + x3 + x4 + x5 + g)
fn model() -> (TermMap, Mat<f64>, Col<f64>, Mat<f64>) {
    let mut t = TermMap::new();
    t.add_main("x1", &[false, true, true]);
    for name in ["x2", "x3", "x4", "x5"] {
        t.add_main(name, &[false]);
    }
    t.add_main("g", &[false, false]);
    let k = COEF.len();
    let p = k - 1;
    let n = LP.len();
    let x = Mat::from_fn(n, p, |i, j| X[i * p + j]);
    let lp = Col::from_fn(n, |i| LP[i]);
    let v = Mat::from_fn(k, k, |a, b| VCOV[a * k + b]);
    (t, x, lp, v)
}

#[test]
fn heuristic_shrinkage_matches_rms_formula() {
    let [lr, df, gamma] = LR_DF_GAMMA;
    close("gamma", heuristic_shrinkage(lr, df), gamma, 1e-14);
    let beta = Col::from_fn(COEF.len(), |j| COEF[j]);
    let shrunk = shrink_slopes(&beta, 1, gamma);
    assert_eq!(shrunk[0], beta[0]);
    for j in 1..COEF.len() {
        close(&format!("slope {j}"), shrunk[j], gamma * beta[j], 1e-15);
    }
}

#[test]
fn approximation_path_matches_fastbw_on_ols_sigma_1() {
    let (t, x, lp, _) = model();
    let path = approximation_path(&x, &lp, &t).unwrap();
    let order: Vec<usize> = path.iter().map(|s| s.term).collect();
    assert_eq!(order, PATH_DELETED);
    for (k, step) in path.iter().enumerate() {
        close(
            &format!("R² after {} deletions", k + 1),
            step.r_squared,
            PATH_R2[k],
            1e-10,
        );
    }
}

#[test]
fn approximate_model_matches_rms_5_5_2() {
    let (t, x, lp, v) = model();
    // x1 (rcs, 3 columns) and x2
    let a = approximate_model(&x, &lp, &v, &t, &[1, 0]).unwrap();
    assert_eq!(a.kept_terms, vec![0, 1]);
    assert_eq!(a.columns, vec![0, 1, 2, 3]);
    close("R²", a.r_squared, APPROX_R2[0], 1e-10);
    let m = APPROX_COEF.len();
    for j in 0..m {
        close(
            &format!("coef {j}"),
            a.coefficients[j],
            APPROX_COEF[j],
            1e-9,
        );
        for l in 0..m {
            close(
                &format!("vcov {j},{l}"),
                a.vcov[(j, l)],
                APPROX_VCOV[j * m + l],
                1e-9,
            );
        }
    }
    // Approximating by every term reproduces the full model exactly
    let full = approximate_model(&x, &lp, &v, &t, &[0, 1, 2, 3, 4, 5]).unwrap();
    close("full R²", full.r_squared, 1.0, 1e-12);
    for j in 0..COEF.len() {
        close(
            &format!("full coef {j}"),
            full.coefficients[j],
            COEF[j],
            1e-9,
        );
        for l in 0..COEF.len() {
            close(
                &format!("full vcov {j},{l}"),
                full.vcov[(j, l)],
                v[(j, l)],
                1e-9,
            );
        }
    }
}

#[test]
fn approximation_rejects_bad_input() {
    let (t, x, lp, v) = model();
    let short = Col::from_fn(10, |i| lp[i]);
    assert!(matches!(
        approximate_model(&x, &short, &v, &t, &[0]),
        Err(ApproximationError::Dimension(_))
    ));
    assert!(matches!(
        approximate_model(&x, &lp, &v, &t, &[9]),
        Err(ApproximationError::Term(9))
    ));
    let small_v = Mat::<f64>::identity(3, 3);
    assert!(matches!(
        approximate_model(&x, &lp, &small_v, &t, &[0]),
        Err(ApproximationError::Dimension(_))
    ));
    let zero = Mat::<f64>::zeros(x.nrows(), x.ncols());
    assert!(matches!(
        approximation_path(&zero, &lp, &t),
        Err(ApproximationError::Singular)
    ));
}
