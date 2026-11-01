#![allow(clippy::large_const_arrays, clippy::excessive_precision)]

//! #104: bootstrap covariance against `rms::bootcov`, replaying R's
//! ordinary and cluster resamples.
//!
//! Reference values come from `tests/r_scripts/generate_bootcov_validation.R`.

use anofox_regression::solvers::LogisticRegression;
use anofox_regression::validation::{bootcov, bootstrap_rows, cluster_bootstrap_rows};
use faer::{Col, Mat};

include!("fixtures/bootcov_validation.rs");

fn fit(rows: &[usize]) -> Option<Vec<f64>> {
    let x = Mat::from_fn(rows.len(), 2, |i, j| [X1[rows[i]], X2[rows[i]]][j]);
    let y = Col::from_fn(rows.len(), |i| Y[rows[i]]);
    let f = LogisticRegression::builder()
        .tolerance(1e-12)
        .build()
        .fit(&x, &y)
        .ok()?;
    Some(vec![
        f.intercept()?,
        f.coefficients()[0],
        f.coefficients()[1],
    ])
}

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

#[test]
fn ordinary_bootstrap_matches_bootcov() {
    let n = Y.len();
    let samples: Vec<Vec<usize>> = BOOT_IDX.chunks(n).map(<[usize]>::to_vec).collect();
    let b = bootcov(&samples, fit).unwrap();
    assert_eq!(b.failed, 0);
    for i in 0..b.replicates.nrows() {
        for j in 0..3 {
            close(
                &format!("coef[{i},{j}]"),
                b.replicates[(i, j)],
                BOOT_COEF[i * 3 + j],
                1e-5,
            );
        }
    }
    for a in 0..3 {
        for c in 0..3 {
            close(
                &format!("var[{a},{c}]"),
                b.vcov[(a, c)],
                BOOT_VAR[a * 3 + c],
                1e-4,
            );
        }
    }
    for (j, (lo, hi)) in b.percentile_ci(0.95).iter().enumerate() {
        close(&format!("q lo {j}"), *lo, BOOT_Q[2 * j], 1e-5);
        close(&format!("q hi {j}"), *hi, BOOT_Q[2 * j + 1], 1e-5);
    }
    let (lo, hi) = b.linear_combination_ci(&[0.0, 1.0, 0.0], 0.95);
    close("lincomb", lo, BOOT_Q[2], 1e-5);
    close("lincomb", hi, BOOT_Q[3], 1e-5);
}

#[test]
fn cluster_bootstrap_matches_bootcov() {
    let mut at = 0;
    let samples: Vec<Vec<usize>> = CL_LENS
        .iter()
        .map(|&l| {
            let s = CL_ROWS[at..at + l].to_vec();
            at += l;
            s
        })
        .collect();
    let b = bootcov(&samples, fit).unwrap();
    for a in 0..3 {
        for c in 0..3 {
            close(
                &format!("cluster var[{a},{c}]"),
                b.vcov[(a, c)],
                CL_VAR[a * 3 + c],
                1e-4,
            );
        }
    }
}

#[test]
fn seeded_samples_run() {
    assert_eq!(bootstrap_rows(10, 3, 1).len(), 3);
    let rows = cluster_bootstrap_rows(&ID, 20, 9);
    assert!(rows.iter().all(|r| r.len() % 4 == 0));
    assert!(bootcov(&rows, fit).unwrap().vcov[(1, 1)] > 0.0);
}
