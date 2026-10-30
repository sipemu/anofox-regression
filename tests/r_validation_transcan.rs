#![allow(clippy::excessive_precision)]

//! #107: `Hmisc::transcan(method = "canonical")` (RMS §4.7.3, §8.4):
//! optimal transformations by alternating canonical correlations, with and
//! without missing values. Reference values come from
//! `tests/r_scripts/generate_transcan_validation.R`. Signs of the
//! transformations are arbitrary, so each R column is aligned to ours.

use anofox_regression::reduction::{transcan, TranscanOptions};
use faer::Mat;

mod fixture {
    include!("fixtures/transcan_validation.rs");
}
use fixture::*;

const N: usize = 150;
const P: usize = 5;

fn check(tag: &str, x: &[f64], trans: &[f64], rsq: &[f64], rsq_adj: &[f64]) {
    let xm = Mat::from_fn(N, P, |i, j| x[i * P + j]);
    let cat = [false, false, false, true, false];
    let t = transcan(&xm, &cat, &TranscanOptions::default()).unwrap();
    for j in 0..P {
        let dot: f64 = (0..N)
            .map(|i| t.transformed[(i, j)] * trans[j * N + i])
            .sum();
        let s = dot.signum();
        for i in 0..N {
            let (u, v) = (t.transformed[(i, j)], s * trans[j * N + i]);
            assert!(
                (u - v).abs() < 1e-6 * (1.0 + v.abs()),
                "{tag} trans {i},{j}: {u} vs {v}"
            );
        }
        assert!(
            (t.r_squared[j] - rsq[j]).abs() < 1e-8,
            "{tag} rsq {j}: {} vs {}",
            t.r_squared[j],
            rsq[j]
        );
        assert!(
            (t.r_squared_adj[j] - rsq_adj[j]).abs() < 1e-8,
            "{tag} rsq.adj {j}"
        );
    }
}

#[test]
fn transcan_matches_hmisc() {
    check("full", &X, &FULL_TRANS, &FULL_RSQ, &FULL_RSQADJ);
}

#[test]
fn transcan_with_missing_values_matches_hmisc() {
    check("missing", &XM, &MISS_TRANS, &MISS_RSQ, &MISS_RSQADJ);
}

#[test]
fn rejects_bad_input() {
    let x = Mat::from_fn(N, P, |i, j| X[i * P + j]);
    assert!(transcan(&x, &[false; 2], &TranscanOptions::default()).is_err());
    let one = Mat::from_fn(N, 1, |i, _| X[i * P]);
    assert!(transcan(&one, &[false], &TranscanOptions::default()).is_err());
}
