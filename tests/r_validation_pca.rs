#![allow(clippy::excessive_precision)]

//! #107: principal components (`stats::prcomp`) and summary scores of
//! variable clusters (RMS §4.7). Reference values come from
//! `tests/r_scripts/generate_pca_validation.R`. Component signs are
//! arbitrary, so each R component is aligned to ours before comparing.

use anofox_regression::reduction::{cluster_scores, principal_components};
use faer::Mat;

mod fixture {
    include!("fixtures/pca_validation.rs");
}
use fixture::*;

const N: usize = 80;
const P: usize = 5;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn x() -> Mat<f64> {
    Mat::from_fn(N, P, |i, j| X[i * P + j])
}

/// Sign that turns R's column `c` (column-major `r`, `rows` rows) into ours.
fn sign(ours: &Mat<f64>, r: &[f64], rows: usize, c: usize) -> f64 {
    let dot: f64 = (0..rows).map(|i| ours[(i, c)] * r[c * rows + i]).sum();
    dot.signum()
}

fn check(tag: &str, scale: bool, sdev: &[f64], rot: &[f64], scores: &[f64]) {
    let pc = principal_components(&x(), scale).unwrap();
    for c in 0..P {
        close(&format!("{tag} sdev {c}"), pc.sdev[c], sdev[c], 1e-12);
        let s = sign(&pc.rotation, rot, P, c);
        for j in 0..P {
            close(
                &format!("{tag} rot {j},{c}"),
                pc.rotation[(j, c)],
                s * rot[c * P + j],
                1e-10,
            );
        }
        for i in 0..N {
            close(
                &format!("{tag} score {i},{c}"),
                pc.scores[(i, c)],
                s * scores[c * N + i],
                1e-10,
            );
        }
    }
    let pv = pc.proportion_of_variance();
    let total: f64 = sdev.iter().map(|s| s * s).sum();
    close(
        &format!("{tag} pve"),
        pv[0],
        sdev[0] * sdev[0] / total,
        1e-12,
    );
}

#[test]
fn principal_components_match_prcomp() {
    check("scaled", true, &SC_SDEV, &SC_ROT, &SC_SCORES);
    check("unscaled", false, &UN_SDEV, &UN_ROT, &UN_SCORES);

    let pc = principal_components(&x(), true).unwrap();
    let nd = Mat::from_fn(3, P, |i, j| NEW[i * P + j]);
    let pred = pc.transform(&nd, 2);
    for c in 0..2 {
        let s = sign(&pc.rotation, &SC_ROT, P, c);
        for i in 0..3 {
            close(
                &format!("predict {i},{c}"),
                pred[(i, c)],
                s * SC_PRED[c * 3 + i],
                1e-10,
            );
        }
    }
}

#[test]
fn cluster_scores_are_first_components() {
    let s = cluster_scores(&x(), &[vec![0, 1], vec![2, 3], vec![4]], 1).unwrap();
    assert_eq!(s.ncols(), 3);
    for c in 0..3 {
        let sg = sign(&s, &CL_SCORES, N, c);
        for i in 0..N {
            close(
                &format!("cluster {i},{c}"),
                s[(i, c)],
                sg * CL_SCORES[c * N + i],
                1e-10,
            );
        }
    }
    assert!(cluster_scores(&x(), &[vec![]], 1).is_err());
    assert!(cluster_scores(&x(), &[vec![P]], 1).is_err());
    let constant = Mat::from_fn(N, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
    assert!(principal_components(&constant, true).is_err());
    assert!(principal_components(&constant, false).is_ok());
}
