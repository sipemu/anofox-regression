#![allow(clippy::excessive_precision, clippy::large_const_arrays)]

//! #115: survival calibration at a horizon (`rms::calibrate.cph`,
//! `cmethod = "KM"`, RMS §20.10), replaying R's bootstrap resamples.
//! Reference values come from
//! `tests/r_scripts/generate_calibrate_cph_validation.R`.
//!
//! `rms::cph` (also inside `calibrate.cph`'s refits) stops its Newton
//! iterations at `eps = 1e-4` after at most 10 steps; with the same
//! `CoxOptions` the fits, and so the whole table, agree to 1e-9.

use anofox_regression::survival::cox::CoxOptions;
use anofox_regression::validation::{calibrate_cph, Resamples};
use faer::Mat;

mod fixture {
    include!("fixtures/calibrate_cph_validation.rs");
}
use fixture::*;

const N: usize = 300;

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        assert!(
            (u - v).abs() <= tol * (1.0 + v.abs()),
            "{name}[{k}]: {u} vs {v}"
        );
    }
}

fn data() -> (Mat<f64>, Vec<bool>) {
    (
        Mat::from_fn(N, 3, |i, j| X[i * 3 + j]),
        EVENT.iter().map(|&e| e == 1.0).collect(),
    )
}

#[test]
fn calibrate_cph_matches_rms() {
    let (x, event) = data();
    let b = BOOT_IDX.len() / N;
    let samples = (0..b)
        .map(|r| (0..N).map(|i| BOOT_IDX[r * N + i] as usize).collect())
        .collect();
    let res = Resamples::bootstrap(N, samples).unwrap();
    let cph = CoxOptions {
        eps: 1e-4,
        max_iter: 10,
        ..CoxOptions::default()
    };
    let cal = calibrate_cph(&x, &TIME, &event, U_M[0], &res, U_M[1] as usize, None, &cph).unwrap();
    all_close("predicted", &cal.predicted, &PREDICTED, 1e-9);
    let g = cal.km.len();
    let col = |k: usize| -> Vec<f64> { (0..g).map(|r| CAL[r * 10 + k]).collect() };
    let n: Vec<f64> = cal.n.iter().map(|&v| v as f64).collect();
    for (name, got, k) in [
        ("index.orig", &cal.index_orig, 0),
        ("training", &cal.training, 1),
        ("test", &cal.test, 2),
        ("optimism", &cal.optimism, 3),
        ("corrected", &cal.corrected, 4),
        ("n", &n, 5),
        ("mean.predicted", &cal.mean_predicted, 6),
        ("KM", &cal.km, 7),
        ("KM.corrected", &cal.km_corrected, 8),
        ("std.err", &cal.std_err, 9),
    ] {
        all_close(name, got, &col(k), 1e-9);
    }
}

#[test]
fn explicit_cuts_and_bad_input() {
    let (x, event) = data();
    let res = Resamples::bootstrap_seeded(N, 5, 3).unwrap();
    let o = CoxOptions::default();
    let cal = calibrate_cph(
        &x,
        &TIME,
        &event,
        5.0,
        &res,
        60,
        Some(vec![0.0, 0.5, 1.0]),
        &o,
    )
    .unwrap();
    assert_eq!(cal.km.len(), 2);
    assert!(cal.km.iter().all(|k| (0.0..=1.0).contains(k)));
    assert!(calibrate_cph(&x, &TIME[..10], &event, 5.0, &res, 60, None, &o).is_err());
    assert!(calibrate_cph(&x, &TIME, &event, 5.0, &res, 60, Some(vec![0.5]), &o).is_err());
    let none = vec![false; N];
    assert!(calibrate_cph(&x, &TIME, &none, 5.0, &res, 60, None, &o).is_err());
}
