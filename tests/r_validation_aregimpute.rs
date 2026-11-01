#![allow(clippy::excessive_precision)]

//! #106: multiple imputation by predictive mean matching
//! (`Hmisc::aregImpute`, RMS §3.7), replaying R's random numbers.
//! Reference values come from
//! `tests/r_scripts/generate_aregimpute_validation.R`: columns
//! `y, x1, x2, b, g` with missing values in `x1`, `x2` and the factor `g`.

use anofox_regression::imputation::{areg_impute, AregImputeOptions, DonorMatch, ImputeVariable};
use faer::Mat;

mod fixture {
    include!("fixtures/aregimpute_validation.rs");
}
use fixture::*;

const N: usize = 120;
const P: usize = 5;

fn data() -> Mat<f64> {
    Mat::from_fn(N, P, |i, j| X[i * P + j])
}

fn kinds() -> Vec<ImputeVariable> {
    let mut k = vec![ImputeVariable::Numeric; P];
    k[4] = ImputeVariable::Categorical;
    k
}

fn check(tag: &str, opts: AregImputeOptions, x1: &[f64], x2: &[f64], g: &[f64], rsq: &[f64]) {
    check_kinds(tag, &kinds(), opts, x1, x2, g, rsq);
}

fn check_kinds(
    tag: &str,
    kinds: &[ImputeVariable],
    opts: AregImputeOptions,
    x1: &[f64],
    x2: &[f64],
    g: &[f64],
    rsq: &[f64],
) {
    let a = areg_impute(&data(), kinds, &opts).unwrap();
    for (col, expected) in [(1, x1), (2, x2), (4, g)] {
        let got: Vec<f64> = a.imputed[col].iter().flatten().copied().collect();
        assert_eq!(got.len(), expected.len(), "{tag} column {col}");
        for (k, (u, v)) in got.iter().zip(expected).enumerate() {
            assert!(
                (u - v).abs() <= 1e-12 * (1.0 + v.abs()),
                "{tag} column {col}[{k}]: {u} vs {v}"
            );
        }
    }
    let r: Vec<f64> = [1, 2, 4].iter().map(|&c| a.r_squared[c]).collect();
    for (u, v) in r.iter().zip(rsq) {
        assert!((u - v).abs() < 1e-8, "{tag} R²: {u} vs {v}");
    }
    // a completed data set has no missing values and keeps observed ones
    let c = a.complete(&data(), 0);
    assert!((0..N).all(|i| (0..P).all(|j| !c[(i, j)].is_nan())));
    assert_eq!(c[(0, 0)], X[0]);
}

#[test]
fn weighted_matching_replays_aregimpute() {
    let opts = AregImputeOptions {
        seed: 7,
        ..Default::default()
    };
    check("weighted", opts, &W_X1, &W_X2, &W_G, &W_RSQ);
}

#[test]
fn closest_matching_replays_aregimpute() {
    let opts = AregImputeOptions {
        n_impute: 3,
        matching: DonorMatch::Closest,
        seed: 8,
        ..Default::default()
    };
    check("closest", opts, &C_X1, &C_X2, &C_G, &C_RSQ);
}

/// #210: `I(x1)` (always linear) and `burnin = 1`.
#[test]
fn linear_targets_and_burnin_replay_aregimpute() {
    let mut linear = kinds();
    linear[1] = ImputeVariable::Linear;
    let opts = AregImputeOptions {
        n_impute: 3,
        seed: 9,
        ..Default::default()
    };
    check_kinds("I(x1)", &linear, opts, &L_X1, &L_X2, &L_G, &L_RSQ);
    let opts = AregImputeOptions {
        n_impute: 3,
        tlinear: false,
        seed: 10,
        ..Default::default()
    };
    check("tlinear = FALSE", opts, &T_X1, &T_X2, &T_G, &T_RSQ);
    let opts = AregImputeOptions {
        n_impute: 2,
        burnin: Some(1),
        seed: 11,
        ..Default::default()
    };
    check("burnin = 1", opts, &B_X1, &B_X2, &B_G, &B_RSQ);
}

#[test]
fn rejects_bad_input() {
    let x = data();
    assert!(areg_impute(&x, &kinds()[..2], &AregImputeOptions::default()).is_err());
    let constant = Mat::from_fn(N, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
    let k = vec![ImputeVariable::Numeric; 2];
    assert!(areg_impute(&constant, &k, &AregImputeOptions::default()).is_err());
}
