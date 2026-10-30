#![allow(clippy::excessive_precision)]

//! Simultaneous confidence bands (`rms::Predict(conf.type =
//! "simultaneous")`, multcomp's single-step max-|t|), follow-up of #116 in
//! #135. Reference values come from
//! `tests/r_scripts/generate_simultaneous_validation.R`.
//!
//! R's critical values come from mvtnorm's randomised algorithm (about 1e-3
//! accurate by default); ours are deterministic. Each `*_CRIT` holds
//! `Predict`'s value and a tight four-seed mvtnorm average; ours must be
//! within 2e-4 of the latter and 2e-3 of the former, and the bands within
//! 2e-3.

use anofox_regression::effects::{simultaneous_critical, Critical, ModelSpec};
use anofox_regression::splines::Rcs;
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/simultaneous_validation.rs");
}
use fixture::*;

const N: usize = 220;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn mats(beta: &[f64], vcov: &[f64]) -> (Col<f64>, Mat<f64>) {
    let p = beta.len();
    (
        Col::from_fn(p, |j| beta[j]),
        Mat::from_fn(p, p, |a, b| vcov[b * p + a]),
    )
}

#[allow(clippy::too_many_arguments)]
fn check(
    tag: &str,
    spec: &ModelSpec,
    k: usize,
    grid: &[f64],
    beta: &[f64],
    vcov: &[f64],
    crit: Critical,
    pred: &[f64],
    r_crit: &[f64],
) {
    let (b, v) = mats(beta, vcov);
    let (_, est, c) =
        spec.predict_grid_simultaneous(&b, &v, true, k, Some(grid.to_vec()), crit, 0.95);
    close(&format!("{tag} crit vs tight mvtnorm"), c, r_crit[1], 2e-4);
    close(&format!("{tag} crit vs Predict"), c, r_crit[0], 2e-3);
    let g = grid.len();
    for (i, e) in est.iter().enumerate() {
        close(&format!("{tag} yhat {i}"), e.estimate, pred[i], 1e-10);
        close(&format!("{tag} lower {i}"), e.lower, pred[g + i], 2e-3);
        close(&format!("{tag} upper {i}"), e.upper, pred[2 * g + i], 2e-3);
    }
}

fn spec(age_rcs: bool, with_w: bool) -> ModelSpec {
    let mut s = ModelSpec::new();
    if age_rcs {
        s.rcs("age", Rcs::from_data(&AGE, 4).unwrap(), &AGE);
    } else {
        s.linear("age", &AGE);
    }
    s.linear("sex", &SEX);
    if with_w {
        s.linear("w", &W);
    }
    s
}

#[test]
fn simultaneous_bands_match_predict() {
    // rank 4 (intercept + rcs(age, 4)): Halton rule
    check(
        "LRM",
        &spec(true, false),
        0,
        &LRM_GRID,
        &LRM_BETA,
        &LRM_VCOV,
        Critical::Normal,
        &LRM_PRED,
        &LRM_CRIT,
    );
    // rank 2 with a t reference: angular rule
    check(
        "OLS",
        &spec(false, true),
        2,
        &OLS_GRID,
        &OLS_BETA,
        &OLS_VCOV,
        Critical::StudentT((N - 4) as f64),
        &OLS_PRED,
        &OLS_CRIT,
    );
    // two levels of a factor: rank 2
    check(
        "CAT",
        &spec(true, false),
        1,
        &CAT_GRID,
        &CAT_BETA,
        &CAT_VCOV,
        Critical::Normal,
        &CAT_PRED,
        &CAT_CRIT,
    );
}

#[test]
fn rank_one_is_pointwise_and_bad_input_is_nan() {
    let v = Mat::from_fn(2, 2, |a, b| if a == b { 1.0 } else { 0.3 });
    let rows = vec![vec![1.0, 0.0], vec![2.0, 0.0]];
    let c = simultaneous_critical(&rows, &v, Critical::Normal, 0.95);
    assert!((c - 1.959963984540054).abs() < 1e-12);
    assert!(simultaneous_critical(&[], &v, Critical::Normal, 0.95).is_nan());
    assert!(simultaneous_critical(&rows, &v, Critical::Normal, 1.5).is_nan());
}
