#![allow(clippy::excessive_precision, clippy::too_many_arguments, dead_code)]

//! #116: datadist limits, summary.rms effects, contrast.rms and Predict.
//!
//! Reference values come from `tests/r_scripts/generate_effects_validation.R`.

use anofox_regression::effects::{Critical, ModelSpec};
use anofox_regression::splines::Rcs;
use faer::{Col, Mat};

include!("fixtures/effects_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    let ok = (a - b).abs() <= tol * (1.0 + b.abs());
    assert!(ok, "{name}: {a} vs {b}");
}

/// `y ~ rcs(x1, 4) * x2 + g`, rms column order.
fn spec(knots: &[f64]) -> ModelSpec {
    let mut s = ModelSpec::new();
    let x1 = s.rcs("x1", Rcs::new(knots).unwrap(), &X1);
    let x2 = s.linear("x2", &X2);
    s.categorical("g", &G);
    s.interaction(x1, x2);
    s
}

fn check(
    id: &str,
    coef: &[f64],
    vcov: &[f64],
    knots: &[f64],
    crit: Critical,
    summary: &[f64],
    contrast: &[f64],
    double: &[f64],
    cg: &[f64],
    px: &[f64],
    py: &[f64],
    pl: &[f64],
    pu: &[f64],
) {
    let s = spec(knots);
    let k = coef.len();
    let beta = Col::from_fn(k, |j| coef[j]);
    let v = Mat::from_fn(k, k, |a, b| vcov[a * k + b]);
    let rows = s.summary(&beta, &v, true, crit, 0.95);
    assert_eq!(rows.len() * 7, summary.len(), "{id} rows");
    for (r, row) in rows.iter().enumerate() {
        let w = &summary[r * 7..r * 7 + 7];
        close(&format!("{id} {} low", row.label), row.low, w[0], 1e-12);
        close(&format!("{id} {} high", row.label), row.high, w[1], 1e-12);
        close(
            &format!("{id} {} effect", row.label),
            row.effect.estimate,
            w[3],
            1e-9,
        );
        close(&format!("{id} {} se", row.label), row.effect.se, w[4], 1e-9);
        close(
            &format!("{id} {} lower", row.label),
            row.effect.lower,
            w[5],
            1e-9,
        );
        close(
            &format!("{id} {} upper", row.label),
            row.effect.upper,
            w[6],
            1e-9,
        );
    }
    let c = s.contrast(
        &beta,
        &v,
        true,
        &[(0, 60.0), (1, 2.0)],
        &[(0, 45.0), (1, 2.0)],
        crit,
        0.95,
    );
    for (name, a, b) in [
        ("est", c.estimate, contrast[0]),
        ("se", c.se, contrast[1]),
        ("lo", c.lower, contrast[2]),
        ("hi", c.upper, contrast[3]),
        ("stat", c.statistic, contrast[4]),
    ] {
        close(&format!("{id} contrast {name}"), a, b, 1e-6);
    }
    let d = s.double_contrast(
        &beta,
        &v,
        true,
        &[(0, 60.0), (1, 8.0)],
        &[(0, 45.0), (1, 8.0)],
        &[(0, 60.0), (1, 2.0)],
        &[(0, 45.0), (1, 2.0)],
        crit,
        0.95,
    );
    for (name, a, b) in [
        ("est", d.estimate, double[0]),
        ("se", d.se, double[1]),
        ("lo", d.lower, double[2]),
        ("hi", d.upper, double[3]),
    ] {
        close(&format!("{id} double {name}"), a, b, 1e-9);
    }
    close(&format!("{id} double p"), d.p_value, double[4], 1e-6);
    let g = s.contrast(&beta, &v, true, &[(2, 3.0)], &[(2, 2.0)], crit, 0.95);
    for (name, a, b) in [
        ("est", g.estimate, cg[0]),
        ("se", g.se, cg[1]),
        ("lo", g.lower, cg[2]),
        ("hi", g.upper, cg[3]),
    ] {
        close(&format!("{id} contrast g {name}"), a, b, 1e-9);
    }
    let (vals, est) = s.predict_grid(&beta, &v, true, 0, None, crit, 0.95);
    for i in 0..px.len() {
        close(&format!("{id} grid x[{i}]"), vals[i], px[i], 1e-12);
        close(&format!("{id} yhat[{i}]"), est[i].estimate, py[i], 1e-9);
        close(&format!("{id} lower[{i}]"), est[i].lower, pl[i], 1e-8);
        close(&format!("{id} upper[{i}]"), est[i].upper, pu[i], 1e-8);
    }
}

#[test]
fn datadist_limits() {
    let s = spec(&OLS_KNOTS);
    for (k, want) in [(0, &LIM_X1), (1, &LIM_X2)] {
        let l = s.predictors()[k].limits;
        let got = [
            l.low_effect,
            l.adjust_to,
            l.high_effect,
            l.low_prediction,
            l.high_prediction,
        ];
        for j in 0..5 {
            close(&format!("limits {k}[{j}]"), got[j], want[j], 1e-12);
        }
    }
}

#[test]
fn ols_effects_match_rms() {
    check(
        "ols",
        &OLS_COEF,
        &OLS_VCOV,
        &OLS_KNOTS,
        Critical::StudentT(X1.len() as f64 - OLS_COEF.len() as f64),
        &OLS_SUMMARY,
        &OLS_CONTRAST,
        &OLS_DOUBLE,
        &OLS_CONTRAST_G,
        &OLS_PRED_X1,
        &OLS_PRED_YHAT,
        &OLS_PRED_LOWER,
        &OLS_PRED_UPPER,
    );
}

#[test]
fn lrm_effects_and_odds_ratios_match_rms() {
    check(
        "lrm",
        &LRM_COEF,
        &LRM_VCOV,
        &LRM_KNOTS,
        Critical::Normal,
        &LRM_SUMMARY,
        &LRM_CONTRAST,
        &LRM_DOUBLE,
        &LRM_CONTRAST_G,
        &LRM_PRED_X1,
        &LRM_PRED_YHAT,
        &LRM_PRED_LOWER,
        &LRM_PRED_UPPER,
    );
    let s = spec(&LRM_KNOTS);
    let k = LRM_COEF.len();
    let beta = Col::from_fn(k, |j| LRM_COEF[j]);
    let v = Mat::from_fn(k, k, |a, b| LRM_VCOV[a * k + b]);
    for (r, row) in s
        .summary(&beta, &v, true, Critical::Normal, 0.95)
        .iter()
        .enumerate()
    {
        close("OR", row.effect.estimate.exp(), LRM_OR[3 * r], 1e-9);
        close("OR lower", row.effect.lower.exp(), LRM_OR[3 * r + 1], 1e-9);
        close("OR upper", row.effect.upper.exp(), LRM_OR[3 * r + 2], 1e-9);
    }
}
