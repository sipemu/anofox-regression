#![allow(clippy::excessive_precision)]

//! #108: fast backward step-down against `rms::fastbw`.
//!
//! Reference values come from `tests/r_scripts/generate_fastbw_validation.R`.

use anofox_regression::design::TermMap;
use anofox_regression::inference::{fastbw, StepRule, StepType};
use faer::{Col, Mat};

include!("fixtures/fastbw_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn check(rule: StepRule, kind: StepType, deleted: &[usize], result: &[f64], coef: &[f64]) {
    // lrm(y ~ rcs(x1, 4) + x2 + x3 + x4 + x5 + g)
    let mut t = TermMap::new();
    t.add_main("x1", &[false, true, true]);
    for name in ["x2", "x3", "x4", "x5"] {
        t.add_main(name, &[false]);
    }
    t.add_main("g", &[false, false]);
    let k = COEF.len();
    let beta = Col::from_fn(k, |j| COEF[j]);
    let v = Mat::from_fn(k, k, |a, b| VCOV[a * k + b]);
    let s = fastbw(&beta, &v, 1, &t, rule, kind);
    let got: Vec<usize> = s.deleted.iter().map(|d| d.term).collect();
    assert_eq!(got, deleted, "{rule:?} {kind:?}");
    for (r, d) in s.deleted.iter().enumerate() {
        let w = &result[r * 7..r * 7 + 7];
        let row = [
            d.chisq,
            d.df as f64,
            d.p_value,
            d.residual_chisq,
            d.residual_df as f64,
            d.residual_p,
            d.aic,
        ];
        for (c, (a, b)) in row.iter().zip(w).enumerate() {
            close(&format!("{rule:?} {kind:?} step {r} col {c}"), *a, *b, 1e-9);
        }
    }
    for (j, (a, b)) in s.coefficients.iter().zip(coef).enumerate() {
        close(&format!("{rule:?} {kind:?} coef {j}"), *a, *b, 1e-9);
    }
}

#[test]
fn fastbw_matches_rms() {
    let aic = StepRule::Aic {
        k: 2.0,
        threshold: 0.0,
    };
    let p = StepRule::P { sls: 0.05 };
    check(
        aic,
        StepType::Residual,
        &AIC_RESIDUAL_DELETED,
        &AIC_RESIDUAL_RESULT,
        &AIC_RESIDUAL_COEF,
    );
    check(
        aic,
        StepType::Individual,
        &AIC_INDIVIDUAL_DELETED,
        &AIC_INDIVIDUAL_RESULT,
        &AIC_INDIVIDUAL_COEF,
    );
    check(
        p,
        StepType::Residual,
        &P_RESIDUAL_DELETED,
        &P_RESIDUAL_RESULT,
        &P_RESIDUAL_COEF,
    );
    check(
        p,
        StepType::Individual,
        &P_INDIVIDUAL_DELETED,
        &P_INDIVIDUAL_RESULT,
        &P_INDIVIDUAL_COEF,
    );
}
