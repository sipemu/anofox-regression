#![allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]

//! #114: Cox proportional hazards against `survival::coxph`,
//! `survfit.coxph` and `rms::cph`.
//!
//! Reference values come from `tests/r_scripts/generate_cox_validation.R`.

use anofox_regression::survival::cox::{coxph, CoxFit, CoxInput, CoxOptions, Ties};
use faer::Mat;

include!("fixtures/cox_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    let ok = (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs());
    assert!(ok, "{name}: {a} vs {b}");
}

fn vclose(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name} length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *x, *y, tol);
    }
}

fn mclose(name: &str, m: &Mat<f64>, want: &[f64], tol: f64) {
    let k = m.ncols();
    assert_eq!(m.nrows() * k, want.len(), "{name} size");
    for i in 0..m.nrows() {
        for j in 0..k {
            close(&format!("{name}[{i},{j}]"), m[(i, j)], want[i * k + j], tol);
        }
    }
}

fn lung_x(cols: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(TIME.len(), cols.len(), |i, j| cols[j][i])
}

fn events(v: &[f64]) -> Vec<bool> {
    v.iter().map(|&s| s == 1.0).collect()
}

fn opts(ties: Ties) -> CoxOptions {
    CoxOptions {
        ties,
        eps: 1e-12,
        max_iter: 50,
        ..Default::default()
    }
}

fn check_fit(
    id: &str,
    f: &CoxFit,
    coef: &[f64],
    var: &[f64],
    loglik: &[f64],
    score: f64,
    wald: f64,
) {
    assert!(f.converged, "{id} converged");
    vclose(&format!("{id} coef"), &f.coefficients, coef, 1e-8);
    mclose(&format!("{id} var"), &f.var, var, 1e-7);
    close(&format!("{id} loglik0"), f.loglik.0, loglik[0], 1e-12);
    close(&format!("{id} loglik"), f.loglik.1, loglik[1], 1e-11);
    close(&format!("{id} score"), f.score_test, score, 1e-9);
    close(&format!("{id} wald"), f.wald_test, wald, 1e-7);
}

#[test]
fn efron_matches_coxph_and_cph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &SEX, &ECOG, &WTLOSS]);
    let f = coxph(&x, &CoxInput::new(&TIME, &ev), &opts(Ties::Efron)).unwrap();
    check_fit(
        "efron",
        &f,
        &EFRON_COEF,
        &EFRON_VAR,
        &EFRON_LOGLIK,
        EFRON_SCORE,
        EFRON_WALD,
    );
    close("concordance", f.concordance(), EFRON_CONCORDANCE, 1e-12);
    close("LR", f.lr_test, CPH_STATS[0], 1e-9);
    close("R2", f.r2(), CPH_STATS[1], 1e-9);
    close("Dxy", f.dxy(), CPH_STATS[2], 1e-12);
}

#[test]
fn breslow_matches_coxph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &SEX, &ECOG, &WTLOSS]);
    let f = coxph(&x, &CoxInput::new(&TIME, &ev), &opts(Ties::Breslow)).unwrap();
    check_fit(
        "breslow",
        &f,
        &BRESLOW_COEF,
        &BRESLOW_VAR,
        &BRESLOW_LOGLIK,
        BRESLOW_SCORE,
        BRESLOW_WALD,
    );
}

#[test]
fn weights_and_offset_match_coxph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &SEX, &ECOG, &WTLOSS]);
    let input = CoxInput::new(&TIME, &ev).weights(&W).offset(&OFF);
    let f = coxph(&x, &input, &opts(Ties::Efron)).unwrap();
    check_fit(
        "weighted",
        &f,
        &WEIGHTED_COEF,
        &WEIGHTED_VAR,
        &WEIGHTED_LOGLIK,
        WEIGHTED_SCORE,
        WEIGHTED_WALD,
    );
}

#[test]
fn strata_match_coxph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &ECOG, &WTLOSS]);
    let sex: Vec<usize> = SEX.iter().map(|&s| s as usize).collect();
    let f = coxph(
        &x,
        &CoxInput::new(&TIME, &ev).strata(&sex),
        &opts(Ties::Efron),
    )
    .unwrap();
    check_fit(
        "strata",
        &f,
        &STRATA_COEF,
        &STRATA_VAR,
        &STRATA_LOGLIK,
        STRATA_SCORE,
        STRATA_WALD,
    );
}

#[test]
fn score_residuals_and_robust_variance_match_coxph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &SEX, &ECOG, &WTLOSS]);
    let f = coxph(&x, &CoxInput::new(&TIME, &ev), &opts(Ties::Efron)).unwrap();
    mclose("score residuals", &f.score_residuals(), &SCORE_RESID, 1e-7);
    mclose("robust", &f.robust_vcov(&ID).unwrap(), &ROBUST_VAR, 1e-7);
    let fb = coxph(&x, &CoxInput::new(&TIME, &ev), &opts(Ties::Breslow)).unwrap();
    mclose(
        "robust breslow",
        &fb.robust_vcov(&ID).unwrap(),
        &ROBUST_BRESLOW_VAR,
        1e-7,
    );
    let fw = coxph(
        &x,
        &CoxInput::new(&TIME, &ev).weights(&W),
        &opts(Ties::Efron),
    )
    .unwrap();
    mclose(
        "robust weighted",
        &fw.robust_vcov(&ID).unwrap(),
        &ROBUST_WEIGHTED_VAR,
        1e-7,
    );
}

#[test]
fn ridge_matches_coxph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &SEX, &ECOG, &WTLOSS]);
    let o = CoxOptions {
        ridge: 2.0,
        ..opts(Ties::Efron)
    };
    let f = coxph(&x, &CoxInput::new(&TIME, &ev), &o).unwrap();
    vclose("ridge coef", &f.coefficients, &RIDGE_COEF, 1e-8);
    mclose("ridge var", &f.var, &RIDGE_VAR, 1e-7);
}

#[test]
fn predicted_survival_matches_survfit_coxph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &SEX, &ECOG, &WTLOSS]);
    let f = coxph(&x, &CoxInput::new(&TIME, &ev), &opts(Ties::Efron)).unwrap();
    let nd = [[60.0, 1.0, 1.0, 5.0], [70.0, 2.0, 2.0, 10.0]];
    let m = PRED_TIME.len();
    for (c, row) in nd.iter().enumerate() {
        let s = f.predict_survival(row, None).unwrap();
        vclose("time", &s.time, &PRED_TIME, 0.0);
        let col = |v: &[f64]| v[c * m..(c + 1) * m].to_vec();
        vclose(&format!("surv {c}"), &s.surv, &col(&PRED_SURV), 1e-8);
        vclose(&format!("cumhaz {c}"), &s.cumhaz, &col(&PRED_CUMHAZ), 1e-8);
        vclose(&format!("se {c}"), &s.std_err, &col(&PRED_SE), 1e-7);
        vclose(&format!("lower {c}"), &s.lower, &col(&PRED_LOWER), 1e-7);
        vclose(&format!("upper {c}"), &s.upper, &col(&PRED_UPPER), 1e-7);
        vclose(&format!("n.risk {c}"), &s.n_risk, &PRED_NRISK, 0.0);
    }
    let fb = coxph(&x, &CoxInput::new(&TIME, &ev), &opts(Ties::Breslow)).unwrap();
    let s = fb.predict_survival(&nd[0], None).unwrap();
    vclose("breslow surv", &s.surv, &PREDB_SURV, 1e-8);
    vclose("breslow se", &s.std_err, &PREDB_SE, 1e-7);
}

#[test]
fn counting_process_matches_coxph() {
    let ev = events(&H_EVENT);
    let x = Mat::from_fn(H_STOP.len(), 3, |i, j| {
        [H_AGE[i], H_TRANSPLANT[i], H_SURGERY[i]][j]
    });
    let input = CoxInput::new(&H_STOP, &ev).entry(&H_START);
    let f = coxph(&x, &input, &opts(Ties::Efron)).unwrap();
    check_fit(
        "heart",
        &f,
        &HEART_COEF,
        &HEART_VAR,
        &HEART_LOGLIK,
        HEART_SCORE,
        HEART_WALD,
    );
    mclose(
        "heart robust",
        &f.robust_vcov(&H_ID).unwrap(),
        &HEART_ROBUST_VAR,
        1e-7,
    );
    let fb = coxph(&x, &input, &opts(Ties::Breslow)).unwrap();
    check_fit(
        "heart breslow",
        &fb,
        &HEARTB_COEF,
        &HEARTB_VAR,
        &HEARTB_LOGLIK,
        HEARTB_SCORE,
        HEARTB_WALD,
    );
}
