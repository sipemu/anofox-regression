#![allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]

//! #115: Cox residuals, the `cox.zph` proportional-hazards test and AFT
//! residuals against `survival`.
//!
//! Reference values come from `tests/r_scripts/generate_coxdiag_validation.R`.

use anofox_regression::solvers::aft::{AftDistribution, AftRegressor};
use anofox_regression::survival::cox::{coxph, CoxFit, CoxInput, CoxOptions, Ties, ZphTransform};
use faer::{Col, Mat};

include!("fixtures/coxdiag_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    let ok = (a - b).abs() <= tol * (1.0 + b.abs());
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

fn opts(ties: Ties) -> CoxOptions {
    CoxOptions {
        ties,
        eps: 1e-12,
        max_iter: 50,
        ..Default::default()
    }
}

fn lung(cols: &[&[f64]], ties: Ties, weights: bool, strata: bool) -> CoxFit {
    let ev: Vec<bool> = STATUS.iter().map(|&s| s == 1.0).collect();
    let x = Mat::from_fn(TIME.len(), cols.len(), |i, j| cols[j][i]);
    let sex: Vec<usize> = SEX.iter().map(|&s| s as usize).collect();
    let mut input = CoxInput::new(&TIME, &ev);
    if weights {
        input = input.weights(&W);
    }
    if strata {
        input = input.strata(&sex);
    }
    // `input` borrows `ev` and `sex`, which live until the fit is built.
    coxph(&x, &input, &opts(ties)).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn check_residuals(
    id: &str,
    f: &CoxFit,
    mart: &[f64],
    dev: &[f64],
    sch: &[f64],
    scsch: &[f64],
    dfb: &[f64],
    dfbs: &[f64],
) {
    vclose(
        &format!("{id} martingale"),
        &f.martingale_residuals(),
        mart,
        1e-8,
    );
    vclose(
        &format!("{id} deviance"),
        &f.deviance_residuals(),
        dev,
        1e-8,
    );
    mclose(
        &format!("{id} schoenfeld"),
        &f.schoenfeld_residuals(false).2,
        sch,
        1e-8,
    );
    mclose(
        &format!("{id} scaledsch"),
        &f.schoenfeld_residuals(true).2,
        scsch,
        1e-6,
    );
    mclose(&format!("{id} dfbeta"), &f.dfbeta(), dfb, 1e-7);
    mclose(&format!("{id} dfbetas"), &f.dfbetas(), dfbs, 1e-7);
}

fn check_zph(id: &str, f: &CoxFit, t: ZphTransform, want: &[f64]) {
    let z = f.cox_zph(t, None).unwrap();
    let rows: Vec<_> = z.terms.iter().chain(std::iter::once(&z.global)).collect();
    assert_eq!(rows.len() * 3, want.len(), "{id} zph rows");
    for (k, r) in rows.iter().enumerate() {
        close(
            &format!("{id} {t:?} chisq[{k}]"),
            r.chisq,
            want[3 * k],
            1e-7,
        );
        assert_eq!(r.df as f64, want[3 * k + 1]);
        close(
            &format!("{id} {t:?} p[{k}]"),
            r.p_value,
            want[3 * k + 2],
            1e-6,
        );
    }
}

const ALL: [&[f64]; 4] = [&AGE, &SEX, &ECOG, &WTLOSS];

#[test]
fn efron_residuals_and_zph() {
    let f = lung(&ALL, Ties::Efron, false, false);
    check_residuals(
        "efron",
        &f,
        &EFRON_MART,
        &EFRON_DEV,
        &EFRON_SCH,
        &EFRON_SCSCH,
        &EFRON_DFBETA,
        &EFRON_DFBETAS,
    );
    check_zph("efron", &f, ZphTransform::Km, &EFRON_ZPH_KM);
    check_zph("efron", &f, ZphTransform::Rank, &EFRON_ZPH_RANK);
    check_zph("efron", &f, ZphTransform::Identity, &EFRON_ZPH_IDENTITY);
    check_zph("efron", &f, ZphTransform::Log, &EFRON_ZPH_LOG);
    let z = f.cox_zph(ZphTransform::Km, None).unwrap();
    vclose("zph x", &z.x, &EFRON_ZPH_X, 1e-12);
    mclose("zph y", &z.y, &EFRON_ZPH_Y, 1e-6);
}

#[test]
fn breslow_weighted_and_stratified() {
    let f = lung(&ALL, Ties::Breslow, false, false);
    check_residuals(
        "breslow",
        &f,
        &BRESLOW_MART,
        &BRESLOW_DEV,
        &BRESLOW_SCH,
        &BRESLOW_SCSCH,
        &BRESLOW_DFBETA,
        &BRESLOW_DFBETAS,
    );
    check_zph("breslow", &f, ZphTransform::Km, &BRESLOW_ZPH_KM);
    let f = lung(&ALL, Ties::Efron, true, false);
    check_residuals(
        "weighted",
        &f,
        &WEIGHTED_MART,
        &WEIGHTED_DEV,
        &WEIGHTED_SCH,
        &WEIGHTED_SCSCH,
        &WEIGHTED_DFBETA,
        &WEIGHTED_DFBETAS,
    );
    check_zph("weighted", &f, ZphTransform::Km, &WEIGHTED_ZPH_KM);
    let f = lung(&[&AGE, &ECOG, &WTLOSS], Ties::Efron, false, true);
    check_residuals(
        "strata",
        &f,
        &STRATA_MART,
        &STRATA_DEV,
        &STRATA_SCH,
        &STRATA_SCSCH,
        &STRATA_DFBETA,
        &STRATA_DFBETAS,
    );
    check_zph("strata", &f, ZphTransform::Km, &STRATA_ZPH_KM);
}

#[test]
fn counting_process_residuals_and_zph() {
    let ev: Vec<bool> = H_EVENT.iter().map(|&e| e == 1.0).collect();
    let x = Mat::from_fn(H_STOP.len(), 3, |i, j| {
        [H_AGE[i], H_TRANSPLANT[i], H_SURGERY[i]][j]
    });
    let f = coxph(
        &x,
        &CoxInput::new(&H_STOP, &ev).entry(&H_START),
        &opts(Ties::Efron),
    )
    .unwrap();
    check_residuals(
        "heart",
        &f,
        &HEART_MART,
        &HEART_DEV,
        &HEART_SCH,
        &HEART_SCSCH,
        &HEART_DFBETA,
        &HEART_DFBETAS,
    );
    check_zph("heart", &f, ZphTransform::Km, &HEART_ZPH_KM);
    check_zph("heart", &f, ZphTransform::Identity, &HEART_ZPH_IDENTITY);
}

#[test]
fn aft_residuals() {
    let ev: Vec<bool> = STATUS.iter().map(|&s| s == 1.0).collect();
    let x = Mat::from_fn(TIME.len(), 4, |i, j| ALL[j][i]);
    let t = Col::from_fn(TIME.len(), |i| TIME[i]);
    let fit = AftRegressor::builder()
        .distribution(AftDistribution::Weibull)
        .tolerance(1e-12)
        .build()
        .fit(&x, &t, &ev)
        .unwrap();
    let z = fit.standardized_residuals(&x, &t);
    let cs = fit.cox_snell_residuals(&x, &t);
    for i in 0..TIME.len() {
        close(&format!("std[{i}]"), z[i], AFT_STD[i], 1e-6);
        close(&format!("cox-snell[{i}]"), cs[i], AFT_COXSNELL[i], 1e-6);
    }
}
