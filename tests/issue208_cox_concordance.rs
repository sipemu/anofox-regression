#![allow(clippy::excessive_precision, dead_code)]

//! #208: `CoxFit::concordance` matches `coxph`'s `concordance` with case
//! weights, an offset, strata and `(start, stop]` data.
//!
//! Reference values: `tests/r_scripts/generate_cox_concordance_validation.R`
//! (data of `tests/fixtures/cox_validation.rs`).

use anofox_regression::survival::cox::{coxph, CoxInput, CoxOptions, Ties};
use faer::Mat;

include!("fixtures/cox_validation.rs");

const CONC_WEIGHTED: f64 = 0.65099991218989173;
const CONC_WEIGHTED_OFFSET: f64 = 0.63532247647319784;
const CONC_STRATA: f64 = 0.61657162026165713;
const CONC_HEART: f64 = 0.5998008408940031;
const CONC_HEART_W: f64 = 0.6299659250350772;
const CONC_TIES: f64 = 0.57967647058823535;

fn opts() -> CoxOptions {
    CoxOptions {
        ties: Ties::Efron,
        eps: 1e-12,
        max_iter: 50,
        ..Default::default()
    }
}

fn events(v: &[f64]) -> Vec<bool> {
    v.iter().map(|&s| s == 1.0).collect()
}

fn lung_x(cols: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(TIME.len(), cols.len(), |i, j| cols[j][i])
}

fn close(name: &str, a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{name}: {a} vs R {b}");
}

#[test]
fn weighted_and_offset_concordance_match_coxph() {
    let ev = events(&STATUS);
    let x = lung_x(&[&AGE, &SEX, &ECOG, &WTLOSS]);
    let f = coxph(&x, &CoxInput::new(&TIME, &ev).weights(&W), &opts()).unwrap();
    close("weighted", f.concordance(), CONC_WEIGHTED);
    let f = coxph(
        &x,
        &CoxInput::new(&TIME, &ev).weights(&W).offset(&OFF),
        &opts(),
    )
    .unwrap();
    close("weighted + offset", f.concordance(), CONC_WEIGHTED_OFFSET);
}

#[test]
fn stratified_and_tied_concordance_match_coxph() {
    let ev = events(&STATUS);
    let sex: Vec<usize> = SEX.iter().map(|&s| s as usize).collect();
    let x = lung_x(&[&AGE, &ECOG, &WTLOSS]);
    let f = coxph(&x, &CoxInput::new(&TIME, &ev).strata(&sex), &opts()).unwrap();
    close("strata", f.concordance(), CONC_STRATA);
    let f = coxph(&lung_x(&[&SEX]), &CoxInput::new(&TIME, &ev), &opts()).unwrap();
    close("tied linear predictor", f.concordance(), CONC_TIES);
}

#[test]
fn counting_process_concordance_matches_coxph() {
    let ev = events(&H_EVENT);
    let x = Mat::from_fn(H_STOP.len(), 3, |i, j| {
        [H_AGE[i], H_TRANSPLANT[i], H_SURGERY[i]][j]
    });
    let input = CoxInput::new(&H_STOP, &ev).entry(&H_START);
    let f = coxph(&x, &input, &opts()).unwrap();
    close("(start, stop]", f.concordance(), CONC_HEART);
    let w: Vec<f64> = (0..H_STOP.len()).map(|i| [1.0, 2.0, 0.5][i % 3]).collect();
    let f = coxph(&x, &input.weights(&w), &opts()).unwrap();
    close("(start, stop], weighted", f.concordance(), CONC_HEART_W);
}
