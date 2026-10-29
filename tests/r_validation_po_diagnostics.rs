#![allow(clippy::excessive_precision)]

//! #110: proportional-odds diagnostics (RMS §13.3). The score test is checked
//! against the score and information of the partial proportional-odds
//! likelihood computed by `numDeriv` at `rms::orm`'s estimate, and
//! `xmean_ordinaly` against `rms::plot.xmean.ordinaly`'s computation. Reference
//! values come from `tests/r_scripts/generate_po_diagnostics_validation.R`.

use anofox_regression::solvers::{xmean_ordinaly, OrdinalLink, OrdinalRegressor};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/po_diagnostics_validation.rs");
}
use fixture::*;

const N: usize = 150;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

#[test]
fn proportional_odds_score_test_matches_numderiv() {
    let x = Mat::from_fn(N, 3, |i, j| X[i * 3 + j]);
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    let logit = OrdinalRegressor::builder().build().fit(&x, &y).unwrap();
    let cases: [(&str, &[usize], f64, usize); 3] = [
        ("all", &[], SCORE_ALL[0], 9),
        ("x1", &[0], SCORE_X1[0], 3),
        ("x3", &[2], SCORE_X3[0], 3),
    ];
    for (id, cols, expected, df) in cases {
        let t = logit.proportional_odds_score_test(&x, &y, cols).unwrap();
        // numDeriv's Richardson extrapolation is accurate to ~1e-7 here
        close(&format!("score {id}"), t.statistic, expected, 1e-6);
        assert_eq!(t.df, df, "{id}");
        assert!(t.p_value > 0.0 && t.p_value < 1.0);
    }
    let probit = OrdinalRegressor::builder()
        .link(OrdinalLink::Probit)
        .build()
        .fit(&x, &y)
        .unwrap();
    let t = probit
        .proportional_odds_score_test(&x, &y, &[0, 1])
        .unwrap();
    close("probit x1, x2", t.statistic, SCORE_PROBIT_X1_X2[0], 1e-6);
    assert_eq!(t.df, 6);
    // bad input
    assert!(logit.proportional_odds_score_test(&x, &y, &[3]).is_err());
    let y_bad = Col::from_fn(N, |i| Y_CAT[i] + 0.5);
    assert!(logit.proportional_odds_score_test(&x, &y_bad, &[]).is_err());
    let x_short = Mat::from_fn(10, 3, |i, j| X[i * 3 + j]);
    assert!(logit
        .proportional_odds_score_test(&x_short, &Col::from_fn(10, |i| Y_CAT[i]), &[])
        .is_err());
}

#[test]
fn xmean_ordinaly_matches_rms() {
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    for (id, col, expected) in [("x1", 0, &XMEAN_X1), ("x3", 2, &XMEAN_X3)] {
        let x = Col::from_fn(N, |i| X[i * 3 + col]);
        let r = xmean_ordinaly(&x, &y, true).unwrap();
        assert_eq!(r.levels, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let cr = r.expected_cr.as_ref().unwrap();
        for j in 0..5 {
            close(
                &format!("{id} observed {j}"),
                r.observed[j],
                expected[j],
                1e-12,
            );
            close(
                &format!("{id} PO {j}"),
                r.expected_po[j],
                expected[5 + j],
                1e-7,
            );
            close(&format!("{id} CR {j}"), cr[j], expected[10 + j], 1e-7);
        }
        assert!(xmean_ordinaly(&x, &y, false).unwrap().expected_cr.is_none());
    }
}
