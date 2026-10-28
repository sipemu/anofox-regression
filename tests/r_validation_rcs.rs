#![allow(
    clippy::excessive_precision,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

//! #99: restricted cubic / linear spline bases against Hmisc and rms.
//!
//! Reference values come from `tests/r_scripts/generate_rcs_validation.R`
//! (`Hmisc::rcspline.eval`, `Hmisc::rcspline.restate`, `rms::rcs`,
//! `rms::lsp`, `rms::%ia%`).

use anofox_regression::splines::{rcs_default_knots, restricted_interaction, Lsp, Rcs, RcsNorm};

include!("fixtures/rcs_validation.rs");

fn assert_close(name: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length {got:?} vs {want:?}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + b.abs()),
            "{name}[{i}]: {a} vs {b}"
        );
    }
}

fn row_major(m: &faer::Mat<f64>) -> Vec<f64> {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
        .collect()
}

#[test]
fn default_knots_match_rcspline_eval() {
    let cases: [(&str, &[f64], [&[f64]; 5]); 6] = [
        (
            "SMALL",
            &X_SMALL,
            [
                &KNOTS_SMALL_3,
                &KNOTS_SMALL_4,
                &KNOTS_SMALL_5,
                &KNOTS_SMALL_6,
                &KNOTS_SMALL_7,
            ],
        ),
        (
            "LARGE",
            &X_LARGE,
            [
                &KNOTS_LARGE_3,
                &KNOTS_LARGE_4,
                &KNOTS_LARGE_5,
                &KNOTS_LARGE_6,
                &KNOTS_LARGE_7,
            ],
        ),
        (
            "TIEDLOW",
            &X_TIEDLOW,
            [
                &KNOTS_TIEDLOW_3,
                &KNOTS_TIEDLOW_4,
                &KNOTS_TIEDLOW_5,
                &KNOTS_TIEDLOW_6,
                &KNOTS_TIEDLOW_7,
            ],
        ),
        (
            "TIEDBOTH",
            &X_TIEDBOTH,
            [
                &KNOTS_TIEDBOTH_3,
                &KNOTS_TIEDBOTH_4,
                &KNOTS_TIEDBOTH_5,
                &KNOTS_TIEDBOTH_6,
                &KNOTS_TIEDBOTH_7,
            ],
        ),
        (
            "FEWU",
            &X_FEWU,
            [
                &KNOTS_FEWU_3,
                &KNOTS_FEWU_4,
                &KNOTS_FEWU_5,
                &KNOTS_FEWU_6,
                &KNOTS_FEWU_7,
            ],
        ),
        (
            "INTS",
            &X_INTS,
            [
                &KNOTS_INTS_3,
                &KNOTS_INTS_4,
                &KNOTS_INTS_5,
                &KNOTS_INTS_6,
                &KNOTS_INTS_7,
            ],
        ),
    ];
    for (id, x, expected) in cases {
        for (nk, want) in (3..=7).zip(expected) {
            match rcs_default_knots(x, nk) {
                Ok(k) => assert_close(&format!("{id} nk={nk}"), &k, want, 1e-14),
                Err(e) => assert!(want.is_empty(), "{id} nk={nk}: {e}"),
            }
        }
    }
}

#[test]
fn basis_matches_rcspline_eval_for_every_norm() {
    let rcs = Rcs::from_data(&X_SMALL, 5).unwrap();
    assert_close("knots", rcs.knots(), &KNOTS_SMALL_5, 1e-14);
    for (norm, want) in [
        (RcsNorm::None, &BASIS_SMALL_5_NORM0[..]),
        (RcsNorm::LastInterval, &BASIS_SMALL_5_NORM1[..]),
        (RcsNorm::Range, &BASIS_SMALL_5_NORM2[..]),
    ] {
        let b = rcs.clone().with_norm(norm).basis(&X_SMALL, "x");
        assert_close(&format!("{norm:?}"), &row_major(&b.columns), want, 1e-12);
    }
}

#[test]
fn basis_matches_rms_rcs() {
    let rcs = Rcs::from_data(&X_LARGE, 4).unwrap();
    assert_close("knots", rcs.knots(), &RMS_RCS_LARGE_4_KNOTS, 1e-14);
    let b = rcs.basis(&X_LARGE, "x");
    assert_eq!(b.names, ["x", "x'", "x''"]);
    assert_eq!(b.nonlinear, [false, true, true]);
    assert_eq!(b.nonlinear_columns(), [1, 2]);
    assert_close("rcs", &row_major(&b.columns), &RMS_RCS_LARGE_4, 1e-12);
}

#[test]
fn restate_matches_rcspline_restate() {
    let rcs = Rcs::from_data(&X_SMALL, 5).unwrap();
    let r = rcs
        .restate(RESTATE_COEF_IN[0], &RESTATE_COEF_IN[1..])
        .unwrap();
    // R returns the intercept, the linear coefficient and one per knot.
    let mut got = vec![r.intercept, r.linear];
    got.extend(&r.cubic);
    assert_close("restate", &got, &RESTATE_COEF_OUT, 1e-12);
    // The algebraic form reproduces the basis fit.
    let b = rcs.basis(&X_SMALL, "x");
    for i in 0..X_SMALL.len() {
        let fit: f64 = RESTATE_COEF_IN[0]
            + (0..4)
                .map(|j| RESTATE_COEF_IN[j + 1] * b.columns[(i, j)])
                .sum::<f64>();
        assert!((r.eval(X_SMALL[i]) - fit).abs() < 1e-10);
    }
}

#[test]
fn linear_spline_matches_rms_lsp() {
    let b = Lsp::new(&LSP_KNOTS).unwrap().basis(&X_SMALL, "x");
    assert_eq!(b.names, ["x", "x'", "x''", "x'''"]);
    assert_close("lsp", &row_major(&b.columns), &LSP_SMALL, 1e-14);
}

#[test]
fn restricted_interaction_matches_rms_ia() {
    let a = Rcs::from_data(&X_SMALL, 4).unwrap().basis(&X_SMALL, "x1");
    let b = Rcs::from_data(&IA_X2, 3).unwrap().basis(&IA_X2, "x2");
    let ia = restricted_interaction(&a, &b).unwrap();
    assert_eq!(ia.n_columns(), IA_NCOL);
    assert_eq!(ia.names, ["x1 * x2", "x1 * x2'", "x1' * x2", "x1'' * x2"]);
    assert_eq!(ia.nonlinear, IA_NONLINEAR);
    assert_close("ia", &row_major(&ia.columns), &IA, 1e-12);
}
