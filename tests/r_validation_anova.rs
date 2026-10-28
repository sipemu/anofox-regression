#![allow(clippy::excessive_precision, clippy::needless_range_loop, dead_code)]

//! #100: Wald chunk tests, general linear hypotheses and the `rms::anova.rms`
//! table.
//!
//! Reference values come from `tests/r_scripts/generate_anova_validation.R`.

use anofox_regression::design::{Design, TermMap};
use anofox_regression::inference::{
    anova, drop_intercept, wald_chunk, wald_test, AnovaOptions, WaldDistribution,
};
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use anofox_regression::splines::{Basis, Rcs};
use faer::{Col, Mat};

include!("fixtures/anova_validation.rs");

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn mat(v: &[f64], n: usize) -> Mat<f64> {
    Mat::from_fn(v.len() / n, n, |i, j| v[i * n + j])
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// Tail probabilities: statrs' upper tails are accurate to ~1e-6 relative
/// for p near 1e-11 (the statistics themselves agree to 1e-9).
fn p_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-5 * b + 1e-15
}

const RCS4: [bool; 3] = [false, true, true];
const RCS3: [bool; 2] = [false, true];

/// rms term order of `y ~ rcs(x1,4) * x3 + rcs(x2,3) + x4`.
fn terms_m1() -> TermMap {
    let mut t = TermMap::new();
    let x1 = t.add_main("x1", &RCS4);
    let x3 = t.add_main("x3", &[false]);
    t.add_main("x2", &RCS3);
    t.add_main("x4", &[false]);
    t.add_interaction(x1, x3, false).unwrap();
    t
}

/// `y ~ rcs(x1,4) * rcs(x2,3) + rcs(x1,4) * x3 + x4`.
fn terms_m3() -> TermMap {
    let mut t = TermMap::new();
    let x1 = t.add_main("x1", &RCS4);
    let x2 = t.add_main("x2", &RCS3);
    let x3 = t.add_main("x3", &[false]);
    t.add_main("x4", &[false]);
    t.add_interaction(x1, x2, false).unwrap();
    t.add_interaction(x1, x3, false).unwrap();
    t
}

/// `y ~ rcs(x1,4) + rcs(x2,3) + rcs(x1,4) %ia% rcs(x2,3)`.
fn terms_m4() -> TermMap {
    let mut t = TermMap::new();
    let x1 = t.add_main("x1", &RCS4);
    let x2 = t.add_main("x2", &RCS3);
    t.add_interaction(x1, x2, true).unwrap();
    t
}

#[allow(clippy::too_many_arguments)]
fn check_table(
    id: &str,
    terms: &TermMap,
    beta: &[f64],
    vcov: &[f64],
    labels: &[&str],
    chisq: &[f64],
    df: &[f64],
    p: &[f64],
    which_len: &[usize],
    which: &[usize],
) {
    let n = beta.len();
    let table = anova(&col(beta), &mat(vcov, n), terms, &AnovaOptions::default()).unwrap();
    let got: Vec<&str> = table.rows.iter().map(|r| r.label.trim_end()).collect();
    let want: Vec<&str> = labels.iter().map(|l| l.trim_end()).collect();
    assert_eq!(got, want, "{id} labels");
    let mut at = 0;
    for (k, r) in table.rows.iter().enumerate() {
        let w = &which[at..at + which_len[k]];
        at += which_len[k];
        assert_eq!(r.columns, w, "{id} {}: columns", r.label);
        assert_eq!(r.test.df as f64, df[k], "{id} {}: df", r.label);
        assert!(
            close(r.test.chi_square, chisq[k], 1e-9),
            "{id} {}: {} vs {}",
            r.label,
            r.test.chi_square,
            chisq[k]
        );
        assert!(
            p_close(r.test.p_value, p[k]),
            "{id} {}: p {} vs {}",
            r.label,
            r.test.p_value,
            p[k]
        );
    }
}

#[test]
fn ols_anova_matches_rms() {
    check_table(
        "M1",
        &terms_m1(),
        &BETA_M1,
        &VCOV_M1,
        &LABELS_M1,
        &CHISQ_M1,
        &DF_M1,
        &P_M1,
        &WHICH_LEN_M1,
        &WHICH_M1,
    );
}

#[test]
fn lrm_anova_matches_rms() {
    check_table(
        "M2",
        &terms_m1(),
        &BETA_M2,
        &VCOV_M2,
        &LABELS_M2,
        &CHISQ_M2,
        &DF_M2,
        &P_M2,
        &WHICH_LEN_M2,
        &WHICH_M2,
    );
}

#[test]
fn tensor_spline_interactions_match_rms() {
    check_table(
        "M3",
        &terms_m3(),
        &BETA_M3,
        &VCOV_M3,
        &LABELS_M3,
        &CHISQ_M3,
        &DF_M3,
        &P_M3,
        &WHICH_LEN_M3,
        &WHICH_M3,
    );
}

#[test]
fn restricted_interaction_matches_rms() {
    check_table(
        "M4",
        &terms_m4(),
        &BETA_M4,
        &VCOV_M4,
        &LABELS_M4,
        &CHISQ_M4,
        &DF_M4,
        &P_M4,
        &WHICH_LEN_M4,
        &WHICH_M4,
    );
}

#[test]
fn f_tests_and_partial_ss_match_rms() {
    let n = BETA_M1.len();
    let opts = AnovaOptions {
        distribution: WaldDistribution::F {
            df_residual: DFE_M1,
        },
        sigma2: Some(SIGMA2_M1),
    };
    let table = anova(&col(&BETA_M1), &mat(&VCOV_M1, n), &terms_m1(), &opts).unwrap();
    for (k, r) in table.rows.iter().enumerate() {
        assert!(close(r.test.f.unwrap(), F_M1[k], 1e-9), "{}: F", r.label);
        assert!(
            p_close(r.test.p_value, FP_M1[k]),
            "{}: p {} vs {}",
            r.label,
            r.test.p_value,
            FP_M1[k]
        );
        assert!(
            close(r.partial_ss.unwrap(), SS_M1[k], 1e-9),
            "{}: SS",
            r.label
        );
    }
    let (dfe, sse, mse) = table.error.unwrap();
    assert_eq!(dfe, DFE_M1);
    assert!(close(sse, SS_M1[SS_M1.len() - 1], 1e-12));
    assert!(close(mse, SIGMA2_M1, 1e-12));
}

/// Build the M1 design in Rust, fit OLS, and get the same ANOVA from the
/// crate's own estimates.
#[test]
fn end_to_end_design_fit_anova() {
    let rcs1 = Rcs::new(&KNOTS_X1).unwrap();
    let rcs2 = Rcs::new(&KNOTS_X2).unwrap();
    let mut d = Design::new();
    let x1 = d.add("x1", rcs1.basis(&X1, "x1")).unwrap();
    let x3 = d.add("x3", Basis::linear(&X3, "x3")).unwrap();
    d.add("x2", rcs2.basis(&X2, "x2")).unwrap();
    d.add("x4", Basis::linear(&X4, "x4")).unwrap();
    d.add_interaction(x1, x3).unwrap();
    assert_eq!(d.terms(), &terms_m1());

    let fit = OlsRegressor::builder()
        .compute_inference(true)
        .build()
        .fit(&d.matrix(), &col(&Y))
        .unwrap();
    let mut beta_full = vec![fit.result().intercept.unwrap()];
    beta_full.extend((0..BETA_M1.len()).map(|j| fit.result().coefficients[j]));
    let (beta, vcov) = drop_intercept(&col(&beta_full), &fit.vcov().unwrap());
    let n = BETA_M1.len();
    for j in 0..n {
        assert!(close(beta[j], BETA_M1[j], 1e-9), "beta {j}");
        for k in 0..n {
            assert!(
                close(vcov[(j, k)], VCOV_M1[j * n + k], 1e-9),
                "vcov {j},{k}"
            );
        }
    }
    let table = anova(&beta, &vcov, d.terms(), &AnovaOptions::default()).unwrap();
    for (k, r) in table.rows.iter().enumerate() {
        assert!(close(r.test.chi_square, CHISQ_M1[k], 1e-8), "{}", r.label);
    }
    assert!(close(
        table.row("TOTAL").unwrap().test.chi_square,
        CHISQ_M1[CHISQ_M1.len() - 1],
        1e-8
    ));
}

#[test]
fn chunk_f_equals_nested_model_f() {
    let n = BETA_M1.len();
    let t = wald_chunk(
        &col(&BETA_M1),
        &mat(&VCOV_M1, n),
        &[0, 1, 2],
        WaldDistribution::F {
            df_residual: DFE_M1,
        },
    )
    .unwrap();
    assert!(
        close(t.f.unwrap(), NESTED_F, 1e-9),
        "{} vs {NESTED_F}",
        t.f.unwrap()
    );
    assert!(
        (t.p_value / NESTED_P - 1.0).abs() < 1e-6,
        "{} vs {NESTED_P}",
        t.p_value
    );
}

#[test]
fn general_linear_hypothesis() {
    let p = LM_BETA.len();
    let c = mat(&GLH_C, p);
    let beta = col(&LM_BETA);
    let vcov = mat(&LM_VCOV, p);
    let r = col(&GLH_R);
    let chi = wald_test(&beta, &vcov, &c, Some(&r), WaldDistribution::ChiSquare).unwrap();
    assert_eq!(chi.df, 2);
    assert!(close(chi.chi_square, GLH_CHISQ, 1e-10));
    assert!(close(chi.p_value, GLH_CHISQ_P, 1e-9));
    let f = wald_test(
        &beta,
        &vcov,
        &c,
        Some(&r),
        WaldDistribution::F {
            df_residual: LM_DFE,
        },
    )
    .unwrap();
    assert!(close(f.f.unwrap(), GLH_CHISQ / 2.0, 1e-10));
    assert!(close(f.p_value, GLH_F_P, 1e-9));
}
