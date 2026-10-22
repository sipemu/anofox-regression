//! Issue #89: GLMM inference against lme4 / lmerTest — the fixed-effect
//! covariance, Satterthwaite t-tests for Gaussian LMMs, conditional standard
//! errors of the BLUPs, and the training data's fitted values and residuals.
//!
//! References: `tests/r_scripts/generate_glmm_satterthwaite_validation.R`
//! (lme4 2.0.1, lmerTest), on the scenario data of
//! `tests/fixtures/glmm_validation.rs` and `glmm_families_validation.rs`.

#![allow(dead_code)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]

use anofox_regression::solvers::{FittedGlmm, GlmmDdf, GlmmRegressor, GlmmRegressorBuilder};
use faer::{Col, Mat};

include!("fixtures/glmm_validation.rs");
include!("fixtures/glmm_families_validation.rs");
include!("fixtures/glmm_inference_validation.rs");

/// Fixed-effect covariance, BLUP standard errors, fitted values and residuals
/// (relative). The references are fitted with tight optimizer / PIRLS
/// tolerances and so are the fits here (`tolerance(1e-10)`).
const SE_TOL: f64 = 1e-6;
/// Satterthwaite degrees of freedom (relative; they agree with lmerTest to
/// about 1e-6).
const DF_TOL: f64 = 1e-5;
/// Negative binomial with θ estimated: the size is found by golden-section
/// search on log θ, which `glmer.nb` reaches by a different route.
const NB_TOL: f64 = 1e-4;

fn close(tag: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{tag}: {got} vs {want} (rel {:.3e})",
        (got - want).abs() / want.abs().max(1e-300)
    );
}

fn close_all(tag: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    for (j, (g, w)) in got.iter().zip(want).enumerate() {
        close(&format!("{tag}[{j}]"), *g, *w, tol);
    }
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn design(cols: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(cols[0].len(), cols.len(), |i, j| cols[j][i])
}

fn check_vcov(tag: &str, fit: &FittedGlmm, want: &[f64], tol: f64) {
    let flat: Vec<f64> = fit.vcov().iter().flatten().copied().collect();
    close_all(&format!("{tag} vcov"), &flat, want, tol);
}

fn check_satterthwaite(tag: &str, fit: &FittedGlmm, df: &[f64], p: &[f64], lo: &[f64], hi: &[f64]) {
    close_all(
        &format!("{tag} df"),
        fit.satterthwaite_df().unwrap(),
        df,
        DF_TOL,
    );
    let inf = fit.inference(GlmmDdf::Satterthwaite, 0.95).unwrap();
    close_all(
        &format!("{tag} df (inference)"),
        inf.df.as_ref().unwrap(),
        df,
        DF_TOL,
    );
    close_all(&format!("{tag} p"), &inf.p_values, p, 1e-6);
    close_all(&format!("{tag} lo95"), &inf.conf_low, lo, 1e-7);
    close_all(&format!("{tag} hi95"), &inf.conf_high, hi, 1e-7);
    close_all(&format!("{tag} t"), &inf.statistics, &fit.z_values(), 0.0);
}

/// BLUP SEs of a factor, comparing level `l` of R's (sorted) order with the
/// fit's first-seen compaction of `ids`.
fn check_re_se(tag: &str, got: &[Vec<f64>], ids: &[usize], want: &[f64], tol: f64) {
    let mut order: Vec<usize> = Vec::new();
    for &g in ids {
        if !order.contains(&g) {
            order.push(g);
        }
    }
    let q = got[0].len();
    assert_eq!(got.len() * q, want.len(), "{tag}: shape");
    for (c, &level) in order.iter().enumerate() {
        for a in 0..q {
            close(
                &format!("{tag}[{level}][{a}]"),
                got[c][a],
                want[level * q + a],
                tol,
            );
        }
    }
}

fn check_observations(
    tag: &str,
    fit: &FittedGlmm,
    fitted: &[f64],
    pop: &[f64],
    pearson: &[f64],
    df_resid: f64,
    tol: f64,
) {
    close_all(&format!("{tag} fitted"), fit.fitted_values(), fitted, tol);
    close_all(
        &format!("{tag} fitted_population"),
        fit.fitted_population(),
        pop,
        tol,
    );
    close_all(
        &format!("{tag} pearson"),
        fit.pearson_residuals(),
        pearson,
        tol,
    );
    // Response residuals are y − fitted, and equal the Pearson residuals where
    // the variance function is 1 (Gaussian).
    let resid = fit.residuals();
    let y: Vec<f64> = resid
        .iter()
        .zip(fit.fitted_values())
        .map(|(r, m)| r + m)
        .collect();
    for (i, r) in resid.iter().enumerate() {
        close(&format!("{tag} resid[{i}]"), *r, y[i] - fitted[i], tol);
    }
    assert_eq!(fit.df_residual(), df_resid, "{tag} df_residual");
}

fn lmm() -> GlmmRegressorBuilder {
    GlmmRegressor::gaussian()
        .tolerance(1e-10)
        .satterthwaite(true)
}

#[test]
fn lmm_random_intercept_matches_lmertest() {
    let fit = lmm()
        .build()
        .fit(&design(&[&X_LMM_A]), &col(&Y_LMM_A), &G_LMM_A)
        .unwrap();
    check_vcov("LMM_A", &fit, &LMM_A_VCOV, SE_TOL);
    check_satterthwaite(
        "LMM_A",
        &fit,
        &LMM_A_SATT_DF,
        &LMM_A_SATT_P,
        &LMM_A_SATT_LO95,
        &LMM_A_SATT_HI95,
    );
    check_re_se(
        "LMM_A re_se",
        fit.random_effects_se_matrix(),
        &G_LMM_A,
        &LMM_A_RE_SE_G,
        SE_TOL,
    );
    check_observations(
        "LMM_A",
        &fit,
        &LMM_A_FITTED,
        &LMM_A_FITTED_POP,
        &LMM_A_PEARSON,
        LMM_A_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn lmm_two_covariates_matches_lmertest() {
    let fit = lmm()
        .build()
        .fit(&design(&[&X1_LMM_B, &X2_LMM_B]), &col(&Y_LMM_B), &G_LMM_B)
        .unwrap();
    check_vcov("LMM_B", &fit, &LMM_B_VCOV, SE_TOL);
    check_satterthwaite(
        "LMM_B",
        &fit,
        &LMM_B_SATT_DF,
        &LMM_B_SATT_P,
        &LMM_B_SATT_LO95,
        &LMM_B_SATT_HI95,
    );
    check_re_se(
        "LMM_B re_se",
        fit.random_effects_se_matrix(),
        &G_LMM_B,
        &LMM_B_RE_SE_G,
        SE_TOL,
    );
}

#[test]
fn lmm_random_slope_matches_lmertest() {
    let fit = lmm()
        .random_slopes(vec![0])
        .build()
        .fit(&design(&[&X_SLOPE_E]), &col(&Y_SLOPE_E), &G_SLOPE_E)
        .unwrap();
    check_vcov("SLOPE_E", &fit, &SLOPE_E_VCOV, SE_TOL);
    check_satterthwaite(
        "SLOPE_E",
        &fit,
        &SLOPE_E_SATT_DF,
        &SLOPE_E_SATT_P,
        &SLOPE_E_SATT_LO95,
        &SLOPE_E_SATT_HI95,
    );
    check_re_se(
        "SLOPE_E re_se",
        fit.random_effects_se_matrix(),
        &G_SLOPE_E,
        &SLOPE_E_RE_SE_G,
        SE_TOL,
    );
    check_observations(
        "SLOPE_E",
        &fit,
        &SLOPE_E_FITTED,
        &SLOPE_E_FITTED_POP,
        &SLOPE_E_PEARSON,
        SLOPE_E_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn lmm_crossed_matches_lmertest() {
    let fit = lmm()
        .build()
        .fit_crossed(
            &design(&[&X_CROSS_F]),
            &col(&Y_CROSS_F),
            &[&A_CROSS_F, &B_CROSS_F],
        )
        .unwrap();
    check_vcov("CROSS_F", &fit, &CROSS_F_VCOV, SE_TOL);
    check_satterthwaite(
        "CROSS_F",
        &fit,
        &CROSS_F_SATT_DF,
        &CROSS_F_SATT_P,
        &CROSS_F_SATT_LO95,
        &CROSS_F_SATT_HI95,
    );
    check_re_se(
        "CROSS_F re_se a",
        fit.factor_random_effects_se_matrix(0).unwrap(),
        &A_CROSS_F,
        &CROSS_F_RE_SE_A,
        SE_TOL,
    );
    check_re_se(
        "CROSS_F re_se b",
        fit.factor_random_effects_se_matrix(1).unwrap(),
        &B_CROSS_F,
        &CROSS_F_RE_SE_B,
        SE_TOL,
    );
    check_observations(
        "CROSS_F",
        &fit,
        &CROSS_F_FITTED,
        &CROSS_F_FITTED_POP,
        &CROSS_F_PEARSON,
        CROSS_F_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn lmm_ml_unbalanced_and_no_intercept_match_lmertest() {
    let fit = lmm()
        .reml(false)
        .build()
        .fit(&design(&[&X_ML_H]), &col(&Y_ML_H), &G_ML_H)
        .unwrap();
    check_vcov("ML_H", &fit, &ML_H_VCOV, SE_TOL);
    check_satterthwaite(
        "ML_H",
        &fit,
        &ML_H_SATT_DF,
        &ML_H_SATT_P,
        &ML_H_SATT_LO95,
        &ML_H_SATT_HI95,
    );

    let fit = lmm()
        .build()
        .fit(&design(&[&X_UNBAL_I]), &col(&Y_UNBAL_I), &G_UNBAL_I)
        .unwrap();
    check_vcov("UNBAL_I", &fit, &UNBAL_I_VCOV, SE_TOL);
    check_satterthwaite(
        "UNBAL_I",
        &fit,
        &UNBAL_I_SATT_DF,
        &UNBAL_I_SATT_P,
        &UNBAL_I_SATT_LO95,
        &UNBAL_I_SATT_HI95,
    );
    check_re_se(
        "UNBAL_I re_se",
        fit.random_effects_se_matrix(),
        &G_UNBAL_I,
        &UNBAL_I_RE_SE_G,
        SE_TOL,
    );

    let fit = lmm()
        .with_intercept(false)
        .build()
        .fit(&design(&[&X_NOINT_K]), &col(&Y_NOINT_K), &G_NOINT_K)
        .unwrap();
    check_vcov("NOINT_K", &fit, &NOINT_K_VCOV, SE_TOL);
    check_satterthwaite(
        "NOINT_K",
        &fit,
        &NOINT_K_SATT_DF,
        &NOINT_K_SATT_P,
        &NOINT_K_SATT_LO95,
        &NOINT_K_SATT_HI95,
    );
}

#[test]
fn lmm_with_offset_matches_lmertest() {
    let fit = lmm()
        .offset(col(&OFF_M))
        .build()
        .fit(&design(&[&X]), &col(&Y_M), &G)
        .unwrap();
    check_vcov("LMM_M", &fit, &LMM_M_VCOV, SE_TOL);
    check_satterthwaite(
        "LMM_M",
        &fit,
        &LMM_M_SATT_DF,
        &LMM_M_SATT_P,
        &LMM_M_SATT_LO95,
        &LMM_M_SATT_HI95,
    );
    check_re_se(
        "LMM_M re_se",
        fit.random_effects_se_matrix(),
        &G,
        &LMM_M_RE_SE_G,
        SE_TOL,
    );
    check_observations(
        "LMM_M",
        &fit,
        &LMM_M_FITTED,
        &LMM_M_FITTED_POP,
        &LMM_M_PEARSON,
        LMM_M_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn lmm_crossed_with_random_slope_matches_lmertest() {
    let fit = lmm()
        .random_slopes_per_factor(vec![vec![0], vec![]])
        .build()
        .fit_crossed(&design(&[&XC]), &col(&Y_N), &[&A, &B])
        .unwrap();
    check_vcov("LMM_N", &fit, &LMM_N_VCOV, SE_TOL);
    check_satterthwaite(
        "LMM_N",
        &fit,
        &LMM_N_SATT_DF,
        &LMM_N_SATT_P,
        &LMM_N_SATT_LO95,
        &LMM_N_SATT_HI95,
    );
    check_re_se(
        "LMM_N re_se a",
        fit.factor_random_effects_se_matrix(0).unwrap(),
        &A,
        &LMM_N_RE_SE_A,
        SE_TOL,
    );
    check_re_se(
        "LMM_N re_se b",
        fit.factor_random_effects_se_matrix(1).unwrap(),
        &B,
        &LMM_N_RE_SE_B,
        SE_TOL,
    );
    check_observations(
        "LMM_N",
        &fit,
        &LMM_N_FITTED,
        &LMM_N_FITTED_POP,
        &LMM_N_PEARSON,
        LMM_N_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn glmm_wald_inference_and_blup_se_match_glmer() {
    let fit = GlmmRegressor::poisson()
        .tolerance(1e-10)
        .build()
        .fit(&design(&[&X_POIS_C]), &col(&Y_POIS_C), &G_POIS_C)
        .unwrap();
    assert!(fit.satterthwaite_df().is_none());
    assert!(fit.inference(GlmmDdf::Satterthwaite, 0.95).is_err());
    let wald = fit.inference(GlmmDdf::Wald, 0.95).unwrap();
    assert!(wald.df.is_none());
    check_vcov("POIS_C", &fit, &POIS_C_VCOV, SE_TOL);
    check_re_se(
        "POIS_C re_se",
        fit.random_effects_se_matrix(),
        &G_POIS_C,
        &POIS_C_RE_SE_G,
        SE_TOL,
    );
    check_observations(
        "POIS_C",
        &fit,
        &POIS_C_FITTED,
        &POIS_C_FITTED_POP,
        &POIS_C_PEARSON,
        POIS_C_DF_RESID,
        SE_TOL,
    );

    let fit = GlmmRegressor::binomial()
        .tolerance(1e-10)
        .build()
        .fit(&design(&[&X_BIN_D]), &col(&Y_BIN_D), &G_BIN_D)
        .unwrap();
    check_vcov("BIN_D", &fit, &BIN_D_VCOV, SE_TOL);
    check_re_se(
        "BIN_D re_se",
        fit.random_effects_se_matrix(),
        &G_BIN_D,
        &BIN_D_RE_SE_G,
        SE_TOL,
    );
    check_observations(
        "BIN_D",
        &fit,
        &BIN_D_FITTED,
        &BIN_D_FITTED_POP,
        &BIN_D_PEARSON,
        BIN_D_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn glmm_other_families_match_glmer() {
    let x = design(&[&X]);
    let fit = GlmmRegressor::negative_binomial()
        .tolerance(1e-10)
        .build()
        .fit(&x, &col(&Y_NB), &G)
        .unwrap();
    check_vcov("NB_H", &fit, &NB_H_VCOV, NB_TOL);
    check_re_se(
        "NB_H re_se",
        fit.random_effects_se_matrix(),
        &G,
        &NB_H_RE_SE_G,
        NB_TOL,
    );
    check_observations(
        "NB_H",
        &fit,
        &NB_H_FITTED,
        &NB_H_FITTED_POP,
        &NB_H_PEARSON,
        NB_H_DF_RESID,
        10.0 * NB_TOL,
    );

    let fit = GlmmRegressor::gamma()
        .tolerance(1e-10)
        .build()
        .fit(&x, &col(&Y_GA), &G)
        .unwrap();
    check_vcov("GA_J", &fit, &GA_J_VCOV, SE_TOL);
    check_re_se(
        "GA_J re_se",
        fit.random_effects_se_matrix(),
        &G,
        &GA_J_RE_SE_G,
        SE_TOL,
    );
    check_observations(
        "GA_J",
        &fit,
        &GA_J_FITTED,
        &GA_J_FITTED_POP,
        &GA_J_PEARSON,
        GA_J_DF_RESID,
        10.0 * SE_TOL,
    );

    let expo: Vec<f64> = EXPO.iter().map(|e| e.ln()).collect();
    let fit = GlmmRegressor::poisson()
        .tolerance(1e-10)
        .offset(col(&expo))
        .build()
        .fit(&x, &col(&Y_PO_OFF), &G)
        .unwrap();
    check_vcov("PO_L", &fit, &PO_L_VCOV, SE_TOL);
    check_re_se(
        "PO_L re_se",
        fit.random_effects_se_matrix(),
        &G,
        &PO_L_RE_SE_G,
        SE_TOL,
    );
    check_observations(
        "PO_L",
        &fit,
        &PO_L_FITTED,
        &PO_L_FITTED_POP,
        &PO_L_PEARSON,
        PO_L_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn glmm_crossed_with_random_slope_matches_glmer() {
    let fit = GlmmRegressor::poisson()
        .tolerance(1e-10)
        .random_slopes_per_factor(vec![vec![0], vec![]])
        .build()
        .fit_crossed(&design(&[&XC]), &col(&Y_O), &[&A, &B])
        .unwrap();
    check_vcov("PO_O", &fit, &PO_O_VCOV, SE_TOL);
    check_re_se(
        "PO_O re_se a",
        fit.factor_random_effects_se_matrix(0).unwrap(),
        &A,
        &PO_O_RE_SE_A,
        SE_TOL,
    );
    check_re_se(
        "PO_O re_se b",
        fit.factor_random_effects_se_matrix(1).unwrap(),
        &B,
        &PO_O_RE_SE_B,
        SE_TOL,
    );
    check_observations(
        "PO_O",
        &fit,
        &PO_O_FITTED,
        &PO_O_FITTED_POP,
        &PO_O_PEARSON,
        PO_O_DF_RESID,
        SE_TOL,
    );
}

#[test]
fn wald_inference_matches_the_existing_accessors() {
    let fit = GlmmRegressor::gaussian()
        .build()
        .fit(&design(&[&X_LMM_A]), &col(&Y_LMM_A), &G_LMM_A)
        .unwrap();
    // Without satterthwaite(true) there are no df, and Satterthwaite is an error.
    assert!(fit.satterthwaite_df().is_none());
    assert!(fit.inference(GlmmDdf::Satterthwaite, 0.95).is_err());
    let w = fit.inference(GlmmDdf::Wald, 0.9).unwrap();
    let (lo, hi) = fit.conf_int(0.9);
    assert_eq!(w.std_errors, fit.std_errors());
    assert_eq!(w.statistics, fit.z_values());
    assert_eq!(w.p_values, fit.p_values());
    assert_eq!(w.conf_low, lo);
    assert_eq!(w.conf_high, hi);
    // An invalid level gives NaN bounds, not an error.
    let fit = lmm()
        .build()
        .fit(&design(&[&X_LMM_A]), &col(&Y_LMM_A), &G_LMM_A)
        .unwrap();
    let bad = fit.inference(GlmmDdf::Satterthwaite, 1.5).unwrap();
    assert!(bad.conf_low.iter().all(|v| v.is_nan()));
    assert!(bad.p_values.iter().all(|v| v.is_finite()));
}
