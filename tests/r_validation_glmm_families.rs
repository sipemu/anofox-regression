//! GLMM validation against R `lme4` for issue #29: the negative-binomial,
//! Gamma and Tweedie families, offsets, and random slopes combined with
//! several crossed grouping factors.
//!
//! Reference values come from `tests/r_scripts/generate_glmm_families_validation.R`
//! (embedded in `tests/fixtures/glmm_families_validation.rs`). All GLMMs are
//! `nAGQ = 0` fits:
//!
//! * negative binomial — `glmer.nb` (θ estimated) and
//!   `glmer(family = MASS::negative.binomial(2))` (θ fixed);
//! * Gamma — `glmer(family = Gamma(link = "log"))`;
//! * Tweedie (p = 1.5) — `glmer` with `statmod::tweedie(1.5, link.power = 0)`
//!   whose `aic` is the Tweedie series log-density (`tweedie::dtweedie`) at
//!   dispersion `deviance / n` (+2), the same criterion lme4 uses for Gamma.
//!   (`glmmTMB` is a different estimator — joint ML dispersion — and is not
//!   used as the reference.)
//!
//! Regenerate with:
//!
//! ```bash
//! Rscript tests/r_scripts/generate_glmm_families_validation.R \
//!     > tests/fixtures/glmm_families_validation.rs
//! ```

#![allow(dead_code)]
#![allow(clippy::approx_constant)]

use anofox_regression::solvers::{FittedGlmm, GlmmRegressor};
use faer::{Col, Mat};

include!("fixtures/glmm_families_validation.rs");

fn close(label: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / (1.0 + want.abs());
    assert!(
        err < tol,
        "{label}: got {got:.8}, R {want:.8} (scaled err {err:.2e} >= {tol:.0e})"
    );
}

fn close_vec(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(&format!("{label}[{i}]"), g, w, tol);
    }
}

/// Reorder R's per-level values (sorted level ids) into the solver's level
/// order, which is the order of first appearance in `ids`.
fn in_solver_order(ids: &[usize], r_values: &[f64]) -> Vec<f64> {
    let mut seen = Vec::new();
    for &id in ids {
        if !seen.contains(&id) {
            seen.push(id);
        }
    }
    seen.iter().map(|&id| r_values[id]).collect()
}

fn x_single() -> Mat<f64> {
    Mat::from_fn(X.len(), 1, |i, _| X[i])
}

fn x_crossed() -> Mat<f64> {
    Mat::from_fn(XC.len(), 1, |i, _| XC[i])
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

#[allow(clippy::too_many_arguments)]
fn check_single(
    tag: &str,
    m: &FittedGlmm,
    fixef: &[f64],
    se: &[f64],
    loglik: f64,
    sigma: f64,
    sd_re: f64,
    blup: &[f64],
    tol: f64,
) {
    assert!(m.converged(), "{tag}: not converged");
    close_vec(&format!("{tag} fixef"), m.fixed_effects(), fixef, tol);
    close_vec(&format!("{tag} se"), m.std_errors(), se, tol);
    close(&format!("{tag} loglik"), m.log_likelihood(), loglik, tol);
    close(&format!("{tag} sigma"), m.sigma(), sigma, tol);
    close(&format!("{tag} sd_re"), m.sd_random(), sd_re, tol);
    close_vec(&format!("{tag} blup"), m.random_effects(), blup, tol);
}

// PIRLS / θ-search resolution; glmer.nb's own θ search (optimize) is the
// coarsest step, hence the looser NB-estimated tolerance.
const TOL: f64 = 2e-4;
const TOL_NB_EST: f64 = 2e-3;

#[test]
fn negative_binomial_estimated_theta_matches_glmer_nb() {
    let m = GlmmRegressor::negative_binomial()
        .build()
        .fit(&x_single(), &col(&Y_NB), &G)
        .expect("NB GLMM fit");
    close(
        "NB theta",
        m.nb_theta().expect("theta"),
        NB_H_THETA,
        TOL_NB_EST,
    );
    check_single(
        "NB_H",
        &m,
        &NB_H_FIXEF,
        &NB_H_SE,
        NB_H_LOGLIK,
        NB_H_SIGMA,
        NB_H_SD_RE,
        &NB_H_BLUP,
        TOL_NB_EST,
    );
}

#[test]
fn negative_binomial_fixed_theta_matches_glmer() {
    let m = GlmmRegressor::negative_binomial()
        .nb_theta(2.0)
        .build()
        .fit(&x_single(), &col(&Y_NB), &G)
        .expect("NB GLMM fit");
    assert_eq!(m.nb_theta(), Some(2.0));
    check_single(
        "NB_I",
        &m,
        &NB_I_FIXEF,
        &NB_I_SE,
        NB_I_LOGLIK,
        NB_I_SIGMA,
        NB_I_SD_RE,
        &NB_I_BLUP,
        TOL,
    );
}

#[test]
fn gamma_matches_glmer() {
    let m = GlmmRegressor::gamma()
        .build()
        .fit(&x_single(), &col(&Y_GA), &G)
        .expect("Gamma GLMM fit");
    assert_eq!(m.nb_theta(), None);
    check_single(
        "GA_J",
        &m,
        &GA_J_FIXEF,
        &GA_J_SE,
        GA_J_LOGLIK,
        GA_J_SIGMA,
        GA_J_SD_RE,
        &GA_J_BLUP,
        TOL,
    );
}

#[test]
fn tweedie_matches_glmer_with_tweedie_family() {
    let m = GlmmRegressor::tweedie(1.5)
        .build()
        .fit(&x_single(), &col(&Y_TW), &G)
        .expect("Tweedie GLMM fit");
    check_single(
        "TW_K",
        &m,
        &TW_K_FIXEF,
        &TW_K_SE,
        TW_K_LOGLIK,
        TW_K_SIGMA,
        TW_K_SD_RE,
        &TW_K_BLUP,
        TOL,
    );
}

#[test]
fn poisson_offset_matches_glmer() {
    let off = Col::from_fn(EXPO.len(), |i| EXPO[i].ln());
    let m = GlmmRegressor::poisson()
        .offset(off.clone())
        .build()
        .fit(&x_single(), &col(&Y_PO_OFF), &G)
        .expect("Poisson GLMM with offset");
    check_single(
        "PO_L",
        &m,
        &PO_L_FIXEF,
        &PO_L_SE,
        PO_L_LOGLIK,
        PO_L_SIGMA,
        PO_L_SD_RE,
        &PO_L_BLUP,
        TOL,
    );
    // predict_fixed_with_offset adds the offset to x'β.
    let eta = m.predict_fixed(&x_single());
    let eta_off = m.predict_fixed_with_offset(&x_single(), &off);
    for i in 0..off.nrows() {
        assert!((eta_off[i] - eta[i] - off[i]).abs() < 1e-12);
    }
}

#[test]
fn lmm_offset_matches_lmer() {
    let m = GlmmRegressor::gaussian()
        .offset(col(&OFF_M))
        .build()
        .fit(&x_single(), &col(&Y_M), &G)
        .expect("LMM with offset");
    check_single(
        "LMM_M",
        &m,
        &LMM_M_FIXEF,
        &LMM_M_SE,
        LMM_M_LOGLIK,
        LMM_M_SIGMA,
        LMM_M_SD_RE,
        &LMM_M_BLUP,
        TOL,
    );
}

fn crossed_fit(builder: anofox_regression::solvers::GlmmRegressorBuilder, y: &[f64]) -> FittedGlmm {
    builder
        .build()
        .fit_crossed(&x_crossed(), &col(y), &[&A, &B])
        .expect("crossed fit")
}

#[allow(clippy::too_many_arguments)]
fn check_crossed_slopes(
    tag: &str,
    m: &FittedGlmm,
    fixef: &[f64],
    se: &[f64],
    loglik: f64,
    sigma: f64,
    sd_a: [f64; 2],
    corr_a: f64,
    sd_b: f64,
    blup_a0: &[f64],
    blup_a1: &[f64],
    blup_b: &[f64],
    tol: f64,
) {
    assert!(m.converged(), "{tag}: not converged");
    close_vec(&format!("{tag} fixef"), m.fixed_effects(), fixef, tol);
    close_vec(&format!("{tag} se"), m.std_errors(), se, tol);
    close(&format!("{tag} loglik"), m.log_likelihood(), loglik, tol);
    close(&format!("{tag} sigma"), m.sigma(), sigma, tol);
    assert_eq!(m.n_factors(), 2);
    let sd = m.factor_random_sd(0).expect("factor a");
    close_vec(&format!("{tag} sd_a"), &sd, &sd_a, tol);
    let cov = m.factor_random_cov(0).expect("factor a cov");
    close(
        &format!("{tag} corr_a"),
        cov[0][1] / (sd[0] * sd[1]),
        corr_a,
        tol,
    );
    close(&format!("{tag} sd_b"), m.factors()[1].sd, sd_b, tol);
    let re_a = m.factor_random_effects_matrix(0).expect("factor a blups");
    let a0: Vec<f64> = re_a.iter().map(|r| r[0]).collect();
    let a1: Vec<f64> = re_a.iter().map(|r| r[1]).collect();
    close_vec(
        &format!("{tag} blup_a0"),
        &a0,
        &in_solver_order(&A, blup_a0),
        tol,
    );
    close_vec(
        &format!("{tag} blup_a1"),
        &a1,
        &in_solver_order(&A, blup_a1),
        tol,
    );
    close_vec(
        &format!("{tag} blup_b"),
        &m.factors()[1].blups,
        &in_solver_order(&B, blup_b),
        tol,
    );
    // Single-factor accessors mirror factor 0 (intercept + slope).
    assert_eq!(m.n_random_effects(), 2);
    assert_eq!(m.factor_random_cov(1).map(<[Vec<f64>]>::len), Some(1));
}

#[test]
fn lmm_crossed_with_random_slope_matches_lmer() {
    let m = crossed_fit(
        GlmmRegressor::gaussian().random_slopes_per_factor(vec![vec![0], vec![]]),
        &Y_N,
    );
    check_crossed_slopes(
        "LMM_N",
        &m,
        &LMM_N_FIXEF,
        &LMM_N_SE,
        LMM_N_LOGLIK,
        LMM_N_SIGMA,
        [LMM_N_SD_A0, LMM_N_SD_A1],
        LMM_N_CORR_A,
        LMM_N_SD_B,
        &LMM_N_BLUP_A0,
        &LMM_N_BLUP_A1,
        &LMM_N_BLUP_B,
        2e-3,
    );
}

#[test]
fn poisson_crossed_with_random_slope_matches_glmer() {
    let m = crossed_fit(
        GlmmRegressor::poisson().random_slopes_per_factor(vec![vec![0], vec![]]),
        &Y_O,
    );
    check_crossed_slopes(
        "PO_O",
        &m,
        &PO_O_FIXEF,
        &PO_O_SE,
        PO_O_LOGLIK,
        PO_O_SIGMA,
        [PO_O_SD_A0, PO_O_SD_A1],
        PO_O_CORR_A,
        PO_O_SD_B,
        &PO_O_BLUP_A0,
        &PO_O_BLUP_A1,
        &PO_O_BLUP_B,
        2e-3,
    );
}

#[test]
fn gamma_crossed_matches_glmer() {
    let m = crossed_fit(GlmmRegressor::gamma(), &Y_P);
    assert!(m.converged());
    close_vec("GA_P fixef", m.fixed_effects(), &GA_P_FIXEF, TOL);
    close_vec("GA_P se", m.std_errors(), &GA_P_SE, TOL);
    close("GA_P loglik", m.log_likelihood(), GA_P_LOGLIK, TOL);
    close("GA_P sigma", m.sigma(), GA_P_SIGMA, TOL);
    close("GA_P sd_a", m.factors()[0].sd, GA_P_SD_A0, TOL);
    close("GA_P sd_b", m.factors()[1].sd, GA_P_SD_B, TOL);
    close_vec(
        "GA_P blup_a",
        &m.factors()[0].blups,
        &in_solver_order(&A, &GA_P_BLUP_A0),
        TOL,
    );
    close_vec(
        "GA_P blup_b",
        &m.factors()[1].blups,
        &in_solver_order(&B, &GA_P_BLUP_B),
        TOL,
    );
}

#[test]
fn input_validation_for_new_options() {
    let x = x_single();
    // Gamma needs y > 0.
    let mut y = col(&Y_GA);
    y[0] = 0.0;
    assert!(GlmmRegressor::gamma().build().fit(&x, &y, &G).is_err());
    // Tweedie power must be in (1, 2).
    assert!(GlmmRegressor::tweedie(2.5)
        .build()
        .fit(&x, &col(&Y_TW), &G)
        .is_err());
    // NB θ must be positive.
    assert!(GlmmRegressor::negative_binomial()
        .nb_theta(-1.0)
        .build()
        .fit(&x, &col(&Y_NB), &G)
        .is_err());
    // Offset length must match.
    assert!(GlmmRegressor::poisson()
        .offset(Col::zeros(3))
        .build()
        .fit(&x, &col(&Y_NB), &G)
        .is_err());
    // Per-factor slopes must have one entry per factor.
    let xc = x_crossed();
    assert!(GlmmRegressor::gaussian()
        .random_slopes_per_factor(vec![vec![0]])
        .build()
        .fit_crossed(&xc, &col(&Y_N), &[&A, &B])
        .is_err());
    // The single-factor random_slopes still refuses multiple factors.
    assert!(GlmmRegressor::gaussian()
        .random_slopes(vec![0])
        .build()
        .fit_crossed(&xc, &col(&Y_N), &[&A, &B])
        .is_err());
}

#[test]
fn per_factor_slopes_with_one_factor_equals_random_slopes() {
    let xc = x_crossed();
    let a = GlmmRegressor::gaussian()
        .random_slopes(vec![0])
        .build()
        .fit(&xc, &col(&Y_N), &A)
        .unwrap();
    let b = GlmmRegressor::gaussian()
        .random_slopes_per_factor(vec![vec![0]])
        .build()
        .fit_crossed(&xc, &col(&Y_N), &[&A])
        .unwrap();
    assert_eq!(a.fixed_effects(), b.fixed_effects());
    assert_eq!(a.random_sd(), b.random_sd());
}
