//! Issue #91: `solvers::mediation` against R's `mediation::mediate`, `lavaan` and
//! `bda::mediation.test`. Reference values from
//! `tests/r_scripts/generate_mediation_validation.R`.
//!
//! Point estimates must match exactly. The bootstrap and Monte Carlo draws come
//! from a different generator than R's, so the intervals are compared two ways:
//! exactly, by summarising R's own draws with [`MediationEffect::from_draws`],
//! and within simulation error, from our own draws.

#![allow(clippy::excessive_precision, dead_code)]

use anofox_regression::solvers::mediation::{
    mediate, BootCi, MediationEffect, MediationMethod, MediationOptions,
};

include!("fixtures/mediation_validation.rs");

const TOL: f64 = 1e-10;

fn close(name: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{name}: got {got:e}, want {want:e}"
    );
}

fn covs() -> [&'static [f64]; 2] {
    [&C1, &C2]
}

fn run(
    method: MediationMethod,
    boot_ci: BootCi,
    seed: u64,
) -> anofox_regression::solvers::mediation::MediationResult {
    let o = MediationOptions {
        method,
        boot_ci,
        n_sims: 2000,
        seed,
        ..Default::default()
    };
    mediate(&Y, &M, &X, &covs(), &o).unwrap()
}

#[test]
fn point_estimates_match_mediate_and_lavaan() {
    let r = run(MediationMethod::Bootstrap, BootCi::Percentile, 42);
    close("a", r.a, A, TOL);
    close("b", r.b, B, TOL);
    close("c'", r.c_prime, C_PRIME, TOL);
    close("acme", r.acme.estimate, PERC_ACME, TOL);
    close("ade", r.ade.estimate, PERC_ADE, TOL);
    close("total", r.total.estimate, PERC_TOTAL, TOL);
    close("prop", r.prop_mediated.estimate, PERC_PROP, TOL);
    close("lavaan a*b", r.acme.estimate, LAVAAN_AB, 1e-8);
    close("lavaan total", r.total.estimate, LAVAAN_TOTAL, 1e-8);
    close("total = c of y ~ x + c", r.total.estimate, C_TOTAL, TOL);
    close("r2 mediator", r.r_squared_mediator, R2_MEDIATOR, TOL);
    close("r2 outcome", r.r_squared_outcome, R2_OUTCOME, TOL);
    assert_eq!(r.n_obs, N);
    assert_eq!(r.n_sims, 2000);
}

fn check_summary(name: &str, est: f64, draws: &[f64], ci: BootCi, want_ci: [f64; 2], want_p: f64) {
    let e = MediationEffect::from_draws(est, draws, 0.95, ci);
    close(&format!("{name} conf_low"), e.conf_low, want_ci[0], TOL);
    close(&format!("{name} conf_high"), e.conf_high, want_ci[1], TOL);
    close(&format!("{name} p"), e.p_value, want_p, TOL);
}

#[test]
fn summaries_of_r_draws_match_mediate_exactly() {
    for (ci, acme, ade, total, prop, p) in [
        (
            BootCi::Percentile,
            PERC_ACME_CI,
            PERC_ADE_CI,
            PERC_TOTAL_CI,
            PERC_PROP_CI,
            [PERC_ACME_P, PERC_ADE_P, PERC_TOTAL_P, PERC_PROP_P],
        ),
        (
            BootCi::Bca,
            BCA_ACME_CI,
            BCA_ADE_CI,
            BCA_TOTAL_CI,
            BCA_PROP_CI,
            [BCA_ACME_P, BCA_ADE_P, BCA_TOTAL_P, BCA_PROP_P],
        ),
    ] {
        check_summary("acme", PERC_ACME, &PERC_ACME_DRAWS, ci, acme, p[0]);
        check_summary("ade", PERC_ADE, &PERC_ADE_DRAWS, ci, ade, p[1]);
        check_summary("total", PERC_TOTAL, &PERC_TOTAL_DRAWS, ci, total, p[2]);
        check_summary("prop", PERC_PROP, &PERC_PROP_DRAWS, ci, prop, p[3]);
    }
    let e = MediationEffect::from_draws(PERC_ACME, &PERC_ACME_DRAWS, 0.95, BootCi::Percentile);
    close("acme bootstrap sd", e.std_error, PERC_ACME_SD, TOL);
}

/// Our own draws: within simulation error of R's (2000 resamples each). The
/// bound is a fraction of the bootstrap SD, generous enough for any seed.
fn within_mc(name: &str, got: MediationEffect, want_ci: [f64; 2], sd: f64) {
    assert!(
        (got.conf_low - want_ci[0]).abs() < 0.25 * sd,
        "{name} conf_low: got {}, R {}",
        got.conf_low,
        want_ci[0]
    );
    assert!(
        (got.conf_high - want_ci[1]).abs() < 0.25 * sd,
        "{name} conf_high: got {}, R {}",
        got.conf_high,
        want_ci[1]
    );
}

#[test]
fn bootstrap_intervals_within_simulation_error() {
    let r = run(MediationMethod::Bootstrap, BootCi::Percentile, 42);
    within_mc("acme", r.acme, PERC_ACME_CI, r.acme.std_error);
    within_mc("ade", r.ade, PERC_ADE_CI, r.ade.std_error);
    within_mc("total", r.total, PERC_TOTAL_CI, r.total.std_error);
    within_mc(
        "prop",
        r.prop_mediated,
        PERC_PROP_CI,
        r.prop_mediated.std_error,
    );
    // lavaan's bootstrap SE of a*b and R's draws
    assert!((r.acme.std_error - LAVAAN_AB_SE).abs() < 0.1 * LAVAAN_AB_SE);
    assert!((r.acme.std_error - PERC_ACME_SD).abs() < 0.1 * PERC_ACME_SD);
    assert!(r.acme.p_value < 0.01 && r.ade.p_value > 0.05);

    let bca = run(MediationMethod::Bootstrap, BootCi::Bca, 42);
    within_mc("bca acme", bca.acme, BCA_ACME_CI, r.acme.std_error);
    // The BCa acceleration of the heavy-tailed proportion draws is too unstable
    // to compare across generators; mediate's formula is checked exactly above.
    within_mc("bca ade", bca.ade, BCA_ADE_CI, r.ade.std_error);
    assert_eq!(bca.acme.estimate, r.acme.estimate);
}

#[test]
fn continuous_treatment_contrast() {
    let o = MediationOptions {
        n_sims: 2000,
        treat_value: 2.0,
        control_value: 0.5,
        ..Default::default()
    };
    let r = mediate(&YC, &MC, &XC, &covs(), &o).unwrap();
    close("acme", r.acme.estimate, CONT_ACME, TOL);
    close("ade", r.ade.estimate, CONT_ADE, TOL);
    close("total", r.total.estimate, CONT_TOTAL, TOL);
    close("prop", r.prop_mediated.estimate, CONT_PROP, TOL);
    within_mc("acme", r.acme, CONT_ACME_CI, r.acme.std_error);
    within_mc("ade", r.ade, CONT_ADE_CI, r.ade.std_error);
}

#[test]
fn quasi_bayesian_within_simulation_error() {
    let r = run(MediationMethod::QuasiBayesian, BootCi::Percentile, 42);
    // The point estimates are means of the draws, as in mediate.
    let sd = r.acme.std_error;
    assert!((r.acme.estimate - QB_ACME).abs() < 0.1 * sd);
    assert!((r.ade.estimate - QB_ADE).abs() < 0.1 * r.ade.std_error);
    assert!((r.total.estimate - QB_TOTAL).abs() < 0.1 * r.total.std_error);
    assert!((r.prop_mediated.estimate - QB_PROP).abs() < 0.1 * r.prop_mediated.std_error);
    within_mc("acme", r.acme, QB_ACME_CI, sd);
    within_mc("ade", r.ade, QB_ADE_CI, r.ade.std_error);
    within_mc("total", r.total, QB_TOTAL_CI, r.total.std_error);
    // Delta-method SE of a*b, the QB draws' asymptotic SD
    let sobel = (A * A * SE_B * SE_B + B * B * SE_A * SE_A).sqrt();
    assert!((sd - sobel).abs() < 0.1 * sobel);
}

#[test]
fn sobel_matches_bda_and_ols_inference() {
    let o = MediationOptions {
        method: MediationMethod::Sobel,
        ..Default::default()
    };
    // bda::mediation.test has no covariates
    let r = mediate(&Y, &M, &X, &[] as &[Vec<f64>], &o).unwrap();
    close("sobel z", r.acme.statistic, SOBEL_NOCOV_Z, 1e-8);
    close("sobel p", r.acme.p_value, SOBEL_NOCOV_P, 1e-8);
    assert_eq!(r.n_sims, 0);

    let r = mediate(&Y, &M, &X, &covs(), &o).unwrap();
    let se_ab = (A * A * SE_B * SE_B + B * B * SE_A * SE_A).sqrt();
    close("acme se", r.acme.std_error, se_ab, TOL);
    close("ade se", r.ade.std_error, SE_C_PRIME, TOL);
    close("ade t", r.ade.statistic, C_PRIME / SE_C_PRIME, TOL);
    close("total se", r.total.std_error, SE_C_TOTAL, TOL);
    // Delta method for ab / (ab + c')
    let t = A * B + C_PRIME;
    let (ga, gb, gc) = (
        B * C_PRIME / (t * t),
        A * C_PRIME / (t * t),
        -A * B / (t * t),
    );
    let var = ga * ga * SE_A * SE_A
        + gb * gb * SE_B * SE_B
        + gc * gc * SE_C_PRIME * SE_C_PRIME
        + 2.0 * gb * gc * COV_B_C_PRIME;
    close("prop se", r.prop_mediated.std_error, var.sqrt(), TOL);
}
