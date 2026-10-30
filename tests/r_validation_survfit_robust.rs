#![allow(clippy::excessive_precision)]

//! #113: the robust (infinitesimal-jackknife) variance of `survfit`, the
//! default for non-integer case weights, with clusters, the
//! Fleming–Harrington hazard and left truncation. Reference values come
//! from `tests/r_scripts/generate_survfit_robust_validation.R`.
//!
//! With the robust variance R's `std.err` is the SE of `S` itself
//! (`logse = FALSE`); ours stays on the `−log S` scale, so `std_err · S` is
//! compared. Greenwood (`robust = FALSE`) is unchanged.
//!
//! For the Fleming–Harrington hazard (`ctype = 2`) survival's `survfitKM`
//! marks the robust `std.err` as log-scale (`logse = TRUE`) although its C
//! code returns the SE of `S`, so R's intervals there are
//! `exp(log S ± z·SE(S))`; ours use `SE(S)/S` as for the Nelson–Aalen
//! hazard, and are checked against that formula instead.

use anofox_regression::survival::{survfit, HazardType, SurvCurve, SurvInput, SurvfitOptions};

mod fixture {
    include!("fixtures/survfit_robust_validation.rs");
}
use fixture::*;

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        if !v.is_finite() {
            assert!(
                u == v || (u.is_nan() && v.is_nan()),
                "{name}[{k}]: {u} vs {v}"
            );
            continue;
        }
        assert!(
            (u - v).abs() <= tol * (1.0 + v.abs()),
            "{name}[{k}]: {u} vs {v}"
        );
    }
}

fn event() -> Vec<bool> {
    EVENT.iter().map(|&e| e == 1.0).collect()
}

#[allow(clippy::too_many_arguments)]
fn check(
    tag: &str,
    c: &SurvCurve,
    robust: bool,
    surv: &[f64],
    se: &[f64],
    chaz: &[f64],
    sechaz: &[f64],
    lower: &[f64],
    upper: &[f64],
) {
    all_close(&format!("{tag} surv"), &c.surv, surv, 1e-12);
    let got: Vec<f64> = if robust {
        c.std_err.iter().zip(&c.surv).map(|(e, s)| e * s).collect()
    } else {
        c.std_err.clone()
    };
    all_close(&format!("{tag} std.err"), &got, se, 1e-10);
    all_close(&format!("{tag} cumhaz"), &c.cumhaz, chaz, 1e-12);
    all_close(&format!("{tag} std.chaz"), &c.std_chaz, sechaz, 1e-10);
    // Where S = 0 the robust SE of S is zero; R's limits are then 0 when
    // its sum comes out exactly zero (no clusters) but NaN when a rounding
    // residue remains (clusters), so the limits are compared where S > 0.
    let pos: Vec<usize> = (0..surv.len())
        .filter(|&k| surv[k] > 0.0 || !robust)
        .collect();
    let pick = |v: &[f64]| pos.iter().map(|&k| v[k]).collect::<Vec<f64>>();
    all_close(
        &format!("{tag} lower"),
        &pick(&c.lower),
        &pick(lower),
        1e-10,
    );
    all_close(
        &format!("{tag} upper"),
        &pick(&c.upper),
        &pick(upper),
        1e-10,
    );
}

fn curve(input: &SurvInput, options: SurvfitOptions) -> SurvCurve {
    survfit(input, None, &options).unwrap().remove(0)
}

#[test]
fn robust_variance_matches_survfit() {
    let ev = event();
    let id: Vec<usize> = ID.iter().map(|&v| v as usize).collect();
    let d = SurvfitOptions::default();
    let base = SurvInput::new(&TIME, &ev).weights(&W);

    check(
        "W",
        &curve(&base, d),
        true,
        &W_SURV,
        &W_SE,
        &W_CHAZ,
        &W_SECHAZ,
        &W_LOWER,
        &W_UPPER,
    );
    check(
        "ID",
        &curve(&base.cluster(&id), d),
        true,
        &ID_SURV,
        &ID_SE,
        &ID_CHAZ,
        &ID_SECHAZ,
        &ID_LOWER,
        &ID_UPPER,
    );
    let fh = SurvfitOptions {
        hazard: HazardType::FlemingHarrington,
        ..d
    };
    let c = curve(&base, fh);
    let lower: Vec<f64> = c
        .surv
        .iter()
        .zip(&c.std_err)
        .map(|(s, e)| {
            if *e == 0.0 {
                *s
            } else {
                (s.ln() - 1.959963984540054 * e).exp()
            }
        })
        .collect();
    let upper: Vec<f64> = c
        .surv
        .iter()
        .zip(&c.std_err)
        .map(|(s, e)| {
            if *e == 0.0 {
                *s
            } else {
                (s.ln() + 1.959963984540054 * e).exp().min(1.0)
            }
        })
        .collect();
    check(
        "FH", &c, true, &FH_SURV, &FH_SE, &FH_CHAZ, &FH_SECHAZ, &lower, &upper,
    );
    // R's FH limits: SE(S) used as a log-scale SE
    let z = 1.959963984540054;
    let r_limit = |sign: f64, k: usize| {
        if FH_SE[k] == 0.0 {
            FH_SURV[k]
        } else {
            (FH_SURV[k].ln() + sign * z * FH_SE[k]).exp().min(1.0)
        }
    };
    let pos: Vec<usize> = (0..FH_SURV.len()).filter(|&k| FH_SURV[k] > 0.0).collect();
    let r_lower: Vec<f64> = pos.iter().map(|&k| r_limit(-1.0, k)).collect();
    let r_upper: Vec<f64> = pos.iter().map(|&k| r_limit(1.0, k)).collect();
    let pick = |v: &[f64]| pos.iter().map(|&k| v[k]).collect::<Vec<f64>>();
    all_close("R FH lower", &r_lower, &pick(&FH_LOWER), 1e-10);
    all_close("R FH upper", &r_upper, &pick(&FH_UPPER), 1e-10);
    check(
        "LT",
        &curve(&base.entry(&ENTRY), d),
        true,
        &LT_SURV,
        &LT_SE,
        &LT_CHAZ,
        &LT_SECHAZ,
        &LT_LOWER,
        &LT_UPPER,
    );
    let forced = SurvfitOptions {
        robust: Some(true),
        ..d
    };
    check(
        "UNW",
        &curve(&SurvInput::new(&TIME, &ev), forced),
        true,
        &UNW_SURV,
        &UNW_SE,
        &UNW_CHAZ,
        &UNW_SECHAZ,
        &UNW_LOWER,
        &UNW_UPPER,
    );
    let greenwood = SurvfitOptions {
        robust: Some(false),
        ..d
    };
    check(
        "GW",
        &curve(&base, greenwood),
        false,
        &GW_SURV,
        &GW_SE,
        &GW_CHAZ,
        &GW_SECHAZ,
        &GW_LOWER,
        &GW_UPPER,
    );
}

#[test]
fn cluster_length_is_checked() {
    let ev = event();
    let short = vec![0usize; 3];
    let input = SurvInput::new(&TIME, &ev).cluster(&short);
    assert!(survfit(&input, None, &SurvfitOptions::default()).is_err());
}
