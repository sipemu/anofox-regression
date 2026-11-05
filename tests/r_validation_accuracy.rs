#![allow(clippy::excessive_precision, dead_code)]

//! #102: predictive accuracy indexes against Hmisc / rms / survival.
//!
//! Reference values come from `tests/r_scripts/generate_accuracy_validation.R`.

use anofox_regression::diagnostics::accuracy::{
    brier_score, concordance, effective_sample_size, gini_md, lrm_stats, r2_measures, rcorr_cens,
    somers2, val_prob,
};

include!("fixtures/accuracy_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

#[test]
fn concordance_counts_match_concordancefit() {
    for (w, want) in [(None, &CONC), (Some(&W[..]), &CONC_W)] {
        let c = concordance(&XC, &YC, w).unwrap();
        let got = [
            c.concordant,
            c.discordant,
            c.tied_x,
            c.tied_y,
            c.tied_xy,
            c.c(),
        ];
        for (k, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            close(
                &format!("conc[{k}] weighted={}", w.is_some()),
                *a,
                *b,
                1e-12,
            );
        }
    }
}

#[test]
fn somers2_matches_hmisc() {
    let p: Vec<f64> = LP.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let s = somers2(&p, &Y, None).unwrap();
    close("C", s.c, SOMERS2[0], 1e-12);
    close("Dxy", s.dxy, SOMERS2[1], 1e-12);
    let s = somers2(&p, &Y, Some(&W)).unwrap();
    close("C w", s.c, SOMERS2_W[0], 1e-12);
    close("Dxy w", s.dxy, SOMERS2_W[1], 1e-12);
}

#[test]
fn lrm_stats_match_rms() {
    let s = lrm_stats(&Y, &LP, None, 2.0).unwrap();
    close("LR", s.model_lr, LRM_MODEL_L_R_, 1e-9);
    // R reports `1 - pchisq(LR, df)`, which at p ≈ 6e-15 is rounded to a
    // multiple of machine epsilon; the upper tail used here is accurate.
    assert!(
        (s.p_value - LRM_P).abs() < 1e-15,
        "{} vs {LRM_P}",
        s.p_value
    );
    close("C", s.c, LRM_C, 1e-12);
    close("Dxy", s.dxy, LRM_DXY, 1e-12);
    close("Gamma", s.gamma, LRM_GAMMA, 1e-12);
    close("Tau-a", s.tau_a, LRM_TAU_A, 1e-12);
    close("R2", s.r2, LRM_R2, 1e-9);
    let m = s.r2_measures;
    let got = [m.r2, m.r2_adj, m.r2_ess.unwrap(), m.r2_adj_ess.unwrap()];
    for k in 0..4 {
        close(&format!("R2m[{k}]"), got[k], LRM_R2M[k], 1e-9);
    }
    close("Brier", s.brier, LRM_BRIER, 1e-12);
    close("g", s.g, LRM_G, 1e-12);
    close("gr", s.gr, LRM_GR, 1e-12);
    close("gp", s.gp, LRM_GP, 1e-12);
    close("ess", s.effective_sample_size, LRM_ESS, 1e-12);
    close("GiniMd", gini_md(&LP), GINI_LP, 1e-12);
}

#[test]
fn weighted_lrm_stats_match_rms() {
    let s = lrm_stats(&Y, &LPW, Some(&W), 2.0).unwrap();
    close("LR", s.model_lr, LRMW_MODEL_L_R_, 1e-9);
    close("C", s.c, LRMW_C, 1e-12);
    close("Dxy", s.dxy, LRMW_DXY, 1e-12);
    close("Gamma", s.gamma, LRMW_GAMMA, 1e-12);
    close("Tau-a", s.tau_a, LRMW_TAU_A, 1e-12);
    close("R2", s.r2, LRMW_R2, 1e-9);
    close("Brier", s.brier, LRMW_BRIER, 1e-12);
    close("ess", s.effective_sample_size, LRMW_ESS, 1e-12);
    let m = s.r2_measures;
    let got = [m.r2, m.r2_adj, m.r2_ess.unwrap(), m.r2_adj_ess.unwrap()];
    for k in 0..4 {
        close(&format!("R2m[{k}]"), got[k], LRMW_R2M[k], 1e-9);
    }
    let p: Vec<f64> = LPW.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    close(
        "brier fn",
        brier_score(&Y, &p, Some(&W)).unwrap(),
        LRMW_BRIER,
        1e-12,
    );
}

#[test]
fn rcorr_cens_matches_hmisc() {
    let ev: Vec<bool> = EV.iter().map(|&e| e == 1.0).collect();
    for (event, want) in [(ev, &RCORR), (vec![true; TT.len()], &RCORR_UNC)] {
        let r = rcorr_cens(&PX, &TT, &event).unwrap();
        let got = [
            r.c,
            r.dxy,
            r.sd,
            r.n as f64,
            r.missing as f64,
            r.uncensored as f64,
            r.relevant_pairs,
            r.concordant,
            r.uncertain,
        ];
        for (k, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            close(&format!("rcorr.cens[{k}]"), *a, *b, 1e-12);
        }
    }
}

#[test]
fn r2_measures_match_hmisc() {
    let m = r2_measures(25.3, 4.0, 200.0, Some(60.0));
    let got = [m.r2, m.r2_adj, m.r2_ess.unwrap(), m.r2_adj_ess.unwrap()];
    for k in 0..4 {
        close("ess", got[k], R2M_ESS[k], 1e-14);
    }
    let ess = effective_sample_size(200.0, &[60.0, 140.0]);
    let m = r2_measures(25.3, 4.0, 200.0, Some(ess));
    let got = [m.r2, m.r2_adj, m.r2_ess.unwrap(), m.r2_adj_ess.unwrap()];
    for k in 0..4 {
        close("freq", got[k], R2M_FREQ[k], 1e-14);
    }
    let m = r2_measures(25.3, 4.0, 200.0, None);
    close("none", m.r2, R2M_NONE[0], 1e-14);
    close("none adj", m.r2_adj, R2M_NONE[1], 1e-14);
    assert!(m.r2_ess.is_none());
}

fn val_prob_vector(v: &anofox_regression::diagnostics::accuracy::ValProb) -> [f64; 18] {
    [
        v.dxy,
        v.c,
        v.r2,
        v.d,
        v.d_chisq,
        v.d_p,
        v.u,
        v.u_chisq,
        v.u_p,
        v.q,
        v.brier,
        v.intercept,
        v.slope,
        v.emax,
        v.e90,
        v.eavg,
        v.s_z,
        v.s_p,
    ]
}

/// Equal, or the same infinity.
fn close_or_inf(name: &str, a: f64, b: f64, tol: f64) {
    if b.is_infinite() {
        assert_eq!(a, b, "{name}");
    } else {
        close(name, a, b, tol);
    }
}

fn check_val_prob(p: &[f64], want: &[f64]) {
    let got = val_prob_vector(&val_prob(p, &Y).unwrap());
    for (k, (a, b)) in got.iter().zip(want).enumerate() {
        let tol = if VALPROB_NAMES[k].ends_with(":p") {
            1e-6
        } else {
            1e-8
        };
        close_or_inf(VALPROB_NAMES[k], *a, *b, tol);
    }
}

#[test]
fn val_prob_matches_rms() {
    let v = val_prob(&PV, &Y).unwrap();
    let got = val_prob_vector(&v);
    for (k, (a, b)) in got.iter().zip(VALPROB.iter()).enumerate() {
        let tol = if VALPROB_NAMES[k].ends_with(":p") {
            1e-6
        } else {
            1e-8
        };
        close(VALPROB_NAMES[k], *a, *b, tol);
    }
}

/// `L01` is summed over all rows as in rms: `p = 0` with `y = 1` makes
/// `U = ∞`, `U:p = 0`, `Q = −∞`; the other 0 / 1 cases drop out.
#[test]
fn val_prob_with_probabilities_of_zero_and_one_matches_rms() {
    let idx: Vec<usize> = VALPROB_EDGE_IDX.iter().map(|&i| i as usize).collect();
    let (one1, one2, zero1, zero2) = (idx[0], idx[1], idx[2], idx[3]);
    assert!(Y[one1] == 1.0 && Y[one2] == 1.0 && Y[zero1] == 0.0 && Y[zero2] == 0.0);

    let mut p = PV.to_vec();
    p[one1] = 1.0;
    p[one2] = 0.0;
    check_val_prob(&p, &VALPROB_EDGE_INF);
    let v = val_prob(&p, &Y).unwrap();
    assert!(v.u == f64::INFINITY && v.u_p == 0.0 && v.q == f64::NEG_INFINITY);

    let mut p = PV.to_vec();
    p[one1] = 1.0;
    p[zero1] = 0.0;
    p[zero2] = 1.0;
    check_val_prob(&p, &VALPROB_EDGE_FIN);
}

/// Constant `p = 1`: every `L01` term is `NaN` in R and dropped (`L01 = 0`).
/// rms's constant-`p` vector has one value too many (a stray 0 before
/// `Eavg`), so only `Dxy` .. `Intercept` and Spiegelhalter's test (its last
/// two values) are compared.
#[test]
fn val_prob_constant_one_matches_rms() {
    let got = val_prob_vector(&val_prob(&vec![1.0; Y.len()], &Y).unwrap());
    let want = &VALPROB_CONST_ONE;
    for k in 0..12 {
        close_or_inf(VALPROB_NAMES[k], got[k], want[k], 1e-8);
    }
    close_or_inf("S:z", got[16], want[17], 1e-8);
    close_or_inf("S:p", got[17], want[18], 1e-6);
}
