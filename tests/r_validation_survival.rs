#![allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]

//! #113: Kaplan–Meier, Nelson–Aalen / Fleming–Harrington and the log-rank /
//! G-ρ tests against `survival::survfit` and `survival::survdiff`.
//!
//! Reference values come from `tests/r_scripts/generate_survival_validation.R`.

use anofox_regression::survival::{
    survdiff, survfit, ConfType, HazardType, SurvCurve, SurvInput, SurvfitOptions,
};

include!("fixtures/survival_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    let ok = (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= tol * (1.0 + b.abs());
    assert!(ok, "{name}: {a} vs {b}");
}

fn vclose(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name} length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *x, *y, tol);
    }
}

fn events() -> Vec<bool> {
    STATUS.iter().map(|&s| s == 1.0).collect()
}

#[allow(clippy::too_many_arguments)]
fn check_curve(
    id: &str,
    c: &SurvCurve,
    time: &[f64],
    nrisk: &[f64],
    nevent: &[f64],
    ncens: &[f64],
    surv: &[f64],
    se: &[f64],
    lower: &[f64],
    upper: &[f64],
    cumhaz: &[f64],
    sechaz: &[f64],
) {
    let tol = 1e-12;
    vclose(&format!("{id} time"), &c.time, time, tol);
    vclose(&format!("{id} n.risk"), &c.n_risk, nrisk, tol);
    vclose(&format!("{id} n.event"), &c.n_event, nevent, tol);
    vclose(&format!("{id} n.censor"), &c.n_censor, ncens, tol);
    vclose(&format!("{id} surv"), &c.surv, surv, tol);
    vclose(&format!("{id} std.err"), &c.std_err, se, tol);
    vclose(&format!("{id} lower"), &c.lower, lower, 1e-10);
    vclose(&format!("{id} upper"), &c.upper, upper, 1e-10);
    vclose(&format!("{id} cumhaz"), &c.cumhaz, cumhaz, tol);
    vclose(&format!("{id} std.chaz"), &c.std_chaz, sechaz, tol);
}

#[test]
fn kaplan_meier_matches_survfit() {
    let ev = events();
    let opts = SurvfitOptions::default();
    let c = &survfit(&SurvInput::new(&TIME, &ev), None, &opts).unwrap()[0];
    check_curve(
        "KM",
        c,
        &KM_TIME,
        &KM_NRISK,
        &KM_NEVENT,
        &KM_NCENSOR,
        &KM_SURV,
        &KM_SE,
        &KM_LOWER,
        &KM_UPPER,
        &KM_CUMHAZ,
        &KM_SECHAZ,
    );
    let s = c.summary(*TIME.iter().max_by(|a, b| a.total_cmp(b)).unwrap());
    let got = [
        s.records as f64,
        s.n_max,
        s.n_start,
        s.events,
        s.rmean,
        s.se_rmean,
        s.median,
        s.median_lower,
        s.median_upper,
    ];
    vclose("KM table", &got, &KM_TABLE, 1e-10);
}

#[test]
fn confidence_types_and_fleming_harrington() {
    let ev = events();
    let input = SurvInput::new(&TIME, &ev);
    let ll = SurvfitOptions {
        conf_level: 0.9,
        conf_type: ConfType::LogLog,
        ..Default::default()
    };
    let c = &survfit(&input, None, &ll).unwrap()[0];
    check_curve(
        "LOGLOG",
        c,
        &KM_LOGLOG_TIME,
        &KM_LOGLOG_NRISK,
        &KM_LOGLOG_NEVENT,
        &KM_LOGLOG_NCENSOR,
        &KM_LOGLOG_SURV,
        &KM_LOGLOG_SE,
        &KM_LOGLOG_LOWER,
        &KM_LOGLOG_UPPER,
        &KM_LOGLOG_CUMHAZ,
        &KM_LOGLOG_SECHAZ,
    );
    let pl = SurvfitOptions {
        conf_type: ConfType::Plain,
        ..Default::default()
    };
    let c = &survfit(&input, None, &pl).unwrap()[0];
    check_curve(
        "PLAIN",
        c,
        &KM_PLAIN_TIME,
        &KM_PLAIN_NRISK,
        &KM_PLAIN_NEVENT,
        &KM_PLAIN_NCENSOR,
        &KM_PLAIN_SURV,
        &KM_PLAIN_SE,
        &KM_PLAIN_LOWER,
        &KM_PLAIN_UPPER,
        &KM_PLAIN_CUMHAZ,
        &KM_PLAIN_SECHAZ,
    );
    let fh = SurvfitOptions {
        hazard: HazardType::FlemingHarrington,
        ..Default::default()
    };
    let c = &survfit(&input, None, &fh).unwrap()[0];
    vclose("FH cumhaz", &c.cumhaz, &FH_CUMHAZ, 1e-12);
    vclose("FH std.chaz", &c.std_chaz, &FH_SECHAZ, 1e-12);
}

#[test]
fn strata_and_summary_table() {
    let ev = events();
    let sex: Vec<usize> = SEX.to_vec();
    let curves = survfit(
        &SurvInput::new(&TIME, &ev),
        Some(&sex),
        &SurvfitOptions::default(),
    )
    .unwrap();
    assert_eq!(curves.len(), 2);
    let mut at = 0;
    let all = [
        &KMS_TIME[..],
        &KMS_NRISK,
        &KMS_NEVENT,
        &KMS_NCENSOR,
        &KMS_SURV,
        &KMS_SE,
        &KMS_LOWER,
        &KMS_UPPER,
        &KMS_CUMHAZ,
        &KMS_SECHAZ,
    ];
    for (s, c) in curves.iter().enumerate() {
        assert_eq!(c.stratum, Some(s));
        let m = KMS_STRATA[s];
        let p: Vec<&[f64]> = all.iter().map(|v| &v[at..at + m]).collect();
        check_curve(
            &format!("stratum {s}"),
            c,
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            p[9],
        );
        at += m;
        let t = c.summary(KMS_RMEAN_END[0]);
        let got = [
            t.records as f64,
            t.n_max,
            t.n_start,
            t.events,
            t.rmean,
            t.se_rmean,
            t.median,
            t.median_lower,
            t.median_upper,
        ];
        // summary(fit)$table is (strata x 9), emitted column-major by t().
        let want: Vec<f64> = (0..9).map(|j| KMS_TABLE[s * 9 + j]).collect();
        vclose(&format!("table {s}"), &got, &want, 1e-10);
    }
}

#[test]
fn case_weights_match_survfit() {
    let ev = events();
    let c = &survfit(
        &SurvInput::new(&TIME, &ev).weights(&W),
        None,
        &SurvfitOptions::default(),
    )
    .unwrap()[0];
    check_curve(
        "W",
        c,
        &KMW_TIME,
        &KMW_NRISK,
        &KMW_NEVENT,
        &KMW_NCENSOR,
        &KMW_SURV,
        &KMW_SE,
        &KMW_LOWER,
        &KMW_UPPER,
        &KMW_CUMHAZ,
        &KMW_SECHAZ,
    );
}

#[test]
fn left_truncation_matches_survfit() {
    let ev: Vec<bool> = LT_EVENT.iter().map(|&e| e == 1.0).collect();
    let input = SurvInput::new(&LT_STOP, &ev).entry(&LT_ENTRY);
    let c = &survfit(&input, None, &SurvfitOptions::default()).unwrap()[0];
    check_curve(
        "LT",
        c,
        &LT_TIME,
        &LT_NRISK,
        &LT_NEVENT,
        &LT_NCENSOR,
        &LT_SURV,
        &LT_SE,
        &LT_LOWER,
        &LT_UPPER,
        &LT_CUMHAZ,
        &LT_SECHAZ,
    );
}

#[test]
fn median_midpoint_rule() {
    let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ev = [true; 6];
    let c = &survfit(
        &SurvInput::new(&time, &ev),
        None,
        &SurvfitOptions::default(),
    )
    .unwrap()[0];
    let t = c.summary(6.0);
    let got = [
        t.records as f64,
        t.n_max,
        t.n_start,
        t.events,
        t.rmean,
        t.se_rmean,
        t.median,
        t.median_lower,
        t.median_upper,
    ];
    vclose("midpoint table", &got, &MID_TABLE, 1e-12);
}

fn check_diff(
    id: &str,
    d: &anofox_regression::survival::SurvDiff,
    obs: &[f64],
    exp: &[f64],
    var: &[f64],
    chisq: f64,
    p: f64,
) {
    vclose(&format!("{id} obs"), &d.observed, obs, 1e-12);
    vclose(&format!("{id} exp"), &d.expected, exp, 1e-12);
    let k = obs.len();
    for a in 0..k {
        for b in 0..k {
            close(
                &format!("{id} var[{a},{b}]"),
                d.variance[(a, b)],
                var[a * k + b],
                1e-10,
            );
        }
    }
    close(&format!("{id} chisq"), d.chisq, chisq, 1e-10);
    assert!(
        (d.p_value - p).abs() <= 1e-6 * p + 1e-15,
        "{id} p {} vs {p}",
        d.p_value
    );
}

#[test]
fn survdiff_matches_r() {
    let ev = events();
    let sex: Vec<usize> = SEX.to_vec();
    let ecog: Vec<usize> = ECOG.to_vec();
    check_diff(
        "log-rank 2",
        &survdiff(&TIME, &ev, &sex, None, 0.0).unwrap(),
        &LR2_OBS,
        &LR2_EXP,
        &LR2_VAR,
        LR2_CHISQ,
        LR2_P,
    );
    check_diff(
        "log-rank k",
        &survdiff(&TIME, &ev, &ecog, None, 0.0).unwrap(),
        &LRK_OBS,
        &LRK_EXP,
        &LRK_VAR,
        LRK_CHISQ,
        LRK_P,
    );
    check_diff(
        "Peto",
        &survdiff(&TIME, &ev, &ecog, None, 1.0).unwrap(),
        &PETO_OBS,
        &PETO_EXP,
        &PETO_VAR,
        PETO_CHISQ,
        PETO_P,
    );
    check_diff(
        "stratified",
        &survdiff(&TIME, &ev, &ecog, Some(&sex), 0.0).unwrap(),
        &STRAT_OBS,
        &STRAT_EXP,
        &STRAT_VAR,
        STRAT_CHISQ,
        STRAT_P,
    );
}
