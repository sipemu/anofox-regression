#![allow(clippy::excessive_precision)]

//! Nomogram points (`rms::nomogram`, follow-up of #116 in #135):
//! `lrm(y ~ rcs(age, 4) + sex + chol + grade)` with a risk axis. Reference
//! values come from `tests/r_scripts/generate_nomogram_validation.R`.

use anofox_regression::effects::ModelSpec;
use anofox_regression::splines::Rcs;
use faer::Col;

mod fixture {
    include!("fixtures/nomogram_validation.rs");
}
use fixture::*;

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length {a:?} vs {b:?}");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        assert!(
            (u - v).abs() <= tol * (1.0 + v.abs()),
            "{name}[{k}]: {u} vs {v}"
        );
    }
}

fn plogis(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn nomogram_matches_rms() {
    let mut spec = ModelSpec::new();
    spec.rcs("age", Rcs::from_data(&AGE, 4).unwrap(), &AGE);
    spec.categorical("sex", &SEX);
    spec.linear("chol", &CHOL);
    spec.linear("grade", &GRADE);
    let beta = Col::from_fn(BETA.len(), |j| BETA[j]);
    let nom = spec.nomogram(&beta, true, &LP, &[], 100.0, 10).unwrap();

    let expected = [
        (&AGE_VALUES[..], &AGE_XBETA[..], &AGE_POINTS[..]),
        (&SEX_VALUES, &SEX_XBETA, &SEX_POINTS),
        (&CHOL_VALUES, &CHOL_XBETA, &CHOL_POINTS),
        (&GRADE_VALUES, &GRADE_XBETA, &GRADE_POINTS),
    ];
    for (axis, (values, xbeta, points)) in nom.axes.iter().zip(expected) {
        all_close(
            &format!("{} values", axis.name),
            &axis.values,
            values,
            1e-12,
        );
        all_close(&format!("{} xbeta", axis.name), &axis.xbeta, xbeta, 1e-10);
        all_close(
            &format!("{} points", axis.name),
            &axis.points,
            points,
            1e-10,
        );
    }
    all_close(
        "scale",
        &[nom.scale, nom.intercept],
        &SCALE_INTERCEPT,
        1e-10,
    );
    all_close("total points", &nom.total_points, &TOTAL_POINTS, 1e-12);
    all_close("lp.at", &nom.lp_at, &LP_AT, 1e-12);
    all_close("lp points", &nom.lp_points, &LP_POINTS, 1e-10);
    let (risk, at) = nom.function_axis(plogis, 10);
    all_close("risk ticks", &risk, &RISK_AT, 1e-12);
    all_close("risk points", &at, &RISK_POINTS, 1e-10);

    // the score of a patient maps back to its linear predictor
    let total: f64 = nom.axes.iter().map(|a| a.points[0]).sum();
    let lp: f64 = BETA[0] + nom.axes.iter().map(|a| a.xbeta[0]).sum::<f64>();
    assert!((nom.points_to_lp(total) - lp).abs() < 1e-10);
    assert!((nom.lp_to_points(lp) - total).abs() < 1e-8);
}

#[test]
fn explicit_ticks_and_interactions() {
    let mut spec = ModelSpec::new();
    let a = spec.linear("chol", &CHOL);
    let b = spec.linear("grade", &GRADE);
    let beta = Col::from_fn(3, |j| [0.5, 0.01, 0.3][j]);
    let nom = spec
        .nomogram(
            &beta,
            true,
            &[0.0, 3.0],
            &[Some(vec![150.0, 250.0]), None],
            100.0,
            10,
        )
        .unwrap();
    assert_eq!(nom.axes[0].values, vec![150.0, 250.0]);
    assert_eq!(nom.axes[1].values, vec![1.0, 2.0, 3.0, 4.0]);
    spec.interaction(a, b);
    assert!(spec
        .nomogram(&beta, true, &[0.0, 3.0], &[], 100.0, 10)
        .is_err());
}
