#![allow(clippy::excessive_precision)]

//! #106: Rubin's rules (RMS §3.8). Six `aregImpute` imputations of
//! `y ~ x1 + x2 + x3` fitted with `ols`, pooled as
//! `Hmisc::fit.mult.impute`; Barnard–Rubin d.f., `fmi` and the D1 test as
//! `mice::pool` / `mice::D1`. Reference values come from
//! `tests/r_scripts/generate_pool_validation.R`.

use anofox_regression::inference::pool::{pool_rubin, pooled_wald_d1, PooledFit};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/pool_validation.rs");
}
use fixture::*;

const P: usize = 4;

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        assert!(
            (u - v).abs() <= tol * (1.0 + v.abs()),
            "{name}[{k}]: {u} vs {v}"
        );
    }
}

fn v(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn fits() -> (Vec<Col<f64>>, Vec<Mat<f64>>) {
    let m = M[0] as usize;
    let coefs = (0..m)
        .map(|i| Col::from_fn(P, |j| COEFS[i * P + j]))
        .collect();
    let vcovs = (0..m)
        .map(|i| Mat::from_fn(P, P, |a, b| VCOVS[i * P * P + b * P + a]))
        .collect();
    (coefs, vcovs)
}

fn pooled(df_complete: Option<f64>) -> PooledFit {
    let (c, v) = fits();
    pool_rubin(&c, &v, df_complete).unwrap()
}

#[test]
fn rubin_rules_match_fit_mult_impute() {
    let f = pooled(None);
    all_close("coef", &v(&f.coefficients), &POOL_COEF, 1e-12);
    let t: Vec<f64> = (0..P * P).map(|k| f.vcov[(k % P, k / P)]).collect();
    all_close("vcov", &t, &POOL_VCOV, 1e-12);
    all_close(
        "variance inflation",
        &v(&f.variance_inflation),
        &POOL_R,
        1e-12,
    );
    all_close("missing info", &v(&f.lambda), &POOL_FMI, 1e-12);
    all_close("dfmi", &v(&f.df_rubin), &POOL_DF, 1e-10);
    assert!(f.df_barnard_rubin.is_none());
    let se = f.std_errors();
    assert!((se[1] - POOL_VCOV[P + 1].sqrt()).abs() < 1e-14);
}

#[test]
fn barnard_rubin_matches_mice() {
    let f = pooled(Some(150.0 - 4.0));
    all_close("riv", &v(&f.riv), &MICE_RIV, 1e-12);
    all_close("lambda", &v(&f.lambda), &MICE_LAMBDA, 1e-12);
    all_close(
        "df",
        &v(f.df_barnard_rubin.as_ref().unwrap()),
        &MICE_DF,
        1e-10,
    );
    all_close("fmi", &v(&f.fmi), &MICE_FMI, 1e-10);
}

#[test]
fn d1_matches_mice() {
    let f = pooled(None);
    for (idx, expected) in [(&[1usize, 2][..], &D1_X1X2[..]), (&[1, 2, 3], &D1_ALL)] {
        let t = pooled_wald_d1(&f, idx).unwrap();
        all_close("D1", &[t.f, t.df1, t.df2, t.p_value, t.riv], expected, 1e-9);
    }
    // a single coefficient with few imputations: the small-ν₁ branch
    let (c, v) = fits();
    let small = pool_rubin(&c[..3], &v[..3], None).unwrap();
    let t = pooled_wald_d1(&small, &[3]).unwrap();
    let r1 = small.riv[3];
    let df2 = 2.0 * (1.0 + 1.0) * (1.0 + 1.0 / r1).powi(2) / 2.0;
    all_close("small df2", &[t.df2], &[df2], 1e-12);
    all_close(
        "small F",
        &[t.f],
        &[small.coefficients[3].powi(2) / ((1.0 + r1) * small.within[(3, 3)])],
        1e-12,
    );
}

#[test]
fn rejects_bad_input() {
    let (c, v) = fits();
    assert!(pool_rubin(&c[..1], &v[..1], None).is_err());
    assert!(pool_rubin(&c, &v[..2], None).is_err());
    assert!(pool_rubin(&c, &v, Some(0.0)).is_err());
    let mut c2 = c.clone();
    c2[0] = Col::zeros(P + 1);
    assert!(pool_rubin(&c2, &v, None).is_err());
    let f = pooled(None);
    assert!(pooled_wald_d1(&f, &[]).is_err());
    assert!(pooled_wald_d1(&f, &[P]).is_err());
}
