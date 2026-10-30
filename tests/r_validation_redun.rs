#![allow(clippy::excessive_precision)]

//! #107: redundancy analysis (`Hmisc::redun`, RMS §4.7.1). Reference values
//! come from `tests/r_scripts/generate_redun_validation.R`: columns
//! `a, b, ab ≈ a + b, a2 ≈ exp(a), g` (3 levels, from `cc`), `cc, e`.

use anofox_regression::reduction::{redun, Redun, RedunOptions, RedunVariable};
use faer::Mat;

mod fixture {
    include!("fixtures/redun_validation.rs");
}
use fixture::*;

const N: usize = 200;
const P: usize = 7;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length {a:?} vs {b:?}");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *u, *v, tol);
    }
}

fn kinds(linear_a2: bool) -> Vec<RedunVariable> {
    let mut k = vec![RedunVariable::Numeric; P];
    k[4] = RedunVariable::Categorical;
    if linear_a2 {
        k[3] = RedunVariable::Linear;
    }
    k
}

fn run(opts: RedunOptions, linear_a2: bool) -> Redun {
    let x = Mat::from_fn(N, P, |i, j| X[i * P + j]);
    redun(&x, &kinds(linear_a2), &opts).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn check(
    tag: &str,
    r: &Redun,
    out: &[f64],
    kept: &[f64],
    rsq: &[f64],
    rsq1: &[f64],
    later: &[f64],
) {
    let f = |v: &[usize]| v.iter().map(|&i| i as f64).collect::<Vec<_>>();
    assert_eq!(f(&r.removed), out, "{tag} removed");
    assert_eq!(f(&r.kept), kept, "{tag} kept");
    all_close(&format!("{tag} rsq"), &r.r_squared, rsq, 1e-8);
    all_close(&format!("{tag} rsq1"), &r.r_squared_first, rsq1, 1e-8);
    let l: Vec<f64> = r.r_squared_later.iter().flatten().copied().collect();
    all_close(&format!("{tag} later"), &l, later, 1e-8);
    assert!(r.too_few.is_empty());
}

#[test]
fn redun_matches_hmisc() {
    let d = RedunOptions::default();
    let cases = [
        (
            "DEF",
            RedunOptions { r2: 0.8, ..d },
            false,
            &DEF_OUT[..],
            &DEF_IN[..],
            &DEF_RSQ[..],
            &DEF_RSQ1[..],
            &DEF_LATER[..],
        ),
        (
            "ADJ",
            RedunOptions {
                r2: 0.8,
                adjusted: true,
                nk: 4,
                ..d
            },
            false,
            &ADJ_OUT,
            &ADJ_IN,
            &ADJ_RSQ,
            &ADJ_RSQ1,
            &ADJ_LATER,
        ),
        (
            "NOTL",
            RedunOptions {
                r2: 0.7,
                tlinear: false,
                ..d
            },
            false,
            &NOTL_OUT,
            &NOTL_IN,
            &NOTL_RSQ,
            &NOTL_RSQ1,
            &NOTL_LATER,
        ),
        (
            "ALLCAT",
            RedunOptions {
                r2: 0.5,
                allcat: true,
                ..d
            },
            false,
            &ALLCAT_OUT,
            &ALLCAT_IN,
            &ALLCAT_RSQ,
            &ALLCAT_RSQ1,
            &ALLCAT_LATER,
        ),
        (
            "RANK",
            RedunOptions {
                r2: 0.7,
                rank: true,
                ..d
            },
            false,
            &RANK_OUT,
            &RANK_IN,
            &RANK_RSQ,
            &RANK_RSQ1,
            &RANK_LATER,
        ),
        (
            "QRANK",
            RedunOptions {
                r2: 0.7,
                qrank: true,
                ..d
            },
            false,
            &QRANK_OUT,
            &QRANK_IN,
            &QRANK_RSQ,
            &QRANK_RSQ1,
            &QRANK_LATER,
        ),
        (
            "LIN",
            RedunOptions { r2: 0.8, ..d },
            true,
            &LIN_OUT,
            &LIN_IN,
            &LIN_RSQ,
            &LIN_RSQ1,
            &LIN_LATER,
        ),
    ];
    for (tag, opts, lin, out, kept, rsq, rsq1, later) in cases {
        check(tag, &run(opts, lin), out, kept, rsq, rsq1, later);
    }
}

#[test]
fn constant_columns_are_set_aside_and_input_checked() {
    let x = Mat::from_fn(N, P + 1, |i, j| if j < P { X[i * P + j] } else { 2.0 });
    let mut k = kinds(false);
    k.push(RedunVariable::Numeric);
    let r = redun(
        &x,
        &k,
        &RedunOptions {
            r2: 0.8,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.too_few, vec![P]);
    assert_eq!(r.removed, vec![0, 2, 4]);
    assert!(redun(&x, &k[..2], &RedunOptions::default()).is_err());
    // nothing above the cutoff: all kept
    let r = redun(
        &x,
        &k,
        &RedunOptions {
            r2: 0.999,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(r.removed.is_empty() && r.kept.len() == P);
}
