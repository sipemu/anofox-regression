#![allow(clippy::excessive_precision)]

//! #110: penalized cumulative link models against `rms::orm(penalty = ,
//! penalty.matrix = )`. Reference values come from
//! `tests/r_scripts/generate_ordinal_penalty_validation.R`.

use anofox_regression::solvers::{FittedOrdinal, OrdinalLink, OrdinalRegressor};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/ordinal_penalty_validation.rs");
}
use fixture::*;

const N: usize = 150;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

struct Reference<'a> {
    coef: &'a [f64],
    deviance: &'a [f64],
    lr_df: &'a [f64],
    effective_df: &'a [f64],
    vcov_slopes: &'a [f64],
    vcov: Option<&'a [f64]>,
}

fn check(id: &str, f: &FittedOrdinal, r: &Reference) {
    let m = f.intercepts().nrows();
    for j in 0..m {
        close(&format!("{id} α{j}"), f.intercepts()[j], r.coef[j], 1e-6);
    }
    for j in 0..3 {
        close(
            &format!("{id} β{j}"),
            f.coefficients()[j],
            r.coef[m + j],
            1e-6,
        );
        close(
            &format!("{id} effective df {j}"),
            f.effective_df_diagonal()[j],
            r.effective_df[j],
            1e-6,
        );
    }
    close(
        &format!("{id} null deviance"),
        -2.0 * f.null_log_likelihood(),
        r.deviance[0],
        1e-10,
    );
    close(
        &format!("{id} penalized deviance"),
        -2.0 * f.penalized_log_likelihood(),
        r.deviance[1],
        1e-9,
    );
    close(&format!("{id} LR χ²"), f.lr_chisq(), r.lr_df[0], 1e-7);
    close(&format!("{id} d.f."), f.effective_df(), r.lr_df[1], 1e-6);
    let vs = f.vcov_slopes();
    for a in 0..3 {
        for b in 0..3 {
            close(
                &format!("{id} vcov slopes {a},{b}"),
                vs[(a, b)],
                r.vcov_slopes[a * 3 + b],
                1e-6,
            );
        }
    }
    if let Some(v_ref) = r.vcov {
        let v = f.vcov();
        let d = m + 3;
        for a in 0..d {
            for b in 0..d {
                close(
                    &format!("{id} vcov {a},{b}"),
                    v[(a, b)],
                    v_ref[a * d + b],
                    1e-6,
                );
            }
        }
    }
}

#[test]
fn penalized_fits_match_orm() {
    let x = Mat::from_fn(N, 3, |i, j| X[i * 3 + j]);
    let y_cat = Col::from_fn(N, |i| Y_CAT[i]);
    let y_cont = Col::from_fn(N, |i| Y_CONT[i]);

    let f = OrdinalRegressor::builder()
        .penalty(2.0)
        .build()
        .fit(&x, &y_cat)
        .unwrap();
    check(
        "penalty 2",
        &f,
        &Reference {
            coef: &PEN2_COEF,
            deviance: &PEN2_DEVIANCE,
            lr_df: &PEN2_LR_DF,
            effective_df: &PEN2_EFFECTIVE_DF,
            vcov_slopes: &PEN2_VCOV_SLOPES,
            vcov: Some(&PEN2_VCOV),
        },
    );
    let pm = f.penalty_matrix().unwrap();
    assert_eq!(pm[(0, 1)], 0.0);

    let custom = Mat::from_fn(3, 3, |a, b| CUSTOM_PENALTY[a * 3 + b]);
    let f = OrdinalRegressor::builder()
        .link(OrdinalLink::Probit)
        .penalty_matrix(custom)
        .build()
        .fit(&x, &y_cat)
        .unwrap();
    check(
        "probit, penalty matrix",
        &f,
        &Reference {
            coef: &PROBIT_MATRIX_COEF,
            deviance: &PROBIT_MATRIX_DEVIANCE,
            lr_df: &PROBIT_MATRIX_LR_DF,
            effective_df: &PROBIT_MATRIX_EFFECTIVE_DF,
            vcov_slopes: &PROBIT_MATRIX_VCOV_SLOPES,
            vcov: Some(&PROBIT_MATRIX_VCOV),
        },
    );

    let f = OrdinalRegressor::builder()
        .penalty(1.0)
        .build()
        .fit(&x, &y_cont)
        .unwrap();
    check(
        "continuous, penalty 1",
        &f,
        &Reference {
            coef: &CONT_PEN1_COEF,
            deviance: &CONT_PEN1_DEVIANCE,
            lr_df: &CONT_PEN1_LR_DF,
            effective_df: &CONT_PEN1_EFFECTIVE_DF,
            vcov_slopes: &CONT_PEN1_VCOV_SLOPES,
            vcov: None,
        },
    );
}

#[test]
fn penalty_edge_cases() {
    let x = Mat::from_fn(N, 3, |i, j| X[i * 3 + j]);
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    // a penalty matrix of the wrong size
    assert!(OrdinalRegressor::builder()
        .penalty_matrix(Mat::identity(2, 2))
        .build()
        .fit(&x, &y)
        .is_err());
    // without a penalty: effective d.f. 1 each, the penalized objective is ℓ
    let f = OrdinalRegressor::builder().build().fit(&x, &y).unwrap();
    assert!(f.penalty_matrix().is_none());
    assert_eq!(f.effective_df(), 3.0);
    assert_eq!(f.penalized_log_likelihood(), f.log_likelihood());
    // a penalty with an offset: the intercepts-only null is fitted unpenalized
    let off = Col::from_fn(N, |i| 0.3 * ((i + 1) as f64).sin());
    let g = OrdinalRegressor::builder()
        .penalty(2.0)
        .offset(off)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!(g.lr_chisq() > 0.0 && g.effective_df() < 3.0);
    assert!(g.aic().is_finite());
}
