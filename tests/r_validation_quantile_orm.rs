#![allow(clippy::excessive_precision)]

//! Confidence limits of predicted quantiles from a cumulative link model
//! (`rms::Quantile.orm(conf.int = )`, follow-up of #110 in #135).
//! Reference values come from
//! `tests/r_scripts/generate_quantile_orm_validation.R`; `orm` converges to
//! about 1e-7, hence the tolerance.

use anofox_regression::solvers::{OrdinalLink, OrdinalQuantileMethod, OrdinalRegressor};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/quantile_orm_validation.rs");
}
use fixture::*;

const N: usize = 200;

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        assert!(
            (u - v).abs() <= tol * (1.0 + v.abs()),
            "{name}[{k}]: {u} vs {v}"
        );
    }
}

#[test]
fn quantile_limits_match_quantile_orm() {
    let x = Mat::from_fn(N, 2, |i, j| X[i * 2 + j]);
    let nd = Mat::from_fn(4, 2, |i, j| NEW[i * 2 + j]);
    let cases = [
        (
            OrdinalLink::Logistic,
            [
                &LOGISTIC_Q50_INTE,
                &LOGISTIC_Q50_DISC,
                &LOGISTIC_Q90_INTE,
                &LOGISTIC_Q90_DISC,
            ],
        ),
        (
            OrdinalLink::Probit,
            [
                &PROBIT_Q50_INTE,
                &PROBIT_Q50_DISC,
                &PROBIT_Q90_INTE,
                &PROBIT_Q90_DISC,
            ],
        ),
    ];
    for (link, refs) in cases {
        let f = OrdinalRegressor::builder()
            .link(link)
            .build()
            .fit(&x, &Col::from_fn(N, |i| Y[i]))
            .unwrap();
        let settings = [
            (0.5, OrdinalQuantileMethod::Interpolated),
            (0.5, OrdinalQuantileMethod::Discrete),
            (0.9, OrdinalQuantileMethod::Interpolated),
            (0.9, OrdinalQuantileMethod::Discrete),
        ];
        for ((q, method), expected) in settings.into_iter().zip(refs) {
            let (est, lo, hi) = f.predict_quantile_ci(&nd, q, method, 0.95);
            let got: Vec<f64> = est
                .iter()
                .chain(lo.iter())
                .chain(hi.iter())
                .copied()
                .collect();
            all_close(&format!("{link:?} q={q} {method:?}"), &got, expected, 1e-5);
            let point = f.predict_quantile(&nd, q, method);
            assert_eq!(point, est);
        }
    }
}
