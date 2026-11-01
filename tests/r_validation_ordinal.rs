#![allow(clippy::excessive_precision)]

//! #110: cumulative link models against `rms::orm` / `lrm` and `MASS::polr`.
//! Reference values come from `tests/r_scripts/generate_ordinal_validation.R`
//! (`orm` converged with `eps = 1e-10`, `gradtol = 1e-9`).

use anofox_regression::solvers::{
    FittedOrdinal, OrdinalLink, OrdinalQuantileMethod, OrdinalRegressor,
};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/ordinal_validation.rs");
}
use fixture::*;

const N: usize = 150;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 3, |i, j| X[i * 3 + j])
}

fn xnew() -> Mat<f64> {
    Mat::from_fn(3, 3, |i, j| XNEW[i * 3 + j])
}

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{name}: {a} vs {b} (diff {:e})",
        (a - b).abs()
    );
}

fn fit(
    y: &[f64],
    link: OrdinalLink,
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
) -> FittedOrdinal {
    let mut b = OrdinalRegressor::builder().link(link);
    if let Some(w) = weights {
        b = b.weights(Col::from_fn(N, |i| w[i]));
    }
    if let Some(o) = offset {
        b = b.offset(Col::from_fn(N, |i| o[i]));
    }
    let f = b.build().fit(&x(), &Col::from_fn(N, |i| y[i])).unwrap();
    assert!(f.converged(), "not converged");
    f
}

/// Coefficients (intercepts then slopes), log-likelihoods and covariance.
fn check_fit(
    id: &str,
    f: &FittedOrdinal,
    coef: &[f64],
    ll: &[f64],
    vcov: Option<&[f64]>,
    vslopes: &[f64],
    tol: f64,
) {
    let m = f.intercepts().nrows();
    assert_eq!(m + 3, coef.len(), "{id}: number of parameters");
    for (j, &want) in coef.iter().enumerate().take(m) {
        close(&format!("{id} α{j}"), f.intercepts()[j], want, tol);
    }
    for j in 0..3 {
        close(&format!("{id} β{j}"), f.coefficients()[j], coef[m + j], tol);
    }
    // orm's deviance: null, [null with the offset,] model
    let l = ll.len();
    close(
        &format!("{id} null ll"),
        f.null_log_likelihood(),
        ll[l - 2],
        1e-9,
    );
    close(&format!("{id} ll"), f.log_likelihood(), ll[l - 1], 1e-9);
    let vs = f.vcov_slopes();
    for a in 0..3 {
        for b in 0..3 {
            close(
                &format!("{id} vcov slopes {a},{b}"),
                vs[(a, b)],
                vslopes[a * 3 + b],
                tol,
            );
        }
    }
    if let Some(v_ref) = vcov {
        let v = f.vcov();
        let d = m + 3;
        for a in 0..d {
            for b in 0..d {
                close(
                    &format!("{id} vcov {a},{b}"),
                    v[(a, b)],
                    v_ref[a * d + b],
                    tol,
                );
            }
        }
    }
}

/// Name, link and seven reference vectors of one `orm` family.
type OrmCase<'a> = (
    &'a str,
    OrdinalLink,
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
);

#[test]
fn orm_continuous_response_every_family() {
    let cases: [OrmCase; 5] = [
        (
            "logistic",
            OrdinalLink::Logistic,
            &ORM_LOGISTIC_COEF,
            &ORM_LOGISTIC_LL,
            &ORM_LOGISTIC_VCOV_SLOPES,
            &ORM_LOGISTIC_MEAN,
            &ORM_LOGISTIC_MEAN_SE,
            &ORM_LOGISTIC_Q_INTERP,
            &ORM_LOGISTIC_Q_DISCRETE,
        ),
        (
            "probit",
            OrdinalLink::Probit,
            &ORM_PROBIT_COEF,
            &ORM_PROBIT_LL,
            &ORM_PROBIT_VCOV_SLOPES,
            &ORM_PROBIT_MEAN,
            &ORM_PROBIT_MEAN_SE,
            &ORM_PROBIT_Q_INTERP,
            &ORM_PROBIT_Q_DISCRETE,
        ),
        (
            "loglog",
            OrdinalLink::LogLog,
            &ORM_LOGLOG_COEF,
            &ORM_LOGLOG_LL,
            &ORM_LOGLOG_VCOV_SLOPES,
            &ORM_LOGLOG_MEAN,
            &ORM_LOGLOG_MEAN_SE,
            &ORM_LOGLOG_Q_INTERP,
            &ORM_LOGLOG_Q_DISCRETE,
        ),
        (
            "cloglog",
            OrdinalLink::CLogLog,
            &ORM_CLOGLOG_COEF,
            &ORM_CLOGLOG_LL,
            &ORM_CLOGLOG_VCOV_SLOPES,
            &ORM_CLOGLOG_MEAN,
            &ORM_CLOGLOG_MEAN_SE,
            &ORM_CLOGLOG_Q_INTERP,
            &ORM_CLOGLOG_Q_DISCRETE,
        ),
        (
            "cauchit",
            OrdinalLink::Cauchit,
            &ORM_CAUCHIT_COEF,
            &ORM_CAUCHIT_LL,
            &ORM_CAUCHIT_VCOV_SLOPES,
            &ORM_CAUCHIT_MEAN,
            &ORM_CAUCHIT_MEAN_SE,
            &ORM_CAUCHIT_Q_INTERP,
            &ORM_CAUCHIT_Q_DISCRETE,
        ),
    ];
    let xn = xnew();
    for (id, link, coef, ll, vs, mean, mean_se, q_interp, q_disc) in cases {
        let f = fit(&Y_CONT, link, None, None);
        assert_eq!(f.levels().len(), N, "{id}: one level per value");
        check_fit(id, &f, coef, ll, None, vs, 1e-5);
        let (m, se) = f.predict_mean_with_se(&xn);
        for i in 0..3 {
            close(&format!("{id} mean {i}"), m[i], mean[i], 1e-5);
            close(&format!("{id} mean se {i}"), se[i], mean_se[i], 1e-5);
        }
        for (k, q) in [0.25, 0.5, 0.9].into_iter().enumerate() {
            let qi = f.predict_quantile(&xn, q, OrdinalQuantileMethod::Interpolated);
            for i in 0..3 {
                close(
                    &format!("{id} Q{q} interp {i}"),
                    qi[i],
                    q_interp[k * 3 + i],
                    1e-5,
                );
            }
        }
        for (k, q) in [0.25, 0.5].into_iter().enumerate() {
            let qd = f.predict_quantile(&xn, q, OrdinalQuantileMethod::Discrete);
            for i in 0..3 {
                close(
                    &format!("{id} Q{q} discrete {i}"),
                    qd[i],
                    q_disc[k * 3 + i],
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn five_categories_logit_probit_weights_offset() {
    let f = fit(&Y_CAT, OrdinalLink::Logistic, None, None);
    check_fit(
        "logit",
        &f,
        &CAT_COEF,
        &CAT_LL,
        Some(&CAT_VCOV),
        &CAT_VCOV_SLOPES,
        1e-6,
    );
    // MASS::polr: P(Y ≤ j) = F(ζ_j − xβ), so ζ = −α and the same β
    for (j, &want) in POLR_ZETA.iter().enumerate() {
        close(&format!("polr ζ{j}"), -f.intercepts()[j], want, 1e-5);
    }
    for (j, &want) in POLR_BETA.iter().enumerate() {
        close(&format!("polr β{j}"), f.coefficients()[j], want, 1e-5);
    }
    let f = fit(&Y_CAT, OrdinalLink::Probit, None, None);
    check_fit(
        "probit",
        &f,
        &CAT_PROBIT_COEF,
        &CAT_PROBIT_LL,
        Some(&CAT_PROBIT_VCOV),
        &CAT_PROBIT_VCOV_SLOPES,
        1e-6,
    );
    let f = fit(&Y_CAT, OrdinalLink::Logistic, Some(&W), None);
    check_fit(
        "weights",
        &f,
        &CAT_W_COEF,
        &CAT_W_LL,
        Some(&CAT_W_VCOV),
        &CAT_W_VCOV_SLOPES,
        1e-6,
    );
    let f = fit(&Y_CAT, OrdinalLink::Logistic, None, Some(&OFFSET));
    check_fit(
        "offset",
        &f,
        &CAT_OFF_COEF,
        &CAT_OFF_LL,
        Some(&CAT_OFF_VCOV),
        &CAT_OFF_VCOV_SLOPES,
        1e-6,
    );
}

#[test]
fn probabilities_are_consistent() {
    let f = fit(&Y_CAT, OrdinalLink::Logistic, None, None);
    let xn = xnew();
    let p = f.category_probabilities(&xn);
    let e = f.exceedance_probabilities(&xn);
    for i in 0..3 {
        let total: f64 = (0..5).map(|j| p[(i, j)]).sum();
        close("probabilities sum to 1", total, 1.0, 1e-12);
        for j in 0..4 {
            let tail: f64 = (j + 1..5).map(|c| p[(i, c)]).sum();
            close("exceedance", e[(i, j)], tail, 1e-12);
        }
    }
    close("LR χ²", f.lr_chisq(), 2.0 * (CAT_LL[1] - CAT_LL[0]), 1e-8);
    let off = fit(&Y_CAT, OrdinalLink::Logistic, None, Some(&OFFSET));
    close(
        "LR χ² with offset",
        off.lr_chisq(),
        2.0 * (CAT_OFF_LL[2] - CAT_OFF_LL[1]),
        1e-8,
    );
    assert_eq!(f.lr_df(), 3);
    close("AIC", f.aic(), -2.0 * CAT_LL[1] + 2.0 * 7.0, 1e-8);
}

#[test]
fn ordinal_rejects_bad_input() {
    let x = x();
    let one_level = Col::from_fn(N, |_| 1.0);
    assert!(OrdinalRegressor::builder()
        .build()
        .fit(&x, &one_level)
        .is_err());
    let short = Col::from_fn(10, |i| i as f64);
    assert!(OrdinalRegressor::builder().build().fit(&x, &short).is_err());
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    let bad_w = Col::from_fn(N, |i| if i == 0 { -1.0 } else { 1.0 });
    assert!(OrdinalRegressor::builder()
        .weights(bad_w)
        .build()
        .fit(&x, &y)
        .is_err());
}

#[test]
fn accessors_and_summary_statistics() {
    let f = fit(&Y_CAT, OrdinalLink::Probit, Some(&W), None);
    assert_eq!(f.link(), OrdinalLink::Probit);
    assert_eq!(f.n_observations(), N);
    assert!(f.iterations() > 0);
    let total: f64 = W.iter().sum();
    close("total weight", f.counts().iter().sum::<f64>(), total, 1e-12);
    let v = f.vcov_slopes();
    let se = f.std_errors();
    for j in 0..3 {
        close("se", se[j], v[(j, j)].sqrt(), 1e-15);
    }
    let p = f.lr_p_value();
    assert!(p > 0.0 && p < 1e-6, "LR p-value {p}");
    // intercept-only: no slopes, so no LR test
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    let f0 = OrdinalRegressor::builder()
        .build()
        .fit(&Mat::zeros(N, 0), &y)
        .unwrap();
    assert!(f0.lr_p_value().is_nan());
    close(
        "intercept-only ll",
        f0.log_likelihood(),
        f0.null_log_likelihood(),
        1e-9,
    );
}

#[test]
fn iteration_limit_reports_non_convergence() {
    let y = Col::from_fn(N, |i| Y_CONT[i]);
    let f = OrdinalRegressor::builder()
        .max_iterations(1)
        .tolerance(1e-14)
        .build()
        .fit(&x(), &y)
        .unwrap();
    assert!(!f.converged());
    assert_eq!(f.iterations(), 1);
}

#[test]
fn ordinal_rejects_more_bad_input() {
    let x = x();
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    let mut y_nan = y.clone();
    y_nan[3] = f64::NAN;
    assert!(OrdinalRegressor::builder().build().fit(&x, &y_nan).is_err());
    let short_w = Col::from_fn(5, |_| 1.0);
    assert!(OrdinalRegressor::builder()
        .weights(short_w)
        .build()
        .fit(&x, &y)
        .is_err());
    // a level whose observations all have zero weight
    let w = Col::from_fn(N, |i| if Y_CAT[i] == 1.0 { 0.0 } else { 1.0 });
    assert!(OrdinalRegressor::builder()
        .weights(w)
        .build()
        .fit(&x, &y)
        .is_err());
    // an offset so extreme that the start has zero likelihood
    let off = Col::from_fn(N, |i| if i == 0 { 1e300 } else { 0.0 });
    assert!(OrdinalRegressor::builder()
        .offset(off)
        .build()
        .fit(&x, &y)
        .is_err());
}
