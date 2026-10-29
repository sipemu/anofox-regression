//! ALM fits against greybox's own optimum (#144).
//!
//! greybox's `alm()` maximises the likelihood directly, with the scale profiled
//! out by its `scalerInternal()` (denominator n). The reference values are
//! `alm()` converged to that optimum (nloptr `xtol_rel = ftol_rel = 1e-14`);
//! generator `tests/r_scripts/generate_alm_greybox_parity.R`. Not covered:
//! the Student t and Beta, which are different models from greybox's (checked
//! against their own MLEs in `r_validation_alm*.rs`), the S (multimodal: we
//! reach a higher likelihood) and the binomial (`r_validation_alm_extended.rs`).

use anofox_regression::solvers::{AlmDistribution as D, AlmRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

include!("fixtures/alm_greybox_parity.rs");

fn distribution(name: &str) -> D {
    match name {
        "Normal" => D::Normal,
        "Laplace" => D::Laplace,
        "LogNormal" => D::LogNormal,
        "Poisson" => D::Poisson,
        "Gamma" => D::Gamma,
        "Logistic" => D::Logistic,
        "Geometric" => D::Geometric,
        "Exponential" => D::Exponential,
        "FoldedNormal" => D::FoldedNormal,
        "RectifiedNormal" => D::RectifiedNormal,
        "LogitNormal" => D::LogitNormal,
        "GeneralisedNormal" => D::GeneralisedNormal,
        "LogLaplace" => D::LogLaplace,
        "LogGeneralisedNormal" => D::LogGeneralisedNormal,
        "CumulativeLogistic" => D::CumulativeLogistic,
        "CumulativeNormal" => D::CumulativeNormal,
        "NegativeBinomial" => D::NegativeBinomial,
        "InverseGaussian" => D::InverseGaussian,
        "AsymmetricLaplace" => D::AsymmetricLaplace,
        "BoxCoxNormal" => D::BoxCoxNormal,
        other => panic!("no ALM distribution {other}"),
    }
}

#[test]
fn alm_reaches_the_greybox_optimum() {
    let mut failures = Vec::new();
    for case in CASES {
        let n = case.x.len();
        let x = Mat::from_fn(n, 1, |i, _| case.x[i]);
        let y = Col::from_fn(n, |i| case.y[i]);
        let mut builder = AlmRegressor::builder()
            .distribution(distribution(case.distribution))
            .with_intercept(true);
        if let Some(extra) = case.extra {
            builder = builder.extra_parameter(extra);
        }
        let fitted = builder.build().fit(&x, &y).expect(case.name);
        let r = fitted.result();
        let (b0, b1, ll) = (r.intercept.unwrap(), r.coefficients[0], r.log_likelihood);
        // the folded normal depends on |μ| only
        let sign = if case.distribution == "FoldedNormal" {
            b0.signum() * case.intercept.signum()
        } else {
            1.0
        };
        // The log-likelihood to 1e-5; the coefficients to 1e-4, because the
        // Laplace likelihood is flat around the least-absolute-deviation fit
        // (2e-5 apart at a log-likelihood difference of 2e-6); the other
        // coefficients agree to 1e-5.
        let ok = (sign * b0 - case.intercept).abs() < 1e-4
            && (sign * b1 - case.slope).abs() < 1e-4
            && (ll - case.log_likelihood).abs() < 1e-5;
        if !ok {
            failures.push(format!(
                "{}: ({b0}, {b1}, ll {ll}) vs greybox ({}, {}, ll {})",
                case.name, case.intercept, case.slope, case.log_likelihood
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
