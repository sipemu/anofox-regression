//! `HasModelInfo` reports the estimator, family and link of every fitted model.

use anofox_regression::prelude::*;
use anofox_regression::solvers::{
    ArdRegression, BayesianRidge, LarsRegressor, LmDynamicRegressor, PassiveAggressiveRegressor,
    RansacRegressor, TheilSenRegressor,
};
use faer::{Col, Mat};

const N: usize = 40;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| {
        let t = i as f64 / N as f64;
        if j == 0 {
            t
        } else {
            ((i * 7 % 11) as f64) / 11.0
        }
    })
}

/// Positive continuous response with noise.
fn y_pos() -> Col<f64> {
    let x = x();
    Col::from_fn(N, |i| {
        1.0 + 2.0 * x[(i, 0)] + 0.5 * x[(i, 1)] + 0.1 * ((i * 13 % 7) as f64)
    })
}

/// Count response.
fn y_count() -> Col<f64> {
    Col::from_fn(N, |i| ((i * 5 % 9) as f64) + (i / 10) as f64)
}

/// Binary response without perfect separation.
fn y_binary() -> Col<f64> {
    Col::from_fn(N, |i| if (i * 7 % 5) < 2 + i / 20 { 1.0 } else { 0.0 })
}

fn groups() -> Vec<usize> {
    (0..N).map(|i| i % 4).collect()
}

fn info(model_type: &'static str, family: Option<&'static str>, link: &'static str) -> ModelInfo {
    ModelInfo::new(model_type, family, link)
}

#[test]
fn linear_models() {
    let (x, y) = (x(), y_pos());
    let gaussian = |m: &'static str| info(m, Some("gaussian"), "identity");

    let f = OlsRegressor::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), gaussian("ols"));
    let f = WlsRegressor::builder()
        .weights(Col::from_fn(N, |i| 1.0 + (i % 3) as f64))
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), gaussian("wls"));
    let f = RidgeRegressor::builder()
        .lambda(0.1)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), gaussian("ridge"));
    let f = ElasticNetRegressor::builder()
        .lambda(0.01)
        .alpha(0.5)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), gaussian("elastic_net"));
    let f = LarsRegressor::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), gaussian("lars"));
    let f = BayesianRidge::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), gaussian("bayesian_ridge"));
    let f = ArdRegression::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), gaussian("ard"));
    let f = PlsRegressor::builder()
        .n_components(1)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), gaussian("pls"));
    let f = RlsRegressor::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), gaussian("rls"));
    let x1 = Mat::from_fn(N, 1, |i, _| x[(i, 0)]);
    let f = PSplineRegressor::builder().build().fit(&x1, &y).unwrap();
    assert_eq!(f.model_info(), gaussian("pspline"));
    let f = LmDynamicRegressor::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), gaussian("lm_dynamic"));
}

#[test]
fn bls_reports_nnls_for_non_negative_bounds() {
    let (x, y) = (x(), y_pos());
    let f = BlsRegressor::nnls().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), info("nnls", Some("gaussian"), "identity"));
    let f = BlsRegressor::builder()
        .lower_bound_all(0.0)
        .upper_bound_all(10.0)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), info("bls", Some("gaussian"), "identity"));
}

#[test]
fn non_likelihood_estimators_have_no_family() {
    let (x, y) = (x(), y_pos());
    let linear = |m: &'static str| info(m, None, "identity");

    let f = HuberRegressor::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), linear("huber"));
    let f = RansacRegressor::builder()
        .random_state(1)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), linear("ransac"));
    let f = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f.model_info(), linear("theil_sen"));
    let f = QuantileRegressor::builder()
        .tau(0.5)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), linear("quantile"));
    let f = IsotonicRegressor::builder()
        .build()
        .fit_1d(&Col::from_fn(N, |i| i as f64), &y)
        .unwrap();
    assert_eq!(f.model_info(), linear("isotonic"));
    let f = PassiveAggressiveRegressor::builder()
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.model_info(), linear("passive_aggressive"));
}

#[test]
fn glm_families_and_links() {
    let x = x();
    let (yb, yc, yp) = (y_binary(), y_count(), y_pos());

    let f = BinomialRegressor::logistic().build().fit(&x, &yb).unwrap();
    assert_eq!(f.model_info(), info("binomial", Some("binomial"), "logit"));
    let f = BinomialRegressor::probit().build().fit(&x, &yb).unwrap();
    assert_eq!(f.model_info(), info("binomial", Some("binomial"), "probit"));
    let f = BinomialRegressor::cloglog().build().fit(&x, &yb).unwrap();
    assert_eq!(
        f.model_info(),
        info("binomial", Some("binomial"), "cloglog")
    );

    let f = LogisticRegression::builder().build().fit(&x, &yb).unwrap();
    assert_eq!(f.model_info(), info("logistic", Some("binomial"), "logit"));

    let f = PoissonRegressor::log().build().fit(&x, &yc).unwrap();
    assert_eq!(f.model_info(), info("poisson", Some("poisson"), "log"));
    let f = PoissonRegressor::sqrt().build().fit(&x, &yc).unwrap();
    assert_eq!(f.model_info(), info("poisson", Some("poisson"), "sqrt"));
    let f = PoissonRegressor::identity().build().fit(&x, &yp).unwrap();
    assert_eq!(f.model_info(), info("poisson", Some("poisson"), "identity"));

    let f = NegativeBinomialRegressor::builder()
        .build()
        .fit(&x, &yc)
        .unwrap();
    assert_eq!(
        f.model_info(),
        info("negative_binomial", Some("negative_binomial"), "log")
    );

    let f = GammaRegressor::builder().build().fit(&x, &yp).unwrap();
    assert_eq!(f.model_info(), info("gamma", Some("gamma"), "log"));
    let f = GammaRegressor::builder()
        .link_power(-1.0)
        .build()
        .fit(&x, &yp)
        .unwrap();
    assert_eq!(f.model_info(), info("gamma", Some("gamma"), "inverse"));

    let f = TweedieRegressor::builder()
        .var_power(1.5)
        .link_power(0.0)
        .build()
        .fit(&x, &yp)
        .unwrap();
    assert_eq!(f.model_info(), info("tweedie", Some("tweedie"), "log"));
    let f = TweedieRegressor::gamma().build().fit(&x, &yp).unwrap();
    assert_eq!(f.model_info(), info("tweedie", Some("gamma"), "log"));
    let f = TweedieRegressor::gaussian().build().fit(&x, &yp).unwrap();
    assert_eq!(
        f.model_info(),
        info("tweedie", Some("gaussian"), "identity")
    );
    let f = TweedieRegressor::inverse_gaussian()
        .build()
        .fit(&x, &yp)
        .unwrap();
    assert_eq!(
        f.model_info(),
        info("tweedie", Some("inverse_gaussian"), "log")
    );
}

#[test]
fn penalized_glm_families() {
    let x = x();
    let cases = [
        (
            PenalizedGlmFamily::Poisson(PoissonLink::Log),
            y_count(),
            "poisson",
            "log",
        ),
        (
            PenalizedGlmFamily::Binomial(BinomialLink::Probit),
            y_binary(),
            "binomial",
            "probit",
        ),
        (
            PenalizedGlmFamily::Logistic { threshold: 0.5 },
            y_binary(),
            "binomial",
            "logit",
        ),
        (
            PenalizedGlmFamily::NegativeBinomial { theta: Some(2.0) },
            y_count(),
            "negative_binomial",
            "log",
        ),
        (PenalizedGlmFamily::Gamma, y_pos(), "gamma", "log"),
        (
            PenalizedGlmFamily::Tweedie { power: 1.5 },
            y_pos(),
            "tweedie",
            "log",
        ),
    ];
    for (family, y, name, link) in cases {
        let f = PenalizedGlmRegressor::builder()
            .family(family)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert_eq!(f.model_info(), info("penalized_glm", Some(name), link));
    }
}

#[test]
fn alm_distribution_and_link() {
    let x = x();
    let f = AlmRegressor::builder().build().fit(&x, &y_pos()).unwrap();
    assert_eq!(f.model_info(), info("alm", Some("normal"), "identity"));
    let f = AlmRegressor::builder()
        .distribution(AlmDistribution::Laplace)
        .build()
        .fit(&x, &y_pos())
        .unwrap();
    assert_eq!(f.model_info(), info("alm", Some("laplace"), "identity"));
    let f = AlmRegressor::builder()
        .distribution(AlmDistribution::Poisson)
        .build()
        .fit(&x, &y_count())
        .unwrap();
    assert_eq!(f.model_info(), info("alm", Some("poisson"), "log"));
    let f = AlmRegressor::builder()
        .distribution(AlmDistribution::LogNormal)
        .build()
        .fit(&x, &y_pos())
        .unwrap();
    assert_eq!(f.model_info().family, Some("log_normal"));
}

#[test]
fn aft_distributions() {
    let x = x();
    let time = y_pos();
    let event: Vec<bool> = (0..N).map(|i| i % 4 != 0).collect();
    for (dist, name) in [
        (AftDistribution::Weibull, "weibull"),
        (AftDistribution::LogNormal, "lognormal"),
        (AftDistribution::LogLogistic, "loglogistic"),
        (AftDistribution::Exponential, "exponential"),
    ] {
        let f = AftRegressor::builder()
            .distribution(dist)
            .build()
            .fit(&x, &time, &event)
            .unwrap();
        assert_eq!(f.model_info(), info("aft", Some(name), "log"));
    }
}

#[test]
fn glmm_families() {
    let x = x();
    let g = groups();
    let f = GlmmRegressor::gaussian()
        .build()
        .fit(&x, &y_pos(), &g)
        .unwrap();
    assert_eq!(f.model_info(), info("glmm", Some("gaussian"), "identity"));
    let f = GlmmRegressor::poisson()
        .build()
        .fit(&x, &y_count(), &g)
        .unwrap();
    assert_eq!(f.model_info(), info("glmm", Some("poisson"), "log"));
    let f = GlmmRegressor::binomial()
        .build()
        .fit(&x, &y_binary(), &g)
        .unwrap();
    assert_eq!(f.model_info(), info("glmm", Some("binomial"), "logit"));
    let f = GlmmRegressor::negative_binomial()
        .nb_theta(2.0)
        .build()
        .fit(&x, &y_count(), &g)
        .unwrap();
    assert_eq!(
        f.model_info(),
        info("glmm", Some("negative_binomial"), "log")
    );
    let f = GlmmRegressor::gamma()
        .build()
        .fit(&x, &y_pos(), &g)
        .unwrap();
    assert_eq!(f.model_info(), info("glmm", Some("gamma"), "log"));
    let f = GlmmRegressor::tweedie(1.5)
        .build()
        .fit(&x, &y_pos(), &g)
        .unwrap();
    assert_eq!(f.model_info(), info("glmm", Some("tweedie"), "log"));
}
