//! A uniform description of a fitted model: what was fitted, under which error
//! family and through which link.
//!
//! Every fitted model in the crate implements [`HasModelInfo`], so code that
//! handles fits generically (reporting, serialisation, SQL bindings) can label a
//! result without matching on its concrete type.
//!
//! The strings are stable, lower-case `snake_case` identifiers:
//!
//! * `model_type` names the estimator, e.g. `"ols"`, `"ridge"`, `"poisson"`,
//!   `"glmm"`.
//! * `family` names the error distribution the estimator assumes, using R's
//!   family names where one exists (`"gaussian"`, `"binomial"`, `"poisson"`,
//!   `"negative_binomial"`, `"gamma"`, `"tweedie"`, …). It is `None` for
//!   estimators that are not likelihood based (Huber, RANSAC, Theil–Sen,
//!   quantile, isotonic, passive-aggressive, …).
//! * `link` names the link between the linear predictor and the mean
//!   (`"identity"`, `"log"`, `"logit"`, `"probit"`, `"cloglog"`, `"inverse"`,
//!   `"sqrt"`, …). Linear models report `"identity"`.

/// Static description of a fitted model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelInfo {
    /// Estimator name, e.g. `"ols"`, `"elastic_net"`, `"binomial"`.
    pub model_type: &'static str,
    /// Error family (R naming), `None` for estimators without a likelihood.
    pub family: Option<&'static str>,
    /// Link function between the linear predictor and the mean.
    pub link: &'static str,
}

impl ModelInfo {
    /// Construct a description.
    pub const fn new(
        model_type: &'static str,
        family: Option<&'static str>,
        link: &'static str,
    ) -> Self {
        Self {
            model_type,
            family,
            link,
        }
    }

    /// A Gaussian model with identity link (least squares and its penalised
    /// variants).
    pub const fn gaussian(model_type: &'static str) -> Self {
        Self::new(model_type, Some("gaussian"), "identity")
    }

    /// A non-likelihood estimator of a linear mean (identity link, no family).
    pub const fn linear(model_type: &'static str) -> Self {
        Self::new(model_type, None, "identity")
    }
}

/// Implemented by every fitted model: describes what was fitted.
pub trait HasModelInfo {
    /// Model type, family and link of this fit.
    fn model_info(&self) -> ModelInfo;
}
