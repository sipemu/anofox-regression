//! Builder-style front end over the penalized GLM engine.

use crate::core::{HasModelInfo, ModelInfo};
use faer::{Col, Mat};

use super::error::{GlmEngineError, GlmEngineResult};
use super::families::{
    fit_binomial, fit_gamma, fit_logistic, fit_negbinomial, fit_poisson, fit_tweedie, GlmResult,
};
use super::types::{
    BinomialOptions, GammaOptions, GlmInferenceResult, GlmPriorOptions, LogisticOptions,
    NegBinomialOptions, PoissonOptions, PriorSpec, TweedieOptions, VcovType,
};
use crate::core::{
    BinomialFamily, BinomialLink, GlmFamily, PoissonFamily, PoissonLink, TweedieFamily,
};
use crate::diagnostics::{check_binary_separation, SeparationCheck};

/// Response family (and link) fitted by [`PenalizedGlmRegressor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenalizedGlmFamily {
    /// Poisson counts; dispersion fixed at 1 (as R's `glm(family = poisson)`), or
    /// the Pearson estimate with [`PenalizedGlmRegressorBuilder::estimate_dispersion`].
    Poisson(PoissonLink),
    /// Binomial proportions / 0-1 outcomes in `[0, 1]`; dispersion fixed at 1.
    Binomial(BinomialLink),
    /// Binary 0/1 outcomes with logit link; additionally reports the training
    /// accuracy at `threshold`.
    Logistic {
        /// Classification threshold on the fitted probability.
        threshold: f64,
    },
    /// Negative Binomial (log link). `theta: None` estimates `theta` by maximum
    /// likelihood as `MASS::glm.nb` does; `Some(theta)` holds it fixed.
    NegativeBinomial {
        /// Fixed shape parameter, or `None` to estimate it.
        theta: Option<f64>,
    },
    /// Gamma with log link; dispersion is the Pearson estimate.
    Gamma,
    /// Tweedie with log link and variance power `1 <= power <= 2`.
    Tweedie {
        /// Variance power.
        power: f64,
    },
}

impl PenalizedGlmFamily {
    /// Poisson with log link.
    pub fn poisson() -> Self {
        Self::Poisson(PoissonLink::Log)
    }
    /// Binomial with logit link.
    pub fn binomial() -> Self {
        Self::Binomial(BinomialLink::Logit)
    }
    /// Logistic regression with threshold 0.5.
    pub fn logistic() -> Self {
        Self::Logistic { threshold: 0.5 }
    }
    /// Negative Binomial with `theta` estimated (`MASS::glm.nb`).
    pub fn negative_binomial() -> Self {
        Self::NegativeBinomial { theta: None }
    }
    /// Gamma with log link.
    pub fn gamma() -> Self {
        Self::Gamma
    }
    /// Tweedie with log link and the given variance power.
    pub fn tweedie(power: f64) -> Self {
        Self::Tweedie { power }
    }

    /// Inverse link applied to a linear predictor.
    fn link_inverse(&self, eta: f64) -> f64 {
        match *self {
            Self::Poisson(link) => PoissonFamily::new(link).link_inverse(eta),
            Self::Binomial(link) => BinomialFamily::new(link).link_inverse(eta),
            Self::Logistic { .. } => BinomialFamily::logistic().link_inverse(eta),
            Self::NegativeBinomial { .. } => eta.exp(),
            Self::Gamma => TweedieFamily::new(2.0, 0.0).link_inverse(eta),
            Self::Tweedie { power } => TweedieFamily::new(power, 0.0).link_inverse(eta),
        }
    }

    fn is_binary(&self) -> bool {
        matches!(self, Self::Binomial(_) | Self::Logistic { .. })
    }
}

impl Default for PenalizedGlmFamily {
    fn default() -> Self {
        Self::poisson()
    }
}

/// Penalized GLM (IRLS with explicit priors) for any supported family.
///
/// Without priors and with `lambda = 0` this is a plain maximum-likelihood GLM
/// that matches R's `glm` / `MASS::glm.nb`; with Gaussian priors it returns the
/// posterior mode (as `arm::bayesglm` with fixed prior scales). See the
/// [module documentation](super) for the details.
#[derive(Debug, Clone)]
pub struct PenalizedGlmRegressor {
    family: PenalizedGlmFamily,
    fit_intercept: bool,
    max_iterations: u32,
    tolerance: f64,
    compute_inference: bool,
    confidence_level: f64,
    lambda: f64,
    priors: Vec<PriorSpec>,
    vcov: VcovType,
    offset: Option<Col<f64>>,
    estimate_dispersion: bool,
}

impl Default for PenalizedGlmRegressor {
    fn default() -> Self {
        Self {
            family: PenalizedGlmFamily::default(),
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: Vec::new(),
            vcov: VcovType::default(),
            offset: None,
            estimate_dispersion: false,
        }
    }
}

impl PenalizedGlmRegressor {
    /// Start a builder with the defaults (Poisson/log, intercept, 100 iterations,
    /// tolerance `1e-8`, no priors, Laplace covariance, no inference).
    pub fn builder() -> PenalizedGlmRegressorBuilder {
        PenalizedGlmRegressorBuilder::default()
    }

    /// Fit the model. `x` is `n x p` (no intercept column), `y` has length `n`.
    /// Rows with a non-finite value in `y`, `x` or the offset are dropped.
    pub fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> GlmEngineResult<FittedPenalizedGlm> {
        let n = x.nrows();
        if y.nrows() != n {
            return Err(GlmEngineError::DimensionMismatch {
                y_len: y.nrows(),
                x_rows: n,
            });
        }
        let p = x.ncols();
        let mut cols: Vec<Vec<f64>> = (0..p)
            .map(|j| (0..n).map(|i| x[(i, j)]).collect())
            .collect();
        let offset_column = match &self.offset {
            Some(off) => {
                if off.nrows() != n {
                    return Err(GlmEngineError::DimensionMismatch {
                        y_len: off.nrows(),
                        x_rows: n,
                    });
                }
                cols.push((0..n).map(|i| off[i]).collect());
                Some(p + 1)
            }
            None => None,
        };
        let yv: Vec<f64> = (0..n).map(|i| y[i]).collect();
        let prior_opts = GlmPriorOptions {
            priors: self.priors.clone(),
            vcov: self.vcov,
        };

        let mut accuracy = None;
        let result = match self.family {
            PenalizedGlmFamily::Poisson(link) => fit_poisson(
                &yv,
                &cols,
                &PoissonOptions {
                    fit_intercept: self.fit_intercept,
                    link,
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    compute_inference: self.compute_inference,
                    confidence_level: self.confidence_level,
                    lambda: self.lambda,
                    prior_opts,
                    offset_column,
                    estimate_dispersion: self.estimate_dispersion,
                },
            )?,
            PenalizedGlmFamily::Binomial(link) => fit_binomial(
                &yv,
                &cols,
                &BinomialOptions {
                    fit_intercept: self.fit_intercept,
                    link,
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    compute_inference: self.compute_inference,
                    confidence_level: self.confidence_level,
                    lambda: self.lambda,
                    prior_opts,
                    offset_column,
                },
            )?,
            PenalizedGlmFamily::Logistic { threshold } => {
                let r = fit_logistic(
                    &yv,
                    &cols,
                    &LogisticOptions {
                        fit_intercept: self.fit_intercept,
                        lambda: self.lambda,
                        threshold,
                        max_iterations: self.max_iterations,
                        tolerance: self.tolerance,
                        compute_inference: self.compute_inference,
                        confidence_level: self.confidence_level,
                        prior_opts,
                        offset_column,
                    },
                )?;
                accuracy = Some(r.accuracy);
                r.fit
            }
            PenalizedGlmFamily::NegativeBinomial { theta } => fit_negbinomial(
                &yv,
                &cols,
                &NegBinomialOptions {
                    fit_intercept: self.fit_intercept,
                    alpha: theta,
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    compute_inference: self.compute_inference,
                    confidence_level: self.confidence_level,
                    lambda: self.lambda,
                    prior_opts,
                    offset_column,
                },
            )?,
            PenalizedGlmFamily::Gamma => fit_gamma(
                &yv,
                &cols,
                &GammaOptions {
                    fit_intercept: self.fit_intercept,
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    compute_inference: self.compute_inference,
                    confidence_level: self.confidence_level,
                    lambda: self.lambda,
                    prior_opts,
                    offset_column,
                },
            )?,
            PenalizedGlmFamily::Tweedie { power } => fit_tweedie(
                &yv,
                &cols,
                &TweedieOptions {
                    fit_intercept: self.fit_intercept,
                    power,
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    compute_inference: self.compute_inference,
                    confidence_level: self.confidence_level,
                    lambda: self.lambda,
                    prior_opts,
                    offset_column,
                },
            )?,
        };

        let separation = if self.family.is_binary() {
            let rows = &result.valid_rows;
            let xs = Mat::from_fn(rows.len(), p, |i, j| x[(rows[i], j)]);
            let ys = Col::from_fn(rows.len(), |i| y[rows[i]]);
            check_binary_separation(&xs, &ys)
        } else {
            SeparationCheck::default()
        };

        let coefficients = Col::from_fn(result.core.coefficients.len(), |j| {
            result.core.coefficients[j]
        });
        Ok(FittedPenalizedGlm {
            family: self.family,
            coefficients,
            result,
            accuracy,
            separation,
        })
    }
}

/// Builder for [`PenalizedGlmRegressor`].
#[derive(Debug, Clone, Default)]
pub struct PenalizedGlmRegressorBuilder {
    inner: PenalizedGlmRegressor,
}

impl PenalizedGlmRegressorBuilder {
    /// Response family and link.
    pub fn family(mut self, family: PenalizedGlmFamily) -> Self {
        self.inner.family = family;
        self
    }
    /// Whether to fit an intercept (default `true`).
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.inner.fit_intercept = fit_intercept;
        self
    }
    /// Maximum IRLS iterations (default 100). Non-convergence is reported through
    /// [`FittedPenalizedGlm::converged`], not as an error.
    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.inner.max_iterations = max_iterations;
        self
    }
    /// Convergence tolerance (default `1e-8`).
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.inner.tolerance = tolerance;
        self
    }
    /// Compute standard errors, z / p values, CIs and the covariance.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.inner.compute_inference = compute;
        self
    }
    /// Confidence level for the intervals (default 0.95).
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.inner.confidence_level = level;
        self
    }
    /// Uniform ridge `lambda` on every non-intercept coefficient (equivalent to a
    /// `N(0, 1/sqrt(lambda))` prior), composed additively with [`Self::priors`].
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.inner.lambda = lambda;
        self
    }
    /// Per-coefficient priors in feature order, optionally with a leading entry
    /// for the intercept (`p` or `p + 1` entries).
    pub fn priors(mut self, priors: Vec<PriorSpec>) -> Self {
        self.inner.priors = priors;
        self
    }
    /// Covariance at the mode (default [`VcovType::Laplace`]; identical to the
    /// others when no prior / lambda is set).
    pub fn vcov_type(mut self, vcov: VcovType) -> Self {
        self.inner.vcov = vcov;
        self
    }
    /// Offset added to the linear predictor (on the link scale, e.g. `log(exposure)`).
    pub fn offset(mut self, offset: Col<f64>) -> Self {
        self.inner.offset = Some(offset);
        self
    }
    /// Poisson only: scale the covariance by the Pearson dispersion
    /// `sum(pearson^2) / df_resid`, i.e. R's `quasipoisson` standard errors.
    /// Default `false` (dispersion 1, as R's `glm(family = poisson)`). Ignored by
    /// the other families, whose dispersion rule is fixed by the family.
    pub fn estimate_dispersion(mut self, estimate: bool) -> Self {
        self.inner.estimate_dispersion = estimate;
        self
    }
    /// Finish the builder.
    pub fn build(self) -> PenalizedGlmRegressor {
        self.inner
    }
}

/// A fitted [`PenalizedGlmRegressor`].
#[derive(Debug, Clone)]
pub struct FittedPenalizedGlm {
    family: PenalizedGlmFamily,
    coefficients: Col<f64>,
    result: GlmResult,
    accuracy: Option<f64>,
    separation: SeparationCheck,
}

impl PenalizedGlmFamily {
    /// What a fit of this family is: model type `"glm"`, the R family name and
    /// the link. `Logistic` reports the `"binomial"` family with the logit
    /// link, and a Tweedie power of 1 or 2 reports `"poisson"` / `"gamma"` (see
    /// [`TweedieFamily::family_name`]).
    pub fn model_info(&self) -> ModelInfo {
        let (family, link) = match *self {
            PenalizedGlmFamily::Poisson(link) => ("poisson", link.as_str()),
            PenalizedGlmFamily::Binomial(link) => ("binomial", link.as_str()),
            PenalizedGlmFamily::Logistic { .. } => ("binomial", "logit"),
            PenalizedGlmFamily::NegativeBinomial { .. } => ("negative_binomial", "log"),
            PenalizedGlmFamily::Gamma => ("gamma", "log"),
            PenalizedGlmFamily::Tweedie { power } => (
                TweedieFamily {
                    var_power: power,
                    link_power: 0.0,
                }
                .family_name(),
                "log",
            ),
        };
        ModelInfo::new("glm", Some(family), link)
    }
}

impl HasModelInfo for FittedPenalizedGlm {
    /// See [`PenalizedGlmFamily::model_info`].
    fn model_info(&self) -> ModelInfo {
        self.family.model_info()
    }
}

impl FittedPenalizedGlm {
    /// Feature coefficients (`NaN` for columns dropped as constant / aliased).
    pub fn coefficients(&self) -> &Col<f64> {
        &self.coefficients
    }
    /// Intercept, if fitted.
    pub fn intercept(&self) -> Option<f64> {
        self.result.core.intercept
    }
    /// Family the model was fitted with.
    pub fn family(&self) -> PenalizedGlmFamily {
        self.family
    }
    /// Residual deviance.
    pub fn deviance(&self) -> f64 {
        self.result.core.residual_deviance
    }
    /// Null deviance.
    pub fn null_deviance(&self) -> f64 {
        self.result.core.null_deviance
    }
    /// Deviance pseudo-R², `1 - deviance / null_deviance`.
    pub fn pseudo_r_squared(&self) -> f64 {
        self.result.core.pseudo_r_squared
    }
    /// Log-likelihood at the fitted means.
    pub fn log_likelihood(&self) -> f64 {
        self.result.log_likelihood
    }
    /// AIC.
    pub fn aic(&self) -> f64 {
        self.result.core.aic
    }
    /// BIC.
    pub fn bic(&self) -> f64 {
        self.result.bic
    }
    /// Dispersion: Pearson estimate (Gamma, Tweedie, Poisson with
    /// `estimate_dispersion(true)`), 1 (Poisson, Binomial, Logistic), or `theta`
    /// (Negative Binomial).
    pub fn dispersion(&self) -> f64 {
        self.result.core.dispersion.unwrap_or(1.0)
    }
    /// Negative Binomial `theta` (fixed or estimated); `None` for other families.
    pub fn theta(&self) -> Option<f64> {
        match self.family {
            PenalizedGlmFamily::NegativeBinomial { .. } => self.result.core.dispersion,
            _ => None,
        }
    }
    /// IRLS iterations of the final fit.
    pub fn iterations(&self) -> u32 {
        self.result.core.iterations
    }
    /// Whether IRLS converged (the last iterate is returned either way).
    pub fn converged(&self) -> bool {
        self.result.core.converged
    }
    /// Number of rows used.
    pub fn n_observations(&self) -> usize {
        self.result.core.n_observations
    }
    /// Training accuracy (Logistic family only).
    pub fn accuracy(&self) -> Option<f64> {
        self.accuracy
    }
    /// Inference, when requested.
    pub fn inference(&self) -> Option<&GlmInferenceResult> {
        self.result.inference.as_ref()
    }
    /// Feature standard errors, when inference was requested.
    pub fn std_errors(&self) -> Option<&[f64]> {
        self.inference().map(|i| i.std_errors.as_slice())
    }
    /// Full parameter covariance in fitted order (see
    /// [`GlmInferenceResult::matrix_parameters`]), when inference was requested.
    pub fn vcov(&self) -> Option<&Mat<f64>> {
        self.inference().and_then(|i| i.vcov.as_ref())
    }
    /// Separation check of the training data (Binomial / Logistic only; empty for
    /// the other families).
    pub fn separation(&self) -> &SeparationCheck {
        &self.separation
    }
    /// The full result as the per-family functions return it.
    pub fn result(&self) -> &GlmResult {
        &self.result
    }
    /// Fitted means over the rows used.
    pub fn fitted_values(&self) -> &[f64] {
        &self.result.fitted_values
    }

    /// Linear predictor `intercept + x * beta` (dropped columns contribute 0).
    pub fn predict_linear(&self, x: &Mat<f64>) -> Col<f64> {
        let b0 = self.intercept().unwrap_or(0.0);
        Col::from_fn(x.nrows(), |i| {
            b0 + (0..x.ncols().min(self.coefficients.nrows()))
                .map(|j| {
                    let b = self.coefficients[j];
                    if b.is_nan() {
                        0.0
                    } else {
                        b * x[(i, j)]
                    }
                })
                .sum::<f64>()
        })
    }

    /// Predicted means on the response scale.
    pub fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let eta = self.predict_linear(x);
        Col::from_fn(eta.nrows(), |i| self.family.link_inverse(eta[i]))
    }

    /// Predicted means with an offset added to the linear predictor.
    pub fn predict_with_offset(&self, x: &Mat<f64>, offset: &Col<f64>) -> Col<f64> {
        let eta = self.predict_linear(x);
        Col::from_fn(eta.nrows(), |i| {
            self.family.link_inverse(eta[i] + offset[i])
        })
    }
}
