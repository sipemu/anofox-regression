//! Builder-style front end over the AFT model.

use crate::core::{HasModelInfo, ModelInfo};
use faer::{Col, Mat};

use super::distribution::AftDistribution;
use super::model::{fit_aft, AftFitResult, AftInference, AftOptions, AftResult};
use crate::solvers::penalized_glm::{GlmEngineError, GlmEngineResult, PriorSpec, VcovType};

/// Accelerated failure time regression with right censoring (`survival::survreg`).
#[derive(Debug, Clone, Default)]
pub struct AftRegressor {
    options: AftOptions,
}

impl AftRegressor {
    /// Start a builder with the defaults (Weibull, intercept, 100 iterations,
    /// tolerance `1e-9`, no priors, Laplace covariance, no inference).
    pub fn builder() -> AftRegressorBuilder {
        AftRegressorBuilder::default()
    }

    /// The options this regressor fits with.
    pub fn options(&self) -> &AftOptions {
        &self.options
    }

    /// Fit to survival times `time` (> 0) with `event[i] == true` for an observed
    /// event and `false` for a right-censored time. Rows with a non-finite value
    /// are dropped.
    pub fn fit(&self, x: &Mat<f64>, time: &Col<f64>, event: &[bool]) -> GlmEngineResult<FittedAft> {
        let n = x.nrows();
        if time.nrows() != n {
            return Err(GlmEngineError::DimensionMismatch {
                y_len: time.nrows(),
                x_rows: n,
            });
        }
        if event.len() != n {
            return Err(GlmEngineError::DimensionMismatch {
                y_len: event.len(),
                x_rows: n,
            });
        }
        let cols: Vec<Vec<f64>> = (0..x.ncols())
            .map(|j| (0..n).map(|i| x[(i, j)]).collect())
            .collect();
        let t: Vec<f64> = (0..n).map(|i| time[i]).collect();
        let ev: Vec<f64> = event.iter().map(|&e| f64::from(u8::from(e))).collect();
        let result = fit_aft(&t, &cols, &ev, &self.options)?;
        let coefficients = Col::from_fn(result.core.coefficients.len(), |j| {
            result.core.coefficients[j]
        });
        Ok(FittedAft {
            distribution: self.options.dist,
            coefficients,
            result,
        })
    }
}

/// Builder for [`AftRegressor`].
#[derive(Debug, Clone, Default)]
pub struct AftRegressorBuilder {
    inner: AftRegressor,
}

impl AftRegressorBuilder {
    /// Error distribution (default Weibull).
    pub fn distribution(mut self, dist: AftDistribution) -> Self {
        self.inner.options.dist = dist;
        self
    }
    /// Whether to fit an intercept (default `true`).
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.inner.options.fit_intercept = fit_intercept;
        self
    }
    /// Maximum Newton iterations (default 100).
    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.inner.options.max_iterations = max_iterations;
        self
    }
    /// Convergence tolerance on the log-likelihood (default `1e-9`).
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.inner.options.tolerance = tolerance;
        self
    }
    /// Compute standard errors, z / p values, CIs and the covariance.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.inner.options.compute_inference = compute;
        self
    }
    /// Confidence level (default 0.95).
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.inner.options.confidence_level = level;
        self
    }
    /// Per-coefficient priors in feature order, optionally with a leading entry
    /// for the intercept.
    pub fn priors(mut self, priors: Vec<PriorSpec>) -> Self {
        self.inner.options.priors = priors;
        self
    }
    /// Covariance at the mode (default [`VcovType::Laplace`]).
    pub fn vcov_type(mut self, vcov: VcovType) -> Self {
        self.inner.options.vcov = vcov;
        self
    }
    /// Finish the builder.
    pub fn build(self) -> AftRegressor {
        self.inner
    }
}

/// A fitted [`AftRegressor`].
#[derive(Debug, Clone)]
pub struct FittedAft {
    distribution: AftDistribution,
    coefficients: Col<f64>,
    result: AftResult,
}

impl HasModelInfo for FittedAft {
    /// The family is the distribution of the survival time (`"weibull"`,
    /// `"lognormal"`, `"loglogistic"`, `"exponential"`); AFT models are linear
    /// in `log T`, so the link is `"log"`.
    fn model_info(&self) -> ModelInfo {
        super::model::aft_model_info(self.distribution)
    }
}

impl FittedAft {
    /// Feature coefficients on the log-time scale.
    pub fn coefficients(&self) -> &Col<f64> {
        &self.coefficients
    }
    /// Per-observation score contributions (`n × k`) at the estimate, in the
    /// parameter order of the covariance: intercept (if fitted), the
    /// coefficients, then `log σ` when the scale is estimated. `x`, `time`
    /// and `event` are the data of the fit.
    pub fn score_contributions(
        &self,
        x: &Mat<f64>,
        time: &Col<f64>,
        event: &[bool],
    ) -> Result<Mat<f64>, crate::inference::ClusterError> {
        let n = x.nrows();
        if time.nrows() != n || event.len() != n || x.ncols() != self.coefficients.nrows() {
            return Err(crate::inference::ClusterError::Dimension(format!(
                "x is {}x{}, time has {} and event {} entries",
                n,
                x.ncols(),
                time.nrows(),
                event.len()
            )));
        }
        let off = usize::from(self.intercept().is_some());
        let p = x.ncols();
        let fit_scale = self.distribution != AftDistribution::Exponential;
        let k = off + p + usize::from(fit_scale);
        let sigma = self.scale();
        let eta = self.predict_linear(x);
        let mut s = Mat::<f64>::zeros(n, k);
        for i in 0..n {
            let z = (time[i].ln() - eta[i]) / sigma;
            let u = if event[i] {
                self.distribution.d_log_density(z)
            } else {
                self.distribution.d_log_survival(z)
            };
            if off == 1 {
                s[(i, 0)] = -u / sigma;
            }
            for j in 0..p {
                s[(i, off + j)] = -u * x[(i, j)] / sigma;
            }
            if fit_scale {
                s[(i, k - 1)] = -u * z - f64::from(u8::from(event[i]));
            }
        }
        Ok(s)
    }

    /// Cluster-robust covariance of all parameters (order of
    /// [`Self::score_contributions`]), `A (Σ_g s_g s_gᵀ) A` with `A` the
    /// fitted covariance; [`crate::inference::ClusterOptions::robcov`]
    /// reproduces `survreg(..., cluster = id)`. Needs a fit with inference.
    pub fn cluster_vcov(
        &self,
        x: &Mat<f64>,
        time: &Col<f64>,
        event: &[bool],
        clusters: &[&[usize]],
        options: &crate::inference::ClusterOptions,
    ) -> Result<Mat<f64>, crate::inference::ClusterError> {
        let bread = self.inference().and_then(|i| i.vcov.as_ref()).ok_or(
            crate::inference::ClusterError::Unsupported(
                "the fit has no covariance; fit with compute_inference(true)",
            ),
        )?;
        let scores = self.score_contributions(x, time, event)?;
        crate::inference::cluster_sandwich(bread, &scores, clusters, options)
    }

    /// Standardized residuals `zᵢ = (log tᵢ − ηᵢ)/σ` (RMS §18.3); for a
    /// correct model their censored distribution is the error
    /// distribution, which a Kaplan–Meier curve of `(z, event)` checks.
    pub fn standardized_residuals(&self, x: &Mat<f64>, time: &Col<f64>) -> Col<f64> {
        let eta = self.predict_linear(x);
        let sigma = self.scale();
        Col::from_fn(time.nrows(), |i| (time[i].ln() - eta[i]) / sigma)
    }

    /// Cox–Snell residuals `−log S(tᵢ | xᵢ)`: unit exponential (censored)
    /// for a correct model.
    pub fn cox_snell_residuals(&self, x: &Mat<f64>, time: &Col<f64>) -> Col<f64> {
        let z = self.standardized_residuals(x, time);
        Col::from_fn(z.nrows(), |i| -self.distribution.log_survival(z[i]))
    }

    /// Intercept, if fitted.
    pub fn intercept(&self) -> Option<f64> {
        self.result.core.intercept
    }
    /// Scale `sigma` (1 for the exponential distribution).
    pub fn scale(&self) -> f64 {
        self.result.core.scale
    }
    /// Error distribution.
    pub fn distribution(&self) -> AftDistribution {
        self.distribution
    }
    /// Log-likelihood (on the time scale, as `survreg` reports it).
    pub fn log_likelihood(&self) -> f64 {
        self.result.core.log_likelihood
    }
    /// Log-likelihood of the intercept-only model.
    pub fn null_log_likelihood(&self) -> f64 {
        self.result.core.null_log_likelihood
    }
    /// AIC.
    pub fn aic(&self) -> f64 {
        self.result.core.aic
    }
    /// BIC.
    pub fn bic(&self) -> f64 {
        self.result.core.bic
    }
    /// Whether the Newton iterations converged.
    pub fn converged(&self) -> bool {
        self.result.core.converged
    }
    /// Core fit summary (counts of events / censored rows, iterations, ...).
    pub fn fit_result(&self) -> &AftFitResult {
        &self.result.core
    }
    /// Inference, when requested.
    pub fn inference(&self) -> Option<&AftInference> {
        self.result.inference.as_ref()
    }
    /// The full result as [`fit_aft`] returns it.
    pub fn result(&self) -> &AftResult {
        &self.result
    }

    /// Linear predictor `eta = intercept + x * beta` (the location of log T).
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

    /// Predicted `p`-quantile of the survival time (`p = 0.5` is the median).
    pub fn predict_quantile(&self, x: &Mat<f64>, p: f64) -> Col<f64> {
        let eta = self.predict_linear(x);
        let s = self.scale();
        Col::from_fn(eta.nrows(), |i| {
            self.distribution.quantile_time(p, eta[i], s)
        })
    }

    /// Predicted median survival time.
    pub fn predict_median(&self, x: &Mat<f64>) -> Col<f64> {
        self.predict_quantile(x, 0.5)
    }

    /// Predicted survival probability `S(t | x)`.
    pub fn predict_survival(&self, x: &Mat<f64>, t: f64) -> Col<f64> {
        let eta = self.predict_linear(x);
        let s = self.scale();
        Col::from_fn(eta.nrows(), |i| {
            self.distribution.survival_time(t, eta[i], s)
        })
    }

    /// Predicted failure probability `F(t | x) = 1 - S(t | x)`.
    pub fn predict_cdf(&self, x: &Mat<f64>, t: f64) -> Col<f64> {
        let eta = self.predict_linear(x);
        let s = self.scale();
        Col::from_fn(eta.nrows(), |i| self.distribution.cdf_time(t, eta[i], s))
    }
}
