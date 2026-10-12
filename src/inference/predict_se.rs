//! Predictions with standard errors and intervals, one API across models (#80).
//!
//! [`PredictWithSe::predict_se`] returns, per row of new data, the prediction,
//! its standard error and the lower / upper interval bound, as R's
//! `predict.lm(se.fit = TRUE, interval = ...)` and
//! `predict.glm(se.fit = TRUE, type = "link" | "response")`.
//!
//! # Linear models (OLS, WLS, Ridge)
//!
//! * `se_fit` is the standard error of the fitted mean, `sqrt(x0' V x0)`, for
//!   both interval types (R's `se.fit`).
//! * A confidence interval is `fit ± t(df) · se_fit`; a prediction interval is
//!   `fit ± t(df) · sqrt(se_fit² + σ²)`, with `σ²` the residual variance (for WLS,
//!   a new observation of weight 1, as R's `predict.lm` default `pred.var`).
//! * The link and response scales coincide (identity link).
//!
//! # GLMs
//!
//! * The standard error on the link scale is `sqrt(x0' V x0)` with `V` the
//!   coefficient covariance (including the dispersion where it is estimated). On
//!   the response scale it is the delta-method value `|dμ/dη| · se_link`, which
//!   is exactly what `predict.glm(se.fit = TRUE, type = "response")` reports.
//! * Intervals are built on the link scale with the normal quantile,
//!   `η ± z · se_link`, and on the response scale they are that interval mapped
//!   through the inverse link (so they stay inside the range of the mean and are
//!   not symmetric around `fit`). This is the usual construction (`ggeffects`,
//!   `emmeans`' `type = "response"`); R's `predict.glm` itself returns no interval.
//! * Prediction intervals are not defined for GLMs and are rejected with
//!   [`PredictSeError::PredictionIntervalUnsupported`].
//! * An offset adds to `η` without contributing to the standard error.

use crate::core::{IntervalType, PredictionType};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use thiserror::Error;

/// Scale, interval type and level of [`PredictWithSe::predict_se`].
///
/// The default is the response scale, a confidence interval and level 0.95.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictSeOptions {
    /// `Link` (linear predictor η) or `Response` (mean μ). Identical for linear models.
    pub scale: PredictionType,
    /// `Confidence` (mean response) or `Prediction` (new observation; linear models only).
    pub interval: IntervalType,
    /// Confidence level in `(0, 1)`.
    pub level: f64,
}

impl Default for PredictSeOptions {
    fn default() -> Self {
        Self {
            scale: PredictionType::Response,
            interval: IntervalType::Confidence,
            level: 0.95,
        }
    }
}

impl PredictSeOptions {
    /// Response scale, confidence interval, level 0.95.
    pub fn new() -> Self {
        Self::default()
    }
    /// Predict on the link scale (R `type = "link"`).
    pub fn link(mut self) -> Self {
        self.scale = PredictionType::Link;
        self
    }
    /// Predict on the response scale (R `type = "response"`).
    pub fn response(mut self) -> Self {
        self.scale = PredictionType::Response;
        self
    }
    /// Confidence interval for the mean (R `interval = "confidence"`).
    pub fn confidence(mut self) -> Self {
        self.interval = IntervalType::Confidence;
        self
    }
    /// Prediction interval for a new observation (R `interval = "prediction"`).
    pub fn prediction(mut self) -> Self {
        self.interval = IntervalType::Prediction;
        self
    }
    /// Confidence level in `(0, 1)`.
    pub fn level(mut self, level: f64) -> Self {
        self.level = level;
        self
    }

    fn check_level(&self) -> Result<(), PredictSeError> {
        if self.level > 0.0 && self.level < 1.0 {
            Ok(())
        } else {
            Err(PredictSeError::InvalidLevel(self.level))
        }
    }
}

/// Why [`PredictWithSe::predict_se`] could not produce a result.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PredictSeError {
    /// The level is not in `(0, 1)`.
    #[error("level must be in (0, 1), got {0}")]
    InvalidLevel(f64),
    /// A prediction interval was requested for a GLM.
    #[error("prediction intervals are only defined for linear models; use a confidence interval")]
    PredictionIntervalUnsupported,
    /// The fit stores no coefficient covariance.
    #[error("the fit has no coefficient covariance: {0}")]
    MissingCovariance(&'static str),
    /// The new data do not match the fit.
    #[error("new data have {got} columns, the fit expects {expected}")]
    DimensionMismatch {
        /// Columns the fit expects.
        expected: usize,
        /// Columns given.
        got: usize,
    },
}

/// Predictions with standard errors and intervals.
#[derive(Debug, Clone)]
pub struct PredictionSe {
    /// Prediction on the requested scale (R `fit`).
    pub fit: Col<f64>,
    /// Standard error of the fitted mean on the requested scale (R `se.fit`).
    /// For a prediction interval this is still the standard error of the mean;
    /// the interval uses `sqrt(se_fit² + σ²)`.
    pub se_fit: Col<f64>,
    /// Lower interval bound.
    pub lower: Col<f64>,
    /// Upper interval bound.
    pub upper: Col<f64>,
    /// Scale of `fit`, `se_fit` and the bounds.
    pub scale: PredictionType,
    /// Interval type.
    pub interval: IntervalType,
    /// Confidence level.
    pub level: f64,
    /// Degrees of freedom of the t quantile, `f64::INFINITY` when the normal
    /// quantile is used (GLMs).
    pub df: f64,
    /// Residual standard deviation σ for linear models (R `residual.scale`),
    /// `None` for GLMs.
    pub residual_scale: Option<f64>,
}

impl PredictionSe {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.fit.nrows()
    }
    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.fit.nrows() == 0
    }
}

/// Predictions with standard errors and confidence / prediction intervals.
///
/// Implemented by the linear models ([`FittedOls`](crate::solvers::FittedOls),
/// [`FittedWls`](crate::solvers::FittedWls), [`FittedRidge`](crate::solvers::FittedRidge)),
/// the dedicated GLM regressors ([`FittedPoisson`](crate::solvers::FittedPoisson),
/// [`FittedBinomial`](crate::solvers::FittedBinomial),
/// [`FittedNegativeBinomial`](crate::solvers::FittedNegativeBinomial),
/// [`FittedTweedie`](crate::solvers::FittedTweedie),
/// [`FittedGamma`](crate::solvers::FittedGamma)) and the penalized GLM engine
/// ([`FittedPenalizedGlm`](crate::solvers::penalized_glm::FittedPenalizedGlm); for
/// the family-function result see
/// [`GlmResult::predict_se`](crate::solvers::penalized_glm::GlmResult::predict_se)).
///
/// [`PredictWithSe::predict_se`] returns, per row of new data, the prediction,
/// its standard error and the lower / upper interval bound, as R's
/// `predict.lm(se.fit = TRUE, interval = ...)` and
/// `predict.glm(se.fit = TRUE, type = "link" | "response")`.
///
/// # Linear models (OLS, WLS, Ridge)
///
/// * `se_fit` is the standard error of the fitted mean, `sqrt(x0' V x0)`, for
///   both interval types (R's `se.fit`).
/// * A confidence interval is `fit ± t(df) · se_fit`; a prediction interval is
///   `fit ± t(df) · sqrt(se_fit² + σ²)`, with `σ²` the residual variance (for WLS,
///   a new observation of weight 1, as R's `predict.lm` default `pred.var`).
/// * The link and response scales coincide (identity link).
///
/// # GLMs
///
/// * The standard error on the link scale is `sqrt(x0' V x0)` with `V` the
///   coefficient covariance (including the dispersion where it is estimated). On
///   the response scale it is the delta-method value `|dμ/dη| · se_link`, which
///   is exactly what `predict.glm(se.fit = TRUE, type = "response")` reports.
/// * Intervals are built on the link scale with the normal quantile,
///   `η ± z · se_link`, and on the response scale they are that interval mapped
///   through the inverse link (so they stay inside the range of the mean and are
///   not symmetric around `fit`). This is the usual construction (`ggeffects`,
///   `emmeans`' `type = "response"`); R's `predict.glm` itself returns no interval.
/// * Prediction intervals are not defined for GLMs and are rejected with
///   [`PredictSeError::PredictionIntervalUnsupported`].
/// * An offset adds to `η` without contributing to the standard error.
pub trait PredictWithSe {
    /// Predict `x` (same feature columns as the fit, no intercept column).
    fn predict_se(
        &self,
        x: &Mat<f64>,
        options: &PredictSeOptions,
    ) -> Result<PredictionSe, PredictSeError>;
}

/// Linear-model result from point predictions, the standard error of the mean,
/// the residual variance and the residual degrees of freedom.
pub(crate) fn linear_prediction(
    fit: Col<f64>,
    se_fit: Col<f64>,
    sigma2: f64,
    df: f64,
    options: &PredictSeOptions,
) -> Result<PredictionSe, PredictSeError> {
    options.check_level()?;
    let n = fit.nrows();
    let t = if df > 0.0 {
        StudentsT::new(0.0, 1.0, df)
            .map(|d| d.inverse_cdf(0.5 + options.level / 2.0))
            .unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    let half = |i: usize| {
        let s = match options.interval {
            IntervalType::Confidence => se_fit[i],
            IntervalType::Prediction => (se_fit[i] * se_fit[i] + sigma2).sqrt(),
        };
        t * s
    };
    let lower = Col::from_fn(n, |i| fit[i] - half(i));
    let upper = Col::from_fn(n, |i| fit[i] + half(i));
    Ok(PredictionSe {
        fit,
        se_fit,
        lower,
        upper,
        scale: options.scale,
        interval: options.interval,
        level: options.level,
        df,
        residual_scale: Some(sigma2.max(0.0).sqrt()),
    })
}

/// [`linear_prediction`] for a fitted linear model, using its confidence-interval
/// standard errors (`se` of `predict_with_interval(.., Confidence, ..)` is
/// `sqrt(σ² x0'(X'X)⁻¹x0)`).
pub(crate) fn linear_predict_se<F: crate::solvers::FittedRegressor + ?Sized>(
    fitted: &F,
    x: &Mat<f64>,
    options: &PredictSeOptions,
) -> Result<PredictionSe, PredictSeError> {
    options.check_level()?;
    let expected = fitted.result().coefficients.nrows();
    if x.ncols() != expected {
        return Err(PredictSeError::DimensionMismatch {
            expected,
            got: x.ncols(),
        });
    }
    let r = fitted.predict_with_interval(x, Some(IntervalType::Confidence), options.level);
    let res = fitted.result();
    linear_prediction(r.fit, r.se, res.mse, res.residual_df() as f64, options)
}

/// GLM result from the link-scale prediction and its standard error.
pub(crate) fn glm_prediction(
    eta: Col<f64>,
    se_eta: Col<f64>,
    link_inverse: &dyn Fn(f64) -> f64,
    mu_eta: &dyn Fn(f64) -> f64,
    options: &PredictSeOptions,
) -> Result<PredictionSe, PredictSeError> {
    options.check_level()?;
    if options.interval == IntervalType::Prediction {
        return Err(PredictSeError::PredictionIntervalUnsupported);
    }
    let z = Normal::new(0.0, 1.0)
        .map(|d| d.inverse_cdf(0.5 + options.level / 2.0))
        .unwrap_or(f64::NAN);
    let n = eta.nrows();
    let lo = Col::from_fn(n, |i| eta[i] - z * se_eta[i]);
    let hi = Col::from_fn(n, |i| eta[i] + z * se_eta[i]);
    let (fit, se_fit, lower, upper) = match options.scale {
        PredictionType::Link => (eta, se_eta, lo, hi),
        PredictionType::Response => {
            let fit = Col::from_fn(n, |i| link_inverse(eta[i]));
            let se = Col::from_fn(n, |i| se_eta[i] * mu_eta(eta[i]).abs());
            let a = Col::from_fn(n, |i| link_inverse(lo[i]));
            let b = Col::from_fn(n, |i| link_inverse(hi[i]));
            let lower = Col::from_fn(n, |i| a[i].min(b[i]));
            let upper = Col::from_fn(n, |i| a[i].max(b[i]));
            (fit, se, lower, upper)
        }
    };
    Ok(PredictionSe {
        fit,
        se_fit,
        lower,
        upper,
        scale: options.scale,
        interval: options.interval,
        level: options.level,
        df: f64::INFINITY,
        residual_scale: None,
    })
}

/// `sqrt(x0' V x0)` for every row of `design` (already including any intercept column).
pub(crate) fn quadratic_se(design: &Mat<f64>, vcov: &Mat<f64>) -> Col<f64> {
    let p = vcov.nrows();
    Col::from_fn(design.nrows(), |i| {
        let mut v = 0.0;
        for j in 0..p {
            let xj = design[(i, j)];
            if xj == 0.0 {
                continue;
            }
            for k in 0..p {
                v += xj * vcov[(j, k)] * design[(i, k)];
            }
        }
        if v >= 0.0 {
            v.sqrt()
        } else {
            f64::NAN
        }
    })
}

macro_rules! linear_predict_with_se {
    ($($t:ty),*) => {$(
        impl PredictWithSe for $t {
            fn predict_se(
                &self,
                x: &Mat<f64>,
                options: &PredictSeOptions,
            ) -> Result<PredictionSe, PredictSeError> {
                linear_predict_se(self, x, options)
            }
        }
    )*};
}

linear_predict_with_se!(
    crate::solvers::FittedOls,
    crate::solvers::FittedWls,
    crate::solvers::FittedRidge
);

impl PredictWithSe for crate::solvers::FittedGamma {
    fn predict_se(
        &self,
        x: &Mat<f64>,
        options: &PredictSeOptions,
    ) -> Result<PredictionSe, PredictSeError> {
        self.inner().predict_se(x, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_builder() {
        let o = PredictSeOptions::new().link().prediction().level(0.9);
        assert_eq!(o.scale, PredictionType::Link);
        assert_eq!(o.interval, IntervalType::Prediction);
        assert_eq!(o.level, 0.9);
        assert_eq!(
            PredictSeOptions::default().interval,
            IntervalType::Confidence
        );
    }

    #[test]
    fn glm_rejects_prediction_interval_and_bad_level() {
        let eta = Col::from_fn(1, |_| 0.0);
        let se = Col::from_fn(1, |_| 1.0);
        let id = |e: f64| e;
        let one = |_: f64| 1.0;
        let r = glm_prediction(
            eta.clone(),
            se.clone(),
            &id,
            &one,
            &PredictSeOptions::new().prediction(),
        );
        assert_eq!(
            r.unwrap_err(),
            PredictSeError::PredictionIntervalUnsupported
        );
        let r = glm_prediction(eta, se, &id, &one, &PredictSeOptions::new().level(1.0));
        assert_eq!(r.unwrap_err(), PredictSeError::InvalidLevel(1.0));
    }

    #[test]
    fn response_interval_is_transformed_link_interval() {
        let eta = Col::from_fn(1, |_| 0.5);
        let se = Col::from_fn(1, |_| 0.2);
        let r = glm_prediction(
            eta,
            se,
            &|e: f64| e.exp(),
            &|e: f64| e.exp(),
            &PredictSeOptions::new(),
        )
        .unwrap();
        let z = 1.959963984540054;
        assert!((r.lower[0] - (0.5 - z * 0.2f64).exp()).abs() < 1e-12);
        assert!((r.upper[0] - (0.5 + z * 0.2f64).exp()).abs() < 1e-12);
        assert!((r.se_fit[0] - 0.2 * 0.5f64.exp()).abs() < 1e-12);
    }
}
