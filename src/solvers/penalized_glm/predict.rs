//! Predictions with standard errors for the penalized GLM engine (#80).

use faer::{Col, Mat};

use super::engine::DispersionRule;
use super::families::GlmResult;
use super::regressor::FittedPenalizedGlm;
use crate::inference::{
    glm_prediction, quadratic_se, PredictSeError, PredictSeOptions, PredictWithSe, PredictionSe,
};

impl GlmResult {
    /// Predictions with standard errors and confidence intervals for new data,
    /// as R's `predict.glm(se.fit = TRUE, type = ...)`; see
    /// [`crate::inference::PredictWithSe`] for the conventions.
    ///
    /// `x` is column-major in the layout the `fit_*` function was given: the same
    /// feature columns and, when the fit used [`Self::offset_column`], the offset
    /// column at that position (its value is added to the linear predictor). A row
    /// with a non-finite value gives `NaN`.
    ///
    /// The standard error uses the stored covariance ([`GlmInferenceResult::vcov`](
    /// super::GlmInferenceResult::vcov)), which carries the Pearson dispersion for
    /// Gamma, Tweedie and quasi-Poisson, i.e. the covariance behind the reported
    /// standard errors, so
    /// the fit must have been run with `compute_inference = true`; for a penalized
    /// fit this is the covariance at the posterior mode of the selected
    /// [`VcovType`](super::VcovType).
    ///
    /// # Errors
    ///
    /// [`PredictSeError::MissingCovariance`] without inference,
    /// [`PredictSeError::DimensionMismatch`] when `x` has the wrong number of
    /// columns or ragged columns, [`PredictSeError::PredictionIntervalUnsupported`]
    /// for a prediction interval and [`PredictSeError::InvalidLevel`].
    pub fn predict_se(
        &self,
        x: &[Vec<f64>],
        options: &PredictSeOptions,
    ) -> Result<PredictionSe, PredictSeError> {
        let n_features = self.core.coefficients.len();
        let offset_idx = self.offset_column.and_then(|c| c.checked_sub(1));
        let expected = n_features + usize::from(offset_idx.is_some());
        if x.len() != expected {
            return Err(PredictSeError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let n = x.first().map_or(0, Vec::len);
        if x.iter().any(|c| c.len() != n) {
            return Err(PredictSeError::DimensionMismatch {
                expected: n,
                got: x.iter().map(Vec::len).find(|&l| l != n).unwrap_or(0),
            });
        }
        let features: Vec<&Vec<f64>> = x
            .iter()
            .enumerate()
            .filter(|(j, _)| Some(*j) != offset_idx)
            .map(|(_, c)| c)
            .collect();
        let offset = offset_idx.map(|o| &x[o]);
        predict_se_core(
            self,
            n,
            &|i, j| features[j][i],
            &|i| offset.map_or(0.0, |o| o[i]),
            options,
        )
    }
}

impl FittedPenalizedGlm {
    /// [`PredictWithSe::predict_se`] with an offset added to the linear predictor.
    pub fn predict_se_with_offset(
        &self,
        x: &Mat<f64>,
        offset: &Col<f64>,
        options: &PredictSeOptions,
    ) -> Result<PredictionSe, PredictSeError> {
        if offset.nrows() != x.nrows() {
            return Err(PredictSeError::DimensionMismatch {
                expected: x.nrows(),
                got: offset.nrows(),
            });
        }
        self.predict_se_impl(x, &|i| offset[i], options)
    }

    fn predict_se_impl(
        &self,
        x: &Mat<f64>,
        offset: &dyn Fn(usize) -> f64,
        options: &PredictSeOptions,
    ) -> Result<PredictionSe, PredictSeError> {
        let expected = self.result().core.coefficients.len();
        if x.ncols() != expected {
            return Err(PredictSeError::DimensionMismatch {
                expected,
                got: x.ncols(),
            });
        }
        predict_se_core(self.result(), x.nrows(), &|i, j| x[(i, j)], offset, options)
    }
}

impl PredictWithSe for FittedPenalizedGlm {
    /// See [`GlmResult::predict_se`]; `x` holds the feature columns only (use
    /// [`FittedPenalizedGlm::predict_se_with_offset`] for an offset). Needs
    /// `compute_inference(true)`.
    fn predict_se(
        &self,
        x: &Mat<f64>,
        options: &PredictSeOptions,
    ) -> Result<PredictionSe, PredictSeError> {
        self.predict_se_impl(x, &|_| 0.0, options)
    }
}

fn predict_se_core(
    fit: &GlmResult,
    n: usize,
    feature: &dyn Fn(usize, usize) -> f64,
    offset: &dyn Fn(usize) -> f64,
    options: &PredictSeOptions,
) -> Result<PredictionSe, PredictSeError> {
    let inf = fit
        .inference
        .as_ref()
        .ok_or(PredictSeError::MissingCovariance(
            "fit with compute_inference = true",
        ))?;
    let vcov = inf.vcov.as_ref().ok_or(PredictSeError::MissingCovariance(
        "fit with compute_inference = true",
    ))?;
    let params = &inf.matrix_parameters;
    let coefs = &fit.core.coefficients;
    let b0 = fit.core.intercept.unwrap_or(0.0);
    let eta = Col::from_fn(n, |i| {
        b0 + offset(i)
            + coefs
                .iter()
                .enumerate()
                .filter(|(_, b)| !b.is_nan())
                .map(|(j, b)| b * feature(i, j))
                .sum::<f64>()
    });
    let design = Mat::from_fn(n, params.len(), |i, k| match params[k] {
        None => 1.0,
        Some(j) => feature(i, j),
    });
    // The stored covariance already carries the dispersion (Gamma, Tweedie,
    // quasi-Poisson), as the standard errors do.
    let se_eta = quadratic_se(&design, vcov);

    let theta = match fit.dispersion_rule {
        DispersionRule::Given(t) => Some(t),
        _ => fit.core.dispersion,
    };
    let family = fit.family.glm_family(theta);
    let linkinv = |e: f64| family.link_inverse(e);
    let mu_eta = |e: f64| 1.0 / family.link_derivative(family.link_inverse(e));
    glm_prediction(eta, se_eta, &linkinv, &mu_eta, options)
}
