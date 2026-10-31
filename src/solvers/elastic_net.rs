//! Elastic Net solver (combined L1 and L2 regularization).

use crate::core::{HasModelInfo, ModelInfo};
use crate::core::{
    IntervalType, LambdaScaling, PredictionResult, RegressionOptions, RegressionOptionsBuilder,
    RegressionResult,
};
use crate::inference::{
    compute_prediction_intervals, compute_xtx_inverse_augmented_reduced,
    compute_xtx_inverse_reduced,
};
use crate::solvers::ridge::RidgeRegressor;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::detect_constant_columns;
use faer::{Col, Mat};

/// Elastic Net regression estimator, solved by coordinate descent.
///
/// Minimizes: ||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)
/// by coordinate descent. With `LambdaScaling::Glmnet` the fit is
/// `glmnet(standardize = FALSE)` at `λ`, the same as [`path`](Self::path).
///
/// Where:
/// - α = 1 is pure Lasso (L1)
/// - α = 0 is pure Ridge (L2)
/// - 0 < α < 1 is a mix of both
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{ElasticNetRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 5, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = ElasticNetRegressor::builder()
///     .with_intercept(true)
///     .lambda(1.0)
///     .alpha(0.5)  // 50% L1, 50% L2
///     .build()
///     .fit(&x, &y)?;
///
/// println!("Non-zero coefficients: {:?}", fitted.coefficients());
/// ```
#[derive(Debug, Clone)]
pub struct ElasticNetRegressor {
    options: RegressionOptions,
}

impl ElasticNetRegressor {
    /// Create a new Elastic Net regressor with the given options.
    pub fn new(options: RegressionOptions) -> Self {
        Self { options }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> ElasticNetRegressorBuilder {
        ElasticNetRegressorBuilder::default()
    }
}

impl Regressor for ElasticNetRegressor {
    type Fitted = FittedElasticNet;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        if let Some((xc, yc, na_info)) =
            crate::core::apply_na_action(x, y, self.options.na_action, false)?
        {
            let mut fitted = self.fit(&xc, &yc)?;
            fitted.result.na_info = Some(na_info);
            return Ok(fitted);
        }
        // When alpha = 0 (or lambda = 0), delegate to Ridge
        if self.options.alpha == 0.0 || self.options.lambda == 0.0 {
            let ridge = RidgeRegressor::new(self.options.clone());
            let ridge_fitted = ridge.fit(x, y)?;
            // For pure Ridge, use the same aliased mask from result
            let aliased = ridge_fitted.result().aliased.clone();
            let xtx_inverse = compute_xtx_inverse_augmented_reduced(x, &aliased).ok();
            return Ok(FittedElasticNet {
                options: self.options.clone(),
                result: ridge_fitted.result().clone(),
                xtx_inverse,
                aliased,
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        if n_samples < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: n_samples,
            });
        }

        if self.options.with_intercept {
            // Detect constant columns (aliased with intercept)
            let constant_cols = detect_constant_columns(x, self.options.rank_tolerance);

            // Solve Elastic Net by coordinate descent (intercept unpenalised)
            let (intercept, coefficients) = self.fit_cd(x, y)?;

            // Mark aliased columns (constant or zero coefficient due to L1)
            let mut aliased = constant_cols.clone();
            for j in 0..n_features {
                if coefficients[j].abs() < 1e-10 {
                    aliased[j] = true;
                }
            }

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    if !constant_cols[j] {
                        pred += x[(i, j)] * coefficients[j];
                    }
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            // Count non-zero coefficients for effective rank
            let n_nonzero = coefficients
                .iter()
                .enumerate()
                .filter(|(j, &c)| !constant_cols[*j] && c.abs() > 1e-10)
                .count();
            let n_params = n_nonzero + 1; // +1 for intercept

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                Some(intercept),
                &residuals,
                &fitted_values,
                &constant_cols,
                n_nonzero,
                n_params,
            )?;

            // Compute (X_aug'X_aug)⁻¹ for prediction intervals (using non-aliased columns)
            // Note: This is approximate for L1-penalized models
            let xtx_inverse = compute_xtx_inverse_augmented_reduced(x, &constant_cols).ok();

            Ok(FittedElasticNet {
                options: self.options.clone(),
                result,
                xtx_inverse,
                aliased: constant_cols,
            })
        } else {
            // No intercept case - detect zero-variance columns
            let mut aliased = vec![false; n_features];
            for j in 0..n_features {
                let mut col_sq = 0.0;
                for i in 0..n_samples {
                    col_sq += x[(i, j)] * x[(i, j)];
                }
                if col_sq < self.options.rank_tolerance {
                    aliased[j] = true;
                }
            }

            let (_, coefficients) = self.fit_cd(x, y)?;

            // Mark zero coefficients as aliased
            for j in 0..n_features {
                if coefficients[j].abs() < 1e-10 {
                    aliased[j] = true;
                }
            }

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = 0.0;
                for j in 0..n_features {
                    pred += x[(i, j)] * coefficients[j];
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            let n_nonzero = coefficients
                .iter()
                .enumerate()
                .filter(|(j, &c)| !aliased[*j] && c.abs() > 1e-10)
                .count();
            let n_params = n_nonzero;

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                None,
                &residuals,
                &fitted_values,
                &aliased,
                n_nonzero,
                n_params,
            )?;

            // Compute (X'X)⁻¹ for prediction intervals (using non-aliased columns)
            // Note: This is approximate for L1-penalized models
            let xtx_inverse = compute_xtx_inverse_reduced(x, &aliased).ok();

            Ok(FittedElasticNet {
                options: self.options.clone(),
                result,
                xtx_inverse,
                aliased,
            })
        }
    }
}

impl ElasticNetRegressor {
    /// Elastic-net coefficient path at `lambdas` (glmnet scale and
    /// objective, see [`path`](crate::solvers::path)) with this regressor's
    /// `alpha` and intercept setting and no standardisation, computed by
    /// coordinate descent with warm starts. Equivalent to
    /// `path::fit_path(x, y, lambdas, &PathOptions::elastic_net(alpha).with_intercept(..))`.
    pub fn path(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        lambdas: &[f64],
    ) -> Result<crate::solvers::path::RegPath, RegressionError> {
        crate::solvers::path::fit_path(x, y, lambdas, &self.path_options())
    }

    /// k-fold cross-validation of the elastic-net path (see
    /// [`path::cv_path`](crate::solvers::path::cv_path)).
    pub fn cv_path(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        lambdas: &[f64],
        fold_ids: &[usize],
    ) -> Result<crate::solvers::path::CvPath, RegressionError> {
        crate::solvers::path::cv_path(x, y, lambdas, fold_ids, &self.path_options())
    }

    fn path_options(&self) -> crate::solvers::path::PathOptions {
        crate::solvers::path::PathOptions::elastic_net(self.options.alpha)
            .with_intercept(self.options.with_intercept)
    }

    /// Solve the elastic net exactly by coordinate descent.
    ///
    /// With `LambdaScaling::Raw` the objective is
    /// `||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)`. With `LambdaScaling::Glmnet`
    /// the result is `glmnet(x, y, alpha, lambda, standardize = FALSE)`, i.e.
    /// [`path`](Self::path) at `λ`: glmnet's
    /// `1/(2n)||y - Xβ||² + λ((1-α)/2 ||β||₂² + α||β||₁)` with the ridge part
    /// divided by `sd(y)` (see [`path`](crate::solvers::path)).
    /// The intercept is never penalised. Converges when the largest
    /// `(1/n)||x_j||² · Δβ_j²` of a pass, relative to `var(y)`, is below
    /// `tolerance` (glmnet's `thresh`), within
    /// `max(max_iterations, 100 000)` passes.
    fn fit_cd(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<(f64, Col<f64>), RegressionError> {
        let (lambda, alpha) = (self.options.lambda, self.options.alpha);
        let tol = self.options.tolerance;
        let max_iter = self.options.max_iterations.max(100_000);
        match self.options.lambda_scaling {
            LambdaScaling::Raw => {
                // 1/(2n)·RSS + l1·||β||₁ + l2/2·||β||²
                let n = x.nrows() as f64;
                crate::solvers::path::fit_penalized(
                    x,
                    y,
                    lambda * alpha / (2.0 * n),
                    lambda * (1.0 - alpha) / n,
                    self.options.with_intercept,
                    tol,
                    max_iter,
                )
            }
            LambdaScaling::Glmnet => {
                let mut opts = self.path_options();
                opts.tol = tol;
                opts.max_iter = max_iter;
                let path = crate::solvers::path::fit_path(x, y, &[lambda], &opts)?;
                Ok((path.intercepts[0], path.coefficients_at(0)))
            }
        }
    }

    /// Compute fit statistics.
    #[allow(clippy::too_many_arguments)]
    fn compute_statistics(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        coefficients: &Col<f64>,
        intercept: Option<f64>,
        residuals: &Col<f64>,
        fitted_values: &Col<f64>,
        aliased: &[bool],
        rank: usize,
        n_params: usize,
    ) -> Result<RegressionResult, RegressionError> {
        let n = y.nrows();
        let n_features = x.ncols();

        let stats = super::fit_stats::linear_fit_stats(
            y,
            residuals,
            None,
            intercept.is_some(),
            n_params,
            true,
        );

        let mut result = RegressionResult::empty(n_features, n);
        result.coefficients = coefficients.clone();
        result.intercept = intercept;
        result.residuals = residuals.clone();
        result.fitted_values = fitted_values.clone();
        result.rank = rank;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.aliased = aliased.to_vec();
        result.rank_tolerance = self.options.rank_tolerance;
        stats.apply(&mut result);
        result.confidence_level = self.options.confidence_level;

        // Elastic Net doesn't have simple standard errors
        // (would require bootstrap or other methods)

        Ok(result)
    }
}

/// A fitted Elastic Net model.
#[derive(Debug, Clone)]
pub struct FittedElasticNet {
    options: RegressionOptions,
    result: RegressionResult,
    /// (X'X)⁻¹ or (X_aug'X_aug)⁻¹ for prediction intervals (reduced to non-aliased columns)
    /// Note: Standard errors for L1-penalized models are approximate
    xtx_inverse: Option<Mat<f64>>,
    /// Which columns are aliased (constant or zero coefficient)
    aliased: Vec<bool>,
}

impl HasModelInfo for FittedElasticNet {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::gaussian("elastic_net")
    }
}

impl FittedElasticNet {
    /// Get the options used to fit this model.
    pub fn options(&self) -> &RegressionOptions {
        &self.options
    }

    /// Get the lambda (regularization) parameter.
    pub fn lambda(&self) -> f64 {
        self.options.lambda
    }

    /// Get the alpha (L1/L2 mixing) parameter.
    pub fn alpha(&self) -> f64 {
        self.options.alpha
    }

    /// Count non-zero coefficients (sparsity).
    pub fn n_nonzero(&self) -> usize {
        self.result
            .coefficients
            .iter()
            .filter(|&&c| c.abs() > 1e-10)
            .count()
    }
}

impl FittedRegressor for FittedElasticNet {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut predictions = Col::zeros(n_samples);

        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n_samples {
            let mut pred = intercept;
            for j in 0..n_features {
                pred += x[(i, j)] * self.result.coefficients[j];
            }
            predictions[i] = pred;
        }

        predictions
    }

    fn result(&self) -> &RegressionResult {
        &self.result
    }

    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        let predictions = self.predict(x);

        match interval {
            None => PredictionResult::point_only(predictions),
            Some(interval_type) => {
                // Note: Prediction intervals for L1-penalized models are approximate
                match &self.xtx_inverse {
                    Some(xtx_inv) => {
                        let df = self.result.residual_df() as f64;
                        let has_intercept = self.result.intercept.is_some();

                        // Filter x to only non-aliased columns (to match xtx_inverse dimensions)
                        let x_reduced = self.reduce_to_non_aliased(x);

                        compute_prediction_intervals(
                            &x_reduced,
                            xtx_inv,
                            &predictions,
                            self.result.mse,
                            df,
                            level,
                            interval_type,
                            has_intercept,
                        )
                    }
                    None => {
                        let n = x.nrows();
                        let mut lower = Col::zeros(n);
                        let mut upper = Col::zeros(n);
                        let mut se = Col::zeros(n);
                        for i in 0..n {
                            lower[i] = f64::NAN;
                            upper[i] = f64::NAN;
                            se[i] = f64::NAN;
                        }
                        PredictionResult::with_intervals(predictions, lower, upper, se)
                    }
                }
            }
        }
    }
}

impl FittedElasticNet {
    /// Reduce x to only non-aliased columns.
    fn reduce_to_non_aliased(&self, x: &Mat<f64>) -> Mat<f64> {
        let n_samples = x.nrows();
        let non_aliased_cols: Vec<usize> = self
            .aliased
            .iter()
            .enumerate()
            .filter(|(_, &is_aliased)| !is_aliased)
            .map(|(j, _)| j)
            .collect();

        let n_reduced = non_aliased_cols.len();
        let mut x_reduced = Mat::zeros(n_samples, n_reduced);

        for i in 0..n_samples {
            for (k, &j) in non_aliased_cols.iter().enumerate() {
                x_reduced[(i, k)] = x[(i, j)];
            }
        }

        x_reduced
    }
}

/// Builder for `ElasticNetRegressor`.
#[derive(Debug, Clone, Default)]
pub struct ElasticNetRegressorBuilder {
    builder: RegressionOptionsBuilder,
}

impl ElasticNetRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.builder = self.builder.with_intercept(include);
        self
    }

    /// Set the regularization parameter (lambda).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.builder = self.builder.lambda(lambda);
        self
    }

    /// Set the L1/L2 mixing parameter (alpha).
    /// - alpha = 1: pure Lasso (L1)
    /// - alpha = 0: pure Ridge (L2)
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.builder = self.builder.alpha(alpha);
        self
    }

    /// Set the lambda scaling convention.
    ///
    /// Use `LambdaScaling::Glmnet` to match R's glmnet package behavior.
    pub fn lambda_scaling(mut self, scaling: LambdaScaling) -> Self {
        self.builder = self.builder.lambda_scaling(scaling);
        self
    }

    /// Set maximum coordinate-descent passes (at least 100 000 are allowed,
    /// as in [`PathOptions`](crate::solvers::path::PathOptions)).
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.builder = self.builder.max_iterations(max_iter);
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.builder = self.builder.tolerance(tol);
        self
    }

    /// Build the Elastic Net regressor.
    pub fn build(self) -> ElasticNetRegressor {
        ElasticNetRegressor::new(self.builder.build_unchecked())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_net_basic() {
        let x = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
        let mut y = Col::zeros(20);
        for i in 0..20 {
            y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)];
        }

        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(0.01)
            .alpha(0.5)
            .build();

        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!(fitted.r_squared() > 0.9);
    }

    #[test]
    fn test_lasso_sparsity() {
        // With high lambda and alpha=1 (Lasso), should get sparse solution
        let x = Mat::from_fn(50, 5, |i, j| ((i + j) as f64) * 0.1);
        let mut y = Col::zeros(50);
        // Only first 2 features matter
        for i in 0..50 {
            y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)];
        }

        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(1.0)
            .alpha(1.0) // Pure Lasso
            .build();

        let fitted = model.fit(&x, &y).expect("model should fit");

        // Some coefficients should be zero (sparse)
        let n_zero = fitted
            .coefficients()
            .iter()
            .filter(|&&c| c.abs() < 1e-10)
            .count();

        // At least some sparsity expected
        assert!(
            n_zero >= 1,
            "Expected some zero coefficients, got {}",
            n_zero
        );
    }
}
