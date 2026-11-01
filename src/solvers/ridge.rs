//! Ridge regression solver (L2 regularization).

use crate::core::{HasModelInfo, ModelInfo};
use crate::core::{
    IntervalType, LambdaScaling, PredictionResult, RegressionOptions, RegressionOptionsBuilder,
    RegressionResult, SolverType,
};
use crate::diagnostics::{augment_with, AugmentSpec, Augmentable, Press, ResidualType};
use crate::inference::{compute_variance_factor, intervals_from_variance_factor, leverage_new};
use crate::solvers::moments::MomentAccumulator;
use crate::solvers::ols::OlsRegressor;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::{center_columns, center_vector};
use faer::prelude::Solve;
use faer::{Col, Mat};

/// Ridge regression estimator with L2 regularization.
///
/// Minimizes: ||y - Xβ||² + λ||β||²
///
/// The solution is: β = (X'X + λI)^(-1) X'y
///
/// When λ = 0, this reduces to OLS.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{RidgeRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = RidgeRegressor::builder()
///     .with_intercept(true)
///     .lambda(0.1)
///     .build()
///     .fit(&x, &y)?;
///
/// println!("R² = {}", fitted.r_squared());
/// ```
#[derive(Debug, Clone)]
pub struct RidgeRegressor {
    options: RegressionOptions,
    conditioning: Option<ConditioningLambda>,
}

/// Default `κ(XᵀX)` threshold for [`ConditioningLambda`]: at or below it the
/// design counts as well-conditioned (the `cond > 1000` heuristic of the
/// anofox-orchestration DIAG-03 ridge sweep).
pub const DEFAULT_CONDITION_THRESHOLD: f64 = 1000.0;
/// Default small λ for [`ConditioningLambda`] on a well-conditioned design.
pub const DEFAULT_LAMBDA_SMALL: f64 = 0.1;

/// Conditioning-aware λ selection (DataZooDE/anofox-forecast#258).
///
/// A fixed λ over-shrinks coefficients on a well-conditioned design (e.g.
/// the seasonal amplitude of a seasonal-dummy fit on a fold that sees every
/// season often enough), where regularisation buys no stability. With this
/// set, [`RidgeRegressor::fit`] measures the condition number `κ(XᵀX)` of
/// the design it penalises (`X` centered when an intercept is fitted) and
/// uses `lambda_small` when `κ ≤ cond_threshold`, the configured λ
/// otherwise. The λ actually used is reported by [`FittedRidge::lambda`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditioningLambda {
    /// λ used on a well-conditioned design.
    pub lambda_small: f64,
    /// `κ(XᵀX)` at or below which the design counts as well-conditioned.
    pub cond_threshold: f64,
}

impl Default for ConditioningLambda {
    fn default() -> Self {
        Self {
            lambda_small: DEFAULT_LAMBDA_SMALL,
            cond_threshold: DEFAULT_CONDITION_THRESHOLD,
        }
    }
}

/// Condition number `κ(XᵀX) = (σ_max / σ_min)²` of the design a ridge fit
/// penalises: `X` with each column centered when `with_intercept`, `X`
/// itself otherwise. `INFINITY` for a rank-deficient (or empty) design.
pub fn gram_condition_number(x: &Mat<f64>, with_intercept: bool) -> f64 {
    if x.nrows() == 0 || x.ncols() == 0 {
        return f64::INFINITY;
    }
    let design = if with_intercept {
        center_columns(x).0
    } else {
        x.clone()
    };
    let svd = match design.thin_svd() {
        Ok(svd) => svd,
        Err(_) => return f64::INFINITY,
    };
    let s = svd.S().column_vector();
    let (mut s_max, mut s_min) = (0.0_f64, f64::INFINITY);
    for i in 0..s.nrows() {
        s_max = s_max.max(s[i]);
        s_min = s_min.min(s[i]);
    }
    let tolerance = s_max * f64::EPSILON * (x.nrows().max(x.ncols()) as f64);
    if s_min.is_nan() || s_min <= tolerance {
        return f64::INFINITY;
    }
    (s_max / s_min).powi(2)
}

impl RidgeRegressor {
    /// Create a new Ridge regressor with the given options.
    pub fn new(options: RegressionOptions) -> Self {
        Self {
            options,
            conditioning: None,
        }
    }

    /// Enable conditioning-aware λ selection (see [`ConditioningLambda`]).
    pub fn with_conditioning(mut self, conditioning: ConditioningLambda) -> Self {
        self.conditioning = Some(conditioning);
        self
    }

    /// The λ [`Self::fit`] uses on `x`: the configured λ, or, with
    /// [`ConditioningLambda`] set, `lambda_small` when the design is
    /// well-conditioned.
    pub fn select_lambda(&self, x: &Mat<f64>) -> f64 {
        match self.conditioning {
            Some(c)
                if gram_condition_number(x, self.options.with_intercept) <= c.cond_threshold =>
            {
                c.lambda_small
            }
            _ => self.options.lambda,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> RidgeRegressorBuilder {
        RidgeRegressorBuilder::default()
    }
}

impl Regressor for RidgeRegressor {
    type Fitted = FittedRidge;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        if let Some((xc, yc, na_info)) =
            crate::core::apply_na_action(x, y, self.options.na_action, false)?
        {
            let mut fitted = self.fit(&xc, &yc)?;
            fitted.result.na_info = Some(na_info);
            return Ok(fitted);
        }
        crate::core::validate_xy(x, y)?;
        // Conditioning-aware λ (#258): resolve λ for this design, then fit
        // with that fixed λ.
        if self.conditioning.is_some() {
            let mut fixed = self.clone();
            fixed.options.lambda = self.select_lambda(x);
            fixed.conditioning = None;
            return fixed.fit(x, y);
        }

        // When lambda = 0, delegate to OLS
        if self.options.lambda == 0.0 {
            let ols = OlsRegressor::new(self.options.clone());
            let ols_fitted = ols.fit(x, y)?;
            let variance_factor = compute_variance_factor(
                x,
                None,
                self.options.with_intercept,
                &ols_fitted.result().aliased,
                0.0,
            )
            .ok();
            return Ok(FittedRidge {
                options: self.options.clone(),
                result: ols_fitted.result().clone(),
                variance_factor,
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
            // Center the data
            let (x_centered, x_means) = center_columns(x);
            let (y_centered, y_mean) = center_vector(y);

            // Solve Ridge regression on centered data (dispatch to selected solver)
            let coefficients = match self.options.solver {
                SolverType::Qr => self.solve_ridge_qr(&x_centered, &y_centered)?,
                SolverType::Svd => self.solve_ridge_svd(&x_centered, &y_centered)?,
                SolverType::Cholesky => self.solve_ridge_cholesky(&x_centered, &y_centered)?,
            };

            // Compute intercept: intercept = y_mean - x_means' * coefficients
            let mut intercept = y_mean;
            for j in 0..n_features {
                intercept -= x_means[j] * coefficients[j];
            }

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += x[(i, j)] * coefficients[j];
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            // Compute statistics
            let aliased = vec![false; n_features]; // Ridge has no aliased coefficients
            let rank = n_features; // Ridge is always full rank
            let n_params = n_features + 1; // +1 for intercept

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                Some(intercept),
                &residuals,
                &fitted_values,
                &aliased,
                rank,
                n_params,
            )?;

            Ok(FittedRidge {
                options: self.options.clone(),
                result,
                variance_factor: self.ridge_variance_factor(x),
            })
        } else {
            // No intercept case
            let coefficients = match self.options.solver {
                SolverType::Qr => self.solve_ridge_qr(x, y)?,
                SolverType::Svd => self.solve_ridge_svd(x, y)?,
                SolverType::Cholesky => self.solve_ridge_cholesky(x, y)?,
            };

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

            let aliased = vec![false; n_features];
            let rank = n_features;
            let n_params = n_features;

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                None,
                &residuals,
                &fitted_values,
                &aliased,
                rank,
                n_params,
            )?;

            Ok(FittedRidge {
                options: self.options.clone(),
                result,
                variance_factor: self.ridge_variance_factor(x),
            })
        }
    }
}

impl RidgeRegressor {
    /// Fit Ridge from pre-accumulated moments.
    ///
    /// This is the streaming entry point for very large panels where the
    /// full `N × p` design matrix can't be materialised. Mathematically
    /// identical to [`Self::fit`] when `with_intercept = true`: builds the
    /// centered Gram `Gc = XᵀX − Σx Σxᵀ / n` and centered cross-moment
    /// `cc = Xᵀy − Σx Σy / n`, then solves `(Gc + λ_eff · I) β = cc` via
    /// Cholesky and recovers the intercept as `ȳ − x̄ᵀβ`. `λ_eff` honors
    /// the configured [`LambdaScaling`].
    ///
    /// Statistics that require per-row data (residuals, R², MSE, AIC, …)
    /// are returned as `NaN` / empty since the input rows are not
    /// retained.
    ///
    /// See also: [`MomentAccumulator`], [`Self::fit_from_accumulator`].
    pub fn fit_from_moments(
        &self,
        xtx: &Mat<f64>,
        xty: &Col<f64>,
        sum_x: &Col<f64>,
        sum_y: f64,
        n: usize,
    ) -> Result<FittedRidge, RegressionError> {
        let p = xtx.nrows();

        // Dimension validation.
        if xtx.ncols() != p {
            return Err(RegressionError::NumericalError(format!(
                "xtx must be square: got {}x{}",
                p,
                xtx.ncols()
            )));
        }
        if xty.nrows() != p {
            return Err(RegressionError::DimensionMismatch {
                x_rows: p,
                y_len: xty.nrows(),
            });
        }
        if sum_x.nrows() != p {
            return Err(RegressionError::DimensionMismatch {
                x_rows: p,
                y_len: sum_x.nrows(),
            });
        }
        let min_n = if self.options.with_intercept { 2 } else { 1 };
        if n < min_n {
            return Err(RegressionError::InsufficientObservations {
                needed: min_n,
                got: n,
            });
        }

        let lambda_eff = self.effective_lambda(n);
        let n_f = n as f64;

        let (coefficients, intercept) = if self.options.with_intercept {
            // Centered Gram and centered cross-moment.
            let mut gc = Mat::<f64>::zeros(p, p);
            for i in 0..p {
                for j in 0..p {
                    gc[(i, j)] = xtx[(i, j)] - sum_x[i] * sum_x[j] / n_f;
                }
            }
            let mut cc = Col::<f64>::zeros(p);
            for i in 0..p {
                cc[i] = xty[i] - sum_x[i] * sum_y / n_f;
            }
            // (Gc + λI)
            for i in 0..p {
                gc[(i, i)] += lambda_eff;
            }
            let llt = gc
                .llt(faer::Side::Lower)
                .map_err(|_| RegressionError::SingularMatrix)?;
            let beta = llt.solve(&cc);
            // Intercept on the original scale.
            let y_mean = sum_y / n_f;
            let mut b0 = y_mean;
            for i in 0..p {
                b0 -= (sum_x[i] / n_f) * beta[i];
            }
            (beta, Some(b0))
        } else {
            // (XᵀX + λI) β = Xᵀy
            let mut g = xtx.clone();
            for i in 0..p {
                g[(i, i)] += lambda_eff;
            }
            let llt = g
                .llt(faer::Side::Lower)
                .map_err(|_| RegressionError::SingularMatrix)?;
            let beta = llt.solve(xty);
            (beta, None)
        };

        // Defensive #21 guard on the produced coefficients.
        let aliased = vec![false; p];
        crate::solvers::ols::check_coefficients_finite(&coefficients, &aliased)?;

        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.rank = p;
        result.n_parameters = p + if self.options.with_intercept { 1 } else { 0 };
        result.n_observations = n;
        result.aliased = aliased;
        result.rank_tolerance = self.options.rank_tolerance;
        result.confidence_level = self.options.confidence_level;
        // Per-row data is unavailable in the streaming flow — leave the
        // statistics as NaN so consumers can't accidentally trust stale
        // values from `RegressionResult::empty`.
        result.r_squared = f64::NAN;
        result.adj_r_squared = f64::NAN;
        result.rmse = f64::NAN;
        result.mse = f64::NAN;
        result.f_statistic = f64::NAN;
        result.f_pvalue = f64::NAN;
        result.aic = f64::NAN;
        result.aicc = f64::NAN;
        result.bic = f64::NAN;
        result.log_likelihood = f64::NAN;

        Ok(FittedRidge {
            options: self.options.clone(),
            result,
            // The sandwich needs the rows (or X'X with the intercept terms),
            // and without a meaningful `mse` intervals would be NaN anyway.
            variance_factor: None,
        })
    }

    /// Ridge coefficient path at `lambdas` (glmnet scale, see
    /// [`path`](crate::solvers::path)), with this regressor's intercept
    /// setting and no standardisation. Equivalent to
    /// `path::fit_path(x, y, lambdas, &PathOptions::ridge().with_intercept(..))`.
    pub fn path(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        lambdas: &[f64],
    ) -> Result<crate::solvers::path::RegPath, RegressionError> {
        crate::solvers::path::fit_path(x, y, lambdas, &self.path_options())
    }

    /// k-fold cross-validation of the ridge path (see
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
        crate::solvers::path::PathOptions::ridge().with_intercept(self.options.with_intercept)
    }

    /// Fit Ridge directly from a [`MomentAccumulator`]. Convenience wrapper
    /// around [`Self::fit_from_moments`].
    pub fn fit_from_accumulator(
        &self,
        acc: &MomentAccumulator,
    ) -> Result<FittedRidge, RegressionError> {
        self.fit_from_moments(acc.xtx(), acc.xty(), acc.sum_x(), acc.sum_y(), acc.n())
    }

    /// Fit ridge from a mergeable [`crate::solvers::SufficientStats`] state.
    ///
    /// λ (with the configured [`LambdaScaling`], `n` = rows in the state) is
    /// applied here, so one state serves any λ grid. Gives the same result
    /// as [`Regressor::fit`] on the rows the state was built from, up to
    /// floating-point summation order, including the sandwich standard
    /// errors and the variance factor for intervals. Weights recorded in the
    /// state are honoured (weighted ridge). `residuals` and `fitted_values`
    /// are empty.
    pub fn fit_from_state(
        &self,
        state: &crate::solvers::SufficientStats,
    ) -> Result<FittedRidge, RegressionError> {
        let lambda = if self.options.lambda == 0.0 {
            0.0
        } else {
            self.effective_lambda(state.n_observations())
        };
        let fit = crate::solvers::suff_stats::fit_state(
            state,
            &self.options,
            crate::utils::LM_ALIAS_TOLERANCE,
            lambda,
        )?;
        let variance_factor = fit.reduced_factor.map(|m| {
            crate::inference::expand_reduced_factor(&m, &fit.aliased, self.options.with_intercept)
        });
        Ok(FittedRidge {
            options: self.options.clone(),
            result: fit.result,
            variance_factor,
        })
    }

    /// Ridge sandwich variance factor `A X'X A`, `A = (X'X + λP)⁻¹` (augmented
    /// with an unpenalised intercept when one is fitted).
    fn ridge_variance_factor(&self, x: &Mat<f64>) -> Option<Mat<f64>> {
        let lambda = self.effective_lambda(x.nrows());
        compute_variance_factor(
            x,
            None,
            self.options.with_intercept,
            &vec![false; x.ncols()],
            lambda,
        )
        .ok()
    }

    /// Get the effective lambda after applying scaling convention.
    fn effective_lambda(&self, n_samples: usize) -> f64 {
        effective_lambda(&self.options, n_samples)
    }

    /// Solve Ridge regression using QR: β = (X'X + λI)^(-1) X'y
    fn solve_ridge_qr(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lambda = self.effective_lambda(n_samples);

        // Compute X'X + λI
        let xtx = x.transpose() * x;
        let mut xtx_reg = xtx.clone();
        for i in 0..n_features {
            xtx_reg[(i, i)] += lambda;
        }

        // Compute X'y
        let xty = x.transpose() * y;

        // Solve (X'X + λI) β = X'y using QR decomposition
        let qr = xtx_reg.qr();
        let q = qr.compute_thin_Q();
        let r = qr.R();

        // Check if R is singular
        for i in 0..n_features {
            if r[(i, i)].abs() < 1e-14 {
                return Err(RegressionError::SingularMatrix);
            }
        }

        // Solve R β = Q' (X'y)
        let qty = q.transpose() * &xty;

        // Back-substitution
        let mut coefficients = Col::zeros(n_features);
        for i in (0..n_features).rev() {
            let mut sum = qty[i];
            for j in (i + 1)..n_features {
                sum -= r[(i, j)] * coefficients[j];
            }
            coefficients[i] = sum / r[(i, i)];
        }

        Ok(coefficients)
    }

    /// Solve Ridge regression using SVD of X.
    ///
    /// Uses SVD of X directly: β = V (S² + λI)⁻¹ S U' y
    /// This is the most numerically stable approach for Ridge.
    fn solve_ridge_svd(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let lambda = self.effective_lambda(n_samples);

        // Compute SVD: X = U S V'
        let svd = x.thin_svd().map_err(|_| RegressionError::SingularMatrix)?;
        let u = svd.U();
        let s = svd.S();
        let s_col = s.column_vector();
        let v = svd.V();

        let n_sv = s_col.nrows();

        // Compute U' y
        let uty = u.transpose() * y;

        // Compute d_i = s_i / (s_i² + λ) * (U'y)_i  for each singular value
        let mut d = Col::zeros(n_sv);
        for i in 0..n_sv {
            let si = s_col[i];
            d[i] = si / (si * si + lambda) * uty[i];
        }

        // β = V * d
        let coefficients = v * &d;

        // Convert from Mat to Col
        let mut coeff = Col::zeros(coefficients.nrows());
        for i in 0..coefficients.nrows() {
            coeff[i] = coefficients[i];
        }

        Ok(coeff)
    }

    /// Solve Ridge regression using Cholesky decomposition.
    ///
    /// Solves (X'X + λI) β = X'y via Cholesky. Since λ > 0, the matrix
    /// X'X + λI is always positive definite, making Cholesky the fastest option.
    fn solve_ridge_cholesky(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let lambda = self.effective_lambda(n_samples);

        // Compute X'X + λI
        let xtx = x.transpose() * x;
        let mut xtx_reg = xtx.clone();
        for i in 0..n_features {
            xtx_reg[(i, i)] += lambda;
        }

        // Compute X'y
        let xty = x.transpose() * y;

        // Solve via Cholesky (should always succeed for λ > 0)
        let llt = xtx_reg
            .llt(faer::Side::Lower)
            .map_err(|_| RegressionError::SingularMatrix)?;

        let coefficients = llt.solve(&xty);

        // Convert from Mat to Col
        let mut coeff = Col::zeros(coefficients.nrows());
        for i in 0..coefficients.nrows() {
            coeff[i] = coefficients[i];
        }

        Ok(coeff)
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

        // Compute inference statistics if requested
        if self.options.compute_inference {
            self.compute_inference(x, &mut result)?;
        }

        Ok(result)
    }

    /// Compute inference statistics for Ridge regression.
    ///
    /// For a fixed λ the sampling covariance of the ridge estimator is the
    /// sandwich `σ² · A X'X A`, `A = (X'X + λP)⁻¹` (P leaves the intercept
    /// unpenalised), with `σ² = RSS / (n − p)`. Its square-rooted diagonal is
    /// reported as `std_errors` / `intercept_std_error`; it converges to the
    /// OLS standard errors as λ → 0.
    ///
    /// t-statistics, p-values and confidence intervals are **not** reported
    /// for λ > 0 (they stay `None`): a t-test centred on the shrunken, biased
    /// coefficient is not a valid test of `β_j = 0`.
    fn compute_inference(
        &self,
        x: &Mat<f64>,
        result: &mut RegressionResult,
    ) -> Result<(), RegressionError> {
        let df = result.residual_df() as f64;
        if df <= 0.0 || !result.mse.is_finite() {
            return Ok(());
        }
        let Some(m) = self.ridge_variance_factor(x) else {
            return Ok(());
        };
        let off = usize::from(result.intercept.is_some());
        let sd = |v: f64| {
            let var = result.mse * v;
            if var >= 0.0 {
                var.sqrt()
            } else {
                f64::NAN
            }
        };
        let std_errors = Col::from_fn(x.ncols(), |j| sd(m[(j + off, j + off)]));
        if result.intercept.is_some() {
            result.intercept_std_error = Some(sd(m[(0, 0)]));
        }
        result.std_errors = Some(std_errors);
        Ok(())
    }
}

/// Penalty `λ` on the sum-of-squares scale: the configured lambda, times `n`
/// for [`LambdaScaling::Glmnet`].
fn effective_lambda(options: &RegressionOptions, n_samples: usize) -> f64 {
    match options.lambda_scaling {
        LambdaScaling::Raw => options.lambda,
        LambdaScaling::Glmnet => options.lambda * n_samples as f64,
    }
}

/// Leverage above which a row determines its own fitted value (as in
/// [`crate::diagnostics::augment`], R's `lm.influence` threshold).
const LOO_HAT_MAX: f64 = 1.0 - 10.0 * f64::EPSILON;

/// Exact leave-one-out diagnostics of a ridge fit, from [`FittedRidge::loo`].
#[derive(Debug, Clone)]
pub struct RidgeLoo {
    /// Diagonal `hᵢ` of the ridge hat matrix `H = X̃(X̃'X̃ + λP)⁻¹X̃'`, `X̃` the
    /// design with the intercept column and `P` the identity with a zero for
    /// the (unpenalised) intercept. Equivalently `1/n + x̃ᵢ'(Xc'Xc + λI)⁻¹x̃ᵢ`
    /// with the centred design `Xc` (no `1/n` term without an intercept).
    pub leverage: Col<f64>,
    /// `(yᵢ − ŷᵢ) / (1 − hᵢ)` = `yᵢ − ŷ₍ᵢ₎`, the error of the ridge fit
    /// without row `i` (same `λ`); `NaN` for rows with `hᵢ = 1`.
    pub loo_residual: Col<f64>,
    /// `ŷ₍ᵢ₎ = yᵢ − loo_residual`; `NaN` where `loo_residual` is.
    pub loo_fitted: Col<f64>,
    /// PRESS, its RMSE and the predicted R² `1 − press / SST` (SST centred
    /// with an intercept, uncentred without, as the fit's R²).
    pub press: Press,
}

/// A fit's result as a plain linear model, for the λ = 0 (OLS) case of
/// [`FittedRidge::loo`].
struct AsLinear<'a>(&'a RegressionResult);

impl Augmentable for AsLinear<'_> {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        AugmentSpec::linear(self.0, None)
    }
}

/// A fitted Ridge regression model.
#[derive(Debug, Clone)]
pub struct FittedRidge {
    options: RegressionOptions,
    result: RegressionResult,
    /// Variance factor `M` (ridge sandwich, or `(X'X)⁻¹` for λ = 0), full
    /// dimension, augmented if with_intercept.
    variance_factor: Option<Mat<f64>>,
}

impl HasModelInfo for FittedRidge {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::gaussian("ridge")
    }
}

impl FittedRidge {
    /// Get the options used to fit this model.
    pub fn options(&self) -> &RegressionOptions {
        &self.options
    }

    /// Get the lambda (regularization) parameter.
    pub fn lambda(&self) -> f64 {
        self.options.lambda
    }

    /// Variance factor `M` with `Var(x₀'β̂) = σ² · x₀' M x₀`.
    ///
    /// For λ > 0 this is the ridge sandwich `A X'X A`, `A = (X'X + λP)⁻¹`; for
    /// λ = 0 it is `(X'X)⁻¹`. `x₀` is augmented by a leading 1 when an
    /// intercept is fitted. Rows/columns of aliased columns are zero. `None`
    /// for fits from moments or when the matrix is singular.
    pub fn variance_factor(&self) -> Option<&Mat<f64>> {
        self.variance_factor.as_ref()
    }

    /// Exact leave-one-out diagnostics, without refitting.
    ///
    /// The ridge fit is a linear smoother `ŷ = Hy` of a quadratic objective
    /// with a fixed penalty, so the fit without row `i` predicts it with error
    /// `(yᵢ − ŷᵢ) / (1 − hᵢ)` exactly. `H` and the penalty follow the fit: the
    /// intercept is unpenalised (the centring of [`RidgeRegressor::fit`]) and
    /// `λ` is the effective penalty on the sum-of-squares scale. With
    /// [`LambdaScaling::Glmnet`] that is `λ·n` of the full fit, held fixed for
    /// the leave-one-out fits (a refit on `n − 1` rows through the builder
    /// would use `λ·(n − 1)`). For λ = 0 these are the OLS values of
    /// [`crate::diagnostics::augment`] / [`crate::diagnostics::press`].
    ///
    /// `x` and `y` must be the data the model was fitted on. `press` is `NaN`
    /// when a row has `hᵢ = 1` (e.g. no residual degrees of freedom at λ = 0).
    ///
    /// # Errors
    ///
    /// [`RegressionError::DimensionMismatch`] when `x` / `y` do not match the
    /// fit (including fits from moments, which keep no rows), and
    /// [`RegressionError::SingularMatrix`] if the SVD of the design fails.
    pub fn loo(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<RidgeLoo, RegressionError> {
        let n = x.nrows();
        let result = &self.result;
        for len in [y.nrows(), result.fitted_values.nrows()] {
            if len != n {
                return Err(RegressionError::DimensionMismatch {
                    x_rows: n,
                    y_len: len,
                });
            }
        }
        if x.ncols() != result.coefficients.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.ncols(),
                y_len: result.coefficients.nrows(),
            });
        }
        let with_intercept = result.intercept.is_some();
        let lambda = effective_lambda(&self.options, n);

        let (leverage, loo_residual, loo_fitted) = if lambda == 0.0 {
            let a = augment_with(&AsLinear(result), x, y, None, ResidualType::Response)?;
            (a.leverage, a.loo_residual, a.loo_fitted)
        } else {
            // Thin SVD of the (centred) design: h = [1/n] + Σₖ Uᵢₖ² sₖ² / (sₖ² + λ).
            let design = if with_intercept {
                center_columns(x).0
            } else {
                x.clone()
            };
            let base = if with_intercept { 1.0 / n as f64 } else { 0.0 };
            let leverage = if design.ncols() == 0 {
                Col::from_fn(n, |_| base)
            } else {
                let svd = design
                    .thin_svd()
                    .map_err(|_| RegressionError::SingularMatrix)?;
                let (u, s) = (svd.U(), svd.S().column_vector());
                let shrink: Vec<f64> = (0..s.nrows())
                    .map(|k| s[k] * s[k] / (s[k] * s[k] + lambda))
                    .collect();
                Col::from_fn(n, |i| {
                    let h: f64 = shrink
                        .iter()
                        .enumerate()
                        .map(|(k, &d)| u[(i, k)] * u[(i, k)] * d)
                        .sum();
                    (base + h).min(1.0)
                })
            };
            let loo_residual = Col::from_fn(n, |i| {
                let v = result.residuals[i] / (1.0 - leverage[i]);
                if leverage[i] > LOO_HAT_MAX || !v.is_finite() {
                    f64::NAN
                } else {
                    v
                }
            });
            let loo_fitted = Col::from_fn(n, |i| y[i] - loo_residual[i]);
            (leverage, loo_residual, loo_fitted)
        };
        let press = crate::diagnostics::augment::press_from(&loo_residual, y, None, with_intercept);
        Ok(RidgeLoo {
            leverage,
            loo_residual,
            loo_fitted,
            press,
        })
    }

    /// Estimated covariance of the coefficients, `σ̂² · M` with `M` from
    /// [`Self::variance_factor`]: intercept first when fitted, zero rows and
    /// columns for aliased coefficients. `None` without a variance factor.
    pub fn vcov(&self) -> Option<Mat<f64>> {
        let mse = self.result.mse;
        self.variance_factor()
            .map(|m| Mat::from_fn(m.nrows(), m.ncols(), |i, j| mse * m[(i, j)]))
    }

    /// Leverage `x₀' M x₀` of new rows (NaN if no variance factor is available).
    pub fn leverage_new(&self, x_new: &Mat<f64>) -> Col<f64> {
        match &self.variance_factor {
            Some(m) => leverage_new(m, x_new, self.result.intercept.is_some()),
            None => Col::from_fn(x_new.nrows(), |_| f64::NAN),
        }
    }
}

impl FittedRegressor for FittedRidge {
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
            Some(interval_type) => match &self.variance_factor {
                Some(m) => {
                    let df = self.result.residual_df() as f64;
                    let has_intercept = self.result.intercept.is_some();

                    intervals_from_variance_factor(
                        x,
                        m,
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
            },
        }
    }
}

/// Builder for `RidgeRegressor`.
#[derive(Debug, Clone, Default)]
pub struct RidgeRegressorBuilder {
    builder: RegressionOptionsBuilder,
    conditioning: Option<ConditioningLambda>,
}

impl RidgeRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.builder = self.builder.with_intercept(include);
        self
    }

    /// Set the L2 regularization parameter (lambda).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.builder = self.builder.lambda(lambda);
        self
    }

    /// Set the lambda scaling convention.
    ///
    /// Use `LambdaScaling::Glmnet` to match R's glmnet package behavior.
    pub fn lambda_scaling(mut self, scaling: LambdaScaling) -> Self {
        self.builder = self.builder.lambda_scaling(scaling);
        self
    }

    /// Set whether to compute inference statistics.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.builder = self.builder.compute_inference(compute);
        self
    }

    /// Set the confidence level for confidence intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.builder = self.builder.confidence_level(level);
        self
    }

    /// Set the solver method (QR, SVD, or Cholesky).
    pub fn solve_method(mut self, solver: SolverType) -> Self {
        self.builder = self.builder.solver(solver);
        self
    }

    /// Choose λ by the design's conditioning (see [`ConditioningLambda`]):
    /// `lambda_small` when `κ(XᵀX) ≤ cond_threshold`, the configured
    /// [`Self::lambda`] otherwise. Applies to [`RidgeRegressor::fit`].
    pub fn conditioning_aware(mut self, lambda_small: f64, cond_threshold: f64) -> Self {
        self.conditioning = Some(ConditioningLambda {
            lambda_small,
            cond_threshold,
        });
        self
    }

    /// Set the handling of rows with NaN in `x` or `y` (R's `na.action`;
    /// default [`NaAction::Omit`](crate::core::NaAction::Omit)).
    pub fn na_action(mut self, action: crate::core::NaAction) -> Self {
        self.builder = self.builder.na_action(action);
        self
    }

    /// Build the Ridge regressor.
    pub fn build(self) -> RidgeRegressor {
        RidgeRegressor {
            options: self.builder.build_unchecked(),
            conditioning: self.conditioning,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_basic() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| 2.0 + 3.0 * i as f64);

        let model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(0.01)
            .build();

        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!(fitted.r_squared() > 0.99);
    }

    // ------------------------------------------------------------------
    // Conditioning-aware λ (DataZooDE/anofox-forecast#258)
    // ------------------------------------------------------------------

    /// One fold window of a seasonal-dummy design: `cycles` x 12 monthly
    /// rows, 11 month dummies (January is the baseline), y = base level with
    /// a 3x spike in December plus small deterministic noise.
    fn seasonal_dummy_fold(cycles: usize) -> (Mat<f64>, Col<f64>) {
        let n = cycles * 12;
        let x = Mat::from_fn(n, 11, |i, j| if i % 12 == j + 1 { 1.0 } else { 0.0 });
        let y = Col::from_fn(n, |i| {
            let noise = ((i * 37 % 11) as f64 - 5.0) * 0.05;
            if i % 12 == 11 {
                30.0 + noise
            } else {
                10.0 + noise
            }
        });
        (x, y)
    }

    #[test]
    fn conditioning_aware_uses_small_lambda_on_well_conditioned_seasonal_dummies() {
        // Four cycles (48 rows): every month is seen 4 times.
        let (x, y) = seasonal_dummy_fold(4);
        let kappa = gram_condition_number(&x, true);
        assert!(kappa < DEFAULT_CONDITION_THRESHOLD, "kappa = {kappa}");

        let aware = RidgeRegressor::builder()
            .lambda(1.0)
            .conditioning_aware(DEFAULT_LAMBDA_SMALL, DEFAULT_CONDITION_THRESHOLD)
            .build();
        assert_eq!(aware.select_lambda(&x), DEFAULT_LAMBDA_SMALL);
        let fitted_aware = aware.fit(&x, &y).unwrap();
        assert_eq!(fitted_aware.lambda(), DEFAULT_LAMBDA_SMALL);

        let fixed = RidgeRegressor::builder().lambda(1.0).build();
        let fitted_fixed = fixed.fit(&x, &y).unwrap();
        let ols = RidgeRegressor::builder()
            .lambda(0.0)
            .build()
            .fit(&x, &y)
            .unwrap();

        // December dummy = the seasonal spike amplitude.
        let dec = |f: &FittedRidge| f.coefficients()[10];
        let (a_ols, a_fixed, a_aware) = (dec(&ols), dec(&fitted_fixed), dec(&fitted_aware));
        // lambda = 1 damps the spike noticeably on this well-conditioned fold ...
        assert!(a_fixed < 0.85 * a_ols, "fixed {a_fixed} vs ols {a_ols}");
        // ... the conditioning-aware fit keeps it within 5% of the unpenalised fit.
        assert!(a_aware > 0.95 * a_ols, "aware {a_aware} vs ols {a_ols}");
    }

    #[test]
    fn conditioning_aware_keeps_configured_lambda_on_ill_conditioned_design() {
        // Two nearly collinear regressors.
        let n = 24;
        let x = Mat::from_fn(n, 2, |i, j| {
            let t = i as f64;
            if j == 0 {
                t
            } else {
                t + 1e-3 * ((i * 7 % 5) as f64 - 2.0)
            }
        });
        let y = Col::from_fn(n, |i| 2.0 * i as f64 + 1.0);
        let kappa = gram_condition_number(&x, true);
        assert!(kappa > DEFAULT_CONDITION_THRESHOLD, "kappa = {kappa}");

        let aware = RidgeRegressor::builder()
            .lambda(1.0)
            .conditioning_aware(DEFAULT_LAMBDA_SMALL, DEFAULT_CONDITION_THRESHOLD)
            .build();
        assert_eq!(aware.select_lambda(&x), 1.0);
        assert_eq!(aware.fit(&x, &y).unwrap().lambda(), 1.0);
    }

    #[test]
    fn ridge_without_conditioning_is_unchanged() {
        let (x, y) = seasonal_dummy_fold(2);
        let plain = RidgeRegressor::builder().lambda(1.0).build();
        assert_eq!(plain.select_lambda(&x), 1.0);
        let a = plain.fit(&x, &y).unwrap();
        let b = RidgeRegressor::new(
            RegressionOptionsBuilder::default()
                .lambda(1.0)
                .build_unchecked(),
        )
        .fit(&x, &y)
        .unwrap();
        assert_eq!(a.lambda(), 1.0);
        for j in 0..11 {
            assert_eq!(a.coefficients()[j], b.coefficients()[j]);
        }
    }

    #[test]
    fn gram_condition_number_flags_rank_deficiency() {
        // A full set of 12 dummies plus an intercept is rank-deficient.
        let x = Mat::from_fn(24, 12, |i, j| if i % 12 == j { 1.0 } else { 0.0 });
        assert!(gram_condition_number(&x, true).is_infinite());
        assert!(gram_condition_number(&Mat::<f64>::zeros(0, 0), false).is_infinite());
    }
}
