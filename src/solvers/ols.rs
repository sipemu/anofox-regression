//! Ordinary Least Squares regression solver.

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
    SolverType,
};
use crate::inference::{
    compute_hc_inference, compute_hc_standard_errors, compute_prediction_intervals,
    compute_xtx_inverse_augmented_reduced, compute_xtx_inverse_reduced, CoefficientInference,
    HcInference, HcResult, HcType,
};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::{center_columns, center_vector, detect_constant_columns};
use faer::prelude::Solve;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Ordinary Least Squares regression estimator.
///
/// Uses QR decomposition with column pivoting to handle rank-deficient matrices.
/// Aliased (collinear) coefficients are set to NaN.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{OlsRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = OlsRegressor::builder()
///     .with_intercept(true)
///     .build()
///     .fit(&x, &y)?;
///
/// println!("R² = {}", fitted.r_squared());
/// println!("Coefficients: {:?}", fitted.coefficients());
/// ```
#[derive(Debug, Clone)]
pub struct OlsRegressor {
    options: RegressionOptions,
}

impl OlsRegressor {
    /// Create a new OLS regressor with the given options.
    pub fn new(options: RegressionOptions) -> Self {
        Self { options }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> OlsRegressorBuilder {
        OlsRegressorBuilder::default()
    }

    /// Detect constant columns in the design matrix.
    pub fn detect_constant_columns(x: &Mat<f64>) -> Vec<bool> {
        detect_constant_columns(x, 1e-10)
    }

    /// Check if a matrix has full column rank.
    pub fn is_full_rank(x: &Mat<f64>) -> bool {
        let qr = x.col_piv_qr();
        let r = qr.R();
        let n_cols = x.ncols();

        // Check diagonal elements of R
        for i in 0..n_cols.min(x.nrows()) {
            if r[(i, i)].abs() < 1e-10 {
                return false;
            }
        }
        true
    }

    /// Fit OLS from pre-accumulated moments.
    ///
    /// This is the streaming entry point for very large panels where the
    /// full `N × p` design matrix can't be materialised. Mathematically
    /// identical to [`Self::fit`] when the centered Gram is well-
    /// conditioned: builds `Gc = XᵀX − Σx Σxᵀ / n` and
    /// `cc = Xᵀy − Σx Σy / n`, then solves `Gc β = cc` via Cholesky and
    /// recovers the intercept as `ȳ − x̄ᵀβ`.
    ///
    /// Unlike [`Self::fit`], the streaming path **cannot detect rank-
    /// deficiency through pivoting** — only the moments survive, and column
    /// permutations don't show up. If `Gc` (or `XᵀX` when `with_intercept
    /// = false`) is singular within tolerance, this method returns
    /// [`RegressionError::SingularMatrix`]. Callers that need rank-deficient
    /// handling should fall back to [`Self::fit`] on the explicit rows.
    ///
    /// Statistics that require per-row data (residuals, R², MSE, AIC, …)
    /// are returned as `NaN` / empty since the input rows are not
    /// retained.
    ///
    /// See also: [`crate::solvers::MomentAccumulator`],
    /// [`Self::fit_from_accumulator`].
    pub fn fit_from_moments(
        &self,
        xtx: &Mat<f64>,
        xty: &Col<f64>,
        sum_x: &Col<f64>,
        sum_y: f64,
        n: usize,
    ) -> Result<FittedOls, RegressionError> {
        let p = xtx.nrows();

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
        let min_n = p + if self.options.with_intercept { 1 } else { 0 };
        if n < min_n {
            return Err(RegressionError::InsufficientObservations {
                needed: min_n,
                got: n,
            });
        }

        let n_f = n as f64;

        let (coefficients, intercept) = if self.options.with_intercept {
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
            // Diagonal singularity check for a cleaner error than faer's.
            for i in 0..p {
                if gc[(i, i)].abs() < self.options.rank_tolerance {
                    return Err(RegressionError::SingularMatrix);
                }
            }
            let llt = gc
                .llt(faer::Side::Lower)
                .map_err(|_| RegressionError::SingularMatrix)?;
            let beta = llt.solve(&cc);
            let y_mean = sum_y / n_f;
            let mut b0 = y_mean;
            for i in 0..p {
                b0 -= (sum_x[i] / n_f) * beta[i];
            }
            (beta, Some(b0))
        } else {
            let g = xtx.clone();
            for i in 0..p {
                if g[(i, i)].abs() < self.options.rank_tolerance {
                    return Err(RegressionError::SingularMatrix);
                }
            }
            let llt = g
                .llt(faer::Side::Lower)
                .map_err(|_| RegressionError::SingularMatrix)?;
            let beta = llt.solve(xty);
            (beta, None)
        };

        // Defensive #21 guard on the produced coefficients.
        let aliased = vec![false; p];
        check_coefficients_finite(&coefficients, &aliased)?;

        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.rank = p;
        result.n_parameters = p + if self.options.with_intercept { 1 } else { 0 };
        result.n_observations = n;
        result.aliased = aliased;
        result.rank_tolerance = self.options.rank_tolerance;
        result.confidence_level = self.options.confidence_level;
        // Per-row data unavailable in the streaming flow.
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

        Ok(FittedOls {
            options: self.options.clone(),
            result,
            xtx_inverse: None,
            aliased: vec![false; p],
        })
    }

    /// Fit OLS directly from a [`crate::solvers::MomentAccumulator`].
    /// Convenience wrapper around [`Self::fit_from_moments`].
    pub fn fit_from_accumulator(
        &self,
        acc: &crate::solvers::moments::MomentAccumulator,
    ) -> Result<FittedOls, RegressionError> {
        self.fit_from_moments(acc.xtx(), acc.xty(), acc.sum_x(), acc.sum_y(), acc.n())
    }
}

/// Guard against the "garbage coefficient" failure mode reported in #21:
/// a coefficient that comes back NaN/±∞ or implausibly large for a column
/// the model marked as active. Aliased coefficients are skipped because
/// they are deliberately set to NaN.
pub(crate) fn check_coefficients_finite(
    coefficients: &Col<f64>,
    aliased: &[bool],
) -> Result<(), RegressionError> {
    // 1e120 is well above any legitimate OLS coefficient on well-conditioned
    // data (the reporter saw -1.2e+149) and well below f64::MAX.
    const ABSURD: f64 = 1.0e120;
    for j in 0..coefficients.nrows() {
        if j < aliased.len() && aliased[j] {
            continue;
        }
        let c = coefficients[j];
        if !c.is_finite() {
            return Err(RegressionError::NumericalError(format!(
                "non-finite coefficient at index {}: {} \
                 (likely a numerical / memory corruption issue; see anofox-regression#21)",
                j, c
            )));
        }
        if c.abs() > ABSURD {
            return Err(RegressionError::NumericalError(format!(
                "implausibly large coefficient at index {}: {} \
                 (likely a numerical / memory corruption issue; see anofox-regression#21)",
                j, c
            )));
        }
    }
    Ok(())
}

impl Regressor for OlsRegressor {
    type Fitted = FittedOls;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        // Need at least 2 observations
        if n_samples < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: n_samples,
            });
        }

        // Minimum observations: need at least n_params + 1 for residual df
        // But allow exact fit (n_params = n_samples) for edge cases
        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        if n_samples < n_params {
            return Err(RegressionError::InsufficientObservations {
                needed: n_params,
                got: n_samples,
            });
        }

        // Reject inputs that contain non-finite values. Without this guard
        // a single NaN or ±∞ in `x`/`y` can propagate silently through the
        // centering + QR/SVD path and produce coefficients that look like
        // legitimate floats (often huge values in the 1e120+ range). Catching
        // it here turns a confusing downstream symptom into a clean error.
        // See #21.
        for j in 0..n_features {
            for i in 0..n_samples {
                if !x[(i, j)].is_finite() {
                    return Err(RegressionError::NumericalError(format!(
                        "X contains a non-finite value at row {}, column {}",
                        i, j
                    )));
                }
            }
        }
        for i in 0..n_samples {
            if !y[i].is_finite() {
                return Err(RegressionError::NumericalError(format!(
                    "y contains a non-finite value at row {}",
                    i
                )));
            }
        }

        // Detect constant columns
        let constant_cols = detect_constant_columns(x, self.options.rank_tolerance);

        // If all columns are constant and we have an intercept, all features are aliased
        let all_constant = constant_cols.iter().all(|&c| c);

        if self.options.with_intercept {
            // Center the data
            let (x_centered, x_means) = center_columns(x);
            let (y_centered, y_mean) = center_vector(y);

            // Dispatch to selected solver
            let (coefficients, aliased, rank) = match self.options.solver {
                SolverType::Qr => self.solve_with_qr(&x_centered, &y_centered, &constant_cols)?,
                SolverType::Svd => self.solve_with_svd(&x_centered, &y_centered, &constant_cols)?,
                SolverType::Cholesky => {
                    self.solve_with_cholesky(&x_centered, &y_centered, &constant_cols)?
                }
            };

            // Sanity-check the solver output. Non-aliased coefficients must
            // be finite and within a reasonable magnitude; aliased ones are
            // intentionally NaN. This catches the rare "garbage coefficient"
            // failure mode reported in #21 (e.g. coef = -1.2e+149) and
            // surfaces it to the caller as a clean error instead of a
            // silent corrupt prediction.
            check_coefficients_finite(&coefficients, &aliased)?;

            // Compute intercept: intercept = y_mean - x_means' * coefficients
            let mut intercept = y_mean;
            for j in 0..n_features {
                if !aliased[j] && !coefficients[j].is_nan() {
                    intercept -= x_means[j] * coefficients[j];
                }
            }

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    if !aliased[j] && !coefficients[j].is_nan() {
                        pred += x[(i, j)] * coefficients[j];
                    }
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            // Compute statistics
            let n_params = rank + 1; // +1 for intercept
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

            // Compute (X_aug'X_aug)⁻¹ for prediction intervals (using non-aliased columns)
            let xtx_inverse = compute_xtx_inverse_augmented_reduced(x, &aliased).ok();

            Ok(FittedOls {
                options: self.options.clone(),
                result,
                xtx_inverse,
                aliased: aliased.clone(),
            })
        } else {
            // No intercept case
            if all_constant {
                return Err(RegressionError::AllFeaturesConstant);
            }

            let (coefficients, aliased, rank) = match self.options.solver {
                SolverType::Qr => self.solve_with_qr(x, y, &constant_cols)?,
                SolverType::Svd => self.solve_with_svd(x, y, &constant_cols)?,
                SolverType::Cholesky => self.solve_with_cholesky(x, y, &constant_cols)?,
            };

            // See the corresponding intercept branch above and #21.
            check_coefficients_finite(&coefficients, &aliased)?;

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = 0.0;
                for j in 0..n_features {
                    if !aliased[j] && !coefficients[j].is_nan() {
                        pred += x[(i, j)] * coefficients[j];
                    }
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            let n_params = rank;
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

            // Compute (X'X)⁻¹ for prediction intervals (using non-aliased columns)
            let xtx_inverse = compute_xtx_inverse_reduced(x, &aliased).ok();

            Ok(FittedOls {
                options: self.options.clone(),
                result,
                xtx_inverse,
                aliased: aliased.clone(),
            })
        }
    }
}

impl OlsRegressor {
    /// Solve the least squares problem using QR decomposition with column pivoting.
    fn solve_with_qr(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        constant_cols: &[bool],
    ) -> Result<(Col<f64>, Vec<bool>, usize), RegressionError> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Initialize aliased flags with constant columns
        let mut aliased = constant_cols.to_vec();

        // Perform column-pivoted QR decomposition
        let qr = x.col_piv_qr();
        let q = qr.compute_Q();
        let r = qr.R();
        let perm = qr.P();

        // Build permutation mapping: perm_fwd[i] = which original column is at position i
        // perm_inv[j] = where original column j ended up
        let perm_arr = perm.arrays().0;
        let mut perm_inv: Vec<usize> = vec![0; n_features];
        perm_inv[..n_features].copy_from_slice(&perm_arr[..n_features]);

        // Determine numerical rank from R diagonal
        let mut rank = 0;
        for i in 0..n_features.min(n_samples) {
            if r[(i, i)].abs() > self.options.rank_tolerance {
                rank += 1;
            } else {
                break;
            }
        }

        if rank == 0 {
            // All features are linearly dependent
            let mut coefficients = Col::zeros(n_features);
            for j in 0..n_features {
                coefficients[j] = f64::NAN;
                aliased[j] = true;
            }
            return Ok((coefficients, aliased, 0));
        }

        // Mark aliased columns based on rank
        // If original column j maps to position perm_inv[j] >= rank, it's aliased
        for j in 0..n_features {
            if constant_cols[j] || perm_inv[j] >= rank {
                aliased[j] = true;
            }
        }

        // Solve R * beta_perm = Q' * y for the non-aliased part
        let qty = q.transpose() * y;

        // Back-substitution for upper triangular system
        let mut beta_reduced = Col::zeros(rank);
        for i in (0..rank).rev() {
            let mut sum = qty[i];
            for j in (i + 1)..rank {
                sum -= r[(i, j)] * beta_reduced[j];
            }
            beta_reduced[i] = sum / r[(i, i)];
        }

        // Map back to original column order
        // beta_reduced[perm_inv[j]] is the coefficient for original column j
        let mut coefficients = Col::zeros(n_features);
        for j in 0..n_features {
            if aliased[j] {
                coefficients[j] = f64::NAN;
            } else {
                coefficients[j] = beta_reduced[perm_inv[j]];
            }
        }

        Ok((coefficients, aliased, rank))
    }

    /// Solve the least squares problem using SVD decomposition.
    ///
    /// Uses the pseudoinverse: β = V S⁺ U' y, where S⁺ is the pseudoinverse
    /// of the singular value matrix (thresholded by rank_tolerance).
    fn solve_with_svd(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        constant_cols: &[bool],
    ) -> Result<(Col<f64>, Vec<bool>, usize), RegressionError> {
        let n_features = x.ncols();

        // Initialize aliased flags with constant columns
        let mut aliased = constant_cols.to_vec();

        // Compute SVD: X = U S V'
        let svd = x.svd().map_err(|_| RegressionError::SingularMatrix)?;
        let u = svd.U();
        let s = svd.S();
        let s_col = s.column_vector();
        let v = svd.V();

        let n_sv = s_col.nrows();

        // Determine numerical rank from singular values
        let s_max = (0..n_sv).fold(0.0_f64, |acc, i| acc.max(s_col[i]));
        let threshold = self.options.rank_tolerance * s_max;

        let mut rank = 0;
        for i in 0..n_sv {
            if s_col[i] > threshold {
                rank += 1;
            }
        }

        if rank == 0 {
            let mut coefficients = Col::zeros(n_features);
            for j in 0..n_features {
                coefficients[j] = f64::NAN;
                aliased[j] = true;
            }
            return Ok((coefficients, aliased, 0));
        }

        // Compute β = V S⁺ U' y
        // First: U' y
        let uty = u.transpose() * y;

        // Then: S⁺ U' y (apply pseudoinverse of singular values)
        let mut s_inv_uty = Col::zeros(n_sv);
        for i in 0..n_sv {
            if s_col[i] > threshold {
                s_inv_uty[i] = uty[i] / s_col[i];
            }
        }

        // Finally: V * (S⁺ U' y)
        let coefficients_full = v * &s_inv_uty;

        // Determine aliased columns: columns where all V rows for non-zero
        // singular values contribute negligibly
        // For SVD, we mark constant columns as aliased and set rank accordingly
        for j in 0..n_features {
            if constant_cols[j] {
                aliased[j] = true;
            }
        }

        // Extract coefficients, setting aliased to NaN
        let mut coefficients = Col::zeros(n_features);
        for j in 0..n_features {
            if aliased[j] {
                coefficients[j] = f64::NAN;
            } else {
                coefficients[j] = coefficients_full[j];
            }
        }

        Ok((coefficients, aliased, rank))
    }

    /// Solve the least squares problem using Cholesky decomposition.
    ///
    /// Computes X'X and solves (X'X) β = X'y via Cholesky.
    /// Falls back to QR if X'X is not positive definite (singular).
    fn solve_with_cholesky(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        constant_cols: &[bool],
    ) -> Result<(Col<f64>, Vec<bool>, usize), RegressionError> {
        let n_features = x.ncols();

        // Compute X'X and X'y
        let xtx = x.transpose() * x;
        let xty = x.transpose() * y;

        // Attempt Cholesky decomposition
        match xtx.llt(faer::Side::Lower) {
            Ok(llt) => {
                // Solve (X'X) β = X'y
                let coefficients = llt.solve(&xty);

                let aliased = constant_cols.to_vec();
                let rank = aliased.iter().filter(|&&a| !a).count();

                // Set aliased coefficients to NaN
                let mut coeff_out = Col::zeros(n_features);
                for j in 0..n_features {
                    if aliased[j] {
                        coeff_out[j] = f64::NAN;
                    } else {
                        coeff_out[j] = coefficients[j];
                    }
                }

                Ok((coeff_out, aliased, rank))
            }
            Err(_) => {
                // Cholesky failed (X'X not positive definite), fall back to QR
                self.solve_with_qr(x, y, constant_cols)
            }
        }
    }

    /// Compute fit statistics and optionally inference statistics.
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

        // Compute y mean
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        // Compute TSS (total sum of squares)
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        // Compute RSS (residual sum of squares)
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        // R-squared
        let r_squared = if tss > 0.0 {
            (1.0 - rss / tss).clamp(0.0, 1.0)
        } else if rss < 1e-10 {
            1.0
        } else {
            0.0
        };

        // Adjusted R-squared
        let df_total = (n - 1) as f64;
        let df_resid = (n - n_params) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        // MSE and RMSE
        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        // F-statistic
        let ess = tss - rss; // Explained sum of squares
        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 }) as f64;
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && mse > 0.0 {
            (ess / df_model) / mse
        } else {
            f64::NAN
        };

        // F p-value
        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            let f_dist = FisherSnedecor::new(df_model, df_resid).ok();
            f_dist.map_or(f64::NAN, |d| 1.0 - d.cdf(f_statistic))
        } else {
            f64::NAN
        };

        // Information criteria
        let log_likelihood = if mse > 0.0 {
            -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + mse.ln())
        } else {
            f64::NAN
        };

        let k = n_params as f64; // Number of parameters
        let aic = if log_likelihood.is_finite() {
            2.0 * k - 2.0 * log_likelihood
        } else {
            f64::NAN
        };

        let aicc = if log_likelihood.is_finite() && (n as f64 - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n as f64 - k - 1.0)
        } else {
            f64::NAN
        };

        let bic = if log_likelihood.is_finite() {
            k * (n as f64).ln() - 2.0 * log_likelihood
        } else {
            f64::NAN
        };

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
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.f_statistic = f_statistic;
        result.f_pvalue = f_pvalue;
        result.aic = aic;
        result.aicc = aicc;
        result.bic = bic;
        result.log_likelihood = log_likelihood;
        result.confidence_level = self.options.confidence_level;

        // Compute inference statistics if requested
        if self.options.compute_inference {
            self.compute_inference(x, &mut result)?;
        }

        Ok(result)
    }

    /// Compute inference statistics (standard errors, t-stats, p-values, CIs).
    fn compute_inference(
        &self,
        x: &Mat<f64>,
        result: &mut RegressionResult,
    ) -> Result<(), RegressionError> {
        let df = result.residual_df() as f64;

        if df <= 0.0 || !result.mse.is_finite() {
            return Ok(());
        }

        // Use the augmented design matrix method for models with intercept
        // This computes SE for both intercept and coefficients correctly
        if let Some(intercept) = result.intercept {
            match CoefficientInference::standard_errors_with_intercept(
                x,
                result.mse,
                &result.aliased,
            ) {
                Ok((se, se_int)) => {
                    // t-statistics
                    let t_stats = CoefficientInference::t_statistics(&result.coefficients, &se);

                    // p-values
                    let p_vals = CoefficientInference::p_values(&t_stats, df);

                    // Confidence intervals
                    let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
                        &result.coefficients,
                        &se,
                        df,
                        self.options.confidence_level,
                    );

                    result.std_errors = Some(se);
                    result.t_statistics = Some(t_stats);
                    result.p_values = Some(p_vals);
                    result.conf_interval_lower = Some(ci_lower);
                    result.conf_interval_upper = Some(ci_upper);

                    // Intercept inference
                    let t_int = if se_int > 0.0 {
                        intercept / se_int
                    } else {
                        f64::NAN
                    };

                    use statrs::distribution::StudentsT;
                    let t_dist = StudentsT::new(0.0, 1.0, df).ok();
                    let p_int = if t_int.is_finite() {
                        t_dist.map_or(f64::NAN, |d| 2.0 * (1.0 - d.cdf(t_int.abs())))
                    } else {
                        f64::NAN
                    };

                    let t_crit = t_dist.map_or(f64::NAN, |d| {
                        d.inverse_cdf(1.0 - (1.0 - self.options.confidence_level) / 2.0)
                    });
                    let ci_int = (intercept - t_crit * se_int, intercept + t_crit * se_int);

                    result.intercept_std_error = Some(se_int);
                    result.intercept_t_statistic = Some(t_int);
                    result.intercept_p_value = Some(p_int);
                    result.intercept_conf_interval = Some(ci_int);
                }
                Err(_) => {
                    // Failed to compute standard errors, leave as None
                }
            }
        } else {
            // No intercept case - use regular SE computation
            match CoefficientInference::standard_errors(x, result.mse, &result.aliased) {
                Ok(se) => {
                    // t-statistics
                    let t_stats = CoefficientInference::t_statistics(&result.coefficients, &se);

                    // p-values
                    let p_vals = CoefficientInference::p_values(&t_stats, df);

                    // Confidence intervals
                    let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
                        &result.coefficients,
                        &se,
                        df,
                        self.options.confidence_level,
                    );

                    result.std_errors = Some(se);
                    result.t_statistics = Some(t_stats);
                    result.p_values = Some(p_vals);
                    result.conf_interval_lower = Some(ci_lower);
                    result.conf_interval_upper = Some(ci_upper);
                }
                Err(_) => {
                    // Failed to compute standard errors, leave as None
                }
            }
        }

        Ok(())
    }
}

/// A fitted OLS regression model.
#[derive(Debug, Clone)]
pub struct FittedOls {
    options: RegressionOptions,
    result: RegressionResult,
    /// (X'X)⁻¹ or (X_aug'X_aug)⁻¹ for prediction intervals (reduced to non-aliased columns)
    xtx_inverse: Option<Mat<f64>>,
    /// Which columns are aliased (collinear or constant)
    aliased: Vec<bool>,
}

impl FittedOls {
    /// Get the options used to fit this model.
    pub fn options(&self) -> &RegressionOptions {
        &self.options
    }
}

impl FittedRegressor for FittedOls {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut predictions = Col::zeros(n_samples);

        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n_samples {
            let mut pred = intercept;
            for j in 0..n_features {
                if !self.result.aliased[j] && !self.result.coefficients[j].is_nan() {
                    pred += x[(i, j)] * self.result.coefficients[j];
                }
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
                // Need xtx_inverse to compute intervals
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
                        // Cannot compute intervals without stored inverse
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

impl FittedOls {
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

impl FittedOls {
    /// Compute HC (heteroskedasticity-consistent) standard errors.
    ///
    /// Requires the original feature matrix `x` (without intercept column).
    /// This is a post-hoc computation — you can try different HC variants
    /// without re-fitting the model.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix used to fit the model
    /// * `hc_type` - Which HC variant to compute (HC0–HC3)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let fitted = OlsRegressor::builder()
    ///     .with_intercept(true)
    ///     .build()
    ///     .fit(&x, &y)?;
    ///
    /// let hc_se = fitted.hc_standard_errors(&x, HcType::HC1)?;
    /// println!("Robust SE: {:?}", hc_se.std_errors);
    /// ```
    pub fn hc_standard_errors(
        &self,
        x: &Mat<f64>,
        hc_type: HcType,
    ) -> Result<HcResult, &'static str> {
        let with_intercept = self.result.intercept.is_some();
        compute_hc_standard_errors(
            x,
            &self.result.residuals,
            &self.aliased,
            with_intercept,
            hc_type,
        )
    }

    /// Compute full HC inference: standard errors, t-statistics, p-values,
    /// and confidence intervals.
    ///
    /// Requires the original feature matrix `x` (without intercept column).
    /// Uses the model's confidence level by default.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix used to fit the model
    /// * `hc_type` - Which HC variant to compute (HC0–HC3)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let fitted = OlsRegressor::builder()
    ///     .with_intercept(true)
    ///     .compute_inference(true)
    ///     .confidence_level(0.95)
    ///     .build()
    ///     .fit(&x, &y)?;
    ///
    /// let hc = fitted.hc_inference(&x, HcType::HC3)?;
    /// println!("HC3 t-stats: {:?}", hc.t_statistics);
    /// println!("HC3 p-values: {:?}", hc.p_values);
    /// ```
    pub fn hc_inference(&self, x: &Mat<f64>, hc_type: HcType) -> Result<HcInference, &'static str> {
        let with_intercept = self.result.intercept.is_some();
        compute_hc_inference(
            x,
            &self.result.coefficients,
            self.result.intercept,
            &self.result.residuals,
            &self.aliased,
            with_intercept,
            hc_type,
            self.options.confidence_level,
        )
    }
}

/// Builder for `OlsRegressor`.
#[derive(Debug, Clone, Default)]
pub struct OlsRegressorBuilder {
    builder: RegressionOptionsBuilder,
}

impl OlsRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.builder = self.builder.with_intercept(include);
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

    /// Set the rank tolerance for QR decomposition.
    pub fn rank_tolerance(mut self, tol: f64) -> Self {
        self.builder = self.builder.rank_tolerance(tol);
        self
    }

    /// Set the solver method (QR, SVD, or Cholesky).
    pub fn solve_method(mut self, solver: SolverType) -> Self {
        self.builder = self.builder.solver(solver);
        self
    }

    /// Build the OLS regressor.
    pub fn build(self) -> OlsRegressor {
        // Use unchecked build since OLS doesn't use lambda/alpha
        OlsRegressor::new(self.builder.build_unchecked())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fit() {
        let x = Mat::from_fn(5, 1, |i, _| i as f64);
        let y = Col::from_fn(5, |i| 2.0 + 3.0 * i as f64);

        let model = OlsRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!((fitted.coefficients()[0] - 3.0).abs() < 1e-10);
        assert!((fitted.intercept().expect("intercept exists") - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict() {
        let x = Mat::from_fn(5, 1, |i, _| i as f64);
        let y = Col::from_fn(5, |i| 2.0 + 3.0 * i as f64);

        let model = OlsRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        let x_new = Mat::from_fn(2, 1, |i, _| (i + 10) as f64);
        let preds = fitted.predict(&x_new);

        assert!((preds[0] - (2.0 + 3.0 * 10.0)).abs() < 1e-10);
        assert!((preds[1] - (2.0 + 3.0 * 11.0)).abs() < 1e-10);
    }

    // Tests for the #21 input/output guards.

    #[test]
    fn rejects_nan_in_x() {
        let mut x = Mat::from_fn(5, 1, |i, _| i as f64);
        x[(2, 0)] = f64::NAN;
        let y = Col::from_fn(5, |i| 2.0 + 3.0 * i as f64);
        let err = OlsRegressor::builder()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap_err();
        match err {
            RegressionError::NumericalError(msg) => {
                assert!(msg.contains("X contains a non-finite value"));
            }
            other => panic!("expected NumericalError, got {:?}", other),
        }
    }

    #[test]
    fn rejects_inf_in_y() {
        let x = Mat::from_fn(5, 1, |i, _| i as f64);
        let y = Col::from_fn(5, |i| if i == 3 { f64::INFINITY } else { i as f64 });
        let err = OlsRegressor::builder()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap_err();
        match err {
            RegressionError::NumericalError(msg) => {
                assert!(msg.contains("y contains a non-finite value"));
            }
            other => panic!("expected NumericalError, got {:?}", other),
        }
    }

    #[test]
    fn coefficient_guard_accepts_aliased_nan() {
        // A perfectly-aliased coefficient is intentionally NaN; the guard
        // must skip those.
        let coefs = Col::from_fn(3, |i| if i == 1 { f64::NAN } else { 1.5 });
        let aliased = [false, true, false];
        check_coefficients_finite(&coefs, &aliased).expect("aliased NaN must pass");
    }

    #[test]
    fn coefficient_guard_rejects_active_nan() {
        let coefs = Col::from_fn(3, |i| if i == 1 { f64::NAN } else { 1.5 });
        let aliased = [false, false, false];
        let err = check_coefficients_finite(&coefs, &aliased).unwrap_err();
        match err {
            RegressionError::NumericalError(msg) => {
                assert!(msg.contains("non-finite coefficient at index 1"));
                assert!(msg.contains("#21"));
            }
            other => panic!("expected NumericalError, got {:?}", other),
        }
    }

    #[test]
    fn coefficient_guard_rejects_implausibly_large() {
        // The reporter's symptom was values like -1.2e+149.
        let coefs = Col::from_fn(3, |i| if i == 2 { -1.2e149 } else { 0.5 });
        let aliased = [false, false, false];
        let err = check_coefficients_finite(&coefs, &aliased).unwrap_err();
        match err {
            RegressionError::NumericalError(msg) => {
                assert!(msg.contains("implausibly large coefficient at index 2"));
                assert!(msg.contains("#21"));
            }
            other => panic!("expected NumericalError, got {:?}", other),
        }
    }
}
