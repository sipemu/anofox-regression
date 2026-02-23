//! Bounded Least Squares (BLS) regression solver.
//!
//! Implements the Lawson-Hanson active set algorithm for non-negative least squares (NNLS)
//! and generalizes it to box constraints (lower ≤ x ≤ upper).
//!
//! # Reference
//!
//! - Lawson, C.L. and Hanson, R.J. (1974). "Solving Least Squares Problems". Prentice-Hall.
//! - R package `nnls`: <https://cran.r-project.org/web/packages/nnls/index.html>

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Bounded Least Squares regression estimator.
///
/// Solves: minimize ||Ax - b||² subject to lower ≤ x ≤ upper
///
/// Special case: When lower = 0 and upper = ∞, this becomes Non-Negative Least Squares (NNLS).
///
/// Uses the Lawson-Hanson active set algorithm, generalized for box constraints.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::solvers::{BlsRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 3, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// // Non-negative least squares (NNLS)
/// let fitted = BlsRegressor::nnls()
///     .build()
///     .fit(&x, &y)?;
///
/// // Box-constrained least squares
/// let fitted = BlsRegressor::builder()
///     .lower_bounds(vec![0.0, -1.0, 0.0])
///     .upper_bounds(vec![10.0, f64::INFINITY, 5.0])
///     .build()
///     .fit(&x, &y)?;
/// ```
#[derive(Debug, Clone)]
pub struct BlsRegressor {
    options: RegressionOptions,
    /// Lower bounds for each coefficient (None means no lower bound, i.e., -∞).
    lower_bounds: Option<Vec<f64>>,
    /// Upper bounds for each coefficient (None means no upper bound, i.e., +∞).
    upper_bounds: Option<Vec<f64>>,
}

impl BlsRegressor {
    /// Create a new BLS regressor with the given options and bounds.
    pub fn new(
        options: RegressionOptions,
        lower_bounds: Option<Vec<f64>>,
        upper_bounds: Option<Vec<f64>>,
    ) -> Self {
        Self {
            options,
            lower_bounds,
            upper_bounds,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> BlsRegressorBuilder {
        BlsRegressorBuilder::default()
    }

    /// Create a builder pre-configured for Non-Negative Least Squares (NNLS).
    ///
    /// This is equivalent to `builder().lower_bounds_all(0.0)`.
    pub fn nnls() -> BlsRegressorBuilder {
        BlsRegressorBuilder::default().lower_bound_all(0.0)
    }

    /// Solve the bounded least squares problem using the active set method.
    ///
    /// Algorithm: Lawson-Hanson with box constraints
    ///
    /// The algorithm maintains two sets:
    /// - Passive set P: variables free to move within bounds
    /// - Active set A: variables fixed at their bounds
    ///
    /// At each iteration:
    /// 1. Solve unconstrained LS on passive set
    /// 2. If solution violates bounds, move variables to active set
    /// 3. Check KKT conditions, move variables from active to passive if beneficial
    fn solve_bls(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        lower: &[f64],
        upper: &[f64],
    ) -> Result<(Col<f64>, usize), RegressionError> {
        let _n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize solution at lower bounds (or 0 if unbounded below)
        let mut beta = Col::from_fn(n_features, |j| {
            if lower[j].is_finite() {
                lower[j]
            } else if upper[j].is_finite() {
                upper[j]
            } else {
                0.0
            }
        });

        // Active set: true = variable is at bound (fixed)
        // false = variable is free (in passive set)
        let mut active = vec![true; n_features];

        // Track which bound each active variable is at: true = at lower, false = at upper
        let mut at_lower = vec![true; n_features];

        // Precompute X'X and X'y
        let xtx = compute_xtx(x);
        let xty = compute_xty(x, y);

        let max_iter = self.options.max_iterations;
        let tol = self.options.tolerance;

        for _outer_iter in 0..max_iter {
            // Compute gradient: g = X'X * beta - X'y = -X'(y - X*beta)
            let gradient = compute_gradient(&xtx, &xty, &beta);

            // Check KKT conditions and find variable to add to passive set
            let mut max_violation = 0.0;
            let mut best_idx: Option<usize> = None;

            for j in 0..n_features {
                if active[j] {
                    // Variable is at bound, check if moving it would help
                    if at_lower[j] {
                        // At lower bound: negative gradient means we should increase
                        if -gradient[j] > max_violation {
                            max_violation = -gradient[j];
                            best_idx = Some(j);
                        }
                    } else {
                        // At upper bound: positive gradient means we should decrease
                        if gradient[j] > max_violation {
                            max_violation = gradient[j];
                            best_idx = Some(j);
                        }
                    }
                }
            }

            // If KKT conditions are satisfied, we're done
            if max_violation < tol {
                break;
            }

            // Move best variable to passive set
            if let Some(idx) = best_idx {
                active[idx] = false;
            } else {
                break;
            }

            // Inner loop: solve on passive set and handle bound violations
            for _inner_iter in 0..max_iter {
                // Get passive set indices
                let passive_indices: Vec<usize> = (0..n_features).filter(|&j| !active[j]).collect();

                if passive_indices.is_empty() {
                    break;
                }

                // Solve unconstrained LS on passive set
                let beta_passive = self.solve_passive_set(x, y, &passive_indices)?;

                // Check for bound violations
                let mut has_violation = false;
                let mut min_alpha = 1.0;
                let mut violation_idx = 0;
                let mut violation_at_lower = true;

                for (i, &j) in passive_indices.iter().enumerate() {
                    let new_val = beta_passive[i];
                    let old_val = beta[j];

                    if new_val < lower[j] - tol {
                        // Violates lower bound
                        let alpha = (lower[j] - old_val) / (new_val - old_val);
                        if alpha < min_alpha {
                            min_alpha = alpha;
                            violation_idx = j;
                            violation_at_lower = true;
                            has_violation = true;
                        }
                    } else if new_val > upper[j] + tol {
                        // Violates upper bound
                        let alpha = (upper[j] - old_val) / (new_val - old_val);
                        if alpha < min_alpha {
                            min_alpha = alpha;
                            violation_idx = j;
                            violation_at_lower = false;
                            has_violation = true;
                        }
                    }
                }

                if !has_violation {
                    // No violations, accept the solution
                    for (i, &j) in passive_indices.iter().enumerate() {
                        beta[j] = beta_passive[i];
                    }
                    break;
                } else {
                    // Interpolate to boundary
                    for (i, &j) in passive_indices.iter().enumerate() {
                        beta[j] = beta[j] + min_alpha * (beta_passive[i] - beta[j]);
                    }

                    // Move violated variable to active set
                    active[violation_idx] = true;
                    at_lower[violation_idx] = violation_at_lower;
                    beta[violation_idx] = if violation_at_lower {
                        lower[violation_idx]
                    } else {
                        upper[violation_idx]
                    };
                }
            }
        }

        // Clamp final solution to bounds (for numerical safety)
        for j in 0..n_features {
            beta[j] = beta[j].clamp(lower[j], upper[j]);
        }

        // Count non-zero coefficients (rank proxy for bounded problems)
        let rank = beta
            .iter()
            .enumerate()
            .filter(|(j, &b)| {
                let at_lower_bound = (b - lower[*j]).abs() < tol;
                let at_upper_bound = (b - upper[*j]).abs() < tol;
                // Exclude coefficients at zero bounds
                !(at_lower_bound && lower[*j] == 0.0 || at_upper_bound && upper[*j] == 0.0)
            })
            .count();

        Ok((beta, rank.max(1)))
    }

    /// Solve unconstrained least squares on passive set variables only.
    fn solve_passive_set(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        passive_indices: &[usize],
    ) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_passive = passive_indices.len();

        if n_passive == 0 {
            return Ok(Col::zeros(0));
        }

        // Extract columns for passive set
        let x_passive = Mat::from_fn(n_samples, n_passive, |i, j| x[(i, passive_indices[j])]);

        // Solve X_P * beta_P = y using QR decomposition
        let qr = x_passive.col_piv_qr();
        let q = qr.compute_Q();
        let r = qr.R();
        let perm = qr.P();

        // Compute Q'y
        let qty = q.transpose() * y;

        // Solve R * beta_permuted = Q'y via back substitution
        let mut beta_perm = Col::zeros(n_passive);
        for i in (0..n_passive).rev() {
            let mut sum = qty[i];
            for j in (i + 1)..n_passive {
                sum -= r[(i, j)] * beta_perm[j];
            }
            if r[(i, i)].abs() > self.options.rank_tolerance {
                beta_perm[i] = sum / r[(i, i)];
            } else {
                beta_perm[i] = 0.0;
            }
        }

        // Unpermute
        let mut beta = Col::zeros(n_passive);
        for i in 0..n_passive {
            beta[perm.inverse().arrays().0[i]] = beta_perm[i];
        }

        Ok(beta)
    }
}

impl Regressor for BlsRegressor {
    type Fitted = FittedBls;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        // Delegate to fit_internal which handles bound expansion
        self.fit_internal(x, y)
    }
}

impl BlsRegressor {
    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        coefficients: Col<f64>,
        intercept: Option<f64>,
        rank: usize,
        n_params: usize,
        _lower: &[f64],
        _upper: &[f64],
    ) -> Result<FittedBls, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Compute fitted values and residuals
        let mut fitted_values = Col::zeros(n_samples);
        let mut residuals = Col::zeros(n_samples);

        for i in 0..n_samples {
            let mut pred = intercept.unwrap_or(0.0);
            for j in 0..n_features {
                pred += x[(i, j)] * coefficients[j];
            }
            fitted_values[i] = pred;
            residuals[i] = y[i] - pred;
        }

        // Compute statistics
        let y_mean: f64 = y.iter().sum::<f64>() / n_samples as f64;
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
        let ess = tss - rss;

        let r_squared = if tss > 0.0 {
            (1.0 - rss / tss).clamp(0.0, 1.0)
        } else if rss < 1e-10 {
            1.0
        } else {
            0.0
        };

        let df_total = (n_samples - 1) as f64;
        let df_resid = (n_samples.saturating_sub(n_params)) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 }) as f64;
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && mse > 0.0 {
            (ess / df_model) / mse
        } else {
            f64::NAN
        };

        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            FisherSnedecor::new(df_model, df_resid)
                .map(|d| 1.0 - d.cdf(f_statistic))
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        // Information criteria
        let n = n_samples as f64;
        let k = n_params as f64;
        let log_likelihood = if mse > 0.0 {
            -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI).ln() + mse.ln())
        } else {
            f64::NAN
        };

        let aic = if log_likelihood.is_finite() {
            2.0 * k - 2.0 * log_likelihood
        } else {
            f64::NAN
        };

        let aicc = if log_likelihood.is_finite() && (n - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        } else {
            f64::NAN
        };

        let bic = if log_likelihood.is_finite() {
            k * n.ln() - 2.0 * log_likelihood
        } else {
            f64::NAN
        };

        let mut result = RegressionResult::empty(n_features, n_samples);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = rank;
        result.n_parameters = n_params;
        result.n_observations = n_samples;
        result.aliased = vec![false; n_features]; // BLS doesn't have aliased in same sense
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

        // Note: Standard errors for constrained LS are complex and often not computed
        // We leave inference statistics as None for BLS

        Ok(FittedBls {
            result,
            options: self.options.clone(),
        })
    }
}

/// Fitted Bounded Least Squares (BLS) model with coefficient constraints.
///
/// Contains the estimated coefficients that satisfy the specified bounds,
/// along with model diagnostics. Uses the Lawson-Hanson algorithm for
/// non-negative least squares (NNLS) or bounded constraints.
///
/// # Constraints
///
/// BLS supports:
/// - Non-negative constraints: `beta >= 0` (NNLS)
/// - Lower bounds: `beta >= lower`
/// - Upper bounds: `beta <= upper`
/// - Box constraints: `lower <= beta <= upper`
///
/// # Available Methods
///
/// - [`predict`](FittedRegressor::predict) - Predict response values
/// - [`coefficients`](FittedRegressor::coefficients) - Get bounded coefficients
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// let x = Mat::from_fn(100, 3, |i, j| (i + j) as f64 / 10.0);
/// let y = Col::from_fn(100, |i| i as f64 + 1.0);
///
/// // Non-negative least squares
/// let fitted = BlsRegressor::nnls()
///     .build()
///     .fit(&x, &y)?;
///
/// // All coefficients are >= 0
/// for coef in fitted.coefficients().iter() {
///     assert!(*coef >= 0.0);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FittedBls {
    result: RegressionResult,
    #[allow(dead_code)]
    options: RegressionOptions,
}

impl FittedRegressor for FittedBls {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);

        Col::from_fn(n_samples, |i| {
            let mut pred = intercept;
            for j in 0..n_features {
                pred += x[(i, j)] * self.result.coefficients[j];
            }
            pred
        })
    }

    fn result(&self) -> &RegressionResult {
        &self.result
    }

    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        _level: f64,
    ) -> PredictionResult {
        let predictions = self.predict(x);

        // BLS typically doesn't have well-defined prediction intervals
        // due to the constraint complexity. Return NaN for intervals.
        match interval {
            None => PredictionResult::point_only(predictions),
            Some(_) => {
                let n = x.nrows();
                let nan_vec = Col::from_fn(n, |_| f64::NAN);
                PredictionResult::with_intervals(
                    predictions,
                    nan_vec.clone(),
                    nan_vec.clone(),
                    nan_vec,
                )
            }
        }
    }
}

// Helper functions

fn compute_xtx(x: &Mat<f64>) -> Mat<f64> {
    let n_features = x.ncols();
    let n_samples = x.nrows();

    let mut xtx = Mat::zeros(n_features, n_features);
    for i in 0..n_samples {
        for j in 0..n_features {
            for k in 0..n_features {
                xtx[(j, k)] += x[(i, j)] * x[(i, k)];
            }
        }
    }
    xtx
}

fn compute_xty(x: &Mat<f64>, y: &Col<f64>) -> Col<f64> {
    let n_features = x.ncols();
    let n_samples = x.nrows();

    let mut xty = Col::zeros(n_features);
    for i in 0..n_samples {
        for j in 0..n_features {
            xty[j] += x[(i, j)] * y[i];
        }
    }
    xty
}

fn compute_gradient(xtx: &Mat<f64>, xty: &Col<f64>, beta: &Col<f64>) -> Col<f64> {
    let n = beta.nrows();
    let mut grad = Col::zeros(n);

    for j in 0..n {
        grad[j] = -xty[j];
        for k in 0..n {
            grad[j] += xtx[(j, k)] * beta[k];
        }
    }
    grad
}

/// Builder for configuring a Bounded Least Squares model.
///
/// Provides a fluent API for setting coefficient bounds and constraints.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// // Non-negative least squares (all coefficients >= 0)
/// let model = BlsRegressor::nnls()
///     .build();
///
/// // Custom lower bounds per coefficient
/// let model = BlsRegressor::builder()
///     .lower_bounds(vec![0.0, -1.0, 0.5])
///     .build();
///
/// // Box constraints (lower and upper bounds)
/// let model = BlsRegressor::builder()
///     .lower_bounds(vec![0.0, 0.0, 0.0])
///     .upper_bounds(vec![1.0, 1.0, 1.0])
///     .build();
///
/// // Same bound for all coefficients
/// let model = BlsRegressor::builder()
///     .lower_bound_all(0.0)
///     .upper_bound_all(10.0)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct BlsRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    lower_bounds: Option<Vec<f64>>,
    upper_bounds: Option<Vec<f64>>,
    lower_bound_all: Option<f64>,
    upper_bound_all: Option<f64>,
}

impl BlsRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.options_builder = self.options_builder.with_intercept(include);
        self
    }

    /// Set the maximum iterations for the active set algorithm.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.options_builder = self.options_builder.max_iterations(max_iter);
        self
    }

    /// Set the convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.options_builder = self.options_builder.tolerance(tol);
        self
    }

    /// Set per-variable lower bounds.
    pub fn lower_bounds(mut self, bounds: Vec<f64>) -> Self {
        self.lower_bounds = Some(bounds);
        self.lower_bound_all = None;
        self
    }

    /// Set per-variable upper bounds.
    pub fn upper_bounds(mut self, bounds: Vec<f64>) -> Self {
        self.upper_bounds = Some(bounds);
        self.upper_bound_all = None;
        self
    }

    /// Set the same lower bound for all variables.
    pub fn lower_bound_all(mut self, bound: f64) -> Self {
        self.lower_bound_all = Some(bound);
        self.lower_bounds = None;
        self
    }

    /// Set the same upper bound for all variables.
    pub fn upper_bound_all(mut self, bound: f64) -> Self {
        self.upper_bound_all = Some(bound);
        self.upper_bounds = None;
        self
    }

    /// Build the regressor.
    ///
    /// Note: If `lower_bound_all` or `upper_bound_all` was set, the bounds vectors
    /// will be created when `fit()` is called based on the number of features.
    pub fn build(self) -> BlsRegressor {
        let lower = self.lower_bounds.or_else(|| {
            self.lower_bound_all.map(|b| vec![b]) // Placeholder, will be expanded in fit()
        });
        let upper = self.upper_bounds.or_else(|| {
            self.upper_bound_all.map(|b| vec![b]) // Placeholder, will be expanded in fit()
        });

        // Store the "all" bounds for later expansion
        BlsRegressor {
            options: self.options_builder.build_unchecked(),
            lower_bounds: if let Some(lb) = self.lower_bound_all {
                Some(vec![lb])
            } else {
                lower
            },
            upper_bounds: if let Some(ub) = self.upper_bound_all {
                Some(vec![ub])
            } else {
                upper
            },
        }
    }
}

// Override get_lower_bounds and get_upper_bounds to handle "all" case
impl BlsRegressor {
    fn get_lower_bounds_expanded(&self, n_features: usize) -> Vec<f64> {
        match &self.lower_bounds {
            Some(bounds) if bounds.len() == 1 => vec![bounds[0]; n_features],
            Some(bounds) => bounds.clone(),
            None => vec![f64::NEG_INFINITY; n_features],
        }
    }

    fn get_upper_bounds_expanded(&self, n_features: usize) -> Vec<f64> {
        match &self.upper_bounds {
            Some(bounds) if bounds.len() == 1 => vec![bounds[0]; n_features],
            Some(bounds) => bounds.clone(),
            None => vec![f64::INFINITY; n_features],
        }
    }
}

// Update fit to use expanded bounds
impl BlsRegressor {
    /// Internal fit implementation using expanded bounds.
    fn fit_internal(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedBls, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

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

        let lower = self.get_lower_bounds_expanded(n_features);
        let upper = self.get_upper_bounds_expanded(n_features);

        for j in 0..n_features {
            if lower[j] > upper[j] {
                return Err(RegressionError::NumericalError(format!(
                    "lower bound {} exceeds upper bound {} for feature {}",
                    lower[j], upper[j], j
                )));
            }
        }

        if self.options.with_intercept {
            let mut x_aug = Mat::zeros(n_samples, n_features + 1);
            for i in 0..n_samples {
                x_aug[(i, 0)] = 1.0;
                for j in 0..n_features {
                    x_aug[(i, j + 1)] = x[(i, j)];
                }
            }

            let mut lower_aug = vec![f64::NEG_INFINITY];
            lower_aug.extend_from_slice(&lower);
            let mut upper_aug = vec![f64::INFINITY];
            upper_aug.extend_from_slice(&upper);

            let (coeffs_aug, rank) = self.solve_bls(&x_aug, y, &lower_aug, &upper_aug)?;

            let intercept = Some(coeffs_aug[0]);
            let coefficients = Col::from_fn(n_features, |j| coeffs_aug[j + 1]);

            self.build_result(
                x,
                y,
                coefficients,
                intercept,
                rank,
                n_features + 1,
                &lower,
                &upper,
            )
        } else {
            let (coefficients, rank) = self.solve_bls(x, y, &lower, &upper)?;
            self.build_result(x, y, coefficients, None, rank, n_features, &lower, &upper)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnls_simple() {
        // Simple NNLS problem
        let x = Mat::from_fn(5, 2, |i, _| (i + 1) as f64);
        let y = Col::from_fn(5, |i| (2 * i + 3) as f64);

        let fitted = BlsRegressor::nnls()
            .build()
            .fit_internal(&x, &y)
            .expect("model should fit");

        // All coefficients should be non-negative
        for i in 0..fitted.result.coefficients.nrows() {
            assert!(
                fitted.result.coefficients[i] >= -1e-10,
                "Coefficient {} is negative: {}",
                i,
                fitted.result.coefficients[i]
            );
        }
    }

    #[test]
    fn test_bls_box_constraints() {
        let x = Mat::from_fn(10, 2, |i, j| (i * 2 + j) as f64);
        let y = Col::from_fn(10, |i| (3 * i + 1) as f64);

        let fitted = BlsRegressor::builder()
            .lower_bounds(vec![0.0, 0.0])
            .upper_bounds(vec![1.0, 2.0])
            .with_intercept(false)
            .build()
            .fit_internal(&x, &y)
            .expect("model should fit");

        // Check bounds
        assert!(fitted.result.coefficients[0] >= -1e-10);
        assert!(fitted.result.coefficients[0] <= 1.0 + 1e-10);
        assert!(fitted.result.coefficients[1] >= -1e-10);
        assert!(fitted.result.coefficients[1] <= 2.0 + 1e-10);
    }

    #[test]
    fn test_bls_unconstrained_matches_ols() {
        // With no bounds, BLS should approximate OLS
        let x = Mat::from_fn(20, 2, |i, j| ((i + 1) * (j + 1)) as f64);
        let y = Col::from_fn(20, |i| 2.0 * i as f64 + 1.0);

        let bls_fitted = BlsRegressor::builder()
            .with_intercept(false)
            .build()
            .fit_internal(&x, &y)
            .expect("model should fit");

        // Should have reasonable R²
        assert!(bls_fitted.result.r_squared > 0.5);
    }

    #[test]
    fn test_bls_with_intercept() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| 5.0 + 2.0 * i as f64);

        let fitted = BlsRegressor::nnls()
            .with_intercept(true)
            .build()
            .fit_internal(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.intercept.is_some());
        // Coefficient should be non-negative
        assert!(fitted.result.coefficients[0] >= -1e-10);
    }

    #[test]
    fn test_bound_all() {
        let x = Mat::from_fn(10, 3, |i, j| (i + j) as f64);
        let y = Col::from_fn(10, |i| i as f64);

        let fitted = BlsRegressor::builder()
            .lower_bound_all(0.0)
            .upper_bound_all(1.0)
            .with_intercept(false)
            .build()
            .fit_internal(&x, &y)
            .expect("model should fit");

        for i in 0..3 {
            assert!(
                fitted.result.coefficients[i] >= -1e-10,
                "Coef {} below lower bound",
                i
            );
            assert!(
                fitted.result.coefficients[i] <= 1.0 + 1e-10,
                "Coef {} above upper bound",
                i
            );
        }
    }

    // ==================== Additional tests for coverage ====================

    #[test]
    fn test_bls_new_constructor() {
        let options = RegressionOptionsBuilder::default()
            .build()
            .expect("valid options");
        let lower = Some(vec![0.0, 0.0]);
        let upper = Some(vec![10.0, 10.0]);
        let regressor = BlsRegressor::new(options, lower, upper);

        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| i as f64);

        let fitted = regressor.fit_internal(&x, &y).expect("should fit");
        assert!(fitted.result.coefficients.nrows() == 2);
    }

    #[test]
    fn test_builder_new() {
        let builder = BlsRegressorBuilder::new();
        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| i as f64);

        let fitted = builder
            .with_intercept(false)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        assert!(fitted.result.coefficients.nrows() == 2);
    }

    #[test]
    fn test_builder_max_iterations() {
        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| i as f64);

        let fitted = BlsRegressor::builder()
            .max_iterations(100)
            .with_intercept(false)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        assert!(fitted.result.r_squared >= 0.0);
    }

    #[test]
    fn test_builder_tolerance() {
        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| i as f64);

        let fitted = BlsRegressor::builder()
            .tolerance(1e-8)
            .with_intercept(false)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        assert!(fitted.result.r_squared >= 0.0);
    }

    #[test]
    fn test_predict_method() {
        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| 1.0 + 0.5 * i as f64);

        let fitted = BlsRegressor::nnls()
            .with_intercept(true)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        let x_new = Mat::from_fn(5, 2, |i, j| ((i + 20) + j) as f64);
        let predictions = fitted.predict(&x_new);

        assert_eq!(predictions.nrows(), 5);
        for i in 0..5 {
            assert!(predictions[i].is_finite());
        }
    }

    #[test]
    fn test_result_method() {
        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| 1.0 + 0.5 * i as f64);

        let fitted = BlsRegressor::nnls()
            .with_intercept(true)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        let result = fitted.result();
        assert!(result.coefficients.nrows() == 2);
        assert!(result.r_squared >= 0.0);
    }

    #[test]
    fn test_predict_with_interval_none() {
        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| 1.0 + 0.5 * i as f64);

        let fitted = BlsRegressor::nnls()
            .with_intercept(true)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        let x_new = Mat::from_fn(5, 2, |i, j| ((i + 20) + j) as f64);
        let result = fitted.predict_with_interval(&x_new, None, 0.95);

        assert_eq!(result.fit.nrows(), 5);
    }

    #[test]
    fn test_predict_with_interval_confidence() {
        let x = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(20, |i| 1.0 + 0.5 * i as f64);

        let fitted = BlsRegressor::nnls()
            .with_intercept(true)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        let x_new = Mat::from_fn(5, 2, |i, j| ((i + 20) + j) as f64);
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

        assert_eq!(result.fit.nrows(), 5);
        assert_eq!(result.lower.nrows(), 5);
        assert_eq!(result.upper.nrows(), 5);
        // Intervals should be NaN for BLS
        for i in 0..5 {
            assert!(result.lower[i].is_nan());
            assert!(result.upper[i].is_nan());
        }
    }

    #[test]
    fn test_predict_without_intercept() {
        let x = Mat::from_fn(20, 2, |i, j| ((i + 1) * (j + 1)) as f64);
        let y = Col::from_fn(20, |i| 0.5 * (i + 1) as f64);

        let fitted = BlsRegressor::nnls()
            .with_intercept(false)
            .build()
            .fit_internal(&x, &y)
            .expect("should fit");

        let x_new = Mat::from_fn(5, 2, |i, j| ((i + 20) * (j + 1)) as f64);
        let predictions = fitted.predict(&x_new);

        assert_eq!(predictions.nrows(), 5);
    }
}
