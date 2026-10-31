//! Bounded Least Squares (BLS) regression solver.
//!
//! Implements the Lawson-Hanson active set algorithm for non-negative least squares (NNLS)
//! and generalizes it to box constraints (lower ≤ x ≤ upper).
//!
//! # Reference
//!
//! - Lawson, C.L. and Hanson, R.J. (1974). "Solving Least Squares Problems". Prentice-Hall.
//! - R package `nnls`: <https://cran.r-project.org/web/packages/nnls/index.html>

use crate::core::{HasModelInfo, ModelInfo};
use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};

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

    /// Solve the bounded least squares problem with the active-set method of
    /// Lawson & Hanson as generalised to box constraints by Stark & Parker
    /// (BVLS).
    ///
    /// The algorithm maintains two sets:
    /// - Free set F: variables solved for (unbounded variables are always free)
    /// - Bound set B: variables fixed at their lower or upper bound
    ///
    /// At each iteration:
    /// 1. Solve least squares for F on `y − X_B β_B`
    /// 2. If that leaves a bound, step to the boundary and fix the variable
    /// 3. When feasible, free the bound variable that most violates the KKT
    ///    conditions; stop when none does
    fn solve_bls(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        lower: &[f64],
        upper: &[f64],
    ) -> Result<(Col<f64>, usize), RegressionError> {
        let n_features = x.ncols();

        // Bounded variables start at a finite bound, unbounded ones are free.
        let mut beta = Col::from_fn(n_features, |j| {
            if lower[j].is_finite() {
                lower[j]
            } else if upper[j].is_finite() {
                upper[j]
            } else {
                0.0
            }
        });
        // true = fixed at a bound, false = free
        let mut active: Vec<bool> = (0..n_features)
            .map(|j| lower[j].is_finite() || upper[j].is_finite())
            .collect();
        let mut at_lower: Vec<bool> = (0..n_features).map(|j| lower[j].is_finite()).collect();

        let xtx = compute_xtx(x);
        let xty = compute_xty(x, y);
        let max_iter = self.options.max_iterations;
        // KKT test on the scale-free gradient x_j'(y − Xβ) / (‖x_j‖·‖y‖).
        let tol = self.options.tolerance;
        let y_norm = y
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let scale: Vec<f64> = (0..n_features)
            .map(|j| xtx[(j, j)].sqrt().max(f64::MIN_POSITIVE) * y_norm)
            .collect();
        // A variable that was just freed and immediately returned to its bound
        // is not freed again until something else changes (anti-cycling).
        let mut blocked: Option<usize> = None;
        let mut first = true;

        for _outer_iter in 0..max_iter {
            if !first {
                // Gradient of ½||y − Xβ||²: X'Xβ − X'y
                let gradient = compute_gradient(&xtx, &xty, &beta);
                let mut max_violation = tol;
                let mut best_idx: Option<usize> = None;
                for j in (0..n_features).filter(|&j| active[j] && Some(j) != blocked) {
                    // At lower: freeing helps if β_j should increase (−g > 0);
                    // at upper: if it should decrease (g > 0).
                    let v = if at_lower[j] {
                        -gradient[j]
                    } else {
                        gradient[j]
                    } / scale[j];
                    if v > max_violation && lower[j] < upper[j] {
                        max_violation = v;
                        best_idx = Some(j);
                    }
                }
                match best_idx {
                    Some(idx) => active[idx] = false,
                    None => break,
                }
                blocked = None;
                // Solve on the free set, stepping back to the boundary on
                // violations (`freed` is the variable just released).
                let freed = best_idx;
                self.free_set_steps(
                    x,
                    y,
                    lower,
                    upper,
                    &mut beta,
                    &mut active,
                    &mut at_lower,
                    freed,
                    &mut blocked,
                )?;
            } else {
                first = false;
                self.free_set_steps(
                    x,
                    y,
                    lower,
                    upper,
                    &mut beta,
                    &mut active,
                    &mut at_lower,
                    None,
                    &mut blocked,
                )?;
            }
        }

        // Clamp final solution to bounds (for numerical safety)
        for j in 0..n_features {
            beta[j] = beta[j].clamp(lower[j], upper[j]);
        }

        // Count coefficients not fixed at a zero bound (rank proxy)
        let rank = (0..n_features)
            .filter(|&j| !(active[j] && beta[j] == 0.0))
            .count();

        Ok((beta, rank.max(1)))
    }

    /// Inner loop of [`solve_bls`]: least squares on the free set with the
    /// bound variables' contribution removed, moving variables that leave
    /// their box to the bound set until the free solution is feasible.
    #[allow(clippy::too_many_arguments)]
    fn free_set_steps(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        lower: &[f64],
        upper: &[f64],
        beta: &mut Col<f64>,
        active: &mut [bool],
        at_lower: &mut [bool],
        freed: Option<usize>,
        blocked: &mut Option<usize>,
    ) -> Result<(), RegressionError> {
        let n_features = x.ncols();
        for _inner_iter in 0..=n_features {
            let free: Vec<usize> = (0..n_features).filter(|&j| !active[j]).collect();
            if free.is_empty() {
                return Ok(());
            }
            // y − X_B β_B
            let y_res = Col::from_fn(x.nrows(), |i| {
                y[i] - (0..n_features)
                    .filter(|&j| active[j])
                    .map(|j| x[(i, j)] * beta[j])
                    .sum::<f64>()
            });
            let z = self.solve_passive_set(x, &y_res, &free)?;

            // Largest feasible step from β_F towards z.
            let mut min_alpha = 1.0;
            let mut hit: Option<(usize, bool)> = None;
            for (i, &j) in free.iter().enumerate() {
                let (new_val, old_val) = (z[i], beta[j]);
                let bound = if new_val < lower[j] {
                    Some((lower[j], true))
                } else if new_val > upper[j] {
                    Some((upper[j], false))
                } else {
                    None
                };
                if let Some((b, lo)) = bound {
                    let alpha = ((b - old_val) / (new_val - old_val)).clamp(0.0, 1.0);
                    if alpha < min_alpha || hit.is_none() {
                        min_alpha = alpha;
                        hit = Some((j, lo));
                    }
                }
            }
            match hit {
                None => {
                    for (i, &j) in free.iter().enumerate() {
                        beta[j] = z[i];
                    }
                    return Ok(());
                }
                Some((j_hit, lo)) => {
                    for (i, &j) in free.iter().enumerate() {
                        beta[j] += min_alpha * (z[i] - beta[j]);
                    }
                    active[j_hit] = true;
                    at_lower[j_hit] = lo;
                    beta[j_hit] = if lo { lower[j_hit] } else { upper[j_hit] };
                    // Also fix any other free variable that landed on a bound.
                    for &j in &free {
                        if !active[j] && (beta[j] <= lower[j] || beta[j] >= upper[j]) {
                            active[j] = true;
                            at_lower[j] = beta[j] <= lower[j];
                            beta[j] = if at_lower[j] { lower[j] } else { upper[j] };
                        }
                    }
                    if Some(j_hit) == freed && min_alpha == 0.0 {
                        *blocked = freed;
                    }
                }
            }
        }
        Ok(())
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
        let q = qr.compute_thin_Q();
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

        // Unpermute. `col_piv_qr` factorises A·P = Q·R, so the back-substituted
        // `beta_perm[i]` is the coefficient for the column at pivot position `i`,
        // which is original column `perm.arrays().0[i]` (the forward array).
        // Writing through `perm.inverse()` scatters coefficients into the wrong
        // slots whenever the pivot permutation is not an involution.
        let mut beta = Col::zeros(n_passive);
        let perm_fwd = perm.arrays().0;
        for i in 0..n_passive {
            beta[perm_fwd[i]] = beta_perm[i];
        }

        Ok(beta)
    }
}

impl Regressor for BlsRegressor {
    type Fitted = FittedBls;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        if let Some((xc, yc, na_info)) =
            crate::core::apply_na_action(x, y, self.options.na_action, false)?
        {
            let mut fitted = self.fit(&xc, &yc)?;
            fitted.result.na_info = Some(na_info);
            return Ok(fitted);
        }
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
        lower: &[f64],
        upper: &[f64],
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

        let stats = super::fit_stats::linear_fit_stats(
            y,
            &residuals,
            None,
            intercept.is_some(),
            n_params,
            true,
        );

        let mut result = RegressionResult::empty(n_features, n_samples);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = rank;
        result.n_parameters = n_params;
        result.n_observations = n_samples;
        result.aliased = vec![false; n_features]; // BLS doesn't have aliased in same sense
        stats.apply(&mut result);
        result.confidence_level = self.options.confidence_level;

        // Note: Standard errors for constrained LS are complex and often not computed
        // We leave inference statistics as None for BLS

        Ok(FittedBls {
            result,
            options: self.options.clone(),
            non_negative: is_non_negative(lower, upper),
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
    /// Every coefficient bound is `[0, ∞)`, i.e. this is an NNLS fit.
    non_negative: bool,
}

/// True when the bounds are exactly the non-negativity constraints of NNLS.
fn is_non_negative(lower: &[f64], upper: &[f64]) -> bool {
    lower.iter().all(|&l| l == 0.0) && upper.iter().all(|&u| u == f64::INFINITY)
}

impl HasModelInfo for FittedBls {
    /// `"nnls"` when every coefficient is bounded to `[0, ∞)`, `"bls"` otherwise.
    fn model_info(&self) -> ModelInfo {
        ModelInfo::gaussian(if self.non_negative { "nnls" } else { "bls" })
    }
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
    use faer::linalg::solvers::SolveLstsq;

    /// Brute-force bounded least squares for small `p`: every assignment of
    /// each variable to free / lower / upper, keeping the feasible one with the
    /// smallest RSS.
    fn brute_force(x: &Mat<f64>, y: &Col<f64>, lower: &[f64], upper: &[f64]) -> (Vec<f64>, f64) {
        let (n, p) = (x.nrows(), x.ncols());
        let mut best = (vec![f64::NAN; p], f64::INFINITY);
        for code in 0..3usize.pow(p as u32) {
            let state: Vec<usize> = (0..p).map(|j| code / 3usize.pow(j as u32) % 3).collect();
            if (0..p).any(|j| {
                (state[j] == 1 && !lower[j].is_finite()) || (state[j] == 2 && !upper[j].is_finite())
            }) {
                continue;
            }
            let mut beta = vec![0.0; p];
            for j in 0..p {
                beta[j] = match state[j] {
                    1 => lower[j],
                    2 => upper[j],
                    _ => 0.0,
                };
            }
            let free: Vec<usize> = (0..p).filter(|&j| state[j] == 0).collect();
            if !free.is_empty() {
                let r = Col::from_fn(n, |i| {
                    y[i] - (0..p)
                        .filter(|&j| state[j] != 0)
                        .map(|j| x[(i, j)] * beta[j])
                        .sum::<f64>()
                });
                let xf = Mat::from_fn(n, free.len(), |i, k| x[(i, free[k])]);
                let z = xf
                    .col_piv_qr()
                    .solve_lstsq(&Mat::from_fn(n, 1, |i, _| r[i]));
                let z = Col::from_fn(free.len(), |k| z[(k, 0)]);
                for (k, &j) in free.iter().enumerate() {
                    beta[j] = z[k];
                }
            }
            if (0..p).any(|j| beta[j] < lower[j] - 1e-12 || beta[j] > upper[j] + 1e-12) {
                continue;
            }
            let rss: f64 = (0..n)
                .map(|i| (y[i] - (0..p).map(|j| x[(i, j)] * beta[j]).sum::<f64>()).powi(2))
                .sum();
            if rss < best.1 {
                best = (beta, rss);
            }
        }
        best
    }

    #[test]
    fn freed_variables_pushed_below_their_bound_step_back() {
        // NNLS problem (found by random search) in which a free coefficient
        // crosses its lower bound after another variable is freed, so the
        // solver must step back to the boundary.
        const X: [[f64; 7]; 15] = [
            [
                -0.16098249377483176,
                0.251518704520466,
                0.42056745927743566,
                0.22099218443177776,
                0.3763599348737777,
                0.21264963775753176,
                -0.067370308696967,
            ],
            [
                -0.5235181330117988,
                -0.558329460475367,
                -0.15495341953678066,
                -0.5647570969035437,
                -0.7071233925180227,
                -0.2328148975722465,
                -0.5631675392738258,
            ],
            [
                -1.2618736745426027,
                -0.5838820521487946,
                -0.5548520177979424,
                -0.8481490931437942,
                -0.8705104977220242,
                -0.5133120433319687,
                -1.232767502672429,
            ],
            [
                0.20065581112372705,
                -0.023844084196413956,
                0.04482706559558597,
                0.17150416082010894,
                0.08038557936749248,
                0.033898985202652776,
                0.35494885310488944,
            ],
            [
                -0.6147230709068469,
                -0.21562386000740202,
                -0.1081878111181242,
                -0.1364366419410753,
                -0.6231757723553183,
                -0.4004300771333976,
                -0.4681839863650206,
            ],
            [
                -0.5711363145247752,
                -0.23204499773959725,
                -0.7055606730428099,
                -0.3383733919860915,
                -0.5521283983403111,
                -0.611866264254344,
                -0.6158362286436174,
            ],
            [
                0.7295088109528851,
                0.6772847685548982,
                0.30403926914888146,
                0.8827201701365028,
                0.9384348459104026,
                0.7454721935544495,
                0.5363073200586539,
            ],
            [
                0.2603953708381935,
                0.3063622941370078,
                0.4243647859163513,
                0.23740409737313728,
                0.22399770244524175,
                0.7739343297586609,
                0.763342577617762,
            ],
            [
                0.14990081560353863,
                0.040943874425136695,
                0.6996295935239851,
                0.5626198895332492,
                0.1165980649987336,
                0.7212102336208046,
                0.49202555023722305,
            ],
            [
                0.49288222249726354,
                0.2785821756910104,
                0.8131969628331341,
                0.4994565443937768,
                0.6769510091903232,
                0.13855185991113872,
                0.24147222425257997,
            ],
            [
                0.9126092872627465,
                0.7996417036261565,
                0.649036236867109,
                0.6694582481057828,
                0.8903856043364282,
                1.1082305962347923,
                0.8822547791629812,
            ],
            [
                0.4954624305778384,
                0.12449259996124729,
                0.2521542825474289,
                0.10007481192463485,
                0.5383125614887472,
                0.735709903677529,
                0.2746094213932972,
            ],
            [
                0.1390897550430899,
                -0.421468198614003,
                0.23346166904309673,
                -0.3197510057975518,
                0.05424391093969705,
                0.045974793423488175,
                0.21740114902687635,
            ],
            [
                -0.3789536070468992,
                -0.40388534925621233,
                -0.06627020507548913,
                0.22685004228854366,
                -0.06995164363302857,
                -0.003401187485030016,
                -0.3452247767326364,
            ],
            [
                -0.7237357963007621,
                -0.5676261679381589,
                -0.4046603250752958,
                -0.8998458207929134,
                -0.6156334564360277,
                -0.4577830328282269,
                -1.0871254493739018,
            ],
        ];
        const Y: [f64; 15] = [
            1.1774089120983202,
            -0.9955973176829187,
            -1.5948492075103253,
            1.3397940798783452,
            -1.4700991813225075,
            -1.6108340948798596,
            -0.8557519523774682,
            1.878548323950867,
            -1.7631870534989198,
            1.921340390580684,
            -0.5877456332458828,
            0.9348558042344357,
            0.033977372698815955,
            -1.1908730927256945,
            -0.023914390753328352,
        ];
        let x = Mat::from_fn(15, 7, |i, j| X[i][j]);
        let y = Col::from_fn(15, |i| Y[i]);
        let (lower, upper) = (vec![0.0; 7], vec![f64::INFINITY; 7]);
        let (expected, _) = brute_force(&x, &y, &lower, &upper);
        let fit = BlsRegressor::nnls()
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .unwrap();
        for j in 0..7 {
            assert!(
                (fit.coefficients()[j] - expected[j]).abs() < 1e-9,
                "coef {j}: {} vs {}",
                fit.coefficients()[j],
                expected[j]
            );
        }
    }

    #[test]
    fn bvls_matches_brute_force_on_random_problems() {
        // Deterministic LCG; correlated columns so that freeing one variable
        // pushes others across their bounds.
        let mut seed: u64 = 191;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let bound_sets: [(f64, f64); 5] = [
            (0.0, f64::INFINITY),
            (f64::NEG_INFINITY, f64::INFINITY),
            (-0.5, 0.5),
            (0.3, 2.0),
            (f64::NEG_INFINITY, -0.2),
        ];
        for case in 0..200 {
            let (n, p) = (12, 4);
            let base: Vec<f64> = (0..n).map(|_| rnd()).collect();
            let mut x = Mat::<f64>::zeros(n, p);
            for i in 0..n {
                for j in 0..p {
                    x[(i, j)] = base[i] + 0.6 * rnd();
                }
            }
            let y = Col::from_fn(n, |_| 2.0 * rnd());
            let lower: Vec<f64> = (0..p).map(|j| bound_sets[(case + j) % 5].0).collect();
            let upper: Vec<f64> = (0..p)
                .map(|j| bound_sets[(case * 3 + j) % 5].1.max(lower[j]))
                .collect();
            let (expected, rss_bf) = brute_force(&x, &y, &lower, &upper);
            let fit = BlsRegressor::builder()
                .with_intercept(false)
                .lower_bounds(lower.clone())
                .upper_bounds(upper.clone())
                .tolerance(1e-12)
                .build()
                .fit(&x, &y)
                .unwrap();
            let b = fit.coefficients();
            let rss: f64 = (0..n)
                .map(|i| (y[i] - (0..p).map(|j| x[(i, j)] * b[j]).sum::<f64>()).powi(2))
                .sum();
            assert!(
                rss <= rss_bf * (1.0 + 1e-9) + 1e-12,
                "case {case}: rss {rss} vs brute force {rss_bf}, {b:?} vs {expected:?}"
            );
            for j in 0..p {
                assert!(
                    (b[j] - expected[j]).abs() < 1e-7,
                    "case {case}: coef {j} {} vs {}",
                    b[j],
                    expected[j]
                );
            }
        }
    }

    #[test]
    fn nnls_recovers_exact_solution_under_nontrivial_pivot() {
        // Regression test for the col-pivot unpermute bug (#26): a 3-cycle pivot
        // order with an exact, strictly-positive solution [7, 2, 0.5], so the
        // NNLS active set is empty and it must reduce to OLS. A wrong unpermute
        // scatters coefficients into the wrong slots (RSS ~ 1e8 instead of 0).
        let n = 12;
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => ((i % 4) as f64) * 0.001 + 0.001,
            1 => (((i * 3) % 5) as f64) * 1000.0 + 5.0,
            _ => (((i * 7) % 6) as f64) * 10.0 + 1.0,
        });
        let truth = [7.0, 2.0, 0.5];
        let y = Col::from_fn(n, |i| (0..3).map(|j| truth[j] * x[(i, j)]).sum());
        let fit = BlsRegressor::nnls()
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .unwrap();
        for j in 0..3 {
            assert!(
                (fit.coefficients()[j] - truth[j]).abs() < 1e-6,
                "coef[{j}] = {} vs {}",
                fit.coefficients()[j],
                truth[j]
            );
        }
    }

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
