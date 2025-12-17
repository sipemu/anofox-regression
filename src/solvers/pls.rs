//! Partial Least Squares (PLS) regression solver.
//!
//! Implements PLS regression using the SIMPLS algorithm (de Jong, 1993).
//! PLS is particularly useful when predictors are highly collinear or when
//! there are more predictors than observations.
//!
//! # Algorithm
//!
//! SIMPLS (Straightforward Implementation of a statistically inspired
//! Modification of the Partial Least Squares method) maximizes the covariance
//! between X scores and y directly, without deflating X and y.
//!
//! # References
//!
//! - de Jong, S. (1993). SIMPLS: an alternative approach to partial least squares regression.
//!   Chemometrics and Intelligent Laboratory Systems, 18, 251-263.
//! - Validated against R's `pls` package: https://cran.r-project.org/package=pls

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::{center_columns, center_vector};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Partial Least Squares regression estimator.
///
/// Uses the SIMPLS algorithm to find latent components that maximize
/// the covariance between X scores and y.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{PlsRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 10, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = PlsRegressor::builder()
///     .n_components(3)
///     .build()
///     .fit(&x, &y)?;
///
/// println!("R² = {}", fitted.r_squared());
/// println!("Coefficients: {:?}", fitted.coefficients());
/// ```
#[derive(Debug, Clone)]
pub struct PlsRegressor {
    /// Number of latent components to extract
    n_components: usize,
    /// Whether to include an intercept term
    with_intercept: bool,
    /// Tolerance for convergence and rank detection
    tolerance: f64,
    /// Whether to scale X to unit variance
    scale: bool,
}

impl PlsRegressor {
    /// Create a new PLS regressor with the given number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            with_intercept: true,
            tolerance: 1e-10,
            scale: false,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> PlsRegressorBuilder {
        PlsRegressorBuilder::default()
    }

    /// Compute the SIMPLS algorithm.
    ///
    /// Returns (weights W, x_loadings P, y_loadings Q, scores T, coefficients B).
    fn simpls(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        n_components: usize,
    ) -> (Mat<f64>, Mat<f64>, Col<f64>, Mat<f64>, Col<f64>) {
        let n = x.nrows();
        let p = x.ncols();
        let ncomp = n_components.min(n).min(p);

        // Initialize output matrices
        let mut weights = Mat::zeros(p, ncomp); // W: p x ncomp
        let mut x_loadings = Mat::zeros(p, ncomp); // P: p x ncomp
        let mut y_loadings = Col::zeros(ncomp); // q: ncomp (scalar for each component)
        let mut scores = Mat::zeros(n, ncomp); // T: n x ncomp

        // V matrix for orthogonalization
        let mut v_mat = Mat::zeros(p, ncomp);

        // Compute initial cross-product s = X'y
        let mut s = Col::zeros(p);
        for j in 0..p {
            let mut sum = 0.0;
            for i in 0..n {
                sum += x[(i, j)] * y[i];
            }
            s[j] = sum;
        }

        for a in 0..ncomp {
            // r = s (weight direction before normalization)
            let mut r = s.clone();

            // Orthogonalize r with respect to previous v's
            if a > 0 {
                for k in 0..a {
                    // r = r - v_k * (v_k' * r)
                    let mut vtr = 0.0;
                    for j in 0..p {
                        vtr += v_mat[(j, k)] * r[j];
                    }
                    for j in 0..p {
                        r[j] -= v_mat[(j, k)] * vtr;
                    }
                }
            }

            // Normalize to get weight w = r / ||r||
            let r_norm = r.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if r_norm < self.tolerance {
                // Remaining components cannot be computed
                break;
            }
            for j in 0..p {
                weights[(j, a)] = r[j] / r_norm;
            }

            // Compute score t = X * w
            for i in 0..n {
                let mut ti = 0.0;
                for j in 0..p {
                    ti += x[(i, j)] * weights[(j, a)];
                }
                scores[(i, a)] = ti;
            }

            // Normalize t
            let t_norm = (0..n).map(|i| scores[(i, a)].powi(2)).sum::<f64>().sqrt();
            if t_norm < self.tolerance {
                break;
            }
            for i in 0..n {
                scores[(i, a)] /= t_norm;
            }

            // Also scale w by t_norm to maintain X*W = T relationship
            for j in 0..p {
                weights[(j, a)] /= t_norm;
            }

            // Compute x-loading p = X' * t
            for j in 0..p {
                let mut pj = 0.0;
                for i in 0..n {
                    pj += x[(i, j)] * scores[(i, a)];
                }
                x_loadings[(j, a)] = pj;
            }

            // Compute y-loading q = y' * t (scalar for univariate y)
            let mut q = 0.0;
            for i in 0..n {
                q += y[i] * scores[(i, a)];
            }
            y_loadings[a] = q;

            // Update v for orthogonalization: v = p
            for j in 0..p {
                v_mat[(j, a)] = x_loadings[(j, a)];
            }

            // Orthogonalize v with respect to previous v's
            if a > 0 {
                for k in 0..a {
                    let mut vtv = 0.0;
                    for j in 0..p {
                        vtv += v_mat[(j, k)] * v_mat[(j, a)];
                    }
                    for j in 0..p {
                        v_mat[(j, a)] -= v_mat[(j, k)] * vtv;
                    }
                }
            }

            // Normalize v
            let v_norm = (0..p).map(|j| v_mat[(j, a)].powi(2)).sum::<f64>().sqrt();
            if v_norm > self.tolerance {
                for j in 0..p {
                    v_mat[(j, a)] /= v_norm;
                }
            }

            // Deflate s: s = s - v * (v' * s)
            let mut vts = 0.0;
            for j in 0..p {
                vts += v_mat[(j, a)] * s[j];
            }
            for j in 0..p {
                s[j] -= v_mat[(j, a)] * vts;
            }
        }

        // Compute final regression coefficients: B = W * (P'W)^(-1) * Q
        // For SIMPLS with proper orthogonalization: B = W * Q (approximately)
        // More precisely, we need to solve the system
        let coefficients = self.compute_coefficients(&weights, &x_loadings, &y_loadings, ncomp);

        (weights, x_loadings, y_loadings, scores, coefficients)
    }

    /// Compute regression coefficients from PLS components.
    fn compute_coefficients(
        &self,
        weights: &Mat<f64>,
        x_loadings: &Mat<f64>,
        y_loadings: &Col<f64>,
        ncomp: usize,
    ) -> Col<f64> {
        let p = weights.nrows();

        // Compute P'W matrix (ncomp x ncomp)
        let mut ptw = Mat::zeros(ncomp, ncomp);
        for i in 0..ncomp {
            for j in 0..ncomp {
                let mut sum = 0.0;
                for k in 0..p {
                    sum += x_loadings[(k, i)] * weights[(k, j)];
                }
                ptw[(i, j)] = sum;
            }
        }

        // Solve (P'W) * c = Q for c, then B = W * c
        // Use simple Gaussian elimination for the small system
        let c = self.solve_linear_system(&ptw, y_loadings, ncomp);

        // B = W * c
        let mut coefficients = Col::zeros(p);
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..ncomp {
                sum += weights[(j, k)] * c[k];
            }
            coefficients[j] = sum;
        }

        coefficients
    }

    /// Solve a small linear system Ax = b using Gaussian elimination with partial pivoting.
    fn solve_linear_system(&self, a: &Mat<f64>, b: &Col<f64>, n: usize) -> Col<f64> {
        // Create augmented matrix
        let mut aug = Mat::zeros(n, n + 1);
        for i in 0..n {
            for j in 0..n {
                aug[(i, j)] = a[(i, j)];
            }
            aug[(i, n)] = b[i];
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = aug[(k, k)].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[(i, k)].abs() > max_val {
                    max_val = aug[(i, k)].abs();
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in 0..=n {
                    let tmp = aug[(k, j)];
                    aug[(k, j)] = aug[(max_row, j)];
                    aug[(max_row, j)] = tmp;
                }
            }

            // Check for singular matrix
            if aug[(k, k)].abs() < self.tolerance {
                continue;
            }

            // Eliminate
            for i in (k + 1)..n {
                let factor = aug[(i, k)] / aug[(k, k)];
                for j in k..=n {
                    aug[(i, j)] -= factor * aug[(k, j)];
                }
            }
        }

        // Back substitution
        let mut x = Col::zeros(n);
        for i in (0..n).rev() {
            if aug[(i, i)].abs() < self.tolerance {
                x[i] = 0.0;
            } else {
                let mut sum = aug[(i, n)];
                for j in (i + 1)..n {
                    sum -= aug[(i, j)] * x[j];
                }
                x[i] = sum / aug[(i, i)];
            }
        }

        x
    }
}

impl Regressor for PlsRegressor {
    type Fitted = FittedPls;

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

        // Determine actual number of components
        let max_components = n_samples.min(n_features);
        let n_components = self.n_components.min(max_components);

        if n_components == 0 {
            return Err(RegressionError::InsufficientObservations {
                needed: 1,
                got: 0,
            });
        }

        // Center data (and optionally scale X)
        let (x_centered, x_means) = center_columns(x);
        let (y_centered, y_mean) = center_vector(y);

        // Optionally scale X to unit variance
        let (x_processed, x_scales) = if self.scale {
            let mut scales = Col::zeros(n_features);
            let mut x_scaled = x_centered.clone();
            for j in 0..n_features {
                let mut var = 0.0;
                for i in 0..n_samples {
                    var += x_centered[(i, j)].powi(2);
                }
                var /= (n_samples - 1) as f64;
                let std = var.sqrt();
                if std > self.tolerance {
                    scales[j] = std;
                    for i in 0..n_samples {
                        x_scaled[(i, j)] /= std;
                    }
                } else {
                    scales[j] = 1.0;
                }
            }
            (x_scaled, Some(scales))
        } else {
            (x_centered, None)
        };

        // Run SIMPLS algorithm
        let (weights, x_loadings, y_loadings, scores, mut coefficients) =
            self.simpls(&x_processed, &y_centered, n_components);

        // If scaling was applied, adjust coefficients
        if let Some(ref scales) = x_scales {
            for j in 0..n_features {
                coefficients[j] /= scales[j];
            }
        }

        // Compute intercept
        let intercept = if self.with_intercept {
            let mut int = y_mean;
            for j in 0..n_features {
                int -= x_means[j] * coefficients[j];
            }
            Some(int)
        } else {
            None
        };

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
        let result = self.compute_statistics(
            n_samples,
            n_features,
            n_components,
            &coefficients,
            intercept,
            y,
            &residuals,
            &fitted_values,
        );

        Ok(FittedPls {
            n_components,
            with_intercept: self.with_intercept,
            result,
            x_means,
            y_mean,
            x_scales,
            weights,
            x_loadings,
            y_loadings,
            scores,
        })
    }
}

impl PlsRegressor {
    /// Compute fit statistics.
    #[allow(clippy::too_many_arguments)]
    fn compute_statistics(
        &self,
        n_samples: usize,
        n_features: usize,
        n_components: usize,
        coefficients: &Col<f64>,
        intercept: Option<f64>,
        y: &Col<f64>,
        residuals: &Col<f64>,
        fitted_values: &Col<f64>,
    ) -> RegressionResult {
        let n = n_samples;

        // Number of parameters: n_components latent variables + intercept
        let n_params = if intercept.is_some() {
            n_components + 1
        } else {
            n_components
        };

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
        let df_resid = (n.saturating_sub(n_params)) as f64;
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

        let k = n_params as f64;
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
        result.rank = n_components;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.aliased = vec![false; n_features];
        result.rank_tolerance = self.tolerance;
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
        result.confidence_level = 0.95;

        result
    }
}

/// A fitted PLS regression model.
#[derive(Debug, Clone)]
pub struct FittedPls {
    /// Number of components used
    n_components: usize,
    /// Whether model includes intercept
    #[allow(dead_code)]
    with_intercept: bool,
    /// Regression result
    result: RegressionResult,
    /// Mean of X columns (for centering new data)
    x_means: Col<f64>,
    /// Mean of y (for centering)
    y_mean: f64,
    /// Standard deviations of X columns (if scaled)
    x_scales: Option<Col<f64>>,
    /// Weight matrix W (p x n_components)
    weights: Mat<f64>,
    /// X-loadings matrix P (p x n_components)
    x_loadings: Mat<f64>,
    /// Y-loadings vector Q (n_components)
    y_loadings: Col<f64>,
    /// Score matrix T (n x n_components)
    scores: Mat<f64>,
}

impl FittedPls {
    /// Get the number of components used in the model.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the weight matrix W (p x n_components).
    ///
    /// W contains the weights used to compute scores from X.
    pub fn weights(&self) -> &Mat<f64> {
        &self.weights
    }

    /// Get the X-loadings matrix P (p x n_components).
    ///
    /// P contains the loadings of X on the components.
    pub fn x_loadings(&self) -> &Mat<f64> {
        &self.x_loadings
    }

    /// Get the Y-loadings vector Q (n_components).
    ///
    /// Q contains the loadings of y on the components.
    pub fn y_loadings(&self) -> &Col<f64> {
        &self.y_loadings
    }

    /// Get the score matrix T (n x n_components).
    ///
    /// T contains the scores (latent variables) from training data.
    pub fn scores(&self) -> &Mat<f64> {
        &self.scores
    }

    /// Get the mean of X columns used for centering.
    pub fn x_means(&self) -> &Col<f64> {
        &self.x_means
    }

    /// Get the mean of y used for centering.
    pub fn y_mean(&self) -> f64 {
        self.y_mean
    }

    /// Compute scores for new X data.
    pub fn transform(&self, x: &Mat<f64>) -> Mat<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let ncomp = self.n_components;

        // Center (and scale) the new data
        let mut x_centered = Mat::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x_centered[(i, j)] = x[(i, j)] - self.x_means[j];
                if let Some(ref scales) = self.x_scales {
                    x_centered[(i, j)] /= scales[j];
                }
            }
        }

        // Compute scores: T = X * W
        let mut new_scores = Mat::zeros(n, ncomp);
        for i in 0..n {
            for k in 0..ncomp {
                let mut sum = 0.0;
                for j in 0..p {
                    sum += x_centered[(i, j)] * self.weights[(j, k)];
                }
                new_scores[(i, k)] = sum;
            }
        }

        new_scores
    }

    /// Get the explained variance ratio for each component.
    ///
    /// Returns the proportion of total variance in X explained by each component.
    pub fn explained_variance_ratio(&self) -> Col<f64> {
        let n = self.scores.nrows();
        let ncomp = self.n_components;

        // Compute variance of each score column
        let mut variances = Col::zeros(ncomp);
        let mut total_var = 0.0;

        for k in 0..ncomp {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for i in 0..n {
                sum += self.scores[(i, k)];
                sum_sq += self.scores[(i, k)].powi(2);
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean.powi(2);
            variances[k] = var;
            total_var += var;
        }

        // Normalize to get ratios
        if total_var > 0.0 {
            for k in 0..ncomp {
                variances[k] /= total_var;
            }
        }

        variances
    }
}

impl FittedRegressor for FittedPls {
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
            Some(_interval_type) => {
                // PLS prediction intervals are complex to compute properly
                // For now, return NaN intervals (similar to how some implementations handle this)
                let n = x.nrows();
                let mut lower = Col::zeros(n);
                let mut upper = Col::zeros(n);
                let mut se = Col::zeros(n);
                for i in 0..n {
                    lower[i] = f64::NAN;
                    upper[i] = f64::NAN;
                    se[i] = f64::NAN;
                }
                let _ = level; // suppress unused warning
                PredictionResult::with_intervals(predictions, lower, upper, se)
            }
        }
    }
}

/// Builder for `PlsRegressor`.
#[derive(Debug, Clone)]
pub struct PlsRegressorBuilder {
    n_components: usize,
    with_intercept: bool,
    tolerance: f64,
    scale: bool,
}

impl Default for PlsRegressorBuilder {
    fn default() -> Self {
        Self {
            n_components: 2,
            with_intercept: true,
            tolerance: 1e-10,
            scale: false,
        }
    }
}

impl PlsRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of latent components to extract.
    ///
    /// Default is 2. Will be clamped to min(n_samples, n_features).
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set whether to include an intercept term.
    ///
    /// Default is true.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Set the tolerance for numerical computations.
    ///
    /// Default is 1e-10.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set whether to scale X to unit variance before fitting.
    ///
    /// Default is false. When true, each column of X is scaled to have
    /// unit variance, which can be useful when features are on different scales.
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Build the PLS regressor.
    pub fn build(self) -> PlsRegressor {
        PlsRegressor {
            n_components: self.n_components,
            with_intercept: self.with_intercept,
            tolerance: self.tolerance,
            scale: self.scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fit() {
        // Simple linear relationship: y = 2*x1 + 3*x2 + 1
        let x = Mat::from_fn(20, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                (i as f64) * 0.5
            }
        });
        let y = Col::from_fn(20, |i| 1.0 + 2.0 * (i as f64) + 3.0 * (i as f64) * 0.5);

        let model = PlsRegressor::builder().n_components(2).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        // Check that R² is high (should be perfect or near-perfect)
        assert!(fitted.r_squared() > 0.99);
    }

    #[test]
    fn test_predict() {
        let x = Mat::from_fn(20, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                (i as f64) * 0.5
            }
        });
        let y = Col::from_fn(20, |i| 1.0 + 2.0 * (i as f64) + 3.0 * (i as f64) * 0.5);

        let model = PlsRegressor::builder().n_components(2).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        // Predict on new data
        let x_new = Mat::from_fn(5, 2, |i, j| {
            if j == 0 {
                (i + 100) as f64
            } else {
                ((i + 100) as f64) * 0.5
            }
        });
        let preds = fitted.predict(&x_new);

        // Check predictions match expected pattern
        for i in 0..5 {
            let expected = 1.0 + 2.0 * ((i + 100) as f64) + 3.0 * ((i + 100) as f64) * 0.5;
            assert!((preds[i] - expected).abs() < 1.0); // Allow some tolerance
        }
    }

    #[test]
    fn test_n_components() {
        let x = Mat::from_fn(50, 5, |i, j| ((i + j) as f64).sin());
        let y = Col::from_fn(50, |i| (i as f64).cos() + 0.1 * (i as f64));

        let model = PlsRegressor::builder().n_components(3).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        assert_eq!(fitted.n_components(), 3);
        assert_eq!(fitted.weights().ncols(), 3);
        assert_eq!(fitted.x_loadings().ncols(), 3);
        assert_eq!(fitted.y_loadings().nrows(), 3);
        assert_eq!(fitted.scores().ncols(), 3);
    }

    #[test]
    fn test_transform() {
        let x = Mat::from_fn(30, 4, |i, j| ((i * j + 1) as f64).sqrt());
        let y = Col::from_fn(30, |i| (i as f64) * 2.0 + 1.0);

        let model = PlsRegressor::builder().n_components(2).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        // Transform new data
        let x_new = Mat::from_fn(10, 4, |i, j| ((i * j + 5) as f64).sqrt());
        let new_scores = fitted.transform(&x_new);

        assert_eq!(new_scores.nrows(), 10);
        assert_eq!(new_scores.ncols(), 2);
    }

    #[test]
    fn test_with_scaling() {
        // Create data with features on very different scales
        let x = Mat::from_fn(30, 3, |i, j| match j {
            0 => i as f64,               // scale ~30
            1 => (i as f64) * 1000.0,    // scale ~30000
            2 => (i as f64) * 0.001,     // scale ~0.03
            _ => 0.0,
        });
        let y = Col::from_fn(30, |i| (i as f64) * 2.5 + 10.0);

        // Without scaling
        let model_no_scale = PlsRegressor::builder()
            .n_components(2)
            .scale(false)
            .build();
        let fitted_no_scale = model_no_scale.fit(&x, &y).expect("model should fit");

        // With scaling
        let model_scaled = PlsRegressor::builder()
            .n_components(2)
            .scale(true)
            .build();
        let fitted_scaled = model_scaled.fit(&x, &y).expect("model should fit");

        // Both should fit, but potentially different R²
        assert!(fitted_no_scale.r_squared() > 0.0);
        assert!(fitted_scaled.r_squared() > 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(5, |i| i as f64); // Wrong length

        let model = PlsRegressor::builder().build();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_observations() {
        let x = Mat::from_fn(1, 2, |_, j| j as f64);
        let y = Col::from_fn(1, |_| 1.0);

        let model = PlsRegressor::builder().build();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_without_intercept() {
        let x = Mat::from_fn(20, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                (i as f64) * 0.5
            }
        });
        let y = Col::from_fn(20, |i| 2.0 * (i as f64) + 1.5 * (i as f64) * 0.5);

        let model = PlsRegressor::builder()
            .n_components(2)
            .with_intercept(false)
            .build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!(fitted.intercept().is_none());
    }
}
