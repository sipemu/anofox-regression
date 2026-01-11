//! Isotonic (Monotonic) Regression solver.
//!
//! Implements isotonic regression using the Pool Adjacent Violators Algorithm (PAVA).
//! Isotonic regression fits a non-decreasing (or non-increasing) step function
//! to the data by pooling adjacent observations that violate the monotonicity constraint.
//!
//! # Algorithm
//!
//! The PAVA algorithm works as follows:
//! 1. Start with the original y values
//! 2. Scan for adjacent pairs that violate monotonicity
//! 3. Pool violating pairs by replacing them with their (weighted) mean
//! 4. Repeat until no violations remain
//!
//! # References
//!
//! - Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972).
//!   Statistical Inference under Order Restrictions. Wiley.
//! - Validated against R's `isoreg()` function

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};

/// Isotonic Regression estimator.
///
/// Fits a monotonic (non-decreasing or non-increasing) function to the data.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{IsotonicRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// // Simple 1D data
/// let x = Col::from_fn(20, |i| i as f64);
/// let y = Col::from_fn(20, |i| 2.0 + 0.5 * i as f64 + rand::random::<f64>());
///
/// let fitted = IsotonicRegressor::builder()
///     .increasing(true)
///     .build()
///     .fit_1d(&x, &y)?;
///
/// println!("Fitted values: {:?}", fitted.fitted_values());
/// ```
#[derive(Debug, Clone)]
pub struct IsotonicRegressor {
    /// Whether to fit increasing (true) or decreasing (false) monotonic function
    increasing: bool,
    /// Tolerance for convergence
    tolerance: f64,
    /// Optional observation weights
    weights: Option<Col<f64>>,
    /// Out-of-bounds handling for prediction
    out_of_bounds: OutOfBounds,
}

/// How to handle predictions outside the training range.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutOfBounds {
    /// Clip predictions to the range [min(fitted), max(fitted)]
    #[default]
    Clip,
    /// Return NaN for out-of-bounds predictions
    Nan,
    /// Extrapolate using the nearest boundary value
    Extrapolate,
}

impl IsotonicRegressor {
    /// Create a new isotonic regressor.
    pub fn new() -> Self {
        Self {
            increasing: true,
            tolerance: 1e-10,
            weights: None,
            out_of_bounds: OutOfBounds::Clip,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> IsotonicRegressorBuilder {
        IsotonicRegressorBuilder::default()
    }

    /// Fit isotonic regression on 1D data.
    ///
    /// # Arguments
    /// * `x` - Feature values (will be sorted)
    /// * `y` - Target values
    ///
    /// # Returns
    /// A fitted isotonic regression model
    pub fn fit_1d(&self, x: &Col<f64>, y: &Col<f64>) -> Result<FittedIsotonic, RegressionError> {
        let n = x.nrows();

        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        if n < 2 {
            return Err(RegressionError::InsufficientObservations { needed: 2, got: n });
        }

        // Validate weights if provided
        let weights = if let Some(ref w) = self.weights {
            if w.nrows() != n {
                return Err(RegressionError::DimensionMismatch {
                    x_rows: n,
                    y_len: w.nrows(),
                });
            }
            if w.iter().any(|&wi| wi < 0.0) {
                return Err(RegressionError::InvalidWeights);
            }
            w.clone()
        } else {
            Col::from_fn(n, |_| 1.0)
        };

        // Sort by x values
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));

        let x_sorted = Col::from_fn(n, |i| x[indices[i]]);
        let y_sorted = Col::from_fn(n, |i| y[indices[i]]);
        let w_sorted = Col::from_fn(n, |i| weights[indices[i]]);

        // Handle ties in x by averaging y values
        let (x_unique, y_unique, w_unique) = self.handle_ties(&x_sorted, &y_sorted, &w_sorted);

        // Apply PAVA algorithm
        let fitted_unique = if self.increasing {
            self.pava_increasing(&y_unique, &w_unique)
        } else {
            // For decreasing, negate, apply PAVA, negate back
            let y_neg = Col::from_fn(y_unique.nrows(), |i| -y_unique[i]);
            let fitted_neg = self.pava_increasing(&y_neg, &w_unique);
            Col::from_fn(fitted_neg.nrows(), |i| -fitted_neg[i])
        };

        // Expand fitted values back to original (sorted) indices
        let fitted_sorted = self.expand_to_original(&x_sorted, &x_unique, &fitted_unique);

        // Create unsorted fitted values (back to original order)
        let mut fitted = Col::zeros(n);
        for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
            fitted[orig_idx] = fitted_sorted[sorted_idx];
        }

        // Compute residuals
        let residuals = Col::from_fn(n, |i| y[i] - fitted[i]);

        // Compute R-squared
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
        let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

        // Build result (using empty coefficients since this is non-parametric)
        let mut result = RegressionResult::empty(0, n);
        result.residuals = residuals;
        result.fitted_values = fitted.clone();
        result.n_observations = n;
        result.r_squared = r_squared;

        Ok(FittedIsotonic {
            increasing: self.increasing,
            x_thresholds: x_unique,
            y_values: fitted_unique,
            result,
            out_of_bounds: self.out_of_bounds,
        })
    }

    /// Handle ties in x by computing weighted average of y values.
    fn handle_ties(
        &self,
        x: &Col<f64>,
        y: &Col<f64>,
        w: &Col<f64>,
    ) -> (Col<f64>, Col<f64>, Col<f64>) {
        let n = x.nrows();

        let mut x_unique = Vec::new();
        let mut y_unique = Vec::new();
        let mut w_unique = Vec::new();

        let mut i = 0;
        while i < n {
            let x_val = x[i];
            let mut y_sum = 0.0;
            let mut w_sum = 0.0;

            // Accumulate all values with the same x
            while i < n && (x[i] - x_val).abs() < self.tolerance {
                y_sum += w[i] * y[i];
                w_sum += w[i];
                i += 1;
            }

            x_unique.push(x_val);
            y_unique.push(y_sum / w_sum);
            w_unique.push(w_sum);
        }

        (
            Col::from_fn(x_unique.len(), |i| x_unique[i]),
            Col::from_fn(y_unique.len(), |i| y_unique[i]),
            Col::from_fn(w_unique.len(), |i| w_unique[i]),
        )
    }

    /// Pool Adjacent Violators Algorithm for increasing constraint.
    fn pava_increasing(&self, y: &Col<f64>, w: &Col<f64>) -> Col<f64> {
        let n = y.nrows();
        if n == 0 {
            return Col::zeros(0);
        }
        if n == 1 {
            return y.clone();
        }

        // Initialize blocks: each observation starts as its own block
        // Each block has: (weighted sum of y, sum of weights, start index, end index)
        let mut blocks: Vec<(f64, f64, usize, usize)> =
            (0..n).map(|i| (y[i] * w[i], w[i], i, i)).collect();

        // Iteratively merge blocks that violate the ordering constraint
        loop {
            let mut merged = false;
            let mut new_blocks = Vec::new();

            let mut i = 0;
            while i < blocks.len() {
                if i + 1 < blocks.len() {
                    let mean_i = blocks[i].0 / blocks[i].1;
                    let mean_j = blocks[i + 1].0 / blocks[i + 1].1;

                    if mean_i > mean_j {
                        // Merge blocks i and i+1
                        let merged_block = (
                            blocks[i].0 + blocks[i + 1].0,
                            blocks[i].1 + blocks[i + 1].1,
                            blocks[i].2,
                            blocks[i + 1].3,
                        );
                        new_blocks.push(merged_block);
                        merged = true;
                        i += 2;
                        continue;
                    }
                }
                new_blocks.push(blocks[i]);
                i += 1;
            }

            blocks = new_blocks;

            if !merged {
                break;
            }
        }

        // Expand blocks back to individual fitted values
        let mut fitted = Col::zeros(n);
        for (y_sum, w_sum, start, end) in blocks {
            let mean = y_sum / w_sum;
            for i in start..=end {
                fitted[i] = mean;
            }
        }

        fitted
    }

    /// Expand unique fitted values back to original indices (handling ties).
    fn expand_to_original(
        &self,
        x_sorted: &Col<f64>,
        x_unique: &Col<f64>,
        fitted_unique: &Col<f64>,
    ) -> Col<f64> {
        let n = x_sorted.nrows();
        let mut fitted = Col::zeros(n);

        let mut unique_idx = 0;
        for i in 0..n {
            // Find the matching unique x value
            while unique_idx + 1 < x_unique.nrows()
                && x_sorted[i] > x_unique[unique_idx] + self.tolerance
            {
                unique_idx += 1;
            }
            fitted[i] = fitted_unique[unique_idx];
        }

        fitted
    }
}

impl Default for IsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Regressor for IsotonicRegressor {
    type Fitted = FittedIsotonic;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        // For matrix input, use only the first column
        if x.ncols() != 1 {
            return Err(RegressionError::NumericalError(
                "Isotonic regression requires exactly one feature. Use fit_1d() for 1D data."
                    .to_string(),
            ));
        }

        let x_col = Col::from_fn(x.nrows(), |i| x[(i, 0)]);
        self.fit_1d(&x_col, y)
    }
}

/// A fitted Isotonic Regression model.
#[derive(Debug, Clone)]
pub struct FittedIsotonic {
    /// Whether the fit is increasing or decreasing
    increasing: bool,
    /// X thresholds for the step function
    x_thresholds: Col<f64>,
    /// Y values at each threshold
    y_values: Col<f64>,
    /// Regression result
    result: RegressionResult,
    /// Out-of-bounds handling
    out_of_bounds: OutOfBounds,
}

impl FittedIsotonic {
    /// Get whether the fit is increasing or decreasing.
    pub fn is_increasing(&self) -> bool {
        self.increasing
    }

    /// Get the X thresholds of the step function.
    pub fn x_thresholds(&self) -> &Col<f64> {
        &self.x_thresholds
    }

    /// Get the Y values at each threshold.
    pub fn y_values(&self) -> &Col<f64> {
        &self.y_values
    }

    /// Get the fitted values for the training data.
    pub fn fitted_values(&self) -> &Col<f64> {
        &self.result.fitted_values
    }

    /// Predict for a single x value using step function interpolation.
    pub fn predict_single(&self, x: f64) -> f64 {
        let n = self.x_thresholds.nrows();
        if n == 0 {
            return f64::NAN;
        }

        let x_min = self.x_thresholds[0];
        let x_max = self.x_thresholds[n - 1];

        // Handle out-of-bounds
        if x < x_min {
            return match self.out_of_bounds {
                OutOfBounds::Clip | OutOfBounds::Extrapolate => self.y_values[0],
                OutOfBounds::Nan => f64::NAN,
            };
        }
        if x > x_max {
            return match self.out_of_bounds {
                OutOfBounds::Clip | OutOfBounds::Extrapolate => self.y_values[n - 1],
                OutOfBounds::Nan => f64::NAN,
            };
        }

        // Binary search for the appropriate interval
        let mut lo = 0;
        let mut hi = n - 1;
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.x_thresholds[mid] <= x {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        // Use step function (return value at lower bound of interval)
        self.y_values[lo]
    }

    /// Predict for 1D input.
    pub fn predict_1d(&self, x: &Col<f64>) -> Col<f64> {
        Col::from_fn(x.nrows(), |i| self.predict_single(x[i]))
    }
}

impl FittedRegressor for FittedIsotonic {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        // Use first column only
        let x_col = Col::from_fn(x.nrows(), |i| x[(i, 0)]);
        self.predict_1d(&x_col)
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
                // Isotonic regression intervals are complex (bootstrap required)
                let n = x.nrows();
                let mut lower = Col::zeros(n);
                let mut upper = Col::zeros(n);
                let mut se = Col::zeros(n);
                for i in 0..n {
                    lower[i] = f64::NAN;
                    upper[i] = f64::NAN;
                    se[i] = f64::NAN;
                }
                let _ = level;
                PredictionResult::with_intervals(predictions, lower, upper, se)
            }
        }
    }
}

/// Builder for `IsotonicRegressor`.
#[derive(Debug, Clone)]
pub struct IsotonicRegressorBuilder {
    increasing: bool,
    tolerance: f64,
    weights: Option<Col<f64>>,
    out_of_bounds: OutOfBounds,
}

impl Default for IsotonicRegressorBuilder {
    fn default() -> Self {
        Self {
            increasing: true,
            tolerance: 1e-10,
            weights: None,
            out_of_bounds: OutOfBounds::Clip,
        }
    }
}

impl IsotonicRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to fit an increasing or decreasing function.
    ///
    /// Default is true (increasing).
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set the tolerance for numerical comparisons.
    ///
    /// Default is 1e-10.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set observation weights.
    pub fn weights(mut self, w: Col<f64>) -> Self {
        self.weights = Some(w);
        self
    }

    /// Set how to handle out-of-bounds predictions.
    ///
    /// Default is `Clip`.
    pub fn out_of_bounds(mut self, oob: OutOfBounds) -> Self {
        self.out_of_bounds = oob;
        self
    }

    /// Build the isotonic regressor.
    pub fn build(self) -> IsotonicRegressor {
        IsotonicRegressor {
            increasing: self.increasing,
            tolerance: self.tolerance,
            weights: self.weights,
            out_of_bounds: self.out_of_bounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_increasing() {
        // Data with general increasing trend
        let x = Col::from_fn(10, |i| i as f64);
        let y = Col::from_fn(10, |i| (i as f64) * 1.5);

        let model = IsotonicRegressor::builder().increasing(true).build();
        let fitted = model.fit_1d(&x, &y).expect("model should fit");

        // Fitted values should be monotonically increasing
        let fv = fitted.fitted_values();
        for i in 1..fv.nrows() {
            assert!(fv[i] >= fv[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_decreasing() {
        // Decreasing data
        let x = Col::from_fn(10, |i| i as f64);
        let y = Col::from_fn(10, |i| 20.0 - (i as f64) * 1.5);

        let model = IsotonicRegressor::builder().increasing(false).build();
        let fitted = model.fit_1d(&x, &y).expect("model should fit");

        // Fitted values should be monotonically decreasing
        let fv = fitted.fitted_values();
        for i in 1..fv.nrows() {
            assert!(fv[i] <= fv[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_already_monotonic() {
        // Already monotonic data - should pass through unchanged
        let x = Col::from_fn(5, |i| i as f64);
        let y = Col::from_fn(5, |i| (i + 1) as f64);

        let model = IsotonicRegressor::builder().build();
        let fitted = model.fit_1d(&x, &y).expect("model should fit");

        let fv = fitted.fitted_values();
        for i in 0..5 {
            assert!((fv[i] - y[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pooling() {
        // Data that requires pooling
        let x = Col::from_fn(4, |i| i as f64);
        let y = Col::from_fn(4, |_| 0.0);
        let mut y_data = y.clone();
        y_data[0] = 5.0; // Higher than next
        y_data[1] = 3.0;
        y_data[2] = 3.0;
        y_data[3] = 4.0;

        let y = Col::from_fn(4, |i| y_data[i]);

        let model = IsotonicRegressor::builder().build();
        let fitted = model.fit_1d(&x, &y).expect("model should fit");

        let fv = fitted.fitted_values();
        // First two should be pooled
        assert!((fv[0] - fv[1]).abs() < 1e-10);
        // Should be monotonically increasing
        for i in 1..fv.nrows() {
            assert!(fv[i] >= fv[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_predict() {
        let x = Col::from_fn(10, |i| i as f64);
        let y = Col::from_fn(10, |i| (i as f64) * 2.0);

        let model = IsotonicRegressor::builder().build();
        let fitted = model.fit_1d(&x, &y).expect("model should fit");

        // Predict at training points
        let preds = fitted.predict_1d(&x);
        assert_eq!(preds.nrows(), 10);

        // Predict at new points
        let x_new = Col::from_fn(3, |i| (i as f64) + 0.5);
        let preds_new = fitted.predict_1d(&x_new);
        assert_eq!(preds_new.nrows(), 3);
    }

    #[test]
    fn test_out_of_bounds_clip() {
        let x = Col::from_fn(5, |i| (i + 1) as f64);
        let y = Col::from_fn(5, |i| (i + 1) as f64 * 2.0);

        let model = IsotonicRegressor::builder()
            .out_of_bounds(OutOfBounds::Clip)
            .build();
        let fitted = model.fit_1d(&x, &y).expect("model should fit");

        // Below range
        assert_eq!(fitted.predict_single(0.0), fitted.predict_single(1.0));
        // Above range
        assert_eq!(fitted.predict_single(10.0), fitted.predict_single(5.0));
    }

    #[test]
    fn test_weighted() {
        let x = Col::from_fn(4, |i| i as f64);
        let y = Col::from_fn(4, |i| (i + 1) as f64);
        let w = Col::from_fn(4, |i| if i == 0 { 10.0 } else { 1.0 });

        let model = IsotonicRegressor::builder().weights(w).build();
        let fitted = model.fit_1d(&x, &y).expect("model should fit");

        assert!(fitted.fitted_values()[0] <= fitted.fitted_values()[1]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = Col::from_fn(10, |i| i as f64);
        let y = Col::from_fn(5, |i| i as f64);

        let model = IsotonicRegressor::builder().build();
        let result = model.fit_1d(&x, &y);
        assert!(result.is_err());
    }
}
