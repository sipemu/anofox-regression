//! Core traits for regression estimators.

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use faer::{Col, Mat};
use thiserror::Error;

/// Errors that can occur during regression fitting.
#[derive(Debug, Error)]
pub enum RegressionError {
    /// `x` and `y` (or a row-aligned extra) have different lengths.
    #[error("dimension mismatch: X has {x_rows} rows but y has {y_len} elements")]
    DimensionMismatch {
        /// Rows of `x`.
        x_rows: usize,
        /// Length of `y` (or of the extra).
        y_len: usize,
    },

    /// Too few observations for the model.
    #[error("insufficient observations: need at least {needed}, got {got}")]
    InsufficientObservations {
        /// Observations the fit needs.
        needed: usize,
        /// Observations available.
        got: usize,
    },

    /// A matrix that must be inverted is singular or nearly so.
    #[error("matrix is singular or nearly singular")]
    SingularMatrix,

    /// Every predictor is constant.
    #[error("all features are constant")]
    AllFeaturesConstant,

    /// The options failed validation.
    #[error("invalid options: {0}")]
    InvalidOptions(#[from] crate::core::OptionsError),

    /// An iterative fit did not converge.
    #[error("convergence failed after {iterations} iterations")]
    ConvergenceFailed {
        /// Iterations performed.
        iterations: usize,
    },

    /// Weights are negative (or otherwise invalid).
    #[error("invalid weights: all weights must be non-negative")]
    InvalidWeights,

    /// Invalid input or a numerical failure, with a description.
    #[error("numerical error: {0}")]
    NumericalError(String),
}

/// A regression estimator that can be fit to data.
///
/// This trait follows the sklearn pattern where fitting returns a fitted model
/// that can then make predictions.
pub trait Regressor {
    /// The type of the fitted model.
    type Fitted: FittedRegressor;

    /// Fit the model to the data.
    ///
    /// # Arguments
    /// * `x` - Design matrix of shape (n_samples, n_features)
    /// * `y` - Target vector of length n_samples
    ///
    /// # Returns
    /// A fitted model that can make predictions.
    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError>;
}

/// A fitted regression model that can make predictions.
pub trait FittedRegressor {
    /// Make predictions on new data.
    ///
    /// # Arguments
    /// * `x` - Design matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Predicted values vector of length n_samples.
    fn predict(&self, x: &Mat<f64>) -> Col<f64>;

    /// Access the regression results (coefficients, statistics, etc.).
    fn result(&self) -> &RegressionResult;

    /// Get the coefficients (convenience method).
    fn coefficients(&self) -> &Col<f64> {
        &self.result().coefficients
    }

    /// Get the intercept (convenience method).
    fn intercept(&self) -> Option<f64> {
        self.result().intercept
    }

    /// Get R² (convenience method).
    fn r_squared(&self) -> f64 {
        self.result().r_squared
    }

    /// Calculate the score (R²) on new data.
    ///
    /// # Arguments
    /// * `x` - Design matrix
    /// * `y` - True target values
    fn score(&self, x: &Mat<f64>, y: &Col<f64>) -> f64 {
        let predictions = self.predict(x);
        let n = y.nrows();

        // Calculate mean of y
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        // Calculate TSS and RSS
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = y
            .iter()
            .zip(predictions.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();

        if tss == 0.0 {
            // Perfect prediction of constant target
            if rss == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - rss / tss
        }
    }

    /// Make predictions with confidence or prediction intervals.
    ///
    /// This method follows R's `predict(..., interval = "confidence" | "prediction")` API.
    ///
    /// # Arguments
    /// * `x` - Design matrix of shape (n_samples, n_features)
    /// * `interval` - Type of interval: `None` for point predictions only,
    ///   `Some(IntervalType::Confidence)` for confidence intervals on the mean response,
    ///   `Some(IntervalType::Prediction)` for prediction intervals on new observations
    /// * `level` - Confidence level (e.g., 0.95 for 95% intervals)
    ///
    /// # Returns
    /// A `PredictionResult` containing:
    /// - `fit`: Point predictions
    /// - `lower`: Lower bounds (same as fit if interval is None)
    /// - `upper`: Upper bounds (same as fit if interval is None)
    /// - `se`: Standard errors (zeros if interval is None)
    ///
    /// # Example
    /// ```rust
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use faer::{Col, Mat};
    /// # let x = Mat::from_fn(50, 2, |i, j| if j == 0 { i as f64 / 10.0 } else { ((i * 7) % 11) as f64 / 10.0 });
    /// # let y = Col::from_fn(50, |i| 1.0 + 2.0 * x[(i, 0)] - x[(i, 1)] + ((i * 13) % 7) as f64 / 20.0);
    /// use anofox_regression::prelude::*;
    /// # let x_new = Mat::from_fn(3, 2, |i, j| (i + j) as f64 / 4.0);
    ///
    /// let fitted = OlsRegressor::builder().build().fit(&x, &y)?;
    ///
    /// // Point predictions only
    /// let pred = fitted.predict_with_interval(&x_new, None, 0.95);
    ///
    /// // 95% prediction intervals
    /// let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    /// println!("Lower: {:?}, Upper: {:?}", pred.lower, pred.upper);
    ///
    /// // 99% confidence intervals for mean response
    /// let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.99);
    /// # let _ = pred;
    /// # Ok(())
    /// # }
    /// ```
    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult;
}

/// Convergence of an iterative fit: whether it met its stopping rule and
/// after how many iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Convergence {
    pub(crate) converged: bool,
    pub(crate) iterations: usize,
}
