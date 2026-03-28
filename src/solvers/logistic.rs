//! Logistic regression classifier.
//!
//! A convenience wrapper around [`BinomialRegressor`] with logit link that
//! provides an sklearn-like classifier API: `predict` returns class labels,
//! `predict_proba` returns probabilities, and `score` returns accuracy.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{LogisticRegression, Regressor};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 1, |i, _| (i as f64) / 50.0 - 1.0);
//! let y = Col::from_fn(100, |i| if i >= 50 { 1.0 } else { 0.0 });
//!
//! let model = LogisticRegression::builder()
//!     .l2(0.01)
//!     .threshold(0.5)
//!     .build();
//!
//! let fitted = model.fit(&x, &y).unwrap();
//!
//! let labels = fitted.predict(&x);      // 0.0 or 1.0
//! let probs = fitted.predict_proba(&x);  // probabilities in [0, 1]
//! let acc = fitted.score(&x, &y);        // classification accuracy
//! ```

use crate::solvers::binomial::{BinomialRegressor, FittedBinomial};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};

/// L2 regularization penalty for logistic regression.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Penalty {
    /// No regularization.
    #[default]
    None,
    /// L2 (Ridge) regularization with the given strength lambda.
    L2(f64),
}

/// Logistic regression classifier.
///
/// A wrapper around [`BinomialRegressor`] with logit link that provides a
/// classifier-oriented API. Unlike the generic [`FittedRegressor`] trait,
/// [`FittedLogistic`] returns class labels from `predict` and accuracy from
/// `score`.
///
/// Use [`LogisticRegression::builder()`] or [`LogisticRegressionBuilder`] to
/// configure the model before fitting.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    penalty: Penalty,
    with_intercept: bool,
    threshold: f64,
    max_iterations: usize,
    tolerance: f64,
    compute_inference: bool,
    confidence_level: f64,
}

impl LogisticRegression {
    /// Create a new builder for configuring logistic regression.
    pub fn builder() -> LogisticRegressionBuilder {
        LogisticRegressionBuilder::default()
    }

    /// Fit the logistic regression model to the data.
    ///
    /// # Arguments
    /// * `x` - Design matrix of shape (n_samples, n_features)
    /// * `y` - Binary target vector (values must be 0.0 or 1.0)
    ///
    /// # Errors
    /// Returns [`RegressionError::InvalidWeights`] if `y` contains values other
    /// than 0.0 and 1.0. May also return errors from the underlying binomial
    /// regression fit.
    pub fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedLogistic, RegressionError> {
        // Validate that y is binary (only 0.0 and 1.0)
        for i in 0..y.nrows() {
            let val = y[i];
            if val != 0.0 && val != 1.0 {
                return Err(RegressionError::NumericalError(format!(
                    "y must be binary (0.0 or 1.0), found {} at index {}",
                    val, i
                )));
            }
        }

        // Build the underlying BinomialRegressor with logit link
        let mut builder = BinomialRegressor::logistic()
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level);

        if let Penalty::L2(lambda) = self.penalty {
            builder = builder.lambda(lambda);
        }

        let binomial = builder.build();
        let inner = binomial.fit(x, y)?;

        Ok(FittedLogistic {
            inner,
            threshold: self.threshold,
        })
    }
}

/// A fitted logistic regression classifier.
///
/// This struct does **not** implement [`FittedRegressor`] because:
/// - `FittedRegressor::predict` returns continuous values, but a classifier's
///   `predict` should return class labels (0.0 or 1.0).
/// - `FittedRegressor::score` returns R-squared, but a classifier's `score` should
///   return classification accuracy.
///
/// Use [`inner()`](FittedLogistic::inner) to access the underlying
/// [`FittedBinomial`] when you need the full GLM interface.
#[derive(Debug)]
pub struct FittedLogistic {
    inner: FittedBinomial,
    threshold: f64,
}

impl FittedLogistic {
    /// Predict class labels (0.0 or 1.0) for the given data.
    ///
    /// Applies the classification threshold to predicted probabilities:
    /// label = 1.0 if P(Y=1|X) >= threshold, else 0.0.
    pub fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let probs = self.predict_proba(x);
        let threshold = self.threshold;
        Col::from_fn(
            probs.nrows(),
            |i| {
                if probs[i] >= threshold {
                    1.0
                } else {
                    0.0
                }
            },
        )
    }

    /// Predict probabilities P(Y=1|X) for the given data.
    ///
    /// Returns values in [0, 1] representing the estimated probability of
    /// belonging to class 1.
    pub fn predict_proba(&self, x: &Mat<f64>) -> Col<f64> {
        self.inner.predict_probability(x)
    }

    /// Compute the decision function (log-odds / linear predictor) for the given data.
    ///
    /// Positive values correspond to class 1 predictions (when threshold = 0.5),
    /// negative values to class 0.
    pub fn decision_function(&self, x: &Mat<f64>) -> Col<f64> {
        self.inner.predict_linear(x)
    }

    /// Compute classification accuracy on the given data.
    ///
    /// accuracy = (number of correct predictions) / (total predictions)
    pub fn score(&self, x: &Mat<f64>, y: &Col<f64>) -> f64 {
        let predictions = self.predict(x);
        let n = y.nrows();
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();
        correct as f64 / n as f64
    }

    /// Access the regression coefficients (excluding intercept).
    pub fn coefficients(&self) -> &Col<f64> {
        &self.inner.result().coefficients
    }

    /// Access the intercept term, if the model was fit with one.
    pub fn intercept(&self) -> Option<f64> {
        self.inner.result().intercept
    }

    /// Access the underlying [`FittedBinomial`] model.
    ///
    /// Useful when you need full GLM diagnostics, residuals, or the
    /// [`FittedRegressor`] trait methods.
    pub fn inner(&self) -> &FittedBinomial {
        &self.inner
    }

    /// Return the number of IRLS iterations used to fit the model.
    pub fn n_iter(&self) -> usize {
        self.inner.iterations
    }
}

/// Builder for configuring a [`LogisticRegression`] model.
///
/// # Defaults
///
/// | Parameter          | Default |
/// |--------------------|---------|
/// | penalty            | None    |
/// | with_intercept     | true    |
/// | threshold          | 0.5     |
/// | max_iterations     | 100     |
/// | tolerance          | 1e-8    |
/// | compute_inference  | true    |
/// | confidence_level   | 0.95    |
#[derive(Debug, Clone)]
pub struct LogisticRegressionBuilder {
    penalty: Penalty,
    with_intercept: bool,
    threshold: f64,
    max_iterations: usize,
    tolerance: f64,
    compute_inference: bool,
    confidence_level: f64,
}

impl Default for LogisticRegressionBuilder {
    fn default() -> Self {
        Self {
            penalty: Penalty::None,
            with_intercept: true,
            threshold: 0.5,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: true,
            confidence_level: 0.95,
        }
    }
}

impl LogisticRegressionBuilder {
    /// Set the regularization penalty.
    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set L2 regularization with the given lambda (regularization strength).
    ///
    /// Equivalent to `penalty(Penalty::L2(lambda))`.
    pub fn l2(mut self, lambda: f64) -> Self {
        self.penalty = Penalty::L2(lambda);
        self
    }

    /// Set L2 regularization using sklearn's C convention (inverse regularization strength).
    ///
    /// `C = 1 / lambda`, so larger C means less regularization.
    ///
    /// # Panics
    /// Panics if `c_value` is zero or negative.
    pub fn c(mut self, c_value: f64) -> Self {
        assert!(c_value > 0.0, "C must be positive, got {}", c_value);
        self.penalty = Penalty::L2(1.0 / c_value);
        self
    }

    /// Set whether to include an intercept (bias) term. Default: true.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Set the classification threshold. Default: 0.5.
    ///
    /// Predicted class is 1 if P(Y=1|X) >= threshold, else 0.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum number of IRLS iterations. Default: 100.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set the convergence tolerance. Default: 1e-8.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set whether to compute inference statistics (standard errors, p-values, etc.).
    /// Default: true.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.compute_inference = compute;
        self
    }

    /// Set the confidence level for confidence intervals. Default: 0.95.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Build the [`LogisticRegression`] model.
    pub fn build(self) -> LogisticRegression {
        LogisticRegression {
            penalty: self.penalty,
            with_intercept: self.with_intercept,
            threshold: self.threshold,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            compute_inference: self.compute_inference,
            confidence_level: self.confidence_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create binary classification data with overlapping classes.
    ///
    /// Uses a probabilistic relationship (not perfectly separable) to ensure
    /// the logistic regression converges reliably.
    fn create_test_data(n: usize) -> (Mat<f64>, Col<f64>) {
        let x = Mat::from_fn(n, 1, |i, _| (i as f64) / (n as f64) * 4.0 - 2.0);
        let y = Col::from_fn(n, |i| {
            let xi = (i as f64) / (n as f64) * 4.0 - 2.0;
            // Probabilistic: P(y=1) increases with x, with some overlap
            let prob = 1.0 / (1.0 + (-xi).exp());
            if prob > 0.5 + 0.1 * ((i % 5) as f64 - 2.0) / 2.0 {
                1.0
            } else {
                0.0
            }
        });
        (x, y)
    }

    #[test]
    fn test_logistic_defaults() {
        let builder = LogisticRegressionBuilder::default();
        assert_eq!(builder.penalty, Penalty::None);
        assert!(builder.with_intercept);
        assert!((builder.threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(builder.max_iterations, 100);
        assert!((builder.tolerance - 1e-8).abs() < f64::EPSILON);
        assert!(builder.compute_inference);
        assert!((builder.confidence_level - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logistic_binary_classification() {
        let (x, y) = create_test_data(100);

        let model = LogisticRegression::builder().build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        // Coefficient should be positive (higher x -> higher P(y=1))
        assert!(
            fitted.coefficients()[0] > 0.0,
            "coefficient should be positive, got {}",
            fitted.coefficients()[0]
        );

        // Intercept should exist
        assert!(fitted.intercept().is_some());

        // Accuracy should be high on separable data
        let acc = fitted.score(&x, &y);
        assert!(acc > 0.8, "accuracy should be > 0.8, got {}", acc);

        // n_iter should be positive
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_predict_proba_range() {
        let (x, y) = create_test_data(100);

        let fitted = LogisticRegression::builder()
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let probs = fitted.predict_proba(&x);
        for i in 0..probs.nrows() {
            assert!(
                probs[i] >= 0.0 && probs[i] <= 1.0,
                "probability at index {} is {}, expected [0, 1]",
                i,
                probs[i]
            );
        }
    }

    #[test]
    fn test_predict_class_labels() {
        let (x, y) = create_test_data(100);

        let fitted = LogisticRegression::builder()
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let labels = fitted.predict(&x);
        for i in 0..labels.nrows() {
            assert!(
                labels[i] == 0.0 || labels[i] == 1.0,
                "label at index {} is {}, expected 0.0 or 1.0",
                i,
                labels[i]
            );
        }
    }

    #[test]
    fn test_decision_function_sign() {
        let (x, y) = create_test_data(100);

        let fitted = LogisticRegression::builder()
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let decision = fitted.decision_function(&x);
        let labels = fitted.predict(&x);

        for i in 0..decision.nrows() {
            if decision[i] > 0.0 {
                assert_eq!(
                    labels[i], 1.0,
                    "positive decision at index {} should yield class 1",
                    i
                );
            } else if decision[i] < 0.0 {
                assert_eq!(
                    labels[i], 0.0,
                    "negative decision at index {} should yield class 0",
                    i
                );
            }
            // decision == 0.0 is on the boundary; either class is acceptable
        }
    }

    #[test]
    fn test_score_accuracy() {
        let (x, y) = create_test_data(100);

        let fitted = LogisticRegression::builder()
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let acc = fitted.score(&x, &y);

        // Manually compute accuracy for verification
        let predictions = fitted.predict(&x);
        let n = y.nrows();
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| p == a)
            .count();
        let expected_acc = correct as f64 / n as f64;

        assert!(
            (acc - expected_acc).abs() < f64::EPSILON,
            "score() returned {}, expected {}",
            acc,
            expected_acc
        );
    }

    #[test]
    fn test_threshold() {
        let (x, y) = create_test_data(100);

        let fitted_default = LogisticRegression::builder()
            .threshold(0.5)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // A very high threshold should predict more 0s
        let fitted_high = LogisticRegression::builder()
            .threshold(0.9)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let labels_default = fitted_default.predict(&x);
        let labels_high = fitted_high.predict(&x);

        let count_ones_default: usize = labels_default.iter().filter(|&&v| v == 1.0).count();
        let count_ones_high: usize = labels_high.iter().filter(|&&v| v == 1.0).count();

        assert!(
            count_ones_high <= count_ones_default,
            "higher threshold should predict fewer 1s: {} (threshold=0.9) vs {} (threshold=0.5)",
            count_ones_high,
            count_ones_default
        );
    }

    #[test]
    fn test_l2_regularization() {
        let (x, y) = create_test_data(100);

        let fitted_no_reg = LogisticRegression::builder()
            .build()
            .fit(&x, &y)
            .expect("model should fit without regularization");

        let fitted_l2 = LogisticRegression::builder()
            .l2(10.0)
            .build()
            .fit(&x, &y)
            .expect("model should fit with L2 regularization");

        // L2 regularization should shrink coefficients toward zero
        let coef_no_reg = fitted_no_reg.coefficients()[0].abs();
        let coef_l2 = fitted_l2.coefficients()[0].abs();

        assert!(
            coef_l2 < coef_no_reg,
            "L2 regularization should shrink coefficient: |{}| (L2) should be < |{}| (no reg)",
            coef_l2,
            coef_no_reg
        );
    }

    #[test]
    fn test_invalid_y_values() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| i as f64 * 0.1); // not binary

        let model = LogisticRegression::builder().build();
        let result = model.fit(&x, &y);

        assert!(result.is_err(), "should reject non-binary y values");
    }

    #[test]
    fn test_c_convention() {
        // C = 1/lambda, so C = 0.1 means lambda = 10.0
        let builder = LogisticRegressionBuilder::default().c(0.1);
        assert_eq!(builder.penalty, Penalty::L2(10.0));
    }

    #[test]
    fn test_inner_access() {
        let (x, y) = create_test_data(100);

        let fitted = LogisticRegression::builder()
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // inner() should return the underlying FittedBinomial
        let inner = fitted.inner();

        // FittedBinomial implements FittedRegressor, so result() is available
        let result = inner.result();
        assert_eq!(result.coefficients.nrows(), 1);
    }
}
