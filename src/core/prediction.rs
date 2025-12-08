//! Prediction types for interval estimation.

use faer::Col;

/// Type of prediction scale for GLM models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredictionType {
    /// Predictions on the response scale (μ).
    /// For binomial: probabilities in (0, 1).
    /// For Poisson: counts > 0.
    #[default]
    Response,

    /// Predictions on the link scale (η = g(μ)).
    /// The linear predictor: η = Xβ.
    Link,
}

/// Type of interval to compute for predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntervalType {
    /// Confidence interval for the mean response E[Y|X=x₀].
    /// Narrower - only accounts for uncertainty in coefficient estimates.
    Confidence,

    /// Prediction interval for a new observation Y|X=x₀.
    /// Wider - also accounts for residual variance (irreducible error).
    #[default]
    Prediction,
}

/// Result of prediction with optional intervals.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Point predictions (fitted values).
    pub fit: Col<f64>,
    /// Lower bounds of the interval.
    pub lower: Col<f64>,
    /// Upper bounds of the interval.
    pub upper: Col<f64>,
    /// Standard errors of predictions.
    pub se: Col<f64>,
}

impl PredictionResult {
    /// Create a new prediction result with only point predictions (no intervals).
    pub fn point_only(fit: Col<f64>) -> Self {
        let n = fit.nrows();
        Self {
            fit,
            lower: Col::zeros(n),
            upper: Col::zeros(n),
            se: Col::zeros(n),
        }
    }

    /// Create a new prediction result with intervals.
    pub fn with_intervals(fit: Col<f64>, lower: Col<f64>, upper: Col<f64>, se: Col<f64>) -> Self {
        Self {
            fit,
            lower,
            upper,
            se,
        }
    }

    /// Number of predictions.
    pub fn len(&self) -> usize {
        self.fit.nrows()
    }

    /// Returns true if there are no predictions.
    pub fn is_empty(&self) -> bool {
        self.fit.nrows() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_only() {
        let fit = Col::from_fn(5, |i| i as f64);
        let result = PredictionResult::point_only(fit);
        assert_eq!(result.len(), 5);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_with_intervals() {
        let fit = Col::from_fn(3, |i| i as f64);
        let lower = Col::from_fn(3, |i| (i as f64) - 1.0);
        let upper = Col::from_fn(3, |i| (i as f64) + 1.0);
        let se = Col::from_fn(3, |_| 0.5);
        let result = PredictionResult::with_intervals(fit, lower, upper, se);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_empty_result() {
        let fit = Col::<f64>::zeros(0);
        let result = PredictionResult::point_only(fit);
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_interval_type_default() {
        let interval = IntervalType::default();
        assert_eq!(interval, IntervalType::Prediction);
    }
}
