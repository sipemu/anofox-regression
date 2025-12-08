//! Prediction interval calculations.

use crate::core::{IntervalType, PredictionResult};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Computes prediction intervals for new data points.
///
/// # Arguments
/// * `x_new` - New data points (n_new × n_features)
/// * `xtx_inv` - (X'X)⁻¹ or (X_aug'X_aug)⁻¹ if has_intercept
/// * `predictions` - Point predictions for x_new
/// * `mse` - Mean squared error from the fitted model
/// * `df` - Degrees of freedom for t-distribution
/// * `confidence_level` - Confidence level (e.g., 0.95)
/// * `interval_type` - Confidence or Prediction interval
/// * `has_intercept` - Whether the model has an intercept (x_new needs augmentation)
///
/// # Returns
/// PredictionResult containing fit, lower, upper bounds and standard errors
#[allow(clippy::too_many_arguments)]
pub fn compute_prediction_intervals(
    x_new: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    predictions: &Col<f64>,
    mse: f64,
    df: f64,
    confidence_level: f64,
    interval_type: IntervalType,
    has_intercept: bool,
) -> PredictionResult {
    let n_new = x_new.nrows();
    let n_features = x_new.ncols();

    let mut se = Col::zeros(n_new);
    let mut lower = Col::zeros(n_new);
    let mut upper = Col::zeros(n_new);

    // Handle edge cases
    if df <= 0.0 || mse <= 0.0 {
        for i in 0..n_new {
            se[i] = f64::NAN;
            lower[i] = f64::NAN;
            upper[i] = f64::NAN;
        }
        return PredictionResult::with_intervals(predictions.clone(), lower, upper, se);
    }

    // Get t-critical value
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let alpha = 1.0 - confidence_level;
    let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);

    // For each new observation, compute interval
    for i in 0..n_new {
        // Build x₀ vector (possibly augmented with 1 for intercept)
        let x0: Col<f64> = if has_intercept {
            let mut x0_aug = Col::zeros(n_features + 1);
            x0_aug[0] = 1.0;
            for j in 0..n_features {
                x0_aug[j + 1] = x_new[(i, j)];
            }
            x0_aug
        } else {
            let mut x0 = Col::zeros(n_features);
            for j in 0..n_features {
                x0[j] = x_new[(i, j)];
            }
            x0
        };

        // Compute h = x₀'(X'X)⁻¹x₀ (leverage for this point)
        // This is a scalar: h = x₀ᵀ × (X'X)⁻¹ × x₀
        let h = compute_leverage_single(&x0, xtx_inv);

        // Compute variance
        let var = match interval_type {
            IntervalType::Confidence => mse * h,
            IntervalType::Prediction => mse * (1.0 + h),
        };

        // Standard error
        se[i] = if var >= 0.0 { var.sqrt() } else { f64::NAN };

        // Confidence/Prediction bounds
        let margin = t_crit * se[i];
        lower[i] = predictions[i] - margin;
        upper[i] = predictions[i] + margin;
    }

    PredictionResult::with_intervals(predictions.clone(), lower, upper, se)
}

/// Compute leverage h = x₀'(X'X)⁻¹x₀ for a single observation.
fn compute_leverage_single(x0: &Col<f64>, xtx_inv: &Mat<f64>) -> f64 {
    let p = x0.nrows();

    // Compute (X'X)⁻¹ × x₀
    let mut xtx_inv_x0 = Col::zeros(p);
    for i in 0..p {
        let mut sum = 0.0;
        for j in 0..p {
            sum += xtx_inv[(i, j)] * x0[j];
        }
        xtx_inv_x0[i] = sum;
    }

    // Compute x₀' × ((X'X)⁻¹ × x₀)
    let mut h = 0.0;
    for i in 0..p {
        h += x0[i] * xtx_inv_x0[i];
    }

    h
}

/// Compute (X'X)⁻¹ for the augmented design matrix [1 | X].
///
/// This is used by fitted models to store the inverse for prediction intervals.
pub fn compute_xtx_inverse_augmented(x: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let aug_size = n_features + 1;

    // Build augmented design matrix [1 | X]
    let mut x_aug = Mat::zeros(n_samples, aug_size);
    for i in 0..n_samples {
        x_aug[(i, 0)] = 1.0;
        for j in 0..n_features {
            x_aug[(i, j + 1)] = x[(i, j)];
        }
    }

    // Compute X_aug'X_aug
    let xtx_aug = x_aug.transpose() * &x_aug;

    // Compute inverse using QR decomposition
    compute_matrix_inverse(&xtx_aug)
}

/// Compute (X'WX)⁻¹ for the weighted augmented design matrix.
pub fn compute_xtwx_inverse_augmented(
    x: &Mat<f64>,
    weights: &Col<f64>,
) -> Result<Mat<f64>, &'static str> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let aug_size = n_features + 1;

    // Build X'WX for augmented matrix [1 | X]
    let mut xtwx_aug: Mat<f64> = Mat::zeros(aug_size, aug_size);

    for i in 0..n_samples {
        let w = weights[i];

        // (0,0): sum of weights
        xtwx_aug[(0, 0)] += w;

        // (0,j+1) and (j+1,0): weighted sum of x_j
        for j in 0..n_features {
            xtwx_aug[(0, j + 1)] += w * x[(i, j)];
            xtwx_aug[(j + 1, 0)] += w * x[(i, j)];
        }

        // (j+1, k+1): weighted x_j * x_k
        for j in 0..n_features {
            for k in 0..n_features {
                xtwx_aug[(j + 1, k + 1)] += w * x[(i, j)] * x[(i, k)];
            }
        }
    }

    compute_matrix_inverse(&xtwx_aug)
}

/// Compute (X'X)⁻¹ for a non-augmented design matrix.
pub fn compute_xtx_inverse(x: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
    let xtx = x.transpose() * x;
    compute_matrix_inverse(&xtx)
}

/// General matrix inverse using QR decomposition.
fn compute_matrix_inverse(matrix: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
    let n = matrix.nrows();

    let qr: faer::linalg::solvers::Qr<f64> = matrix.qr();
    let q = qr.compute_q();
    let r = qr.compute_r();

    // Check if R is singular
    for i in 0..n {
        if r[(i, i)].abs() < 1e-10 {
            return Err("Matrix is singular");
        }
    }

    // Solve R * X = Q' for each column of identity to get inverse
    let mut inv = Mat::zeros(n, n);
    let qt = q.transpose();

    for col in 0..n {
        for i in (0..n).rev() {
            let mut sum = qt[(i, col)];
            for j in (i + 1)..n {
                sum -= r[(i, j)] * inv[(j, col)];
            }
            inv[(i, col)] = sum / r[(i, i)];
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leverage_single() {
        // Simple 2x2 identity matrix case
        let x0 = Col::from_fn(2, |i| (i + 1) as f64);
        let xtx_inv = Mat::identity(2, 2);

        let h = compute_leverage_single(&x0, &xtx_inv);

        // h = x₀'Ix₀ = ||x₀||² = 1² + 2² = 5
        assert!((h - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_interval_wider_than_confidence() {
        // Create simple test data
        let x_new = Mat::from_fn(3, 1, |_, _| 1.0);
        let xtx_inv = Mat::identity(2, 2); // For augmented [1|X]
        let predictions = Col::from_fn(3, |i| i as f64);
        let mse = 1.0;
        let df = 10.0;

        let ci = compute_prediction_intervals(
            &x_new,
            &xtx_inv,
            &predictions,
            mse,
            df,
            0.95,
            IntervalType::Confidence,
            true,
        );

        let pi = compute_prediction_intervals(
            &x_new,
            &xtx_inv,
            &predictions,
            mse,
            df,
            0.95,
            IntervalType::Prediction,
            true,
        );

        // Prediction interval should be wider
        for i in 0..3 {
            let ci_width = ci.upper[i] - ci.lower[i];
            let pi_width = pi.upper[i] - pi.lower[i];
            assert!(pi_width > ci_width, "PI should be wider than CI");
        }
    }
}
