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

    if df <= 0.0 || mse < 0.0 {
        return create_nan_result(predictions, n_new);
    }

    let t_crit = compute_t_critical(df, confidence_level);
    let (se, lower, upper) = compute_all_intervals(
        x_new,
        xtx_inv,
        predictions,
        mse,
        t_crit,
        interval_type,
        has_intercept,
    );

    PredictionResult::with_intervals(predictions.clone(), lower, upper, se)
}

/// Create a result with NaN values for invalid parameters.
fn create_nan_result(predictions: &Col<f64>, n: usize) -> PredictionResult {
    let se = Col::from_fn(n, |_| f64::NAN);
    let lower = Col::from_fn(n, |_| f64::NAN);
    let upper = Col::from_fn(n, |_| f64::NAN);
    PredictionResult::with_intervals(predictions.clone(), lower, upper, se)
}

/// Compute the t-critical value for confidence intervals.
fn compute_t_critical(df: f64, confidence_level: f64) -> f64 {
    let t_dist = StudentsT::new(0.0, 1.0, df).expect("valid t-distribution parameters");
    let alpha = 1.0 - confidence_level;
    t_dist.inverse_cdf(1.0 - alpha / 2.0)
}

/// Compute intervals for all observations.
fn compute_all_intervals(
    x_new: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    predictions: &Col<f64>,
    mse: f64,
    t_crit: f64,
    interval_type: IntervalType,
    has_intercept: bool,
) -> (Col<f64>, Col<f64>, Col<f64>) {
    let n_new = x_new.nrows();
    let mut se = Col::zeros(n_new);
    let mut lower = Col::zeros(n_new);
    let mut upper = Col::zeros(n_new);

    for i in 0..n_new {
        let (s, l, u) = compute_single_interval(
            x_new,
            xtx_inv,
            predictions[i],
            mse,
            t_crit,
            interval_type,
            has_intercept,
            i,
        );
        se[i] = s;
        lower[i] = l;
        upper[i] = u;
    }

    (se, lower, upper)
}

/// Compute interval for a single observation.
#[allow(clippy::too_many_arguments)]
fn compute_single_interval(
    x_new: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    prediction: f64,
    mse: f64,
    t_crit: f64,
    interval_type: IntervalType,
    has_intercept: bool,
    row: usize,
) -> (f64, f64, f64) {
    let x0 = build_observation_vector(x_new, row, has_intercept);
    let h = compute_leverage_single(&x0, xtx_inv);
    let var = compute_interval_variance(mse, h, interval_type);
    let se = if var >= 0.0 { var.sqrt() } else { f64::NAN };
    let margin = t_crit * se;
    (se, prediction - margin, prediction + margin)
}

/// Build the observation vector, optionally augmented with intercept.
fn build_observation_vector(x_new: &Mat<f64>, row: usize, has_intercept: bool) -> Col<f64> {
    let n_features = x_new.ncols();
    if has_intercept {
        let mut x0 = Col::zeros(n_features + 1);
        x0[0] = 1.0;
        for j in 0..n_features {
            x0[j + 1] = x_new[(row, j)];
        }
        x0
    } else {
        Col::from_fn(n_features, |j| x_new[(row, j)])
    }
}

/// Compute variance based on interval type.
fn compute_interval_variance(mse: f64, leverage: f64, interval_type: IntervalType) -> f64 {
    match interval_type {
        IntervalType::Confidence => mse * leverage,
        IntervalType::Prediction => mse * (1.0 + leverage),
    }
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
/// This function computes the inverse of X'X where X is augmented with a
/// column of ones for the intercept term. Used for computing standard errors
/// and prediction intervals in models with intercepts.
///
/// # Arguments
/// * `x` - Feature matrix (n × p), WITHOUT the intercept column
///
/// # Returns
/// * `Ok(Mat<f64>)` - The (p+1) × (p+1) inverse matrix
/// * `Err(&'static str)` - If the matrix is singular
///
/// # Example
///
/// ```rust,ignore
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let xtx_inv = compute_xtx_inverse_augmented(&x)?;
/// // xtx_inv is 3×3 (intercept + 2 features)
/// ```
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
///
/// This function computes the inverse of X'WX where X is augmented with a
/// column of ones and W is a diagonal weight matrix. Used for weighted
/// least squares (WLS) and generalized linear models (GLM).
///
/// # Arguments
/// * `x` - Feature matrix (n × p), WITHOUT the intercept column
/// * `weights` - Weight vector (n × 1), typically IRLS weights
///
/// # Returns
/// * `Ok(Mat<f64>)` - The (p+1) × (p+1) inverse matrix
/// * `Err(&'static str)` - If the matrix is singular
///
/// # Example
///
/// ```rust,ignore
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let weights = Col::from_fn(100, |_| 1.0);  // uniform weights
/// let xtwx_inv = compute_xtwx_inverse_augmented(&x, &weights)?;
/// ```
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
///
/// This function computes the inverse of X'X directly without adding
/// an intercept column. Used for models without intercepts.
///
/// # Arguments
/// * `x` - Design matrix (n × p)
///
/// # Returns
/// * `Ok(Mat<f64>)` - The p × p inverse matrix
/// * `Err(&'static str)` - If the matrix is singular
///
/// # Example
///
/// ```rust,ignore
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let xtx_inv = compute_xtx_inverse(&x)?;
/// // xtx_inv is 2×2
/// ```
pub fn compute_xtx_inverse(x: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
    let xtx = x.transpose() * x;
    compute_matrix_inverse(&xtx)
}

/// Compute (X'X)⁻¹ for the augmented design matrix, excluding aliased columns.
///
/// When some features are aliased (collinear or constant), we compute the inverse
/// using only the non-aliased columns. This allows prediction intervals to be
/// computed correctly even when the full matrix is singular.
///
/// # Arguments
/// * `x` - Feature matrix (n × p), WITHOUT the intercept column
/// * `aliased` - Boolean mask indicating which columns are aliased
///
/// # Returns
/// * `Ok(Mat<f64>)` - The (1 + n_non_aliased) × (1 + n_non_aliased) inverse matrix
/// * `Err(&'static str)` - If the reduced matrix is still singular
pub fn compute_xtx_inverse_augmented_reduced(
    x: &Mat<f64>,
    aliased: &[bool],
) -> Result<Mat<f64>, &'static str> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Count non-aliased columns
    let non_aliased_cols: Vec<usize> = (0..n_features).filter(|&j| !aliased[j]).collect();
    let n_reduced = non_aliased_cols.len();
    let aug_size = n_reduced + 1; // +1 for intercept

    // Build reduced augmented design matrix [1 | X_reduced]
    let mut x_aug = Mat::zeros(n_samples, aug_size);
    for i in 0..n_samples {
        x_aug[(i, 0)] = 1.0;
        for (k, &j) in non_aliased_cols.iter().enumerate() {
            x_aug[(i, k + 1)] = x[(i, j)];
        }
    }

    // Compute X_aug'X_aug
    let xtx_aug = x_aug.transpose() * &x_aug;

    // Compute inverse
    compute_matrix_inverse(&xtx_aug)
}

/// Compute (X'X)⁻¹ for a non-augmented design matrix, excluding aliased columns.
///
/// # Arguments
/// * `x` - Design matrix (n × p)
/// * `aliased` - Boolean mask indicating which columns are aliased
///
/// # Returns
/// * `Ok(Mat<f64>)` - The n_non_aliased × n_non_aliased inverse matrix
/// * `Err(&'static str)` - If the reduced matrix is still singular
pub fn compute_xtx_inverse_reduced(
    x: &Mat<f64>,
    aliased: &[bool],
) -> Result<Mat<f64>, &'static str> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Count non-aliased columns
    let non_aliased_cols: Vec<usize> = (0..n_features).filter(|&j| !aliased[j]).collect();
    let n_reduced = non_aliased_cols.len();

    if n_reduced == 0 {
        return Err("All columns are aliased");
    }

    // Build reduced design matrix
    let mut x_reduced = Mat::zeros(n_samples, n_reduced);
    for i in 0..n_samples {
        for (k, &j) in non_aliased_cols.iter().enumerate() {
            x_reduced[(i, k)] = x[(i, j)];
        }
    }

    // Compute X'X and its inverse
    let xtx = x_reduced.transpose() * &x_reduced;
    compute_matrix_inverse(&xtx)
}

/// Compute (X'WX)⁻¹ for the weighted augmented design matrix, excluding aliased columns.
///
/// When some features are aliased (collinear or constant), we compute the inverse
/// using only the non-aliased columns. This allows prediction intervals to be
/// computed correctly even when the full matrix is singular.
///
/// # Arguments
/// * `x` - Feature matrix (n × p), WITHOUT the intercept column
/// * `weights` - Weight vector (n × 1), typically IRLS weights
/// * `aliased` - Boolean mask indicating which columns are aliased
///
/// # Returns
/// * `Ok(Mat<f64>)` - The (1 + n_non_aliased) × (1 + n_non_aliased) inverse matrix
/// * `Err(&'static str)` - If the reduced matrix is still singular
pub fn compute_xtwx_inverse_augmented_reduced(
    x: &Mat<f64>,
    weights: &Col<f64>,
    aliased: &[bool],
) -> Result<Mat<f64>, &'static str> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Count non-aliased columns
    let non_aliased_cols: Vec<usize> = (0..n_features).filter(|&j| !aliased[j]).collect();
    let n_reduced = non_aliased_cols.len();
    let aug_size = n_reduced + 1; // +1 for intercept

    // Build X'WX for reduced augmented matrix [1 | X_reduced]
    let mut xtwx_aug: Mat<f64> = Mat::zeros(aug_size, aug_size);

    for i in 0..n_samples {
        let w = weights[i];

        // (0,0): sum of weights
        xtwx_aug[(0, 0)] += w;

        // (0,k+1) and (k+1,0): weighted sum of x_j (for non-aliased j)
        for (k, &j) in non_aliased_cols.iter().enumerate() {
            xtwx_aug[(0, k + 1)] += w * x[(i, j)];
            xtwx_aug[(k + 1, 0)] += w * x[(i, j)];
        }

        // (k+1, m+1): weighted x_j * x_l (for non-aliased j, l)
        for (k, &j) in non_aliased_cols.iter().enumerate() {
            for (m, &l) in non_aliased_cols.iter().enumerate() {
                xtwx_aug[(k + 1, m + 1)] += w * x[(i, j)] * x[(i, l)];
            }
        }
    }

    compute_matrix_inverse(&xtwx_aug)
}

/// Compute (X'WX)⁻¹ for a weighted non-augmented design matrix, excluding aliased columns.
///
/// # Arguments
/// * `x` - Design matrix (n × p)
/// * `weights` - Weight vector (n × 1)
/// * `aliased` - Boolean mask indicating which columns are aliased
///
/// # Returns
/// * `Ok(Mat<f64>)` - The n_non_aliased × n_non_aliased inverse matrix
/// * `Err(&'static str)` - If the reduced matrix is still singular
pub fn compute_xtwx_inverse_reduced(
    x: &Mat<f64>,
    weights: &Col<f64>,
    aliased: &[bool],
) -> Result<Mat<f64>, &'static str> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Count non-aliased columns
    let non_aliased_cols: Vec<usize> = (0..n_features).filter(|&j| !aliased[j]).collect();
    let n_reduced = non_aliased_cols.len();

    if n_reduced == 0 {
        return Err("All columns are aliased");
    }

    // Build X'WX for reduced matrix
    let mut xtwx: Mat<f64> = Mat::zeros(n_reduced, n_reduced);

    for i in 0..n_samples {
        let w = weights[i];

        for (k, &j) in non_aliased_cols.iter().enumerate() {
            for (m, &l) in non_aliased_cols.iter().enumerate() {
                xtwx[(k, m)] += w * x[(i, j)] * x[(i, l)];
            }
        }
    }

    compute_matrix_inverse(&xtwx)
}

/// General matrix inverse using QR decomposition.
pub(crate) fn compute_matrix_inverse(matrix: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
    let n = matrix.nrows();

    let qr: faer::linalg::solvers::Qr<f64> = matrix.qr();
    let q = qr.compute_Q();
    let r = qr.R();

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

    #[test]
    fn test_compute_matrix_inverse_1x1() {
        // Test 1x1 matrix inversion (the failing case scenario)
        let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
        let xtx = x.transpose() * &x;

        // xtx should be 1x1 with sum of squares = 385
        assert_eq!(xtx.nrows(), 1);
        assert_eq!(xtx.ncols(), 1);
        let expected_sum = (1..=10).map(|i| (i * i) as f64).sum::<f64>();
        assert!((xtx[(0, 0)] - expected_sum).abs() < 1e-10);

        // Now compute inverse
        let inv = compute_matrix_inverse(&xtx).expect("Should not fail");

        // Check dimensions
        assert_eq!(inv.nrows(), 1);
        assert_eq!(inv.ncols(), 1);

        // Check value: inverse of 385 should be 1/385
        let expected_inv = 1.0 / expected_sum;
        assert!(
            (inv[(0, 0)] - expected_inv).abs() < 1e-10,
            "Expected inv = {}, got {}",
            expected_inv,
            inv[(0, 0)]
        );
    }
}
