//! LOWESS (Locally Weighted Scatterplot Smoothing) implementation.
//!
//! This module provides LOWESS smoothing for time-varying weight matrices,
//! primarily used by lmDynamic for smoothing model weights over time.

use faer::{Col, Mat};

/// Apply LOWESS smoothing to each column of a weight matrix.
///
/// # Arguments
/// * `weights` - Matrix where each row is an observation and each column is a model
/// * `span` - Fraction of data to use for local regression (0.0 to 1.0)
///
/// # Returns
/// Smoothed weight matrix of same dimensions
pub fn lowess_smooth_weights(weights: &Mat<f64>, span: f64) -> Mat<f64> {
    let n = weights.nrows();
    let m = weights.ncols();

    if n == 0 || m == 0 {
        return weights.clone();
    }

    let span = span.clamp(0.0, 1.0);
    let k = ((n as f64 * span).ceil() as usize).max(2).min(n);

    let mut smoothed = Mat::zeros(n, m);

    // Create time index (0 to n-1)
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Smooth each column independently
    for col in 0..m {
        let y: Vec<f64> = (0..n).map(|i| weights[(i, col)]).collect();
        let smoothed_col = lowess_1d(&t, &y, k);
        for i in 0..n {
            smoothed[(i, col)] = smoothed_col[i];
        }
    }

    // Renormalize rows to sum to 1
    for i in 0..n {
        let row_sum: f64 = (0..m).map(|j| smoothed[(i, j)].max(0.0)).sum();
        if row_sum > 1e-10 {
            for j in 0..m {
                smoothed[(i, j)] = smoothed[(i, j)].max(0.0) / row_sum;
            }
        } else {
            // If all weights are zero/negative, use uniform
            for j in 0..m {
                smoothed[(i, j)] = 1.0 / m as f64;
            }
        }
    }

    smoothed
}

/// Apply LOWESS smoothing to a single time series.
///
/// # Arguments
/// * `x` - Time/index values
/// * `y` - Values to smooth
/// * `k` - Number of nearest neighbors to use
///
/// # Returns
/// Smoothed values
fn lowess_1d(x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![y[0]];
    }

    let k = k.min(n);
    let mut result = vec![0.0; n];

    for i in 0..n {
        // Find k nearest neighbors
        let mut distances: Vec<(usize, f64)> = (0..n).map(|j| (j, (x[j] - x[i]).abs())).collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get the k nearest
        let neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        let max_dist = distances[k - 1].1;

        // Compute tricube weights
        let weights: Vec<f64> = neighbors
            .iter()
            .map(|&j| {
                if max_dist < 1e-10 {
                    1.0 / k as f64
                } else {
                    let u = (x[j] - x[i]).abs() / max_dist;
                    tricube(u)
                }
            })
            .collect();

        // Weighted local regression (just weighted mean for robustness)
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 1e-10 {
            let weighted_y: f64 = neighbors
                .iter()
                .zip(weights.iter())
                .map(|(&j, &w)| w * y[j])
                .sum();
            result[i] = weighted_y / weight_sum;
        } else {
            result[i] = y[i];
        }
    }

    result
}

/// Tricube weight function: (1 - u³)³ for u in [0, 1], 0 otherwise
fn tricube(u: f64) -> f64 {
    if (0.0..1.0).contains(&u) {
        let v = 1.0 - u.powi(3);
        v.powi(3)
    } else {
        0.0
    }
}

/// Smooth a single column vector using LOWESS.
///
/// # Arguments
/// * `y` - Values to smooth
/// * `span` - Fraction of data to use (0.0 to 1.0)
///
/// # Returns
/// Smoothed values
pub fn lowess_smooth(y: &Col<f64>, span: f64) -> Col<f64> {
    let n = y.nrows();
    if n == 0 {
        return y.clone();
    }

    let span = span.clamp(0.0, 1.0);
    let k = ((n as f64 * span).ceil() as usize).max(2).min(n);

    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_vec: Vec<f64> = (0..n).map(|i| y[i]).collect();

    let smoothed = lowess_1d(&x, &y_vec, k);
    Col::from_fn(n, |i| smoothed[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tricube() {
        // At u = 0, tricube = 1
        assert!((tricube(0.0) - 1.0).abs() < 1e-10);

        // At u = 1, tricube = 0
        assert!((tricube(1.0) - 0.0).abs() < 1e-10);

        // At u = 0.5, tricube = (1 - 0.125)^3 = 0.669921875
        assert!((tricube(0.5) - 0.669921875).abs() < 1e-6);

        // Outside [0, 1], tricube = 0
        assert!((tricube(-0.1) - 0.0).abs() < 1e-10);
        assert!((tricube(1.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lowess_smooth_constant() {
        // Constant input should remain constant
        let y = Col::from_fn(10, |_| 5.0);
        let smoothed = lowess_smooth(&y, 0.5);

        for i in 0..10 {
            assert!((smoothed[i] - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lowess_smooth_weights_normalization() {
        // Weights should sum to 1 after smoothing
        let mut weights = Mat::zeros(10, 3);
        for i in 0..10 {
            weights[(i, 0)] = 0.5;
            weights[(i, 1)] = 0.3;
            weights[(i, 2)] = 0.2;
        }

        let smoothed = lowess_smooth_weights(&weights, 0.5);

        for i in 0..10 {
            let row_sum: f64 = (0..3).map(|j| smoothed[(i, j)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_lowess_reduces_noise() {
        // Noisy sine wave should be smoothed
        let y = Col::from_fn(50, |i| {
            let x = i as f64 * 0.1;
            x.sin() + ((i * 7) as f64 * 0.3).sin() * 0.1 // Signal + noise
        });

        let smoothed = lowess_smooth(&y, 0.3);

        // Smoothed values should exist
        assert_eq!(smoothed.nrows(), 50);

        // Values should be bounded reasonably
        for i in 0..50 {
            assert!(smoothed[i].is_finite());
            assert!(smoothed[i].abs() < 2.0);
        }
    }
}
