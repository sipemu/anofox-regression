//! Common test utilities and data generators.

use faer::{Col, Mat};

/// Generate simple linear data: y = x * beta + intercept + noise
#[allow(dead_code)]
pub fn generate_linear_data(
    n_samples: usize,
    n_features: usize,
    intercept: f64,
    noise_std: f64,
    seed: u64,
) -> (Mat<f64>, Col<f64>, Col<f64>) {
    // Simple deterministic "random" for reproducibility
    let mut rng_state = seed;
    let next_rand = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
    };

    let mut x = Mat::zeros(n_samples, n_features);
    let mut y = Col::zeros(n_samples);
    let mut true_coefficients = Col::zeros(n_features);

    // Generate true coefficients
    for j in 0..n_features {
        true_coefficients[j] = (j + 1) as f64;
    }

    // Generate X and y
    for i in 0..n_samples {
        let mut yi = intercept;
        for j in 0..n_features {
            x[(i, j)] = next_rand(&mut rng_state);
            yi += x[(i, j)] * true_coefficients[j];
        }
        yi += noise_std * next_rand(&mut rng_state);
        y[i] = yi;
    }

    (x, y, true_coefficients)
}

/// Generate data with collinear features.
#[allow(dead_code)]
pub fn generate_collinear_data(n_samples: usize) -> (Mat<f64>, Col<f64>) {
    let mut x = Mat::zeros(n_samples, 3);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        x[(i, 0)] = i as f64;
        x[(i, 1)] = 2.0 * i as f64; // Perfectly collinear with x0
        x[(i, 2)] = (i * i) as f64;
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 2)];
    }

    (x, y)
}

/// Generate data with constant columns.
#[allow(dead_code)]
pub fn generate_constant_column_data(n_samples: usize) -> (Mat<f64>, Col<f64>) {
    let mut x = Mat::zeros(n_samples, 3);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        x[(i, 0)] = i as f64;
        x[(i, 1)] = 5.0; // Constant column
        x[(i, 2)] = (i * 2) as f64;
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 2)];
    }

    (x, y)
}

/// Approximate equality check for floating point values.
#[allow(dead_code)]
pub fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

// ============================================================================
// DuckDB Scenario Data Generators
// ============================================================================

/// Generate alternating dummy variables: [[1,0],[0,1],[1,0],[0,1],...]
/// With intercept, this creates rank deficiency since col1 + col2 = 1 (intercept)
#[allow(dead_code)]
pub fn generate_alternating_dummies(n_samples: usize) -> (Mat<f64>, Col<f64>) {
    let mut x = Mat::zeros(n_samples, 2);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        if i % 2 == 0 {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = 0.0;
        } else {
            x[(i, 0)] = 0.0;
            x[(i, 1)] = 1.0;
        }
        y[i] = (i + 1) as f64;
    }

    (x, y)
}

/// Generate data with specified zero-variance columns.
/// `zero_cols` contains indices of columns that should be all zeros.
#[allow(dead_code)]
pub fn generate_zero_variance_columns(
    n_samples: usize,
    n_features: usize,
    zero_cols: &[usize],
) -> (Mat<f64>, Col<f64>) {
    let mut x = Mat::zeros(n_samples, n_features);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        let mut yi = 0.0;
        for j in 0..n_features {
            if zero_cols.contains(&j) {
                x[(i, j)] = 0.0;
            } else {
                x[(i, j)] = (i + j + 1) as f64;
                yi += x[(i, j)] * ((j + 1) as f64);
            }
        }
        y[i] = yi;
    }

    (x, y)
}

/// Generate changepoint-like production data.
/// 11 month dummies (rotating) + 11 changepoint segments (9 are all zeros).
#[allow(dead_code)]
pub fn generate_changepoint_data(n_samples: usize) -> (Mat<f64>, Col<f64>) {
    let n_features = 22; // 11 month dummies + 11 changepoint segments
    let mut x = Mat::zeros(n_samples, n_features);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        // Month dummies (rotating, indices 0-10)
        for m in 0..11 {
            x[(i, m)] = if i % 12 == (m + 1) { 1.0 } else { 0.0 };
        }

        // Changepoint segments (indices 11-21)
        // Only first 2 have variance, rest are zeros
        x[(i, 11)] = if i >= n_samples / 3 { 1.0 } else { 0.0 }; // seg_1: has variance
        x[(i, 12)] = if i >= 2 * n_samples / 3 { 1.0 } else { 0.0 }; // seg_2: has variance
        for seg in 13..22 {
            x[(i, seg)] = 0.0; // seg_3 to seg_11: all zeros
        }

        // y based on month dummies and first changepoint
        let month_effect = (i % 12) as f64;
        let changepoint_effect = if i >= n_samples / 3 { 10.0 } else { 0.0 };
        y[i] = 100.0 + month_effect * 5.0 + changepoint_effect;
    }

    (x, y)
}
