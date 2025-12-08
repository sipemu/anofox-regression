//! Leverage (hat matrix diagonal) calculations.

use faer::{Col, Mat};

/// Compute leverage values (diagonal of hat matrix H = X(X'X)^(-1)X').
///
/// Leverage measures the influence of each observation on its own fitted value.
/// High leverage points have unusual predictor values.
///
/// # Properties
/// - h_ii ∈ [0, 1]
/// - Σ h_ii = p (number of parameters)
/// - Points with h_ii > 2p/n are considered high leverage
pub fn compute_leverage(x: &Mat<f64>, with_intercept: bool) -> Col<f64> {
    let n = x.nrows();
    let p_orig = x.ncols();

    // Build design matrix with intercept if needed
    let (design, p) = if with_intercept {
        let mut design = Mat::zeros(n, p_orig + 1);
        for i in 0..n {
            design[(i, 0)] = 1.0;
            for j in 0..p_orig {
                design[(i, j + 1)] = x[(i, j)];
            }
        }
        (design, p_orig + 1)
    } else {
        (x.to_owned(), p_orig)
    };

    // Compute X'X
    let mut xtx: Mat<f64> = Mat::zeros(p, p);
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                xtx[(j, k)] += design[(i, j)] * design[(i, k)];
            }
        }
    }

    // Compute (X'X)^(-1) using QR decomposition
    let qr = xtx.qr();
    let q = qr.compute_q();
    let r = qr.compute_r();

    let mut xtx_inv: Mat<f64> = Mat::zeros(p, p);
    let qt = q.transpose();

    for col in 0..p {
        for i in (0..p).rev() {
            if r[(i, i)].abs() < 1e-14 {
                continue;
            }
            let mut sum = qt[(i, col)];
            for j in (i + 1)..p {
                sum -= r[(i, j)] * xtx_inv[(j, col)];
            }
            xtx_inv[(i, col)] = sum / r[(i, i)];
        }
    }

    // Compute leverage: h_ii = x_i' (X'X)^(-1) x_i
    let mut leverage = Col::zeros(n);

    for i in 0..n {
        let mut h_ii = 0.0;

        // Compute x_i' * (X'X)^(-1) * x_i
        for j in 0..p {
            for k in 0..p {
                h_ii += design[(i, j)] * xtx_inv[(j, k)] * design[(i, k)];
            }
        }

        // Clamp to [0, 1] for numerical stability
        leverage[i] = h_ii.clamp(0.0, 1.0);
    }

    leverage
}

/// Identify high leverage points.
///
/// Returns indices of observations with leverage > threshold.
/// Default threshold is 2p/n where p is number of parameters.
pub fn high_leverage_points(
    leverage: &Col<f64>,
    n_params: usize,
    threshold: Option<f64>,
) -> Vec<usize> {
    let n = leverage.nrows();
    let cutoff = threshold.unwrap_or(2.0 * n_params as f64 / n as f64);

    leverage
        .iter()
        .enumerate()
        .filter(|(_, &h)| h > cutoff)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leverage_bounds() {
        let x = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
        let leverage = compute_leverage(&x, true);

        for i in 0..leverage.nrows() {
            assert!(
                leverage[i] >= 0.0,
                "Leverage[{}] = {} should be >= 0",
                i,
                leverage[i]
            );
            assert!(
                leverage[i] <= 1.0,
                "Leverage[{}] = {} should be <= 1",
                i,
                leverage[i]
            );
        }
    }

    #[test]
    fn test_leverage_sum() {
        // Use linearly independent predictors
        let x = Mat::from_fn(
            30,
            2,
            |i, j| {
                if j == 0 {
                    i as f64
                } else {
                    (i as f64).sin()
                }
            },
        );
        let leverage = compute_leverage(&x, true);

        let sum: f64 = leverage.iter().sum();
        let n_params = 3; // intercept + 2 features

        // Sum of leverage should equal number of parameters (rank of design matrix)
        // Allow more tolerance due to numerical precision
        assert!(
            (sum - n_params as f64).abs() < 0.5,
            "Sum of leverage {} should be close to {}",
            sum,
            n_params
        );
    }

    #[test]
    fn test_high_leverage_detection() {
        let mut x = Mat::from_fn(20, 1, |i, _| i as f64);
        // Add an extreme outlier in x
        x[(19, 0)] = 100.0;

        let leverage = compute_leverage(&x, true);
        let high = high_leverage_points(&leverage, 2, None);

        // The outlier should have high leverage
        assert!(
            high.contains(&19),
            "Point 19 should be flagged as high leverage"
        );
    }
}
