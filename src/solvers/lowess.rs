//! LOWESS (Locally Weighted Scatterplot Smoothing) implementation.
//!
//! [`lowess`] is Cleveland's (1979, 1981) robust locally weighted regression
//! scatterplot smoother, a port of R's `stats::lowess` (the `clowess` routine):
//! a local linear fit with tricube weights over the `frac · n` nearest
//! neighbours of each point, `iterations` robustifying passes with bisquare
//! weights on the residuals (scaled by six times their median absolute value),
//! and linear interpolation between points closer than `delta`.
//!
//! [`lowess_smooth_weights`] / [`lowess_smooth`] are the simpler index-based
//! smoothers lmDynamic uses for its model weights over time.

use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};

/// Options for [`lowess`], with the defaults of R's `stats::lowess`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowessOptions {
    /// Smoother span: the fraction of points that influence each fitted
    /// value (R's `f`, default 2/3). Larger values give more smoothness.
    pub frac: f64,
    /// Number of robustifying iterations (R's `iter`, default 3); 0 gives
    /// the plain, non-robust fit.
    pub iterations: usize,
    /// Points within `delta` of the last fitted point are interpolated
    /// instead of fitted (R's `delta`); `None` uses R's default
    /// `0.01 · (max(x) − min(x))`.
    pub delta: Option<f64>,
}

impl Default for LowessOptions {
    fn default() -> Self {
        Self {
            frac: 2.0 / 3.0,
            iterations: 3,
            delta: None,
        }
    }
}

/// Result of [`lowess`]: the smoothed curve at the sorted `x` values, as the
/// `list(x, y)` R's `stats::lowess` returns.
#[derive(Debug, Clone, PartialEq)]
pub struct LowessFit {
    /// The input `x` values in increasing order (ties keep their input order).
    pub x: Vec<f64>,
    /// The smoothed value at each `x`.
    pub fitted: Vec<f64>,
    /// `order[k]` is the input position of the `k`-th sorted point, so
    /// `fitted[k]` is the smoothed value of input row `order[k]`.
    pub order: Vec<usize>,
}

/// Robust locally weighted scatterplot smoothing of `y` on `x`, matching R's
/// `stats::lowess(x, y, f = frac, iter = iterations, delta = delta)`.
///
/// # Errors
/// `DimensionMismatch` when `x` and `y` differ in length,
/// `InsufficientObservations` for empty input, and `NumericalError` for
/// non-finite values or invalid options (`frac` must be finite and positive,
/// `delta` finite and non-negative).
pub fn lowess(x: &[f64], y: &[f64], options: &LowessOptions) -> Result<LowessFit, RegressionError> {
    let n = x.len();
    if y.len() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: y.len(),
        });
    }
    if n == 0 {
        return Err(RegressionError::InsufficientObservations { needed: 1, got: 0 });
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(RegressionError::NumericalError(
            "lowess: x and y must be finite".to_string(),
        ));
    }
    if !(options.frac.is_finite() && options.frac > 0.0) {
        return Err(RegressionError::NumericalError(
            "lowess: frac must be finite and > 0".to_string(),
        ));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();

    let delta = options.delta.unwrap_or_else(|| 0.01 * (xs[n - 1] - xs[0]));
    if !(delta.is_finite() && delta >= 0.0) {
        return Err(RegressionError::NumericalError(
            "lowess: delta must be finite and >= 0".to_string(),
        ));
    }

    let fitted = clowess(&xs, &ys, options.frac, options.iterations, delta);
    Ok(LowessFit {
        x: xs,
        fitted,
        order,
    })
}

fn fcube(x: f64) -> f64 {
    x * x * x
}

/// Fitted value at `xs` from the points `nleft..=nright` (1-based; ties to
/// the right of `nright` are picked up), R's `lowest`. `None` when every
/// weight is zero.
#[allow(clippy::too_many_arguments)]
fn lowest(
    x: &[f64],
    y: &[f64],
    n: usize,
    xs: f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
    rw: Option<&[f64]>,
) -> Option<f64> {
    let range = x[n] - x[1];
    let h = (xs - x[nleft]).max(x[nright] - xs);
    let h9 = 0.999 * h;
    let h1 = 0.001 * h;

    // Tricube weights, picking up all ties on the right.
    let mut a = 0.0;
    let mut j = nleft;
    while j <= n {
        w[j] = 0.0;
        let r = (x[j] - xs).abs();
        if r <= h9 {
            w[j] = if r <= h1 {
                1.0
            } else {
                fcube(1.0 - fcube(r / h))
            };
            if let Some(rw) = rw {
                w[j] *= rw[j];
            }
            a += w[j];
        } else if x[j] > xs {
            break;
        }
        j += 1;
    }
    let nrt = j - 1;
    if a <= 0.0 {
        return None;
    }

    // Weighted least squares: normalise the weights, then fold in the local
    // slope when the points are spread out enough to estimate it.
    for w_j in &mut w[nleft..=nrt] {
        *w_j /= a;
    }
    if h > 0.0 {
        let center: f64 = (nleft..=nrt).map(|j| w[j] * x[j]).sum();
        let b = xs - center;
        let c: f64 = (nleft..=nrt)
            .map(|j| w[j] * (x[j] - center) * (x[j] - center))
            .sum();
        if c.sqrt() > 0.001 * range {
            let b = b / c;
            for j in nleft..=nrt {
                w[j] *= b * (x[j] - center) + 1.0;
            }
        }
    }
    Some((nleft..=nrt).map(|j| w[j] * y[j]).sum())
}

/// R's `clowess` on `x` sorted increasingly; returns the fitted values.
fn clowess(x0: &[f64], y0: &[f64], frac: f64, nsteps: usize, delta: f64) -> Vec<f64> {
    let n = x0.len();
    if n < 2 {
        return vec![y0[0]];
    }
    // 1-based copies, as in the C code.
    let mut x = vec![0.0; n + 1];
    let mut y = vec![0.0; n + 1];
    x[1..].copy_from_slice(x0);
    y[1..].copy_from_slice(y0);
    let mut ys = vec![0.0; n + 1];
    let mut w = vec![0.0; n + 1];
    let mut rw = vec![1.0; n + 1];
    let mut res = vec![0.0; n];

    // At least two, at most n points.
    let ns = ((frac * n as f64 + 1e-7) as usize).clamp(2, n);

    let mut iter = 1;
    while iter <= nsteps + 1 {
        let mut nleft = 1;
        let mut nright = ns;
        let mut last = 0; // index of the previously estimated point
        let mut i = 1; // index of the current point
        loop {
            if nright < n {
                // Move the window right while that shrinks its radius.
                let d1 = x[i] - x[nleft];
                let d2 = x[nright + 1] - x[i];
                if d1 > d2 {
                    nleft += 1;
                    nright += 1;
                    continue;
                }
            }

            let robust = (iter > 1).then_some(rw.as_slice());
            ys[i] = lowest(&x, &y, n, x[i], nleft, nright, &mut w, robust).unwrap_or(y[i]);

            // Interpolate the points skipped since the last estimate.
            if last + 1 < i {
                let denom = x[i] - x[last];
                for j in last + 1..i {
                    let alpha = (x[j] - x[last]) / denom;
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last];
                }
            }
            last = i;

            // Skip ahead over points within delta; ties share the fit.
            let cut = x[last] + delta;
            i = last + 1;
            while i <= n {
                if x[i] > cut {
                    break;
                }
                if x[i] == x[last] {
                    ys[i] = ys[last];
                    last = i;
                }
                i += 1;
            }
            i = (last + 1).max(i - 1);
            if last >= n {
                break;
            }
        }

        for i in 0..n {
            res[i] = y[i + 1] - ys[i + 1];
        }
        let sc = res.iter().map(|r| r.abs()).sum::<f64>() / n as f64;
        if iter > nsteps {
            break;
        }

        // Bisquare robustness weights from 6 · median |residual|.
        let mut abs_res: Vec<f64> = res.iter().map(|r| r.abs()).collect();
        abs_res.sort_by(f64::total_cmp);
        let m1 = n / 2;
        // `n % 2 == 0` rather than `usize::is_multiple_of`, which needs
        // Rust >= 1.87 (the DuckDB wasm toolchain is older). See #20.
        #[allow(clippy::manual_is_multiple_of)]
        let cmad = if n % 2 == 0 {
            3.0 * (abs_res[m1] + abs_res[n - m1 - 1])
        } else {
            6.0 * abs_res[m1]
        };
        if cmad < 1e-7 * sc {
            break;
        }
        let c9 = 0.999 * cmad;
        let c1 = 0.001 * cmad;
        for i in 0..n {
            let r = res[i].abs();
            rw[i + 1] = if r <= c1 {
                1.0
            } else if r <= c9 {
                let u = r / cmad;
                (1.0 - u * u) * (1.0 - u * u)
            } else {
                0.0
            };
        }
        iter += 1;
    }
    ys[1..].to_vec()
}

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
