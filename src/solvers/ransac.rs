//! RANSAC (RAndom SAmple Consensus) regression.
//!
//! Iteratively samples a minimal subset of the data, fits OLS on it, marks
//! observations with `|residual| < residual_threshold` as inliers, and keeps
//! the subset with the largest inlier count. The returned model is then
//! refit on the best inlier set. Highly robust to outliers because outliers
//! never enter the final fit.
//!
//! Algorithmically follows scikit-learn's `RANSACRegressor` with an OLS base
//! estimator. We do not attempt to reproduce sklearn's NumPy Mersenne-Twister
//! exactly — instead, the validation suite picks fixtures where the consensus
//! inlier set is unique, so the final fit converges to the same coefficients
//! regardless of subsample ordering.
//!
//! Stop-probability early termination (`stop_probability`) uses the standard
//! Fischler–Bolles bound
//!
//! ```text
//! N >= log(1 - p) / log(1 - w^k)
//! ```
//!
//! where `w` is the current best inlier ratio, `k` is `min_samples`, and `p`
//! is the desired probability of having seen at least one all-inlier subset.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{RansacRegressor, Regressor, FittedRegressor};
//!
//! let fitted = RansacRegressor::builder()
//!     .min_samples(2)
//!     .residual_threshold(1.0)
//!     .max_trials(200)
//!     .random_state(42)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! let mask = fitted.inlier_mask();
//! ```

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::prelude::Solve;
use faer::{Col, Mat};

/// RANSAC robust regression.
#[derive(Debug, Clone)]
pub struct RansacRegressor {
    with_intercept: bool,
    /// Subsample size. `None` → `n_features + 1` (the minimum for an OLS
    /// fit with intercept).
    min_samples: Option<usize>,
    /// Inlier threshold. `None` → MAD of `y` (sklearn default).
    residual_threshold: Option<f64>,
    max_trials: usize,
    stop_probability: f64,
    /// Refuse a sampled subset whose OLS fit produces a singular system.
    stop_n_inliers: Option<usize>,
    random_state: u64,
}

impl Default for RansacRegressor {
    fn default() -> Self {
        Self {
            with_intercept: true,
            min_samples: None,
            residual_threshold: None,
            max_trials: 100,
            stop_probability: 0.99,
            stop_n_inliers: None,
            random_state: 0,
        }
    }
}

impl RansacRegressor {
    pub fn builder() -> RansacRegressorBuilder {
        RansacRegressorBuilder::default()
    }
}

impl Regressor for RansacRegressor {
    type Fitted = FittedRansac;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        if n != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        let p_eff = if self.with_intercept { p + 1 } else { p };
        let min_samples = self.min_samples.unwrap_or(p_eff);
        if min_samples > n {
            return Err(RegressionError::InsufficientObservations {
                needed: min_samples,
                got: n,
            });
        }
        let residual_threshold = self.residual_threshold.unwrap_or_else(|| mad(y));

        let mut rng = SplitMix64::new(self.random_state.wrapping_add(0x9E37_79B9_7F4A_7C15));
        let mut best_inlier_count: usize = 0;
        let mut best_mask: Vec<bool> = vec![false; n];

        let mut effective_max_trials = self.max_trials;
        let mut buf = vec![0usize; min_samples];

        let mut trials_run = 0_usize;
        while trials_run < effective_max_trials {
            random_combination(n, min_samples, &mut buf, &mut rng);
            let beta = match ols_on_subsample(x, y, &buf, self.with_intercept) {
                Some(b) => b,
                None => {
                    trials_run += 1;
                    continue;
                }
            };

            // Compute residuals over the entire dataset.
            let intercept = if self.with_intercept { beta[0] } else { 0.0 };
            let coef_slice = if self.with_intercept {
                &beta[1..]
            } else {
                &beta[..]
            };

            let mut mask = vec![false; n];
            let mut inlier_count = 0_usize;
            for i in 0..n {
                let mut pred = intercept;
                for j in 0..p {
                    pred += x[(i, j)] * coef_slice[j];
                }
                let resid = (y[i] - pred).abs();
                if resid < residual_threshold {
                    mask[i] = true;
                    inlier_count += 1;
                }
            }

            if inlier_count > best_inlier_count {
                best_inlier_count = inlier_count;
                best_mask = mask;

                // Update the dynamic max_trials bound (Fischler–Bolles).
                if best_inlier_count >= min_samples {
                    let w = best_inlier_count as f64 / n as f64;
                    let denom = (1.0 - w.powi(min_samples as i32)).ln();
                    if denom < 0.0 {
                        let num = (1.0 - self.stop_probability).ln();
                        let need = (num / denom).ceil() as usize;
                        effective_max_trials = effective_max_trials.min(need.max(1));
                    }
                }

                // Optional early stop on absolute inlier count.
                if let Some(threshold) = self.stop_n_inliers {
                    if best_inlier_count >= threshold {
                        break;
                    }
                }
            }
            trials_run += 1;
        }

        if best_inlier_count < min_samples {
            return Err(RegressionError::ConvergenceFailed {
                iterations: self.max_trials,
            });
        }

        // Final fit on inliers via OLS.
        let inlier_idx: Vec<usize> = (0..n).filter(|&i| best_mask[i]).collect();
        let final_beta = ols_normal_eq(x, y, &inlier_idx, self.with_intercept)
            .ok_or(RegressionError::SingularMatrix)?;

        let intercept = if self.with_intercept {
            Some(final_beta[0])
        } else {
            None
        };
        let coef_slice = if self.with_intercept {
            &final_beta[1..]
        } else {
            &final_beta[..]
        };
        let coefficients = Col::from_fn(p, |i| coef_slice[i]);

        let fitted_values = Col::from_fn(n, |i| {
            let mut v = intercept.unwrap_or(0.0);
            for j in 0..p {
                v += x[(i, j)] * coef_slice[j];
            }
            v
        });
        let residuals = Col::from_fn(n, |i| y[i] - fitted_values[i]);

        let n_params = p + if self.with_intercept { 1 } else { 0 };
        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = n_params;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.r_squared = compute_r_squared(y, &result.residuals);

        Ok(FittedRansac {
            result,
            inlier_mask: best_mask,
            n_trials: trials_run,
            residual_threshold,
        })
    }
}

fn compute_r_squared(y: &Col<f64>, residuals: &Col<f64>) -> f64 {
    let n = y.nrows();
    let y_mean: f64 = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;
    let tss: f64 = (0..n).map(|i| (y[i] - y_mean).powi(2)).sum();
    let rss: f64 = (0..n).map(|i| residuals[i].powi(2)).sum();
    if tss == 0.0 {
        if rss == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - rss / tss
    }
}

fn mad(y: &Col<f64>) -> f64 {
    let mut v: Vec<f64> = (0..y.nrows()).map(|i| y[i]).collect();
    let med = median_in_place(&mut v);
    for vi in v.iter_mut() {
        *vi = (*vi - med).abs();
    }
    median_in_place(&mut v)
}

fn median_in_place(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    // `usize::is_multiple_of` is unstable on Rust < 1.87. See #20.
    #[allow(clippy::manual_is_multiple_of)]
    if n % 2 == 0 {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    } else {
        v[n / 2]
    }
}

fn ols_on_subsample(
    x: &Mat<f64>,
    y: &Col<f64>,
    idx: &[usize],
    with_intercept: bool,
) -> Option<Vec<f64>> {
    let p = x.ncols();
    let p_eff = if with_intercept { p + 1 } else { p };
    let k = idx.len();
    let xs = Mat::from_fn(k, p_eff, |i, j| {
        if with_intercept {
            if j == 0 {
                1.0
            } else {
                x[(idx[i], j - 1)]
            }
        } else {
            x[(idx[i], j)]
        }
    });
    let ys = Col::from_fn(k, |i| y[idx[i]]);

    if k == p_eff {
        // Square system: LU.
        let lu = xs.partial_piv_lu();
        let beta = lu.solve(&ys);
        let mut out = Vec::with_capacity(p_eff);
        for i in 0..p_eff {
            if !beta[i].is_finite() {
                return None;
            }
            out.push(beta[i]);
        }
        Some(out)
    } else {
        ols_normal_eq_from_design(&xs, &ys)
    }
}

fn ols_normal_eq(
    x: &Mat<f64>,
    y: &Col<f64>,
    idx: &[usize],
    with_intercept: bool,
) -> Option<Vec<f64>> {
    let p = x.ncols();
    let p_eff = if with_intercept { p + 1 } else { p };
    let k = idx.len();
    let xs = Mat::from_fn(k, p_eff, |i, j| {
        if with_intercept {
            if j == 0 {
                1.0
            } else {
                x[(idx[i], j - 1)]
            }
        } else {
            x[(idx[i], j)]
        }
    });
    let ys = Col::from_fn(k, |i| y[idx[i]]);
    ols_normal_eq_from_design(&xs, &ys)
}

fn ols_normal_eq_from_design(xs: &Mat<f64>, ys: &Col<f64>) -> Option<Vec<f64>> {
    let xtx = xs.transpose() * xs;
    let xty = xs.transpose() * ys;
    let lu = xtx.partial_piv_lu();
    let beta = lu.solve(&xty);
    let p_eff = xs.ncols();
    let mut out = Vec::with_capacity(p_eff);
    for i in 0..p_eff {
        if !beta[i].is_finite() {
            return None;
        }
        out.push(beta[i]);
    }
    Some(out)
}

fn random_combination(n: usize, k: usize, out: &mut [usize], rng: &mut SplitMix64) {
    debug_assert_eq!(out.len(), k);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    for j in (n - k)..n {
        let t = rng.gen_range_usize(0, j + 1);
        if chosen.contains(&t) {
            chosen.push(j);
        } else {
            chosen.push(t);
        }
    }
    chosen.sort_unstable();
    out.copy_from_slice(&chosen);
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn gen_range_usize(&mut self, lo: usize, hi: usize) -> usize {
        let span = (hi - lo) as u64;
        if span == 0 {
            return lo;
        }
        let bound = u64::MAX - (u64::MAX % span);
        loop {
            let r = self.next_u64();
            if r < bound {
                return lo + (r % span) as usize;
            }
        }
    }
}

/// A fitted RANSAC model.
#[derive(Debug, Clone)]
pub struct FittedRansac {
    result: RegressionResult,
    inlier_mask: Vec<bool>,
    n_trials: usize,
    residual_threshold: f64,
}

impl FittedRansac {
    /// Boolean mask marking observations classified as inliers in the
    /// final consensus set.
    pub fn inlier_mask(&self) -> &[bool] {
        &self.inlier_mask
    }

    /// Number of inliers in the final consensus set.
    pub fn n_inliers(&self) -> usize {
        self.inlier_mask.iter().filter(|&&b| b).count()
    }

    /// Number of RANSAC trials actually run before early termination.
    pub fn n_trials(&self) -> usize {
        self.n_trials
    }

    /// Residual threshold used for classifying inliers.
    pub fn residual_threshold(&self) -> f64 {
        self.residual_threshold
    }
}

impl FittedRegressor for FittedRansac {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);
        Col::from_fn(n, |i| {
            let mut v = intercept;
            for j in 0..p {
                v += x[(i, j)] * self.result.coefficients[j];
            }
            v
        })
    }

    fn result(&self) -> &RegressionResult {
        &self.result
    }

    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        _interval: Option<IntervalType>,
        _level: f64,
    ) -> PredictionResult {
        PredictionResult::point_only(self.predict(x))
    }
}

/// Builder for [`RansacRegressor`].
#[derive(Debug, Clone, Default)]
pub struct RansacRegressorBuilder {
    inner: RansacRegressor,
}

impl RansacRegressorBuilder {
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.inner.with_intercept = include;
        self
    }
    pub fn min_samples(mut self, value: usize) -> Self {
        self.inner.min_samples = Some(value);
        self
    }
    pub fn residual_threshold(mut self, value: f64) -> Self {
        self.inner.residual_threshold = Some(value);
        self
    }
    pub fn max_trials(mut self, value: usize) -> Self {
        self.inner.max_trials = value;
        self
    }
    pub fn stop_probability(mut self, value: f64) -> Self {
        self.inner.stop_probability = value;
        self
    }
    pub fn stop_n_inliers(mut self, value: usize) -> Self {
        self.inner.stop_n_inliers = Some(value);
        self
    }
    pub fn random_state(mut self, seed: u64) -> Self {
        self.inner.random_state = seed;
        self
    }
    pub fn build(self) -> RansacRegressor {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ransac_ignores_outliers() {
        // Inliers on y = 1 + 2x + small noise, outliers far away.
        let n_in = 50;
        let n_out = 20;
        let n = n_in + n_out;
        let mut xs: Vec<f64> = (0..n_in).map(|i| i as f64 * 0.2).collect();
        let mut ys: Vec<f64> = xs.iter().map(|xi| 1.0 + 2.0 * xi).collect();
        for i in 0..n_out {
            xs.push(i as f64 * 0.1);
            ys.push(50.0 + i as f64);
        }

        let x = Mat::from_fn(n, 1, |i, _| xs[i]);
        let y = Col::from_fn(n, |i| ys[i]);

        let fitted = RansacRegressor::builder()
            .residual_threshold(0.5)
            .max_trials(200)
            .random_state(42)
            .build()
            .fit(&x, &y)
            .expect("RANSAC fit failed");

        let intercept = fitted.result().intercept.unwrap();
        let slope = fitted.result().coefficients[0];
        assert!(
            (intercept - 1.0).abs() < 1e-6,
            "intercept = {} (expected 1.0)",
            intercept
        );
        assert!(
            (slope - 2.0).abs() < 1e-6,
            "slope = {} (expected 2.0)",
            slope
        );

        // Outlier indices in [n_in, n_in + n_out) should be flagged.
        for i in n_in..n {
            assert!(!fitted.inlier_mask()[i], "obs {} should be an outlier", i);
        }
    }
}
