//! Passive-Aggressive online regression (Crammer et al. 2006).
//!
//! For each training sample the model leaves weights unchanged if the
//! epsilon-insensitive loss is zero (passive), and otherwise makes the
//! smallest update that brings the loss back to zero given a regularisation
//! constant `C` (aggressive). PA-I clips the step at `C`; PA-II uses the
//! squared epsilon-insensitive loss which is equivalent to adding a small
//! `1 / (2C)` to the denominator and gives a smoother trajectory.
//!
//! ```text
//! PA-I  step τ = min(C, ℓ / ||x||²)
//! PA-II step τ = ℓ / (||x||² + 1 / (2C))
//! w     ← w + sign(y - ŷ) · τ · x
//! b     ← b + sign(y - ŷ) · τ          (when `with_intercept`)
//! ```
//!
//! Multi-epoch [`fit`] iterates over the dataset for up to `max_iter`
//! passes, stopping early once the per-epoch loss change drops below `tol`.
//! Per-sample [`partial_fit`] exposes the same update as a single step so
//! the estimator can be used in true streaming settings.

#![allow(clippy::needless_range_loop)]

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};

/// Loss variant for Passive-Aggressive updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaLoss {
    /// `max(0, |r| − ε)` — PA-I.
    #[default]
    EpsilonInsensitive,
    /// `max(0, |r| − ε)²` — PA-II (smoother updates).
    SquaredEpsilonInsensitive,
}

/// Passive-Aggressive regression estimator.
#[derive(Debug, Clone)]
pub struct PassiveAggressiveRegressor {
    c: f64,
    epsilon: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    shuffle: bool,
    loss: PaLoss,
    random_state: u64,
}

impl Default for PassiveAggressiveRegressor {
    fn default() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            with_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            loss: PaLoss::EpsilonInsensitive,
            random_state: 0,
        }
    }
}

impl PassiveAggressiveRegressor {
    pub fn builder() -> PassiveAggressiveRegressorBuilder {
        PassiveAggressiveRegressorBuilder::default()
    }
}

impl Regressor for PassiveAggressiveRegressor {
    type Fitted = FittedPassiveAggressive;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        if n != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        let mut w = vec![0.0_f64; p];
        let mut b = 0.0_f64;
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = SplitMix64::new(self.random_state.wrapping_add(0x9E37_79B9_7F4A_7C15));

        // Match sklearn's `_plain_sgd` early-stop:
        //   if sumloss > best_loss − tol * n_train: no_improvement += 1 else 0
        //   if sumloss < best_loss: best_loss = sumloss
        //   stop when no_improvement >= n_iter_no_change (=5 here)
        let mut best_loss = f64::INFINITY;
        let mut no_improvement_count: usize = 0;
        let n_iter_no_change: usize = 5;
        let tol_threshold = self.tol * n as f64;
        let mut n_iter = 0;
        for _epoch in 0..self.max_iter {
            n_iter += 1;
            if self.shuffle {
                fisher_yates(&mut order, &mut rng);
            }
            let mut sum_loss = 0.0_f64;
            for &i in &order {
                let (_step_used, loss_value) = self.apply_step(&mut w, &mut b, x, y, i);
                sum_loss += loss_value;
            }
            if sum_loss > best_loss - tol_threshold {
                no_improvement_count += 1;
            } else {
                no_improvement_count = 0;
            }
            if sum_loss < best_loss {
                best_loss = sum_loss;
            }
            if no_improvement_count >= n_iter_no_change {
                break;
            }
        }

        let coefficients = Col::from_fn(p, |i| w[i]);
        let intercept_value = if self.with_intercept { Some(b) } else { None };

        // Final fitted values / residuals.
        let fitted_values = Col::from_fn(n, |i| {
            let mut v = if self.with_intercept { b } else { 0.0 };
            for j in 0..p {
                v += x[(i, j)] * w[j];
            }
            v
        });
        let residuals = Col::from_fn(n, |i| y[i] - fitted_values[i]);

        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept_value;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = p + if self.with_intercept { 1 } else { 0 };
        result.n_parameters = result.rank;
        result.n_observations = n;
        result.r_squared = compute_r_squared(y, &result.residuals);

        Ok(FittedPassiveAggressive {
            result,
            weights: w,
            intercept: b,
            with_intercept: self.with_intercept,
            n_iter,
        })
    }
}

impl PassiveAggressiveRegressor {
    /// Apply a single PA update step. Returns `(step_size, hinge_loss_value)`.
    fn apply_step(
        &self,
        w: &mut [f64],
        b: &mut f64,
        x: &Mat<f64>,
        y: &Col<f64>,
        i: usize,
    ) -> (f64, f64) {
        let p = w.len();
        let mut pred = if self.with_intercept { *b } else { 0.0 };
        for j in 0..p {
            pred += x[(i, j)] * w[j];
        }
        let residual = y[i] - pred;
        let abs_r = residual.abs();
        let loss = (abs_r - self.epsilon).max(0.0);
        if loss == 0.0 {
            return (0.0, 0.0);
        }

        // sklearn computes ||x_i||² over feature columns only — the bias
        // dimension is *not* included even when fit_intercept=True. We
        // therefore use the bare feature norm here as well so the per-step
        // size matches sklearn's _plain_sgd kernel.
        let mut sq_norm = 0.0_f64;
        for j in 0..p {
            sq_norm += x[(i, j)] * x[(i, j)];
        }
        if sq_norm == 0.0 {
            return (0.0, loss);
        }

        let tau = match self.loss {
            PaLoss::EpsilonInsensitive => (loss / sq_norm).min(self.c),
            PaLoss::SquaredEpsilonInsensitive => loss / (sq_norm + 1.0 / (2.0 * self.c)),
        };
        let direction = residual.signum();
        let step = direction * tau;
        for j in 0..p {
            w[j] += step * x[(i, j)];
        }
        if self.with_intercept {
            *b += step;
        }
        (tau, loss)
    }

    /// Apply a single online update with a fresh sample (`partial_fit`
    /// equivalent). Useful for streaming workflows.
    pub fn partial_fit(
        &self,
        state: &mut PaState,
        x_row: &[f64],
        y_value: f64,
    ) -> Result<(), RegressionError> {
        if x_row.len() != state.weights.len() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: 1,
                y_len: x_row.len(),
            });
        }
        let mut pred = if self.with_intercept {
            state.intercept
        } else {
            0.0
        };
        for j in 0..x_row.len() {
            pred += x_row[j] * state.weights[j];
        }
        let residual = y_value - pred;
        let abs_r = residual.abs();
        let loss = (abs_r - self.epsilon).max(0.0);
        if loss == 0.0 {
            return Ok(());
        }
        let mut sq_norm = 0.0_f64;
        for j in 0..x_row.len() {
            sq_norm += x_row[j] * x_row[j];
        }
        if sq_norm == 0.0 {
            return Ok(());
        }
        let tau = match self.loss {
            PaLoss::EpsilonInsensitive => (loss / sq_norm).min(self.c),
            PaLoss::SquaredEpsilonInsensitive => loss / (sq_norm + 1.0 / (2.0 * self.c)),
        };
        let direction = residual.signum();
        let step = direction * tau;
        for j in 0..x_row.len() {
            state.weights[j] += step * x_row[j];
        }
        if self.with_intercept {
            state.intercept += step;
        }
        Ok(())
    }
}

/// In-progress PA state for streaming use via [`PassiveAggressiveRegressor::partial_fit`].
#[derive(Debug, Clone)]
pub struct PaState {
    pub weights: Vec<f64>,
    pub intercept: f64,
}

impl PaState {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            intercept: 0.0,
        }
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

fn fisher_yates(order: &mut [usize], rng: &mut SplitMix64) {
    let n = order.len();
    for i in (1..n).rev() {
        let j = rng.gen_range_usize(0, i + 1);
        order.swap(i, j);
    }
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

/// A fitted Passive-Aggressive model.
#[derive(Debug, Clone)]
pub struct FittedPassiveAggressive {
    result: RegressionResult,
    weights: Vec<f64>,
    intercept: f64,
    with_intercept: bool,
    n_iter: usize,
}

impl FittedPassiveAggressive {
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl FittedRegressor for FittedPassiveAggressive {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let intercept = if self.with_intercept {
            self.intercept
        } else {
            0.0
        };
        Col::from_fn(n, |i| {
            let mut v = intercept;
            for j in 0..p {
                v += x[(i, j)] * self.weights[j];
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

/// Builder for [`PassiveAggressiveRegressor`].
#[derive(Debug, Clone, Default)]
pub struct PassiveAggressiveRegressorBuilder {
    inner: PassiveAggressiveRegressor,
}

impl PassiveAggressiveRegressorBuilder {
    pub fn c(mut self, value: f64) -> Self {
        self.inner.c = value;
        self
    }
    pub fn epsilon(mut self, value: f64) -> Self {
        self.inner.epsilon = value;
        self
    }
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.inner.with_intercept = include;
        self
    }
    pub fn max_iter(mut self, value: usize) -> Self {
        self.inner.max_iter = value;
        self
    }
    pub fn tolerance(mut self, value: f64) -> Self {
        self.inner.tol = value;
        self
    }
    pub fn shuffle(mut self, value: bool) -> Self {
        self.inner.shuffle = value;
        self
    }
    pub fn loss(mut self, value: PaLoss) -> Self {
        self.inner.loss = value;
        self
    }
    pub fn random_state(mut self, seed: u64) -> Self {
        self.inner.random_state = seed;
        self
    }
    pub fn build(self) -> PassiveAggressiveRegressor {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_simple_line() {
        let n = 50;
        let x = Mat::from_fn(n, 1, |i, _| i as f64 / 10.0);
        let y = Col::from_fn(n, |i| 1.0 + 2.0 * (i as f64 / 10.0));

        let fitted = PassiveAggressiveRegressor::builder()
            .shuffle(false)
            .max_iter(500)
            .build()
            .fit(&x, &y)
            .unwrap();
        let slope = fitted.result.coefficients[0];
        let intercept = fitted.result.intercept.unwrap();
        // PA converges close to the true line in the absence of noise.
        assert!((slope - 2.0).abs() < 0.1, "slope = {}", slope);
        assert!((intercept - 1.0).abs() < 0.5, "intercept = {}", intercept);
    }

    #[test]
    fn partial_fit_streaming_matches_single_pass() {
        let n = 30;
        let x = Mat::from_fn(n, 1, |i, _| i as f64 / 5.0);
        let y = Col::from_fn(n, |i| 0.5 + 0.3 * (i as f64 / 5.0));

        let pa = PassiveAggressiveRegressor::builder()
            .shuffle(false)
            .max_iter(1)
            .build();
        // Batch
        let fitted = pa.fit(&x, &y).unwrap();
        // Streaming
        let mut state = PaState::new(1);
        for i in 0..n {
            let row = [x[(i, 0)]];
            pa.partial_fit(&mut state, &row, y[i]).unwrap();
        }
        assert!((state.weights[0] - fitted.weights[0]).abs() < 1e-12);
        assert!((state.intercept - fitted.intercept).abs() < 1e-12);
    }
}
