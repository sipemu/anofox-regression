//! Mergeable, serializable sufficient statistics for OLS / WLS / ridge.
//!
//! A [`SufficientStats`] holds everything a (weighted) least-squares or ridge
//! fit needs, in `O(p²)` memory regardless of the number of rows:
//!
//! ```text
//! n_rows, n_positive        row counts (all rows / rows with w > 0)
//! W    = Σ wᵢ               total weight
//! Σ ln wᵢ  (wᵢ > 0)         for the weighted Gaussian log-likelihood
//! x̄    = Σ wᵢ xᵢ / W        weighted means          (length p)
//! ȳ    = Σ wᵢ yᵢ / W
//! Cxx  = Σ wᵢ (xᵢ − x̄)(xᵢ − x̄)ᵀ   centred co-moments  (p × p)
//! Cxy  = Σ wᵢ (xᵢ − x̄)(yᵢ − ȳ)                        (length p)
//! Cyy  = Σ wᵢ (yᵢ − ȳ)²
//! ```
//!
//! Rows are added with a weighted Welford update and states are combined with
//! the pairwise formula of Chan, Golub & LeVeque, so a merge never forms raw
//! cross-products: data with a large offset (`x ≈ 1e9 + noise`) keeps the
//! precision of the batch fit. States merge in any order and grouping, an
//! empty state is the identity, and [`SufficientStats::to_bytes`] gives a
//! stable, versioned encoding for storing or shipping partial states.
//!
//! # What a state can give
//!
//! [`OlsRegressor::fit_from_state`](crate::solvers::OlsRegressor::fit_from_state),
//! [`WlsRegressor::fit_from_state`](crate::solvers::WlsRegressor::fit_from_state) and
//! [`RidgeRegressor::fit_from_state`](crate::solvers::RidgeRegressor::fit_from_state)
//! return the same result as `fit()` on the concatenated rows (up to
//! floating-point summation order): coefficients, R-`lm`-style aliasing,
//! classical standard errors, t-statistics, p-values and confidence
//! intervals, R², adjusted R², F, MSE, log-likelihood, AIC / AICc / BIC and
//! confidence / prediction intervals for new rows. For ridge, λ is applied at
//! finalize, so one state serves any λ grid.
//!
//! # What a state cannot give
//!
//! Anything defined per row needs the rows: residuals, fitted values,
//! leverage / hat values, LOO and Cook's distance, and the HC0–HC3 sandwich
//! covariance (`Σ eᵢ² xᵢxᵢᵀ` depends on the residuals, which are only known
//! once the coefficients are, so it is not mergeable in one pass). A fit from
//! a state returns empty `residuals` / `fitted_values`, and the HC methods
//! return an error instead of an approximation.
//!
//! # GLMs
//!
//! IRLS needs a pass over the rows per iteration, so no exact single-pass
//! state exists for a GLM. A state still makes each IRLS step distributable:
//! for the current `β` every worker pushes its rows as
//! `(xᵢ, zᵢ, wᵢ)` with the working response `zᵢ = ηᵢ + (yᵢ − μᵢ) / μ'(ηᵢ)` and
//! working weight `wᵢ = pᵢ μ'(ηᵢ)² / V(μᵢ)` (prior weight `pᵢ`); the merged
//! state is the score / Fisher-information summary of that step, and
//! [`WlsRegressor::fit_from_state`](crate::solvers::WlsRegressor::fit_from_state)
//! gives the updated `β`.
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::solvers::{FittedRegressor, OlsRegressor, SufficientStats};
//!
//! let mut a = SufficientStats::new(1);
//! let mut b = SufficientStats::new(1);
//! for i in 0..10 {
//!     let x = i as f64;
//!     let y = 1.0 + 2.0 * x + if i % 2 == 0 { 0.1 } else { -0.1 };
//!     if i < 5 { a.push(&[x], y).unwrap() } else { b.push(&[x], y).unwrap() }
//! }
//! // Ship `b` as bytes, restore and merge.
//! let b = SufficientStats::from_bytes(&b.to_bytes()).unwrap();
//! a.merge(&b).unwrap();
//!
//! let fit = OlsRegressor::builder().build().fit_from_state(&a).unwrap();
//! assert!((fit.result().coefficients[0] - 2.0).abs() < 0.05);
//! assert_eq!(fit.result().n_observations, 10);
//! ```

#![allow(clippy::needless_range_loop)]

use crate::core::{RegressionOptions, RegressionResult};
use crate::inference::CoefficientInference;
use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Magic bytes of the serialized state.
const MAGIC: &[u8; 4] = b"AXSS";
/// Current serialization version.
const VERSION: u16 = 1;
/// Header: magic, version, flags, then p, n_rows, n_positive and five scalars.
const HEADER_LEN: usize = 4 + 2 + 2 + 3 * 8 + 5 * 8;
/// Flag bit: at least one row had a weight other than 1.
const FLAG_WEIGHTED: u16 = 1;

/// Mergeable sufficient statistics of a (weighted) linear model.
///
/// See the [module documentation](self) for the contents, the merge rule and
/// what can and cannot be computed from a state.
#[derive(Debug, Clone, PartialEq)]
pub struct SufficientStats {
    n_features: usize,
    n_rows: u64,
    n_positive: u64,
    weighted: bool,
    sum_w: f64,
    sum_log_w: f64,
    /// Shift `K` (the first row with positive weight); the means are stored
    /// relative to it, so the updates stay accurate for data with a large
    /// offset (`|x̄| ≫ sd(x)`).
    shift_x: Vec<f64>,
    shift_y: f64,
    /// `x̄ − K`.
    mean_x: Vec<f64>,
    /// `ȳ − K_y`.
    mean_y: f64,
    /// Row-major `p × p`, symmetric.
    cxx: Vec<f64>,
    cxy: Vec<f64>,
    cyy: f64,
}

impl SufficientStats {
    /// Empty state for `n_features` predictors (no intercept column).
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            n_rows: 0,
            n_positive: 0,
            weighted: false,
            sum_w: 0.0,
            sum_log_w: 0.0,
            shift_x: vec![0.0; n_features],
            shift_y: 0.0,
            mean_x: vec![0.0; n_features],
            mean_y: 0.0,
            cxx: vec![0.0; n_features * n_features],
            cxy: vec![0.0; n_features],
            cyy: 0.0,
        }
    }

    /// State of a block of rows, computed in two passes (means first, then
    /// centred co-moments) for best accuracy. `weights = None` means unit
    /// weights.
    pub fn from_data(
        x: &Mat<f64>,
        y: &Col<f64>,
        weights: Option<&Col<f64>>,
    ) -> Result<Self, RegressionError> {
        let (n, p) = (x.nrows(), x.ncols());
        if y.nrows() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        if let Some(w) = weights {
            if w.nrows() != n {
                return Err(RegressionError::DimensionMismatch {
                    x_rows: n,
                    y_len: w.nrows(),
                });
            }
        }
        let w = |i: usize| weights.map_or(1.0, |w| w[i]);
        let mut s = Self::new(p);
        for i in 0..n {
            check_row((0..p).map(|j| x[(i, j)]), y[i], w(i))?;
            let wi = w(i);
            s.n_rows += 1;
            s.weighted |= wi != 1.0;
            if wi > 0.0 {
                s.n_positive += 1;
                s.sum_w += wi;
                s.sum_log_w += wi.ln();
            }
        }
        let Some(first) = (0..n).find(|&i| w(i) > 0.0) else {
            return Ok(s);
        };
        s.shift_y = y[first];
        for j in 0..p {
            s.shift_x[j] = x[(first, j)];
        }
        for i in 0..n {
            let wi = w(i);
            s.mean_y += wi * (y[i] - s.shift_y);
            for j in 0..p {
                s.mean_x[j] += wi * (x[(i, j)] - s.shift_x[j]);
            }
        }
        s.mean_y /= s.sum_w;
        for j in 0..p {
            s.mean_x[j] /= s.sum_w;
        }
        let mut dx = vec![0.0; p];
        for i in 0..n {
            let wi = w(i);
            if wi == 0.0 {
                continue;
            }
            let dy = (y[i] - s.shift_y) - s.mean_y;
            for j in 0..p {
                dx[j] = (x[(i, j)] - s.shift_x[j]) - s.mean_x[j];
            }
            s.cyy += wi * dy * dy;
            for j in 0..p {
                s.cxy[j] += wi * dx[j] * dy;
                for k in j..p {
                    s.cxx[j * p + k] += wi * dx[j] * dx[k];
                }
            }
        }
        s.symmetrize();
        Ok(s)
    }

    /// Add one row with unit weight.
    pub fn push(&mut self, x_row: &[f64], y: f64) -> Result<(), RegressionError> {
        self.push_weighted(x_row, y, 1.0)
    }

    /// Add one row with weight `w ≥ 0`. A zero-weight row only counts towards
    /// [`Self::n_observations`], as in `lm(weights =)`.
    pub fn push_weighted(&mut self, x_row: &[f64], y: f64, w: f64) -> Result<(), RegressionError> {
        if x_row.len() != self.n_features {
            return Err(RegressionError::DimensionMismatch {
                x_rows: self.n_features,
                y_len: x_row.len(),
            });
        }
        check_row(x_row.iter().copied(), y, w)?;
        self.n_rows += 1;
        self.weighted |= w != 1.0;
        if w == 0.0 {
            return Ok(());
        }
        self.n_positive += 1;
        self.sum_log_w += w.ln();
        // Weighted Welford (West, 1979): with d = x − x̄_old and W' = W + w,
        // x̄ += d·w/W' and C += d dᵀ · w·W/W'.
        let w_old = self.sum_w;
        if w_old == 0.0 {
            self.shift_x.copy_from_slice(x_row);
            self.shift_y = y;
        }
        self.sum_w += w;
        let f = w / self.sum_w;
        let g = w * w_old / self.sum_w;
        let p = self.n_features;
        let dy = (y - self.shift_y) - self.mean_y;
        let dx: Vec<f64> = (0..p)
            .map(|j| (x_row[j] - self.shift_x[j]) - self.mean_x[j])
            .collect();
        self.mean_y += f * dy;
        self.cyy += g * dy * dy;
        for j in 0..p {
            self.mean_x[j] += f * dx[j];
            self.cxy[j] += g * dx[j] * dy;
            for k in j..p {
                let v = self.cxx[j * p + k] + g * dx[j] * dx[k];
                self.cxx[j * p + k] = v;
                self.cxx[k * p + j] = v;
            }
        }
        Ok(())
    }

    /// Merge another state into this one (Chan–Golub–LeVeque pairwise
    /// update). The result fits exactly like a state of the concatenated
    /// rows, up to floating-point summation order.
    pub fn merge(&mut self, other: &Self) -> Result<(), RegressionError> {
        if other.n_features != self.n_features {
            return Err(RegressionError::DimensionMismatch {
                x_rows: self.n_features,
                y_len: other.n_features,
            });
        }
        self.n_rows += other.n_rows;
        self.n_positive += other.n_positive;
        self.weighted |= other.weighted;
        self.sum_log_w += other.sum_log_w;
        if other.sum_w == 0.0 {
            return Ok(());
        }
        if self.sum_w == 0.0 {
            self.sum_w = other.sum_w;
            self.shift_x.clone_from(&other.shift_x);
            self.shift_y = other.shift_y;
            self.mean_x.clone_from(&other.mean_x);
            self.mean_y = other.mean_y;
            self.cxx.clone_from(&other.cxx);
            self.cxy.clone_from(&other.cxy);
            self.cyy = other.cyy;
            return Ok(());
        }
        let (wa, wb) = (self.sum_w, other.sum_w);
        let w = wa + wb;
        let f = wb / w;
        let g = wa * wb / w;
        let p = self.n_features;
        // Difference of the means, with `other`'s shift rebased onto ours
        // (nearby shifts subtract exactly).
        let dy = ((other.shift_y - self.shift_y) + other.mean_y) - self.mean_y;
        let dx: Vec<f64> = (0..p)
            .map(|j| ((other.shift_x[j] - self.shift_x[j]) + other.mean_x[j]) - self.mean_x[j])
            .collect();
        self.sum_w = w;
        self.mean_y += f * dy;
        self.cyy += other.cyy + g * dy * dy;
        for j in 0..p {
            self.mean_x[j] += f * dx[j];
            self.cxy[j] += other.cxy[j] + g * dx[j] * dy;
            for k in j..p {
                let v = self.cxx[j * p + k] + other.cxx[j * p + k] + g * dx[j] * dx[k];
                self.cxx[j * p + k] = v;
                self.cxx[k * p + j] = v;
            }
        }
        Ok(())
    }

    /// Merge of `self` and `other` as a new state.
    pub fn merged(&self, other: &Self) -> Result<Self, RegressionError> {
        let mut out = self.clone();
        out.merge(other)?;
        Ok(out)
    }

    /// Reset to the empty state, keeping the feature count.
    pub fn clear(&mut self) {
        *self = Self::new(self.n_features);
    }

    /// Number of predictors (excluding the intercept).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of rows added, including rows with zero weight (a fit reports
    /// only the [positive-weight rows](Self::n_positive) as
    /// `RegressionResult::n_observations`, as `lm`).
    pub fn n_observations(&self) -> usize {
        self.n_rows as usize
    }

    /// Number of rows with a positive weight.
    pub fn n_positive(&self) -> usize {
        self.n_positive as usize
    }

    /// `true` if no row has been added.
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// `true` if any row had a weight other than 1.
    pub fn is_weighted(&self) -> bool {
        self.weighted
    }

    /// Total weight `Σ wᵢ`.
    pub fn sum_weights(&self) -> f64 {
        self.sum_w
    }

    /// `Σ ln wᵢ` over rows with `wᵢ > 0`.
    pub fn sum_log_weights(&self) -> f64 {
        self.sum_log_w
    }

    /// Weighted means of the predictors.
    pub fn mean_x(&self) -> Vec<f64> {
        (0..self.n_features)
            .map(|j| self.shift_x[j] + self.mean_x[j])
            .collect()
    }

    /// Weighted mean of the response.
    pub fn mean_y(&self) -> f64 {
        self.shift_y + self.mean_y
    }

    /// Centred co-moment matrix `Σ wᵢ (xᵢ − x̄)(xᵢ − x̄)ᵀ`.
    pub fn centered_xx(&self) -> Mat<f64> {
        let p = self.n_features;
        Mat::from_fn(p, p, |j, k| self.cxx[j * p + k])
    }

    /// Centred co-moments `Σ wᵢ (xᵢ − x̄)(yᵢ − ȳ)`.
    pub fn centered_xy(&self) -> &[f64] {
        &self.cxy
    }

    /// Centred response sum of squares `Σ wᵢ (yᵢ − ȳ)²`.
    pub fn centered_yy(&self) -> f64 {
        self.cyy
    }

    /// Stable little-endian encoding (format version 1):
    ///
    /// ```text
    /// "AXSS" | u16 version | u16 flags | u64 p | u64 n_rows | u64 n_positive
    /// | f64 Σw | f64 Σ ln w | f64 K_y | f64 ȳ − K_y | f64 Cyy
    /// | f64 K[p] | f64 x̄ − K [p] | f64 Cxy[p]
    /// | f64 Cxx upper triangle, row-major [p(p+1)/2]
    /// ```
    ///
    /// `K` is the shift the means are stored against (the state's first row
    /// with positive weight).
    pub fn to_bytes(&self) -> Vec<u8> {
        let p = self.n_features;
        let mut out = Vec::with_capacity(HEADER_LEN + 8 * (3 * p + p * (p + 1) / 2));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        let flags = if self.weighted { FLAG_WEIGHTED } else { 0 };
        out.extend_from_slice(&flags.to_le_bytes());
        for v in [p as u64, self.n_rows, self.n_positive] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in [
            self.sum_w,
            self.sum_log_w,
            self.shift_y,
            self.mean_y,
            self.cyy,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.shift_x.iter().chain(&self.mean_x).chain(&self.cxy) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for j in 0..p {
            for k in j..p {
                out.extend_from_slice(&self.cxx[j * p + k].to_le_bytes());
            }
        }
        out
    }

    /// Decode a state written by [`Self::to_bytes`]. Rejects a wrong magic,
    /// an unknown version, a length that does not match the feature count,
    /// and inconsistent contents.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RegressionError> {
        let bad = |msg: &str| RegressionError::NumericalError(format!("invalid state: {msg}"));
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            return Err(bad("not a SufficientStats encoding"));
        }
        let mut r = Reader { bytes, pos: 4 };
        let version = r.u16();
        if version != VERSION {
            return Err(bad(&format!("unsupported version {version}")));
        }
        let flags = r.u16();
        if flags & !FLAG_WEIGHTED != 0 {
            return Err(bad("unknown flags"));
        }
        let p = r.u64();
        let expected = usize::try_from(p).ok().and_then(|p| {
            let tri = p.checked_add(1)?.checked_mul(p)? / 2;
            let floats = tri.checked_add(p.checked_mul(3)?)?;
            floats.checked_mul(8)?.checked_add(HEADER_LEN)
        });
        if expected != Some(bytes.len()) {
            return Err(bad("length does not match the feature count"));
        }
        let p = p as usize;
        let mut s = Self::new(p);
        s.weighted = flags & FLAG_WEIGHTED != 0;
        s.n_rows = r.u64();
        s.n_positive = r.u64();
        s.sum_w = r.f64();
        s.sum_log_w = r.f64();
        s.shift_y = r.f64();
        s.mean_y = r.f64();
        s.cyy = r.f64();
        for j in 0..p {
            s.shift_x[j] = r.f64();
        }
        for j in 0..p {
            s.mean_x[j] = r.f64();
        }
        for j in 0..p {
            s.cxy[j] = r.f64();
        }
        for j in 0..p {
            for k in j..p {
                s.cxx[j * p + k] = r.f64();
            }
        }
        s.symmetrize();
        let finite = [s.sum_w, s.sum_log_w, s.shift_y, s.mean_y, s.cyy]
            .iter()
            .chain(&s.shift_x)
            .chain(&s.mean_x)
            .chain(&s.cxy)
            .chain(&s.cxx)
            .all(|v| v.is_finite());
        if !finite || s.sum_w < 0.0 || s.cyy < 0.0 || s.n_positive > s.n_rows {
            return Err(bad("inconsistent contents"));
        }
        if (s.n_positive == 0) != (s.sum_w == 0.0) {
            return Err(bad("inconsistent contents"));
        }
        Ok(s)
    }

    fn symmetrize(&mut self) {
        let p = self.n_features;
        for j in 0..p {
            for k in 0..j {
                self.cxx[j * p + k] = self.cxx[k * p + j];
            }
        }
    }

    /// Gram matrix, cross-moment and response sum of squares of the design
    /// the solver works on: centred with an intercept, uncentred without.
    fn design_moments(&self, with_intercept: bool) -> (Mat<f64>, Vec<f64>, f64) {
        let p = self.n_features;
        let w = if with_intercept { 0.0 } else { self.sum_w };
        let (mx, my) = (self.mean_x(), self.mean_y());
        let g = Mat::from_fn(p, p, |j, k| self.cxx[j * p + k] + w * mx[j] * mx[k]);
        let c = (0..p).map(|j| self.cxy[j] + w * mx[j] * my).collect();
        (g, c, self.cyy + w * my * my)
    }

    /// `ȳ − x̄ᵀβ` over the columns `kept`, with the shifts and the relative
    /// means summed separately.
    fn intercept(&self, kept: &[usize], beta: &[f64]) -> f64 {
        let (mut big, mut small) = (self.shift_y, self.mean_y);
        for (a, &j) in kept.iter().enumerate() {
            big -= self.shift_x[j] * beta[a];
            small -= self.mean_x[j] * beta[a];
        }
        big + small
    }

    /// Uncentred weighted squared norm `Σ wᵢ xᵢⱼ²` of column `j`.
    fn column_norm2(&self, j: usize) -> f64 {
        let p = self.n_features;
        let m = self.shift_x[j] + self.mean_x[j];
        self.cxx[j * p + j] + self.sum_w * m * m
    }
}

/// Validate one input row.
fn check_row(x: impl Iterator<Item = f64>, y: f64, w: f64) -> Result<(), RegressionError> {
    if !w.is_finite() || w < 0.0 {
        return Err(RegressionError::InvalidWeights);
    }
    if !y.is_finite() {
        return Err(RegressionError::NumericalError(
            "y contains a non-finite value".into(),
        ));
    }
    for (j, v) in x.enumerate() {
        if !v.is_finite() {
            return Err(RegressionError::NumericalError(format!(
                "X contains a non-finite value in column {j}"
            )));
        }
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut a = [0u8; N];
        a.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        a
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take())
    }
}

/// Output of a fit from a state.
pub(crate) struct StateFit {
    pub result: RegressionResult,
    /// Variance factor over the intercept (first, if fitted) and the
    /// non-aliased columns.
    pub reduced_factor: Option<Mat<f64>>,
    pub aliased: Vec<bool>,
}

/// Fit a least-squares (`ridge_lambda == 0`) or ridge model from a state.
///
/// Least squares aliases columns R-`lm`-style: columns are taken in order and
/// column `j` is aliased when its squared residual norm after projecting out
/// the intercept and the earlier kept columns (the Cholesky pivot of the
/// Gram matrix) is below `alias_tol²` times its squared uncentred norm. Ridge
/// keeps every column and penalises all but the intercept.
pub(crate) fn fit_state(
    state: &SufficientStats,
    options: &RegressionOptions,
    alias_tol: f64,
    ridge_lambda: f64,
) -> Result<StateFit, RegressionError> {
    let p = state.n_features;
    let with_intercept = options.with_intercept;
    let n_rows = state.n_observations();
    if n_rows < 2 {
        return Err(RegressionError::InsufficientObservations {
            needed: 2,
            got: n_rows,
        });
    }
    if state.sum_w < 1e-14 {
        return Err(RegressionError::InvalidWeights);
    }
    let n_params_full = p + usize::from(with_intercept);
    if ridge_lambda == 0.0 && state.n_positive() < n_params_full {
        return Err(RegressionError::InsufficientObservations {
            needed: n_params_full,
            got: state.n_positive(),
        });
    }

    let (g, c, yy) = state.design_moments(with_intercept);
    let ridge = ridge_lambda > 0.0;
    let (beta_k, kept, factor_k, rss) = if ridge {
        solve_ridge(&g, &c, yy, ridge_lambda)?
    } else {
        let norms: Vec<f64> = (0..p).map(|j| state.column_norm2(j)).collect();
        solve_least_squares(&g, &c, yy, &norms, alias_tol)
    };
    if !with_intercept && kept.is_empty() {
        return Err(RegressionError::AllFeaturesConstant);
    }

    let mut aliased = vec![true; p];
    let mut coefficients = Col::from_fn(p, |_| f64::NAN);
    for (a, &j) in kept.iter().enumerate() {
        aliased[j] = false;
        coefficients[j] = beta_k[a];
    }
    super::ols::check_coefficients_finite(&coefficients, &aliased)?;
    let intercept = with_intercept.then(|| state.intercept(&kept, &beta_k));

    let rank = kept.len();
    let n_params = if ridge {
        n_params_full
    } else {
        rank + usize::from(with_intercept)
    };
    let tss = if with_intercept { state.cyy } else { yy };
    let stats = super::fit_stats::linear_fit_stats_from_sums(
        state.n_positive(),
        tss,
        rss,
        state.sum_log_w,
        with_intercept,
        n_params,
        ridge,
    );

    let mut result = RegressionResult::empty(p, 0);
    result.coefficients = coefficients;
    result.intercept = intercept;
    result.rank = rank;
    result.n_parameters = n_params;
    // Rows with zero weight do not count (`nobs` / `df.residual` of `lm`).
    result.n_observations = state.n_positive();
    result.aliased = aliased.clone();
    result.rank_tolerance = options.rank_tolerance;
    stats.apply(&mut result);
    result.confidence_level = options.confidence_level;

    let means = state.mean_x();
    let kept_means: Vec<f64> = kept.iter().map(|&j| means[j]).collect();
    let reduced = factor_k.map(|k| augment_factor(&k, &kept_means, state.sum_w, with_intercept));
    if options.compute_inference {
        if let Some(m) = &reduced {
            apply_inference(&mut result, m, !ridge, options.confidence_level);
        }
    }
    Ok(StateFit {
        result,
        reduced_factor: reduced,
        aliased,
    })
}

/// Least squares with in-order aliasing via a sequential Cholesky
/// factorisation. Returns the kept coefficients, the kept column indices,
/// `G_KK⁻¹` and the RSS.
fn solve_least_squares(
    g: &Mat<f64>,
    c: &[f64],
    yy: f64,
    norms: &[f64],
    tol: f64,
) -> (Vec<f64>, Vec<usize>, Option<Mat<f64>>, f64) {
    let p = g.nrows();
    let mut kept: Vec<usize> = Vec::with_capacity(p);
    // Rows of the lower-triangular Cholesky factor of G_KK.
    let mut l: Vec<Vec<f64>> = Vec::with_capacity(p);
    for j in 0..p {
        if !(norms[j] > 0.0 && norms[j].is_finite()) {
            continue;
        }
        let row = forward_solve(&l, kept.iter().map(|&i| g[(i, j)]));
        let d = g[(j, j)] - row.iter().map(|v| v * v).sum::<f64>();
        if d < tol * tol * norms[j] {
            continue;
        }
        let mut row = row;
        row.push(d.sqrt());
        l.push(row);
        kept.push(j);
    }
    let k = kept.len();
    // z = L⁻¹ c_K, β = L⁻ᵀ z, RSS = yy − zᵀz.
    let z = forward_solve(&l, kept.iter().map(|&j| c[j]));
    let rss = (yy - z.iter().map(|v| v * v).sum::<f64>()).max(0.0);
    let beta = backward_solve(&l, &z);
    // G_KK⁻¹ = L⁻ᵀ L⁻¹ from the columns of L⁻¹.
    let mut linv = Mat::<f64>::zeros(k, k);
    for col in 0..k {
        let e = forward_solve(&l, (0..k).map(|i| f64::from(u8::from(i == col))));
        for i in 0..k {
            linv[(i, col)] = e[i];
        }
    }
    let inv = linv.transpose() * &linv;
    (beta, kept, Some(inv), rss)
}

/// Ridge `(G + λI) β = c` on every column. Returns the coefficients, all
/// column indices, the sandwich factor `B G B` (`B = (G + λI)⁻¹`) and the
/// RSS `yy − 2βᵀc + βᵀGβ`.
#[allow(clippy::type_complexity)]
fn solve_ridge(
    g: &Mat<f64>,
    c: &[f64],
    yy: f64,
    lambda: f64,
) -> Result<(Vec<f64>, Vec<usize>, Option<Mat<f64>>, f64), RegressionError> {
    use faer::linalg::solvers::Solve;
    let p = g.nrows();
    let mut h = g.clone();
    for j in 0..p {
        h[(j, j)] += lambda;
    }
    let llt = h
        .llt(faer::Side::Lower)
        .map_err(|_| RegressionError::SingularMatrix)?;
    let cc = Col::from_fn(p, |j| c[j]);
    let beta = llt.solve(&cc);
    let gb = g * &beta;
    let rss = yy - 2.0 * (0..p).map(|j| beta[j] * c[j]).sum::<f64>()
        + (0..p).map(|j| beta[j] * gb[j]).sum::<f64>();
    let b = llt.solve(Mat::<f64>::identity(p, p));
    let factor = &b * g * &b;
    Ok((
        (0..p).map(|j| beta[j]).collect(),
        (0..p).collect(),
        Some(factor),
        rss.max(0.0),
    ))
}

/// Solve `L v = b` for the lower-triangular factor given by rows.
fn forward_solve(l: &[Vec<f64>], b: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::with_capacity(l.len());
    for (i, bi) in b.enumerate() {
        let s: f64 = (0..i).map(|m| l[i][m] * v[m]).sum();
        v.push((bi - s) / l[i][i]);
    }
    v
}

/// Solve `Lᵀ v = b` for the lower-triangular factor given by rows.
fn backward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let k = b.len();
    let mut v = vec![0.0; k];
    for i in (0..k).rev() {
        let s: f64 = ((i + 1)..k).map(|m| l[m][i] * v[m]).sum();
        v[i] = (b[i] - s) / l[i][i];
    }
    v
}

/// Variance factor of `(intercept, β_K)` from the slope factor `K` in centred
/// coordinates: `Var(b₀) ∝ 1/W + x̄ᵀKx̄`, `Cov(b₀, β) ∝ −Kx̄`.
fn augment_factor(k: &Mat<f64>, means: &[f64], sum_w: f64, with_intercept: bool) -> Mat<f64> {
    if !with_intercept {
        return k.clone();
    }
    let r = k.nrows();
    let kx: Vec<f64> = (0..r)
        .map(|a| (0..r).map(|b| k[(a, b)] * means[b]).sum())
        .collect();
    let m00 = 1.0 / sum_w + (0..r).map(|a| means[a] * kx[a]).sum::<f64>();
    Mat::from_fn(r + 1, r + 1, |a, b| match (a, b) {
        (0, 0) => m00,
        (0, b) => -kx[b - 1],
        (a, 0) => -kx[a - 1],
        (a, b) => k[(a - 1, b - 1)],
    })
}

/// Classical inference from the reduced variance factor. With
/// `wald_tests = false` (ridge, λ > 0) only standard errors are reported,
/// as `RidgeRegressor::fit` does.
fn apply_inference(result: &mut RegressionResult, m: &Mat<f64>, wald_tests: bool, level: f64) {
    let df = result.residual_df() as f64;
    if df <= 0.0 || !result.mse.is_finite() {
        return;
    }
    let mse = result.mse;
    let sd = |v: f64| {
        let var = mse * v;
        if var >= 0.0 {
            var.sqrt()
        } else {
            f64::NAN
        }
    };
    let off = usize::from(result.intercept.is_some());
    let mut idx = off;
    let se = Col::from_fn(result.aliased.len(), |j| {
        if result.aliased[j] {
            f64::NAN
        } else {
            idx += 1;
            sd(m[(idx - 1, idx - 1)])
        }
    });
    let se_int = result.intercept.map(|_| sd(m[(0, 0)]));
    result.intercept_std_error = se_int;
    if wald_tests {
        let t = CoefficientInference::t_statistics(&result.coefficients, &se);
        result.p_values = Some(CoefficientInference::p_values(&t, df));
        let (lo, hi) =
            CoefficientInference::confidence_intervals(&result.coefficients, &se, df, level);
        result.t_statistics = Some(t);
        result.conf_interval_lower = Some(lo);
        result.conf_interval_upper = Some(hi);
        if let (Some(b0), Some(s0)) = (result.intercept, se_int) {
            let t0 = if s0 > 0.0 { b0 / s0 } else { f64::NAN };
            let dist = StudentsT::new(0.0, 1.0, df).ok();
            let p0 = if t0.is_finite() {
                dist.map_or(f64::NAN, |d| 2.0 * (1.0 - d.cdf(t0.abs())))
            } else {
                f64::NAN
            };
            let tc = dist.map_or(f64::NAN, |d| d.inverse_cdf(1.0 - (1.0 - level) / 2.0));
            result.intercept_t_statistic = Some(t0);
            result.intercept_p_value = Some(p0);
            result.intercept_conf_interval = Some((b0 - tc * s0, b0 + tc * s0));
        }
    }
    result.std_errors = Some(se);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<(Vec<f64>, f64, f64)> {
        (0..40)
            .map(|i| {
                let t = i as f64 * 0.37;
                let x = vec![t.sin(), (0.5 * t).cos() + 0.1 * t];
                let y = 1.0 + 2.0 * x[0] - x[1] + 0.1 * (3.1 * t).sin();
                (x, y, 0.5 + (i % 3) as f64)
            })
            .collect()
    }

    fn assert_close(a: &SufficientStats, b: &SufficientStats, tol: f64) {
        assert_eq!(a.n_rows, b.n_rows);
        assert_eq!(a.n_positive, b.n_positive);
        let pairs = [
            (a.sum_w, b.sum_w),
            (a.sum_log_w, b.sum_log_w),
            (a.mean_y, b.mean_y),
            (a.cyy, b.cyy),
        ];
        for (u, v) in pairs.into_iter().chain(
            a.mean_x
                .iter()
                .chain(&a.cxy)
                .chain(&a.cxx)
                .copied()
                .zip(b.mean_x.iter().chain(&b.cxy).chain(&b.cxx).copied()),
        ) {
            assert!((u - v).abs() <= tol * (1.0 + v.abs()), "{u} vs {v}");
        }
    }

    #[test]
    fn welford_push_matches_two_pass() {
        let data = rows();
        let mut s = SufficientStats::new(2);
        for (x, y, w) in &data {
            s.push_weighted(x, *y, *w).unwrap();
        }
        let x = Mat::from_fn(data.len(), 2, |i, j| data[i].0[j]);
        let y = Col::from_fn(data.len(), |i| data[i].1);
        let w = Col::from_fn(data.len(), |i| data[i].2);
        let batch = SufficientStats::from_data(&x, &y, Some(&w)).unwrap();
        assert_close(&s, &batch, 1e-13);
        assert!(s.is_weighted());
    }

    #[test]
    fn merge_with_empty_is_identity() {
        let mut s = SufficientStats::new(2);
        for (x, y, w) in rows() {
            s.push_weighted(&x, y, w).unwrap();
        }
        let empty = SufficientStats::new(2);
        assert_eq!(s.merged(&empty).unwrap(), s);
        assert_eq!(empty.merged(&s).unwrap(), s);
    }

    #[test]
    fn zero_weight_rows_only_count() {
        let mut s = SufficientStats::new(1);
        s.push(&[1.0], 2.0).unwrap();
        s.push_weighted(&[5.0], 9.0, 0.0).unwrap();
        assert_eq!(s.n_observations(), 2);
        assert_eq!(s.n_positive(), 1);
        assert_eq!(s.mean_x(), vec![1.0]);
    }

    #[test]
    fn bytes_round_trip_and_rejections() {
        let mut s = SufficientStats::new(2);
        for (x, y, w) in rows() {
            s.push_weighted(&x, y, w).unwrap();
        }
        let bytes = s.to_bytes();
        assert_eq!(SufficientStats::from_bytes(&bytes).unwrap(), s);
        assert_eq!(
            SufficientStats::from_bytes(&SufficientStats::new(0).to_bytes()).unwrap(),
            SufficientStats::new(0)
        );
        assert!(SufficientStats::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut v2 = bytes.clone();
        v2[4] = 2;
        assert!(SufficientStats::from_bytes(&v2).is_err());
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(SufficientStats::from_bytes(&magic).is_err());
        // A huge feature count must not allocate.
        let mut huge = bytes;
        huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(SufficientStats::from_bytes(&huge).is_err());
    }

    #[test]
    fn rejects_bad_rows() {
        let mut s = SufficientStats::new(1);
        assert!(s.push(&[f64::NAN], 1.0).is_err());
        assert!(s.push(&[1.0], f64::INFINITY).is_err());
        assert!(s.push_weighted(&[1.0], 1.0, -1.0).is_err());
        assert!(s.push(&[1.0, 2.0], 1.0).is_err());
        assert!(s.merge(&SufficientStats::new(2)).is_err());
        assert!(s.is_empty());
    }
}
