//! Theil–Sen estimator: robust regression based on the spatial median of
//! per-subsample OLS coefficient vectors.
//!
//! Univariate case: slope is the median of all pairwise slopes
//! `(y_j - y_i) / (x_j - x_i)`; intercept is `median(y - slope * x)`. Has a
//! 29.3% asymptotic breakdown point and matches `mblm::mblm` (R) and
//! `sklearn.linear_model.TheilSenRegressor` numerically.
//!
//! Multivariate case: for each subsample of `n_subsamples` observations
//! (default `n_features + 1`, or `n_features` without intercept), solve the
//! exact system — least squares when `n_subsamples` is larger — to get a
//! coefficient vector, then take the spatial (geometric) median of all those
//! vectors using a Vardi–Zhang modified Weiszfeld iteration (the ordinary
//! median for a single coefficient, as sklearn). When
//! `C(n, n_subsamples) <= max_subpopulation` the algorithm enumerates every
//! subsample (matching sklearn's deterministic path); otherwise it draws
//! `max_subpopulation` random subsamples using the provided seed.
//!
//! # Inference
//!
//! With [`compute_inference`](TheilSenRegressorBuilder::compute_inference)
//! the fit also carries coefficient intervals at
//! [`confidence_level`](TheilSenRegressorBuilder::confidence_level), from one
//! of two methods ([`TheilSenCiMethod`]):
//!
//! - **Exact** (one feature with an intercept, the median-of-pairwise-slopes
//!   path): the distribution-free interval of Sen (1968). With `N` pairwise
//!   slopes sorted, the bounds are the slopes at ranks `round((N − C)/2)` and
//!   `round((N + C)/2 + 1)`, `C = z_{(1+level)/2} · sqrt(Var S)`, where `S` is
//!   Kendall's score of `(x, y)` and `Var S` its tie-corrected null variance.
//!   The slope also gets Kendall's continuity-corrected `z = sign(S)(|S| − 1) /
//!   sqrt(Var S)` (in `t_statistics`) and its two-sided normal p-value for
//!   slope = 0. This matches R's `trend::sens.slope`. There is no standard
//!   error and no interval for the intercept; a bound whose rank falls outside
//!   `1..N` (very small `n`) is `NaN`.
//! - **Bootstrap** (any number of features, and the intercept): the pairs
//!   bootstrap. `n_bootstrap` resamples of the rows are refitted; each
//!   coefficient gets the bootstrap standard deviation (`std_errors`) and the
//!   percentile interval (R `quantile` type 7). Deterministic given
//!   `random_state` and independent of the row order. No test statistic or
//!   p-value.
//!
//! [`TheilSenCiMethod::Auto`] (the default) uses the exact interval where it
//! applies and the bootstrap otherwise. Theil–Sen has no F-test:
//! `f_statistic` / `f_pvalue` are `NaN`.
//!
//! # Rank-deficient designs
//!
//! Linearly dependent columns (e.g. `x2 = 2·x1`, or a constant column when an
//! intercept is fitted) are detected up front with a column-order
//! Gram–Schmidt (relative tolerance `1e-7`). As in
//! [`OlsRegressor`](crate::solvers::OlsRegressor) and R's `lm`, the later
//! redundant column is **aliased**: the estimator is fitted on the remaining
//! columns, the aliased ones get a `NaN` coefficient and `aliased = true`,
//! and prediction ignores them. (sklearn instead takes the minimum-norm
//! least-squares solution of each singular subsample, which spreads the
//! effect over the collinear columns.)
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{TheilSenRegressor, Regressor, FittedRegressor};
//!
//! let fitted = TheilSenRegressor::builder()
//!     .with_intercept(true)
//!     .build()
//!     .fit(&x, &y)?;
//! ```

use crate::core::{HasModelInfo, ModelInfo, OptionsError};
use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::linalg::solvers::SolveLstsq;
use faer::prelude::Solve;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal};

/// How [`TheilSenRegressor`] computes coefficient intervals when inference is
/// requested.
///
/// - `Exact`: Sen's (1968) distribution-free interval for the single slope of
///   a one-feature fit with an intercept, at the ranks `round((N − C)/2)` and
///   `round((N + C)/2 + 1)` of the `N` sorted pairwise slopes,
///   `C = z_{(1+level)/2} · sqrt(Var S)` with Kendall's tie-corrected `Var S`
///   (R's `trend::sens.slope`). Also Kendall's continuity-corrected z
///   (`t_statistics`) and its two-sided p-value for slope = 0. No standard
///   error, no intercept interval.
/// - `Bootstrap`: pairs bootstrap with `n_bootstrap` refits; bootstrap SDs
///   (`std_errors`, `intercept_std_error`) and percentile intervals (R
///   `quantile` type 7) for every coefficient and the intercept, deterministic
///   given `random_state`. No test statistic.
/// - `Auto`: `Exact` where it applies, `Bootstrap` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TheilSenCiMethod {
    /// `Exact` for one feature with an intercept (the median-of-pairwise-
    /// slopes path), `Bootstrap` otherwise.
    #[default]
    Auto,
    /// Sen's (1968) distribution-free interval for the single slope.
    Exact,
    /// Pairs bootstrap with percentile intervals, for every coefficient and
    /// the intercept.
    Bootstrap,
}

/// Theil–Sen estimator.
#[derive(Debug, Clone)]
pub struct TheilSenRegressor {
    with_intercept: bool,
    /// Cap on number of subsamples examined. Matches sklearn's
    /// `max_subpopulation = 10_000` default.
    max_subpopulation: usize,
    /// Subsample size. `None` means `n_features + 1` (the minimum to identify
    /// an OLS solution with intercept) which mirrors sklearn's default.
    n_subsamples: Option<usize>,
    max_iter: usize,
    tol: f64,
    random_state: u64,
    /// Univariate (one feature, with intercept) fits use the classical
    /// median of pairwise slopes. `false` selects sklearn's spatial-median
    /// algorithm for that case as well.
    univariate_pairwise: bool,
    compute_inference: bool,
    confidence_level: f64,
    ci_method: TheilSenCiMethod,
    n_bootstrap: usize,
}

impl Default for TheilSenRegressor {
    fn default() -> Self {
        Self {
            with_intercept: true,
            max_subpopulation: 10_000,
            n_subsamples: None,
            max_iter: 300,
            tol: 1e-3,
            random_state: 0,
            univariate_pairwise: true,
            compute_inference: false,
            confidence_level: 0.95,
            ci_method: TheilSenCiMethod::Auto,
            n_bootstrap: 999,
        }
    }
}

impl TheilSenRegressor {
    /// Configure a Theil-Sen estimator.
    pub fn builder() -> TheilSenRegressorBuilder {
        TheilSenRegressorBuilder::default()
    }

    fn effective_n_subsamples(&self, n_features_eff: usize) -> usize {
        self.n_subsamples
            .unwrap_or(n_features_eff)
            .max(n_features_eff)
    }

    /// Whether a fit on `p` (non-aliased) features takes the classical
    /// median-of-pairwise-slopes path.
    fn is_pairwise(&self, p: usize) -> bool {
        p == 1 && self.with_intercept && self.univariate_pairwise
    }

    /// The interval method used for a fit on `p` (non-aliased) features.
    fn resolve_ci_method(&self, p: usize) -> Result<TheilSenCiMethod, RegressionError> {
        match self.ci_method {
            TheilSenCiMethod::Auto if self.is_pairwise(p) => Ok(TheilSenCiMethod::Exact),
            TheilSenCiMethod::Auto => Ok(TheilSenCiMethod::Bootstrap),
            TheilSenCiMethod::Exact if !self.is_pairwise(p) => {
                Err(RegressionError::InvalidOptions(OptionsError::Unsupported(
                    "the exact Theil-Sen interval needs exactly one feature and an intercept; \
                     use the bootstrap"
                        .to_string(),
                )))
            }
            m => Ok(m),
        }
    }

    fn validate_inference_options(&self) -> Result<(), RegressionError> {
        if !(self.confidence_level > 0.0 && self.confidence_level < 1.0) {
            return Err(RegressionError::InvalidOptions(
                OptionsError::InvalidConfidenceLevel(self.confidence_level),
            ));
        }
        if self.n_bootstrap < 2 {
            return Err(RegressionError::InvalidOptions(OptionsError::Unsupported(
                format!("n_bootstrap must be at least 2, got {}", self.n_bootstrap),
            )));
        }
        Ok(())
    }

    /// Fit without inference on a design with no aliased columns.
    fn fit_point(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedTheilSen, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        let p_eff = if self.with_intercept { p + 1 } else { p };

        // Univariate with intercept: the classical Theil–Sen estimator —
        // slope = median of all pairwise slopes (pairs with equal x skipped),
        // intercept = median(y − slope·x). sklearn instead applies its
        // spatial-median-over-subsamples algorithm to 1-D inputs too; that
        // behaviour stays available via `univariate_pairwise(false)`.
        if self.is_pairwise(p) {
            let xs: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
            let ys: Vec<f64> = (0..n).map(|i| y[i]).collect();
            let slope = median_pairwise_slope(&xs, &ys).ok_or(RegressionError::SingularMatrix)?;
            let mut res: Vec<f64> = xs.iter().zip(&ys).map(|(xi, yi)| yi - slope * xi).collect();
            let intercept = median_in_place(&mut res);
            return Ok(build_fitted_from_full(vec![intercept, slope], x, y, true));
        }

        // Multivariate (or univariate without intercept): sklearn's spatial
        // median of per-subsample OLS coefficient vectors.
        let n_sub = self.effective_n_subsamples(p_eff);
        if n_sub > n {
            return Err(RegressionError::InsufficientObservations {
                needed: n_sub,
                got: n,
            });
        }

        let total = n_choose_k(n, n_sub);
        let coef_dim = p_eff;
        let mut coef_vectors: Vec<Vec<f64>> = Vec::new();

        if total <= self.max_subpopulation as u128 {
            // Exhaustive enumeration in lexicographic order — matches sklearn's
            // deterministic branch.
            for_each_combination(n, n_sub, |idx| {
                if let Some(beta) = ols_on_subsample(x, y, idx, self.with_intercept) {
                    coef_vectors.push(beta);
                }
            });
        } else {
            // Stochastic branch with a deterministic seed.
            let mut rng = SplitMix64::new(self.random_state.wrapping_add(0x9E37_79B9_7F4A_7C15));
            let mut buf = vec![0usize; n_sub];
            for _ in 0..self.max_subpopulation {
                random_combination(n, n_sub, &mut buf, &mut rng);
                if let Some(beta) = ols_on_subsample(x, y, &buf, self.with_intercept) {
                    coef_vectors.push(beta);
                }
            }
        }

        if coef_vectors.is_empty() {
            return Err(RegressionError::SingularMatrix);
        }

        let median = spatial_median(&coef_vectors, coef_dim, self.max_iter, self.tol);

        Ok(build_fitted_from_full(median, x, y, self.with_intercept))
    }

    /// Sen's exact interval and Kendall's test for the single slope.
    fn exact_inference(&self, fitted: &mut FittedTheilSen, x: &Mat<f64>, y: &Col<f64>) {
        let n = x.nrows();
        let xs: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
        let ys: Vec<f64> = (0..n).map(|i| y[i]).collect();
        let sen = sen_interval(&xs, &ys, self.confidence_level);
        let r = &mut fitted.result;
        r.std_errors = None;
        r.conf_interval_lower = Some(Col::from_fn(1, |_| sen.lower));
        r.conf_interval_upper = Some(Col::from_fn(1, |_| sen.upper));
        r.t_statistics = Some(Col::from_fn(1, |_| sen.z));
        r.p_values = Some(Col::from_fn(1, |_| sen.p_value));
        fitted.ci_method = Some(TheilSenCiMethod::Exact);
    }

    /// Pairs-bootstrap SDs and percentile intervals of every coefficient and
    /// the intercept.
    fn bootstrap_inference(&self, fitted: &mut FittedTheilSen, x: &Mat<f64>, y: &Col<f64>) {
        let n = x.nrows();
        let p = x.ncols();
        let dim = p + usize::from(self.with_intercept);
        let mut inner = self.clone();
        inner.compute_inference = false;
        let mut rng = SplitMix64::new(self.random_state ^ 0xB007_5EED_D1CE_A5E5);
        // Resample from the rows in a canonical (sorted) order, so the result
        // depends on the data and the seed but not on the row order (e.g. the
        // order in which a parallel aggregate collected the rows).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            (0..p)
                .map(|j| x[(a, j)].total_cmp(&x[(b, j)]))
                .find(|o| o.is_ne())
                .unwrap_or_else(|| y[a].total_cmp(&y[b]))
        });
        let mut idx = vec![0usize; n];
        // draws[k] holds the replicates of parameter k (intercept first).
        let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(self.n_bootstrap); dim];
        for _ in 0..self.n_bootstrap {
            idx.iter_mut()
                .for_each(|i| *i = order[rng.gen_range_usize(0, n)]);
            inner.random_state = rng.next_u64();
            let xb = Mat::from_fn(n, p, |i, j| x[(idx[i], j)]);
            let yb = Col::from_fn(n, |i| y[idx[i]]);
            // A resample can be degenerate (e.g. all x equal, or a column that
            // becomes aliased); such replicates are dropped.
            let Ok(fb) = inner.fit(&xb, &yb) else {
                continue;
            };
            let r = &fb.result;
            let params: Vec<f64> = r
                .intercept
                .into_iter()
                .chain(r.coefficients.iter().copied())
                .collect();
            if params.iter().all(|v| v.is_finite()) {
                for (d, v) in draws.iter_mut().zip(params) {
                    d.push(v);
                }
            }
        }

        let a = (1.0 - self.confidence_level) / 2.0;
        let summary: Vec<(f64, f64, f64)> = draws
            .iter_mut()
            .map(|d| {
                if d.len() < 2 {
                    return (f64::NAN, f64::NAN, f64::NAN);
                }
                d.sort_by(|u, v| u.total_cmp(v));
                let m = d.len() as f64;
                let mean = d.iter().sum::<f64>() / m;
                let sd = (d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0)).sqrt();
                (sd, quantile_type7(d, a), quantile_type7(d, 1.0 - a))
            })
            .collect();
        let off = usize::from(self.with_intercept);
        let r = &mut fitted.result;
        if self.with_intercept {
            let (sd, lo, hi) = summary[0];
            r.intercept_std_error = Some(sd);
            r.intercept_conf_interval = Some((lo, hi));
        }
        r.std_errors = Some(Col::from_fn(p, |j| summary[off + j].0));
        r.conf_interval_lower = Some(Col::from_fn(p, |j| summary[off + j].1));
        r.conf_interval_upper = Some(Col::from_fn(p, |j| summary[off + j].2));
        r.t_statistics = None;
        r.p_values = None;
        fitted.ci_method = Some(TheilSenCiMethod::Bootstrap);
        fitted.bootstrap_replicates = Some(draws.first().map_or(0, Vec::len));
    }
}

impl Regressor for TheilSenRegressor {
    type Fitted = FittedTheilSen;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n = x.nrows();
        if n != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        let p = x.ncols();
        let p_eff = if self.with_intercept { p + 1 } else { p };
        if n < p_eff {
            return Err(RegressionError::InsufficientObservations {
                needed: p_eff,
                got: n,
            });
        }
        if self.compute_inference {
            self.validate_inference_options()?;
        }

        // Linearly dependent columns (e.g. x2 = 2·x1, or a constant column
        // with an intercept) make every subsample system singular; partial-
        // pivot LU then returns huge finite "solutions" that dominate the
        // spatial median (#50). Like `OlsRegressor` / R's `lm`, the later
        // redundant columns are aliased: Theil–Sen is fitted on the kept
        // columns and the aliased ones get a `NaN` coefficient.
        let aliased = aliased_columns(x, self.with_intercept, ALIAS_TOL);
        if aliased.iter().any(|&a| a) {
            let keep: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
            if keep.is_empty() && !self.with_intercept {
                return Err(RegressionError::SingularMatrix);
            }
            let xr = Mat::from_fn(n, keep.len(), |i, k| x[(i, keep[k])]);
            let inner = self.fit(&xr, y)?;
            return Ok(expand_aliased(inner, &keep, p));
        }

        let method = if self.compute_inference {
            Some(self.resolve_ci_method(p)?)
        } else {
            None
        };
        let mut fitted = self.fit_point(x, y)?;
        match method {
            Some(TheilSenCiMethod::Exact) => self.exact_inference(&mut fitted, x, y),
            Some(_) => self.bootstrap_inference(&mut fitted, x, y),
            None => {}
        }
        if method.is_some() {
            fitted.result.confidence_level = self.confidence_level;
        }
        Ok(fitted)
    }
}

/// Relative tolerance for declaring a column linearly dependent on the
/// preceding ones (R's `lm` uses 1e-7 for its pivoted QR).
const ALIAS_TOL: f64 = 1e-7;

/// Column-order Gram–Schmidt rank detection on the design (centred when an
/// intercept is fitted, which projects out the intercept column). Column `j`
/// is aliased when its residual after projecting onto the previously kept
/// columns is below `tol` times its own norm — the same "later column is
/// redundant" rule as R's `lm`.
fn aliased_columns(x: &Mat<f64>, with_intercept: bool, tol: f64) -> Vec<bool> {
    let n = x.nrows();
    let p = x.ncols();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut aliased = vec![false; p];
    for (j, flag) in aliased.iter_mut().enumerate() {
        let raw: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
        let scale = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut r = raw;
        if with_intercept {
            let mean = r.iter().sum::<f64>() / n as f64;
            r.iter_mut().for_each(|v| *v -= mean);
        }
        // Two passes of modified Gram–Schmidt for numerical stability.
        for _ in 0..2 {
            for q in &basis {
                let d: f64 = r.iter().zip(q).map(|(a, b)| a * b).sum();
                r.iter_mut().zip(q).for_each(|(a, b)| *a -= d * b);
            }
        }
        let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if scale == 0.0 || norm <= tol * scale {
            *flag = true;
        } else {
            basis.push(r.into_iter().map(|v| v / norm).collect());
        }
    }
    aliased
}

/// Re-expand a fit on the kept columns to all `p` columns: aliased columns
/// get a `NaN` coefficient and `aliased = true`.
fn expand_aliased(inner: FittedTheilSen, keep: &[usize], p: usize) -> FittedTheilSen {
    let mut result = inner.result;
    let mut coefficients = Col::from_fn(p, |_| f64::NAN);
    for (k, &j) in keep.iter().enumerate() {
        coefficients[j] = result.coefficients[k];
    }
    result.coefficients = coefficients;
    let mut aliased = vec![true; p];
    for &j in keep {
        aliased[j] = false;
    }
    result.aliased = aliased;
    // Per-coefficient inference: aliased columns get NaN.
    let expand = |c: Option<Col<f64>>| {
        c.map(|c| {
            let mut out = Col::from_fn(p, |_| f64::NAN);
            for (k, &j) in keep.iter().enumerate() {
                out[j] = c[k];
            }
            out
        })
    };
    result.std_errors = expand(result.std_errors.take());
    result.t_statistics = expand(result.t_statistics.take());
    result.p_values = expand(result.p_values.take());
    result.conf_interval_lower = expand(result.conf_interval_lower.take());
    result.conf_interval_upper = expand(result.conf_interval_upper.take());
    FittedTheilSen { result, ..inner }
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
    if k < p_eff {
        return None;
    }
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

    // Square system: LU (skipped if singular); larger subsets
    // (`n_subsamples > p_eff`): least squares by QR, as sklearn's `lstsq`.
    let beta = if k == p_eff {
        xs.partial_piv_lu().solve(&ys)
    } else {
        let b = xs
            .col_piv_qr()
            .solve_lstsq(Mat::from_fn(k, 1, |i, _| ys[i]));
        Col::from_fn(p_eff, |j| b[(j, 0)])
    };
    let mut out = Vec::with_capacity(p_eff);
    let mut any_nan = false;
    for i in 0..p_eff {
        let v = beta[i];
        if !v.is_finite() {
            any_nan = true;
            break;
        }
        out.push(v);
    }
    if any_nan {
        None
    } else {
        Some(out)
    }
}

/// Vardi–Zhang modified Weiszfeld iteration for the spatial median.
///
/// `points` is a flat list of length-`dim` coefficient vectors. The
/// returned vector minimises `Σ_i ||x − p_i||₂`.
fn spatial_median(points: &[Vec<f64>], dim: usize, max_iter: usize, tol: f64) -> Vec<f64> {
    // One dimension: the ordinary median, as sklearn's `_spatial_median`
    // (for an even count the minimiser is any point between the two middle
    // values; Weiszfeld would stop somewhere in that interval).
    if dim == 1 {
        let mut v: Vec<f64> = points.iter().map(|p| p[0]).collect();
        return vec![median_in_place(&mut v)];
    }
    // Initialise with the coordinate-wise mean (matching sklearn's
    // `_spatial_median` so the iteration trajectory matches term-for-term
    // and convergence terminates at the same point under the same tol).
    let m = points.len() as f64;
    let mut y: Vec<f64> = (0..dim)
        .map(|j| points.iter().map(|p| p[j]).sum::<f64>() / m)
        .collect();

    for _ in 0..max_iter {
        let mut num = vec![0.0_f64; dim];
        let mut denom = 0.0_f64;
        let mut coincident: Option<usize> = None;
        let mut r_vec = vec![0.0_f64; dim];

        for (idx, p) in points.iter().enumerate() {
            let d = euclidean_distance(&y, p);
            if d < f64::EPSILON {
                coincident = Some(idx);
                continue;
            }
            let inv = 1.0 / d;
            denom += inv;
            for j in 0..dim {
                num[j] += p[j] * inv;
                r_vec[j] += (p[j] - y[j]) * inv;
            }
        }

        if denom == 0.0 {
            // All points coincide with y — we are already at the median.
            return y;
        }

        let t_vec: Vec<f64> = (0..dim).map(|j| num[j] / denom).collect();

        let new_y = if let Some(_idx) = coincident {
            // Vardi–Zhang correction: y_new = max(0, 1 − 1/||R||) * T + min(1, 1/||R||) * y
            let r_norm: f64 = (0..dim).map(|j| r_vec[j] * r_vec[j]).sum::<f64>().sqrt();
            if r_norm == 0.0 {
                return y;
            }
            let gamma = 1.0 / r_norm;
            let alpha = (1.0_f64 - gamma).max(0.0);
            let beta = gamma.min(1.0);
            (0..dim).map(|j| alpha * t_vec[j] + beta * y[j]).collect()
        } else {
            t_vec
        };

        let shift = euclidean_distance(&y, &new_y);
        y = new_y;
        if shift < tol {
            break;
        }
    }
    y
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s.sqrt()
}

fn n_choose_k(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc
            .saturating_mul((n - i) as u128)
            .saturating_div((i + 1) as u128);
    }
    acc
}

fn for_each_combination<F: FnMut(&[usize])>(n: usize, k: usize, mut f: F) {
    let mut idx: Vec<usize> = (0..k).collect();
    if k == 0 {
        f(&idx);
        return;
    }
    loop {
        f(&idx);
        // Advance to the next combination in lexicographic order.
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - k {
                idx[i] += 1;
                for j in (i + 1)..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return;
            }
        }
    }
}

fn random_combination(n: usize, k: usize, out: &mut [usize], rng: &mut SplitMix64) {
    debug_assert_eq!(out.len(), k);
    // Floyd's algorithm — O(k) and unbiased.
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
        // Unbiased modulo via rejection.
        let bound = u64::MAX - (u64::MAX % span);
        loop {
            let r = self.next_u64();
            if r < bound {
                return lo + (r % span) as usize;
            }
        }
    }
}

/// R-style median (mean of the two middle values for even length).
fn median_in_place(v: &mut [f64]) -> f64 {
    let n = v.len();
    let mid = n / 2;
    let (_, hi, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *hi;
    if n % 2 == 1 {
        hi
    } else {
        let lo = v[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

/// Above this many pairs the median slope is found by counting instead of
/// materialising every pairwise slope.
const EXPLICIT_PAIR_LIMIT: u64 = 4_000_000;

/// Median of `(y_j − y_i)/(x_j − x_i)` over all pairs with `x_i ≠ x_j`
/// (R `median` convention). Exact; `O(n²)` work/memory for up to
/// [`EXPLICIT_PAIR_LIMIT`] pairs, otherwise `O(n log n)` memory and
/// `O(n log n · log(1/ε))` time via slope-counting bisection.
fn median_pairwise_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    median_pairwise_slope_with_limit(x, y, EXPLICIT_PAIR_LIMIT)
}

fn median_pairwise_slope_with_limit(x: &[f64], y: &[f64], explicit_limit: u64) -> Option<f64> {
    let mut slopes = PairwiseSlopes::new(x, y, explicit_limit)?;
    let m = slopes.n_pairs;
    Some(if m % 2 == 1 {
        slopes.kth(m.div_ceil(2))
    } else {
        0.5 * (slopes.kth(m / 2) + slopes.kth(m / 2 + 1))
    })
}

/// The pairwise slopes `(y_j − y_i)/(x_j − x_i)` of all pairs with
/// `x_i ≠ x_j`, with order statistics either from the materialised slopes
/// (up to `explicit_limit` pairs) or by slope counting.
struct PairwiseSlopes {
    /// Points in (x, y) order.
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Number of pairs with distinct x.
    n_pairs: u64,
    /// Number of pairs of fully duplicated points.
    dup: u64,
    explicit: Option<Vec<f64>>,
}

impl PairwiseSlopes {
    /// `None` when every pair has equal x.
    fn new(x: &[f64], y: &[f64], explicit_limit: u64) -> Option<Self> {
        let n = x.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(y[a].total_cmp(&y[b])));
        let xs: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
        let ys: Vec<f64> = idx.iter().map(|&i| y[i]).collect();

        // Pairs with equal x (excluded) and fully duplicated points.
        let mut same_x: u64 = 0;
        let mut dup: u64 = 0;
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j + 1 < n && xs[j + 1] == xs[i] {
                j += 1;
            }
            let g = (j - i + 1) as u64;
            same_x += g * (g - 1) / 2;
            let mut a = i;
            while a <= j {
                let mut b = a;
                while b < j && ys[b + 1] == ys[a] {
                    b += 1;
                }
                let d = (b - a + 1) as u64;
                dup += d * (d - 1) / 2;
                a = b + 1;
            }
            i = j + 1;
        }
        let total = (n as u64) * (n as u64).saturating_sub(1) / 2;
        let n_pairs = total - same_x;
        if n_pairs == 0 {
            return None;
        }

        let explicit = (n_pairs <= explicit_limit).then(|| {
            let mut slopes = Vec::with_capacity(n_pairs as usize);
            for a in 0..n {
                for b in (a + 1)..n {
                    let dx = xs[b] - xs[a];
                    if dx != 0.0 {
                        slopes.push((ys[b] - ys[a]) / dx);
                    }
                }
            }
            slopes
        });
        Some(Self {
            xs,
            ys,
            n_pairs,
            dup,
            explicit,
        })
    }

    /// Number of pairs (x_a < x_b) with slope ≤ `s`
    ///   = number of position pairs a < b with u_b ≤ u_a, u = y − s·x,
    /// minus the duplicated points (whose u always tie).
    fn count_le(&self, s: f64) -> u64 {
        let mut u: Vec<f64> = self
            .xs
            .iter()
            .zip(&self.ys)
            .map(|(xi, yi)| yi - s * xi)
            .collect();
        let mut buf = vec![0.0; u.len()];
        count_non_increasing_pairs(&mut u, &mut buf) - self.dup
    }

    /// The `k`-th smallest slope (1-based, `1 ≤ k ≤ n_pairs`).
    fn kth(&mut self, k: u64) -> f64 {
        if let Some(slopes) = self.explicit.as_mut() {
            let (_, v, _) = slopes.select_nth_unstable_by((k - 1) as usize, |a, b| a.total_cmp(b));
            return *v;
        }
        let (xs, ys) = (&self.xs, &self.ys);
        let n = xs.len();
        // Bracket with bounds on |slope|.
        let y_range = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - ys.iter().copied().fold(f64::INFINITY, f64::min);
        let mut min_dx = f64::INFINITY;
        for w in xs.windows(2) {
            let d = w[1] - w[0];
            if d > 0.0 && d < min_dx {
                min_dx = d;
            }
        }
        let bound = y_range / min_dx + 1.0;
        let (mut lo, mut hi) = (-bound, bound);
        let (mut g_lo, mut g_hi) = (0_u64, self.n_pairs);
        let cap = (4 * n as u64).max(4096);
        for _ in 0..400 {
            if g_hi - g_lo <= cap {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                return hi; // interval exhausted: all remaining slopes ≈ hi
            }
            let g = self.count_le(mid);
            if g >= k {
                hi = mid;
                g_hi = g;
            } else {
                lo = mid;
                g_lo = g;
            }
        }
        // Enumerate the pairs whose slope lies in (lo, hi]: those whose
        // u-order flips between s = lo and s = hi.
        let mut order: Vec<usize> = (0..n).collect();
        let u_lo: Vec<f64> = xs.iter().zip(ys).map(|(xi, yi)| yi - lo * xi).collect();
        order.sort_by(|&a, &b| u_lo[a].total_cmp(&u_lo[b]).then(xs[b].total_cmp(&xs[a])));
        let mut items: Vec<(f64, usize)> = order.iter().map(|&i| (ys[i] - hi * xs[i], i)).collect();
        let mut buf = items.clone();
        let mut cand: Vec<f64> = Vec::new();
        enumerate_non_increasing_pairs(&mut items, &mut buf, &mut |a, b| {
            let dx = xs[b] - xs[a];
            if dx != 0.0 {
                cand.push((ys[b] - ys[a]) / dx);
            }
        });
        cand.sort_by(|a, b| a.total_cmp(b));
        let r = k.saturating_sub(g_lo + 1) as usize;
        cand.get(r.min(cand.len().saturating_sub(1)))
            .copied()
            .unwrap_or(hi)
    }
}

/// Sen's interval and Kendall's test for the slope of a single regressor.
#[derive(Debug, Clone, Copy)]
struct SenInterval {
    lower: f64,
    upper: f64,
    /// Kendall's continuity-corrected z for slope = 0.
    z: f64,
    p_value: f64,
}

/// Sen (1968) distribution-free confidence interval for the Theil–Sen slope,
/// following R's `trend::sens.slope`: with `N` pairwise slopes, the bounds
/// are the order statistics at ranks `round((N − C)/2)` and
/// `round((N + C)/2 + 1)` (R rounds half to even), `C = z · sqrt(Var S)`.
/// `Var S` is the null variance of Kendall's score with ties in x and in y.
/// A rank outside `1..N` gives a `NaN` bound.
fn sen_interval(x: &[f64], y: &[f64], level: f64) -> SenInterval {
    let nan = SenInterval {
        lower: f64::NAN,
        upper: f64::NAN,
        z: f64::NAN,
        p_value: f64::NAN,
    };
    let Some(mut slopes) = PairwiseSlopes::new(x, y, EXPLICIT_PAIR_LIMIT) else {
        return nan;
    };
    let var_s = kendall_var_s(x, y);
    if var_s.is_nan() || var_s <= 0.0 {
        return nan;
    }
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let c = normal.inverse_cdf(0.5 + level / 2.0) * var_s.sqrt();
    let n_pairs = slopes.n_pairs as f64;
    let rank_lo = ((n_pairs - c) / 2.0).round_ties_even();
    let rank_up = ((n_pairs + c) / 2.0 + 1.0).round_ties_even();
    let mut bound = |r: f64| {
        if r >= 1.0 && r <= n_pairs {
            slopes.kth(r as u64)
        } else {
            f64::NAN
        }
    };
    let lower = bound(rank_lo);
    let upper = bound(rank_up);

    // Kendall's score S = #(slope > 0) − #(slope < 0) over the pairs with
    // distinct x; #(slope ≥ 0) is the "≤ 0" count of the reflected data.
    let neg_y: Vec<f64> = y.iter().map(|v| -v).collect();
    let n_le0 = slopes.count_le(0.0);
    let n_ge0 = PairwiseSlopes::new(x, &neg_y, 0).map_or(0, |r| r.count_le(0.0));
    let s = n_ge0 as f64 - n_le0 as f64;
    let z = s.signum() * (s.abs() - 1.0).max(0.0) / var_s.sqrt();
    let z = if s == 0.0 { 0.0 } else { z };
    let p_value = (2.0 * normal.cdf(-z.abs())).min(1.0);
    SenInterval {
        lower,
        upper,
        z,
        p_value,
    }
}

/// Null variance of Kendall's score `S` of `(x, y)` with tie corrections for
/// both variables (Kendall 1970, eq. 4.3); without ties in `x` it is the
/// Mann–Kendall variance `[n(n−1)(2n+5) − Σ u(u−1)(2u+5)] / 18` over the tie
/// groups `u` of `y`.
fn kendall_var_s(x: &[f64], y: &[f64]) -> f64 {
    fn tie_sums(v: &[f64]) -> (f64, f64, f64) {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let (mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0);
        let mut i = 0;
        while i < v.len() {
            let mut j = i;
            while j + 1 < v.len() && v[j + 1] == v[i] {
                j += 1;
            }
            let t = (j - i + 1) as f64;
            s1 += t * (t - 1.0) * (2.0 * t + 5.0);
            s2 += t * (t - 1.0);
            s3 += t * (t - 1.0) * (t - 2.0);
            i = j + 1;
        }
        (s1, s2, s3)
    }
    let n = x.len() as f64;
    let (tx1, tx2, tx3) = tie_sums(x);
    let (ty1, ty2, ty3) = tie_sums(y);
    let mut v = (n * (n - 1.0) * (2.0 * n + 5.0) - tx1 - ty1) / 18.0;
    if n > 1.0 {
        v += tx2 * ty2 / (2.0 * n * (n - 1.0));
    }
    if n > 2.0 {
        v += tx3 * ty3 / (9.0 * n * (n - 1.0) * (n - 2.0));
    }
    v
}

/// R `quantile(type = 7)` of sorted data.
fn quantile_type7(sorted: &[f64], q: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Number of index pairs `a < b` with `v[b] <= v[a]` (merge sort; sorts `v`).
fn count_non_increasing_pairs(v: &mut [f64], buf: &mut [f64]) -> u64 {
    let n = v.len();
    if n < 2 {
        return 0;
    }
    let mid = n / 2;
    let mut c = {
        let (l, r) = v.split_at_mut(mid);
        let (bl, br) = buf.split_at_mut(mid);
        count_non_increasing_pairs(l, bl) + count_non_increasing_pairs(r, br)
    };
    let (mut i, mut j, mut k) = (0, mid, 0);
    while i < mid && j < n {
        if v[i] < v[j] {
            buf[k] = v[i];
            i += 1;
        } else {
            buf[k] = v[j];
            c += (mid - i) as u64;
            j += 1;
        }
        k += 1;
    }
    buf[k..k + (mid - i)].copy_from_slice(&v[i..mid]);
    k += mid - i;
    buf[k..k + (n - j)].copy_from_slice(&v[j..n]);
    v.copy_from_slice(&buf[..n]);
    c
}

/// Calls `f(i, j)` (original point ids) for every position pair `a < b` with
/// `v[b] <= v[a]`, sorting `items` by value.
fn enumerate_non_increasing_pairs(
    items: &mut [(f64, usize)],
    buf: &mut [(f64, usize)],
    f: &mut impl FnMut(usize, usize),
) {
    let n = items.len();
    if n < 2 {
        return;
    }
    let mid = n / 2;
    {
        let (l, r) = items.split_at_mut(mid);
        let (bl, br) = buf.split_at_mut(mid);
        enumerate_non_increasing_pairs(l, bl, f);
        enumerate_non_increasing_pairs(r, br, f);
    }
    let (mut i, mut j, mut k) = (0, mid, 0);
    while i < mid && j < n {
        if items[i].0 < items[j].0 {
            buf[k] = items[i];
            i += 1;
        } else {
            for l in &items[i..mid] {
                f(l.1, items[j].1);
            }
            buf[k] = items[j];
            j += 1;
        }
        k += 1;
    }
    buf[k..k + (mid - i)].copy_from_slice(&items[i..mid]);
    k += mid - i;
    buf[k..k + (n - j)].copy_from_slice(&items[j..n]);
    items.copy_from_slice(&buf[..n]);
}

fn build_fitted_from_full(
    beta: Vec<f64>,
    x: &Mat<f64>,
    y: &Col<f64>,
    with_intercept: bool,
) -> FittedTheilSen {
    let p = x.ncols();
    let (intercept, coef_slice): (Option<f64>, &[f64]) = if with_intercept {
        (Some(beta[0]), &beta[1..])
    } else {
        (None, &beta[..])
    };
    let coefficients = Col::from_fn(p, |i| coef_slice[i]);
    let n = x.nrows();

    let fitted_values = Col::from_fn(n, |i| {
        let mut v = intercept.unwrap_or(0.0);
        for j in 0..p {
            v += x[(i, j)] * coef_slice[j];
        }
        v
    });
    let residuals = Col::from_fn(n, |i| y[i] - fitted_values[i]);

    let n_params = p + if with_intercept { 1 } else { 0 };
    let y_mean: f64 = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;
    let tss: f64 = (0..n).map(|i| (y[i] - y_mean).powi(2)).sum();
    let rss: f64 = (0..n).map(|i| residuals[i].powi(2)).sum();
    let r_squared = if tss == 0.0 {
        if rss == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - rss / tss
    };
    let df_resid = (n - n_params) as f64;
    let adj_r_squared = if df_resid > 0.0 && tss > 0.0 {
        1.0 - (rss / df_resid) / (tss / (n as f64 - 1.0))
    } else {
        f64::NAN
    };
    let mse = rss / n as f64;
    let rmse = mse.sqrt();

    let mut result = RegressionResult::empty(p, n);
    result.coefficients = coefficients;
    result.intercept = intercept;
    result.residuals = residuals;
    result.fitted_values = fitted_values;
    result.rank = n_params;
    result.n_parameters = n_params;
    result.n_observations = n;
    result.r_squared = r_squared;
    result.adj_r_squared = adj_r_squared;
    result.rmse = rmse;
    result.mse = mse;
    result.f_statistic = f64::NAN;
    result.f_pvalue = f64::NAN;
    result.aic = f64::NAN;
    result.aicc = f64::NAN;
    result.bic = f64::NAN;
    result.log_likelihood = f64::NAN;

    FittedTheilSen {
        result,
        with_intercept,
        ci_method: None,
        bootstrap_replicates: None,
    }
}

/// A fitted Theil–Sen model.
#[derive(Debug, Clone)]
pub struct FittedTheilSen {
    result: RegressionResult,
    with_intercept: bool,
    ci_method: Option<TheilSenCiMethod>,
    bootstrap_replicates: Option<usize>,
}

impl HasModelInfo for FittedTheilSen {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::linear("theil_sen")
    }
}

impl FittedTheilSen {
    /// Whether the model was fit with an intercept.
    pub fn with_intercept(&self) -> bool {
        self.with_intercept
    }
    /// The interval method used (`Exact` or `Bootstrap`), `None` without
    /// inference.
    pub fn ci_method(&self) -> Option<TheilSenCiMethod> {
        self.ci_method
    }
    /// Number of bootstrap replicates that gave a finite fit (degenerate
    /// resamples are dropped); `None` unless the bootstrap was used.
    pub fn bootstrap_replicates(&self) -> Option<usize> {
        self.bootstrap_replicates
    }
}

impl FittedRegressor for FittedTheilSen {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);
        Col::from_fn(n, |i| {
            let mut v = intercept;
            for j in 0..p {
                let c = self.result.coefficients[j];
                // Aliased (redundant) columns carry a NaN coefficient.
                if !c.is_nan() {
                    v += x[(i, j)] * c;
                }
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
        // Theil–Sen confidence bands are not analytically tractable; return
        // point predictions only.
        PredictionResult::point_only(self.predict(x))
    }
}

/// Builder for [`TheilSenRegressor`].
#[derive(Debug, Clone, Default)]
pub struct TheilSenRegressorBuilder {
    inner: TheilSenRegressor,
}

impl TheilSenRegressorBuilder {
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.inner.with_intercept = include;
        self
    }
    pub fn max_subpopulation(mut self, value: usize) -> Self {
        self.inner.max_subpopulation = value;
        self
    }
    pub fn n_subsamples(mut self, value: usize) -> Self {
        self.inner.n_subsamples = Some(value);
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
    pub fn random_state(mut self, seed: u64) -> Self {
        self.inner.random_state = seed;
        self
    }
    /// Use the classical median of pairwise slopes for one-feature fits with
    /// an intercept (default `true`, matching R's Theil–Sen / Sen 1968).
    /// `false` uses sklearn's spatial-median-over-subsamples algorithm, which
    /// gives a different estimate in 1-D.
    pub fn univariate_pairwise(mut self, enabled: bool) -> Self {
        self.inner.univariate_pairwise = enabled;
        self
    }
    /// Compute coefficient intervals (default `false`); see
    /// [`TheilSenCiMethod`].
    pub fn compute_inference(mut self, enabled: bool) -> Self {
        self.inner.compute_inference = enabled;
        self
    }
    /// Level of the coefficient intervals, in (0, 1) (default 0.95).
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.inner.confidence_level = level;
        self
    }
    /// Interval method (default [`TheilSenCiMethod::Auto`]).
    pub fn ci_method(mut self, method: TheilSenCiMethod) -> Self {
        self.inner.ci_method = method;
        self
    }
    /// Number of bootstrap resamples, at least 2 (default 999).
    pub fn n_bootstrap(mut self, value: usize) -> Self {
        self.inner.n_bootstrap = value;
        self
    }
    pub fn build(self) -> TheilSenRegressor {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_pairwise_slope_counting_matches_explicit() {
        // Deterministic pseudo-random data with ties in x and duplicated points.
        let mut s: u64 = 7;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for &n in &[50usize, 301, 1000] {
            let x: Vec<f64> = (0..n).map(|_| (rnd() * 40.0).floor()).collect();
            let mut y: Vec<f64> = x.iter().map(|xi| 0.7 * xi + (rnd() - 0.5) * 10.0).collect();
            y[3] = y[2];
            let exact = median_pairwise_slope_with_limit(&x, &y, u64::MAX).unwrap();
            let counted = median_pairwise_slope_with_limit(&x, &y, 0).unwrap();
            assert!(
                (exact - counted).abs() < 1e-12,
                "n={n}: {exact} vs {counted}"
            );
            // continuous x as well
            let xc: Vec<f64> = (0..n).map(|_| rnd() * 10.0).collect();
            let yc: Vec<f64> = xc.iter().map(|xi| -1.3 * xi + rnd()).collect();
            let exact = median_pairwise_slope_with_limit(&xc, &yc, u64::MAX).unwrap();
            let counted = median_pairwise_slope_with_limit(&xc, &yc, 0).unwrap();
            assert!(
                (exact - counted).abs() < 1e-12,
                "n={n}: {exact} vs {counted}"
            );
        }
    }

    #[test]
    fn pairwise_slope_order_statistics_counting_matches_explicit() {
        // The counting path (used above EXPLICIT_PAIR_LIMIT pairs, e.g. for
        // the bounds of Sen's interval) agrees with sorting every slope.
        let mut s: u64 = 11;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let n = 400;
        let x: Vec<f64> = (0..n).map(|_| (rnd() * 60.0).floor()).collect();
        let y: Vec<f64> = x.iter().map(|xi| 0.3 * xi + (rnd() - 0.5) * 8.0).collect();
        let mut explicit = PairwiseSlopes::new(&x, &y, u64::MAX).unwrap();
        let mut counted = PairwiseSlopes::new(&x, &y, 0).unwrap();
        let m = explicit.n_pairs;
        for k in [1, 2, m / 7, m / 3, m / 2, m - 1, m] {
            let (a, b) = (explicit.kth(k), counted.kth(k));
            assert!((a - b).abs() < 1e-12, "k={k}: {a} vs {b}");
        }
    }

    #[test]
    fn kendall_variance_reduces_to_mann_kendall() {
        // No ties in x: the Mann–Kendall variance with ties in y.
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y = [1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 4.0, 8.0, 9.0];
        let want = (10.0 * 9.0 * 25.0 - 2.0 * 1.0 * 9.0 - 3.0 * 2.0 * 11.0) / 18.0;
        assert!((kendall_var_s(&x, &y) - want).abs() < 1e-12);
        // Symmetric in its arguments.
        assert_eq!(kendall_var_s(&x, &y), kendall_var_s(&y, &x));
    }

    #[test]
    fn univariate_exact_slope() {
        // For a clean line, Theil–Sen recovers slope/intercept exactly.
        let n = 20;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let y = Col::from_fn(n, |i| 3.0 + 2.0 * i as f64);

        let fitted = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();

        assert!((fitted.result.intercept.unwrap() - 3.0).abs() < 1e-10);
        assert!((fitted.result.coefficients[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn univariate_outliers_robust() {
        // Inject outliers — Theil–Sen should ignore them.
        let n = 21;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let mut y_vec: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
        y_vec[0] = 100.0;
        y_vec[20] = -100.0;
        let y = Col::from_fn(n, |i| y_vec[i]);

        let fitted = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();

        assert!((fitted.result.coefficients[0] - 0.5).abs() < 0.05);
    }

    #[test]
    fn enumerate_combinations_count() {
        let mut k = 0;
        for_each_combination(6, 3, |_| k += 1);
        assert_eq!(k as u128, n_choose_k(6, 3));
    }
}
