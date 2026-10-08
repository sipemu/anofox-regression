//! Theil–Sen estimator: robust regression based on the spatial median of
//! per-subsample OLS coefficient vectors.
//!
//! Univariate case: slope is the median of all pairwise slopes
//! `(y_j - y_i) / (x_j - x_i)`; intercept is `median(y - slope * x)`. Has a
//! 29.3% asymptotic breakdown point and matches `mblm::mblm` (R) and
//! `sklearn.linear_model.TheilSenRegressor` numerically.
//!
//! Multivariate case: for each subsample of `n_subsamples = n_features + 1`
//! observations, solve the exact OLS system to get a coefficient vector,
//! then take the spatial (geometric) median of all those vectors using a
//! Vardi–Zhang modified Weiszfeld iteration. When
//! `C(n, n_subsamples) <= max_subpopulation` the algorithm enumerates every
//! subsample (matching sklearn's deterministic path); otherwise it draws
//! `max_subpopulation` random subsamples using the provided seed.
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

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::prelude::Solve;
use faer::{Col, Mat};

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

        // Univariate with intercept: the classical Theil–Sen estimator —
        // slope = median of all pairwise slopes (pairs with equal x skipped),
        // intercept = median(y − slope·x). sklearn instead applies its
        // spatial-median-over-subsamples algorithm to 1-D inputs too; that
        // behaviour stays available via `univariate_pairwise(false)`.
        if p == 1 && self.with_intercept && self.univariate_pairwise {
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
    if k != p_eff {
        // Caller guarantees k = p_eff for the standard configuration.
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

    // Square system: directly solve via LU. Skip if singular.
    let lu = xs.partial_piv_lu();
    let beta = lu.solve(&ys);
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
    let n = x.len();
    // Points in (x, y) order.
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

    if n_pairs <= explicit_limit {
        let mut slopes = Vec::with_capacity(n_pairs as usize);
        for a in 0..n {
            for b in (a + 1)..n {
                let dx = xs[b] - xs[a];
                if dx != 0.0 {
                    slopes.push((ys[b] - ys[a]) / dx);
                }
            }
        }
        return Some(median_in_place(&mut slopes));
    }

    // Counting path. G(s) = #pairs (x_a < x_b) with slope ≤ s
    //                     = #position pairs a < b with u_b ≤ u_a, u = y − s·x,
    // minus the duplicated points (whose u always tie).
    let count_le = |s: f64| -> u64 {
        let mut u: Vec<f64> = xs.iter().zip(&ys).map(|(xi, yi)| yi - s * xi).collect();
        let mut buf = vec![0.0; n];
        count_non_increasing_pairs(&mut u, &mut buf) - dup
    };
    let select = |k: u64| -> f64 {
        // k is 1-based. Bracket with bounds on |slope|.
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
        let (mut g_lo, mut g_hi) = (0_u64, n_pairs);
        let cap = (4 * n as u64).max(4096);
        for _ in 0..400 {
            if g_hi - g_lo <= cap {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                return hi; // interval exhausted: all remaining slopes ≈ hi
            }
            let g = count_le(mid);
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
        let u_lo: Vec<f64> = xs.iter().zip(&ys).map(|(xi, yi)| yi - lo * xi).collect();
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
    };

    let m = n_pairs;
    Some(if m % 2 == 1 {
        select(m.div_ceil(2))
    } else {
        0.5 * (select(m / 2) + select(m / 2 + 1))
    })
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
    }
}

/// A fitted Theil–Sen model.
#[derive(Debug, Clone)]
pub struct FittedTheilSen {
    result: RegressionResult,
    with_intercept: bool,
}

impl FittedTheilSen {
    /// Whether the model was fit with an intercept.
    pub fn with_intercept(&self) -> bool {
        self.with_intercept
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
