//! B-spline basis of one variable, as scikit-learn's `SplineTransformer`
//! (#309).

use super::{Basis, SplineError};
use crate::numeric::quantile::quantile7;
use faer::Mat;

/// Placement of the base knots computed from data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum KnotPlacement {
    /// Equally spaced from the minimum to the maximum (sklearn
    /// `knots="uniform"`).
    #[default]
    Uniform,
    /// At equally spaced quantiles (type 7, as `np.percentile`; sklearn
    /// `knots="quantile"`).
    Quantile,
}

/// Behaviour of a [`BSpline`] basis outside the base knots (sklearn
/// `extrapolation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SplineExtrapolation {
    /// The value at the nearest boundary of the `degree` splines that are
    /// non-zero there; the others are 0 (sklearn `"constant"`, the default).
    #[default]
    Constant,
    /// Linear continuation of the boundary splines (sklearn `"linear"`).
    Linear,
    /// The polynomial pieces of the first / last interval, continued
    /// (sklearn `"continue"`).
    Continue,
    /// Periodic splines with period `last knot − first knot`; `degree − 1`
    /// continuous derivatives at the period boundary (sklearn `"periodic"`).
    Periodic,
    /// Values outside the base knots are an error (sklearn `"error"`).
    Error,
}

/// B-spline basis of degree `degree` on fixed base knots, as
/// `sklearn.preprocessing.SplineTransformer` for one feature.
///
/// The `n_knots` base knots span the data; `degree` further knots are added
/// on each side at the spacing of the outermost base interval (periodic
/// splines wrap the base knots instead). The basis has `n_knots + degree − 1`
/// columns (`n_knots − 1` when periodic), one fewer without the bias column
/// (the last spline is dropped, sklearn `include_bias=False`). Keep the value
/// to rebuild the same basis on new data.
///
/// # Example
///
/// ```rust
/// use anofox_regression::splines::{BSpline, KnotPlacement, SplineExtrapolation};
///
/// let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
/// let spline = BSpline::from_data(&x, 5, 3, KnotPlacement::Uniform)
///     .unwrap()
///     .with_extrapolation(SplineExtrapolation::Linear);
/// let basis = spline.basis(&x, "x").unwrap();
/// assert_eq!(basis.columns.nrows(), 50);
/// assert_eq!(basis.n_columns(), 5 + 3 - 1);
/// // inside the knots the B-splines sum to one
/// let row_sum: f64 = (0..basis.n_columns()).map(|j| basis.columns[(10, j)]).sum();
/// assert!((row_sum - 1.0).abs() < 1e-12);
/// let new = spline.basis(&[-1.0, 7.5], "x").unwrap(); // same knots, extrapolated
/// assert_eq!(new.columns.nrows(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BSpline {
    base_knots: Vec<f64>,
    degree: usize,
    extrapolation: SplineExtrapolation,
    include_bias: bool,
}

impl BSpline {
    /// Basis on explicit base knots (at least 2, finite, strictly
    /// increasing; sklearn `knots=array`).
    pub fn new(base_knots: &[f64], degree: usize) -> Result<Self, SplineError> {
        if base_knots.iter().any(|k| !k.is_finite()) {
            return Err(SplineError::NonFiniteKnots);
        }
        if base_knots.len() < 2 {
            return Err(SplineError::TooFewBaseKnots(base_knots.len()));
        }
        if base_knots.windows(2).any(|w| w[1] <= w[0]) {
            return Err(SplineError::UnsortedKnots);
        }
        Ok(Self {
            base_knots: base_knots.to_vec(),
            degree,
            extrapolation: SplineExtrapolation::default(),
            include_bias: true,
        })
    }

    /// Basis with `n_knots` base knots placed on the finite values of `x`
    /// (sklearn `n_knots` / `knots="uniform" | "quantile"`). Fails when the
    /// knots are not distinct (e.g. constant `x` or heavy ties at quantile
    /// knots).
    pub fn from_data(
        x: &[f64],
        n_knots: usize,
        degree: usize,
        placement: KnotPlacement,
    ) -> Result<Self, SplineError> {
        let mut v: Vec<f64> = x.iter().copied().filter(|t| t.is_finite()).collect();
        if v.is_empty() {
            return Err(SplineError::TooFewObservations);
        }
        if n_knots < 2 {
            return Err(SplineError::TooFewBaseKnots(n_knots));
        }
        v.sort_by(f64::total_cmp);
        let step = |i: usize| i as f64 / (n_knots - 1) as f64;
        let knots: Vec<f64> = match placement {
            KnotPlacement::Uniform => {
                let (lo, hi) = (v[0], v[v.len() - 1]);
                (0..n_knots).map(|i| lo + (hi - lo) * step(i)).collect()
            }
            KnotPlacement::Quantile => (0..n_knots).map(|i| quantile7(&v, step(i))).collect(),
        };
        Self::new(&knots, degree)
    }

    /// Use another extrapolation (default [`SplineExtrapolation::Constant`]).
    pub fn with_extrapolation(mut self, extrapolation: SplineExtrapolation) -> Self {
        self.extrapolation = extrapolation;
        self
    }

    /// Keep (default) or drop the last spline (sklearn `include_bias`).
    /// Without it the columns no longer sum to one, so a model with an
    /// intercept stays identifiable.
    pub fn with_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// The base knots.
    pub fn base_knots(&self) -> &[f64] {
        &self.base_knots
    }

    /// The full knot vector: the base knots with `degree` knots added on
    /// each side (sklearn `bsplines_[i].t`).
    pub fn knots(&self) -> Vec<f64> {
        let (t, k) = (&self.base_knots, self.degree);
        let m = t.len();
        let (first, last) = (t[0], t[m - 1]);
        if self.extrapolation == SplineExtrapolation::Periodic {
            let period = last - first;
            let lo = m.saturating_sub(k + 1);
            let left = t[lo..m - 1].iter().map(|v| v - period);
            let right = t[1..(k + 1).min(m)].iter().map(|v| v + period);
            return left.chain(t.iter().copied()).chain(right).collect();
        }
        let (d_lo, d_hi) = (t[1] - first, last - t[m - 2]);
        let left = (0..k).map(|i| first - (k - i) as f64 * d_lo);
        let right = (1..=k).map(|i| last + i as f64 * d_hi);
        left.chain(t.iter().copied()).chain(right).collect()
    }

    /// The degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The extrapolation.
    pub fn extrapolation(&self) -> SplineExtrapolation {
        self.extrapolation
    }

    /// Whether the last spline is kept.
    pub fn include_bias(&self) -> bool {
        self.include_bias
    }

    /// Number of splines before dropping the bias column.
    fn n_splines(&self) -> usize {
        let m = self.base_knots.len();
        if self.extrapolation == SplineExtrapolation::Periodic {
            m - 1
        } else {
            m + self.degree - 1
        }
    }

    /// Number of basis columns.
    pub fn n_columns(&self) -> usize {
        self.n_splines() - usize::from(!self.include_bias)
    }

    /// The basis at `x`, columns `{name}_sp_0, {name}_sp_1, …` (sklearn's
    /// output order). A non-finite `x` gives a row of NaN. All columns are
    /// flagged nonlinear: a B-spline basis has no separate linear column.
    ///
    /// # Errors
    /// [`SplineError::PeriodicDegree`] for periodic splines with
    /// `degree ≥ n_knots`, [`SplineError::OutsideKnots`] for a value outside
    /// the base knots with [`SplineExtrapolation::Error`].
    pub fn basis(&self, x: &[f64], name: &str) -> Result<Basis, SplineError> {
        let periodic = self.extrapolation == SplineExtrapolation::Periodic;
        if periodic && self.degree >= self.base_knots.len() {
            return Err(SplineError::PeriodicDegree {
                degree: self.degree,
                n_knots: self.base_knots.len(),
            });
        }
        let eval = Evaluator::new(self.knots(), self.degree);
        let ns = self.n_splines();
        let mut columns = Mat::<f64>::zeros(x.len(), ns);
        for (i, &v) in x.iter().enumerate() {
            let row = self.row(&eval, v)?;
            for (j, value) in row.into_iter().enumerate() {
                columns[(i, j)] = value;
            }
        }
        let m = self.n_columns();
        let columns = Mat::from_fn(x.len(), m, |i, j| columns[(i, j)]);
        Ok(Basis {
            columns,
            names: (0..m).map(|j| format!("{name}_sp_{j}")).collect(),
            nonlinear: vec![true; m],
        })
    }

    /// One basis row (all `n_splines` columns).
    fn row(&self, eval: &Evaluator, x: f64) -> Result<Vec<f64>, SplineError> {
        let ns = self.n_splines();
        if !x.is_finite() {
            return Ok(vec![f64::NAN; ns]);
        }
        let (lo, hi) = eval.domain();
        let inside = lo <= x && x <= hi;
        match self.extrapolation {
            SplineExtrapolation::Periodic => Ok(self.periodic_row(eval, x)),
            SplineExtrapolation::Continue => Ok(eval.values(x, 0)),
            _ if inside => Ok(eval.values(x, 0)),
            SplineExtrapolation::Error => Err(SplineError::OutsideKnots(x)),
            SplineExtrapolation::Constant => Ok(self.constant_row(eval, x)),
            SplineExtrapolation::Linear => Ok(self.linear_row(eval, x)),
        }
    }

    /// `x` wrapped into the base period, then the first `degree` splines
    /// folded onto the last.
    fn periodic_row(&self, eval: &Evaluator, x: f64) -> Vec<f64> {
        let (lo, hi) = eval.domain();
        // The base knots are strictly increasing, so the period is positive.
        let xw = lo + (x - lo).rem_euclid(hi - lo);
        let full = eval.values(xw, 0);
        let ns = self.n_splines();
        let mut row = full[..ns].to_vec();
        for j in 0..self.degree.min(full.len() - ns) {
            row[j] += full[ns + j];
        }
        row
    }

    /// Below the knots the first `degree` splines keep their values at the
    /// lower boundary, above the last `degree` theirs at the upper one.
    fn constant_row(&self, eval: &Evaluator, x: f64) -> Vec<f64> {
        let (lo, hi) = eval.domain();
        let ns = self.n_splines();
        let k = self.degree.min(ns);
        let mut row = vec![0.0; ns];
        if x < lo {
            let f = eval.values(lo, 0);
            row[..k].copy_from_slice(&f[..k]);
        } else {
            let f = eval.values(hi, 0);
            row[ns - k..].copy_from_slice(&f[ns - k..]);
        }
        row
    }

    /// The boundary splines continued linearly from the nearer boundary
    /// (sklearn uses `max(degree, 2)` splines for degree ≤ 1).
    fn linear_row(&self, eval: &Evaluator, x: f64) -> Vec<f64> {
        let (lo, hi) = eval.domain();
        let ns = self.n_splines();
        let k = if self.degree <= 1 {
            self.degree + 1
        } else {
            self.degree
        }
        .min(ns);
        let edge = if x < lo { lo } else { hi };
        let (f, fp) = (eval.values(edge, 0), eval.values(edge, 1));
        let mut row = vec![0.0; ns];
        for j in 0..k {
            let c = if x < lo { j } else { ns - 1 - j };
            row[c] = f[c] + (x - edge) * fp[c];
        }
        row
    }
}

/// Evaluates all B-splines (or their first derivatives) of a knot vector.
struct Evaluator {
    t: Vec<f64>,
    k: usize,
}

impl Evaluator {
    fn new(t: Vec<f64>, k: usize) -> Self {
        Self { t, k }
    }

    /// Number of B-splines, `len(t) − k − 1`.
    fn n(&self) -> usize {
        self.t.len() - self.k - 1
    }

    /// `[t_k, t_n]`, where the splines sum to one.
    fn domain(&self) -> (f64, f64) {
        (self.t[self.k], self.t[self.n()])
    }

    /// Knot interval `l` (`t_l ≤ x < t_{l+1}`, `k ≤ l < n`) holding `x`; the
    /// last interval is closed on the right and values outside use the
    /// outermost interval, as scipy's `BSpline`.
    fn interval(&self, x: f64) -> usize {
        let (k, n) = (self.k, self.n());
        let pos = self.t[k + 1..n].partition_point(|&tj| tj <= x);
        k + pos
    }

    /// All `n` spline values (`nu = 0`) or first derivatives (`nu = 1`) at
    /// `x`.
    fn values(&self, x: f64, nu: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.n()];
        if nu > self.k {
            return out;
        }
        let l = self.interval(x);
        let local = self.local(x, l, self.k - nu);
        if nu == 0 {
            out[l - self.k..=l].copy_from_slice(&local);
            return out;
        }
        // B'_{i,k} = k (B_{i,k−1}/(t_{i+k} − t_i) − B_{i+1,k−1}/(t_{i+k+1} − t_{i+1}))
        let (t, k) = (&self.t, self.k);
        let lower = |i: usize| {
            let j = i as isize - (l as isize - (k as isize - 1));
            if (0..k as isize).contains(&j) {
                local[j as usize]
            } else {
                0.0
            }
        };
        for i in l - k..=l {
            let a = ratio(lower(i), t[i + k] - t[i]);
            let b = ratio(lower(i + 1), t[i + k + 1] - t[i + 1]);
            out[i] = k as f64 * (a - b);
        }
        out
    }

    /// The `p + 1` splines of degree `p` that can be non-zero on interval
    /// `l`, `B_{l−p}, …, B_l` (de Boor–Cox recursion; *The NURBS Book* A2.2).
    fn local(&self, x: f64, l: usize, p: usize) -> Vec<f64> {
        let t = &self.t;
        let mut n = vec![0.0; p + 1];
        let mut left = vec![0.0; p + 1];
        let mut right = vec![0.0; p + 1];
        n[0] = 1.0;
        for j in 1..=p {
            left[j] = x - t[l + 1 - j];
            right[j] = t[l + j] - x;
            let mut saved = 0.0;
            for r in 0..j {
                let temp = ratio(n[r], right[r + 1] + left[j - r]);
                n[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            n[j] = saved;
        }
        n
    }
}

fn ratio(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}
