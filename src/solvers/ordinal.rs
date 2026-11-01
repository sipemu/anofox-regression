//! Ordinal regression: cumulative link models (RMS Ch 13, 15; `rms::orm`,
//! `rms::lrm`, `MASS::polr`).
//!
//! For an ordered response with distinct values `y₁ < … < y_k`, the model is
//!
//! ```text
//! P(Y ≥ y_{j+1} | x) = F(α_j + xᵀβ),   j = 1, …, k − 1,
//! ```
//!
//! with decreasing intercepts `α₁ > … > α_{k−1}` and a link `F`
//! ([`OrdinalLink`]); the logistic link is the proportional odds model. This
//! is the parametrisation of `rms::orm` / `lrm` (`MASS::polr` uses
//! `P(Y ≤ y_j) = F(ζ_j − xᵀβ)`, i.e. `ζ_j = −α_j` and the same β).
//!
//! The fit is maximum likelihood by Newton–Raphson with the analytic gradient
//! and observed information. Each observation involves at most two adjacent
//! intercepts, so the intercept block of the information is tridiagonal and
//! a Newton step costs O(k·p + n·p²) with an LDLᵀ factorisation of that block
//! and the Schur complement for the slopes. A continuous response, with one
//! intercept per distinct value (`orm`'s semiparametric regression, RMS
//! Ch 15), is therefore as cheap as a few categories.

use crate::solvers::traits::RegressionError;
use faer::linalg::solvers::Solve;
use faer::{Col, Mat, Side};
use statrs::distribution::{ChiSquared, Continuous, ContinuousCDF, Normal};
use std::f64::consts::PI;

/// The link (cumulative distribution `F`) of a cumulative link model, named
/// as the `family` of `rms::orm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrdinalLink {
    /// `F(x) = 1 / (1 + e⁻ˣ)`: proportional odds (`lrm`, `polr`).
    #[default]
    Logistic,
    /// `F = Φ`, the standard normal distribution.
    Probit,
    /// `F(x) = exp(−exp(−x))` (rms `loglog`).
    LogLog,
    /// `F(x) = 1 − exp(−exp(x))` (rms `cloglog`): proportional hazards in the
    /// discrete-time sense.
    CLogLog,
    /// `F(x) = 1/2 + atan(x)/π`.
    Cauchit,
}

fn std_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal")
}

impl OrdinalLink {
    /// `F(x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            OrdinalLink::Logistic => 1.0 / (1.0 + (-x).exp()),
            OrdinalLink::Probit => std_normal().cdf(x),
            OrdinalLink::LogLog => (-(-x).exp()).exp(),
            OrdinalLink::CLogLog => -(-x.exp()).exp_m1(),
            OrdinalLink::Cauchit => 0.5 + x.atan() / PI,
        }
    }

    /// `1 − F(x)`, accurate in the upper tail.
    pub fn sf(&self, x: f64) -> f64 {
        match self {
            OrdinalLink::Logistic => 1.0 / (1.0 + x.exp()),
            OrdinalLink::Probit => std_normal().cdf(-x),
            OrdinalLink::LogLog => -(-(-x).exp()).exp_m1(),
            OrdinalLink::CLogLog => (-x.exp()).exp(),
            OrdinalLink::Cauchit => 0.5 - x.atan() / PI,
        }
    }

    /// The density `f = F'`.
    pub fn pdf(&self, x: f64) -> f64 {
        match self {
            OrdinalLink::Logistic => {
                let e = (-x.abs()).exp();
                e / ((1.0 + e) * (1.0 + e))
            }
            OrdinalLink::Probit => std_normal().pdf(x),
            OrdinalLink::LogLog => (-x - (-x).exp()).exp(),
            OrdinalLink::CLogLog => (x - x.exp()).exp(),
            OrdinalLink::Cauchit => 1.0 / (PI * (1.0 + x * x)),
        }
    }

    /// `f'`.
    fn dpdf(&self, x: f64) -> f64 {
        let f = self.pdf(x);
        match self {
            OrdinalLink::Logistic => f * (1.0 - 2.0 * self.cdf(x)),
            OrdinalLink::Probit => -x * f,
            OrdinalLink::LogLog => f * ((-x).exp() - 1.0),
            OrdinalLink::CLogLog => f * (1.0 - x.exp()),
            OrdinalLink::Cauchit => -2.0 * x * PI * f * f,
        }
    }

    /// `F⁻¹(p)`.
    pub fn quantile(&self, p: f64) -> f64 {
        match self {
            OrdinalLink::Logistic => (p / (1.0 - p)).ln(),
            OrdinalLink::Probit => std_normal().inverse_cdf(p),
            OrdinalLink::LogLog => -(-p.ln()).ln(),
            OrdinalLink::CLogLog => (-(1.0 - p).ln()).ln(),
            OrdinalLink::Cauchit => (PI * (p - 0.5)).tan(),
        }
    }

    /// `F(a) − F(b)` for `a ≥ b`, from the tail that keeps precision.
    fn prob_between(&self, a: f64, b: f64) -> f64 {
        if b > 0.0 {
            self.sf(b) - self.sf(a)
        } else {
            self.cdf(a) - self.cdf(b)
        }
    }
}

/// How [`FittedOrdinal::predict_quantile`] turns the predicted distribution
/// into a quantile (`rms::Quantile.orm`'s `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrdinalQuantileMethod {
    /// Linear interpolation of the predicted CDF between the response values
    /// (rms default).
    #[default]
    Interpolated,
    /// The smallest response value whose predicted CDF reaches `q`.
    Discrete,
}

/// Cumulative link model regressor.
///
/// ```
/// use anofox_regression::solvers::{OrdinalLink, OrdinalRegressor};
/// use faer::{Col, Mat};
///
/// let x = Mat::from_fn(60, 1, |i, _| (i % 12) as f64 / 4.0);
/// let y = Col::from_fn(60, |i| ((i % 12) / 4 + (i % 5 == 0) as usize) as f64);
/// let fit = OrdinalRegressor::builder()
///     .link(OrdinalLink::Logistic)
///     .build()
///     .fit(&x, &y)
///     .unwrap();
/// assert!(fit.coefficients()[0] > 0.0);
/// assert_eq!(fit.intercepts().nrows(), fit.levels().len() - 1);
/// ```
#[derive(Debug, Clone)]
pub struct OrdinalRegressor {
    link: OrdinalLink,
    weights: Option<Col<f64>>,
    offset: Option<Col<f64>>,
    penalty: Option<OrdinalPenalty>,
    max_iterations: usize,
    tolerance: f64,
}

/// Builder for [`OrdinalRegressor`].
#[derive(Debug, Clone)]
pub struct OrdinalRegressorBuilder {
    inner: OrdinalRegressor,
}

impl Default for OrdinalRegressorBuilder {
    fn default() -> Self {
        Self {
            inner: OrdinalRegressor {
                link: OrdinalLink::Logistic,
                weights: None,
                offset: None,
                penalty: None,
                max_iterations: 100,
                tolerance: 1e-10,
            },
        }
    }
}

impl OrdinalRegressorBuilder {
    /// The link (default [`OrdinalLink::Logistic`]).
    pub fn link(mut self, link: OrdinalLink) -> Self {
        self.inner.link = link;
        self
    }

    /// Case (frequency) weights.
    pub fn weights(mut self, weights: Col<f64>) -> Self {
        self.inner.weights = Some(weights);
        self
    }

    /// An offset added to the linear predictor.
    pub fn offset(mut self, offset: Col<f64>) -> Self {
        self.inner.offset = Some(offset);
        self
    }

    /// Quadratic penalty `λ · diag(var(x_j))` on the slopes (`orm` / `lrm`'s
    /// `penalty = λ` for simple numeric predictors; sample variances).
    pub fn penalty(mut self, lambda: f64) -> Self {
        self.inner.penalty = Some(OrdinalPenalty::Scale(lambda));
        self
    }

    /// Quadratic penalty `½ βᵀPβ` with a given `p × p` matrix `P`
    /// (`penalty.matrix`).
    pub fn penalty_matrix(mut self, penalty: Mat<f64>) -> Self {
        self.inner.penalty = Some(OrdinalPenalty::Matrix(penalty));
        self
    }

    /// Maximum Newton iterations (default 100).
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.inner.max_iterations = n;
        self
    }

    /// Convergence tolerance on the largest absolute score (default 1e-10;
    /// the score is scaled by the total weight).
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.inner.tolerance = tol;
        self
    }

    /// Build the regressor.
    pub fn build(self) -> OrdinalRegressor {
        self.inner
    }
}

/// A quadratic penalty on the slopes.
#[derive(Debug, Clone)]
enum OrdinalPenalty {
    /// `λ · diag(var(x_j))`.
    Scale(f64),
    /// A given matrix.
    Matrix(Mat<f64>),
}

/// The observed information `[[A, B], [Bᵀ, D]]` at the estimate: `A` the
/// tridiagonal intercept block, `B` intercepts × slopes, `D` the slopes.
#[derive(Debug, Clone)]
struct Information {
    a_diag: Vec<f64>,
    a_off: Vec<f64>,
    b: Mat<f64>,
    d: Mat<f64>,
}

/// LDLᵀ factor of the tridiagonal block and the Schur complement inverse.
struct InformationFactor<'a> {
    info: &'a Information,
    d: Vec<f64>,
    l: Vec<f64>,
    ainv_b: Mat<f64>,
    s_inv: Mat<f64>,
}

impl<'a> InformationFactor<'a> {
    fn new(info: &'a Information) -> Option<Self> {
        let m = info.a_diag.len();
        let mut d = vec![0.0; m];
        let mut l = vec![0.0; m.saturating_sub(1)];
        for i in 0..m {
            d[i] = info.a_diag[i]
                - if i > 0 {
                    l[i - 1] * info.a_off[i - 1]
                } else {
                    0.0
                };
            if d[i].is_nan() || d[i] <= 0.0 || !d[i].is_finite() {
                return None;
            }
            if i + 1 < m {
                l[i] = info.a_off[i] / d[i];
            }
        }
        let mut factor = Self {
            info,
            d,
            l,
            ainv_b: Mat::zeros(0, 0),
            s_inv: Mat::zeros(0, 0),
        };
        let p = info.d.nrows();
        let mut ainv_b = Mat::zeros(m, p);
        for c in 0..p {
            let col: Vec<f64> = (0..m).map(|r| info.b[(r, c)]).collect();
            let sol = factor.solve_a(&col);
            for r in 0..m {
                ainv_b[(r, c)] = sol[r];
            }
        }
        let s = &info.d - info.b.transpose() * &ainv_b;
        let s_inv = if p == 0 {
            Mat::zeros(0, 0)
        } else {
            let llt = s.llt(Side::Lower).ok()?;
            llt.solve(Mat::<f64>::identity(p, p))
        };
        factor.ainv_b = ainv_b;
        factor.s_inv = s_inv;
        Some(factor)
    }

    /// `A⁻¹ r` for the tridiagonal block.
    fn solve_a(&self, r: &[f64]) -> Vec<f64> {
        let m = self.d.len();
        let mut z = r.to_vec();
        for i in 1..m {
            z[i] -= self.l[i - 1] * z[i - 1];
        }
        for (zi, di) in z.iter_mut().zip(&self.d) {
            *zi /= di;
        }
        for i in (0..m.saturating_sub(1)).rev() {
            z[i] -= self.l[i] * z[i + 1];
        }
        z
    }

    /// `I⁻¹ (g_α, g_β)`.
    fn solve(&self, g_a: &[f64], g_b: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = g_a.len();
        let p = g_b.len();
        let ainv_ga = self.solve_a(g_a);
        // δβ = S⁻¹ (g_β − Bᵀ A⁻¹ g_α)
        let rhs: Vec<f64> = (0..p)
            .map(|c| {
                g_b[c]
                    - (0..m)
                        .map(|r| self.info.b[(r, c)] * ainv_ga[r])
                        .sum::<f64>()
            })
            .collect();
        let db: Vec<f64> = (0..p)
            .map(|r| (0..p).map(|c| self.s_inv[(r, c)] * rhs[c]).sum())
            .collect();
        // δα = A⁻¹ g_α − (A⁻¹ B) δβ
        let da: Vec<f64> = (0..m)
            .map(|r| ainv_ga[r] - (0..p).map(|c| self.ainv_b[(r, c)] * db[c]).sum::<f64>())
            .collect();
        (da, db)
    }
}

/// A fitted cumulative link model.
#[derive(Debug, Clone)]
pub struct FittedOrdinal {
    link: OrdinalLink,
    levels: Vec<f64>,
    counts: Vec<f64>,
    intercepts: Col<f64>,
    coefficients: Col<f64>,
    log_likelihood: f64,
    penalized_log_likelihood: f64,
    null_log_likelihood: f64,
    iterations: usize,
    converged: bool,
    n_observations: usize,
    information: Information,
    weights: Option<Col<f64>>,
    offset: Option<Col<f64>>,
    penalty: Option<Mat<f64>>,
}

/// Data shared by the likelihood evaluations.
struct Problem<'a> {
    link: OrdinalLink,
    x: &'a Mat<f64>,
    category: Vec<usize>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    k: usize,
    penalty: Option<Mat<f64>>,
}

impl Problem<'_> {
    /// `½ βᵀPβ`.
    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.penalty.as_ref().map_or(0.0, |pm| {
            let p = beta.len();
            0.5 * (0..p)
                .map(|r| (0..p).map(|c| beta[r] * pm[(r, c)] * beta[c]).sum::<f64>())
                .sum::<f64>()
        })
    }

    /// The maximised objective: the log-likelihood minus the penalty.
    fn objective(&self, alpha: &[f64], beta: &[f64]) -> f64 {
        self.log_likelihood(alpha, beta) - self.penalty_value(beta)
    }

    fn eta(&self, beta: &[f64]) -> Vec<f64> {
        let (n, p) = (self.x.nrows(), self.x.ncols());
        (0..n)
            .map(|i| self.offset[i] + (0..p).map(|j| self.x[(i, j)] * beta[j]).sum::<f64>())
            .collect()
    }

    /// The two cumulative-probability arguments of observation `i`:
    /// `(α_{c−1} + η, α_c + η)` with ±∞ at the ends.
    fn bounds(&self, alpha: &[f64], c: usize, eta: f64) -> (f64, f64) {
        let upper = if c == 0 {
            f64::INFINITY
        } else {
            alpha[c - 1] + eta
        };
        let lower = if c + 1 == self.k {
            f64::NEG_INFINITY
        } else {
            alpha[c] + eta
        };
        (upper, lower)
    }

    fn log_likelihood(&self, alpha: &[f64], beta: &[f64]) -> f64 {
        if alpha.windows(2).any(|w| w[0] <= w[1]) {
            return f64::NEG_INFINITY;
        }
        let eta = self.eta(beta);
        let mut ll = 0.0;
        for (i, &c) in self.category.iter().enumerate() {
            let (a, b) = self.bounds(alpha, c, eta[i]);
            let p = self.link.prob_between(a, b);
            if p.is_nan() || p <= 0.0 {
                return f64::NEG_INFINITY;
            }
            ll += self.weights[i] * p.ln();
        }
        ll
    }

    /// Score and observed information at `(alpha, beta)`.
    #[allow(clippy::needless_range_loop)]
    fn score_information(&self, alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<f64>, Information) {
        let m = self.k - 1;
        let p = self.x.ncols();
        let eta = self.eta(beta);
        let mut g_a = vec![0.0; m];
        let mut g_b = vec![0.0; p];
        let mut info = Information {
            a_diag: vec![0.0; m],
            a_off: vec![0.0; m.saturating_sub(1)],
            b: Mat::zeros(m, p),
            d: Mat::zeros(p, p),
        };
        let link = self.link;
        let finite = |v: f64, f: &dyn Fn(f64) -> f64| if v.is_finite() { f(v) } else { 0.0 };
        for (i, &c) in self.category.iter().enumerate() {
            let w = self.weights[i];
            let (a, b) = self.bounds(alpha, c, eta[i]);
            let pr = link.prob_between(a, b);
            let (fa, fb) = (finite(a, &|v| link.pdf(v)), finite(b, &|v| link.pdf(v)));
            let (dfa, dfb) = (finite(a, &|v| link.dpdf(v)), finite(b, &|v| link.dpdf(v)));
            let (ia, ib) = (c.checked_sub(1), (c + 1 < self.k).then_some(c));
            let df = fa - fb;
            // second derivatives of log p (negated below)
            let haa = dfa / pr - fa * fa / pr / pr;
            let hbb = -dfb / pr - fb * fb / pr / pr;
            let hab = fa * fb / pr / pr;
            let hax = dfa / pr - fa * df / pr / pr;
            let hbx = -dfb / pr + fb * df / pr / pr;
            let hxx = (dfa - dfb) / pr - df * df / pr / pr;
            if let Some(ja) = ia {
                g_a[ja] += w * fa / pr;
                info.a_diag[ja] -= w * haa;
            }
            if let Some(jb) = ib {
                g_a[jb] -= w * fb / pr;
                info.a_diag[jb] -= w * hbb;
            }
            if let (Some(ja), Some(_)) = (ia, ib) {
                info.a_off[ja] -= w * hab;
            }
            for r in 0..p {
                let xr = self.x[(i, r)];
                g_b[r] += w * xr * df / pr;
                if let Some(ja) = ia {
                    info.b[(ja, r)] -= w * hax * xr;
                }
                if let Some(jb) = ib {
                    info.b[(jb, r)] -= w * hbx * xr;
                }
                for s in 0..=r {
                    let v = w * hxx * xr * self.x[(i, s)];
                    info.d[(r, s)] -= v;
                }
            }
        }
        for r in 0..p {
            for s in 0..r {
                info.d[(s, r)] = info.d[(r, s)];
            }
        }
        if let Some(pm) = &self.penalty {
            for r in 0..p {
                g_b[r] -= (0..p).map(|c| pm[(r, c)] * beta[c]).sum::<f64>();
                for c in 0..p {
                    info.d[(r, c)] += pm[(r, c)];
                }
            }
        }
        (g_a, g_b, info)
    }
}

impl OrdinalRegressor {
    /// Builder with the defaults (logistic link, no weights or offset).
    pub fn builder() -> OrdinalRegressorBuilder {
        OrdinalRegressorBuilder::default()
    }

    /// Fit the model. `x` holds the predictors (no intercept column; the
    /// intercepts are the cut-points) and `y` the ordered response; its
    /// distinct values are the categories.
    pub fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedOrdinal, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        if y.nrows() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        for v in [&self.weights, &self.offset].into_iter().flatten() {
            if v.nrows() != n {
                return Err(RegressionError::DimensionMismatch {
                    x_rows: n,
                    y_len: v.nrows(),
                });
            }
        }
        if y.iter()
            .chain(x.col_iter().flat_map(|c| c.iter()))
            .any(|v| !v.is_finite())
        {
            return Err(RegressionError::NumericalError(
                "non-finite values in x or y".to_string(),
            ));
        }
        let weights: Vec<f64> = match &self.weights {
            Some(w) => w.iter().copied().collect(),
            None => vec![1.0; n],
        };
        if weights
            .iter()
            .any(|&w| w.is_nan() || w < 0.0 || !w.is_finite())
        {
            return Err(RegressionError::NumericalError(
                "weights must be finite and non-negative".to_string(),
            ));
        }
        let offset: Vec<f64> = match &self.offset {
            Some(o) => o.iter().copied().collect(),
            None => vec![0.0; n],
        };

        let mut levels: Vec<f64> = y.iter().copied().collect();
        levels.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        levels.dedup();
        let k = levels.len();
        if k < 2 {
            return Err(RegressionError::NumericalError(
                "an ordinal response needs at least two distinct values".to_string(),
            ));
        }
        let category: Vec<usize> = y
            .iter()
            .map(|v| levels.partition_point(|l| l < v))
            .collect();
        let mut counts = vec![0.0; k];
        for (i, &c) in category.iter().enumerate() {
            counts[c] += weights[i];
        }
        let total: f64 = counts.iter().sum();
        if counts.iter().any(|&c| c <= 0.0) {
            return Err(RegressionError::NumericalError(
                "every response value needs positive total weight".to_string(),
            ));
        }

        let penalty = match &self.penalty {
            None => None,
            Some(OrdinalPenalty::Matrix(pm)) => {
                if pm.nrows() != p || pm.ncols() != p {
                    return Err(RegressionError::NumericalError(format!(
                        "the penalty matrix is {}×{}, expected {p}×{p}",
                        pm.nrows(),
                        pm.ncols()
                    )));
                }
                Some(pm.clone())
            }
            Some(OrdinalPenalty::Scale(lambda)) => Some(Mat::from_fn(p, p, |r, c| {
                if r != c || n < 2 {
                    return 0.0;
                }
                let mean = (0..n).map(|i| x[(i, r)]).sum::<f64>() / n as f64;
                let ss: f64 = (0..n).map(|i| (x[(i, r)] - mean).powi(2)).sum();
                lambda * ss / (n - 1) as f64
            })),
        };
        let problem = Problem {
            link: self.link,
            x,
            category,
            weights,
            offset,
            k,
            penalty,
        };

        // Start: the intercept-only MLE, α_j = F⁻¹(P̂(Y ≥ y_{j+1})), β = 0.
        let mut alpha: Vec<f64> = (1..k)
            .map(|j| self.link.quantile(counts[j..].iter().sum::<f64>() / total))
            .collect();
        let mut beta = vec![0.0; p];
        // The intercept-only model: closed form, or (with an offset) its own
        // fit, as rms's model LR χ² compares against intercepts + offset.
        let null_ll: f64 = if self.offset.is_some() && p > 0 {
            let intercepts_only = OrdinalRegressor {
                penalty: None,
                ..self.clone()
            };
            intercepts_only.fit(&Mat::zeros(n, 0), y)?.log_likelihood
        } else {
            counts.iter().map(|&c| c * (c / total).ln()).sum()
        };
        let mut ll = problem.objective(&alpha, &beta);
        if !ll.is_finite() {
            return Err(RegressionError::NumericalError(
                "the starting values have zero likelihood (check the offset)".to_string(),
            ));
        }

        let mut converged = false;
        let mut iterations = 0;
        let (mut g_a, mut g_b, mut info) = problem.score_information(&alpha, &beta);
        while iterations < self.max_iterations {
            let max_score = g_a.iter().chain(&g_b).fold(0.0_f64, |m, g| m.max(g.abs()));
            if max_score <= self.tolerance * total.max(1.0) {
                converged = true;
                break;
            }
            iterations += 1;
            let factor = InformationFactor::new(&info).ok_or_else(|| {
                RegressionError::NumericalError(
                    "the information matrix is not positive definite".to_string(),
                )
            })?;
            let (da, db) = factor.solve(&g_a, &g_b);
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..50 {
                let alpha_new: Vec<f64> =
                    alpha.iter().zip(&da).map(|(a, d)| a + step * d).collect();
                let beta_new: Vec<f64> = beta.iter().zip(&db).map(|(b, d)| b + step * d).collect();
                let ll_new = problem.objective(&alpha_new, &beta_new);
                if ll_new.is_finite() && ll_new >= ll - 1e-12 * ll.abs().max(1.0) {
                    alpha = alpha_new;
                    beta = beta_new;
                    ll = ll_new;
                    accepted = true;
                    break;
                }
                step /= 2.0;
            }
            (g_a, g_b, info) = problem.score_information(&alpha, &beta);
            if !accepted {
                break;
            }
        }
        if !converged {
            let max_score = g_a.iter().chain(&g_b).fold(0.0_f64, |m, g| m.max(g.abs()));
            converged = max_score <= self.tolerance * total.max(1.0);
        }
        if InformationFactor::new(&info).is_none() {
            return Err(RegressionError::NumericalError(
                "the information matrix is not positive definite at the estimate".to_string(),
            ));
        }

        Ok(FittedOrdinal {
            link: self.link,
            levels,
            counts,
            intercepts: Col::from_fn(k - 1, |j| alpha[j]),
            coefficients: Col::from_fn(p, |j| beta[j]),
            log_likelihood: problem.log_likelihood(&alpha, &beta),
            penalized_log_likelihood: ll,
            null_log_likelihood: null_ll,
            iterations,
            converged,
            n_observations: n,
            information: info,
            weights: self.weights.clone(),
            offset: self.offset.clone(),
            penalty: problem.penalty,
        })
    }
}

impl FittedOrdinal {
    /// The link.
    pub fn link(&self) -> OrdinalLink {
        self.link
    }

    /// The distinct response values `y₁ < … < y_k`.
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// Total weight of each response value.
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// Intercepts `α_j` of `P(Y ≥ y_{j+1})`, `j = 1, …, k − 1` (decreasing).
    pub fn intercepts(&self) -> &Col<f64> {
        &self.intercepts
    }

    /// Slopes β.
    pub fn coefficients(&self) -> &Col<f64> {
        &self.coefficients
    }

    /// Log-likelihood at the estimate (unpenalized, also for a penalized
    /// fit).
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// The maximised objective `ℓ − ½ βᵀPβ` (the log-likelihood without a
    /// penalty); `orm`'s model deviance is `−2` times it.
    pub fn penalized_log_likelihood(&self) -> f64 {
        self.penalized_log_likelihood
    }

    /// The penalty matrix `P` of a penalized fit.
    pub fn penalty_matrix(&self) -> Option<&Mat<f64>> {
        self.penalty.as_ref()
    }

    /// Effective degrees of freedom of the slopes, `diag(V I_unpenalized)`
    /// with `V` the penalized covariance (`orm`'s `effective.df.diagonal`
    /// without the intercepts, which are 1): `1 − (V_ββ P)_jj`; 1 each
    /// without a penalty.
    pub fn effective_df_diagonal(&self) -> Col<f64> {
        let p = self.coefficients.nrows();
        match &self.penalty {
            None => Col::from_fn(p, |_| 1.0),
            Some(pm) => {
                let vp = self.vcov_slopes() * pm;
                Col::from_fn(p, |j| 1.0 - vp[(j, j)])
            }
        }
    }

    /// Total effective degrees of freedom of the slopes (the d.f. of
    /// [`Self::lr_chisq`] for a penalized fit).
    pub fn effective_df(&self) -> f64 {
        self.effective_df_diagonal().iter().sum()
    }

    /// Log-likelihood of the intercept-only model (with the offset, if any).
    pub fn null_log_likelihood(&self) -> f64 {
        self.null_log_likelihood
    }

    /// Model likelihood-ratio χ² against the intercept-only model.
    pub fn lr_chisq(&self) -> f64 {
        2.0 * (self.log_likelihood - self.null_log_likelihood)
    }

    /// The number of slopes (the d.f. of [`Self::lr_chisq`] without a
    /// penalty; see [`Self::effective_df`]).
    pub fn lr_df(&self) -> usize {
        self.coefficients.nrows()
    }

    /// p-value of [`Self::lr_chisq`] on [`Self::effective_df`] d.f.
    pub fn lr_p_value(&self) -> f64 {
        let df = self.effective_df();
        if df.is_nan() || df <= 0.0 {
            return f64::NAN;
        }
        ChiSquared::new(df).map_or(f64::NAN, |d| d.sf(self.lr_chisq()))
    }

    /// AIC `−2ℓ + 2(k − 1 + d.f.)` with the effective d.f. of the slopes.
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * (self.intercepts.nrows() as f64 + self.effective_df())
    }

    /// Newton iterations used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether the score reached the tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Number of observations.
    pub fn n_observations(&self) -> usize {
        self.n_observations
    }

    fn factor(&self) -> InformationFactor<'_> {
        InformationFactor::new(&self.information).expect("positive definite at the fit")
    }

    /// Covariance of the slopes (inverse observed information, penalized for a
    /// penalized fit: `orm`'s `var.penalty = "simple"`), without
    /// forming the intercept block: O(k·p²).
    pub fn vcov_slopes(&self) -> Mat<f64> {
        self.factor().s_inv.clone()
    }

    /// Standard errors of the slopes.
    pub fn std_errors(&self) -> Col<f64> {
        let v = self.vcov_slopes();
        Col::from_fn(v.nrows(), |j| v[(j, j)].sqrt())
    }

    /// Full covariance of `(α, β)`, intercepts first (dense, O(k²) memory:
    /// for a continuous response prefer [`Self::vcov_slopes`]).
    pub fn vcov(&self) -> Mat<f64> {
        let factor = self.factor();
        let (m, p) = (self.intercepts.nrows(), self.coefficients.nrows());
        let mut v = Mat::zeros(m + p, m + p);
        for c in 0..m + p {
            let (ga, gb): (Vec<f64>, Vec<f64>) = (
                (0..m).map(|r| (r == c) as u8 as f64).collect(),
                (0..p).map(|r| (r + m == c) as u8 as f64).collect(),
            );
            let (da, db) = factor.solve(&ga, &gb);
            for r in 0..m {
                v[(r, c)] = da[r];
            }
            for r in 0..p {
                v[(m + r, c)] = db[r];
            }
        }
        v
    }

    /// Full covariance of a penalized fit by the sandwich `V I V` with `V`
    /// the penalized inverse information and `I` the unpenalized information
    /// (`orm` / `lrm`'s `var.penalty = "sandwich"`): `V − V_{·β} P V_{β·}`.
    /// Equal to [`Self::vcov`] without a penalty. Dense, O(k²) memory.
    pub fn vcov_sandwich(&self) -> Mat<f64> {
        let v = self.vcov();
        let Some(pm) = &self.penalty else {
            return v;
        };
        let (m, p) = (self.intercepts.nrows(), self.coefficients.nrows());
        let d = m + p;
        let v_b = Mat::from_fn(d, p, |r, c| v[(r, m + c)]);
        let correction = &v_b * pm * v_b.transpose();
        Mat::from_fn(d, d, |r, c| v[(r, c)] - correction[(r, c)])
    }

    /// Per-observation score contributions `∂ wᵢ log P(Yᵢ = yᵢ) / ∂(α, β)`
    /// at the estimate: `n × (k − 1 + p)`, intercepts first (the penalty
    /// term is not included). `x` and `y` must be the data of the fit; its
    /// weights and offset are used.
    pub fn score_contributions(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<Mat<f64>, crate::inference::ClusterError> {
        use crate::inference::ClusterError;
        let (n, p, m) = (
            x.nrows(),
            self.coefficients.nrows(),
            self.intercepts.nrows(),
        );
        if y.nrows() != n || x.ncols() != p {
            return Err(ClusterError::Dimension(format!(
                "x is {n}x{}, y has {} rows, the model {p} slopes",
                x.ncols(),
                y.nrows()
            )));
        }
        let lp = self.linear_predictor(x);
        let link = self.link;
        let finite = |v: f64| if v.is_finite() { link.pdf(v) } else { 0.0 };
        let mut u = Mat::zeros(n, m + p);
        for i in 0..n {
            let c = self.levels.iter().position(|&l| l == y[i]).ok_or_else(|| {
                ClusterError::Dimension(format!("y[{i}] is not a level of the fit"))
            })?;
            let eta = lp[i] + self.offset.as_ref().map_or(0.0, |o| o[i]);
            let w = self.weights.as_ref().map_or(1.0, |w| w[i]);
            let a = if c == 0 {
                f64::INFINITY
            } else {
                self.intercepts[c - 1] + eta
            };
            let b = if c + 1 == m + 1 {
                f64::NEG_INFINITY
            } else {
                self.intercepts[c] + eta
            };
            let pr = link.prob_between(a, b);
            let (fa, fb) = (finite(a), finite(b));
            if c > 0 {
                u[(i, c - 1)] = w * fa / pr;
            }
            if c < m {
                u[(i, c)] = -w * fb / pr;
            }
            for r in 0..p {
                u[(i, m + r)] = w * x[(i, r)] * (fa - fb) / pr;
            }
        }
        Ok(u)
    }

    /// Cluster-robust covariance of `(α, β)`, intercepts first: the sandwich
    /// `V (Σ_g u_g u_gᵀ) V` with `V` = [`Self::vcov`] and `u_g` the summed
    /// [`Self::score_contributions`] of cluster `g`.
    /// [`ClusterOptions::robcov`](crate::inference::ClusterOptions::robcov)
    /// gives `rms::robcov(fit, cluster)` for `orm` / `lrm`.
    pub fn cluster_vcov(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        clusters: &[&[usize]],
        options: &crate::inference::ClusterOptions,
    ) -> Result<Mat<f64>, crate::inference::ClusterError> {
        let scores = self.score_contributions(x, y)?;
        crate::inference::cluster_sandwich(&self.vcov(), &scores, clusters, options)
    }

    /// `xᵀβ` for each row of `x` (no intercept, no offset).
    pub fn linear_predictor(&self, x: &Mat<f64>) -> Col<f64> {
        x * &self.coefficients
    }

    /// `P(Y ≥ y_{j+1} | x)` for `j = 1, …, k − 1`: one row per row of `x`.
    pub fn exceedance_probabilities(&self, x: &Mat<f64>) -> Mat<f64> {
        let lp = self.linear_predictor(x);
        let m = self.intercepts.nrows();
        Mat::from_fn(x.nrows(), m, |i, j| {
            self.link.cdf(self.intercepts[j] + lp[i])
        })
    }

    /// `P(Y = y_j | x)` for `j = 1, …, k`.
    pub fn category_probabilities(&self, x: &Mat<f64>) -> Mat<f64> {
        let lp = self.linear_predictor(x);
        let k = self.levels.len();
        Mat::from_fn(x.nrows(), k, |i, j| {
            let a = if j == 0 {
                f64::INFINITY
            } else {
                self.intercepts[j - 1] + lp[i]
            };
            let b = if j + 1 == k {
                f64::NEG_INFINITY
            } else {
                self.intercepts[j] + lp[i]
            };
            self.link.prob_between(a, b)
        })
    }

    /// Predicted mean `Σ_j y_j P(Y = y_j | x)` (`rms::Mean.orm`).
    pub fn predict_mean(&self, x: &Mat<f64>) -> Col<f64> {
        let probs = self.category_probabilities(x);
        Col::from_fn(x.nrows(), |i| {
            (0..self.levels.len())
                .map(|j| probs[(i, j)] * self.levels[j])
                .sum()
        })
    }

    /// Predicted mean with its delta-method standard error (`Mean.orm` with
    /// `conf.int`): `∂mean/∂α_j = f(α_j + xᵀβ)(y_{j+1} − y_j)` and
    /// `∂mean/∂β = x Σ_j ∂mean/∂α_j`.
    pub fn predict_mean_with_se(&self, x: &Mat<f64>) -> (Col<f64>, Col<f64>) {
        let mean = self.predict_mean(x);
        let lp = self.linear_predictor(x);
        let factor = self.factor();
        let m = self.intercepts.nrows();
        let p = self.coefficients.nrows();
        let se = Col::from_fn(x.nrows(), |i| {
            let da: Vec<f64> = (0..m)
                .map(|j| {
                    self.link.pdf(self.intercepts[j] + lp[i])
                        * (self.levels[j + 1] - self.levels[j])
                })
                .collect();
            let total: f64 = da.iter().sum();
            let db: Vec<f64> = (0..p).map(|r| total * x[(i, r)]).collect();
            let (za, zb) = factor.solve(&da, &db);
            let var: f64 = da.iter().zip(&za).map(|(a, b)| a * b).sum::<f64>()
                + db.iter().zip(&zb).map(|(a, b)| a * b).sum::<f64>();
            var.max(0.0).sqrt()
        });
        (mean, se)
    }

    /// Score test of proportional odds (parallel slopes) for the predictor
    /// columns `columns` (all columns if empty), RMS §13.3.
    ///
    /// The alternative is the partial proportional-odds model in which each
    /// tested column has its own slope at every cut-point,
    /// `P(Y ≥ y_{j+1} | x) = F(α_j + x_tᵀβ_{t,j} + x_uᵀβ_u)`; under the null
    /// all `β_{t,j}` are equal and the fitted model is its MLE. The score `U`
    /// and observed information `I` of the alternative at this fit give
    /// `S = Uᵀ I⁻¹ U` on `|columns| · (k − 2)` d.f. `x` and `y` must be the
    /// data of the fit (its weights and offset are reused). The information is
    /// dense in `k`, so this is meant for a modest number of categories.
    pub fn proportional_odds_score_test(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        columns: &[usize],
    ) -> Result<crate::inference::ChiSqTest, RegressionError> {
        let (n, p) = (x.nrows(), x.ncols());
        let m = self.intercepts.nrows();
        if p != self.coefficients.nrows() || y.nrows() != n || n != self.n_observations {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        let tested: Vec<usize> = if columns.is_empty() {
            (0..p).collect()
        } else {
            columns.to_vec()
        };
        if tested.iter().any(|&c| c >= p) {
            return Err(RegressionError::NumericalError(
                "tested column out of range".to_string(),
            ));
        }
        let untested: Vec<usize> = (0..p).filter(|c| !tested.contains(c)).collect();
        let s = tested.len();
        // parameters: α_1..α_m, β_{t,j} (j-major), β_u
        let q = m + m * s + untested.len();
        let lp = self.linear_predictor(x);
        let mut u = vec![0.0; q];
        let mut info = Mat::<f64>::zeros(q, q);
        let gradient_of_bound = |i: usize, j: usize| -> Vec<(usize, f64)> {
            let mut g = vec![(j, 1.0)];
            g.extend(
                tested
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| (m + j * s + k, x[(i, c)])),
            );
            g.extend(
                untested
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| (m + m * s + k, x[(i, c)])),
            );
            g
        };
        let link = self.link;
        for i in 0..n {
            let c = self.levels.partition_point(|l| *l < y[i]);
            if c >= self.levels.len() || self.levels[c] != y[i] {
                return Err(RegressionError::NumericalError(
                    "y has a value that is not a level of the fit".to_string(),
                ));
            }
            let w = self.weights.as_ref().map_or(1.0, |w| w[i]);
            let eta = lp[i] + self.offset.as_ref().map_or(0.0, |o| o[i]);
            let a = if c == 0 {
                f64::INFINITY
            } else {
                self.intercepts[c - 1] + eta
            };
            let b = if c == m {
                f64::NEG_INFINITY
            } else {
                self.intercepts[c] + eta
            };
            let pr = link.prob_between(a, b);
            let fin = |v: f64, f: &dyn Fn(f64) -> f64| if v.is_finite() { f(v) } else { 0.0 };
            let (fa, fb) = (fin(a, &|v| link.pdf(v)), fin(b, &|v| link.pdf(v)));
            let (dfa, dfb) = (fin(a, &|v| link.dpdf(v)), fin(b, &|v| link.dpdf(v)));
            let haa = dfa / pr - fa * fa / pr / pr;
            let hbb = -dfb / pr - fb * fb / pr / pr;
            let hab = fa * fb / pr / pr;
            let ja = (c > 0).then(|| gradient_of_bound(i, c - 1));
            let jb = (c < m).then(|| gradient_of_bound(i, c));
            for (g, score, h) in [(&ja, fa / pr, haa), (&jb, -fb / pr, hbb)] {
                if let Some(g) = g {
                    for &(r, v) in g {
                        u[r] += w * score * v;
                        for &(t, z) in g {
                            info[(r, t)] -= w * h * v * z;
                        }
                    }
                }
            }
            if let (Some(ga), Some(gb)) = (&ja, &jb) {
                for &(r, v) in ga {
                    for &(t, z) in gb {
                        info[(r, t)] -= w * hab * v * z;
                        info[(t, r)] -= w * hab * v * z;
                    }
                }
            }
        }
        let llt = info.llt(Side::Lower).map_err(|_| {
            RegressionError::NumericalError(
                "the information of the alternative is not positive definite".to_string(),
            )
        })?;
        let uc = Col::from_fn(q, |r| u[r]);
        let z = llt.solve(&uc);
        let statistic = (0..q).map(|r| uc[r] * z[r]).sum::<f64>();
        let df = s * (m - 1);
        let p_value = if df == 0 {
            f64::NAN
        } else {
            ChiSquared::new(df as f64).map_or(f64::NAN, |d| d.sf(statistic))
        };
        Ok(crate::inference::ChiSqTest {
            statistic,
            df,
            p_value,
        })
    }

    /// Predicted `q`-quantile of `Y` (`rms::Quantile.orm`).
    pub fn predict_quantile(
        &self,
        x: &Mat<f64>,
        q: f64,
        method: OrdinalQuantileMethod,
    ) -> Col<f64> {
        let exceed = self.exceedance_probabilities(x);
        let m = self.intercepts.nrows();
        Col::from_fn(x.nrows(), |i| {
            let cp: Vec<f64> = (0..m).map(|j| exceed[(i, j)]).collect();
            self.quantile_of(&cp, &cp, q, method)
        })
    }

    /// Predicted `q`-quantile of `Y` with pointwise confidence limits at
    /// `level` (`Quantile.orm(conf.int = level)`): each `α_j + xβ` is moved
    /// by `± z·SE` (delta method, from [`Self::vcov`]) and the shifted
    /// exceedance curves are turned into quantiles as the estimate.
    /// Returns `(estimate, lower, upper)`.
    pub fn predict_quantile_ci(
        &self,
        x: &Mat<f64>,
        q: f64,
        method: OrdinalQuantileMethod,
        level: f64,
    ) -> (Col<f64>, Col<f64>, Col<f64>) {
        let (m, p) = (self.intercepts.nrows(), self.coefficients.nrows());
        let v = self.vcov();
        let lp = self.linear_predictor(x);
        let z = std_normal().inverse_cdf((1.0 + level) / 2.0);
        let n = x.nrows();
        let (mut est, mut lo, mut hi) = (Col::zeros(n), Col::zeros(n), Col::zeros(n));
        for i in 0..n {
            let se = |j: usize| -> f64 {
                // d(α_j + xβ)/dθ = (e_j, x)
                let mut s = v[(j, j)];
                for a in 0..p {
                    s += 2.0 * x[(i, a)] * v[(j, m + a)];
                    for b in 0..p {
                        s += x[(i, a)] * v[(m + a, m + b)] * x[(i, b)];
                    }
                }
                s.max(0.0).sqrt()
            };
            let lb: Vec<f64> = (0..m).map(|j| self.intercepts[j] + lp[i]).collect();
            let ses: Vec<f64> = (0..m).map(se).collect();
            let cp: Vec<f64> = lb.iter().map(|&l| self.link.cdf(l)).collect();
            let shifted = |sign: f64| -> Vec<f64> {
                (0..m)
                    .map(|j| self.link.cdf(lb[j] + sign * z * ses[j]))
                    .collect()
            };
            est[i] = self.quantile_of(&cp, &cp, q, method);
            lo[i] = self.quantile_of(&shifted(-1.0), &cp, q, method);
            hi[i] = self.quantile_of(&shifted(1.0), &cp, q, method);
        }
        (est, lo, hi)
    }

    /// The `q`-quantile from exceedance probabilities `curve`
    /// (`P(Y ≥ y_{j+1})`), interpolating with the response values shifted
    /// in proportion to the estimate's exceedances `cp` (`m.yvals` of
    /// `Quantile.orm`).
    fn quantile_of(&self, curve: &[f64], cp: &[f64], q: f64, method: OrdinalQuantileMethod) -> f64 {
        let m = curve.len();
        let values = &self.levels;
        match method {
            OrdinalQuantileMethod::Discrete => {
                // smallest y_j with P(Y ≤ y_j) = 1 − P(Y ≥ y_{j+1}) ≥ q
                (0..m)
                    .find(|&j| 1.0 - curve[j] >= q)
                    .map_or(values[m], |j| values[j])
            }
            OrdinalQuantileMethod::Interpolated => {
                // Quantile.orm: interpolate (1, curve₁, …, curve_m, 0) → y
                // values shifted within each interval in proportion to cp
                let span = cp[m - 1] - cp[0];
                let w = |j: usize| {
                    if span != 0.0 {
                        (cp[j] - cp[0]) / span
                    } else {
                        0.0
                    }
                };
                let mut xs = Vec::with_capacity(m + 2);
                let mut ys = Vec::with_capacity(m + 2);
                xs.push(1.0);
                ys.push(values[0]);
                for j in 0..m {
                    xs.push(curve[j]);
                    ys.push((1.0 - w(j)) * values[j] + w(j) * values[j + 1]);
                }
                xs.push(0.0);
                ys.push(values[m]);
                interpolate_decreasing(&xs, &ys, 1.0 - q)
            }
        }
    }
}

/// Observed and model-expected means of a predictor within each response
/// level ([`xmean_ordinaly`]).
#[derive(Debug, Clone, PartialEq)]
pub struct XmeanOrdinaly {
    /// The distinct response values.
    pub levels: Vec<f64>,
    /// `mean(x | Y = y_j)`.
    pub observed: Vec<f64>,
    /// `E(x | Y = y_j)` under the proportional odds model `lrm(y ~ x)`.
    pub expected_po: Vec<f64>,
    /// `E(x | Y = y_j)` under the continuation-ratio model (when requested).
    pub expected_cr: Option<Vec<f64>>,
}

/// The numbers of `rms::plot.xmean.ordinaly` (RMS §13.3.3): for one
/// predictor `x`, the observed mean of `x` at each level of `y` and its
/// expectation `Σᵢ xᵢ P̂(Y = y_j | xᵢ) / n_j` under the univariable
/// proportional odds model and, with `cr`, the continuation-ratio model.
/// Systematic gaps between the observed and expected curves indicate that
/// the model's assumption does not hold for `x`.
pub fn xmean_ordinaly(
    x: &Col<f64>,
    y: &Col<f64>,
    cr: bool,
) -> Result<XmeanOrdinaly, RegressionError> {
    let n = x.nrows();
    let xm = Mat::from_fn(n, 1, |i, _| x[i]);
    let po = OrdinalRegressor::builder().build().fit(&xm, y)?;
    let levels = po.levels().to_vec();
    let k = levels.len();
    let counts = po.counts().to_vec();
    let mut observed = vec![0.0; k];
    for i in 0..n {
        observed[levels.partition_point(|l| *l < y[i])] += x[i];
    }
    for (o, c) in observed.iter_mut().zip(&counts) {
        *o /= c;
    }
    let expected = |p: &Mat<f64>| -> Vec<f64> {
        (0..k)
            .map(|j| (0..n).map(|i| x[i] * p[(i, j)]).sum::<f64>() / counts[j])
            .collect()
    };
    let expected_po = expected(&po.category_probabilities(&xm));
    let expected_cr = if cr {
        let f = crate::solvers::ContinuationRatioRegressor::builder()
            .build()
            .fit(&xm, y)?;
        Some(expected(&f.category_probabilities(&xm)))
    } else {
        None
    };
    Ok(XmeanOrdinaly {
        levels,
        observed,
        expected_po,
        expected_cr,
    })
}

/// `approx(xs, ys, xout, rule = 2)` for `xs` in decreasing order: tied `xs`
/// are collapsed to the mean of their `ys` (R's `ties = mean`), then the
/// points are interpolated linearly, constant beyond the ends.
fn interpolate_decreasing(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(xs.len());
    let mut j = 0;
    while j < xs.len() {
        let mut end = j + 1;
        while end < xs.len() && xs[end] == xs[j] {
            end += 1;
        }
        let mean = ys[j..end].iter().sum::<f64>() / (end - j) as f64;
        pts.push((xs[j], mean));
        j = end;
    }
    let last = pts.len() - 1;
    if x >= pts[0].0 {
        return pts[0].1;
    }
    if x <= pts[last].0 {
        return pts[last].1;
    }
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x0 && x >= x1 {
            return y0 + (y1 - y0) * (x0 - x) / (x0 - x1);
        }
    }
    pts[last].1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_functions_are_consistent() {
        let links = [
            OrdinalLink::Logistic,
            OrdinalLink::Probit,
            OrdinalLink::LogLog,
            OrdinalLink::CLogLog,
            OrdinalLink::Cauchit,
        ];
        for link in links {
            for x in [-3.0, -0.7, 0.0, 0.4, 2.5] {
                assert!(
                    (link.cdf(x) + link.sf(x) - 1.0).abs() < 1e-14,
                    "{link:?} sf"
                );
                assert!(
                    (link.quantile(link.cdf(x)) - x).abs() < 1e-8,
                    "{link:?} quantile"
                );
                let h = 1e-5;
                let dcdf = (link.cdf(x + h) - link.cdf(x - h)) / (2.0 * h);
                assert!((link.pdf(x) - dcdf).abs() < 1e-8, "{link:?} pdf");
                let dpdf = (link.pdf(x + h) - link.pdf(x - h)) / (2.0 * h);
                assert!((link.dpdf(x) - dpdf).abs() < 1e-7, "{link:?} dpdf");
                let between = link.prob_between(x + 1.0, x);
                assert!((between - (link.cdf(x + 1.0) - link.cdf(x))).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn interpolation_follows_r_approx() {
        let xs = [1.0, 0.8, 0.8, 0.3, 0.0];
        let ys = [0.0, 1.0, 3.0, 4.0, 5.0];
        // ends are held constant (rule = 2)
        assert_eq!(interpolate_decreasing(&xs, &ys, 1.5), 0.0);
        assert_eq!(interpolate_decreasing(&xs, &ys, -0.2), 5.0);
        // tied x collapse to the mean of their y (ties = mean)
        assert!((interpolate_decreasing(&xs, &ys, 0.8) - 2.0).abs() < 1e-15);
        assert!((interpolate_decreasing(&xs, &ys, 0.9) - 1.0).abs() < 1e-15);
        assert!((interpolate_decreasing(&xs, &ys, 0.55) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn a_non_positive_definite_information_is_rejected() {
        let info = Information {
            a_diag: vec![1.0, -1.0],
            a_off: vec![0.0],
            b: Mat::zeros(2, 0),
            d: Mat::zeros(0, 0),
        };
        assert!(InformationFactor::new(&info).is_none());
    }
}
