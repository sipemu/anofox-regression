//! Penalized B-spline (P-spline) smoother — GAM-style one-dimensional smoothing.
//!
//! The Eilers & Marx (1996) P-spline as `mgcv`'s `bs = "ps"` smooth:
//! `mgcv::gam(y ~ s(x, bs = "ps", k = n_basis, m = c(2, penalty_order)),
//! method = "GCV.Cp")`.
//!
//! * Basis: `n_basis` cubic B-splines on evenly spaced knots. The data range
//!   is widened by 0.1 % on each side and the knots continue three spacings
//!   beyond it, as `smooth.construct.ps.smooth.spec`.
//! * Penalty: `λ ‖Dₘ β‖²` with the order-`m` difference matrix `Dₘ`
//!   (default `m = 2`).
//! * Smoothing parameter: fixed, or the minimiser of
//!   `GCV(λ) = n · RSS / (n − edf)²` (a grid over 16 decades, then a
//!   golden-section refinement in `log λ`). `edf` is the trace of the
//!   smoother matrix (it counts the intercept, as `sum(gam$edf)`).
//! * `σ² = RSS / (n − edf)` (`gam$sig2`). Standard errors of the mean use the
//!   Bayesian posterior covariance `σ² (BᵀB + λ DᵀD)⁻¹` (`predict.gam(se.fit =
//!   TRUE)`).
//! * Outside the data range the smooth is extended linearly from the
//!   boundary, as `Predict.matrix.pspline.smooth`.
//!
//! mgcv's sum-to-zero constraint with a separate intercept spans the same
//! model, so fitted values, `edf`, `σ²` and the GCV score agree; mgcv's `sp`
//! is `λ · S.scale`. (ggplot2's `geom_smooth(method = "gam")` is a different
//! smooth — `bs = "cs"` with REML.)
//!
//! The design matrix passed to [`Regressor::fit`] is the raw predictor as a
//! single column (n × 1); the basis is built internally.

use crate::core::{HasModelInfo, ModelInfo};
use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, StudentsT};

const DEGREE: usize = 3;

/// One penalized-basis solution at a given λ: `(coefficients, Cholesky factor L,
/// residual sum of squares, effective degrees of freedom)`.
type Solved = (Vec<f64>, Vec<Vec<f64>>, f64, f64);

/// Penalized B-spline smoother.
#[derive(Debug, Clone)]
pub struct PSplineRegressor {
    /// Number of basis functions. `0` = choose automatically (~n/4, clamped).
    n_basis: usize,
    /// Difference order of the penalty (default 2).
    penalty_order: usize,
    /// Fixed smoothing parameter; `None` selects it by GCV.
    lambda: Option<f64>,
}

impl Default for PSplineRegressor {
    fn default() -> Self {
        Self {
            n_basis: 0,
            penalty_order: 2,
            lambda: None,
        }
    }
}

impl PSplineRegressor {
    /// A P-spline with automatic basis size and GCV-selected smoothing.
    pub fn new() -> Self {
        Self::default()
    }
    /// Configure the number of basis functions (0 = automatic).
    pub fn with_n_basis(mut self, k: usize) -> Self {
        self.n_basis = k;
        self
    }
    /// Configure the difference-penalty order (default 2).
    pub fn with_penalty_order(mut self, order: usize) -> Self {
        self.penalty_order = order.max(1);
        self
    }
    /// Use a fixed smoothing parameter instead of GCV.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda);
        self
    }
    /// Builder alias, matching the other solvers.
    pub fn builder() -> Self {
        Self::default()
    }
    /// No-op finalizer so `builder()....build()` reads consistently.
    pub fn build(self) -> Self {
        self
    }
}

impl Regressor for PSplineRegressor {
    type Fitted = FittedPSpline;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n = x.nrows();
        if n != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        if n < DEGREE + 3 {
            return Err(RegressionError::InsufficientObservations {
                needed: DEGREE + 3,
                got: n,
            });
        }
        let xs: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
        let ys: Vec<f64> = (0..n).map(|i| y[i]).collect();
        let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if xmax <= xmin {
            return Err(RegressionError::AllFeaturesConstant);
        }

        let p = if self.n_basis == 0 {
            (n / 4 + 4).clamp(6, 40).min(n - 1)
        } else {
            self.n_basis
        }
        .max(DEGREE + self.penalty_order + 1);
        let knots = ps_knots(xmin, xmax, p);

        // Basis matrix rows, BᵀB and Bᵀy.
        let brows: Vec<Vec<f64>> = xs.iter().map(|&x| bspline_basis(x, &knots, p)).collect();
        let mut btb = vec![vec![0.0; p]; p];
        let mut bty = vec![0.0; p];
        for (row, &yi) in brows.iter().zip(&ys) {
            for i in 0..p {
                bty[i] += row[i] * yi;
                for j in 0..p {
                    btb[i][j] += row[i] * row[j];
                }
            }
        }
        let pen = difference_penalty(p, self.penalty_order);

        let tr_btb: f64 = (0..p).map(|i| btb[i][i]).sum();
        let tr_pen: f64 = (0..p).map(|i| pen[i][i]).sum::<f64>().max(1e-12);
        let ln_scale = (tr_btb / tr_pen).ln();

        // Choose λ (fixed or GCV-optimal), keeping the winning Cholesky factor.
        let solve_at = |lambda: f64| -> Option<Solved> {
            let a: Vec<Vec<f64>> = (0..p)
                .map(|i| (0..p).map(|j| btb[i][j] + lambda * pen[i][j]).collect())
                .collect();
            let l = cholesky(&a)?;
            let beta = chol_solve(&l, &bty);
            let rss = rss(&brows, &ys, &beta);
            let edf = trace_influence(&btb, &l, p);
            Some((beta, l, rss, edf))
        };

        let lambda = match self.lambda {
            Some(lambda) => lambda,
            None => {
                let gcv = |rho: f64| -> f64 {
                    match solve_at(rho.exp()) {
                        Some(sol) if (n as f64) - sol.3 > 0.0 => {
                            let denom = n as f64 - sol.3;
                            n as f64 * sol.2 / (denom * denom)
                        }
                        _ => f64::INFINITY,
                    }
                };
                minimise_gcv(&gcv, ln_scale).exp()
            }
        };
        let (beta, l, rss, edf) = solve_at(lambda).ok_or(RegressionError::SingularMatrix)?;

        let sigma2 = rss / (n as f64 - edf).max(1.0);
        let fitted: Vec<f64> = brows
            .iter()
            .map(|row| row.iter().zip(&beta).map(|(b, c)| b * c).sum())
            .collect();

        let mut result = RegressionResult::empty(edf.round() as usize, n);
        result.coefficients = Col::from_fn(p, |i| beta[i]);
        result.fitted_values = Col::from_fn(n, |i| fitted[i]);
        result.residuals = Col::from_fn(n, |i| ys[i] - fitted[i]);
        result.n_parameters = edf.round().max(1.0) as usize;
        let tss: f64 = {
            let ym = ys.iter().sum::<f64>() / n as f64;
            ys.iter().map(|v| (v - ym).powi(2)).sum()
        };
        result.r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };
        result.mse = sigma2;
        result.rmse = sigma2.sqrt();

        Ok(FittedPSpline {
            knots,
            beta,
            chol_l: l,
            sigma2,
            edf,
            n,
            p,
            lambda,
            result,
        })
    }
}

/// A fitted P-spline smoother.
#[derive(Debug, Clone)]
pub struct FittedPSpline {
    knots: Vec<f64>,
    beta: Vec<f64>,
    chol_l: Vec<Vec<f64>>,
    sigma2: f64,
    edf: f64,
    n: usize,
    p: usize,
    lambda: f64,
    result: RegressionResult,
}

impl HasModelInfo for FittedPSpline {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::gaussian("pspline")
    }
}

impl FittedPSpline {
    /// Effective degrees of freedom (trace of the smoother matrix).
    pub fn edf(&self) -> f64 {
        self.edf
    }
    /// Residual variance estimate σ².
    pub fn sigma2(&self) -> f64 {
        self.sigma2
    }
    /// The smoothing parameter `λ` of the difference penalty.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
    fn eval(&self, x: f64) -> f64 {
        let b = design_row(x, &self.knots, self.p);
        b.iter().zip(&self.beta).map(|(bi, ci)| bi * ci).sum()
    }
    /// Pointwise SE of the mean response at `x` (Bayesian posterior).
    fn mean_se(&self, x: f64) -> f64 {
        let b = design_row(x, &self.knots, self.p);
        let z = chol_solve(&self.chol_l, &b);
        let var: f64 = b.iter().zip(&z).map(|(bi, zi)| bi * zi).sum();
        (self.sigma2 * var).max(0.0).sqrt()
    }
}

impl FittedRegressor for FittedPSpline {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        Col::from_fn(x.nrows(), |i| self.eval(x[(i, 0)]))
    }

    fn result(&self) -> &RegressionResult {
        &self.result
    }

    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        let fit = self.predict(x);
        let Some(kind) = interval else {
            return PredictionResult::point_only(fit);
        };
        let df = (self.n as f64 - self.edf).max(1.0);
        let t = StudentsT::new(0.0, 1.0, df)
            .map(|d| d.inverse_cdf(0.5 + level / 2.0))
            .unwrap_or(1.96);
        let n = x.nrows();
        let mut lower = Col::zeros(n);
        let mut upper = Col::zeros(n);
        let mut se = Col::zeros(n);
        for i in 0..n {
            let m_se = self.mean_se(x[(i, 0)]);
            // Prediction intervals add the irreducible residual variance.
            let s = match kind {
                IntervalType::Confidence => m_se,
                IntervalType::Prediction => (m_se * m_se + self.sigma2).sqrt(),
            };
            se[i] = s;
            lower[i] = fit[i] - t * s;
            upper[i] = fit[i] + t * s;
        }
        PredictionResult::with_intervals(fit, lower, upper, se)
    }
}

// ---------- numerics ----------

fn rss(brows: &[Vec<f64>], y: &[f64], beta: &[f64]) -> f64 {
    brows
        .iter()
        .zip(y)
        .map(|(row, &yi)| {
            let yh: f64 = row.iter().zip(beta).map(|(b, c)| b * c).sum();
            (yi - yh).powi(2)
        })
        .sum()
}

/// Effective df = tr(A⁻¹ BᵀB): for each column j, solve A z = (BᵀB)[:,j], take z[j].
fn trace_influence(btb: &[Vec<f64>], l: &[Vec<f64>], p: usize) -> f64 {
    (0..p)
        .map(|j| {
            let col: Vec<f64> = (0..p).map(|i| btb[i][j]).collect();
            chol_solve(l, &col)[j]
        })
        .sum()
}

/// `smooth.construct.ps.smooth.spec`'s knots for `p` cubic B-splines: the
/// range widened by 0.1 % on each side, `p − 2` evenly spaced knots over it
/// and three more spacings beyond each end (`p + 4` knots in all).
fn ps_knots(xmin: f64, xmax: f64, p: usize) -> Vec<f64> {
    let nk = p - (DEGREE - 1);
    let xr = xmax - xmin;
    let (xl, xu) = (xmin - 0.001 * xr, xmax + 0.001 * xr);
    let dx = (xu - xl) / (nk as f64 - 1.0);
    let m = p + DEGREE + 1;
    let start = xl - dx * DEGREE as f64;
    (0..m).map(|i| start + dx * i as f64).collect()
}

/// B-splines of degree `degree` on `knots` at `x` (Cox–de Boor; half-open
/// knot intervals): `knots.len() − degree − 1` values.
fn bspline_basis_degree(x: f64, knots: &[f64], degree: usize) -> Vec<f64> {
    let m = knots.len();
    let mut b: Vec<f64> = (0..m - 1)
        .map(|i| f64::from(u8::from(knots[i] <= x && x < knots[i + 1])))
        .collect();
    for d in 1..=degree {
        for i in 0..m - 1 - d {
            let d1 = knots[i + d] - knots[i];
            let d2 = knots[i + d + 1] - knots[i + 1];
            let t1 = if d1 > 0.0 {
                (x - knots[i]) / d1 * b[i]
            } else {
                0.0
            };
            let t2 = if d2 > 0.0 {
                (knots[i + d + 1] - x) / d2 * b[i + 1]
            } else {
                0.0
            };
            b[i] = t1 + t2;
        }
    }
    b.truncate(m - degree - 1);
    b
}

fn bspline_basis(x: f64, knots: &[f64], p: usize) -> Vec<f64> {
    let mut b = bspline_basis_degree(x, knots, DEGREE);
    b.truncate(p);
    b
}

/// Derivative of the cubic basis: `3 (B_{i,2}/(t_{i+3} − t_i) −
/// B_{i+1,2}/(t_{i+4} − t_{i+1}))`.
fn bspline_basis_derivative(x: f64, knots: &[f64], p: usize) -> Vec<f64> {
    let q = bspline_basis_degree(x, knots, DEGREE - 1);
    let k = DEGREE as f64;
    (0..p)
        .map(|i| {
            let a = q[i] / (knots[i + DEGREE] - knots[i]);
            let b = q[i + 1] / (knots[i + DEGREE + 1] - knots[i + 1]);
            k * (a - b)
        })
        .collect()
}

/// Basis row at `x`; outside `[knots[3], knots[p]]` (the widened data range)
/// the linear extension from the nearer end, as
/// `Predict.matrix.pspline.smooth`.
fn design_row(x: f64, knots: &[f64], p: usize) -> Vec<f64> {
    let (ll, ul) = (knots[DEGREE], knots[p]);
    let edge = if x < ll {
        ll
    } else if x > ul {
        ul
    } else {
        return bspline_basis(x, knots, p);
    };
    let b = bspline_basis(edge, knots, p);
    let d = bspline_basis_derivative(edge, knots, p);
    b.iter()
        .zip(&d)
        .map(|(bi, di)| bi + (x - edge) * di)
        .collect()
}

/// Minimise the GCV score over `ρ = ln λ`: a grid of 16 decades around
/// `ln_scale` (`ln tr(BᵀB)/tr(P)`) in steps of a quarter decade, then golden
/// sections on the bracket around the best grid point. A minimum at the end
/// of the grid is kept there (λ → 0 or ∞).
fn minimise_gcv(gcv: &dyn Fn(f64) -> f64, ln_scale: f64) -> f64 {
    let step = std::f64::consts::LN_10 / 4.0;
    let grid: Vec<f64> = (-32..=32).map(|k| ln_scale + step * k as f64).collect();
    let vals: Vec<f64> = grid.iter().map(|&r| gcv(r)).collect();
    let best = (0..grid.len())
        .min_by(|&a, &b| vals[a].total_cmp(&vals[b]))
        .unwrap_or(0);
    if best == 0 || best == grid.len() - 1 {
        return grid[best];
    }
    let (mut a, mut b) = (grid[best - 1], grid[best + 1]);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (gcv(c), gcv(d));
    while b - a > 1e-10 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = gcv(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = gcv(d);
        }
    }
    (a + b) / 2.0
}

/// Difference penalty P = Dₖᵀ Dₖ (p × p) for difference order `k`.
fn difference_penalty(p: usize, order: usize) -> Vec<Vec<f64>> {
    // Binomial-coefficient row of the k-th difference operator, alternating sign.
    let mut d = vec![1.0];
    for _ in 0..order {
        let mut next = vec![0.0; d.len() + 1];
        for (i, &v) in d.iter().enumerate() {
            next[i] += v;
            next[i + 1] -= v;
        }
        d = next;
    }
    let mut pen = vec![vec![0.0; p]; p];
    if p > order {
        for r in 0..p - order {
            for a in 0..=order {
                for c in 0..=order {
                    pen[r + a][r + c] += d[a] * d[c];
                }
            }
        }
    }
    pen
}

#[allow(clippy::needless_range_loop)]
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::IntervalType;

    #[test]
    fn basis_partition_of_unity() {
        let knots = ps_knots(0.0, 10.0, 8);
        for &x in &[0.0, 2.5, 5.0, 7.3, 10.0] {
            let sum: f64 = bspline_basis(x, &knots, 8).iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "sum at {x} = {sum}");
        }
    }

    #[test]
    fn recovers_sine_trend() {
        let n = 80;
        let x = Mat::from_fn(n, 1, |i, _| {
            i as f64 / (n - 1) as f64 * std::f64::consts::TAU
        });
        let y = Col::from_fn(n, |i| {
            let xv = x[(i, 0)];
            xv.sin() + 0.05 * ((i * 7 % 5) as f64 - 2.0)
        });
        let fit = PSplineRegressor::new().fit(&x, &y).expect("fit");
        let pred = fit.predict(&x);
        let rmse = ((0..n)
            .map(|i| (pred[i] - x[(i, 0)].sin()).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(rmse < 0.1, "rmse vs sin = {rmse}");
        assert!(fit.edf() > 3.0 && fit.edf() < 25.0, "edf = {}", fit.edf());

        let pr = fit.predict_with_interval(&x, Some(IntervalType::Confidence), 0.95);
        assert!((0..n).all(|i| pr.lower[i] <= pr.fit[i] && pr.fit[i] <= pr.upper[i]));
        assert!((0..n).all(|i| pr.se[i].is_finite() && pr.se[i] >= 0.0));
    }

    #[test]
    fn recovers_linear_trend() {
        // A 2nd-difference penalty leaves linear functions unpenalized, so an
        // exactly-linear signal must be recovered to numerical precision.
        let n = 40;
        let x = Mat::from_fn(n, 1, |i, _| i as f64 * 0.25);
        let y = Col::from_fn(n, |i| 2.0 * x[(i, 0)] + 1.0);
        let fit = PSplineRegressor::new().fit(&x, &y).expect("fit");
        let pred = fit.predict(&x);
        let max_err = (0..n).map(|i| (pred[i] - y[i]).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-6, "max abs err on linear data = {max_err}");
    }

    #[test]
    fn fixed_lambda_is_deterministic() {
        let n = 30;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let y = Col::from_fn(n, |i| (i as f64 * 0.3).sin());
        let a = PSplineRegressor::new()
            .with_lambda(1.0)
            .fit(&x, &y)
            .unwrap();
        let b = PSplineRegressor::new()
            .with_lambda(1.0)
            .fit(&x, &y)
            .unwrap();
        assert!((a.edf() - b.edf()).abs() < 1e-12);
    }
}
