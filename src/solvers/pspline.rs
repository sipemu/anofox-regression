//! Penalized B-spline (P-spline) smoother — GAM-style one-dimensional smoothing.
//!
//! A cubic B-spline basis over the predictor range with an order-2 difference
//! penalty (Eilers & Marx, 1996). The smoothing parameter λ is selected by
//! minimizing GCV, and pointwise standard errors use the Bayesian posterior
//! covariance `σ²·(BᵀB + λP)⁻¹` — the default in `mgcv`. This is the estimator
//! behind ggplot2's `geom_smooth(method = "gam")`.
//!
//! The design matrix passed to [`Regressor::fit`] is the raw predictor as a
//! single column (n × 1); the basis is built internally.

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
        let knots = clamped_knots(xmin, xmax, p);

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
        let scale = tr_btb / tr_pen;

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

        let (beta, l, rss, edf) = match self.lambda {
            Some(lambda) => solve_at(lambda).ok_or(RegressionError::SingularMatrix)?,
            None => {
                let mut best: Option<(f64, Solved)> = None;
                for e in -8..=8 {
                    let lambda = 10f64.powi(e) * scale;
                    if let Some(sol) = solve_at(lambda) {
                        let denom = n as f64 - sol.3;
                        if denom <= 0.0 {
                            continue;
                        }
                        let gcv = n as f64 * sol.2 / (denom * denom);
                        if best.as_ref().map(|(g, _)| gcv < *g).unwrap_or(true) {
                            best = Some((gcv, sol));
                        }
                    }
                }
                best.ok_or(RegressionError::SingularMatrix)?.1
            }
        };

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
        result.rmse = (rss / n as f64).sqrt();
        result.mse = sigma2;

        Ok(FittedPSpline {
            knots,
            beta,
            chol_l: l,
            sigma2,
            edf,
            n,
            p,
            xmin,
            xmax,
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
    xmin: f64,
    xmax: f64,
    result: RegressionResult,
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
    fn eval(&self, x: f64) -> f64 {
        let b = bspline_basis(x.clamp(self.xmin, self.xmax), &self.knots, self.p);
        b.iter().zip(&self.beta).map(|(bi, ci)| bi * ci).sum()
    }
    /// Pointwise SE of the mean response at `x` (Bayesian posterior).
    fn mean_se(&self, x: f64) -> f64 {
        let b = bspline_basis(x.clamp(self.xmin, self.xmax), &self.knots, self.p);
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

fn clamped_knots(xmin: f64, xmax: f64, p: usize) -> Vec<f64> {
    let n_interior = p - DEGREE - 1;
    let mut k = Vec::with_capacity(p + DEGREE + 1);
    for _ in 0..=DEGREE {
        k.push(xmin);
    }
    for i in 1..=n_interior {
        k.push(xmin + (xmax - xmin) * i as f64 / (n_interior as f64 + 1.0));
    }
    for _ in 0..=DEGREE {
        k.push(xmax);
    }
    k
}

fn bspline_basis(x: f64, knots: &[f64], p: usize) -> Vec<f64> {
    let m = knots.len();
    let mut b = vec![0.0; m - 1];
    for i in 0..m - 1 {
        if knots[i] <= x && x < knots[i + 1] {
            b[i] = 1.0;
        }
    }
    if x >= knots[m - 1] {
        for i in (0..m - 1).rev() {
            if knots[i] < knots[i + 1] {
                b[i] = 1.0;
                break;
            }
        }
    }
    for d in 1..=DEGREE {
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
    b.truncate(p);
    b
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
        let knots = clamped_knots(0.0, 10.0, 8);
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
