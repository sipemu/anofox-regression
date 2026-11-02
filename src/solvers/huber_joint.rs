//! scikit-learn's Huber estimator: coefficients and scale estimated jointly
//! (#310).
//!
//! Minimises Owen's (2007) concomitant-scale objective
//!
//! ```text
//! F(β, σ) = n σ + Σ_{|rᵢ| ≤ εσ} rᵢ²/σ + Σ_{|rᵢ| > εσ} (2ε|rᵢ| − ε²σ) + α ‖β‖²
//! ```
//!
//! (`rᵢ = yᵢ − xᵢᵀβ`, intercept unpenalized), which is jointly convex in
//! `(β, σ)`, by a damped Newton method on the piecewise Hessian with an
//! Armijo line search and `σ ≥ 10 ε_mach` (sklearn's bound). sklearn solves
//! the same problem by L-BFGS-B.
//!
//! Reference: Owen, A. B. (2007). "A robust hybrid of lasso and ridge
//! regression". Contemporary Mathematics 443: 59–72.

#![allow(clippy::needless_range_loop)]

use crate::solvers::traits::Convergence;
use faer::{Col, Mat};

/// Lower bound of the scale (sklearn: `10 · eps`).
const SIGMA_MIN: f64 = 10.0 * f64::EPSILON;
const MAX_HALVINGS: usize = 60;
const ARMIJO: f64 = 1e-4;

/// The objective and its derivatives for a design `xd` (intercept column
/// first when `intercept`).
pub(crate) struct JointHuber<'a> {
    pub xd: &'a Mat<f64>,
    pub y: &'a Col<f64>,
    pub epsilon: f64,
    pub alpha: f64,
    pub intercept: bool,
}

impl JointHuber<'_> {
    fn penalized(&self, j: usize) -> bool {
        !(self.intercept && j == 0)
    }

    fn residuals(&self, beta: &[f64]) -> Vec<f64> {
        let b = Col::from_fn(beta.len(), |j| beta[j]);
        let fit = self.xd * &b;
        (0..self.y.nrows()).map(|i| self.y[i] - fit[i]).collect()
    }

    /// `F(β, σ)`; the parameters are `θ = (β, σ)`.
    pub(crate) fn objective(&self, theta: &[f64]) -> f64 {
        let k = theta.len() - 1;
        let sigma = theta[k];
        let eps = self.epsilon;
        let r = self.residuals(&theta[..k]);
        let loss: f64 = r
            .iter()
            .map(|&ri| {
                if ri.abs() <= eps * sigma {
                    ri * ri / sigma
                } else {
                    2.0 * eps * ri.abs() - eps * eps * sigma
                }
            })
            .sum();
        let pen: f64 = (0..k)
            .filter(|&j| self.penalized(j))
            .map(|j| theta[j] * theta[j])
            .sum();
        r.len() as f64 * sigma + loss + self.alpha * pen
    }

    /// Gradient and (generalized) Hessian of `F` at `θ`.
    fn derivatives(&self, theta: &[f64]) -> (Vec<f64>, Mat<f64>) {
        let k = theta.len() - 1;
        let sigma = theta[k];
        let eps = self.epsilon;
        let r = self.residuals(&theta[..k]);
        let mut g = vec![0.0; k + 1];
        let mut h = Mat::<f64>::zeros(k + 1, k + 1);
        g[k] = r.len() as f64;
        for (i, &ri) in r.iter().enumerate() {
            let inlier = ri.abs() <= eps * sigma;
            let dr = if inlier {
                -2.0 * ri / sigma
            } else {
                -2.0 * eps * ri.signum()
            };
            for a in 0..k {
                g[a] += dr * self.xd[(i, a)];
            }
            if !inlier {
                g[k] -= eps * eps;
                continue;
            }
            g[k] -= ri * ri / (sigma * sigma);
            let cross = 2.0 * ri / (sigma * sigma);
            for a in 0..k {
                let xa = self.xd[(i, a)];
                for b in 0..k {
                    h[(a, b)] += 2.0 / sigma * xa * self.xd[(i, b)];
                }
                h[(a, k)] += cross * xa;
                h[(k, a)] += cross * xa;
            }
            h[(k, k)] += 2.0 * ri * ri / (sigma * sigma * sigma);
        }
        for j in (0..k).filter(|&j| self.penalized(j)) {
            g[j] += 2.0 * self.alpha * theta[j];
            h[(j, j)] += 2.0 * self.alpha;
        }
        (g, h)
    }

    /// Newton direction `−(H + μI)⁻¹ g`, with the smallest damping `μ` that
    /// makes the system solvable (the Hessian is singular when few rows are
    /// inliers).
    fn direction(g: &[f64], h: &Mat<f64>) -> Option<Vec<f64>> {
        let m = g.len();
        let scale = (0..m).map(|j| h[(j, j)].abs()).fold(0.0, f64::max).max(1.0);
        let rhs = Mat::from_fn(m, 1, |j, _| -g[j]);
        for damping in [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1.0] {
            let mut hd = h.clone();
            for j in 0..m {
                hd[(j, j)] += damping * scale;
            }
            if let Ok(llt) = hd.llt(faer::Side::Lower) {
                let d = faer::prelude::Solve::solve(&llt, &rhs);
                let d: Vec<f64> = (0..m).map(|j| d[(j, 0)]).collect();
                if d.iter().all(|v| v.is_finite()) {
                    return Some(d);
                }
            }
        }
        None
    }

    /// Backtracking along `d`, keeping `σ ≥ SIGMA_MIN`; `None` when no
    /// decrease is possible.
    fn line_search(&self, theta: &[f64], f: f64, g: &[f64], d: &[f64]) -> Option<(Vec<f64>, f64)> {
        let slope: f64 = g.iter().zip(d).map(|(a, b)| a * b).sum();
        if slope.is_nan() || slope >= 0.0 {
            return None;
        }
        let k = theta.len() - 1;
        let mut t = 1.0;
        for _ in 0..MAX_HALVINGS {
            let mut cand: Vec<f64> = theta.iter().zip(d).map(|(a, s)| a + t * s).collect();
            cand[k] = cand[k].max(SIGMA_MIN);
            let fc = self.objective(&cand);
            if fc.is_finite() && fc <= f + ARMIJO * t * slope {
                return Some((cand, fc));
            }
            t *= 0.5;
        }
        None
    }

    /// Minimise `F` from `θ₀ = (β₀, σ₀)`. Converged when a step moves no
    /// parameter by more than `tol · (1 + max |θ|)`, or no descent is left.
    pub(crate) fn minimize(
        &self,
        mut theta: Vec<f64>,
        max_iterations: usize,
        tol: f64,
    ) -> (Vec<f64>, Convergence) {
        let k = theta.len() - 1;
        theta[k] = theta[k].max(SIGMA_MIN);
        let mut f = self.objective(&theta);
        for iter in 1..=max_iterations {
            let (g, h) = self.derivatives(&theta);
            let done = Convergence {
                converged: true,
                iterations: iter,
            };
            let Some(d) = Self::direction(&g, &h) else {
                return (theta, done);
            };
            let Some((next, f_next)) = self.line_search(&theta, f, &g, &d) else {
                return (theta, done);
            };
            let scale = 1.0 + theta.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let step = theta
                .iter()
                .zip(&next)
                .fold(0.0_f64, |a, (u, v)| a.max((u - v).abs()));
            theta = next;
            f = f_next;
            if step <= tol * scale {
                return (theta, done);
            }
        }
        (
            theta,
            Convergence {
                converged: false,
                iterations: max_iterations,
            },
        )
    }
}
