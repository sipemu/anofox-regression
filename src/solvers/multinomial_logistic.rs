//! Multinomial (softmax) and binary logistic regression with an elastic-net
//! penalty, as scikit-learn's `LogisticRegression` (#307).
//!
//! The objective is sklearn's with `lambda = 1 / C`:
//!
//! ```text
//! Σᵢ sᵢ ℓᵢ(B, b) + λ · ( α ‖B‖₁ + (1 − α)/2 ‖B‖²₂ )
//! ```
//!
//! with sample weights `sᵢ`, the mixing `α = l1_ratio` (0: ridge, 1: lasso)
//! and the intercepts `b` unpenalized. With two classes `ℓ` is the binomial
//! deviance of a single linear predictor (sklearn's binary
//! parameterization: one coefficient row, the second class is "positive");
//! with `K ≥ 3` classes it is the multinomial (softmax) cross-entropy with
//! one coefficient row per class (sklearn's symmetric parameterization).
//!
//! The fit is a proximal Newton method (Lee, Sun & Saunders 2014): each
//! iteration minimises the quadratic model of the smooth part plus the L1
//! term by coordinate descent (a plain Newton solve without L1) and takes a
//! backtracking step on the full objective. L1 and elastic-net fits therefore
//! give exact zeros. The softmax intercepts (and, without a penalty, the
//! coefficients of each feature) are only defined up to a common shift; they
//! are reported centred to sum to zero across classes, as sklearn's lbfgs
//! solution.
//!
//! The Hessian has `m (p + 1)` rows (`m = 1` for two classes, `m = K`
//! otherwise), so the method is meant for up to a few thousand parameters.
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::solvers::MultinomialLogisticRegressor;
//! use faer::{Col, Mat};
//!
//! let x = Mat::from_fn(90, 2, |i, j| ((i * (j + 3)) as f64 * 0.37).sin());
//! let y = Col::from_fn(90, |i| (i % 3) as f64);
//! let fitted = MultinomialLogisticRegressor::builder()
//!     .c(1.0)
//!     .build()
//!     .fit(&x, &y)
//!     .unwrap();
//! assert_eq!(fitted.classes(), &[0.0, 1.0, 2.0]);
//! let proba = fitted.predict_proba(&x);
//! assert_eq!((proba.nrows(), proba.ncols()), (90, 3));
//! ```
//!
//! References:
//!  - Lee, J. D., Sun, Y. & Saunders, M. A. (2014). "Proximal Newton-type
//!    methods for minimizing composite functions". SIAM J. Optim. 24(3).
//!  - Friedman, J., Hastie, T. & Tibshirani, R. (2010). "Regularization
//!    paths for generalized linear models via coordinate descent". JSS 33(1).

use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};

/// Multinomial / binary logistic regression with an elastic-net penalty
/// (sklearn `LogisticRegression`).
///
/// Build with [`MultinomialLogisticRegressor::builder`].
#[derive(Debug, Clone)]
pub struct MultinomialLogisticRegressor {
    lambda: f64,
    l1_ratio: f64,
    with_intercept: bool,
    weights: Option<Col<f64>>,
    max_iterations: usize,
    tolerance: f64,
    error_on_non_convergence: bool,
}

impl Default for MultinomialLogisticRegressor {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            l1_ratio: 0.0,
            with_intercept: true,
            weights: None,
            max_iterations: 100,
            tolerance: 1e-8,
            error_on_non_convergence: true,
        }
    }
}

impl MultinomialLogisticRegressor {
    /// A builder with sklearn's defaults: `C = 1` (`lambda = 1`), ridge
    /// (`l1_ratio = 0`), an intercept, no weights.
    pub fn builder() -> MultinomialLogisticRegressorBuilder {
        MultinomialLogisticRegressorBuilder::default()
    }

    /// Fit to the class labels `y` (any finite values; the classes are the
    /// sorted distinct values, at least two).
    ///
    /// # Errors
    /// [`RegressionError::DimensionMismatch`] / [`RegressionError::InvalidInput`]
    /// for inconsistent or non-finite data or fewer than two classes,
    /// [`RegressionError::InvalidWeights`] for negative or non-finite
    /// weights, [`RegressionError::InvalidParameter`] for an out-of-range
    /// option, and [`RegressionError::ConvergenceFailed`] when
    /// `max_iterations` is exhausted (unless
    /// `error_on_non_convergence(false)`).
    pub fn fit(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<FittedMultinomialLogistic, RegressionError> {
        crate::core::validate_xy(x, y)?;
        self.validate_options()?;
        let n = x.nrows();
        let weights = self.sample_weights(n)?;
        let (classes, labels) = encode_classes(y)?;
        let problem = Problem::new(x, labels, classes.len(), weights, self);
        let (theta, iterations, converged) = problem.solve(self.max_iterations, self.tolerance);
        if !converged && self.error_on_non_convergence {
            return Err(RegressionError::ConvergenceFailed {
                iterations: self.max_iterations,
            });
        }
        let objective = problem.objective(&theta);
        let loss = problem.loss(&theta);
        Ok(problem.into_fitted(theta, classes, iterations, converged, objective, loss))
    }

    fn validate_options(&self) -> Result<(), RegressionError> {
        if !(self.lambda.is_finite() && self.lambda >= 0.0) {
            return Err(RegressionError::InvalidParameter {
                name: "lambda",
                reason: format!("must be finite and >= 0, got {}", self.lambda),
            });
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(RegressionError::InvalidParameter {
                name: "l1_ratio",
                reason: format!("must be in [0, 1], got {}", self.l1_ratio),
            });
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(RegressionError::InvalidParameter {
                name: "tolerance",
                reason: format!("must be finite and > 0, got {}", self.tolerance),
            });
        }
        Ok(())
    }

    fn sample_weights(&self, n: usize) -> Result<Vec<f64>, RegressionError> {
        let Some(w) = &self.weights else {
            return Ok(vec![1.0; n]);
        };
        if w.nrows() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: w.nrows(),
            });
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(RegressionError::InvalidWeights);
        }
        Ok(w.iter().copied().collect())
    }
}

/// Sorted distinct labels and each row's class index.
fn encode_classes(y: &Col<f64>) -> Result<(Vec<f64>, Vec<usize>), RegressionError> {
    let mut classes: Vec<f64> = y.iter().copied().collect();
    classes.sort_by(f64::total_cmp);
    classes.dedup();
    if classes.len() < 2 {
        return Err(RegressionError::InvalidInput(
            "y must contain at least two classes".into(),
        ));
    }
    let labels = y
        .iter()
        .map(|v| classes.partition_point(|c| c < v))
        .collect();
    Ok((classes, labels))
}

/// The penalized log-likelihood and its derivatives.
///
/// Parameters are stored per linear predictor: `theta[k * q + a]` is the
/// coefficient of design column `a` (column 0 is the intercept when one is
/// fitted) in predictor `k`.
struct Problem {
    /// Design with the intercept column first (when fitted), `n × q`.
    xd: Mat<f64>,
    labels: Vec<usize>,
    weights: Vec<f64>,
    /// Number of linear predictors: 1 for two classes, K otherwise.
    m: usize,
    q: usize,
    with_intercept: bool,
    l1: f64,
    l2: f64,
}

impl Problem {
    fn new(
        x: &Mat<f64>,
        labels: Vec<usize>,
        n_classes: usize,
        weights: Vec<f64>,
        opts: &MultinomialLogisticRegressor,
    ) -> Self {
        let off = usize::from(opts.with_intercept);
        let q = x.ncols() + off;
        let xd = Mat::from_fn(
            x.nrows(),
            q,
            |i, a| {
                if a < off {
                    1.0
                } else {
                    x[(i, a - off)]
                }
            },
        );
        Self {
            xd,
            labels,
            weights,
            m: if n_classes == 2 { 1 } else { n_classes },
            q,
            with_intercept: opts.with_intercept,
            l1: opts.lambda * opts.l1_ratio,
            l2: opts.lambda * (1.0 - opts.l1_ratio),
        }
    }

    fn n_params(&self) -> usize {
        self.m * self.q
    }

    fn penalized(&self, u: usize) -> bool {
        // `u % q == 0` rather than `usize::is_multiple_of`, which needs
        // Rust >= 1.87 (the DuckDB wasm toolchain is older). See #20.
        #[allow(clippy::manual_is_multiple_of)]
        let is_intercept = u % self.q == 0;
        !(self.with_intercept && is_intercept)
    }

    /// Linear predictors, `n × m`.
    fn eta(&self, theta: &[f64]) -> Mat<f64> {
        let b = Mat::from_fn(self.q, self.m, |a, k| theta[k * self.q + a]);
        &self.xd * &b
    }

    /// Class probabilities of the `m` predictors (binary: of the positive
    /// class), `n × m`.
    fn probabilities(&self, eta: &Mat<f64>) -> Mat<f64> {
        if self.m == 1 {
            return Mat::from_fn(eta.nrows(), 1, |i, _| sigmoid(eta[(i, 0)]));
        }
        let mut p = eta.clone();
        for i in 0..p.nrows() {
            softmax_row(&mut p, i);
        }
        p
    }

    /// `Σᵢ sᵢ ℓᵢ` (the negative weighted log-likelihood).
    fn loss(&self, theta: &[f64]) -> f64 {
        let eta = self.eta(theta);
        (0..eta.nrows())
            .map(|i| self.weights[i] * self.row_loss(&eta, i))
            .sum()
    }

    fn row_loss(&self, eta: &Mat<f64>, i: usize) -> f64 {
        let y = self.labels[i];
        if self.m == 1 {
            let e = eta[(i, 0)];
            return softplus(e) - if y == 1 { e } else { 0.0 };
        }
        let mx = (0..self.m).fold(f64::NEG_INFINITY, |a, k| a.max(eta[(i, k)]));
        let lse = mx
            + (0..self.m)
                .map(|k| (eta[(i, k)] - mx).exp())
                .sum::<f64>()
                .ln();
        lse - eta[(i, y)]
    }

    fn penalty(&self, theta: &[f64]) -> f64 {
        (0..theta.len())
            .filter(|&u| self.penalized(u))
            .map(|u| self.l1 * theta[u].abs() + 0.5 * self.l2 * theta[u] * theta[u])
            .sum()
    }

    fn objective(&self, theta: &[f64]) -> f64 {
        self.loss(theta) + self.penalty(theta)
    }

    fn l1_norm(&self, theta: &[f64]) -> f64 {
        (0..theta.len())
            .filter(|&u| self.penalized(u))
            .map(|u| theta[u].abs())
            .sum()
    }

    /// Residual `y − p` sign convention: returns `p − y` per row and
    /// predictor.
    fn residuals(&self, prob: &Mat<f64>) -> Mat<f64> {
        let target = |i: usize, k: usize| {
            let class = if self.m == 1 { 1 } else { k };
            f64::from(u8::from(self.labels[i] == class))
        };
        Mat::from_fn(prob.nrows(), self.m, |i, k| prob[(i, k)] - target(i, k))
    }

    /// Gradient and Hessian of the smooth part (loss + L2 term).
    fn derivatives(&self, theta: &[f64]) -> (Vec<f64>, Mat<f64>) {
        let prob = self.probabilities(&self.eta(theta));
        let r = self.residuals(&prob);
        let n = self.xd.nrows();
        let sr = Mat::from_fn(n, self.m, |i, k| self.weights[i] * r[(i, k)]);
        let gm = self.xd.transpose() * &sr;
        let mut g: Vec<f64> = (0..self.n_params())
            .map(|u| gm[(u % self.q, u / self.q)])
            .collect();
        let mut h = self.hessian(&prob);
        for (u, gu) in g.iter_mut().enumerate() {
            if self.penalized(u) {
                *gu += self.l2 * theta[u];
                h[(u, u)] += self.l2;
            }
        }
        (g, h)
    }

    /// `Σᵢ sᵢ (diag(pᵢ) − pᵢpᵢᵀ) ⊗ xᵢxᵢᵀ` (binary: `sᵢ pᵢ(1 − pᵢ) xᵢxᵢᵀ`).
    fn hessian(&self, prob: &Mat<f64>) -> Mat<f64> {
        let (n, q, m) = (self.xd.nrows(), self.q, self.m);
        let mut h = Mat::<f64>::zeros(m * q, m * q);
        for k in 0..m {
            for l in k..m {
                let w = |i: usize| {
                    let (pk, pl) = (prob[(i, k)], prob[(i, l)]);
                    let diag = if m == 1 || k == l { pk } else { 0.0 };
                    self.weights[i] * (diag - pk * pl)
                };
                let xw = Mat::from_fn(n, q, |i, a| w(i) * self.xd[(i, a)]);
                let block = self.xd.transpose() * &xw;
                for a in 0..q {
                    for b in 0..q {
                        h[(k * q + a, l * q + b)] = block[(a, b)];
                        h[(l * q + b, k * q + a)] = block[(a, b)];
                    }
                }
            }
        }
        h
    }

    /// Groups of parameters along which the smooth objective is flat: the
    /// softmax intercepts and, without an L2 term, every feature's
    /// coefficients across classes.
    fn null_groups(&self) -> Vec<usize> {
        if self.m == 1 {
            return Vec::new();
        }
        (0..self.q)
            .filter(|&a| !self.penalized(a) || (self.l2 == 0.0 && self.l1 == 0.0))
            .collect()
    }

    /// Newton direction without L1: `(H + c Σ_g 1_g 1_gᵀ) d = −g`. The rank
    /// one terms make `H` invertible along the flat directions, to which the
    /// gradient is orthogonal, so `d` has no component along them.
    fn newton_direction(&self, g: &[f64], h: &Mat<f64>) -> Result<Vec<f64>, RegressionError> {
        let mut hs = h.clone();
        let mp = g.len();
        let c = (0..mp).map(|u| h[(u, u)]).sum::<f64>() / mp as f64;
        let c = if c > 0.0 { c } else { 1.0 };
        for a in self.null_groups() {
            for k in 0..self.m {
                for l in 0..self.m {
                    hs[(k * self.q + a, l * self.q + a)] += c;
                }
            }
        }
        let rhs = Mat::from_fn(mp, 1, |u, _| -g[u]);
        let d = match hs.llt(faer::Side::Lower) {
            Ok(llt) => faer::prelude::Solve::solve(&llt, &rhs),
            Err(_) => faer::prelude::Solve::solve(&hs.partial_piv_lu(), &rhs),
        };
        let d: Vec<f64> = (0..mp).map(|u| d[(u, 0)]).collect();
        if d.iter().all(|v| v.is_finite()) {
            Ok(d)
        } else {
            Err(RegressionError::SingularMatrix)
        }
    }

    /// Minimise `gᵀd + ½ dᵀHd + l1 ‖θ + d‖₁` by cyclic coordinate descent.
    fn prox_newton_direction(&self, theta: &[f64], g: &[f64], h: &Mat<f64>) -> Vec<f64> {
        let mp = g.len();
        let mut d = vec![0.0; mp];
        let mut hd = vec![0.0; mp];
        let scale = 1.0 + theta.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        for _ in 0..CD_MAX_SWEEPS {
            let mut max_change = 0.0_f64;
            for u in 0..mp {
                let delta = self.cd_update(u, theta[u] + d[u], g[u] + hd[u], h[(u, u)]);
                if delta != 0.0 {
                    d[u] += delta;
                    for (v, hdv) in hd.iter_mut().enumerate() {
                        *hdv += delta * h[(v, u)];
                    }
                    max_change = max_change.max(delta.abs());
                }
            }
            if max_change <= CD_TOLERANCE * scale {
                break;
            }
        }
        d
    }

    /// The coordinate step on parameter `u` at value `cur` with model
    /// gradient `r` and curvature `huu`.
    fn cd_update(&self, u: usize, cur: f64, r: f64, huu: f64) -> f64 {
        if huu <= 0.0 {
            return 0.0;
        }
        let target = cur - r / huu;
        let new = if self.penalized(u) {
            soft_threshold(target, self.l1 / huu)
        } else {
            target
        };
        new - cur
    }

    fn direction(&self, theta: &[f64], g: &[f64], h: &Mat<f64>) -> Vec<f64> {
        if self.l1 == 0.0 {
            if let Ok(d) = self.newton_direction(g, h) {
                return d;
            }
        }
        self.prox_newton_direction(theta, g, h)
    }

    /// Proximal Newton iterations from zero; returns the parameters, the
    /// iterations used and whether the step criterion was met.
    fn solve(&self, max_iterations: usize, tol: f64) -> (Vec<f64>, usize, bool) {
        let mut theta = vec![0.0; self.n_params()];
        let mut f = self.objective(&theta);
        for iter in 1..=max_iterations {
            let (g, h) = self.derivatives(&theta);
            let d = self.direction(&theta, &g, &h);
            let Some((next, f_next, step)) = self.line_search(&theta, f, &g, &d) else {
                self.centre(&mut theta);
                return (theta, iter, true);
            };
            let scale = 1.0 + theta.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            theta = next;
            f = f_next;
            if step <= tol * scale {
                self.centre(&mut theta);
                return (theta, iter, true);
            }
        }
        self.centre(&mut theta);
        (theta, max_iterations, false)
    }

    /// Backtracking (Armijo) search along `d` on the full objective. `None`
    /// when `d` is not a descent direction (the current point is optimal to
    /// rounding). Returns the new point, its objective and the largest
    /// parameter change.
    fn line_search(
        &self,
        theta: &[f64],
        f: f64,
        g: &[f64],
        d: &[f64],
    ) -> Option<(Vec<f64>, f64, f64)> {
        let moved: Vec<f64> = theta.iter().zip(d).map(|(t, s)| t + s).collect();
        let decrement = g.iter().zip(d).map(|(a, b)| a * b).sum::<f64>()
            + self.l1 * (self.l1_norm(&moved) - self.l1_norm(theta));
        let max_d = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if decrement.is_nan() || decrement >= 0.0 || max_d == 0.0 {
            return None;
        }
        let mut t = 1.0;
        for _ in 0..MAX_HALVINGS {
            let cand: Vec<f64> = theta.iter().zip(d).map(|(a, s)| a + t * s).collect();
            let fc = self.objective(&cand);
            if fc.is_finite() && fc <= f + ARMIJO * t * decrement {
                return Some((cand, fc, t * max_d));
            }
            t *= 0.5;
        }
        None
    }

    /// Centre the flat parameter groups to sum to zero across classes (the
    /// objective does not depend on their common shift).
    fn centre(&self, theta: &mut [f64]) {
        for a in self.null_groups() {
            let mean = (0..self.m).map(|k| theta[k * self.q + a]).sum::<f64>() / self.m as f64;
            for k in 0..self.m {
                theta[k * self.q + a] -= mean;
            }
        }
    }

    fn into_fitted(
        self,
        theta: Vec<f64>,
        classes: Vec<f64>,
        iterations: usize,
        converged: bool,
        objective: f64,
        loss: f64,
    ) -> FittedMultinomialLogistic {
        let off = usize::from(self.with_intercept);
        let p = self.q - off;
        let coefficients = Mat::from_fn(self.m, p, |k, j| theta[k * self.q + off + j]);
        let intercepts = Col::from_fn(self.m, |k| {
            if self.with_intercept {
                theta[k * self.q]
            } else {
                0.0
            }
        });
        FittedMultinomialLogistic {
            classes,
            coefficients,
            intercepts,
            iterations,
            converged,
            objective,
            log_likelihood: -loss,
        }
    }
}

const CD_MAX_SWEEPS: usize = 10_000;
const CD_TOLERANCE: f64 = 1e-13;
const MAX_HALVINGS: usize = 60;
const ARMIJO: f64 = 1e-4;

fn sigmoid(e: f64) -> f64 {
    if e >= 0.0 {
        1.0 / (1.0 + (-e).exp())
    } else {
        let z = e.exp();
        z / (1.0 + z)
    }
}

/// `ln(1 + eᵉ)` without overflow.
fn softplus(e: f64) -> f64 {
    e.max(0.0) + (-e.abs()).exp().ln_1p()
}

fn softmax_row(p: &mut Mat<f64>, i: usize) {
    let m = p.ncols();
    let mx = (0..m).fold(f64::NEG_INFINITY, |a, k| a.max(p[(i, k)]));
    let mut s = 0.0;
    for k in 0..m {
        p[(i, k)] = (p[(i, k)] - mx).exp();
        s += p[(i, k)];
    }
    for k in 0..m {
        p[(i, k)] /= s;
    }
}

fn soft_threshold(v: f64, t: f64) -> f64 {
    if v > t {
        v - t
    } else if v < -t {
        v + t
    } else {
        0.0
    }
}

/// A fitted [`MultinomialLogisticRegressor`].
#[derive(Debug, Clone)]
pub struct FittedMultinomialLogistic {
    classes: Vec<f64>,
    coefficients: Mat<f64>,
    intercepts: Col<f64>,
    iterations: usize,
    converged: bool,
    objective: f64,
    log_likelihood: f64,
}

impl FittedMultinomialLogistic {
    /// The sorted distinct class labels (sklearn `classes_`).
    pub fn classes(&self) -> &[f64] {
        &self.classes
    }
    /// Coefficients, one row per linear predictor (sklearn `coef_`): `1 × p`
    /// for two classes (the log-odds of the second class), `K × p`
    /// otherwise.
    pub fn coefficients(&self) -> &Mat<f64> {
        &self.coefficients
    }
    /// Intercepts, one per linear predictor (sklearn `intercept_`; all 0
    /// without an intercept). With `K ≥ 3` classes they sum to zero.
    pub fn intercepts(&self) -> &Col<f64> {
        &self.intercepts
    }
    /// The linear predictors `X Bᵀ + b`, `n × 1` for two classes and `n × K`
    /// otherwise (sklearn `decision_function`).
    pub fn decision_function(&self, x: &Mat<f64>) -> Mat<f64> {
        let mut eta = x * self.coefficients.transpose();
        for i in 0..eta.nrows() {
            for k in 0..eta.ncols() {
                eta[(i, k)] += self.intercepts[k];
            }
        }
        eta
    }
    /// Class probabilities, `n × K` with the columns in [`classes`](Self::classes)
    /// order (sklearn `predict_proba`).
    pub fn predict_proba(&self, x: &Mat<f64>) -> Mat<f64> {
        let mut eta = self.decision_function(x);
        if self.coefficients.nrows() == 1 {
            return Mat::from_fn(eta.nrows(), 2, |i, k| {
                let p1 = sigmoid(eta[(i, 0)]);
                if k == 1 {
                    p1
                } else {
                    1.0 - p1
                }
            });
        }
        for i in 0..eta.nrows() {
            softmax_row(&mut eta, i);
        }
        eta
    }
    /// The most probable class of each row (sklearn `predict`).
    pub fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let p = self.predict_proba(x);
        Col::from_fn(p.nrows(), |i| {
            let best = (0..p.ncols()).fold(0, |b, k| if p[(i, k)] > p[(i, b)] { k } else { b });
            self.classes[best]
        })
    }
    /// Classification accuracy on `(x, y)`.
    pub fn score(&self, x: &Mat<f64>, y: &Col<f64>) -> f64 {
        let pred = self.predict(x);
        let hits = (0..y.nrows()).filter(|&i| pred[i] == y[i]).count();
        hits as f64 / y.nrows().max(1) as f64
    }
    /// Weighted log-likelihood `−Σᵢ sᵢ ℓᵢ` at the fit.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }
    /// The minimised objective: `−log_likelihood` plus the penalty.
    pub fn objective(&self) -> f64 {
        self.objective
    }
}

impl crate::solvers::IterativeFit for FittedMultinomialLogistic {
    fn converged(&self) -> bool {
        self.converged
    }
    fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Builder for [`MultinomialLogisticRegressor`].
#[derive(Debug, Clone, Default)]
pub struct MultinomialLogisticRegressorBuilder {
    inner: MultinomialLogisticRegressor,
}

impl MultinomialLogisticRegressorBuilder {
    /// Penalty strength `λ ≥ 0` (default 1; 0: unpenalized).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.inner.lambda = lambda;
        self
    }
    /// sklearn's inverse penalty strength, `λ = 1 / C` (`C = ∞`: no
    /// penalty).
    pub fn c(mut self, c: f64) -> Self {
        self.inner.lambda = 1.0 / c;
        self
    }
    /// Elastic-net mixing in `[0, 1]`: 0 ridge (default), 1 lasso (sklearn
    /// `l1_ratio`).
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.inner.l1_ratio = l1_ratio;
        self
    }
    /// Fit unpenalized intercepts (default `true`).
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.inner.with_intercept = include;
        self
    }
    /// Non-negative sample weights (sklearn `sample_weight`; class weights
    /// are folded in by the caller).
    pub fn weights(mut self, weights: Col<f64>) -> Self {
        self.inner.weights = Some(weights);
        self
    }
    /// Maximum proximal Newton iterations (default 100).
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.inner.max_iterations = max_iterations;
        self
    }
    /// Converged when the largest parameter change of a step is at most
    /// `tolerance · (1 + max |θ|)` (default 1e-8).
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.inner.tolerance = tolerance;
        self
    }
    /// Return `Err(ConvergenceFailed)` when `max_iterations` is exhausted
    /// (default `true`); `false` returns the last iterate.
    pub fn error_on_non_convergence(mut self, error: bool) -> Self {
        self.inner.error_on_non_convergence = error;
        self
    }
    /// The configured estimator.
    pub fn build(self) -> MultinomialLogisticRegressor {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_single_class_and_bad_options() {
        let x = Mat::from_fn(4, 1, |i, _| i as f64);
        let y = Col::from_fn(4, |_| 1.0);
        let m = MultinomialLogisticRegressor::builder().build();
        assert!(m.fit(&x, &y).is_err());
        let y = Col::from_fn(4, |i| (i % 2) as f64);
        for bad in [
            MultinomialLogisticRegressor::builder()
                .l1_ratio(1.5)
                .build(),
            MultinomialLogisticRegressor::builder().lambda(-1.0).build(),
            MultinomialLogisticRegressor::builder()
                .tolerance(0.0)
                .build(),
            MultinomialLogisticRegressor::builder()
                .weights(Col::from_fn(4, |i| i as f64 - 1.0))
                .build(),
            MultinomialLogisticRegressor::builder()
                .weights(Col::from_fn(3, |_| 1.0))
                .build(),
        ] {
            assert!(bad.fit(&x, &y).is_err());
        }
    }

    #[test]
    fn separable_without_penalty_reports_non_convergence() {
        let x = Mat::from_fn(6, 1, |i, _| i as f64);
        let y = Col::from_fn(6, |i| f64::from(u8::from(i >= 3)));
        let m = MultinomialLogisticRegressor::builder()
            .lambda(0.0)
            .max_iterations(5);
        assert!(m.clone().build().fit(&x, &y).is_err());
        let f = m
            .error_on_non_convergence(false)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert!(!crate::solvers::IterativeFit::converged(&f));
    }

    #[test]
    fn helpers_are_stable() {
        assert!((softplus(800.0) - 800.0).abs() < 1e-12);
        assert!(softplus(-800.0) >= 0.0);
        assert!((sigmoid(-800.0)).abs() < 1e-300);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-2.0, 1.0), -1.0);
    }
}
