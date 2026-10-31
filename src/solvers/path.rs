//! Regularisation paths and k-fold cross-validation for the Gaussian
//! elastic net (ridge, lasso and anything in between), following `glmnet`.
//!
//! # Objective
//!
//! For a decreasing grid of penalties `λ`, [`fit_path`] solves
//!
//! ```text
//! 1/(2n) · ||y − β₀ − Xβ||² + λ · [ (1 − α)/2 · ||β||² + α · ||β||₁ ]
//! ```
//!
//! with exactly the conventions of `glmnet(family = "gaussian")`:
//!
//! - `λ` is the per-observation penalty, the scale of
//!   [`LambdaScaling::Glmnet`](crate::core::LambdaScaling::Glmnet) (`1/(2n)` on
//!   the residual sum of squares).
//! - The intercept is never penalised; with an intercept `X` and `y` are
//!   centred.
//! - With `standardize = true` the penalty applies to the coefficients of the
//!   columns scaled to unit variance (`1/n` formula); coefficients are always
//!   returned on the original scale.
//! - Like `glmnet`, `y` is scaled internally to unit variance (`1/n`
//!   formula) and `λ` by the same factor before solving, then the solution is
//!   scaled back. This leaves the lasso part unchanged but divides the ridge
//!   part of the penalty by `sd(y)`; it is what `glmnet` computes (see the
//!   note on the Gaussian objective in `?glmnet`). For `α = 0` and
//!   `standardize = false`, a `y` with unit `1/n` variance gives exactly
//!   [`RidgeRegressor`](crate::solvers::RidgeRegressor) with
//!   `LambdaScaling::Glmnet`.
//!
//! Ridge (`α = 0`) is solved in closed form from one SVD of the centred
//! design, which is exact for every `λ` of the path. For `α > 0` the path is
//! computed by cyclic coordinate descent with warm starts (each `λ` starts
//! from the previous solution) and active-set iterations, as `glmnet`.
//!
//! [`lambda_grid`] gives `glmnet`'s default grid, and [`cv_path`] its
//! `cv.glmnet` aggregates (`type.measure = "mse"`, `grouped = TRUE`):
//! `cv_mean`, `cv_se`, `lambda_min` and `lambda_1se`, with folds given as
//! ids (deterministic) or drawn from a seed by [`kfold_ids`]. Group,
//! stratified and time-series folds, CV on explicit train / test splits and
//! nested CV are in [`solvers::cv`](crate::solvers::cv).
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::solvers::path::{cv_path, fit_path, kfold_ids, lambda_grid, PathOptions};
//! use faer::{Col, Mat};
//!
//! let x = Mat::from_fn(60, 3, |i, j| ((i * (j + 3)) % 11) as f64 + 0.1 * j as f64);
//! let y = Col::from_fn(60, |i| 1.0 + 2.0 * x[(i, 0)] - x[(i, 2)] + ((i % 5) as f64 - 2.0));
//!
//! let opts = PathOptions::elastic_net(0.5);
//! let lambdas = lambda_grid(&x, &y, &opts, 50, None).unwrap();
//! let path = fit_path(&x, &y, &lambdas, &opts).unwrap();
//! assert_eq!(path.coefficients.ncols(), 50);
//!
//! let folds = kfold_ids(60, 5, 42);
//! let cv = cv_path(&x, &y, &lambdas, &folds, &opts).unwrap();
//! assert!(cv.lambda_1se >= cv.lambda_min);
//! ```

use crate::solvers::cv::{cv_path_splits, splits_from_fold_ids, CvMetric};
use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};

/// Options of a regularisation path (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathOptions {
    /// Elastic-net mixing `α ∈ [0, 1]`: 0 is ridge, 1 is the lasso.
    pub alpha: f64,
    /// Fit an unpenalised intercept (centres `X` and `y`). Default `true`.
    pub fit_intercept: bool,
    /// Penalise the coefficients of the unit-variance columns (`glmnet`'s
    /// `standardize`). Default `false`, the crate's convention.
    pub standardize: bool,
    /// Coordinate-descent convergence threshold on the largest weighted
    /// squared coefficient change, relative to the variance of `y`
    /// (`glmnet`'s `thresh`). Default `1e-18`.
    pub tol: f64,
    /// Maximum coordinate-descent passes over the data per `λ`. Default 100 000.
    pub max_iter: usize,
}

impl Default for PathOptions {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            standardize: false,
            tol: 1e-18,
            max_iter: 100_000,
        }
    }
}

impl PathOptions {
    /// Ridge path options (`α = 0`).
    pub fn ridge() -> Self {
        Self {
            alpha: 0.0,
            ..Self::default()
        }
    }

    /// Elastic-net path options with mixing `alpha`.
    pub fn elastic_net(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    /// Set whether an intercept is fitted.
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set whether the columns are standardised for the penalty.
    pub fn with_standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), RegressionError> {
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(RegressionError::NumericalError(format!(
                "alpha must be in [0, 1], got {}",
                self.alpha
            )));
        }
        if self.tol.is_nan() || self.tol <= 0.0 || self.max_iter == 0 {
            return Err(RegressionError::NumericalError(
                "tol must be > 0 and max_iter >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// A coefficient path over a grid of penalties.
#[derive(Debug, Clone)]
pub struct RegPath {
    /// Penalties of the path, in the order the path was computed (decreasing
    /// for [`fit_path`]; the LARS / LassoLars `alphas` for
    /// [`FittedLars::path`](crate::solvers::FittedLars::path)).
    pub lambdas: Vec<f64>,
    /// Intercept at each `λ` (0 without an intercept).
    pub intercepts: Vec<f64>,
    /// Coefficients on the original scale, `p × n_lambda` (column `l` is the
    /// solution at `lambdas[l]`), as `glmnet`'s `beta`.
    pub coefficients: Mat<f64>,
    /// Number of non-zero coefficients at each `λ` (`glmnet`'s `df`).
    pub df: Vec<usize>,
    /// Elastic-net mixing parameter (`NaN` for a LARS path).
    pub alpha: f64,
}

impl RegPath {
    /// Number of penalties on the path.
    pub fn len(&self) -> usize {
        self.lambdas.len()
    }

    /// Whether the path is empty.
    pub fn is_empty(&self) -> bool {
        self.lambdas.is_empty()
    }

    /// Coefficients (without the intercept) at path index `l`.
    pub fn coefficients_at(&self, l: usize) -> Col<f64> {
        Col::from_fn(self.coefficients.nrows(), |j| self.coefficients[(j, l)])
    }

    /// Predictions of every path solution: an `n × n_lambda` matrix.
    pub fn predict(&self, x: &Mat<f64>) -> Mat<f64> {
        let p = self.coefficients.nrows();
        Mat::from_fn(x.nrows(), self.len(), |i, l| {
            let mut v = self.intercepts[l];
            for j in 0..p {
                v += x[(i, j)] * self.coefficients[(j, l)];
            }
            v
        })
    }
}

/// Cross-validation curve over a penalty grid (`cv.glmnet`), from
/// [`cv_path`] or [`cv_path_splits`].
#[derive(Debug, Clone)]
pub struct CvPath {
    /// Penalties (same order as passed to [`cv_path`]).
    pub lambdas: Vec<f64>,
    /// Mean cross-validated error (`cvm`): the fold errors (mean squared
    /// error unless another [`CvMetric`] was asked for) averaged with the fold
    /// sizes as weights.
    pub cv_mean: Vec<f64>,
    /// Standard error of `cv_mean` (`cvsd`):
    /// `sqrt(weighted.mean((mse_k − cvm)², n_k) / (K − 1))`.
    pub cv_se: Vec<f64>,
    /// Number of non-zero coefficients of the full-data fit at each `λ`.
    pub df: Vec<usize>,
    /// Largest `λ` attaining the minimum `cv_mean`.
    pub lambda_min: f64,
    /// Largest `λ` with `cv_mean ≤ cv_mean[min] + cv_se[min]`.
    pub lambda_1se: f64,
    /// Index of `lambda_min` in `lambdas`.
    pub index_min: usize,
    /// Index of `lambda_1se` in `lambdas`.
    pub index_1se: usize,
    /// Number of folds (splits).
    pub n_folds: usize,
    /// Full-data path at the same penalties.
    pub path: RegPath,
}

/// Centred / scaled data for the solver.
struct Prepared {
    /// Centred (and scaled when standardising) columns, column-major.
    xs: Mat<f64>,
    /// Column means (0 without an intercept).
    x_mean: Vec<f64>,
    /// Column scale (1 unless standardising).
    x_scale: Vec<f64>,
    /// `(1/n)·||xs_j||²`.
    v: Vec<f64>,
    /// Whether a column has no variation (always 0 coefficient).
    constant: Vec<bool>,
    /// Centred and scaled response.
    ys: Col<f64>,
    y_mean: f64,
    y_scale: f64,
}

pub(crate) fn validate_xy(x: &Mat<f64>, y: &Col<f64>) -> Result<(), RegressionError> {
    if x.nrows() != y.nrows() {
        return Err(RegressionError::DimensionMismatch {
            x_rows: x.nrows(),
            y_len: y.nrows(),
        });
    }
    if x.nrows() < 2 {
        return Err(RegressionError::InsufficientObservations {
            needed: 2,
            got: x.nrows(),
        });
    }
    if x.ncols() == 0 {
        return Err(RegressionError::NumericalError(
            "x must have at least one column".to_string(),
        ));
    }
    let finite = y.iter().all(|v| v.is_finite())
        && (0..x.ncols()).all(|j| (0..x.nrows()).all(|i| x[(i, j)].is_finite()));
    if !finite {
        return Err(RegressionError::NumericalError(
            "x and y must be finite".to_string(),
        ));
    }
    Ok(())
}

fn prepare(x: &Mat<f64>, y: &Col<f64>, opts: &PathOptions) -> Prepared {
    let n = x.nrows();
    let p = x.ncols();
    let nf = n as f64;
    let y_mean = if opts.fit_intercept {
        y.iter().sum::<f64>() / nf
    } else {
        0.0
    };
    let ss_y = y.iter().map(|v| (v - y_mean) * (v - y_mean)).sum::<f64>() / nf;
    let y_scale = if ss_y > 0.0 { ss_y.sqrt() } else { 1.0 };
    let ys = Col::from_fn(n, |i| (y[i] - y_mean) / y_scale);

    let mut xs = Mat::<f64>::zeros(n, p);
    let mut x_mean = vec![0.0; p];
    let mut x_scale = vec![1.0; p];
    let mut v = vec![0.0; p];
    let mut constant = vec![false; p];
    for j in 0..p {
        let m = if opts.fit_intercept {
            (0..n).map(|i| x[(i, j)]).sum::<f64>() / nf
        } else {
            0.0
        };
        let ms = (0..n).map(|i| (x[(i, j)] - m).powi(2)).sum::<f64>() / nf;
        // Constant column (no variation around the centre): glmnet drops it.
        let scale_ref = (0..n).fold(0.0_f64, |a, i| a.max(x[(i, j)].abs())).max(1.0);
        if ms.sqrt() <= 1e-12 * scale_ref {
            constant[j] = true;
            x_mean[j] = m;
            continue;
        }
        let s = if opts.standardize { ms.sqrt() } else { 1.0 };
        x_mean[j] = m;
        x_scale[j] = s;
        for i in 0..n {
            xs[(i, j)] = (x[(i, j)] - m) / s;
        }
        v[j] = ms / (s * s);
    }
    Prepared {
        xs,
        x_mean,
        x_scale,
        v,
        constant,
        ys,
        y_mean,
        y_scale,
    }
}

/// `glmnet`'s default penalty grid.
///
/// `λ_max = max_j |x̃_jᵀ (y − ȳ)| / (n · max(α, 0.001))`, where `x̃_j` is the
/// centred (and, with `standardize`, unit-variance) column: the smallest `λ`
/// with all coefficients zero for `α > 0` (for ridge `glmnet` uses
/// `α = 0.001`). The grid has `n_lambda` values log-spaced from `λ_max` down to
/// `λ_max · lambda_min_ratio`; `lambda_min_ratio` defaults to `1e-4` when
/// `n > p` and `0.01` otherwise, as in `glmnet`.
pub fn lambda_grid(
    x: &Mat<f64>,
    y: &Col<f64>,
    opts: &PathOptions,
    n_lambda: usize,
    lambda_min_ratio: Option<f64>,
) -> Result<Vec<f64>, RegressionError> {
    validate_xy(x, y)?;
    opts.validate()?;
    if n_lambda == 0 {
        return Err(RegressionError::NumericalError(
            "n_lambda must be >= 1".to_string(),
        ));
    }
    let ratio = lambda_min_ratio.unwrap_or(if x.nrows() > x.ncols() { 1e-4 } else { 1e-2 });
    if !(ratio > 0.0 && ratio < 1.0) {
        return Err(RegressionError::NumericalError(format!(
            "lambda_min_ratio must be in (0, 1), got {ratio}"
        )));
    }
    let prep = prepare(x, y, opts);
    let n = x.nrows() as f64;
    let mut gmax = 0.0_f64;
    for j in 0..x.ncols() {
        if prep.constant[j] {
            continue;
        }
        let g: f64 = (0..x.nrows()).map(|i| prep.xs[(i, j)] * prep.ys[i]).sum();
        gmax = gmax.max(g.abs());
    }
    let lambda_max = gmax * prep.y_scale / (n * opts.alpha.max(1e-3));
    if lambda_max.is_nan() || lambda_max <= 0.0 {
        return Err(RegressionError::NumericalError(
            "lambda_max is zero: y is constant or uncorrelated with every column".to_string(),
        ));
    }
    if n_lambda == 1 {
        return Ok(vec![lambda_max]);
    }
    let (lo, hi) = ((lambda_max * ratio).ln(), lambda_max.ln());
    let step = (hi - lo) / (n_lambda - 1) as f64;
    Ok((0..n_lambda)
        .map(|k| (hi - step * k as f64).exp())
        .collect())
}

pub(crate) fn validate_lambdas(lambdas: &[f64]) -> Result<(), RegressionError> {
    if lambdas.is_empty() {
        return Err(RegressionError::NumericalError(
            "lambdas must not be empty".to_string(),
        ));
    }
    if lambdas.iter().any(|l| !(l.is_finite() && *l >= 0.0)) {
        return Err(RegressionError::NumericalError(
            "lambdas must be finite and >= 0".to_string(),
        ));
    }
    Ok(())
}

/// Compute the elastic-net coefficient path at `lambdas` (see the
/// [module docs](self) for the objective).
///
/// The penalties are processed from the largest to the smallest (warm
/// starts) and the result keeps the order in which they were passed. A
/// `λ = 0` gives the least-squares fit (minimum-norm for ridge when `X` is
/// rank deficient).
pub fn fit_path(
    x: &Mat<f64>,
    y: &Col<f64>,
    lambdas: &[f64],
    opts: &PathOptions,
) -> Result<RegPath, RegressionError> {
    validate_xy(x, y)?;
    opts.validate()?;
    validate_lambdas(lambdas)?;
    let prep = prepare(x, y, opts);
    let scaled = if opts.alpha == 0.0 {
        ridge_path_scaled(&prep, lambdas)?
    } else {
        let pens: Vec<(f64, f64)> = lambdas
            .iter()
            .map(|&lam| {
                let lam = lam / prep.y_scale;
                (lam * opts.alpha, lam * (1.0 - opts.alpha))
            })
            .collect();
        cd_path_scaled(&prep, &pens, opts)?
    };
    Ok(unscale(&prep, scaled, lambdas, opts.alpha))
}

/// Coefficients on the scaled problem, `p × n_lambda`.
fn unscale(prep: &Prepared, scaled: Mat<f64>, lambdas: &[f64], alpha: f64) -> RegPath {
    let p = scaled.nrows();
    let nl = lambdas.len();
    let mut coefficients = Mat::<f64>::zeros(p, nl);
    let mut intercepts = vec![0.0; nl];
    let mut df = vec![0usize; nl];
    for l in 0..nl {
        let mut b0 = prep.y_mean;
        for j in 0..p {
            if prep.constant[j] {
                continue;
            }
            let b = scaled[(j, l)] * prep.y_scale / prep.x_scale[j];
            coefficients[(j, l)] = b;
            b0 -= prep.x_mean[j] * b;
            if b != 0.0 {
                df[l] += 1;
            }
        }
        intercepts[l] = b0;
    }
    RegPath {
        lambdas: lambdas.to_vec(),
        intercepts,
        coefficients,
        df,
        alpha,
    }
}

/// Ridge path in closed form: `b(λ) = V diag(s / (s² + nλ)) Uᵀ y` on the
/// scaled problem, from one thin SVD.
fn ridge_path_scaled(prep: &Prepared, lambdas: &[f64]) -> Result<Mat<f64>, RegressionError> {
    let n = prep.xs.nrows();
    let p = prep.xs.ncols();
    let svd = prep
        .xs
        .thin_svd()
        .map_err(|_| RegressionError::NumericalError("SVD failed".to_string()))?;
    let u = svd.U();
    let s = svd.S().column_vector();
    let v = svd.V();
    let uty = u.transpose() * &prep.ys;
    let k = s.nrows();
    let smax = (0..k).fold(0.0_f64, |m, i| m.max(s[i]));
    let cutoff = smax * 1e-12 * (n.max(p) as f64);
    let mut out = Mat::<f64>::zeros(p, lambdas.len());
    for (l, &lam) in lambdas.iter().enumerate() {
        let lam_s = n as f64 * lam / prep.y_scale;
        for i in 0..k {
            let si = s[i];
            if si <= cutoff {
                continue;
            }
            let d = si / (si * si + lam_s) * uty[i];
            for j in 0..p {
                out[(j, l)] += v[(j, i)] * d;
            }
        }
    }
    Ok(out)
}

fn soft(z: f64, g: f64) -> f64 {
    if z > g {
        z - g
    } else if z < -g {
        z + g
    } else {
        0.0
    }
}

/// Solve `1/(2n) · ||y − β₀ − Xβ||² + l1 · ||β||₁ + l2/2 · ||β||²` (no
/// standardisation, no `sd(y)` rescaling of the ridge part) by coordinate
/// descent. Returns the intercept (0 without one) and the coefficients on
/// the original scale; constant columns get 0. `tol` and `max_iter` are as
/// in [`PathOptions`].
pub(crate) fn fit_penalized(
    x: &Mat<f64>,
    y: &Col<f64>,
    l1: f64,
    l2: f64,
    fit_intercept: bool,
    tol: f64,
    max_iter: usize,
) -> Result<(f64, Col<f64>), RegressionError> {
    validate_xy(x, y)?;
    let opts = PathOptions {
        alpha: 1.0,
        fit_intercept,
        standardize: false,
        tol,
        max_iter,
    };
    opts.validate()?;
    let prep = prepare(x, y, &opts);
    // On the scaled problem (b = β / sd(y)) the lasso part is divided by
    // sd(y) and the ridge part unchanged.
    let scaled = cd_path_scaled(&prep, &[(l1 / prep.y_scale, l2)], &opts)?;
    let path = unscale(&prep, scaled, &[0.0], f64::NAN);
    Ok((path.intercepts[0], path.coefficients_at(0)))
}

/// Coordinate-descent path on the scaled problem with warm starts, for the
/// scaled penalties `(l1, l2)` of each solution.
fn cd_path_scaled(
    prep: &Prepared,
    pens: &[(f64, f64)],
    opts: &PathOptions,
) -> Result<Mat<f64>, RegressionError> {
    let n = prep.xs.nrows();
    let p = prep.xs.ncols();
    let nf = n as f64;
    // Process in decreasing penalty for warm starts.
    let mut order: Vec<usize> = (0..pens.len()).collect();
    order.sort_by(|&a, &b| (pens[b].0 + pens[b].1).total_cmp(&(pens[a].0 + pens[a].1)));

    let mut beta = vec![0.0; p];
    let mut r: Vec<f64> = prep.ys.iter().copied().collect();
    let mut out = Mat::<f64>::zeros(p, pens.len());
    let cols: Vec<&[f64]> = (0..p)
        .map(|j| {
            let c = prep.xs.col(j);
            c.try_as_col_major().expect("contiguous column").as_slice()
        })
        .collect();
    // y is scaled to unit variance, so `tol` is relative to var(y).
    let thresh = opts.tol;

    for &l in &order {
        let (l1, l2) = pens[l];
        let mut passes = 0usize;
        let mut converged = false;
        // Outer loop: full pass, then iterate on the active set until it
        // converges, then a full pass to check for new entrants (glmnet).
        while passes < opts.max_iter {
            let dmax = cd_pass(&cols, prep, &mut beta, &mut r, l1, l2, nf, None);
            passes += 1;
            if dmax < thresh {
                converged = true;
                break;
            }
            let active: Vec<usize> = (0..p).filter(|&j| beta[j] != 0.0).collect();
            while passes < opts.max_iter {
                let d = cd_pass(&cols, prep, &mut beta, &mut r, l1, l2, nf, Some(&active));
                passes += 1;
                if d < thresh {
                    break;
                }
            }
        }
        if !converged {
            return Err(RegressionError::ConvergenceFailed {
                iterations: opts.max_iter,
            });
        }
        for j in 0..p {
            out[(j, l)] = beta[j];
        }
    }
    Ok(out)
}

/// One coordinate-descent pass; returns `max_j v_j · Δβ_j²`.
#[allow(clippy::too_many_arguments)]
fn cd_pass(
    cols: &[&[f64]],
    prep: &Prepared,
    beta: &mut [f64],
    r: &mut [f64],
    l1: f64,
    l2: f64,
    nf: f64,
    subset: Option<&[usize]>,
) -> f64 {
    let p = beta.len();
    let mut dmax = 0.0_f64;
    let mut update = |j: usize| {
        if prep.constant[j] {
            return;
        }
        let xj = cols[j];
        let vj = prep.v[j];
        let g: f64 = xj.iter().zip(r.iter()).map(|(a, b)| a * b).sum::<f64>() / nf;
        let old = beta[j];
        let new = soft(g + vj * old, l1) / (vj + l2);
        if new != old {
            let d = new - old;
            for (ri, xi) in r.iter_mut().zip(xj) {
                *ri -= d * xi;
            }
            beta[j] = new;
            dmax = dmax.max(vj * d * d);
        }
    };
    match subset {
        Some(s) => s.iter().for_each(|&j| update(j)),
        None => (0..p).for_each(&mut update),
    }
    dmax
}

/// Deterministic fold ids `0..k` for `n` rows: a seeded shuffle of
/// `0, 1, …, k−1, 0, 1, …` (the scheme of `cv.glmnet`'s
/// `sample(rep(seq(nfolds), length = N))`, with the crate's own generator, so
/// the ids differ from R's for the same seed).
pub fn kfold_ids(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let k = k.max(1);
    let mut ids: Vec<usize> = (0..n).map(|i| i % k).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..n).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        ids.swap(i, j);
    }
    ids
}

/// k-fold cross-validation of the path at `lambdas` (`cv.glmnet` with
/// `type.measure = "mse"`, `grouped = TRUE`, unit weights).
///
/// `fold_ids[i]` is the fold of row `i`; any labels can be used (the distinct
/// values define the folds, at least 3 are required like `cv.glmnet`). Each
/// fold is predicted from a path fitted on the other folds with the same
/// `lambdas` and options (each training set is centred and standardised on
/// its own). Use [`kfold_ids`] for seeded random folds, or
/// [`lambda_grid`] on the full data for `glmnet`'s default grid.
pub fn cv_path(
    x: &Mat<f64>,
    y: &Col<f64>,
    lambdas: &[f64],
    fold_ids: &[usize],
    opts: &PathOptions,
) -> Result<CvPath, RegressionError> {
    validate_xy(x, y)?;
    opts.validate()?;
    validate_lambdas(lambdas)?;
    let n = x.nrows();
    if fold_ids.len() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: fold_ids.len(),
        });
    }
    let splits = splits_from_fold_ids(fold_ids);
    let k = splits.len();
    if k < 3 {
        return Err(RegressionError::NumericalError(format!(
            "cross-validation needs at least 3 folds, got {k}"
        )));
    }
    cv_path_splits(x, y, lambdas, &splits, opts, CvMetric::Mse)
}

pub(crate) fn largest_lambda_where(lambdas: &[f64], pred: impl Fn(usize) -> bool) -> usize {
    let mut best: Option<usize> = None;
    for l in 0..lambdas.len() {
        if pred(l) && best.is_none_or(|b| lambdas[l] > lambdas[b]) {
            best = Some(l);
        }
    }
    best.unwrap_or(0)
}

/// SplitMix64 generator (deterministic across platforms).
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound` (`bound > 0`), without modulo bias.
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % bound;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Mat<f64>, Col<f64>) {
        let x = Mat::from_fn(40, 3, |i, j| (((i + 1) * (j + 2) * 7) % 13) as f64 / 3.0);
        let y = Col::from_fn(40, |i| {
            1.0 + x[(i, 0)] - 0.5 * x[(i, 1)] + ((i * 5) % 7) as f64 / 4.0
        });
        (x, y)
    }

    #[test]
    fn lasso_grid_top_is_all_zero() {
        let (x, y) = data();
        let opts = PathOptions::elastic_net(1.0);
        let grid = lambda_grid(&x, &y, &opts, 10, None).unwrap();
        let path = fit_path(&x, &y, &grid, &opts).unwrap();
        assert_eq!(path.df[0], 0);
        assert!(path.df[9] > 0);
        // Just below lambda_max a coefficient enters.
        let p2 = fit_path(&x, &y, &[grid[0] * 0.999], &opts).unwrap();
        assert_eq!(p2.df[0], 1);
    }

    #[test]
    fn zero_lambda_is_ols() {
        let (x, y) = data();
        for alpha in [0.0, 0.5] {
            let path = fit_path(&x, &y, &[0.0], &PathOptions::elastic_net(alpha)).unwrap();
            let ols = crate::solvers::OlsRegressor::builder()
                .with_intercept(true)
                .build();
            use crate::solvers::{FittedRegressor, Regressor};
            let f = ols.fit(&x, &y).unwrap();
            for j in 0..3 {
                assert!((path.coefficients[(j, 0)] - f.result().coefficients[j]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn kfold_ids_balanced_and_deterministic() {
        let a = kfold_ids(23, 5, 7);
        assert_eq!(a, kfold_ids(23, 5, 7));
        for f in 0..5 {
            let c = a.iter().filter(|&&v| v == f).count();
            assert!(c == 4 || c == 5);
        }
    }
}
