//! Partial-residual (component-plus-residual) and added-variable plot data (#80).
//!
//! * [`partial_residuals`]: for every fitted term `j`, the component
//!   `b_j (x_j - mean(x_j))` (uncentered without an intercept) and the partial
//!   residual `e + component`, as R's `residuals(fit, type = "partial")`, which is
//!   what `car::crPlots` draws. For GLMs ([`partial_residuals_glm`]) `e` is the
//!   working residual, as `residuals.glm(type = "partial")`.
//! * [`added_variable`]: for every column `j`, the residuals of `x_j` and of `y`
//!   after (weighted) least squares on the other columns (and the intercept), as
//!   `car::avPlots` (`lsfit(..., wt = weights)`); the slope through the origin of
//!   `y_residual` on `x_residual` is the coefficient `b_j` (Frisch-Waugh-Lovell).

use crate::solvers::penalized_glm::{GlmEngineError, GlmEngineResult, GlmResult};
use crate::solvers::{FittedRegressor, RegressionError};
use faer::{Col, Mat};

/// Component-plus-residual data of one term.
#[derive(Debug, Clone)]
pub struct PartialResiduals {
    /// 0-based feature column.
    pub term: usize,
    /// The feature values.
    pub x: Col<f64>,
    /// The term's contribution `b_j (x_j - mean(x_j))` (no centering without an intercept).
    pub component: Col<f64>,
    /// `residual + component` (R `residuals(type = "partial")`).
    pub partial_residual: Col<f64>,
}

/// Added-variable (partial regression) data of one column.
#[derive(Debug, Clone)]
pub struct AddedVariable {
    /// 0-based feature column.
    pub term: usize,
    /// Residual of `x_j` regressed on the other columns.
    pub x_residual: Col<f64>,
    /// Residual of `y` regressed on the other columns.
    pub y_residual: Col<f64>,
    /// Weighted slope of `y_residual` on `x_residual` through the origin; equals
    /// the full-model coefficient of column `j`.
    pub slope: f64,
}

fn centered_component(b: f64, xj: &[f64], center: bool) -> Vec<f64> {
    let mean = if center {
        xj.iter().sum::<f64>() / xj.len().max(1) as f64
    } else {
        0.0
    };
    xj.iter().map(|&v| b * (v - mean)).collect()
}

/// Component-plus-residual data for a fitted linear model (OLS, WLS, Ridge, ...)
/// on its training data, one entry per non-aliased feature.
///
/// The residual is `y - ŷ` (unweighted, as R's `lm` residuals for a weighted
/// fit) and the centering uses the unweighted column mean, as R's
/// `predict(type = "terms")`.
///
/// # Errors
///
/// [`RegressionError::DimensionMismatch`] when `x` / `y` do not match each other
/// or the fit.
pub fn partial_residuals<F: FittedRegressor + ?Sized>(
    fit: &F,
    x: &Mat<f64>,
    y: &Col<f64>,
) -> Result<Vec<PartialResiduals>, RegressionError> {
    let n = x.nrows();
    let coefs = fit.coefficients();
    if y.nrows() != n || x.ncols() != coefs.nrows() {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: y.nrows(),
        });
    }
    let yhat = fit.predict(x);
    let resid: Vec<f64> = (0..n).map(|i| y[i] - yhat[i]).collect();
    let center = fit.intercept().is_some();
    Ok((0..x.ncols())
        .filter(|&j| coefs[j].is_finite())
        .map(|j| {
            let xj: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
            let comp = centered_component(coefs[j], &xj, center);
            PartialResiduals {
                term: j,
                x: Col::from_fn(n, |i| xj[i]),
                component: Col::from_fn(n, |i| comp[i]),
                partial_residual: Col::from_fn(n, |i| resid[i] + comp[i]),
            }
        })
        .collect())
}

/// Component-plus-residual data for a penalized-GLM-engine fit (the result of
/// `fit_poisson`, `fit_binomial`, ...) on its training data: working residual
/// `(y - μ) g'(μ)` plus `b_j (x_j - mean(x_j))`, as R's
/// `residuals(glm, type = "partial")`.
///
/// `y` / `x` are the data the fit was given (column-major, offset column
/// included). Means are over the rows the fit used; dropped rows are `NaN`.
/// One entry per fitted (non-`NaN`) feature, numbered in the feature order
/// (offset excluded).
///
/// # Errors
///
/// [`GlmEngineError::DimensionMismatch`] / [`GlmEngineError::InvalidInput`] when
/// the data do not match the fit.
pub fn partial_residuals_glm(
    fit: &GlmResult,
    y: &[f64],
    x: &[Vec<f64>],
) -> GlmEngineResult<Vec<PartialResiduals>> {
    let n = y.len();
    if let Some(c) = x.iter().find(|c| c.len() != n) {
        return Err(GlmEngineError::DimensionMismatch {
            y_len: n,
            x_rows: c.len(),
        });
    }
    let offset_idx = fit.offset_column.and_then(|c| c.checked_sub(1));
    let features: Vec<&Vec<f64>> = x
        .iter()
        .enumerate()
        .filter(|(j, _)| Some(*j) != offset_idx)
        .map(|(_, c)| c)
        .collect();
    let coefs = &fit.core.coefficients;
    if features.len() != coefs.len() {
        return Err(GlmEngineError::InvalidInput(format!(
            "partial_residuals_glm: x has {} feature columns (offset excluded) but the fit has {}",
            features.len(),
            coefs.len()
        )));
    }
    let rows = &fit.valid_rows;
    if rows.len() != fit.fitted_values.len() || rows.iter().any(|&r| r >= n) {
        return Err(GlmEngineError::DimensionMismatch {
            y_len: n,
            x_rows: rows.iter().max().map_or(0, |&r| r + 1),
        });
    }
    let theta = match fit.dispersion_rule {
        crate::solvers::penalized_glm::DispersionRule::Given(t) => Some(t),
        _ => fit.core.dispersion,
    };
    let family = fit.family.glm_family(theta);
    let mut working = vec![f64::NAN; n];
    for (k, &r) in rows.iter().enumerate() {
        let mu = fit.fitted_values[k];
        working[r] = (y[r] - mu) * family.link_derivative(mu);
    }
    let center = fit.core.intercept.is_some();
    Ok((0..coefs.len())
        .filter(|&j| coefs[j].is_finite())
        .map(|j| {
            let used: Vec<f64> = rows.iter().map(|&r| features[j][r]).collect();
            let mean = if center {
                used.iter().sum::<f64>() / used.len().max(1) as f64
            } else {
                0.0
            };
            let mut comp = vec![f64::NAN; n];
            for &r in rows {
                comp[r] = coefs[j] * (features[j][r] - mean);
            }
            PartialResiduals {
                term: j,
                x: Col::from_fn(n, |i| features[j][i]),
                component: Col::from_fn(n, |i| comp[i]),
                partial_residual: Col::from_fn(n, |i| working[i] + comp[i]),
            }
        })
        .collect())
}

/// Weighted least-squares residualizer: projects vectors off the column space of
/// `z` with weights `w` (rank-revealing, so collinear columns count once).
pub(crate) struct Residualizer {
    q: Mat<f64>,
    rank: usize,
    sw: Vec<f64>,
}

impl Residualizer {
    pub(crate) fn new(z: &Mat<f64>, w: Option<&[f64]>) -> Self {
        let n = z.nrows();
        let sw: Vec<f64> = (0..n)
            .map(|i| w.map_or(1.0, |w| w[i].max(0.0).sqrt()))
            .collect();
        if z.ncols() == 0 {
            return Self {
                q: Mat::zeros(n, 0),
                rank: 0,
                sw,
            };
        }
        let wz = Mat::from_fn(n, z.ncols(), |i, j| sw[i] * z[(i, j)]);
        let qr = wz.col_piv_qr();
        let r = qr.thin_R();
        let d = r.nrows().min(r.ncols());
        let r00 = if d > 0 { r[(0, 0)].abs() } else { 0.0 };
        let rank = (0..d)
            .take_while(|&i| r[(i, i)].abs() > 1e-10 * r00.max(f64::MIN_POSITIVE))
            .count();
        Self {
            q: qr.compute_thin_Q(),
            rank,
            sw,
        }
    }

    /// Rank of the projection.
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// Raw residuals `v - z β̂` of the weighted fit (`NaN` where the weight is 0).
    pub(crate) fn residuals(&self, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let wv: Vec<f64> = (0..n).map(|i| self.sw[i] * v[i]).collect();
        let mut r = wv.clone();
        for k in 0..self.rank {
            let c: f64 = (0..n).map(|i| self.q[(i, k)] * wv[i]).sum();
            for (i, ri) in r.iter_mut().enumerate() {
                *ri -= c * self.q[(i, k)];
            }
        }
        (0..n)
            .map(|i| {
                if self.sw[i] > 0.0 {
                    r[i] / self.sw[i]
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

/// Added-variable (partial regression) coordinates for every column of `x`, as
/// `car::avPlots` on `lm(y ~ x, weights = weights)`: `x_j` and `y` are each
/// regressed by (weighted) least squares on the other columns plus, with
/// `fit_intercept`, a constant, and the raw residuals are returned.
///
/// # Errors
///
/// [`RegressionError::DimensionMismatch`] when `x`, `y` and `weights` differ in
/// length, [`RegressionError::InvalidWeights`] for a negative or non-finite weight.
pub fn added_variable(
    x: &Mat<f64>,
    y: &Col<f64>,
    weights: Option<&Col<f64>>,
    fit_intercept: bool,
) -> Result<Vec<AddedVariable>, RegressionError> {
    let n = x.nrows();
    if y.nrows() != n || weights.is_some_and(|w| w.nrows() != n) {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: y.nrows(),
        });
    }
    let w: Option<Vec<f64>> = weights.map(|w| (0..n).map(|i| w[i]).collect());
    if w.as_ref()
        .is_some_and(|w| w.iter().any(|&v| !(v.is_finite() && v >= 0.0)))
    {
        return Err(RegressionError::InvalidWeights);
    }
    let off = usize::from(fit_intercept);
    let yv: Vec<f64> = (0..n).map(|i| y[i]).collect();
    Ok((0..x.ncols())
        .map(|j| {
            let others: Vec<usize> = (0..x.ncols()).filter(|&k| k != j).collect();
            let z = Mat::from_fn(n, others.len() + off, |i, k| {
                if fit_intercept && k == 0 {
                    1.0
                } else {
                    x[(i, others[k - off])]
                }
            });
            let res = Residualizer::new(&z, w.as_deref());
            let xj: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
            let ex = res.residuals(&xj);
            let ey = res.residuals(&yv);
            let wt = |i: usize| w.as_ref().map_or(1.0, |w| w[i]);
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for i in 0..n {
                if ex[i].is_finite() && ey[i].is_finite() {
                    sxy += wt(i) * ex[i] * ey[i];
                    sxx += wt(i) * ex[i] * ex[i];
                }
            }
            AddedVariable {
                term: j,
                x_residual: Col::from_fn(n, |i| ex[i]),
                y_residual: Col::from_fn(n, |i| ey[i]),
                slope: if sxx > 0.0 { sxy / sxx } else { f64::NAN },
            }
        })
        .collect())
}
