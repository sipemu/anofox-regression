//! Additive regression with optimal transformations of both sides
//! (`Hmisc::areg`, RMS §16.1).
//!
//! Each predictor and the response are expanded into a restricted cubic
//! spline basis (or kept linear, or dummy-coded), and the first canonical
//! correlation between the two bases gives the transformation of `y` and
//! the additive transformations of the predictors that maximise R². With a
//! linear `y` the fit is ordinary least squares on the predictor bases.

use crate::diagnostics::accuracy::quantile7;
use crate::solvers::RegressionError;
use crate::splines::Rcs;
use faer::{Col, Mat};

/// How a variable enters [`areg`] (`xtype` / `ytype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AregType {
    /// Restricted cubic spline with `nk` default knots (`"s"`).
    #[default]
    Spline,
    /// Untransformed (`"l"`).
    Linear,
    /// Dummy columns for every distinct value but the smallest (`"c"`). A
    /// categorical response has no inverse transformation (`inverse_y`,
    /// `mad` and `med` are `None`).
    Categorical,
}

/// Options of [`areg`].
#[derive(Debug, Clone, PartialEq)]
pub struct AregOptions {
    /// Number of spline knots (`nk`, default 4); 0 makes every spline linear.
    pub nk: usize,
    /// Type of each predictor (default: all [`AregType::Spline`]).
    pub x_types: Option<Vec<AregType>>,
    /// Type of the response.
    pub y_type: AregType,
}

impl Default for AregOptions {
    fn default() -> Self {
        Self {
            nk: 4,
            x_types: None,
            y_type: AregType::Spline,
        }
    }
}

/// How one predictor was expanded.
#[derive(Debug, Clone, PartialEq)]
enum Expansion {
    Linear,
    Spline(Rcs),
    Categorical(Vec<f64>),
}

impl Expansion {
    fn columns(&self, z: &[f64]) -> Mat<f64> {
        match self {
            Self::Linear => Mat::from_fn(z.len(), 1, |i, _| z[i]),
            Self::Spline(rcs) => rcs.basis(z, "z").columns,
            Self::Categorical(levels) => Mat::from_fn(z.len(), levels.len() - 1, |i, j| {
                f64::from(u8::from(z[i] == levels[j + 1]))
            }),
        }
    }
}

fn expansion(z: &[f64], kind: AregType, nk: usize) -> Result<Expansion, RegressionError> {
    let mut distinct = z.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    match kind {
        // too few distinct values for nk knots: linear, as areg intends
        AregType::Spline if nk >= 3 && distinct.len() >= nk => Rcs::from_data(z, nk)
            .map(Expansion::Spline)
            .map_err(|e| RegressionError::NumericalError(e.to_string())),
        AregType::Categorical if distinct.len() >= 2 => Ok(Expansion::Categorical(distinct)),
        AregType::Categorical => Err(RegressionError::NumericalError(
            "a categorical predictor needs two or more levels".to_string(),
        )),
        _ => Ok(Expansion::Linear),
    }
}

/// A fitted [`areg`] model.
#[derive(Debug, Clone)]
pub struct Areg {
    /// Intercept followed by the coefficients of the predictor bases
    /// (`xcoefficients`).
    pub x_coefficients: Col<f64>,
    /// Intercept followed by the coefficients of the response basis
    /// (`ycoefficients`; `[0, 1]` for a linear response).
    pub y_coefficients: Col<f64>,
    /// R² of the transformed response on the transformed predictors.
    pub r_squared: f64,
    /// Basis columns of each predictor (`xdf`).
    pub x_df: Vec<usize>,
    /// Basis columns of the response (`ydf`).
    pub y_df: usize,
    /// Transformed response (`ty`).
    pub transformed_y: Col<f64>,
    /// Transformed predictors, each centered (`tx`, `n × p`).
    pub transformed_x: Mat<f64>,
    /// Linear predictors on the transformed-`y` scale.
    pub linear_predictors: Col<f64>,
    /// `ty − lp`.
    pub residuals: Col<f64>,
    /// Mean `|y − ŷ|` with `ŷ` the back-transformed linear predictor;
    /// `None` when the response transformation is not monotone.
    pub mad: Option<f64>,
    /// Median `|y − ŷ|`.
    pub med: Option<f64>,
    x_expansions: Vec<Expansion>,
    y_expansion: Expansion,
    x_design: Mat<f64>,
    y_design: Mat<f64>,
    y: Vec<f64>,
}

/// Bootstrap optimism of an [`Areg`] fit ([`Areg::bootstrap`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AregBootstrap {
    /// Mean optimism of R² (`r2opt`).
    pub r2_optimism: f64,
    /// Optimism-corrected R² (`r2boot`).
    pub r2_corrected: f64,
    /// Mean optimism of the mean |error| (`madopt`).
    pub mad_optimism: f64,
    /// Corrected mean |error| (`madboot`).
    pub mad_corrected: f64,
    /// Mean optimism of the median |error| (`medopt`).
    pub med_optimism: f64,
    /// Corrected median |error| (`medboot`).
    pub med_corrected: f64,
}

/// `Hmisc::areg(x, y, xtype, ytype, nk)`: additive regression with optimal
/// transformations of both sides. Each predictor and the response are
/// expanded into a restricted cubic spline basis (or kept linear, or
/// dummy-coded), and the first canonical correlation between the two bases
/// gives the transformation of `y` and the additive transformations of the
/// predictors that maximise R². With a linear `y` the fit is ordinary least
/// squares on the predictor bases.
pub fn areg(x: &Mat<f64>, y: &[f64], options: &AregOptions) -> Result<Areg, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if y.len() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: y.len(),
        });
    }
    let x_types = options.x_types.clone().unwrap_or(vec![AregType::Spline; p]);
    if x_types.len() != p {
        return Err(RegressionError::NumericalError(
            "x_types needs one entry per predictor".to_string(),
        ));
    }
    let x_expansions = (0..p)
        .map(|j| {
            let col: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
            expansion(&col, x_types[j], options.nk)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let blocks: Vec<Mat<f64>> = (0..p)
        .map(|j| {
            let col: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
            x_expansions[j].columns(&col)
        })
        .collect();
    let x_df: Vec<usize> = blocks.iter().map(|b| b.ncols()).collect();
    let x_design = hcat(&blocks);
    let y_expansion = expansion(y, options.y_type, options.nk)?;
    let y_design = y_expansion.columns(y);
    if n <= x_design.ncols() + y_design.ncols() + 1 {
        return Err(RegressionError::InsufficientObservations {
            needed: x_design.ncols() + y_design.ncols() + 2,
            got: n,
        });
    }
    let (x_coefficients, y_coefficients, r_squared) = match y_expansion {
        Expansion::Linear => least_squares(&x_design, y)?,
        _ => canonical(&x_design, &y_design)?,
    };
    let mut fit = Areg {
        transformed_y: affine(&y_design, &y_coefficients),
        linear_predictors: affine(&x_design, &x_coefficients),
        residuals: Col::zeros(n),
        transformed_x: Mat::zeros(n, p),
        x_coefficients,
        y_coefficients,
        r_squared,
        y_df: y_design.ncols(),
        x_df,
        mad: None,
        med: None,
        x_expansions,
        y_expansion,
        x_design,
        y_design,
        y: y.to_vec(),
    };
    fit.residuals = &fit.transformed_y - &fit.linear_predictors;
    fit.transformed_x = fit.transformed_predictors();
    let lp: Vec<f64> = fit.linear_predictors.iter().copied().collect();
    if let Some(yhat) = fit.inverse_y(&lp) {
        let (mad, med) = abs_error(y, &yhat, 0..n);
        fit.mad = Some(mad);
        fit.med = Some(med);
    }
    Ok(fit)
}

/// Columns side by side.
fn hcat(blocks: &[Mat<f64>]) -> Mat<f64> {
    let n = blocks.first().map_or(0, |b| b.nrows());
    let cols: Vec<(usize, usize)> = blocks
        .iter()
        .enumerate()
        .flat_map(|(b, m)| (0..m.ncols()).map(move |c| (b, c)))
        .collect();
    Mat::from_fn(n, cols.len(), |i, k| {
        let (b, c) = cols[k];
        blocks[b][(i, c)]
    })
}

/// `b₀ + Zb`.
fn affine(z: &Mat<f64>, coef: &Col<f64>) -> Col<f64> {
    Col::from_fn(z.nrows(), |i| {
        coef[0] + (0..z.ncols()).map(|j| z[(i, j)] * coef[j + 1]).sum::<f64>()
    })
}

/// Centered copy of `z` and its column means.
fn center(z: &Mat<f64>) -> (Mat<f64>, Vec<f64>) {
    let n = z.nrows() as f64;
    let means: Vec<f64> = (0..z.ncols())
        .map(|j| (0..z.nrows()).map(|i| z[(i, j)]).sum::<f64>() / n)
        .collect();
    (
        Mat::from_fn(z.nrows(), z.ncols(), |i, j| z[(i, j)] - means[j]),
        means,
    )
}

/// Thin QR with an invertible `R`, or an error when `z` is rank deficient.
fn qr(z: &Mat<f64>) -> Result<(Mat<f64>, Mat<f64>), RegressionError> {
    let qr = z.qr();
    let (q, r) = (qr.compute_thin_Q(), qr.thin_R().to_owned());
    let scale = (0..r.ncols()).fold(0.0_f64, |m, j| m.max(r[(j, j)].abs()));
    if (0..r.ncols()).any(|j| r[(j, j)].abs() <= 1e-10 * scale) {
        return Err(RegressionError::SingularMatrix);
    }
    Ok((q, r))
}

/// Solve `R b = u` for upper-triangular `R`.
fn back_solve(r: &Mat<f64>, u: &[f64]) -> Vec<f64> {
    let m = r.ncols();
    let mut b = vec![0.0; m];
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|k| r[(i, k)] * b[k]).sum();
        b[i] = (u[i] - s) / r[(i, i)];
    }
    b
}

/// Intercept-first coefficients `√(n−1) (−Σ b x̄, b)` of a canonical vector.
fn scaled(b: &[f64], means: &[f64], n: usize) -> Col<f64> {
    let k = ((n - 1) as f64).sqrt();
    let icpt = -b.iter().zip(means).map(|(c, m)| c * m).sum::<f64>();
    Col::from_fn(b.len() + 1, |j| k * if j == 0 { icpt } else { b[j - 1] })
}

/// First canonical correlation of `X` and `Y` (`cancor` as used by areg):
/// both variates have unit variance, the `y` variate increases with the
/// first `y` column; returns the `x` and `y` coefficients and `r²`.
pub(crate) fn canonical(
    x: &Mat<f64>,
    y: &Mat<f64>,
) -> Result<(Col<f64>, Col<f64>, f64), RegressionError> {
    let n = x.nrows();
    let (xc, xm) = center(x);
    let (yc, ym) = center(y);
    let (qx, rx) = qr(&xc)?;
    let (qy, ry) = qr(&yc)?;
    let m = qx.transpose() * &qy;
    let svd = m
        .thin_svd()
        .map_err(|_| RegressionError::NumericalError("SVD failed".to_string()))?;
    let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
    let best = (0..s.nrows())
        .max_by(|&a, &b| s[a].total_cmp(&s[b]))
        .expect("non-empty SVD");
    let ux: Vec<f64> = (0..u.nrows()).map(|i| u[(i, best)]).collect();
    let vy: Vec<f64> = (0..v.nrows()).map(|i| v[(i, best)]).collect();
    let mut xb = scaled(&back_solve(&rx, &ux), &xm, n);
    let mut yb = scaled(&back_solve(&ry, &vy), &ym, n);
    if yb[1] < 0.0 {
        xb = -xb;
        yb = -yb;
    }
    Ok((xb, yb, s[best] * s[best]))
}

/// Least squares of `y` on `[1, X]` (linear response).
fn least_squares(x: &Mat<f64>, y: &[f64]) -> Result<(Col<f64>, Col<f64>, f64), RegressionError> {
    let n = x.nrows();
    let design = Mat::from_fn(
        n,
        x.ncols() + 1,
        |i, j| {
            if j == 0 {
                1.0
            } else {
                x[(i, j - 1)]
            }
        },
    );
    let (q, r) = qr(&design)?;
    let yc = Col::from_fn(n, |i| y[i]);
    let qty = q.transpose() * &yc;
    let b = back_solve(&r, &qty.iter().copied().collect::<Vec<_>>());
    let fitted = &q * &qty;
    let mean = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let sse: f64 = (0..n).map(|i| (y[i] - fitted[i]).powi(2)).sum();
    let ycoef = Col::from_fn(2, |j| if j == 0 { 0.0 } else { 1.0 });
    Ok((Col::from_fn(b.len(), |j| b[j]), ycoef, 1.0 - sse / sst))
}

/// Mean and median of `|y_i − ŷ_i|` over `rows`.
fn abs_error(y: &[f64], yhat: &[f64], rows: impl Iterator<Item = usize>) -> (f64, f64) {
    let mut e: Vec<f64> = rows.map(|i| (y[i] - yhat[i]).abs()).collect();
    let mean = e.iter().sum::<f64>() / e.len() as f64;
    e.sort_by(f64::total_cmp);
    (mean, quantile7(&e, 0.5))
}

/// Squared Pearson correlation.
fn cor2(a: &Col<f64>, b: &Col<f64>) -> f64 {
    let n = a.nrows() as f64;
    let (ma, mb) = (a.sum() / n, b.sum() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..a.nrows() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    sab * sab / (saa * sbb)
}

/// Piecewise-linear inverse of a monotone transformation tabulated on a
/// grid, constant beyond its ends (`approx(rule = 2)`).
struct Inverse {
    t: Vec<f64>,
    y: Vec<f64>,
}

impl Inverse {
    fn eval(&self, v: f64) -> f64 {
        let k = self.t.len();
        if v <= self.t[0] {
            return self.y[0];
        }
        if v >= self.t[k - 1] {
            return self.y[k - 1];
        }
        let j = self.t.partition_point(|t| *t <= v).min(k - 1);
        let (t0, t1) = (self.t[j - 1], self.t[j]);
        if t1 == t0 {
            return self.y[j - 1];
        }
        self.y[j - 1] + (v - t0) * (self.y[j] - self.y[j - 1]) / (t1 - t0)
    }

    fn eval_all(&self, v: impl Iterator<Item = f64>) -> Vec<f64> {
        v.map(|t| self.eval(t)).collect()
    }
}

impl Areg {
    /// Knots of the response spline (`yparms`), `None` for a linear response.
    pub fn y_knots(&self) -> Option<&[f64]> {
        match &self.y_expansion {
            Expansion::Spline(rcs) => Some(rcs.knots()),
            _ => None,
        }
    }

    /// The transformation of `y` at new values.
    pub fn transform_y(&self, y: &[f64]) -> Vec<f64> {
        affine(&self.y_expansion.columns(y), &self.y_coefficients)
            .iter()
            .copied()
            .collect()
    }

    /// Linear predictors (transformed-`y` scale) for new rows of `x`.
    pub fn predict(&self, x: &Mat<f64>) -> Vec<f64> {
        let blocks: Vec<Mat<f64>> = self
            .x_expansions
            .iter()
            .enumerate()
            .map(|(j, e)| {
                let col: Vec<f64> = (0..x.nrows()).map(|i| x[(i, j)]).collect();
                e.columns(&col)
            })
            .collect();
        affine(&hcat(&blocks), &self.x_coefficients)
            .iter()
            .copied()
            .collect()
    }

    /// Back-transform values `t` of the transformed response to the `y`
    /// scale (`yinv`): linear interpolation of the transformation on 1000
    /// points over the range of `y`, constant beyond it; the identity for a
    /// linear response. `None` when the transformation is not monotone over
    /// the range of `y`.
    pub fn inverse_y(&self, t: &[f64]) -> Option<Vec<f64>> {
        match self.y_expansion {
            Expansion::Linear => return Some(t.to_vec()),
            Expansion::Categorical(_) => return None,
            Expansion::Spline(_) => {}
        }
        self.inverse_with(&self.y_coefficients)
            .map(|inv| inv.eval_all(t.iter().copied()))
    }

    /// Tabulated inverse of the `y` transformation with coefficients `coef`.
    fn inverse_with(&self, coef: &Col<f64>) -> Option<Inverse> {
        let lo = self.y.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = self.y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let grid: Vec<f64> = (0..1000)
            .map(|k| lo + (hi - lo) * k as f64 / 999.0)
            .collect();
        let t: Vec<f64> = affine(&self.y_expansion.columns(&grid), coef)
            .iter()
            .copied()
            .collect();
        let increasing = t.windows(2).all(|w| w[1] >= w[0]);
        let decreasing = t.windows(2).all(|w| w[1] <= w[0]);
        match (increasing, decreasing) {
            (true, _) => Some(Inverse { t, y: grid }),
            (_, true) => Some(Inverse {
                t: t.into_iter().rev().collect(),
                y: grid.into_iter().rev().collect(),
            }),
            _ => None,
        }
    }

    /// The centered transformation of each predictor (`tx`).
    fn transformed_predictors(&self) -> Mat<f64> {
        let n = self.x_design.nrows();
        let mut out = Mat::<f64>::zeros(n, self.x_df.len());
        let mut start = 1;
        for (j, &m) in self.x_df.iter().enumerate() {
            let z: Vec<f64> = (0..n)
                .map(|i| {
                    (0..m)
                        .map(|c| self.x_design[(i, start - 1 + c)] * self.x_coefficients[start + c])
                        .sum()
                })
                .collect();
            let mean = z.iter().sum::<f64>() / n as f64;
            for i in 0..n {
                out[(i, j)] = z[i] - mean;
            }
            start += m;
        }
        out
    }

    /// Bootstrap estimates of the optimism of R², mean and median |error|
    /// as `areg(B = )`, refitting on each resample in `indices` (rows of the
    /// original data, 0-based) and evaluating on all rows. Needs a spline
    /// response whose transformation stays monotone.
    pub fn bootstrap(&self, indices: &[Vec<usize>]) -> Result<AregBootstrap, RegressionError> {
        let n = self.y.len();
        let (mad, med) = match (self.mad, self.med) {
            (Some(a), Some(b)) if self.y_expansion != Expansion::Linear => (a, b),
            _ => {
                return Err(RegressionError::NumericalError(
                    "bootstrap needs a monotone spline response transformation".to_string(),
                ))
            }
        };
        if indices.is_empty() || indices.iter().flatten().any(|&i| i >= n) {
            return Err(RegressionError::NumericalError(
                "bootstrap indices must be non-empty and in range".to_string(),
            ));
        }
        let (mut r2o, mut mado, mut medo) = (0.0, 0.0, 0.0);
        for s in indices {
            let xs = Mat::from_fn(s.len(), self.x_design.ncols(), |i, j| {
                self.x_design[(s[i], j)]
            });
            let ys = Mat::from_fn(s.len(), self.y_design.ncols(), |i, j| {
                self.y_design[(s[i], j)]
            });
            let (bx, by, r2b) = canonical(&xs, &ys)?;
            let xb = affine(&self.x_design, &bx);
            let yb = affine(&self.y_design, &by);
            r2o += r2b - cor2(&xb, &yb);
            let inv = self.inverse_with(&by).ok_or_else(|| {
                RegressionError::NumericalError(
                    "a bootstrap response transformation is not monotone".to_string(),
                )
            })?;
            let puy = inv.eval_all(xb.iter().copied());
            let (mb, db) = abs_error(&self.y, &puy, s.iter().copied());
            let (ma, da) = abs_error(&self.y, &puy, 0..n);
            mado += mb - ma;
            medo += db - da;
        }
        let b = indices.len() as f64;
        Ok(AregBootstrap {
            r2_optimism: r2o / b,
            r2_corrected: self.r_squared - r2o / b,
            mad_optimism: mado / b,
            mad_corrected: mad - mado / b,
            med_optimism: medo / b,
            med_corrected: med - medo / b,
        })
    }
}
