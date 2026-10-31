//! Generalized least squares for longitudinal data (`nlme::gls`, RMS Ch 7).
//!
//! The Gaussian linear model `y = Xβ + ε` in which the errors of one subject
//! (group) are correlated, `Var(ε_g) = σ² A_g R_g A_g`. `R_g` comes from a
//! [`GlsCorrelation`] structure (compound symmetry, AR(1), continuous-time
//! AR(1), spatial-type functions of the time distance, unstructured) and the
//! diagonal `A_g` from a [`GlsVariance`] function (`varIdent`, `varPower`,
//! `varExp`). Observations of different groups are independent.
//!
//! The correlation and variance parameters maximise the REML (default) or ML
//! log-likelihood with β and σ profiled out, in nlme's unconstrained
//! parameterisation and from nlme's starting values. Approximate intervals of
//! the variance–covariance parameters invert the numerical Hessian of the
//! full log-likelihood in `(θ, log σ)`, as `intervals.gls`.
//!
//! ```rust,ignore
//! use anofox_regression::solvers::gls::{GlsCorrelation, GlsRegressor};
//!
//! let fit = GlsRegressor::builder()
//!     .correlation(GlsCorrelation::ContinuousAr1)
//!     .build()
//!     .fit(&x, &y, &subject, Some(&time))?;
//! println!("phi = {:?}, AIC = {}", fit.correlation_params(), fit.aic());
//! ```

use crate::diagnostics::accuracy::quantile7;
use crate::solvers::traits::RegressionError;
use basin::{
    CostFunction, Executor, FirstOrderState, Gradient, Lbfgs, MoreThuente, NelderMead,
    SimplexProgress, Termination,
};
use faer::{Col, Mat};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal, StudentsT};
use std::collections::HashMap;

/// Within-group correlation structure `R_g` (nlme `corStruct`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlsCorrelation {
    /// Independent errors (`R_g = I`).
    #[default]
    Independent,
    /// Equal correlation `ρ` between any two observations of a group
    /// (`corCompSymm`).
    CompoundSymmetry,
    /// `φ^|k_i − k_j|` for integer times `k` (`corAR1`); without times the
    /// position within the group.
    Ar1,
    /// `φ^|t_i − t_j|` for continuous times (`corCAR1`).
    ContinuousAr1,
    /// `(1 − nugget) exp(−d/range)` with `d = |t_i − t_j|` (`corExp`).
    Exponential {
        /// Estimate a nugget effect.
        nugget: bool,
    },
    /// `(1 − nugget) exp(−(d/range)²)` (`corGaus`).
    Gaussian {
        /// Estimate a nugget effect.
        nugget: bool,
    },
    /// `(1 − nugget)(1 − d/range)` for `d < range`, else 0 (`corLin`).
    Linear {
        /// Estimate a nugget effect.
        nugget: bool,
    },
    /// `(1 − nugget)(1 − 1.5 d/range + 0.5 (d/range)³)` for `d < range`,
    /// else 0 (`corSpher`).
    Spherical {
        /// Estimate a nugget effect.
        nugget: bool,
    },
    /// A free correlation between every two time positions (`corSymm`). The
    /// positions are the distinct times, in increasing order (or the position
    /// within the group without times).
    Unstructured,
}

impl GlsCorrelation {
    fn spatial(self) -> Option<(SpatialKind, bool)> {
        match self {
            Self::Exponential { nugget } => Some((SpatialKind::Exponential, nugget)),
            Self::Gaussian { nugget } => Some((SpatialKind::Gaussian, nugget)),
            Self::Linear { nugget } => Some((SpatialKind::Linear, nugget)),
            Self::Spherical { nugget } => Some((SpatialKind::Spherical, nugget)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialKind {
    Exponential,
    Gaussian,
    Linear,
    Spherical,
}

impl SpatialKind {
    /// corLin and corSpher parameterise `range − minD`.
    fn offset_range(self) -> bool {
        matches!(self, Self::Linear | Self::Spherical)
    }

    /// Correlation at the scaled distance `r = d/range`.
    fn rho(self, r: f64) -> f64 {
        match self {
            Self::Exponential => (-r).exp(),
            Self::Gaussian => (-r * r).exp(),
            Self::Linear if r < 1.0 => 1.0 - r,
            Self::Spherical if r < 1.0 => 1.0 - 1.5 * r + 0.5 * r.powi(3),
            _ => 0.0,
        }
    }
}

/// Variance function: the standard deviation of observation `i` is
/// `σ · s_i` (nlme `varFunc`).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GlsVariance {
    /// `s_i = 1`.
    #[default]
    Constant,
    /// A separate multiplier per stratum (`varIdent`). The stratum of the
    /// first observation is the reference (`s = 1`); the others follow in
    /// order of first appearance.
    Ident(Vec<usize>),
    /// `s_i = |v_i|^δ` for a covariate `v` (`varPower(form = ~ v)`).
    Power(Vec<f64>),
    /// `s_i = exp(δ v_i)` (`varExp(form = ~ v)`).
    Exponential(Vec<f64>),
}

/// Estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlsMethod {
    /// Restricted maximum likelihood (nlme default).
    #[default]
    Reml,
    /// Maximum likelihood (needed to compare models with different fixed
    /// effects by likelihood ratio).
    Ml,
}

/// Residual scale (nlme `resid(type = )`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlsResidualType {
    /// `y − Xβ̂`.
    Response,
    /// Response residuals divided by `σ̂ s_i`.
    Pearson,
    /// Pearson residuals decorrelated by the inverse Cholesky factor of
    /// `R_g`; independent standard normal under the model.
    #[default]
    Normalized,
}

/// Generalized least squares estimator, see the [module docs](self).
#[derive(Debug, Clone)]
pub struct GlsRegressor {
    correlation: GlsCorrelation,
    variance: GlsVariance,
    method: GlsMethod,
    with_intercept: bool,
    confidence_level: f64,
    max_iterations: usize,
}

/// Builder of [`GlsRegressor`].
#[derive(Debug, Clone)]
pub struct GlsRegressorBuilder {
    inner: GlsRegressor,
}

impl GlsRegressor {
    /// A builder with independent errors, constant variance, REML, an
    /// intercept and 95% intervals.
    pub fn builder() -> GlsRegressorBuilder {
        GlsRegressorBuilder {
            inner: GlsRegressor {
                correlation: GlsCorrelation::Independent,
                variance: GlsVariance::Constant,
                method: GlsMethod::Reml,
                with_intercept: true,
                confidence_level: 0.95,
                max_iterations: 500,
            },
        }
    }
}

impl GlsRegressorBuilder {
    /// The within-group correlation structure.
    pub fn correlation(mut self, correlation: GlsCorrelation) -> Self {
        self.inner.correlation = correlation;
        self
    }

    /// The variance function.
    pub fn variance(mut self, variance: GlsVariance) -> Self {
        self.inner.variance = variance;
        self
    }

    /// REML or ML.
    pub fn method(mut self, method: GlsMethod) -> Self {
        self.inner.method = method;
        self
    }

    /// Prepend an intercept column to `x` (default `true`).
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.inner.with_intercept = include;
        self
    }

    /// Level of the coefficient and variance-parameter intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.inner.confidence_level = level;
        self
    }

    /// Maximum L-BFGS iterations for the variance–covariance parameters.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.inner.max_iterations = max_iter;
        self
    }

    /// Finish the builder.
    pub fn build(self) -> GlsRegressor {
        self.inner
    }
}

// ============================================================ problem setup ==

/// Which parameterisation of the unstructured correlation a parameter
/// vector uses: optimisation runs on unit-diagonal Cholesky rows (always
/// positive definite), the intervals on Fisher's `z = log((1+r)/(1−r))` of
/// each correlation, as nlme's `corNatural`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scale {
    Optimisation,
    Natural,
}

/// The data of one fit in a form the likelihood can evaluate quickly.
struct Problem<'a> {
    x: Mat<f64>,
    y: &'a Col<f64>,
    /// Observation indices of each group, in data order.
    groups: Vec<Vec<usize>>,
    /// Time of each observation (or its position within the group).
    time: Vec<f64>,
    /// Integer position of each observation (AR(1) lags, unstructured
    /// positions).
    position: Vec<usize>,
    correlation: GlsCorrelation,
    variance: &'a GlsVariance,
    /// Stratum index per observation for `varIdent` (0 = reference).
    stratum: Vec<usize>,
    n_strata: usize,
    /// Size of the unstructured correlation matrix.
    n_positions: usize,
    max_group: usize,
    /// Smallest within-group distance.
    min_d: f64,
    reml: bool,
}

/// Compact ids to `0..K` in order of first appearance.
fn compact(ids: &[usize]) -> (Vec<usize>, usize) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let idx = ids
        .iter()
        .map(|&g| {
            let next = map.len();
            *map.entry(g).or_insert(next)
        })
        .collect();
    (idx, map.len())
}

fn invalid(msg: impl Into<String>) -> RegressionError {
    RegressionError::NumericalError(msg.into())
}

impl<'a> Problem<'a> {
    fn new(
        reg: &'a GlsRegressor,
        x: &Mat<f64>,
        y: &'a Col<f64>,
        group: &[usize],
        time: Option<&[f64]>,
    ) -> Result<Self, RegressionError> {
        let n = y.nrows();
        check_lengths(reg, x, n, group, time)?;
        let x = if reg.with_intercept {
            Mat::from_fn(
                n,
                x.ncols() + 1,
                |i, j| {
                    if j == 0 {
                        1.0
                    } else {
                        x[(i, j - 1)]
                    }
                },
            )
        } else {
            x.to_owned()
        };
        if n <= x.ncols() {
            return Err(RegressionError::InsufficientObservations {
                needed: x.ncols() + 1,
                got: n,
            });
        }
        let (gid, n_groups) = compact(group);
        let mut groups = vec![Vec::new(); n_groups];
        for (i, &g) in gid.iter().enumerate() {
            groups[g].push(i);
        }
        let mut within = vec![0.0; n];
        for members in &groups {
            for (k, &i) in members.iter().enumerate() {
                within[i] = k as f64;
            }
        }
        let time = time.map_or(within, <[f64]>::to_vec);
        let (position, n_positions) = positions(reg.correlation, &time)?;
        let (stratum, n_strata) = match &reg.variance {
            GlsVariance::Ident(s) => compact(s),
            _ => (Vec::new(), 1),
        };
        let max_group = groups.iter().map(Vec::len).max().unwrap_or(0);
        let mut problem = Self {
            x,
            y,
            groups,
            time,
            position,
            correlation: reg.correlation,
            variance: &reg.variance,
            stratum,
            n_strata,
            n_positions,
            max_group,
            min_d: 0.0,
            reml: reg.method == GlsMethod::Reml,
        };
        problem.min_d = problem.min_distances().0;
        Ok(problem)
    }

    fn n_cor(&self) -> usize {
        match self.correlation {
            GlsCorrelation::Independent => 0,
            GlsCorrelation::Unstructured => self.n_positions * (self.n_positions - 1) / 2,
            c => match c.spatial() {
                Some((_, nugget)) => 1 + usize::from(nugget),
                None => 1,
            },
        }
    }

    fn n_var(&self) -> usize {
        match self.variance {
            GlsVariance::Constant => 0,
            GlsVariance::Ident(_) => self.n_strata - 1,
            _ => 1,
        }
    }

    /// nlme's starting values on the unconstrained scale.
    fn start(&self) -> Vec<f64> {
        let mut v = match self.correlation {
            GlsCorrelation::Independent => vec![],
            GlsCorrelation::CompoundSymmetry => {
                vec![(1.0 / (self.max_group as f64 - 1.0)).ln()]
            }
            GlsCorrelation::Ar1 => vec![0.0],
            GlsCorrelation::ContinuousAr1 => vec![(0.2_f64 / 0.8).ln()],
            GlsCorrelation::Unstructured => vec![0.0; self.n_cor()],
            c => {
                let (kind, nugget) = c.spatial().expect("spatial structure");
                // corLin / corSpher: range = minD + exp(u) from 1.1 minD
                let (min_all, min_pos) = self.min_distances();
                let start = if kind.offset_range() {
                    0.1 * if min_all > 0.0 { min_all } else { min_pos }
                } else {
                    0.9 * min_pos
                };
                let mut s = vec![start.ln()];
                if nugget {
                    s.push((0.1_f64 / 0.9).ln());
                }
                s
            }
        };
        v.extend(std::iter::repeat_n(0.0, self.n_var()));
        v
    }

    /// Smallest within-group distance (nlme's `minD`) and smallest positive
    /// one (1 without pairs).
    fn min_distances(&self) -> (f64, f64) {
        let (mut all, mut pos) = (f64::INFINITY, f64::INFINITY);
        for members in &self.groups {
            for (a, &i) in members.iter().enumerate() {
                for &j in &members[a + 1..] {
                    let d = (self.time[i] - self.time[j]).abs();
                    all = all.min(d);
                    if d > 0.0 {
                        pos = pos.min(d);
                    }
                }
            }
        }
        let fin = |v: f64| if v.is_finite() { v } else { 1.0 };
        (fin(all), fin(pos))
    }

    /// The spatial `range` of the unconstrained parameter `u`.
    fn range(&self, kind: SpatialKind, u: f64) -> f64 {
        if kind.offset_range() {
            u.exp() + self.min_d
        } else {
            u.exp()
        }
    }

    /// Standard-deviation multiplier of every observation.
    fn sd_multipliers(&self, theta: &[f64]) -> Vec<f64> {
        let n = self.y.nrows();
        match self.variance {
            GlsVariance::Constant => vec![1.0; n],
            GlsVariance::Ident(_) => self
                .stratum
                .iter()
                .map(|&s| if s == 0 { 1.0 } else { theta[s - 1].exp() })
                .collect(),
            GlsVariance::Power(v) => v.iter().map(|vi| vi.abs().powf(theta[0])).collect(),
            GlsVariance::Exponential(v) => v.iter().map(|vi| (theta[0] * vi).exp()).collect(),
        }
    }

    /// Unstructured correlation matrix of the positions, or `None` when the
    /// natural-scale parameters are not positive definite.
    fn unstructured(&self, theta: &[f64], scale: Scale) -> Option<Mat<f64>> {
        let m = self.n_positions;
        let mut r = Mat::<f64>::identity(m, m);
        match scale {
            Scale::Natural => {
                let mut k = 0;
                for i in 0..m {
                    for j in i + 1..m {
                        let v = (theta[k] / 2.0).tanh();
                        r[(i, j)] = v;
                        r[(j, i)] = v;
                        k += 1;
                    }
                }
                cholesky(&r).map(|_| r)
            }
            Scale::Optimisation => {
                let rows = unit_cholesky_rows(theta, m);
                for i in 0..m {
                    for j in i + 1..m {
                        let v: f64 = (0..m).map(|c| rows[(i, c)] * rows[(j, c)]).sum();
                        r[(i, j)] = v;
                        r[(j, i)] = v;
                    }
                }
                Some(r)
            }
        }
    }

    /// Correlation between observations `i` and `j` (`i ≠ j`) of one group.
    fn correlation_of(&self, cor: &[f64], symm: Option<&Mat<f64>>, i: usize, j: usize) -> f64 {
        let dt = (self.time[i] - self.time[j]).abs();
        match self.correlation {
            GlsCorrelation::Independent => 0.0,
            GlsCorrelation::CompoundSymmetry => {
                let inf = -1.0 / (self.max_group as f64 - 1.0);
                let e = cor[0].exp();
                (e + inf) / (e + 1.0)
            }
            GlsCorrelation::Ar1 => {
                let lag = self.position[i].abs_diff(self.position[j]);
                (cor[0] / 2.0).tanh().powi(lag as i32)
            }
            GlsCorrelation::ContinuousAr1 => (1.0 / (1.0 + (-cor[0]).exp())).powf(dt),
            GlsCorrelation::Unstructured => {
                symm.expect("unstructured matrix")[(self.position[i], self.position[j])]
            }
            c => {
                let (kind, nugget) = c.spatial().expect("spatial structure");
                let nug = if nugget {
                    1.0 / (1.0 + (-cor[1]).exp())
                } else {
                    0.0
                };
                (1.0 - nug) * kind.rho(dt / self.range(kind, cor[0]))
            }
        }
    }

    /// Correlation matrix `R_g` of the observations `members`.
    fn group_correlation(
        &self,
        cor: &[f64],
        symm: Option<&Mat<f64>>,
        members: &[usize],
    ) -> Mat<f64> {
        let m = members.len();
        Mat::from_fn(m, m, |a, b| {
            if a == b {
                1.0
            } else {
                self.correlation_of(cor, symm, members[a], members[b])
            }
        })
    }

    /// Whiten the data with `θ`: returns the decorrelated `X̃`, `ỹ`, the
    /// multipliers `s` and `log|Λ|`, or `None` if some `R_g` is singular.
    fn whiten(&self, theta: &[f64], scale: Scale) -> Option<Whitened> {
        let (cor, var) = theta.split_at(self.n_cor());
        let symm = match self.correlation {
            GlsCorrelation::Unstructured => Some(self.unstructured(cor, scale)?),
            _ => None,
        };
        let s = self.sd_multipliers(var);
        if s.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return None;
        }
        let (n, p) = (self.x.nrows(), self.x.ncols());
        let mut xw = Mat::<f64>::zeros(n, p);
        let mut yw = Col::<f64>::zeros(n);
        let mut logdet: f64 = s.iter().map(|v| 2.0 * v.ln()).sum();
        let mut factors = Vec::with_capacity(self.groups.len());
        for members in &self.groups {
            let l = cholesky(&self.group_correlation(cor, symm.as_ref(), members))?;
            logdet += (0..members.len())
                .map(|a| 2.0 * l[(a, a)].ln())
                .sum::<f64>();
            let rhs = Mat::from_fn(members.len(), p + 1, |a, c| {
                let i = members[a];
                let v = if c < p { self.x[(i, c)] } else { self.y[i] };
                v / s[i]
            });
            let z = forward_solve(&l, &rhs);
            for (a, &i) in members.iter().enumerate() {
                for c in 0..p {
                    xw[(i, c)] = z[(a, c)];
                }
                yw[i] = z[(a, p)];
            }
            factors.push(l);
        }
        Some(Whitened {
            xw,
            yw,
            s,
            logdet,
            factors,
        })
    }

    /// Residual degrees of freedom of the likelihood (`N − p` for REML).
    fn n_eff(&self) -> f64 {
        let (n, p) = (self.x.nrows() as f64, self.x.ncols() as f64);
        if self.reml {
            n - p
        } else {
            n
        }
    }

    /// GLS estimate at `θ` with σ profiled out.
    fn profile(&self, theta: &[f64], scale: Scale) -> Option<Profile> {
        let w = self.whiten(theta, scale)?;
        let xtx = w.xw.transpose() * &w.xw;
        let xtx_inv = spd_inverse(&xtx)?;
        let beta = &xtx_inv * (w.xw.transpose() * &w.yw);
        let resid = &w.yw - &w.xw * &beta;
        let rss: f64 = resid.iter().map(|r| r * r).sum();
        let logdet_xtx = 2.0
            * cholesky(&xtx)?
                .diagonal()
                .column_vector()
                .iter()
                .map(|v| v.ln())
                .sum::<f64>();
        Some(Profile {
            beta,
            xtx_inv,
            rss,
            logdet_xtx,
            whitened: w,
        })
    }

    /// Log-likelihood at `θ` and `σ` (profiled when `log_sigma` is `None`).
    fn log_likelihood(&self, theta: &[f64], log_sigma: Option<f64>, scale: Scale) -> Option<f64> {
        let pr = self.profile(theta, scale)?;
        let ne = self.n_eff();
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let core = match log_sigma {
            None => -ne / 2.0 * (ln2pi + 1.0 + (pr.rss / ne).ln()),
            Some(ls) => -ne / 2.0 * (ln2pi + 2.0 * ls) - pr.rss / (2.0 * (2.0 * ls).exp()),
        };
        let reml = if self.reml { pr.logdet_xtx / 2.0 } else { 0.0 };
        let ll = core - pr.whitened.logdet / 2.0 - reml;
        ll.is_finite().then_some(ll)
    }

    /// Optimisation parameters → natural-scale (`corNatural`) parameters.
    fn to_natural(&self, theta: &[f64]) -> Vec<f64> {
        if self.correlation != GlsCorrelation::Unstructured {
            return theta.to_vec();
        }
        let nc = self.n_cor();
        let r = self
            .unstructured(&theta[..nc], Scale::Optimisation)
            .expect("unit Cholesky rows are positive definite");
        let m = self.n_positions;
        let mut out = Vec::with_capacity(theta.len());
        for i in 0..m {
            for j in i + 1..m {
                let v = r[(i, j)];
                out.push(((1.0 + v) / (1.0 - v)).ln());
            }
        }
        out.extend_from_slice(&theta[nc..]);
        out
    }

    /// The natural correlation and variance parameters of `θ` (natural
    /// scale), as `coef(..., unconstrained = FALSE)`.
    fn natural_params(&self, theta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (cor, var) = theta.split_at(self.n_cor());
        let expit = |u: f64| 1.0 / (1.0 + (-u).exp());
        let cor = match self.correlation {
            GlsCorrelation::Independent => vec![],
            GlsCorrelation::CompoundSymmetry => {
                let inf = -1.0 / (self.max_group as f64 - 1.0);
                vec![(cor[0].exp() + inf) / (cor[0].exp() + 1.0)]
            }
            GlsCorrelation::Ar1 => vec![(cor[0] / 2.0).tanh()],
            GlsCorrelation::ContinuousAr1 => vec![expit(cor[0])],
            GlsCorrelation::Unstructured => cor.iter().map(|z| (z / 2.0).tanh()).collect(),
            c => {
                let (kind, _) = c.spatial().expect("spatial structure");
                let mut v = vec![self.range(kind, cor[0])];
                if cor.len() > 1 {
                    v.push(expit(cor[1]));
                }
                v
            }
        };
        let var = match self.variance {
            GlsVariance::Ident(_) => var.iter().map(|u| u.exp()).collect(),
            _ => var.to_vec(),
        };
        (cor, var)
    }
}

fn check_lengths(
    reg: &GlsRegressor,
    x: &Mat<f64>,
    n: usize,
    group: &[usize],
    time: Option<&[f64]>,
) -> Result<(), RegressionError> {
    if x.nrows() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: x.nrows(),
            y_len: n,
        });
    }
    let var_len = match &reg.variance {
        GlsVariance::Constant => n,
        GlsVariance::Ident(s) => s.len(),
        GlsVariance::Power(v) | GlsVariance::Exponential(v) => v.len(),
    };
    if group.len() != n || var_len != n || time.is_some_and(|t| t.len() != n) {
        return Err(invalid(
            "group, time and variance covariates must have one entry per observation",
        ));
    }
    if time.is_some_and(|t| t.iter().any(|v| !v.is_finite())) {
        return Err(invalid("times must be finite"));
    }
    if !(reg.confidence_level > 0.0 && reg.confidence_level < 1.0) {
        return Err(invalid("confidence_level must be in (0, 1)"));
    }
    Ok(())
}

/// Integer positions for AR(1) lags and the unstructured matrix.
fn positions(
    correlation: GlsCorrelation,
    time: &[f64],
) -> Result<(Vec<usize>, usize), RegressionError> {
    match correlation {
        GlsCorrelation::Ar1 => {
            if time.iter().any(|t| t.fract() != 0.0) {
                return Err(invalid("AR(1) needs integer times"));
            }
            let min = time.iter().copied().fold(f64::INFINITY, f64::min);
            Ok((time.iter().map(|t| (t - min) as usize).collect(), 0))
        }
        GlsCorrelation::Unstructured => {
            let mut distinct: Vec<f64> = time.to_vec();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup();
            if distinct.len() < 2 {
                return Err(invalid(
                    "an unstructured correlation needs two or more times",
                ));
            }
            let pos = time
                .iter()
                .map(|t| distinct.partition_point(|d| d < t))
                .collect();
            Ok((pos, distinct.len()))
        }
        _ => Ok((Vec::new(), 0)),
    }
}

/// Rows of `L` (unit last entry before normalisation) from the free lower
/// entries `θ`: row `i` is `(θ_i1, …, θ_i,i−1, 1) / ‖·‖`.
fn unit_cholesky_rows(theta: &[f64], m: usize) -> Mat<f64> {
    let mut l = Mat::<f64>::zeros(m, m);
    let mut k = 0;
    for i in 0..m {
        l[(i, i)] = 1.0;
        for j in 0..i {
            l[(i, j)] = theta[k];
            k += 1;
        }
        let norm: f64 = (0..=i).map(|j| l[(i, j)].powi(2)).sum::<f64>().sqrt();
        for j in 0..=i {
            l[(i, j)] /= norm;
        }
    }
    l
}

struct Whitened {
    xw: Mat<f64>,
    yw: Col<f64>,
    s: Vec<f64>,
    logdet: f64,
    /// Cholesky factor of each `R_g`.
    factors: Vec<Mat<f64>>,
}

struct Profile {
    beta: Col<f64>,
    xtx_inv: Mat<f64>,
    rss: f64,
    logdet_xtx: f64,
    whitened: Whitened,
}

// ======================================================== small linear algebra

/// Lower Cholesky factor of a symmetric positive definite matrix.
fn cholesky(a: &Mat<f64>) -> Option<Mat<f64>> {
    let m = a.nrows();
    let mut l = Mat::<f64>::zeros(m, m);
    for j in 0..m {
        let d = a[(j, j)] - (0..j).map(|k| l[(j, k)].powi(2)).sum::<f64>();
        if !(d > 0.0 && d.is_finite()) {
            return None;
        }
        let ljj = d.sqrt();
        l[(j, j)] = ljj;
        for i in j + 1..m {
            let v = a[(i, j)] - (0..j).map(|k| l[(i, k)] * l[(j, k)]).sum::<f64>();
            l[(i, j)] = v / ljj;
        }
    }
    Some(l)
}

/// Solve `L Z = B` for lower-triangular `L`.
fn forward_solve(l: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let mut z = b.clone();
    for c in 0..b.ncols() {
        for i in 0..l.nrows() {
            let v = z[(i, c)] - (0..i).map(|k| l[(i, k)] * z[(k, c)]).sum::<f64>();
            z[(i, c)] = v / l[(i, i)];
        }
    }
    z
}

/// Inverse of a symmetric positive definite matrix.
fn spd_inverse(a: &Mat<f64>) -> Option<Mat<f64>> {
    let l = cholesky(a)?;
    let m = a.nrows();
    let linv = forward_solve(&l, &Mat::<f64>::identity(m, m));
    Some(linv.transpose() * &linv)
}

// ============================================================ optimisation ===

#[derive(Clone, Copy)]
struct Objective<'a> {
    f: &'a dyn Fn(&[f64]) -> f64,
}

impl CostFunction for Objective<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, p: &Self::Param) -> Result<f64, Self::Error> {
        Ok((self.f)(p))
    }
}

impl Gradient for Objective<'_> {
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Vec<f64>, Self::Error> {
        let mut g = vec![0.0; p.len()];
        for k in 0..p.len() {
            let h = 1e-6 * p[k].abs().max(1.0);
            let (mut a, mut b) = (p.clone(), p.clone());
            a[k] += h;
            b[k] -= h;
            g[k] = ((self.f)(&a) - (self.f)(&b)) / (2.0 * h);
        }
        Ok(g)
    }
}

/// Minimise `f` from `x0` by L-BFGS with a central-difference gradient.
/// When the line search fails (a first step into a singular region), a
/// Nelder–Mead pass moves into the basin and L-BFGS restarts from there.
/// With `simplex_too` (the spatial structures) that second path also runs
/// from `x0` and the better result is kept: their likelihood can have a
/// nearly flat range → ∞ plateau that L-BFGS steps onto from nlme's start,
/// e.g. `corGaus` with a nugget (#210).
fn minimise(
    f: &dyn Fn(&[f64]) -> f64,
    x0: Vec<f64>,
    max_iter: usize,
    simplex_too: bool,
) -> Vec<f64> {
    let (x, failed) = lbfgs(f, x0.clone(), max_iter);
    if !failed && !simplex_too {
        return x;
    }
    let state = SimplexProgress::new(if failed { x.clone() } else { x0 });
    let Ok(res) = Executor::new(Objective { f }, NelderMead::adaptive(), state)
        .max_iter(200 * max_iter as u64)
        .run();
    let y = lbfgs(f, res.best_param().clone(), max_iter).0;
    if failed || f(&y) < f(&x) {
        y
    } else {
        x
    }
}

/// One L-BFGS run: the best point and whether the solver failed.
fn lbfgs(f: &dyn Fn(&[f64]) -> f64, x0: Vec<f64>, max_iter: usize) -> (Vec<f64>, bool) {
    let solver =
        Lbfgs::with_line_search(MoreThuente::new().with_sufficient_decrease_coefficient(1e-4))
            .unbounded()
            .with_m_capacity(7)
            .with_absolute_gradient_tolerance(1e-7)
            .with_absolute_cost_change_tolerance(f64::EPSILON)
            .with_relative_cost_change_tolerance(1e-14);
    let Ok(res) = Executor::new(Objective { f }, solver, FirstOrderState::new(x0))
        .max_iter(max_iter as u64)
        .run();
    let failed = matches!(res.report.termination, Termination::Failed(_));
    (res.best_param().clone(), failed)
}

/// Central-difference Hessian of `f` at `x`.
fn hessian(f: &dyn Fn(&[f64]) -> Option<f64>, x: &[f64]) -> Option<Mat<f64>> {
    let k = x.len();
    let h: Vec<f64> = x.iter().map(|v| 1e-4 * v.abs().max(1.0)).collect();
    let at = |d: &[(usize, f64)]| -> Option<f64> {
        let mut p = x.to_vec();
        for &(i, s) in d {
            p[i] += s * h[i];
        }
        f(&p)
    };
    let f0 = f(x)?;
    let mut hm = Mat::<f64>::zeros(k, k);
    for i in 0..k {
        hm[(i, i)] = (at(&[(i, 1.0)])? - 2.0 * f0 + at(&[(i, -1.0)])?) / (h[i] * h[i]);
        for j in 0..i {
            let v = (at(&[(i, 1.0), (j, 1.0)])?
                - at(&[(i, 1.0), (j, -1.0)])?
                - at(&[(i, -1.0), (j, 1.0)])?
                + at(&[(i, -1.0), (j, -1.0)])?)
                / (4.0 * h[i] * h[j]);
            hm[(i, j)] = v;
            hm[(j, i)] = v;
        }
    }
    Some(hm)
}

// ================================================================== fitting ==

impl GlsRegressor {
    /// Fit the model.
    ///
    /// * `x` — fixed-effects design `(n, p)` without an intercept column
    ///   (one is prepended when `with_intercept` is set).
    /// * `y` — response of length `n`.
    /// * `group` — subject id per observation; errors of different subjects
    ///   are independent. Ids may be any `usize`.
    /// * `time` — the time of each observation for the AR(1), continuous-AR(1),
    ///   spatial and unstructured structures. `None` uses the position of the
    ///   observation within its group (`form = ~ 1 | id`).
    pub fn fit(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        group: &[usize],
        time: Option<&[f64]>,
    ) -> Result<FittedGls, RegressionError> {
        let problem = Problem::new(self, x, y, group, time)?;
        self.check_structure(&problem)?;
        let theta = self.optimise(&problem)?;
        let profile = problem
            .profile(&theta, Scale::Optimisation)
            .ok_or(RegressionError::SingularMatrix)?;
        let natural = problem.to_natural(&theta);
        Ok(self.assemble(&problem, &natural, profile))
    }

    fn check_structure(&self, problem: &Problem<'_>) -> Result<(), RegressionError> {
        if self.correlation == GlsCorrelation::CompoundSymmetry && problem.max_group < 2 {
            return Err(invalid(
                "compound symmetry needs a group with two or more observations",
            ));
        }
        if let GlsVariance::Power(v) = &self.variance {
            if v.iter().any(|vi| *vi == 0.0 || !vi.is_finite()) {
                return Err(invalid("varPower covariate must be finite and non-zero"));
            }
        }
        Ok(())
    }

    fn optimise(&self, problem: &Problem<'_>) -> Result<Vec<f64>, RegressionError> {
        let x0 = problem.start();
        if x0.is_empty() {
            return Ok(x0);
        }
        let f = |t: &[f64]| {
            problem
                .log_likelihood(t, None, Scale::Optimisation)
                .map_or(1e20, |v| -v)
        };
        let spatial = self.correlation.spatial().is_some();
        let theta = minimise(&f, x0, self.max_iterations, spatial);
        if theta.iter().all(|v| v.is_finite())
            && problem.whiten(&theta, Scale::Optimisation).is_some()
        {
            Ok(theta)
        } else {
            Err(invalid("GLS variance–covariance optimisation failed"))
        }
    }

    fn assemble(&self, problem: &Problem<'_>, natural: &[f64], pr: Profile) -> FittedGls {
        let (n, p) = (problem.x.nrows(), problem.x.ncols());
        let ne = problem.n_eff();
        let sigma = (pr.rss / ne).sqrt();
        // nlme's varBeta uses RSS/(N − p) under ML too
        let s2 = pr.rss / (n - p) as f64;
        let vcov = Mat::from_fn(p, p, |r, c| s2 * pr.xtx_inv[(r, c)]);
        let fitted = &problem.x * &pr.beta;
        let residuals = problem.y - &fitted;
        let pearson = Col::from_fn(n, |i| residuals[i] / (sigma * pr.whitened.s[i]));
        let normalized = normalize(&problem.groups, &pr.whitened.factors, &pearson);
        let ll = problem
            .log_likelihood(natural, None, Scale::Natural)
            .unwrap_or(f64::NAN);
        let n_params = p + natural.len() + 1;
        let (correlation_params, variance_params) = problem.natural_params(natural);
        let intervals = self.var_cov_intervals(problem, natural, sigma);
        let df = (n - p) as f64;
        let se = Col::from_fn(p, |j| vcov[(j, j)].sqrt());
        FittedGls {
            coefficients: pr.beta,
            std_errors: se,
            vcov,
            sigma,
            log_likelihood: ll,
            aic: -2.0 * ll + 2.0 * n_params as f64,
            bic: -2.0 * ll + n_params as f64 * ne.ln(),
            correlation_params,
            variance_params,
            intervals,
            fitted,
            residuals,
            pearson,
            normalized,
            groups: problem.groups.clone(),
            df_residual: df,
            confidence_level: self.confidence_level,
            with_intercept: self.with_intercept,
            n_params,
        }
    }

    /// `intervals(fit, which = "var-cov")`.
    fn var_cov_intervals(
        &self,
        problem: &Problem<'_>,
        natural: &[f64],
        sigma: f64,
    ) -> Option<VarCovIntervals> {
        let level = self.confidence_level;
        if natural.is_empty() {
            let nr = problem.n_eff();
            let chi = ChiSquared::new(nr).ok()?;
            let est = sigma * nr.sqrt();
            return Some(VarCovIntervals {
                correlation: vec![],
                variance: vec![],
                sigma: Interval {
                    lower: est / chi.inverse_cdf((1.0 + level) / 2.0).sqrt(),
                    estimate: sigma,
                    upper: est / chi.inverse_cdf((1.0 - level) / 2.0).sqrt(),
                },
            });
        }
        let mut pars = natural.to_vec();
        pars.push(sigma.ln());
        let k = pars.len();
        let f = |q: &[f64]| problem.log_likelihood(&q[..k - 1], Some(q[k - 1]), Scale::Natural);
        let h = hessian(&f, &pars)?;
        let neg = Mat::from_fn(k, k, |r, c| -h[(r, c)]);
        let cov = spd_inverse(&neg)?;
        let zq = Normal::new(0.0, 1.0).ok()?.inverse_cdf((1.0 + level) / 2.0);
        let bound = |sign: f64| -> Vec<f64> {
            (0..k)
                .map(|j| pars[j] + sign * zq * cov[(j, j)].sqrt())
                .collect()
        };
        let (lo, hi) = (bound(-1.0), bound(1.0));
        let (cl, vl) = problem.natural_params(&lo[..k - 1]);
        let (ce, ve) = problem.natural_params(&pars[..k - 1]);
        let (ch, vh) = problem.natural_params(&hi[..k - 1]);
        let zip = |l: Vec<f64>, e: Vec<f64>, h: Vec<f64>| -> Vec<Interval> {
            l.into_iter()
                .zip(e)
                .zip(h)
                .map(|((lower, estimate), upper)| Interval {
                    lower: lower.min(upper),
                    estimate,
                    upper: upper.max(lower),
                })
                .collect()
        };
        Some(VarCovIntervals {
            correlation: zip(cl, ce, ch),
            variance: zip(vl, ve, vh),
            sigma: Interval {
                lower: lo[k - 1].exp(),
                estimate: sigma,
                upper: hi[k - 1].exp(),
            },
        })
    }
}

/// Decorrelate the Pearson residuals group by group: `L_g⁻¹ r_g`.
fn normalize(groups: &[Vec<usize>], factors: &[Mat<f64>], pearson: &Col<f64>) -> Col<f64> {
    let mut out = Col::<f64>::zeros(pearson.nrows());
    for (members, l) in groups.iter().zip(factors) {
        let r = Mat::from_fn(members.len(), 1, |a, _| pearson[members[a]]);
        let z = forward_solve(l, &r);
        for (a, &i) in members.iter().enumerate() {
            out[i] = z[(a, 0)];
        }
    }
    out
}

// ================================================================== results ==

/// A point estimate with its confidence limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Lower limit.
    pub lower: f64,
    /// Estimate.
    pub estimate: f64,
    /// Upper limit.
    pub upper: f64,
}

/// Approximate intervals of the variance–covariance parameters
/// (`intervals(fit, which = "var-cov")`).
#[derive(Debug, Clone, PartialEq)]
pub struct VarCovIntervals {
    /// Correlation parameters, as [`FittedGls::correlation_params`].
    pub correlation: Vec<Interval>,
    /// Variance-function parameters, as [`FittedGls::variance_params`].
    pub variance: Vec<Interval>,
    /// Residual standard error.
    pub sigma: Interval,
}

/// One distance class of an empirical [`FittedGls::variogram`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariogramBin {
    /// Mean semivariogram `(r_i − r_j)² / 2` of the pairs.
    pub variog: f64,
    /// Median distance of the pairs.
    pub dist: f64,
    /// Number of pairs.
    pub n_pairs: usize,
}

/// A fitted [`GlsRegressor`].
#[derive(Debug, Clone)]
pub struct FittedGls {
    coefficients: Col<f64>,
    std_errors: Col<f64>,
    vcov: Mat<f64>,
    sigma: f64,
    log_likelihood: f64,
    aic: f64,
    bic: f64,
    correlation_params: Vec<f64>,
    variance_params: Vec<f64>,
    intervals: Option<VarCovIntervals>,
    fitted: Col<f64>,
    residuals: Col<f64>,
    pearson: Col<f64>,
    normalized: Col<f64>,
    groups: Vec<Vec<usize>>,
    df_residual: f64,
    confidence_level: f64,
    with_intercept: bool,
    n_params: usize,
}

impl FittedGls {
    /// Coefficients, intercept first when fitted with one.
    pub fn coefficients(&self) -> &Col<f64> {
        &self.coefficients
    }

    /// `σ̂² (XᵀΛ⁻¹X)⁻¹`.
    pub fn vcov(&self) -> &Mat<f64> {
        &self.vcov
    }

    /// Standard errors of the coefficients.
    pub fn std_errors(&self) -> &Col<f64> {
        &self.std_errors
    }

    /// t statistics of the coefficients.
    pub fn t_values(&self) -> Col<f64> {
        Col::from_fn(self.coefficients.nrows(), |j| {
            self.coefficients[j] / self.std_errors[j]
        })
    }

    /// Two-sided p-values on `N − p` degrees of freedom (`summary.gls`).
    pub fn p_values(&self) -> Col<f64> {
        let t = StudentsT::new(0.0, 1.0, self.df_residual).expect("positive df");
        let tv = self.t_values();
        Col::from_fn(tv.nrows(), |j| 2.0 * t.sf(tv[j].abs()))
    }

    /// Confidence intervals `(lower, upper)` of the coefficients on `N − p`
    /// degrees of freedom (`intervals(fit, which = "coef")`).
    pub fn conf_int(&self) -> (Col<f64>, Col<f64>) {
        let t = StudentsT::new(0.0, 1.0, self.df_residual).expect("positive df");
        let q = t.inverse_cdf((1.0 + self.confidence_level) / 2.0);
        let p = self.coefficients.nrows();
        (
            Col::from_fn(p, |j| self.coefficients[j] - q * self.std_errors[j]),
            Col::from_fn(p, |j| self.coefficients[j] + q * self.std_errors[j]),
        )
    }

    /// Residual standard error `σ̂`.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Maximised REML or ML log-likelihood (`logLik`).
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// `−2ℓ + 2k` with `k` = coefficients + variance–covariance parameters + σ.
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// `−2ℓ + k log N'`, `N' = N − p` for REML.
    pub fn bic(&self) -> f64 {
        self.bic
    }

    /// Number of estimated parameters `k`.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Residual degrees of freedom `N − p`.
    pub fn df_residual(&self) -> f64 {
        self.df_residual
    }

    /// Correlation parameters on their natural scale: `ρ` (compound
    /// symmetry), `φ` (AR(1), continuous AR(1)), `range` and optionally
    /// `nugget` (spatial), or the correlations `r_12, r_13, …, r_23, …`
    /// (unstructured).
    pub fn correlation_params(&self) -> &[f64] {
        &self.correlation_params
    }

    /// Variance-function parameters: the standard-deviation ratios of the
    /// non-reference strata (`varIdent`) or `δ` (`varPower`, `varExp`).
    pub fn variance_params(&self) -> &[f64] {
        &self.variance_params
    }

    /// Approximate intervals of the variance–covariance parameters; `None`
    /// when the Hessian of the log-likelihood is not negative definite.
    pub fn var_cov_intervals(&self) -> Option<&VarCovIntervals> {
        self.intervals.as_ref()
    }

    /// Fitted values `Xβ̂`.
    pub fn fitted_values(&self) -> &Col<f64> {
        &self.fitted
    }

    /// Residuals on the given scale.
    pub fn residuals(&self, kind: GlsResidualType) -> &Col<f64> {
        match kind {
            GlsResidualType::Response => &self.residuals,
            GlsResidualType::Pearson => &self.pearson,
            GlsResidualType::Normalized => &self.normalized,
        }
    }

    fn design_row(&self, x: &Mat<f64>, i: usize) -> Vec<f64> {
        let mut row = Vec::with_capacity(self.coefficients.nrows());
        if self.with_intercept {
            row.push(1.0);
        }
        row.extend((0..x.ncols()).map(|j| x[(i, j)]));
        row
    }

    /// Predicted means for new rows of `x` (without the intercept column).
    pub fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        self.predict_with_se(x).0
    }

    /// Predicted means and their standard errors `√(xᵀ V x)`.
    pub fn predict_with_se(&self, x: &Mat<f64>) -> (Col<f64>, Col<f64>) {
        let p = self.coefficients.nrows();
        let mut mean = Col::<f64>::zeros(x.nrows());
        let mut se = Col::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let r = self.design_row(x, i);
            mean[i] = (0..p).map(|j| r[j] * self.coefficients[j]).sum();
            let q: f64 = (0..p)
                .map(|a| (0..p).map(|b| r[a] * self.vcov[(a, b)] * r[b]).sum::<f64>())
                .sum();
            se[i] = q.sqrt();
        }
        (mean, se)
    }

    /// Empirical semivariogram of the residuals against the within-group
    /// distance `|t_i − t_j|` (`Variogram(fit, form = ~ t | id)`).
    ///
    /// Pairs are pooled into `nint` classes at quantiles of the distances, as
    /// nlme's `collapse = "quantiles"`; with no more distinct distances than
    /// `nint`, every distinct distance is its own class.
    pub fn variogram(
        &self,
        time: &[f64],
        kind: GlsResidualType,
        nint: usize,
    ) -> Result<Vec<VariogramBin>, RegressionError> {
        let res = self.residuals(kind);
        if time.len() != res.nrows() {
            return Err(invalid("time must have one entry per observation"));
        }
        let mut pairs = Vec::new();
        for members in &self.groups {
            for (a, &i) in members.iter().enumerate() {
                for &j in &members[a + 1..] {
                    pairs.push(((time[i] - time[j]).abs(), (res[i] - res[j]).powi(2) / 2.0));
                }
            }
        }
        if pairs.is_empty() {
            return Err(invalid("no group has two or more observations"));
        }
        Ok(collapse_variogram(pairs, nint.max(1)))
    }
}

/// nlme's `collapse = "quantiles"` of `(distance, semivariance)` pairs.
fn collapse_variogram(pairs: Vec<(f64, f64)>, nint: usize) -> Vec<VariogramBin> {
    let mut dists: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    dists.sort_by(f64::total_cmp);
    let mut distinct = dists.clone();
    distinct.dedup();
    // class of each pair: the distinct distance, or the quantile interval
    // (b_k, b_k+1] (`cut` drops distances equal to the smallest break)
    let class: Box<dyn Fn(f64) -> Option<usize>> = if nint < distinct.len() {
        let mut breaks: Vec<f64> = (0..=nint)
            .map(|k| quantile7(&dists, k as f64 / nint as f64))
            .collect();
        breaks.dedup();
        Box::new(move |d| {
            let k = breaks.partition_point(|b| *b < d);
            (k > 0 && k < breaks.len()).then(|| k - 1)
        })
    } else {
        Box::new(move |d| Some(distinct.partition_point(|b| *b < d)))
    };
    let mut bins: Vec<Vec<(f64, f64)>> = Vec::new();
    for (d, v) in pairs {
        if let Some(k) = class(d) {
            if bins.len() <= k {
                bins.resize(k + 1, Vec::new());
            }
            bins[k].push((d, v));
        }
    }
    bins.into_iter()
        .filter(|b| !b.is_empty())
        .map(|b| {
            let mut ds: Vec<f64> = b.iter().map(|p| p.0).collect();
            ds.sort_by(f64::total_cmp);
            VariogramBin {
                variog: b.iter().map(|p| p.1).sum::<f64>() / b.len() as f64,
                dist: quantile7(&ds, 0.5),
                n_pairs: b.len(),
            }
        })
        .collect()
}
