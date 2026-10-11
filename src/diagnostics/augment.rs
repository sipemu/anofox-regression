//! Per-observation diagnostics of a fitted model, in the shape of R's
//! `broom::augment`.
//!
//! [`augment`] returns, for every training row, the fitted mean, a residual,
//! standardised and studentised residuals, the leverage, Cook's distance and
//! DFFITS. It covers the least-squares models ([`FittedOls`], [`FittedWls`])
//! and the GLMs ([`FittedPoisson`], [`FittedBinomial`], [`FittedLogistic`],
//! [`FittedNegativeBinomial`], [`FittedTweedie`], [`FittedGamma`]).
//!
//! Fits of the penalized GLM engine have their own entry points with the same
//! output: [`augment_glm`] / [`augment_glm_with`] for the [`GlmResult`] of the
//! per-family functions (`penalized_glm::fit_poisson`, ...), and
//! [`FittedPenalizedGlm::augment`](crate::solvers::penalized_glm::FittedPenalizedGlm::augment)
//! for the builder. They follow the same conventions (see [`augment_glm`] for
//! offsets, dropped rows / columns and penalized fits).
//!
//! # Memory
//!
//! The leverages are the squared row norms of the thin `Q` factor of the
//! (weighted) design, `h_i = ‖Q_i·‖²`, so the n×n hat matrix is never formed:
//! memory is O(n·p).
//!
//! # Conventions (R `stats`, 4.6)
//!
//! Let `e_i` be the Pearson residual `√wᵢ (yᵢ − μᵢ) / √V(μᵢ)` (for linear
//! models `√wᵢ (yᵢ − ŷᵢ)`, R's `weighted.residuals`), `h_i` the leverage of
//! the design weighted by the final IRLS weights `wᵢ / (V(μᵢ) g'(μᵢ)²)`, `p` the
//! rank, `n` the number of rows with positive weight, and `φ` the dispersion
//! (`1` for binomial, Poisson and negative binomial; `Σ e² / (n − p)`
//! otherwise, as `summary.glm` / `summary.lm`). With the leave-one-out scale
//! `σ₍ᵢ₎² = (Σ e² − eᵢ² / (1 − hᵢ)) / (n − p − 1)`:
//!
//! | field | formula | R |
//! |---|---|---|
//! | `std_residual` | `rᵢ / √(φ (1 − hᵢ))`, `r` the deviance residual for [`ResidualType::Deviance`], else the Pearson residual | `rstandard` |
//! | `stud_residual` | `eᵢ / (σ₍ᵢ₎ √(1 − hᵢ))`; fixed-dispersion families (binomial, Poisson): `eᵢ / √(1 − hᵢ)` | `rstudent` |
//! | `cooks_d` | `(eᵢ / (1 − hᵢ))² hᵢ / (φ p)` | `cooks.distance` |
//! | `dffits` | `eᵢ √hᵢ / (σ₍ᵢ₎ (1 − hᵢ))`; fixed-dispersion families: `eᵢ √hᵢ / (1 − hᵢ)` | `dffits` |
//!
//! The negative binomial follows R's `glm.nb` objects: `φ = 1` for the
//! standardised residuals and Cook's distance, but the leave-one-out scale for
//! the studentised residuals and DFFITS. Rows with `hᵢ = 1` give `NaN`.

use crate::core::{GlmFamily, RegressionResult};
use crate::solvers::penalized_glm::{DispersionRule, GlmEngineError, GlmEngineResult, GlmResult};
use crate::solvers::{
    FittedBinomial, FittedGamma, FittedLogistic, FittedNegativeBinomial, FittedOls, FittedPoisson,
    FittedRegressor, FittedTweedie, FittedWls, RegressionError,
};
use faer::{Col, Mat};

/// Which residual [`Augment::residual`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualType {
    /// Response residual `y − μ` (default for linear models).
    Response,
    /// Pearson residual `√w (y − μ) / √V(μ)`.
    Pearson,
    /// Deviance residual `sign(y − μ) √dᵢ` (default for GLMs, as R).
    Deviance,
}

impl ResidualType {
    /// Stable lower-case name: `"response"`, `"pearson"` or `"deviance"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ResidualType::Response => "response",
            ResidualType::Pearson => "pearson",
            ResidualType::Deviance => "deviance",
        }
    }
}

/// Per-observation diagnostics returned by [`augment`].
///
/// Every vector has one entry per row of the data passed to [`augment`].
#[derive(Debug, Clone)]
pub struct Augment {
    /// Fitted mean on the response scale (`μ̂`, or `ŷ` for linear models).
    pub fitted: Col<f64>,
    /// Residual of type [`Augment::resid_type`].
    pub residual: Col<f64>,
    /// The residual type of [`Augment::residual`].
    pub resid_type: ResidualType,
    /// Standardised (internally studentised) residual, R's `rstandard`.
    pub std_residual: Col<f64>,
    /// Externally studentised residual, R's `rstudent`.
    pub stud_residual: Col<f64>,
    /// Leverage (diagonal of the hat matrix), R's `hatvalues`.
    pub leverage: Col<f64>,
    /// Cook's distance, R's `cooks.distance`.
    pub cooks_d: Col<f64>,
    /// DFFITS, R's `dffits`.
    pub dffits: Col<f64>,
}

/// What [`augment`] needs to know about a fit.
///
/// Implemented for the least-squares models and the GLMs; see the module
/// documentation for the list.
pub trait Augmentable {
    /// The description of the fit used by [`augment`].
    fn augment_spec(&self) -> AugmentSpec<'_>;
}

/// The parts of a fit [`augment`] works from.
pub struct AugmentSpec<'a> {
    /// The regression result (fitted values, aliased columns, intercept).
    pub result: &'a RegressionResult,
    /// The GLM family, `None` for a Gaussian linear model.
    pub family: Option<&'a dyn GlmFamily>,
    /// Whether the dispersion is fixed at 1 (binomial, Poisson): the
    /// studentised residuals and DFFITS then use `1` instead of the
    /// leave-one-out scale.
    pub fixed_dispersion: bool,
    /// Whether `φ = 1` is used for the standardised residuals and Cook's
    /// distance (binomial, Poisson, negative binomial).
    pub unit_dispersion: bool,
    /// Prior weights the model was fitted with, used when [`augment`] is
    /// called without weights.
    pub weights: Option<&'a Col<f64>>,
}

impl<'a> AugmentSpec<'a> {
    /// A Gaussian linear model.
    pub fn linear(result: &'a RegressionResult, weights: Option<&'a Col<f64>>) -> Self {
        Self {
            result,
            family: None,
            fixed_dispersion: false,
            unit_dispersion: false,
            weights,
        }
    }

    /// A GLM with the given family; `fixed_dispersion` for binomial / Poisson,
    /// `unit_dispersion` for families whose dispersion is 1 by definition.
    pub fn glm(
        result: &'a RegressionResult,
        family: &'a dyn GlmFamily,
        fixed_dispersion: bool,
        unit_dispersion: bool,
    ) -> Self {
        Self {
            result,
            family: Some(family),
            fixed_dispersion,
            unit_dispersion,
            weights: None,
        }
    }
}

/// Per-observation diagnostics with the default residual type (response for
/// linear models, deviance for GLMs, as `broom::augment`).
///
/// `x`, `y` and `weights` must be the data the model was fitted on (without
/// the intercept column). `weights` are prior weights; `None` uses the fit's
/// own weights (WLS) or unit weights.
pub fn augment<M: Augmentable + ?Sized>(
    fit: &M,
    x: &Mat<f64>,
    y: &Col<f64>,
    weights: Option<&Col<f64>>,
) -> Result<Augment, RegressionError> {
    let resid_type = if fit.augment_spec().family.is_some() {
        ResidualType::Deviance
    } else {
        ResidualType::Response
    };
    augment_with(fit, x, y, weights, resid_type)
}

/// [`augment`] with an explicit residual type.
pub fn augment_with<M: Augmentable + ?Sized>(
    fit: &M,
    x: &Mat<f64>,
    y: &Col<f64>,
    weights: Option<&Col<f64>>,
    resid_type: ResidualType,
) -> Result<Augment, RegressionError> {
    let spec = fit.augment_spec();
    let result = spec.result;
    let n = x.nrows();
    if y.nrows() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: y.nrows(),
        });
    }
    if result.fitted_values.nrows() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: result.fitted_values.nrows(),
        });
    }
    let weights = weights.or(spec.weights);
    if let Some(w) = weights {
        if w.nrows() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: w.nrows(),
            });
        }
        if w.iter().any(|&wi| wi.is_nan() || wi < 0.0) {
            return Err(RegressionError::InvalidWeights);
        }
    }
    // Active design columns (aliased ones dropped), intercept first.
    let with_intercept = result.intercept.is_some();
    let active: Vec<usize> = (0..x.ncols())
        .filter(|&j| !result.aliased.get(j).copied().unwrap_or(false))
        .collect();
    let off = usize::from(with_intercept);
    let k = active.len() + off;
    let design = Mat::from_fn(n, k, |i, j| {
        if with_intercept && j == 0 {
            1.0
        } else {
            x[(i, active[j - off])]
        }
    });
    Ok(augment_core(&Core {
        design: &design,
        y,
        mu: &result.fitted_values,
        family: spec.family,
        fixed_dispersion: spec.fixed_dispersion,
        unit_dispersion: spec.unit_dispersion,
        weights,
        resid_type,
        rank_revealing: false,
    }))
}

/// Everything the diagnostics are computed from, after the model-specific
/// parts (aliased columns, offset, dropped rows) have been resolved.
struct Core<'a> {
    /// Unweighted design over the used rows, intercept column included.
    design: &'a Mat<f64>,
    y: &'a Col<f64>,
    mu: &'a Col<f64>,
    family: Option<&'a dyn GlmFamily>,
    fixed_dispersion: bool,
    unit_dispersion: bool,
    weights: Option<&'a Col<f64>>,
    resid_type: ResidualType,
    /// Determine the rank by a column-pivoted QR (as R's `glm.fit`) instead of
    /// assuming the design has full column rank.
    rank_revealing: bool,
}

/// Relative tolerance on the pivoted-QR diagonal below which a column counts as
/// linearly dependent (R's `glm.fit` uses `1e-11` on its own scaling).
const RANK_TOL: f64 = 1e-10;

fn augment_core(c: &Core<'_>) -> Augment {
    let n = c.y.nrows();
    let (y, mu, family, weights, resid_type) = (c.y, c.mu, c.family, c.weights, c.resid_type);
    let prior = |i: usize| weights.map_or(1.0, |w| w[i]);

    // Variance function and IRLS weight per row.
    let variance = |i: usize| family.map_or(1.0, |f| f.variance(mu[i]));
    let irls_w = Col::from_fn(n, |i| match family {
        None => prior(i),
        Some(f) => {
            let d = f.link_derivative(mu[i]);
            prior(i) / (f.variance(mu[i]) * d * d)
        }
    });

    // Leverage from the thin QR of diag(√W) X: `h_i = ‖Q_i·‖²` over the first
    // `rank` columns of Q.
    let k_cols = c.design.ncols();
    let (leverage, k) = if k_cols == 0 {
        (Col::zeros(n), 0)
    } else {
        let wx = Mat::from_fn(n, k_cols, |i, j| {
            irls_w[i].max(0.0).sqrt() * c.design[(i, j)]
        });
        let (q, rank) = if c.rank_revealing {
            let qr = wx.col_piv_qr();
            let r = qr.thin_R();
            let d = r.nrows().min(r.ncols());
            let r00 = if d > 0 { r[(0, 0)].abs() } else { 0.0 };
            let rank = (0..d)
                .take_while(|&i| r[(i, i)].abs() > RANK_TOL * r00.max(f64::MIN_POSITIVE))
                .count();
            (qr.compute_thin_Q(), rank)
        } else {
            (wx.qr().compute_thin_Q(), k_cols)
        };
        let lev = Col::from_fn(n, |i| {
            if prior(i) > 0.0 {
                (0..rank)
                    .map(|j| q[(i, j)] * q[(i, j)])
                    .sum::<f64>()
                    .min(1.0)
            } else {
                0.0
            }
        });
        (lev, rank)
    };

    let pearson = Col::from_fn(n, |i| prior(i).sqrt() * (y[i] - mu[i]) / variance(i).sqrt());
    let deviance = Col::from_fn(n, |i| match family {
        None => prior(i).sqrt() * (y[i] - mu[i]),
        Some(f) => {
            let d = (prior(i) * f.unit_deviance(y[i], mu[i])).max(0.0);
            (y[i] - mu[i]).signum() * d.sqrt()
        }
    });
    let residual = match resid_type {
        ResidualType::Response => Col::from_fn(n, |i| y[i] - mu[i]),
        ResidualType::Pearson => pearson.clone(),
        ResidualType::Deviance => deviance.clone(),
    };

    let n_obs = (0..n).filter(|&i| prior(i) > 0.0).count() as f64;
    let p = k as f64;
    let df = n_obs - p;
    let sum_e2: f64 = (0..n)
        .filter(|&i| prior(i) > 0.0)
        .map(|i| pearson[i] * pearson[i])
        .sum();
    let phi = if c.unit_dispersion {
        1.0
    } else if df > 0.0 {
        sum_e2 / df
    } else {
        f64::NAN
    };
    let loo_sigma = |i: usize| {
        let h = leverage[i];
        let s2 = (sum_e2 - pearson[i] * pearson[i] / (1.0 - h)) / (df - 1.0);
        if s2 >= 0.0 {
            s2.sqrt()
        } else {
            f64::NAN
        }
    };
    let nan_if_inf = |v: f64| if v.is_finite() { v } else { f64::NAN };

    let std_source = match resid_type {
        ResidualType::Deviance => &deviance,
        _ => &pearson,
    };
    let std_residual = Col::from_fn(n, |i| {
        nan_if_inf(std_source[i] / (phi * (1.0 - leverage[i])).sqrt())
    });
    let scale = |i: usize| {
        if c.fixed_dispersion {
            1.0
        } else {
            loo_sigma(i)
        }
    };
    let stud_residual = Col::from_fn(n, |i| {
        nan_if_inf(pearson[i] / (scale(i) * (1.0 - leverage[i]).sqrt()))
    });
    let cooks_d = Col::from_fn(n, |i| {
        let h = leverage[i];
        nan_if_inf((pearson[i] / (1.0 - h)).powi(2) * h / (phi * p))
    });
    let dffits = Col::from_fn(n, |i| {
        let h = leverage[i];
        nan_if_inf(pearson[i] * h.sqrt() / (scale(i) * (1.0 - h)))
    });

    Augment {
        fitted: mu.clone(),
        residual,
        resid_type,
        std_residual,
        stud_residual,
        leverage,
        cooks_d,
        dffits,
    }
}

impl Augmentable for FittedOls {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        AugmentSpec::linear(self.result(), None)
    }
}

impl Augmentable for FittedWls {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        AugmentSpec::linear(self.result(), Some(self.weights()))
    }
}

impl Augmentable for FittedPoisson {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        AugmentSpec::glm(self.result(), self.family(), true, true)
    }
}

impl Augmentable for FittedBinomial {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        AugmentSpec::glm(self.result(), self.family(), true, true)
    }
}

impl Augmentable for FittedLogistic {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        self.inner().augment_spec()
    }
}

impl Augmentable for FittedNegativeBinomial {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        AugmentSpec::glm(self.result(), self.family(), false, true)
    }
}

impl Augmentable for FittedTweedie {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        AugmentSpec::glm(self.result(), self.family(), false, false)
    }
}

impl Augmentable for FittedGamma {
    fn augment_spec(&self) -> AugmentSpec<'_> {
        self.inner().augment_spec()
    }
}

/// Per-observation diagnostics of a penalized-GLM-engine fit, with deviance
/// residuals (as R's `broom::augment` on a `glm`).
///
/// `y` and `x` must be exactly what was passed to the `penalized_glm::fit_*`
/// function that produced `fit` (column-major `x`, including the offset column
/// if one was used). The output has one entry per input row; rows the fit
/// dropped because of a non-finite value are `NaN` in every field.
///
/// Conventions, matching [`augment`] on the dedicated regressors and R's
/// `glm` / `MASS::glm.nb`:
///
/// * The dispersion follows [`GlmResult::dispersion_rule`]: `Fixed` (binomial,
///   logistic, Poisson) uses `φ = 1` everywhere; `Given` (Negative Binomial,
///   fixed or estimated `theta`) uses `φ = 1` for `rstandard` / Cook's distance
///   and the leave-one-out scale for `rstudent` / DFFITS; `Pearson` (Gamma,
///   Tweedie, Poisson with `estimate_dispersion`, as R's `quasipoisson`) uses
///   the Pearson estimate.
/// * An offset enters only through the fitted means; it is not part of the
///   design whose leverage is computed.
/// * Columns dropped as constant (`NaN` coefficient) are left out of the
///   design, and the rank is determined by a column-pivoted QR, so collinear
///   columns count once (as R's `glm`).
/// * The engine has no prior weights; every used row has weight 1.
///
/// # Errors
///
/// [`GlmEngineError::InvalidInput`] for a penalized fit ([`GlmResult::penalized`]:
/// `lambda > 0` or a non-flat prior). The hat matrix of a posterior mode is
/// `W½X(X'WX + P)⁻¹X'W½`, for which R's `rstandard` / `rstudent` / Cook's
/// distance have no counterpart, so it is rejected rather than reported with
/// the unpenalized formulas. [`GlmEngineError::DimensionMismatch`] when `y` /
/// `x` do not match the fit.
pub fn augment_glm(fit: &GlmResult, y: &[f64], x: &[Vec<f64>]) -> GlmEngineResult<Augment> {
    augment_glm_with(fit, y, x, ResidualType::Deviance)
}

/// [`augment_glm`] with an explicit residual type.
pub fn augment_glm_with(
    fit: &GlmResult,
    y: &[f64],
    x: &[Vec<f64>],
    resid_type: ResidualType,
) -> GlmEngineResult<Augment> {
    let n = y.len();
    for col in x {
        if col.len() != n {
            return Err(GlmEngineError::DimensionMismatch {
                y_len: n,
                x_rows: col.len(),
            });
        }
    }
    let offset_idx = fit.offset_column.and_then(|c| c.checked_sub(1));
    let features: Vec<&Vec<f64>> = x
        .iter()
        .enumerate()
        .filter(|(j, _)| Some(*j) != offset_idx)
        .map(|(_, c)| c)
        .collect();
    augment_glm_impl(
        fit,
        y,
        n,
        features.len(),
        &|i, j| features[j][i],
        resid_type,
    )
}

/// Shared implementation of [`augment_glm_with`] and
/// `FittedPenalizedGlm::augment_with`: `feature(i, j)` is feature `j` (offset
/// excluded) of input row `i`.
pub(crate) fn augment_glm_impl(
    fit: &GlmResult,
    y: &[f64],
    n_rows: usize,
    n_features: usize,
    feature: &dyn Fn(usize, usize) -> f64,
    resid_type: ResidualType,
) -> GlmEngineResult<Augment> {
    if fit.penalized {
        return Err(GlmEngineError::InvalidInput(
            "augment is not defined for a penalized GLM fit (lambda > 0 or a non-flat prior): \
             the standardised residuals, leverage and Cook's distance assume the \
             maximum-likelihood fit"
                .to_string(),
        ));
    }
    if y.len() != n_rows {
        return Err(GlmEngineError::DimensionMismatch {
            y_len: y.len(),
            x_rows: n_rows,
        });
    }
    let coefs = &fit.core.coefficients;
    if n_features != coefs.len() {
        return Err(GlmEngineError::InvalidInput(format!(
            "augment: x has {n_features} feature columns (offset excluded) but the fit has {}",
            coefs.len()
        )));
    }
    let rows = &fit.valid_rows;
    if rows.len() != fit.fitted_values.len() || rows.iter().any(|&r| r >= n_rows) {
        return Err(GlmEngineError::DimensionMismatch {
            y_len: n_rows,
            x_rows: rows.iter().max().map_or(0, |&r| r + 1),
        });
    }

    let with_intercept = fit.core.intercept.is_some();
    let active: Vec<usize> = (0..n_features).filter(|&j| !coefs[j].is_nan()).collect();
    let off = usize::from(with_intercept);
    let m = rows.len();
    let design = Mat::from_fn(m, active.len() + off, |i, j| {
        if with_intercept && j == 0 {
            1.0
        } else {
            feature(rows[i], active[j - off])
        }
    });
    let yv = Col::from_fn(m, |i| y[rows[i]]);
    let mu = Col::from_fn(m, |i| fit.fitted_values[i]);

    let theta = match fit.dispersion_rule {
        DispersionRule::Given(t) => Some(t),
        _ => fit.core.dispersion,
    };
    let family = fit.family.glm_family(theta);
    let (fixed_dispersion, unit_dispersion) = match fit.dispersion_rule {
        DispersionRule::Fixed => (true, true),
        DispersionRule::Given(_) => (false, true),
        DispersionRule::Pearson | DispersionRule::PearsonFlooredAtOne => (false, false),
    };

    let a = augment_core(&Core {
        design: &design,
        y: &yv,
        mu: &mu,
        family: Some(family.as_ref()),
        fixed_dispersion,
        unit_dispersion,
        weights: None,
        resid_type,
        rank_revealing: true,
    });
    if m == n_rows && rows.iter().enumerate().all(|(i, &r)| i == r) {
        return Ok(a);
    }
    let expand = |v: &Col<f64>| {
        let mut out = Col::from_fn(n_rows, |_| f64::NAN);
        for (i, &r) in rows.iter().enumerate() {
            out[r] = v[i];
        }
        out
    };
    Ok(Augment {
        fitted: expand(&a.fitted),
        residual: expand(&a.residual),
        resid_type: a.resid_type,
        std_residual: expand(&a.std_residual),
        stud_residual: expand(&a.stud_residual),
        leverage: expand(&a.leverage),
        cooks_d: expand(&a.cooks_d),
        dffits: expand(&a.dffits),
    })
}
