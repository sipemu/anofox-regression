//! Per-observation diagnostics of a fitted model, in the shape of R's
//! `broom::augment`.
//!
//! [`augment`] returns, for every training row, the fitted mean, a residual,
//! standardised and studentised residuals, the leverage, Cook's distance and
//! DFFITS. It covers the least-squares models ([`FittedOls`], [`FittedWls`])
//! and the GLMs ([`FittedPoisson`], [`FittedBinomial`], [`FittedLogistic`],
//! [`FittedNegativeBinomial`], [`FittedTweedie`], [`FittedGamma`]).
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
    let prior = |i: usize| weights.map_or(1.0, |w| w[i]);
    let mu = &result.fitted_values;
    let family = spec.family;

    // Variance function and IRLS weight per row.
    let variance = |i: usize| family.map_or(1.0, |f| f.variance(mu[i]));
    let irls_w = Col::from_fn(n, |i| match family {
        None => prior(i),
        Some(f) => {
            let d = f.link_derivative(mu[i]);
            prior(i) / (f.variance(mu[i]) * d * d)
        }
    });

    // Leverage from the thin QR of diag(√W) X (active columns only).
    let with_intercept = result.intercept.is_some();
    let active: Vec<usize> = (0..x.ncols())
        .filter(|&j| !result.aliased.get(j).copied().unwrap_or(false))
        .collect();
    let off = usize::from(with_intercept);
    let k = active.len() + off;
    let leverage = if k == 0 {
        Col::zeros(n)
    } else {
        let design = Mat::from_fn(n, k, |i, j| {
            let v = if with_intercept && j == 0 {
                1.0
            } else {
                x[(i, active[j - off])]
            };
            irls_w[i].max(0.0).sqrt() * v
        });
        let q = design.qr().compute_thin_Q();
        Col::from_fn(n, |i| {
            if prior(i) > 0.0 {
                (0..k).map(|j| q[(i, j)] * q[(i, j)]).sum::<f64>().min(1.0)
            } else {
                0.0
            }
        })
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
    let phi = if spec.unit_dispersion {
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
        if spec.fixed_dispersion {
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

    Ok(Augment {
        fitted: mu.clone(),
        residual,
        resid_type,
        std_residual,
        stud_residual,
        leverage,
        cooks_d,
        dffits,
    })
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
