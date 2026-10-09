//! Family-generic penalized GLM engine.
//!
//! The upstream `anofox-regression` crate carries six hand-copied IRLS loops, one
//! per family, none of which supports explicit priors, a corrected penalized
//! covariance, or a real per-family log-likelihood. This module implements the loop
//! once against upstream's [`GlmFamily`] trait and adds those three things.
//!
//! It is also the foundation for the rest of issue #107. A Gaussian prior on a
//! coefficient and a Gaussian random effect are the same object — a quadratic
//! precision block in the normal equations — so the mixed-effects work reuses
//! [`irls`] and [`laplace`] rather than forking them, and the survival work reuses
//! [`laplace`] alone.
//!
//! ```text
//!   design::build ──▶ Design ──▶ build_penalty ──▶ Penalty
//!                                     │
//!                                     ▼
//!                          irls::fit_irls (generic over GlmFamily)
//!                                     │
//!                       information at the mode (X'WX + P)
//!                                     │
//!                                     ▼
//!                          laplace::inference ──▶ SE / z / p / CI
//! ```

use super::error::GlmEngineResult;
use super::types::{GlmFitResult, GlmInferenceResult, PriorSpec, VcovType};
use super::{design, irls, laplace, loglik};
use crate::core::GlmFamily;

use super::design::{ConstantColumnPolicy, Design, DesignSpec};
use super::irls::{IrlsConfig, IrlsFit};
use super::laplace::LaplaceInference;
use super::loglik::LogLikKind;

/// Everything the engine needs for one fit, independent of family.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub compute_inference: bool,
    pub confidence_level: f64,
    /// Legacy uniform ridge (`glm_lambda`), composed additively with `priors`.
    pub lambda: f64,
    /// Per-feature priors in the caller's feature order, optionally prefixed with
    /// an entry for the intercept. Empty means no explicit priors.
    pub priors: Vec<PriorSpec>,
    pub vcov: VcovType,
    /// 1-based index into `x` of an offset column, if any.
    pub offset_column: Option<usize>,
    pub constant_policy: ConstantColumnPolicy,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: Vec::new(),
            vcov: VcovType::default(),
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Keep,
        }
    }
}

/// How the dispersion parameter is obtained for a family.
///
/// Note that the value a family *reports* as its dispersion and the factor that
/// scales its coefficient covariance are not always the same number. For Negative
/// Binomial the reported "dispersion" is `theta`, a shape parameter that already
/// enters the IRLS weights through the variance function `mu + mu^2 / theta`;
/// multiplying the covariance by it again would inflate every standard error by
/// `sqrt(theta)`. [`DispersionRule::covariance_scale`] keeps the two separate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DispersionRule {
    /// Fixed at 1.0 (Poisson, Binomial).
    Fixed,
    /// Pearson chi-squared over residual degrees of freedom, floored at 1.0.
    /// No longer used by any family in this crate (Poisson defaults to
    /// [`Self::Fixed`] like R, and uses [`Self::Pearson`] when asked to estimate
    /// the dispersion); kept for callers of [`fit`].
    PearsonFlooredAtOne,
    /// Pearson chi-squared over residual degrees of freedom, unfloored
    /// (Gamma, Tweedie).
    Pearson,
    /// A shape parameter supplied by the caller (Negative Binomial `theta`). It is
    /// reported as the dispersion but does **not** scale the covariance.
    Given(f64),
}

impl DispersionRule {
    /// The multiplier applied to `(X'WX + P)^-1` when forming the covariance.
    pub fn covariance_scale(&self, estimated: f64) -> f64 {
        match self {
            DispersionRule::Fixed | DispersionRule::Given(_) => 1.0,
            DispersionRule::Pearson | DispersionRule::PearsonFlooredAtOne => estimated,
        }
    }
}

/// A completed engine fit, ready to be mapped onto the crate's result types.
#[derive(Debug, Clone)]
pub struct EngineFit {
    pub design: Design,
    pub irls: IrlsFit,
    pub dispersion: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub inference: Option<LaplaceInference>,
}

impl EngineFit {
    /// Map onto the crate-wide [`GlmFitResult`] shape.
    pub fn to_glm_fit_result(&self) -> GlmFitResult {
        let (coefficients, intercept) = self.design.expand(&self.irls.beta);
        let pseudo_r_squared = if self.irls.null_deviance > 0.0 {
            1.0 - self.irls.deviance / self.irls.null_deviance
        } else {
            0.0
        };

        GlmFitResult {
            coefficients,
            intercept,
            null_deviance: self.irls.null_deviance,
            residual_deviance: self.irls.deviance,
            pseudo_r_squared,
            aic: self.aic,
            n_observations: self.design.n_observations(),
            n_features: self.design.n_features_original,
            iterations: self.irls.iterations,
            converged: self.irls.converged,
            dispersion: Some(self.dispersion),
        }
    }

    /// Map the inference onto [`GlmInferenceResult`], expanding dropped columns
    /// back to `NaN`. Intercept entries are stripped, matching the existing shape.
    pub fn to_glm_inference(&self) -> Option<GlmInferenceResult> {
        let inf = self.inference.as_ref()?;
        let expand = |v: &[f64]| self.design.expand(v).0;
        let intercept = |v: &[f64]| self.design.expand(v).1;

        // What each row of the matrices is. The vectors above are expanded into the
        // original feature order with `NaN` for dropped columns and the intercept
        // stripped; the matrices stay in fitted order, so this is what reconciles the
        // two. A dropped column is absent rather than present-and-`NaN`, so a caller
        // cannot index a row that was never fitted.
        let mut matrix_parameters: Vec<Option<usize>> = Vec::new();
        if self.design.fit_intercept {
            matrix_parameters.push(None);
        }
        matrix_parameters.extend(self.design.retained_columns.iter().map(|&j| Some(j)));

        Some(GlmInferenceResult {
            std_errors: expand(&inf.std_errors),
            z_values: expand(&inf.z_values),
            p_values: expand(&inf.p_values),
            ci_lower: expand(&inf.ci_lower),
            ci_upper: expand(&inf.ci_upper),
            confidence_level: inf.confidence_level,
            intercept_std_error: intercept(&inf.std_errors),
            intercept_z_value: intercept(&inf.z_values),
            intercept_p_value: intercept(&inf.p_values),
            intercept_ci_lower: intercept(&inf.ci_lower),
            intercept_ci_upper: intercept(&inf.ci_upper),
            vcov: Some(inf.vcov.clone()),
            information: Some(self.irls.information.clone()),
            matrix_parameters,
        })
    }
}

/// Null deviance as R's `glm.fit` defines it.
///
/// * With an intercept and no offset: the deviance at `mu = mean(y)` (the closed
///   form of the intercept-only fit, for any link).
/// * With an intercept and an offset: the deviance of the unpenalized
///   intercept-only model **with the offset**, fitted by IRLS (R refits
///   `glm.fit(x = intercept, offset = offset)`).
/// * Without an intercept: the deviance at `mu = linkinv(offset)`, or
///   `linkinv(0)` when there is no offset.
pub fn null_deviance<F: GlmFamily + ?Sized>(
    family: &F,
    y: &[f64],
    offset: Option<&[f64]>,
    fit_intercept: bool,
    config: &IrlsConfig,
) -> GlmEngineResult<f64> {
    let n = y.len();
    match (fit_intercept, offset) {
        (true, None) => Ok(family.null_deviance(y)),
        (true, Some(off)) => {
            let ones = faer::Mat::from_fn(n, 1, |_, _| 1.0);
            let null_fit = irls::fit_irls(
                family,
                &ones,
                y,
                Some(off),
                &super::penalty::Penalty::none(1),
                config,
            )?;
            Ok(null_fit.deviance)
        }
        (false, off) => {
            let mu: Vec<f64> = (0..n)
                .map(|i| family.link_inverse(off.map_or(0.0, |o| o[i])))
                .collect();
            Ok(family.deviance(y, &mu))
        }
    }
}

/// Fit a GLM with the given family, priors and covariance policy.
///
/// `x` is column-major (one inner vector per feature), matching the rest of the
/// crate. The `loglik` argument carries any nuisance parameter the family needs
/// (Negative Binomial `theta`, Gamma/Tweedie dispersion); pass the value estimated
/// for this fit.
pub fn fit<F: GlmFamily + ?Sized>(
    family: &F,
    y: &[f64],
    x: &[Vec<f64>],
    options: &EngineOptions,
    dispersion_rule: DispersionRule,
    loglik_for: impl Fn(f64) -> LogLikKind,
) -> GlmEngineResult<EngineFit> {
    fit_impl(
        family,
        y,
        x,
        options,
        dispersion_rule,
        loglik_for,
        None,
        FitDepth::Full,
    )
}

/// How much of a fit [`fit_impl`] computes beyond the IRLS mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FitDepth {
    /// Null deviance, dispersion, log-likelihood / AIC / BIC and (if requested)
    /// inference: a reportable fit.
    Full,
    /// The IRLS mode only (`beta`, `mu`, deviance). Null deviance, log-likelihood,
    /// AIC and BIC are `NaN`, the dispersion is only meaningful for
    /// [`DispersionRule::Fixed`] / [`DispersionRule::Given`], and no inference is
    /// computed. Used for the intermediate fits of the Negative Binomial theta
    /// alternation, whose results are never reported.
    Probe,
}

/// [`fit`] with an optional warm start (`beta_start`, design-matrix order, as
/// returned in [`IrlsFit::beta`]) and a choice of how much to compute.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_impl<F: GlmFamily + ?Sized>(
    family: &F,
    y: &[f64],
    x: &[Vec<f64>],
    options: &EngineOptions,
    dispersion_rule: DispersionRule,
    loglik_for: impl Fn(f64) -> LogLikKind,
    beta_start: Option<&[f64]>,
    depth: FitDepth,
) -> GlmEngineResult<EngineFit> {
    let design = design::build(&DesignSpec {
        y,
        x,
        fit_intercept: options.fit_intercept,
        offset_column: options.offset_column,
        constant_policy: options.constant_policy,
    })?;

    let penalty = design.build_penalty(&options.priors, options.lambda)?;

    let config = IrlsConfig {
        max_iterations: options.max_iterations as usize,
        tolerance: options.tolerance,
        ..IrlsConfig::default()
    };

    let mut irls = irls::fit_irls_from(
        family,
        &design.matrix,
        &design.y,
        design.offset.as_deref(),
        &penalty,
        &config,
        beta_start,
    )?;

    if depth == FitDepth::Probe {
        irls.null_deviance = f64::NAN;
        let dispersion = match dispersion_rule {
            DispersionRule::Given(v) => v,
            _ => 1.0,
        };
        return Ok(EngineFit {
            design,
            irls,
            dispersion,
            log_likelihood: f64::NAN,
            aic: f64::NAN,
            bic: f64::NAN,
            inference: None,
        });
    }

    irls.null_deviance = null_deviance(
        family,
        &design.y,
        design.offset.as_deref(),
        design.fit_intercept,
        &config,
    )?;

    let n = design.n_observations();
    let p = design.n_params();
    let df_resid = n.saturating_sub(p) as f64;

    let dispersion = match dispersion_rule {
        DispersionRule::Fixed => 1.0,
        DispersionRule::Given(v) => v,
        DispersionRule::Pearson | DispersionRule::PearsonFlooredAtOne => {
            let d = if df_resid > 0.0 {
                let chi2: f64 = design
                    .y
                    .iter()
                    .zip(irls.mu.iter())
                    .map(|(&yi, &mui)| {
                        let v = family.variance(mui);
                        if v > 0.0 {
                            (yi - mui).powi(2) / v
                        } else {
                            0.0
                        }
                    })
                    .sum();
                chi2 / df_resid
            } else {
                1.0
            };
            if dispersion_rule == DispersionRule::PearsonFlooredAtOne {
                d.max(1.0)
            } else {
                d
            }
        }
    };

    let kind = loglik_for(dispersion);
    let log_likelihood = loglik::log_likelihood(kind, &design.y, &irls.mu);
    let k = p + kind.n_nuisance();
    let aic = loglik::aic(log_likelihood, k);
    let bic = loglik::bic(log_likelihood, k, n);

    let inference = if options.compute_inference {
        Some(laplace::inference(
            &irls.beta,
            &irls.information,
            Some(&irls.unpenalized_information),
            dispersion_rule.covariance_scale(dispersion),
            options.confidence_level,
            options.vcov,
            &irls.inactive,
        )?)
    } else {
        None
    };

    Ok(EngineFit {
        design,
        irls,
        dispersion,
        log_likelihood,
        aic,
        bic,
        inference,
    })
}
