//! Generalized linear mixed model (`lme4::lmer` / `glmer`) wrapper for
//! WebAssembly.

use anofox_regression::solvers::{
    FittedGlmm as RustFittedGlmm, GlmmDdf, GlmmRegressor as RustGlmmRegressor, GlmmRegressorBuilder,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::{col_vec, fit_error, row_column, to_col, to_js, to_mat};

/// Per-factor summary of a crossed / nested fit.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FactorSummaryResult {
    pub n_levels: usize,
    pub sd: f64,
    pub blups: Vec<f64>,
}

/// Result from fitting a GLMM.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GlmmResult {
    /// Fixed effects, intercept first when present.
    pub fixed_effects: Vec<f64>,
    pub intercept: Option<f64>,
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    /// Covariance of the fixed effects, as nested rows.
    pub vcov: Vec<Vec<f64>>,
    pub satterthwaite_df: Option<Vec<f64>>,
    /// BLUPs, one row of `q` random effects per group.
    pub random_effects: Vec<Vec<f64>>,
    /// Conditional standard errors of the BLUPs, same layout.
    pub random_effects_se: Vec<Vec<f64>>,
    /// Random-effects covariance Σ (`q × q`).
    pub random_cov: Vec<Vec<f64>>,
    pub random_sd: Vec<f64>,
    pub random_corr: Vec<Vec<f64>>,
    pub theta: f64,
    pub sigma: f64,
    pub sd_random: f64,
    pub var_random: f64,
    pub deviance: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_parameters: usize,
    pub n_observations: usize,
    pub n_groups: usize,
    pub n_factors: usize,
    pub factors: Vec<FactorSummaryResult>,
    pub df_residual: f64,
    pub nb_theta: Option<f64>,
    pub is_reml: bool,
    pub fitted_values: Vec<f64>,
    pub fitted_population: Vec<f64>,
    pub residuals: Vec<f64>,
    pub pearson_residuals: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Fixed-effect inference, intercept first.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GlmmInferenceResult {
    pub std_errors: Vec<f64>,
    pub statistics: Vec<f64>,
    pub df: Option<Vec<f64>>,
    pub p_values: Vec<f64>,
    pub conf_low: Vec<f64>,
    pub conf_high: Vec<f64>,
}

/// Lower and upper interval bounds.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds<T> {
    pub lower: T,
    pub upper: T,
}

/// A fitted GLMM.
#[wasm_bindgen]
pub struct FittedGlmm {
    fitted: RustFittedGlmm,
    n_features: usize,
}

fn bad_factor(f: usize) -> JsError {
    JsError::new(&format!(
        "No grouping factor {} (only crossed / nested fits have per-factor results)",
        f
    ))
}

#[wasm_bindgen]
impl FittedGlmm {
    /// Get the full result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        use anofox_regression::solvers::IterativeFit;
        let f = &self.fitted;
        to_js(&GlmmResult {
            fixed_effects: col_vec(f.fixed_effects()),
            intercept: f.intercept(),
            std_errors: col_vec(f.std_errors()),
            z_values: col_vec(f.z_values().as_ref()),
            p_values: col_vec(f.p_values().as_ref()),
            vcov: f.vcov().to_vec(),
            satterthwaite_df: f.satterthwaite_df().map(<[f64]>::to_vec),
            random_effects: f.random_effects_matrix().to_vec(),
            random_effects_se: f.random_effects_se_matrix().to_vec(),
            random_cov: f.random_cov().to_vec(),
            random_sd: col_vec(f.random_sd().as_ref()),
            random_corr: f.random_corr(),
            theta: f.theta(),
            sigma: f.sigma(),
            sd_random: f.sd_random(),
            var_random: f.var_random(),
            deviance: f.deviance(),
            log_likelihood: f.log_likelihood(),
            aic: f.aic(),
            bic: f.bic(),
            n_parameters: f.n_parameters(),
            n_observations: f.n_observations(),
            n_groups: f.n_groups(),
            n_factors: f.n_factors(),
            factors: f
                .factors()
                .iter()
                .map(|s| FactorSummaryResult {
                    n_levels: s.n_levels,
                    sd: s.sd,
                    blups: s.blups.clone(),
                })
                .collect(),
            df_residual: f.df_residual(),
            nb_theta: f.nb_theta(),
            is_reml: f.is_reml(),
            fitted_values: col_vec(f.fitted_values()),
            fitted_population: col_vec(f.fitted_population()),
            residuals: col_vec(f.residuals().as_ref()),
            pearson_residuals: col_vec(f.pearson_residuals()),
            iterations: f.iterations(),
            converged: f.converged(),
        })
    }

    /// Fixed effects, intercept first when present.
    #[wasm_bindgen(js_name = getFixedEffects)]
    pub fn get_fixed_effects(&self) -> Vec<f64> {
        col_vec(self.fitted.fixed_effects())
    }

    /// Random-intercept BLUP of each group.
    #[wasm_bindgen(js_name = getRandomEffects)]
    pub fn get_random_effects(&self) -> Vec<f64> {
        col_vec(self.fitted.random_effects())
    }

    /// The log-likelihood (REML criterion for a REML fit).
    #[wasm_bindgen(js_name = getLogLikelihood)]
    pub fn get_log_likelihood(&self) -> f64 {
        self.fitted.log_likelihood()
    }

    /// Wald confidence intervals of the fixed effects: `{ lower, upper }`.
    #[wasm_bindgen(js_name = confInt)]
    pub fn conf_int(&self, level: f64) -> Result<JsValue, JsError> {
        let (lower, upper) = self.fitted.conf_int(level);
        to_js(&Bounds { lower, upper })
    }

    /// Fixed-effect tests with `ddf` "wald" (z) or "satterthwaite" (t; a
    /// Gaussian fit with `setSatterthwaite(true)`):
    /// `{ stdErrors, statistics, df?, pValues, confLow, confHigh }`.
    #[wasm_bindgen]
    pub fn inference(&self, ddf: &str, level: f64) -> Result<JsValue, JsError> {
        let ddf = match ddf {
            "wald" => GlmmDdf::Wald,
            "satterthwaite" => GlmmDdf::Satterthwaite,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown ddf '{}': expected wald or satterthwaite",
                    other
                )))
            }
        };
        let inf = self.fitted.inference(ddf, level).map_err(fit_error)?;
        to_js(&GlmmInferenceResult {
            std_errors: inf.std_errors,
            statistics: inf.statistics,
            df: inf.df,
            p_values: inf.p_values,
            conf_low: inf.conf_low,
            conf_high: inf.conf_high,
        })
    }

    /// Normal intervals of the BLUPs (`n_groups × q` rows):
    /// `{ lower, upper }`.
    #[wasm_bindgen(js_name = randomEffectsConfInt)]
    pub fn random_effects_conf_int(&self, level: f64) -> Result<JsValue, JsError> {
        let (lower, upper) = self.fitted.random_effects_conf_int(level);
        to_js(&Bounds { lower, upper })
    }

    /// Random-effects covariance of grouping factor `f` (crossed / nested
    /// fits), as nested rows.
    #[wasm_bindgen(js_name = factorRandomCov)]
    pub fn factor_random_cov(&self, f: usize) -> Result<JsValue, JsError> {
        let cov = self
            .fitted
            .factor_random_cov(f)
            .ok_or_else(|| bad_factor(f))?;
        to_js(&cov)
    }

    /// Random-effects standard deviations of grouping factor `f`.
    #[wasm_bindgen(js_name = factorRandomSd)]
    pub fn factor_random_sd(&self, f: usize) -> Result<Vec<f64>, JsError> {
        self.fitted.factor_random_sd(f).ok_or_else(|| bad_factor(f))
    }

    /// BLUPs of grouping factor `f`, one row per level.
    #[wasm_bindgen(js_name = factorRandomEffects)]
    pub fn factor_random_effects(&self, f: usize) -> Result<JsValue, JsError> {
        let re = self
            .fitted
            .factor_random_effects_matrix(f)
            .ok_or_else(|| bad_factor(f))?;
        to_js(&re)
    }

    /// Conditional standard errors of the BLUPs of grouping factor `f`.
    #[wasm_bindgen(js_name = factorRandomEffectsSe)]
    pub fn factor_random_effects_se(&self, f: usize) -> Result<JsValue, JsError> {
        let se = self
            .fitted
            .factor_random_effects_se_matrix(f)
            .ok_or_else(|| bad_factor(f))?;
        to_js(&se)
    }

    /// Normal intervals of the BLUPs of grouping factor `f`.
    #[wasm_bindgen(js_name = factorRandomEffectsConfInt)]
    pub fn factor_random_effects_conf_int(&self, f: usize, level: f64) -> Result<JsValue, JsError> {
        let (lower, upper) = self
            .fitted
            .factor_random_effects_conf_int(f, level)
            .ok_or_else(|| bad_factor(f))?;
        to_js(&Bounds { lower, upper })
    }

    /// Population-level linear predictor `xβ` (random effects at zero).
    #[wasm_bindgen(js_name = predictFixed)]
    pub fn predict_fixed(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = to_mat(x, n_rows, self.n_features)?;
        Ok(col_vec(self.fitted.predict_fixed(&x_mat).as_ref()))
    }

    /// Population-level linear predictor `xβ + offset`.
    #[wasm_bindgen(js_name = predictFixedWithOffset)]
    pub fn predict_fixed_with_offset(
        &self,
        x: &[f64],
        n_rows: usize,
        offset: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        let x_mat = to_mat(x, n_rows, self.n_features)?;
        let offset = to_col(offset, n_rows, "offset")?;
        Ok(col_vec(
            self.fitted
                .predict_fixed_with_offset(&x_mat, &offset)
                .as_ref(),
        ))
    }
}

/// Generalized linear mixed model with random intercepts (and optional
/// random slopes) over one or several crossed / nested grouping factors.
#[wasm_bindgen]
pub struct GlmmRegressor {
    family: String,
    tweedie_power: f64,
    with_intercept: bool,
    random_intercept: bool,
    random_slopes: Vec<usize>,
    random_slopes_per_factor: Option<Vec<Vec<usize>>>,
    nb_theta: Option<f64>,
    offset: Option<Vec<f64>>,
    reml: bool,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    theta_max: Option<f64>,
    satterthwaite: bool,
    n_agq: Option<usize>,
}

#[wasm_bindgen]
impl GlmmRegressor {
    /// Create a GLMM for `family`: "gaussian" (default; a linear mixed
    /// model), "poisson", "binomial", "negative_binomial", "gamma" or
    /// "tweedie" (with `power` in (1, 2)).
    #[wasm_bindgen(constructor)]
    pub fn new(family: Option<String>, power: Option<f64>) -> Result<GlmmRegressor, JsError> {
        let family = family.unwrap_or_else(|| "gaussian".to_string());
        match family.as_str() {
            "gaussian" | "poisson" | "binomial" | "negative_binomial" | "gamma" => {}
            "tweedie" if power.is_some() => {}
            "tweedie" => return Err(JsError::new("The tweedie family needs a power")),
            other => {
                return Err(JsError::new(&format!(
                    "Unknown family '{}': expected gaussian, poisson, binomial, \
                     negative_binomial, gamma or tweedie",
                    other
                )))
            }
        }
        Ok(Self {
            family,
            tweedie_power: power.unwrap_or(1.5),
            with_intercept: true,
            random_intercept: true,
            random_slopes: Vec::new(),
            random_slopes_per_factor: None,
            nb_theta: None,
            offset: None,
            reml: true,
            max_iterations: None,
            tolerance: None,
            theta_max: None,
            satterthwaite: false,
            n_agq: None,
        })
    }

    /// Set whether to include a fixed intercept.
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set whether the random-effects design has a random intercept.
    #[wasm_bindgen(js_name = setRandomIntercept)]
    pub fn set_random_intercept(&mut self, include: bool) {
        self.random_intercept = include;
    }

    /// Columns of `x` that carry a random slope (single grouping factor).
    #[wasm_bindgen(js_name = setRandomSlopes)]
    pub fn set_random_slopes(&mut self, columns: &[u32]) {
        self.random_slopes = columns.iter().map(|&c| c as usize).collect();
    }

    /// Random-slope columns per grouping factor for `fitCrossed`, e.g.
    /// `[[0], []]`.
    #[wasm_bindgen(js_name = setRandomSlopesPerFactor)]
    pub fn set_random_slopes_per_factor(&mut self, columns: JsValue) -> Result<(), JsError> {
        let columns: Vec<Vec<usize>> =
            serde_wasm_bindgen::from_value(columns).map_err(|e| JsError::new(&e.to_string()))?;
        self.random_slopes_per_factor = Some(columns);
        Ok(())
    }

    /// Fix the negative-binomial size θ (estimated by default).
    #[wasm_bindgen(js_name = setNbTheta)]
    pub fn set_nb_theta(&mut self, theta: f64) {
        self.nb_theta = Some(theta);
    }

    /// Set an offset on the linear predictor, one value per row.
    #[wasm_bindgen(js_name = setOffset)]
    pub fn set_offset(&mut self, offset: Vec<f64>) {
        self.offset = Some(offset);
    }

    /// REML (default) or ML for the Gaussian family.
    #[wasm_bindgen(js_name = setReml)]
    pub fn set_reml(&mut self, reml: bool) {
        self.reml = reml;
    }

    /// Set the maximum PIRLS iterations.
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = Some(max_iter);
    }

    /// Set the convergence tolerance.
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = Some(tol);
    }

    /// Set the upper bound of the profiled ratio θ = σ_b / σ.
    #[wasm_bindgen(js_name = setThetaMax)]
    pub fn set_theta_max(&mut self, theta_max: f64) {
        self.theta_max = Some(theta_max);
    }

    /// Compute Satterthwaite degrees of freedom (Gaussian family).
    #[wasm_bindgen(js_name = setSatterthwaite)]
    pub fn set_satterthwaite(&mut self, compute: bool) {
        self.satterthwaite = compute;
    }

    /// lme4's `nAGQ`: 0 = PIRLS fixed effects, 1 = Laplace, > 1 = adaptive
    /// Gauss–Hermite (single grouping factor).
    #[wasm_bindgen(js_name = setNAgq)]
    pub fn set_n_agq(&mut self, n_agq: usize) {
        self.n_agq = Some(n_agq);
    }

    fn builder(&self, n_rows: usize) -> Result<GlmmRegressorBuilder, JsError> {
        let mut b = match self.family.as_str() {
            "poisson" => RustGlmmRegressor::poisson(),
            "binomial" => RustGlmmRegressor::binomial(),
            "negative_binomial" => RustGlmmRegressor::negative_binomial(),
            "gamma" => RustGlmmRegressor::gamma(),
            "tweedie" => RustGlmmRegressor::tweedie(self.tweedie_power),
            _ => RustGlmmRegressor::gaussian(),
        }
        .with_intercept(self.with_intercept)
        .random_intercept(self.random_intercept)
        .random_slopes(self.random_slopes.clone())
        .reml(self.reml)
        .satterthwaite(self.satterthwaite);
        if let Some(per) = &self.random_slopes_per_factor {
            b = b.random_slopes_per_factor(per.clone());
        }
        if let Some(theta) = self.nb_theta {
            b = b.nb_theta(theta);
        }
        if let Some(offset) = row_column(&self.offset, n_rows, "offset")? {
            b = b.offset(offset);
        }
        if let Some(n) = self.max_iterations {
            b = b.max_iterations(n);
        }
        if let Some(tol) = self.tolerance {
            b = b.tolerance(tol);
        }
        if let Some(t) = self.theta_max {
            b = b.theta_max(t);
        }
        if let Some(n) = self.n_agq {
            b = b.n_agq(n);
        }
        Ok(b)
    }

    /// Fit with one grouping factor: `group` holds the group id of each row.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        group: &[u32],
    ) -> Result<FittedGlmm, JsError> {
        let x_mat = to_mat(x, n_rows, n_cols)?;
        let y_col = to_col(y, n_rows, "target")?;
        if group.len() != n_rows {
            return Err(JsError::new(&format!(
                "Expected {} group values, got {}",
                n_rows,
                group.len()
            )));
        }
        let group: Vec<usize> = group.iter().map(|&g| g as usize).collect();
        let fitted = self
            .builder(n_rows)?
            .build()
            .fit(&x_mat, &y_col, &group)
            .map_err(fit_error)?;
        Ok(FittedGlmm {
            fitted,
            n_features: n_cols,
        })
    }

    /// Fit with several crossed or nested grouping factors, an array of
    /// group-id arrays (one per factor, one id per row). For `(1|a/b)` pass
    /// `a` and a unique id per `(a, b)` pair.
    #[wasm_bindgen(js_name = fitCrossed)]
    pub fn fit_crossed(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        groups: JsValue,
    ) -> Result<FittedGlmm, JsError> {
        let x_mat = to_mat(x, n_rows, n_cols)?;
        let y_col = to_col(y, n_rows, "target")?;
        let groups: Vec<Vec<usize>> =
            serde_wasm_bindgen::from_value(groups).map_err(|e| JsError::new(&e.to_string()))?;
        if let Some(g) = groups.iter().find(|g| g.len() != n_rows) {
            return Err(JsError::new(&format!(
                "Expected {} group values per factor, got {}",
                n_rows,
                g.len()
            )));
        }
        let refs: Vec<&[usize]> = groups.iter().map(Vec::as_slice).collect();
        let fitted = self
            .builder(n_rows)?
            .build()
            .fit_crossed(&x_mat, &y_col, &refs)
            .map_err(fit_error)?;
        Ok(FittedGlmm {
            fitted,
            n_features: n_cols,
        })
    }
}

iterative_fit!(FittedGlmm);
