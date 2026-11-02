//! Penalized GLM (IRLS with explicit priors) wrapper for WebAssembly.

use anofox_regression::core::{BinomialLink, HasModelInfo, PoissonLink};
use anofox_regression::solvers::penalized_glm::{
    FittedPenalizedGlm as RustFittedPenalizedGlm, PenalizedGlmFamily,
    PenalizedGlmRegressor as RustPenalizedGlmRegressor, PriorSpec, VcovType,
};
use faer::Mat;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::{
    col_vec, fit_error, mat_rows, parse_priors, parse_vcov_type, row_column, to_col, to_js, to_mat,
};

/// Wald inference of a penalized GLM fit.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PenalizedGlmInference {
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub confidence_level: f64,
    pub intercept_std_error: Option<f64>,
    pub intercept_z_value: Option<f64>,
    pub intercept_p_value: Option<f64>,
    pub intercept_ci_lower: Option<f64>,
    pub intercept_ci_upper: Option<f64>,
    /// Covariance in fitted order (intercept first), as nested rows.
    pub vcov: Option<Vec<Vec<f64>>>,
}

/// Separation check of binary data.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeparationResult {
    pub has_separation: bool,
    pub separated_predictors: Vec<usize>,
    pub warning_message: Option<String>,
}

/// Result from fitting a penalized GLM.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PenalizedGlmResult {
    /// R family name and link, e.g. "poisson" / "log".
    pub family: String,
    pub link: String,
    /// Feature coefficients (`NaN` for dropped columns).
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    pub pseudo_r_squared: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub dispersion: f64,
    pub theta: Option<f64>,
    pub accuracy: Option<f64>,
    pub n_observations: usize,
    pub iterations: usize,
    pub converged: bool,
    pub fitted_values: Vec<f64>,
    pub inference: Option<PenalizedGlmInference>,
    pub separation: SeparationResult,
}

/// A fitted penalized GLM.
#[wasm_bindgen]
pub struct FittedPenalizedGlm {
    fitted: RustFittedPenalizedGlm,
    n_features: usize,
}

impl FittedPenalizedGlm {
    fn mat(&self, x: &[f64], n_rows: usize) -> Result<Mat<f64>, JsError> {
        to_mat(x, n_rows, self.n_features)
    }
}

#[wasm_bindgen]
impl FittedPenalizedGlm {
    /// Get the full result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        use anofox_regression::solvers::IterativeFit;
        let f = &self.fitted;
        let info = f.model_info();
        let sep = f.separation();
        to_js(&PenalizedGlmResult {
            family: info.family.unwrap_or_default().to_string(),
            link: info.link.to_string(),
            coefficients: col_vec(f.coefficients().as_ref()),
            intercept: f.intercept(),
            deviance: f.deviance(),
            null_deviance: f.null_deviance(),
            pseudo_r_squared: f.pseudo_r_squared(),
            log_likelihood: f.log_likelihood(),
            aic: f.aic(),
            bic: f.bic(),
            dispersion: f.dispersion(),
            theta: f.theta(),
            accuracy: f.accuracy(),
            n_observations: f.n_observations(),
            iterations: f.iterations(),
            converged: f.converged(),
            fitted_values: col_vec(f.fitted_values()),
            inference: f.inference().map(|inf| PenalizedGlmInference {
                std_errors: inf.std_errors.clone(),
                z_values: inf.z_values.clone(),
                p_values: inf.p_values.clone(),
                ci_lower: inf.ci_lower.clone(),
                ci_upper: inf.ci_upper.clone(),
                confidence_level: inf.confidence_level,
                intercept_std_error: inf.intercept_std_error,
                intercept_z_value: inf.intercept_z_value,
                intercept_p_value: inf.intercept_p_value,
                intercept_ci_lower: inf.intercept_ci_lower,
                intercept_ci_upper: inf.intercept_ci_upper,
                vcov: inf.vcov.as_ref().map(mat_rows),
            }),
            separation: SeparationResult {
                has_separation: sep.has_separation,
                separated_predictors: sep.separated_predictors.clone(),
                warning_message: sep.warning_message.clone(),
            },
        })
    }

    /// Feature coefficients.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        col_vec(self.fitted.coefficients().as_ref())
    }

    /// The intercept.
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.intercept()
    }

    /// The residual deviance.
    #[wasm_bindgen(js_name = getDeviance)]
    pub fn get_deviance(&self) -> f64 {
        self.fitted.deviance()
    }

    /// Linear predictor `intercept + xβ`.
    #[wasm_bindgen(js_name = predictLinear)]
    pub fn predict_linear(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted.predict_linear(&self.mat(x, n_rows)?).as_ref(),
        ))
    }

    /// Per-observation diagnostics (`broom::augment`) of an unpenalized
    /// fit: fitted mean, residuals of `residType` ("deviance" by default),
    /// standardized and studentized residuals, leverage, Cook's distance,
    /// DFFITS. `x` and `y` are the data of the fit (without the offset).
    #[wasm_bindgen]
    pub fn augment(
        &self,
        x: &[f64],
        n_rows: usize,
        y: &[f64],
        resid_type: Option<String>,
    ) -> Result<JsValue, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let y_col = to_col(y, n_rows, "target")?;
        let kind =
            crate::diagnostics::parse_residual_type(resid_type.as_deref().unwrap_or("deviance"))?;
        let a = self
            .fitted
            .augment_with(&x_mat, &y_col, kind)
            .map_err(fit_error)?;
        Ok(crate::convert::ToJs::to_js(&a))
    }

    /// Component-plus-residual data per predictor on the working scale
    /// (`crPlots` for a GLM): `[{ term, x, component, partialResidual }]`.
    #[wasm_bindgen(js_name = partialResiduals)]
    pub fn partial_residuals(
        &self,
        x: &[f64],
        n_rows: usize,
        y: &[f64],
    ) -> Result<JsValue, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        to_col(y, n_rows, "target")?;
        let cols: Vec<Vec<f64>> = (0..self.n_features)
            .map(|j| (0..n_rows).map(|i| x_mat[(i, j)]).collect())
            .collect();
        let p =
            anofox_regression::diagnostics::partial_residuals_glm(self.fitted.result(), y, &cols)
                .map_err(fit_error)?;
        Ok(crate::convert::ToJs::to_js(&p))
    }

    /// Predicted means on the response scale.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(self.fitted.predict(&self.mat(x, n_rows)?).as_ref()))
    }

    /// Predicted means with a per-row offset on the linear predictor.
    #[wasm_bindgen(js_name = predictWithOffset)]
    pub fn predict_with_offset(
        &self,
        x: &[f64],
        n_rows: usize,
        offset: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        let offset = to_col(offset, n_rows, "offset")?;
        Ok(col_vec(
            self.fitted
                .predict_with_offset(&self.mat(x, n_rows)?, &offset)
                .as_ref(),
        ))
    }
}

/// GLM fitted by IRLS with an optional ridge `lambda` and per-coefficient
/// priors (posterior mode, as `arm::bayesglm` with fixed scales). Without
/// priors it matches `glm` / `MASS::glm.nb`.
#[wasm_bindgen]
pub struct PenalizedGlmRegressor {
    family: PenalizedGlmFamily,
    with_intercept: bool,
    max_iterations: u32,
    tolerance: f64,
    compute_inference: bool,
    confidence_level: f64,
    lambda: f64,
    priors: Vec<PriorSpec>,
    vcov: VcovType,
    offset: Option<Vec<f64>>,
    estimate_dispersion: bool,
}

#[wasm_bindgen]
impl PenalizedGlmRegressor {
    /// Create a Poisson (log link) penalized GLM without priors.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            family: PenalizedGlmFamily::poisson(),
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: Vec::new(),
            vcov: VcovType::default(),
            offset: None,
            estimate_dispersion: false,
        }
    }

    /// Set the family: "poisson", "binomial", "logistic",
    /// "negative_binomial", "gamma" or "tweedie". `link` applies to Poisson
    /// ("log", "identity", "sqrt") and binomial ("logit", "probit",
    /// "cloglog"); `param` is the Tweedie power, the fixed negative-binomial
    /// θ (estimated when omitted) or the logistic threshold.
    #[wasm_bindgen(js_name = setFamily)]
    pub fn set_family(
        &mut self,
        family: &str,
        link: Option<String>,
        param: Option<f64>,
    ) -> Result<(), JsError> {
        let bad_link = |l: &str| JsError::new(&format!("Unknown link '{}' for {}", l, family));
        self.family = match family {
            "poisson" => PenalizedGlmFamily::Poisson(match link.as_deref() {
                None | Some("log") => PoissonLink::Log,
                Some("identity") => PoissonLink::Identity,
                Some("sqrt") => PoissonLink::Sqrt,
                Some(l) => return Err(bad_link(l)),
            }),
            "binomial" => PenalizedGlmFamily::Binomial(match link.as_deref() {
                None | Some("logit") => BinomialLink::Logit,
                Some("probit") => BinomialLink::Probit,
                Some("cloglog") => BinomialLink::Cloglog,
                Some(l) => return Err(bad_link(l)),
            }),
            "logistic" => PenalizedGlmFamily::Logistic {
                threshold: param.unwrap_or(0.5),
            },
            "negative_binomial" => PenalizedGlmFamily::NegativeBinomial { theta: param },
            "gamma" => PenalizedGlmFamily::gamma(),
            "tweedie" => PenalizedGlmFamily::tweedie(
                param.ok_or_else(|| JsError::new("The tweedie family needs a variance power"))?,
            ),
            other => {
                return Err(JsError::new(&format!(
                    "Unknown family '{}': expected poisson, binomial, logistic, \
                     negative_binomial, gamma or tweedie",
                    other
                )))
            }
        };
        Ok(())
    }

    /// Set whether to include an intercept term.
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the maximum IRLS iterations (default 100).
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: u32) {
        self.max_iterations = max_iter;
    }

    /// Set the convergence tolerance (default 1e-8).
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Set whether to compute standard errors, tests and intervals.
    #[wasm_bindgen(js_name = setComputeInference)]
    pub fn set_compute_inference(&mut self, compute: bool) {
        self.compute_inference = compute;
    }

    /// Set the confidence level (default 0.95).
    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    /// Uniform ridge `lambda` on every non-intercept coefficient, added to
    /// the priors.
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// Set per-coefficient priors, an array of
    /// `{ kind: "flat" | "normal" | "laplace", loc, scale }` in feature
    /// order, optionally with a leading entry for the intercept.
    #[wasm_bindgen(js_name = setPriors)]
    pub fn set_priors(&mut self, priors: JsValue) -> Result<(), JsError> {
        self.priors = parse_priors(priors)?;
        Ok(())
    }

    /// Set the covariance at the mode: "laplace" (default), "sandwich" or
    /// "naive".
    #[wasm_bindgen(js_name = setVcovType)]
    pub fn set_vcov_type(&mut self, vcov: &str) -> Result<(), JsError> {
        self.vcov = parse_vcov_type(vcov)?;
        Ok(())
    }

    /// Set an offset on the linear predictor, one value per row.
    #[wasm_bindgen(js_name = setOffset)]
    pub fn set_offset(&mut self, offset: Vec<f64>) {
        self.offset = Some(offset);
    }

    /// Poisson only: quasi-Poisson standard errors (Pearson dispersion).
    #[wasm_bindgen(js_name = setEstimateDispersion)]
    pub fn set_estimate_dispersion(&mut self, estimate: bool) {
        self.estimate_dispersion = estimate;
    }

    /// Fit the model to `x` (no intercept column) and `y`. Rows with a
    /// non-finite value are dropped.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedPenalizedGlm, JsError> {
        let x_mat = to_mat(x, n_rows, n_cols)?;
        let y_col = to_col(y, n_rows, "target")?;
        let mut builder = RustPenalizedGlmRegressor::builder()
            .family(self.family)
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .lambda(self.lambda)
            .priors(self.priors.clone())
            .vcov_type(self.vcov)
            .estimate_dispersion(self.estimate_dispersion);
        if let Some(offset) = row_column(&self.offset, n_rows, "offset")? {
            builder = builder.offset(offset);
        }
        let fitted = builder.build().fit(&x_mat, &y_col).map_err(fit_error)?;
        Ok(FittedPenalizedGlm {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for PenalizedGlmRegressor {
    fn default() -> Self {
        Self::new()
    }
}

iterative_fit!(FittedPenalizedGlm);
predict_se_method!(FittedPenalizedGlm);

#[wasm_bindgen]
impl FittedPenalizedGlm {
    /// `predictSe` with an offset on the linear predictor (one per row).
    #[wasm_bindgen(js_name = predictSeWithOffset)]
    pub fn predict_se_with_offset(
        &self,
        x: &[f64],
        n_rows: usize,
        offset: &[f64],
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let x_mat = super::to_mat(x, n_rows, self.n_features)?;
        let off = super::to_col(offset, n_rows, "offset")?;
        let options = crate::inference::parse_predict_se_options(options)?;
        let p = self
            .fitted
            .predict_se_with_offset(&x_mat, &off, &options)
            .map_err(super::fit_error)?;
        Ok(crate::convert::ToJs::to_js(&p))
    }
}
