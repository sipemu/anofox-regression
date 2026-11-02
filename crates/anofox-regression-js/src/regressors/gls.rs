//! Generalized least squares (`nlme::gls`) wrapper for WebAssembly.

use anofox_regression::solvers::gls::Interval;
use anofox_regression::solvers::{
    FittedGls as RustFittedGls, GlsCorrelation, GlsMethod, GlsRegressor as RustGlsRegressor,
    GlsResidualType, GlsVariance,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::{col_vec, fit_error, mat_rows, to_col, to_js, to_mat};

/// A point estimate with its confidence limits.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IntervalResult {
    pub lower: f64,
    pub estimate: f64,
    pub upper: f64,
}

impl From<&Interval> for IntervalResult {
    fn from(i: &Interval) -> Self {
        Self {
            lower: i.lower,
            estimate: i.estimate,
            upper: i.upper,
        }
    }
}

/// Approximate intervals of the variance–covariance parameters.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VarCovIntervalsResult {
    pub correlation: Vec<IntervalResult>,
    pub variance: Vec<IntervalResult>,
    pub sigma: IntervalResult,
}

/// Result from fitting a GLS model.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GlsResult {
    /// Coefficients, intercept first when fitted with one.
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_interval_lower: Vec<f64>,
    pub conf_interval_upper: Vec<f64>,
    pub vcov: Vec<Vec<f64>>,
    pub sigma: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_params: usize,
    pub df_residual: f64,
    pub correlation_params: Vec<f64>,
    pub variance_params: Vec<f64>,
    pub var_cov_intervals: Option<VarCovIntervalsResult>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
}

/// One distance class of an empirical semivariogram.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VariogramBinResult {
    pub variog: f64,
    pub dist: f64,
    pub n_pairs: usize,
}

/// Predicted means with standard errors.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GlsPrediction {
    pub fit: Vec<f64>,
    pub se: Vec<f64>,
}

fn parse_residual_type(kind: &str) -> Result<GlsResidualType, JsError> {
    match kind {
        "response" => Ok(GlsResidualType::Response),
        "pearson" => Ok(GlsResidualType::Pearson),
        "normalized" => Ok(GlsResidualType::Normalized),
        other => Err(JsError::new(&format!(
            "Unknown residual type '{}': expected response, pearson or normalized",
            other
        ))),
    }
}

fn to_usize(v: &[u32]) -> Vec<usize> {
    v.iter().map(|&g| g as usize).collect()
}

/// A fitted GLS model.
#[wasm_bindgen]
pub struct FittedGls {
    fitted: RustFittedGls,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedGls {
    /// Get the full result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let f = &self.fitted;
        let (lower, upper) = f.conf_int();
        to_js(&GlsResult {
            coefficients: col_vec(f.coefficients().as_ref()),
            std_errors: col_vec(f.std_errors().as_ref()),
            t_values: col_vec(f.t_values().as_ref()),
            p_values: col_vec(f.p_values().as_ref()),
            conf_interval_lower: col_vec(lower.as_ref()),
            conf_interval_upper: col_vec(upper.as_ref()),
            vcov: mat_rows(f.vcov()),
            sigma: f.sigma(),
            log_likelihood: f.log_likelihood(),
            aic: f.aic(),
            bic: f.bic(),
            n_params: f.n_params(),
            df_residual: f.df_residual(),
            correlation_params: col_vec(f.correlation_params()),
            variance_params: col_vec(f.variance_params()),
            var_cov_intervals: f.var_cov_intervals().map(|v| VarCovIntervalsResult {
                correlation: v.correlation.iter().map(IntervalResult::from).collect(),
                variance: v.variance.iter().map(IntervalResult::from).collect(),
                sigma: IntervalResult::from(&v.sigma),
            }),
            fitted_values: col_vec(f.fitted_values().as_ref()),
            residuals: col_vec(f.residuals(GlsResidualType::Response).as_ref()),
        })
    }

    /// Coefficients, intercept first when fitted with one.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        col_vec(self.fitted.coefficients().as_ref())
    }

    /// Correlation parameters on their natural scale.
    #[wasm_bindgen(js_name = getCorrelationParams)]
    pub fn get_correlation_params(&self) -> Vec<f64> {
        col_vec(self.fitted.correlation_params())
    }

    /// Variance-function parameters.
    #[wasm_bindgen(js_name = getVarianceParams)]
    pub fn get_variance_params(&self) -> Vec<f64> {
        col_vec(self.fitted.variance_params())
    }

    /// Residual standard error σ.
    #[wasm_bindgen(js_name = getSigma)]
    pub fn get_sigma(&self) -> f64 {
        self.fitted.sigma()
    }

    /// Maximised REML or ML log-likelihood.
    #[wasm_bindgen(js_name = getLogLikelihood)]
    pub fn get_log_likelihood(&self) -> f64 {
        self.fitted.log_likelihood()
    }

    /// Residuals on the "response", "pearson" or "normalized" scale.
    #[wasm_bindgen]
    pub fn residuals(&self, kind: &str) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted.residuals(parse_residual_type(kind)?).as_ref(),
        ))
    }

    /// Predicted means for new rows of `x` (without an intercept column).
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = to_mat(x, n_rows, self.n_features)?;
        Ok(col_vec(self.fitted.predict(&x_mat).as_ref()))
    }

    /// Predicted means and their standard errors: `{ fit, se }`.
    #[wasm_bindgen(js_name = predictWithSe)]
    pub fn predict_with_se(&self, x: &[f64], n_rows: usize) -> Result<JsValue, JsError> {
        let x_mat = to_mat(x, n_rows, self.n_features)?;
        let (fit, se) = self.fitted.predict_with_se(&x_mat);
        to_js(&GlsPrediction {
            fit: col_vec(fit.as_ref()),
            se: col_vec(se.as_ref()),
        })
    }

    /// Empirical semivariogram of the residuals of type `kind` against the
    /// within-group time distance, pooled into `nint` classes:
    /// `[{ variog, dist, nPairs }]`.
    #[wasm_bindgen]
    pub fn variogram(&self, time: &[f64], kind: &str, nint: usize) -> Result<JsValue, JsError> {
        let bins = self
            .fitted
            .variogram(time, parse_residual_type(kind)?, nint)
            .map_err(fit_error)?;
        let bins: Vec<VariogramBinResult> = bins
            .iter()
            .map(|b| VariogramBinResult {
                variog: b.variog,
                dist: b.dist,
                n_pairs: b.n_pairs,
            })
            .collect();
        to_js(&bins)
    }
}

/// Generalized least squares for grouped (longitudinal) data with a
/// within-group correlation structure and a variance function
/// (`nlme::gls`).
#[wasm_bindgen]
pub struct GlsRegressor {
    correlation: GlsCorrelation,
    variance: GlsVariance,
    method: GlsMethod,
    with_intercept: bool,
    confidence_level: f64,
    max_iterations: usize,
}

#[wasm_bindgen]
impl GlsRegressor {
    /// Create a GLS regressor: independent errors, constant variance, REML,
    /// intercept, 95% intervals.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            correlation: GlsCorrelation::Independent,
            variance: GlsVariance::Constant,
            method: GlsMethod::Reml,
            with_intercept: true,
            confidence_level: 0.95,
            max_iterations: 500,
        }
    }

    /// Set the within-group correlation structure: "independent",
    /// "compound_symmetry" (`corCompSymm`), "ar1" (`corAR1`), "car1"
    /// (`corCAR1`), "exponential", "gaussian", "linear", "spherical"
    /// (spatial, with an optional nugget) or "unstructured" (`corSymm`).
    #[wasm_bindgen(js_name = setCorrelation)]
    pub fn set_correlation(&mut self, name: &str, nugget: Option<bool>) -> Result<(), JsError> {
        let nugget = nugget.unwrap_or(false);
        self.correlation = match name {
            "independent" => GlsCorrelation::Independent,
            "compound_symmetry" => GlsCorrelation::CompoundSymmetry,
            "ar1" => GlsCorrelation::Ar1,
            "car1" => GlsCorrelation::ContinuousAr1,
            "exponential" => GlsCorrelation::Exponential { nugget },
            "gaussian" => GlsCorrelation::Gaussian { nugget },
            "linear" => GlsCorrelation::Linear { nugget },
            "spherical" => GlsCorrelation::Spherical { nugget },
            "unstructured" => GlsCorrelation::Unstructured,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown correlation '{}': expected independent, compound_symmetry, ar1, \
                     car1, exponential, gaussian, linear, spherical or unstructured",
                    other
                )))
            }
        };
        Ok(())
    }

    /// Use constant variance (the default).
    #[wasm_bindgen(js_name = setVarConstant)]
    pub fn set_var_constant(&mut self) {
        self.variance = GlsVariance::Constant;
    }

    /// A separate standard deviation per stratum (`varIdent`), one stratum
    /// id per observation.
    #[wasm_bindgen(js_name = setVarIdent)]
    pub fn set_var_ident(&mut self, strata: &[u32]) {
        self.variance = GlsVariance::Ident(to_usize(strata));
    }

    /// Standard deviation `σ |v|^δ` for a covariate `v` (`varPower`).
    #[wasm_bindgen(js_name = setVarPower)]
    pub fn set_var_power(&mut self, v: Vec<f64>) {
        self.variance = GlsVariance::Power(v);
    }

    /// Standard deviation `σ exp(δ v)` for a covariate `v` (`varExp`).
    #[wasm_bindgen(js_name = setVarExp)]
    pub fn set_var_exp(&mut self, v: Vec<f64>) {
        self.variance = GlsVariance::Exponential(v);
    }

    /// Set the estimation method: "reml" (default) or "ml".
    #[wasm_bindgen(js_name = setMethod)]
    pub fn set_method(&mut self, method: &str) -> Result<(), JsError> {
        self.method = match method {
            "reml" => GlsMethod::Reml,
            "ml" => GlsMethod::Ml,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown method '{}': expected reml or ml",
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

    /// Set the level of the coefficient and variance-parameter intervals.
    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    /// Set the maximum optimiser iterations (default 500).
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Fit the model. `group` holds the subject id of each row; `time`
    /// (optional) the time of each observation for the AR(1), CAR(1),
    /// spatial and unstructured structures (default: position within the
    /// group).
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        group: &[u32],
        time: Option<Vec<f64>>,
    ) -> Result<FittedGls, JsError> {
        let x_mat = to_mat(x, n_rows, n_cols)?;
        let y_col = to_col(y, n_rows, "target")?;
        if group.len() != n_rows {
            return Err(JsError::new(&format!(
                "Expected {} group values, got {}",
                n_rows,
                group.len()
            )));
        }
        let fitted = RustGlsRegressor::builder()
            .correlation(self.correlation)
            .variance(self.variance.clone())
            .method(self.method)
            .with_intercept(self.with_intercept)
            .confidence_level(self.confidence_level)
            .max_iterations(self.max_iterations)
            .build()
            .fit(&x_mat, &y_col, &to_usize(group), time.as_deref())
            .map_err(fit_error)?;
        Ok(FittedGls {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for GlsRegressor {
    fn default() -> Self {
        Self::new()
    }
}

iterative_fit!(FittedGls);
