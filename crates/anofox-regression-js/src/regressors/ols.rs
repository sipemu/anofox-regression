//! OLS Regression wrapper for WebAssembly.

use anofox_regression::core::NaAction;
use anofox_regression::solvers::{FittedRegressor, OlsRegressor as RustOlsRegressor, Regressor};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use super::parse_na_action;

/// Result from fitting an OLS regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OlsResult {
    /// Estimated coefficients (excluding intercept).
    pub coefficients: Vec<f64>,
    /// Intercept term (null if model was fit without intercept).
    pub intercept: Option<f64>,
    /// R-squared (coefficient of determination).
    pub r_squared: f64,
    /// Adjusted R-squared.
    pub adj_r_squared: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean squared error.
    pub mse: f64,
    /// F-statistic for overall model significance.
    pub f_statistic: f64,
    /// P-value for F-statistic.
    pub f_pvalue: f64,
    /// Number of observations.
    pub n_observations: usize,
    /// Number of parameters (including intercept if present).
    pub n_parameters: usize,
    /// Residual degrees of freedom.
    pub residual_df: usize,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Residuals (y - fitted_values).
    pub residuals: Vec<f64>,
    /// Fitted values (predictions on training data).
    pub fitted_values: Vec<f64>,
    /// Standard errors of coefficients (if inference was computed).
    pub std_errors: Option<Vec<f64>>,
    /// T-statistics for coefficients (if inference was computed).
    pub t_statistics: Option<Vec<f64>>,
    /// P-values for coefficient significance tests (if inference was computed).
    pub p_values: Option<Vec<f64>>,
    /// Lower bounds of confidence intervals (if inference was computed).
    pub conf_interval_lower: Option<Vec<f64>>,
    /// Upper bounds of confidence intervals (if inference was computed).
    pub conf_interval_upper: Option<Vec<f64>>,
    /// Confidence level used for intervals.
    pub confidence_level: f64,
}

/// A fitted OLS regression model.
#[wasm_bindgen]
pub struct FittedOls {
    fitted: anofox_regression::solvers::FittedOls,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedOls {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();

        let ols_result = OlsResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            rmse: result.rmse,
            mse: result.mse,
            f_statistic: result.f_statistic,
            f_pvalue: result.f_pvalue,
            n_observations: result.n_observations,
            n_parameters: result.n_parameters,
            residual_df: result.residual_df(),
            aic: result.aic,
            bic: result.bic,
            residuals: result.residuals_expanded().iter().copied().collect(),
            fitted_values: result.fitted_expanded().iter().copied().collect(),
            std_errors: result
                .std_errors
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            t_statistics: result
                .t_statistics
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            p_values: result
                .p_values
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            conf_interval_lower: result
                .conf_interval_lower
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            conf_interval_upper: result
                .conf_interval_upper
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            confidence_level: result.confidence_level,
        };

        serde_wasm_bindgen::to_value(&ols_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the coefficients as a Float64Array.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    /// Get the intercept (returns null if model was fit without intercept).
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    /// Get R-squared.
    #[wasm_bindgen(js_name = getRSquared)]
    pub fn get_r_squared(&self) -> f64 {
        self.fitted.result().r_squared
    }

    /// Get adjusted R-squared.
    #[wasm_bindgen(js_name = getAdjRSquared)]
    pub fn get_adj_r_squared(&self) -> f64 {
        self.fitted.result().adj_r_squared
    }

    /// Make predictions on new data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix as a flat array (row-major order)
    /// * `n_rows` - Number of rows in the matrix
    ///
    /// # Returns
    /// Predictions as a Float64Array
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let n_cols = self.n_features;

        if x.len() != n_rows * n_cols {
            return Err(JsError::new(&format!(
                "Expected {} elements for {}x{} matrix, got {}",
                n_rows * n_cols,
                n_rows,
                n_cols,
                x.len()
            )));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let predictions = self.fitted.predict(&x_mat);

        Ok(predictions.iter().copied().collect())
    }
}

/// Ordinary Least Squares regression.
///
/// # Example (JavaScript)
/// ```javascript
/// const regressor = new OlsRegressor();
/// regressor.setWithIntercept(true);
/// regressor.setComputeInference(true);
///
/// // X is a flat array in row-major order: [x11, x12, x21, x22, ...]
/// const x = [1, 2, 3, 4, 5, 6]; // 3 rows, 2 columns
/// const y = [1, 2, 3];
///
/// const fitted = regressor.fit(x, 3, 2, y);
/// const result = fitted.getResult();
/// console.log(result.rSquared);
/// ```
#[wasm_bindgen]
pub struct OlsRegressor {
    na_action: NaAction,
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    rank_tolerance: f64,
}

#[wasm_bindgen]
impl OlsRegressor {
    /// Create a new OLS regressor with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            na_action: NaAction::default(),
            with_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            rank_tolerance: 1e-10,
        }
    }

    /// Set whether to include an intercept term.
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set whether to compute inference statistics (standard errors, p-values, etc.).
    #[wasm_bindgen(js_name = setComputeInference)]
    pub fn set_compute_inference(&mut self, compute: bool) {
        self.compute_inference = compute;
    }

    /// Set the confidence level for confidence intervals (default: 0.95).
    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    /// Set the rank tolerance for QR decomposition (default: 1e-10).
    #[wasm_bindgen(js_name = setRankTolerance)]
    pub fn set_rank_tolerance(&mut self, tol: f64) {
        self.rank_tolerance = tol;
    }

    /// Set the handling of rows with NaN in `x` or `y` (R's `na.action`):
    /// "omit" (default), "exclude" (residuals and fitted values padded with
    /// NaN at the dropped rows), "fail" or "pass".
    #[wasm_bindgen(js_name = setNaAction)]
    pub fn set_na_action(&mut self, action: &str) -> Result<(), JsError> {
        self.na_action = parse_na_action(action)?;
        Ok(())
    }

    /// Fit the OLS model to the data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix as a flat array (row-major order)
    /// * `n_rows` - Number of rows (observations)
    /// * `n_cols` - Number of columns (features)
    /// * `y` - Target values
    ///
    /// # Returns
    /// A fitted OLS model
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedOls, JsError> {
        // Validate input dimensions
        if x.len() != n_rows * n_cols {
            return Err(JsError::new(&format!(
                "Expected {} elements for {}x{} matrix, got {}",
                n_rows * n_cols,
                n_rows,
                n_cols,
                x.len()
            )));
        }

        if y.len() != n_rows {
            return Err(JsError::new(&format!(
                "Expected {} target values for {} observations, got {}",
                n_rows,
                n_rows,
                y.len()
            )));
        }

        // Convert to faer matrices
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        // Build and fit the model
        let model = self.rust();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedOls {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for OlsRegressor {
    fn default() -> Self {
        Self::new()
    }
}

augment_methods!(FittedOls);
partial_residuals_method!(FittedOls);
predict_se_method!(FittedOls);

#[wasm_bindgen]
impl FittedOls {
    /// Covariance of the coefficients `σ̂² M`, intercept first (nested
    /// rows); undefined without a variance factor.
    #[wasm_bindgen]
    pub fn vcov(&self) -> JsValue {
        crate::convert::ToJs::to_js(&self.fitted.vcov())
    }

    /// Variance factor `M` with `Var(β̂) = σ² M`, intercept first.
    #[wasm_bindgen(js_name = varianceFactor)]
    pub fn variance_factor(&self) -> JsValue {
        crate::convert::ToJs::to_js(&self.fitted.variance_factor())
    }

    /// Leverage `x₀ᵀ M x₀` of new rows.
    #[wasm_bindgen(js_name = leverageNew)]
    pub fn leverage_new(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = super::to_mat(x, n_rows, self.n_features)?;
        Ok(self.fitted.leverage_new(&x_mat).iter().copied().collect())
    }
}

#[wasm_bindgen]
impl FittedOls {
    /// Cluster-robust covariance (`sandwich::vcovCL`), intercept first:
    /// `clusters` is one or two arrays of cluster codes, `options` a preset
    /// ("cr1" default, "hc0", "robcov") or `{ type, cadjust }`. `x` is the
    /// design of the fit.
    #[wasm_bindgen(js_name = clusterVcov)]
    pub fn cluster_vcov(
        &self,
        x: &[f64],
        n_rows: usize,
        clusters: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let x_mat = super::to_mat(x, n_rows, self.n_features)?;
        let c = crate::inference::parse_clusters(clusters, n_rows)?;
        let v = self
            .fitted
            .cluster_vcov(
                &x_mat,
                &crate::inference::cluster_refs(&c),
                &crate::inference::parse_cluster_options(options)?,
            )
            .map_err(super::fit_error)?;
        Ok(crate::convert::ToJs::to_js(&v))
    }
}

#[wasm_bindgen]
impl FittedOls {
    /// Heteroskedasticity-consistent standard errors ("hc0" … "hc3"):
    /// `{ stdErrors, interceptStdError? }`.
    #[wasm_bindgen(js_name = hcStandardErrors)]
    pub fn hc_standard_errors(
        &self,
        x: &[f64],
        n_rows: usize,
        hc_type: &str,
    ) -> Result<JsValue, JsError> {
        let x_mat = super::to_mat(x, n_rows, self.n_features)?;
        let r = self
            .fitted
            .hc_standard_errors(&x_mat, crate::inference::parse_hc_type(hc_type)?)
            .map_err(super::fit_error)?;
        Ok(crate::convert::ToJs::to_js(&r))
    }

    /// HC standard errors with t tests and confidence intervals.
    #[wasm_bindgen(js_name = hcInference)]
    pub fn hc_inference(
        &self,
        x: &[f64],
        n_rows: usize,
        hc_type: &str,
    ) -> Result<JsValue, JsError> {
        let x_mat = super::to_mat(x, n_rows, self.n_features)?;
        let r = self
            .fitted
            .hc_inference(&x_mat, crate::inference::parse_hc_type(hc_type)?)
            .map_err(super::fit_error)?;
        Ok(crate::convert::ToJs::to_js(&r))
    }
}

impl OlsRegressor {
    fn rust(&self) -> RustOlsRegressor {
        RustOlsRegressor::builder()
            .na_action(self.na_action)
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .rank_tolerance(self.rank_tolerance)
            .build()
    }
}

#[wasm_bindgen]
impl OlsRegressor {
    /// Fit from a `SufficientStats` state: the same fit as on the rows it
    /// was built from (weighted states give WLS); residuals and fitted
    /// values are empty.
    #[wasm_bindgen(js_name = fitFromState)]
    pub fn fit_from_state(
        &self,
        state: &crate::streaming::SufficientStats,
    ) -> Result<FittedOls, JsError> {
        let fitted = self
            .rust()
            .fit_from_state(&state.inner)
            .map_err(super::fit_error)?;
        Ok(FittedOls {
            fitted,
            n_features: state.inner.n_features(),
        })
    }

    /// Fit from a `MomentAccumulator` (coefficients and their inference;
    /// row-level statistics are NaN).
    #[wasm_bindgen(js_name = fitFromAccumulator)]
    pub fn fit_from_accumulator(
        &self,
        acc: &crate::streaming::MomentAccumulator,
    ) -> Result<FittedOls, JsError> {
        let fitted = self
            .rust()
            .fit_from_accumulator(&acc.inner)
            .map_err(super::fit_error)?;
        Ok(FittedOls {
            fitted,
            n_features: acc.inner.n_features(),
        })
    }

    /// Fit from raw moments: `xtx` (row-major p × p), `xty`, `sumX`, `sumY`
    /// and `n`.
    #[wasm_bindgen(js_name = fitFromMoments)]
    pub fn fit_from_moments(
        &self,
        xtx: &[f64],
        xty: &[f64],
        sum_x: &[f64],
        sum_y: f64,
        n: usize,
    ) -> Result<FittedOls, JsError> {
        let p = xty.len();
        let fitted = self
            .rust()
            .fit_from_moments(
                &crate::inference::square(xtx, p, "xtx")?,
                &super::to_col(xty, p, "xty")?,
                &super::to_col(sum_x, p, "sumX")?,
                sum_y,
                n,
            )
            .map_err(super::fit_error)?;
        Ok(FittedOls {
            fitted,
            n_features: p,
        })
    }
}
