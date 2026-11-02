//! Ordinal (cumulative link) regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    xmean_ordinaly, FittedOrdinal as RustFittedOrdinal, OrdinalLink, OrdinalQuantileMethod,
    OrdinalRegressor as RustOrdinalRegressor,
};
use faer::{Col, Mat};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::{col_vec, fit_error, mat_flat, mat_rows, to_col, to_js, to_mat};

/// Result from fitting a cumulative link model.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrdinalResult {
    pub link: String,
    /// The distinct response values, in increasing order.
    pub levels: Vec<f64>,
    /// (Weighted) count of each level.
    pub counts: Vec<f64>,
    /// Intercepts `α_j` of `P(Y ≥ y_{j+1})`, one per cut-point.
    pub intercepts: Vec<f64>,
    pub coefficients: Vec<f64>,
    /// Standard errors of the coefficients.
    pub std_errors: Vec<f64>,
    /// Covariance of (intercepts, coefficients), as nested rows.
    pub vcov: Vec<Vec<f64>>,
    pub log_likelihood: f64,
    pub penalized_log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub lr_chisq: f64,
    pub lr_df: usize,
    pub lr_p_value: f64,
    pub aic: f64,
    pub effective_df: f64,
    pub n_observations: usize,
    pub iterations: usize,
    pub converged: bool,
}

/// A chi-square test.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChiSqTestResult {
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
}

/// Predictions with their standard errors.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictionWithSe {
    pub fit: Vec<f64>,
    pub se: Vec<f64>,
}

/// Quantile predictions with confidence limits.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantileCi {
    pub estimate: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// Observed and expected means of a predictor per response level.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct XmeanOrdinalyResult {
    pub levels: Vec<f64>,
    pub observed: Vec<f64>,
    pub expected_po: Vec<f64>,
    pub expected_cr: Option<Vec<f64>>,
}

pub(crate) fn parse_link(link: &str) -> Result<OrdinalLink, JsError> {
    match link {
        "logistic" => Ok(OrdinalLink::Logistic),
        "probit" => Ok(OrdinalLink::Probit),
        "loglog" => Ok(OrdinalLink::LogLog),
        "cloglog" => Ok(OrdinalLink::CLogLog),
        "cauchit" => Ok(OrdinalLink::Cauchit),
        other => Err(JsError::new(&format!(
            "Unknown link '{}': expected logistic, probit, loglog, cloglog or cauchit",
            other
        ))),
    }
}

fn link_name(link: OrdinalLink) -> &'static str {
    match link {
        OrdinalLink::Probit => "probit",
        OrdinalLink::LogLog => "loglog",
        OrdinalLink::CLogLog => "cloglog",
        OrdinalLink::Cauchit => "cauchit",
        _ => "logistic",
    }
}

fn parse_quantile_method(method: &str) -> Result<OrdinalQuantileMethod, JsError> {
    match method {
        "interpolated" => Ok(OrdinalQuantileMethod::Interpolated),
        "discrete" => Ok(OrdinalQuantileMethod::Discrete),
        other => Err(JsError::new(&format!(
            "Unknown quantile method '{}': expected interpolated or discrete",
            other
        ))),
    }
}

/// A fitted cumulative link model.
#[wasm_bindgen]
pub struct FittedOrdinal {
    fitted: RustFittedOrdinal,
    n_features: usize,
}

impl FittedOrdinal {
    /// Wrap a core fit (e.g. the best fit of `pentrace`).
    pub(crate) fn from_rust(fitted: RustFittedOrdinal, n_features: usize) -> Self {
        Self { fitted, n_features }
    }

    fn mat(&self, x: &[f64], n_rows: usize) -> Result<Mat<f64>, JsError> {
        to_mat(x, n_rows, self.n_features)
    }
}

#[wasm_bindgen]
impl FittedOrdinal {
    /// Get the full result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        use anofox_regression::solvers::IterativeFit;
        let f = &self.fitted;
        to_js(&OrdinalResult {
            link: link_name(f.link()).to_string(),
            levels: f.levels().to_vec(),
            counts: f.counts().to_vec(),
            intercepts: col_vec(f.intercepts().as_ref()),
            coefficients: col_vec(f.coefficients().as_ref()),
            std_errors: col_vec(f.std_errors().as_ref()),
            vcov: mat_rows(&f.vcov()),
            log_likelihood: f.log_likelihood(),
            penalized_log_likelihood: f.penalized_log_likelihood(),
            null_log_likelihood: f.null_log_likelihood(),
            lr_chisq: f.lr_chisq(),
            lr_df: f.lr_df(),
            lr_p_value: f.lr_p_value(),
            aic: f.aic(),
            effective_df: f.effective_df(),
            n_observations: f.n_observations(),
            iterations: f.iterations(),
            converged: f.converged(),
        })
    }

    /// Slopes β.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        col_vec(self.fitted.coefficients().as_ref())
    }

    /// Intercepts α_j, one per cut-point.
    #[wasm_bindgen(js_name = getIntercepts)]
    pub fn get_intercepts(&self) -> Vec<f64> {
        col_vec(self.fitted.intercepts().as_ref())
    }

    /// The distinct response values.
    #[wasm_bindgen(js_name = getLevels)]
    pub fn get_levels(&self) -> Vec<f64> {
        self.fitted.levels().to_vec()
    }

    /// The log-likelihood.
    #[wasm_bindgen(js_name = getLogLikelihood)]
    pub fn get_log_likelihood(&self) -> f64 {
        self.fitted.log_likelihood()
    }

    /// Covariance of (intercepts, coefficients), row-major.
    #[wasm_bindgen]
    pub fn vcov(&self) -> Vec<f64> {
        mat_flat(&self.fitted.vcov())
    }

    /// Covariance of the slopes only, row-major `n_cols × n_cols`.
    #[wasm_bindgen(js_name = vcovSlopes)]
    pub fn vcov_slopes(&self) -> Vec<f64> {
        mat_flat(&self.fitted.vcov_slopes())
    }

    /// Per-slope effective degrees of freedom of a penalized fit (1 each
    /// without a penalty).
    #[wasm_bindgen(js_name = effectiveDfDiagonal)]
    pub fn effective_df_diagonal(&self) -> Vec<f64> {
        col_vec(self.fitted.effective_df_diagonal().as_ref())
    }

    /// Sandwich covariance of a penalized fit (`var.penalty = "sandwich"`),
    /// row-major; equal to `vcov()` without a penalty.
    #[wasm_bindgen(js_name = vcovSandwich)]
    pub fn vcov_sandwich(&self) -> Vec<f64> {
        mat_flat(&self.fitted.vcov_sandwich())
    }

    /// Per-observation score contributions at the estimate, row-major
    /// `n_rows × (cut-points + features)`. `x` and `y` are the data of the
    /// fit.
    #[wasm_bindgen(js_name = scoreContributions)]
    pub fn score_contributions(
        &self,
        x: &[f64],
        n_rows: usize,
        y: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let y_col = to_col(y, n_rows, "target")?;
        let s = self
            .fitted
            .score_contributions(&x_mat, &y_col)
            .map_err(fit_error)?;
        Ok(mat_flat(&s))
    }

    /// Cluster-robust covariance of (intercepts, slopes) (`rms::robcov` with
    /// the "robcov" preset), row-major nested rows. `x` and `y` are the data
    /// of the fit; see `clusterSandwich` for `clusters` and `options`.
    #[wasm_bindgen(js_name = clusterVcov)]
    pub fn cluster_vcov(
        &self,
        x: &[f64],
        n_rows: usize,
        y: &[f64],
        clusters: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let y_col = to_col(y, n_rows, "target")?;
        let c = crate::inference::parse_clusters(clusters, n_rows)?;
        let v = self
            .fitted
            .cluster_vcov(
                &x_mat,
                &y_col,
                &crate::inference::cluster_refs(&c),
                &crate::inference::parse_cluster_options(options)?,
            )
            .map_err(fit_error)?;
        Ok(crate::convert::ToJs::to_js(&v))
    }

    /// `xᵀβ` per row (no intercept, no offset).
    #[wasm_bindgen(js_name = linearPredictor)]
    pub fn linear_predictor(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted.linear_predictor(&self.mat(x, n_rows)?).as_ref(),
        ))
    }

    /// `P(Y ≥ y_{j+1} | x)`, row-major `n_rows × cut-points`.
    #[wasm_bindgen(js_name = exceedanceProbabilities)]
    pub fn exceedance_probabilities(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(mat_flat(
            &self.fitted.exceedance_probabilities(&self.mat(x, n_rows)?),
        ))
    }

    /// `P(Y = y_j | x)`, row-major `n_rows × levels`.
    #[wasm_bindgen(js_name = categoryProbabilities)]
    pub fn category_probabilities(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(mat_flat(
            &self.fitted.category_probabilities(&self.mat(x, n_rows)?),
        ))
    }

    /// Predicted mean `Σ_j y_j P(Y = y_j | x)` (`rms::Mean.orm`).
    #[wasm_bindgen(js_name = predictMean)]
    pub fn predict_mean(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted.predict_mean(&self.mat(x, n_rows)?).as_ref(),
        ))
    }

    /// Predicted mean with its delta-method standard error: `{ fit, se }`.
    #[wasm_bindgen(js_name = predictMeanWithSe)]
    pub fn predict_mean_with_se(&self, x: &[f64], n_rows: usize) -> Result<JsValue, JsError> {
        let (fit, se) = self.fitted.predict_mean_with_se(&self.mat(x, n_rows)?);
        to_js(&PredictionWithSe {
            fit: col_vec(fit.as_ref()),
            se: col_vec(se.as_ref()),
        })
    }

    /// Predicted `q`-quantile of `Y` (`rms::Quantile.orm`); `method` is
    /// "interpolated" (default in rms) or "discrete".
    #[wasm_bindgen(js_name = predictQuantile)]
    pub fn predict_quantile(
        &self,
        x: &[f64],
        n_rows: usize,
        q: f64,
        method: &str,
    ) -> Result<Vec<f64>, JsError> {
        let method = parse_quantile_method(method)?;
        Ok(col_vec(
            self.fitted
                .predict_quantile(&self.mat(x, n_rows)?, q, method)
                .as_ref(),
        ))
    }

    /// Predicted `q`-quantile with confidence limits at `level`:
    /// `{ estimate, lower, upper }`.
    #[wasm_bindgen(js_name = predictQuantileCi)]
    pub fn predict_quantile_ci(
        &self,
        x: &[f64],
        n_rows: usize,
        q: f64,
        method: &str,
        level: f64,
    ) -> Result<JsValue, JsError> {
        let method = parse_quantile_method(method)?;
        let (estimate, lower, upper) =
            self.fitted
                .predict_quantile_ci(&self.mat(x, n_rows)?, q, method, level);
        to_js(&QuantileCi {
            estimate: col_vec(estimate.as_ref()),
            lower: col_vec(lower.as_ref()),
            upper: col_vec(upper.as_ref()),
        })
    }

    /// Score test of proportional odds for the predictor `columns` (all if
    /// empty), RMS §13.3: `{ statistic, df, pValue }`. `x` and `y` are the
    /// data of the fit.
    #[wasm_bindgen(js_name = proportionalOddsScoreTest)]
    pub fn proportional_odds_score_test(
        &self,
        x: &[f64],
        n_rows: usize,
        y: &[f64],
        columns: &[u32],
    ) -> Result<JsValue, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let y_col = to_col(y, n_rows, "target")?;
        let columns: Vec<usize> = columns.iter().map(|&c| c as usize).collect();
        let t = self
            .fitted
            .proportional_odds_score_test(&x_mat, &y_col, &columns)
            .map_err(fit_error)?;
        to_js(&ChiSqTestResult {
            statistic: t.statistic,
            df: t.df,
            p_value: t.p_value,
        })
    }
}

/// Cumulative link model for an ordered response (`rms::orm`, `MASS::polr`).
#[wasm_bindgen]
pub struct OrdinalRegressor {
    link: OrdinalLink,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    penalty: Option<f64>,
    penalty_matrix: Option<Vec<f64>>,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl OrdinalRegressor {
    /// Create a proportional-odds (logistic link) regressor.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            link: OrdinalLink::Logistic,
            weights: None,
            offset: None,
            penalty: None,
            penalty_matrix: None,
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }

    /// Set the link: "logistic" (default), "probit", "loglog", "cloglog" or
    /// "cauchit".
    #[wasm_bindgen(js_name = setLink)]
    pub fn set_link(&mut self, link: &str) -> Result<(), JsError> {
        self.link = parse_link(link)?;
        Ok(())
    }

    /// Set case (frequency) weights, one per row.
    #[wasm_bindgen(js_name = setWeights)]
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = Some(weights);
    }

    /// Set an offset on the linear predictor, one per row.
    #[wasm_bindgen(js_name = setOffset)]
    pub fn set_offset(&mut self, offset: Vec<f64>) {
        self.offset = Some(offset);
    }

    /// Quadratic penalty `λ · diag(var(x_j))` on the slopes (`orm`'s
    /// `penalty = λ`).
    #[wasm_bindgen(js_name = setPenalty)]
    pub fn set_penalty(&mut self, lambda: f64) {
        self.penalty = Some(lambda);
        self.penalty_matrix = None;
    }

    /// Quadratic penalty `½ βᵀPβ` with a row-major `n_cols × n_cols`
    /// matrix `P` (`penalty.matrix`).
    #[wasm_bindgen(js_name = setPenaltyMatrix)]
    pub fn set_penalty_matrix(&mut self, penalty: Vec<f64>) {
        self.penalty_matrix = Some(penalty);
        self.penalty = None;
    }

    /// Set the maximum Newton iterations (default 100).
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Set the convergence tolerance on the largest score (default 1e-10).
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Fit the model to predictors `x` (no intercept column) and the
    /// ordered response `y`.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedOrdinal, JsError> {
        let x_mat = to_mat(x, n_rows, n_cols)?;
        let y_col = to_col(y, n_rows, "target")?;
        let mut builder = RustOrdinalRegressor::builder()
            .link(self.link)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance);
        if let Some(w) = &self.weights {
            builder = builder.weights(to_col(w, n_rows, "weight")?);
        }
        if let Some(o) = &self.offset {
            builder = builder.offset(to_col(o, n_rows, "offset")?);
        }
        if let Some(lambda) = self.penalty {
            builder = builder.penalty(lambda);
        }
        if let Some(p) = &self.penalty_matrix {
            builder = builder.penalty_matrix(to_mat(p, n_cols, n_cols)?);
        }
        let fitted = builder.build().fit(&x_mat, &y_col).map_err(fit_error)?;
        Ok(FittedOrdinal {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for OrdinalRegressor {
    fn default() -> Self {
        Self::new()
    }
}

/// `rms::plot.xmean.ordinaly` numbers for one predictor: the observed mean
/// of `x` per level of `y` and its expectation under the proportional-odds
/// model and (with `cr`) the continuation-ratio model:
/// `{ levels, observed, expectedPo, expectedCr? }`.
#[wasm_bindgen(js_name = xmeanOrdinaly)]
pub fn xmean_ordinaly_js(x: &[f64], y: &[f64], cr: bool) -> Result<JsValue, JsError> {
    let n = x.len();
    let x_col = Col::from_fn(n, |i| x[i]);
    let y_col = to_col(y, n, "target")?;
    let r = xmean_ordinaly(&x_col, &y_col, cr).map_err(fit_error)?;
    to_js(&XmeanOrdinalyResult {
        levels: r.levels,
        observed: r.observed,
        expected_po: r.expected_po,
        expected_cr: r.expected_cr,
    })
}

iterative_fit!(FittedOrdinal);

#[wasm_bindgen]
impl FittedOrdinal {
    /// The penalty matrix of a penalized fit, row-major (undefined
    /// otherwise).
    #[wasm_bindgen(js_name = penaltyMatrix)]
    pub fn penalty_matrix(&self) -> Option<Vec<f64>> {
        self.fitted.penalty_matrix().map(super::mat_flat)
    }

    /// (Weighted) count of each response level.
    #[wasm_bindgen]
    pub fn counts(&self) -> Vec<f64> {
        self.fitted.counts().to_vec()
    }
}
