//! Regression model wrappers for WebAssembly.

/// Expose the Rust `IterativeFit` trait (`converged`, `iterations`) on a
/// fitted wrapper whose inner model is in the `fitted` field. Invoked in each
/// wrapper's module, since that field is private.
macro_rules! iterative_fit {
    ($($wrapper:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $wrapper {
            /// Whether the fitting algorithm met its stopping rule.
            #[wasm_bindgen(js_name = getConverged)]
            pub fn get_converged(&self) -> bool {
                anofox_regression::solvers::IterativeFit::converged(&self.fitted)
            }

            /// Iterations (or passes, epochs) the fitting algorithm used.
            #[wasm_bindgen(js_name = getIterations)]
            pub fn get_iterations(&self) -> usize {
                anofox_regression::solvers::IterativeFit::iterations(&self.fitted)
            }
        }
    )*};
}

/// `augment`, `press` and `partialResiduals` on a fitted wrapper whose
/// inner model implements the Rust `Augmentable` trait.
macro_rules! augment_methods {
    ($($wrapper:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $wrapper {
            /// Per-observation diagnostics (`broom::augment`): fitted values,
            /// residuals of `residType` ("response", "pearson" or "deviance";
            /// the model's default when omitted), standardized and
            /// studentized residuals, leverage, Cook's distance, DFFITS and
            /// leave-one-out residuals. `x` and `y` are the data of the fit.
            #[wasm_bindgen]
            pub fn augment(
                &self,
                x: &[f64],
                n_rows: usize,
                y: &[f64],
                resid_type: Option<String>,
                weights: Option<Vec<f64>>,
            ) -> Result<JsValue, JsError> {
                let x_mat = $crate::regressors::to_mat(x, n_rows, self.n_features)?;
                let y_col = $crate::regressors::to_col(y, n_rows, "target")?;
                let w = $crate::regressors::row_column(&weights, n_rows, "weight")?;
                let a = match resid_type {
                    Some(kind) => anofox_regression::diagnostics::augment_with(
                        &self.fitted,
                        &x_mat,
                        &y_col,
                        w.as_ref(),
                        $crate::diagnostics::parse_residual_type(&kind)?,
                    ),
                    None => anofox_regression::diagnostics::augment(
                        &self.fitted,
                        &x_mat,
                        &y_col,
                        w.as_ref(),
                    ),
                }
                .map_err($crate::regressors::fit_error)?;
                Ok($crate::convert::ToJs::to_js(&a))
            }

            /// PRESS statistic, its RMSE and the leave-one-out R².
            #[wasm_bindgen]
            pub fn press(
                &self,
                x: &[f64],
                n_rows: usize,
                y: &[f64],
                weights: Option<Vec<f64>>,
            ) -> Result<JsValue, JsError> {
                let x_mat = $crate::regressors::to_mat(x, n_rows, self.n_features)?;
                let y_col = $crate::regressors::to_col(y, n_rows, "target")?;
                let w = $crate::regressors::row_column(&weights, n_rows, "weight")?;
                let p = anofox_regression::diagnostics::press(&self.fitted, &x_mat, &y_col, w.as_ref())
                    .map_err($crate::regressors::fit_error)?;
                Ok($crate::convert::ToJs::to_js(&p))
            }
        }
    )*};
}

/// `partialResiduals` on a fitted wrapper whose inner model implements the
/// Rust `FittedRegressor` trait.
macro_rules! partial_residuals_method {
    ($($wrapper:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $wrapper {
            /// Component-plus-residual data for every predictor (`crPlots`):
            /// `[{ term, x, component, partialResidual }]`. `x` and `y` are
            /// the data of the fit.
            #[wasm_bindgen(js_name = partialResiduals)]
            pub fn partial_residuals(
                &self,
                x: &[f64],
                n_rows: usize,
                y: &[f64],
            ) -> Result<JsValue, JsError> {
                let x_mat = $crate::regressors::to_mat(x, n_rows, self.n_features)?;
                let y_col = $crate::regressors::to_col(y, n_rows, "target")?;
                let p = anofox_regression::diagnostics::partial_residuals(&self.fitted, &x_mat, &y_col)
                    .map_err($crate::regressors::fit_error)?;
                Ok($crate::convert::ToJs::to_js(&p))
            }
        }
    )*};
}

/// `predictSe` on a fitted wrapper whose inner model implements the Rust
/// `PredictWithSe` trait.
macro_rules! predict_se_method {
    ($($wrapper:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $wrapper {
            /// Predictions with standard errors and intervals (`predict(...,
            /// se.fit = TRUE)`): `options` = `{ scale?: "response" | "link",
            /// interval?: "confidence" | "prediction", level? }`. Returns
            /// `{ fit, seFit, lower, upper, scale, interval, level, df,
            /// residualScale? }`.
            #[wasm_bindgen(js_name = predictSe)]
            pub fn predict_se(
                &self,
                x: &[f64],
                n_rows: usize,
                options: JsValue,
            ) -> Result<JsValue, JsError> {
                let x_mat = $crate::regressors::to_mat(x, n_rows, self.n_features)?;
                let options = $crate::inference::parse_predict_se_options(options)?;
                let p = anofox_regression::inference::PredictWithSe::predict_se(&self.fitted, &x_mat, &options)
                    .map_err($crate::regressors::fit_error)?;
                Ok($crate::convert::ToJs::to_js(&p))
            }
        }
    )*};
}

/// `likelihood` and `modelLrTest` on a fitted wrapper whose inner model
/// implements the Rust `LikelihoodFit` trait.
macro_rules! likelihood_methods {
    ($($wrapper:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $wrapper {
            /// What `lrTest` needs: `{ minusTwoLoglik, nullMinusTwoLoglik?,
            /// lrDispersion, nParameters }`.
            #[wasm_bindgen]
            pub fn likelihood(&self) -> JsValue {
                $crate::inference::likelihood_summary(&self.fitted)
            }

            /// Likelihood-ratio test against the null model:
            /// `{ statistic, df, pValue }` (undefined when the null
            /// likelihood is unknown).
            #[wasm_bindgen(js_name = modelLrTest)]
            pub fn model_lr_test(&self) -> JsValue {
                $crate::inference::model_lr(&self.fitted)
            }
        }
    )*};
}

/// Methods returning a column of the fit (residuals, weights, …) as an
/// array: `{ "jsName" => rust_method, … }`.
macro_rules! col_methods {
    ($w:ident { $($js:literal => $m:ident),* $(,)? }) => {
        #[wasm_bindgen]
        impl $w {
            $(
                #[doc = concat!("`", stringify!($m), "` of the fit.")]
                #[wasm_bindgen(js_name = $js)]
                pub fn $m(&self) -> Vec<f64> {
                    self.fitted.$m().iter().copied().collect()
                }
            )*
        }
    };
}

/// Methods predicting a column for new rows (`x`, `nRows`):
/// `{ "jsName" => rust_method, … }`.
macro_rules! predict_col_methods {
    ($w:ident { $($js:literal => $m:ident),* $(,)? }) => {
        #[wasm_bindgen]
        impl $w {
            $(
                #[doc = concat!("`", stringify!($m), "` for new rows (row-major `x`).")]
                #[wasm_bindgen(js_name = $js)]
                pub fn $m(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
                    let x = $crate::regressors::to_mat(x, n_rows, self.n_features)?;
                    Ok(self.fitted.$m(&x).iter().copied().collect())
                }
            )*
        }
    };
}

/// Methods returning a number: `{ "jsName" => rust_method, … }`.
macro_rules! scalar_methods {
    ($w:ident { $($js:literal => $m:ident),* $(,)? }) => {
        #[wasm_bindgen]
        impl $w {
            $(
                #[doc = concat!("`", stringify!($m), "` of the fit.")]
                #[wasm_bindgen(js_name = $js)]
                pub fn $m(&self) -> f64 {
                    self.fitted.$m() as f64
                }
            )*
        }
    };
}

/// `vcov()`: the coefficient covariance, row-major (undefined when the fit
/// has none).
macro_rules! vcov_method {
    ($($w:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $w {
            /// Covariance of the coefficients, row-major (undefined when
            /// not available).
            #[wasm_bindgen]
            pub fn vcov(&self) -> Option<Vec<f64>> {
                $crate::regressors::FlatMat::flat(self.fitted.vcov())
            }
        }
    )*};
}

/// `predictWithSe(x, nRows, type?, interval?, level?)` on an IRLS GLM:
/// `type` "response" (default) or "link", `interval` "confidence" or
/// "prediction" (default none), `level` 0.95.
macro_rules! predict_with_se_method {
    ($($w:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $w {
            /// Predictions with standard errors and optional intervals:
            /// `{ fit, lower, upper, se }`.
            #[wasm_bindgen(js_name = predictWithSe)]
            pub fn predict_with_se(
                &self,
                x: &[f64],
                n_rows: usize,
                pred_type: Option<String>,
                interval: Option<String>,
                level: Option<f64>,
            ) -> Result<JsValue, JsError> {
                let x = $crate::regressors::to_mat(x, n_rows, self.n_features)?;
                let pt = $crate::regressors::parse_prediction_type(pred_type.as_deref())?;
                let it = interval
                    .as_deref()
                    .map($crate::regressors::parse_interval)
                    .transpose()?;
                let r = self.fitted.predict_with_se(&x, pt, it, level.unwrap_or(0.95));
                Ok($crate::convert::ToJs::to_js(&r))
            }
        }
    )*};
}

/// GLM residual types: `devianceResiduals`, `pearsonResiduals`,
/// `workingResiduals`.
macro_rules! glm_residual_methods {
    ($($w:ident),* $(,)?) => {$(
        col_methods!($w {
            "devianceResiduals" => deviance_residuals,
            "pearsonResiduals" => pearson_residuals,
            "workingResiduals" => working_residuals,
        });
    )*};
}

/// A matrix (or optional matrix) as a row-major array.
pub(crate) trait FlatMat {
    fn flat(self) -> Option<Vec<f64>>;
}
impl FlatMat for Mat<f64> {
    fn flat(self) -> Option<Vec<f64>> {
        Some(mat_flat(&self))
    }
}
impl FlatMat for &Mat<f64> {
    fn flat(self) -> Option<Vec<f64>> {
        Some(mat_flat(self))
    }
}
impl<T: FlatMat> FlatMat for Option<T> {
    fn flat(self) -> Option<Vec<f64>> {
        self.and_then(FlatMat::flat)
    }
}
impl FlatMat for &[Vec<f64>] {
    fn flat(self) -> Option<Vec<f64>> {
        Some(self.iter().flatten().copied().collect())
    }
}

/// "response" (default) or "link".
pub(crate) fn parse_prediction_type(
    v: Option<&str>,
) -> Result<anofox_regression::core::PredictionType, JsError> {
    use anofox_regression::core::PredictionType;
    match v {
        None | Some("response") => Ok(PredictionType::Response),
        Some("link") => Ok(PredictionType::Link),
        Some(other) => Err(JsError::new(&format!(
            "Unknown type '{}': expected response or link",
            other
        ))),
    }
}

mod aft;
mod aid;
mod alm;
mod bayesian;
mod binomial;
mod bls;
mod continuation_ratio;
mod elastic_net;
mod gamma;
mod glmm;
mod gls;
mod huber;
mod isotonic;
mod lars;
mod lm_dynamic;
mod logistic;
mod lowess;
mod negative_binomial;
mod ols;
mod ordinal;
mod passive_aggressive;
mod penalized_glm;
mod pls;
mod poisson;
mod pspline;
mod quantile;
mod ransac;
mod ridge;
mod rls;
mod theil_sen;
mod tweedie;
mod wls;

pub use aft::*;
pub use aid::*;
pub use alm::*;
pub use bayesian::*;
pub use binomial::*;
pub use bls::*;
pub use continuation_ratio::*;
pub use elastic_net::*;
pub use gamma::*;
pub use glmm::*;
pub use gls::*;
pub use huber::*;
pub use isotonic::*;
pub use lars::*;
pub use lm_dynamic::*;
pub use logistic::*;
pub use lowess::*;
pub use negative_binomial::*;
pub use ols::*;
pub use ordinal::*;
pub use passive_aggressive::*;
pub use penalized_glm::*;
pub use pls::*;
pub use poisson::*;
pub use pspline::*;
pub use quantile::*;
pub use ransac::*;
pub use ridge::*;
pub use rls::*;
pub use theil_sen::*;
pub use tweedie::*;
pub use wls::*;

use anofox_regression::core::NaAction;
use anofox_regression::solvers::penalized_glm::PriorSpec;
use faer::{Col, ColRef, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Parse an R `na.action` name: "omit", "exclude", "fail" or "pass".
pub(crate) fn parse_na_action(action: &str) -> Result<NaAction, JsError> {
    match action {
        "omit" => Ok(NaAction::Omit),
        "exclude" => Ok(NaAction::Exclude),
        "fail" => Ok(NaAction::Fail),
        "pass" => Ok(NaAction::Pass),
        other => Err(JsError::new(&format!(
            "Unknown na action '{}': expected omit, exclude, fail or pass",
            other
        ))),
    }
}

/// A row-aligned vector (weights, offset) as a column, checked against the
/// number of rows.
pub(crate) fn row_column(
    v: &Option<Vec<f64>>,
    n_rows: usize,
    what: &str,
) -> Result<Option<Col<f64>>, JsError> {
    match v {
        Some(v) if v.len() != n_rows => Err(JsError::new(&format!(
            "Expected {} {} values, got {}",
            n_rows,
            what,
            v.len()
        ))),
        Some(v) => Ok(Some(Col::from_fn(n_rows, |i| v[i]))),
        None => Ok(None),
    }
}

/// A row-major `n_rows x n_cols` matrix from a flat JS array.
pub(crate) fn to_mat(x: &[f64], n_rows: usize, n_cols: usize) -> Result<Mat<f64>, JsError> {
    if x.len() != n_rows * n_cols {
        return Err(JsError::new(&format!(
            "Expected {} elements for {}x{} matrix, got {}",
            n_rows * n_cols,
            n_rows,
            n_cols,
            x.len()
        )));
    }
    Ok(Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]))
}

/// A column from a JS array that must have one value per row.
pub(crate) fn to_col(v: &[f64], n_rows: usize, what: &str) -> Result<Col<f64>, JsError> {
    if v.len() != n_rows {
        return Err(JsError::new(&format!(
            "Expected {} {} values, got {}",
            n_rows,
            what,
            v.len()
        )));
    }
    Ok(Col::from_fn(n_rows, |i| v[i]))
}

/// A faer column as a JS array.
pub(crate) fn col_vec(c: ColRef<'_, f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

/// A faer matrix as a row-major flat JS array.
pub(crate) fn mat_flat(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
        .collect()
}

/// A faer matrix as nested rows (for serde results).
pub(crate) fn mat_rows(m: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

/// Serialize a result struct to a JS object.
pub(crate) fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(value).map_err(|e| JsError::new(&e.to_string()))
}

/// Map a solver error to a JS error.
pub(crate) fn fit_error(e: impl std::fmt::Display) -> JsError {
    JsError::new(&format!("Regression failed: {}", e))
}

/// A coefficient prior as passed from JS:
/// `{ kind: "flat" | "normal" | "laplace", loc?: number, scale?: number }`.
#[derive(Serialize, Deserialize)]
pub(crate) struct JsPrior {
    kind: String,
    #[serde(default)]
    loc: f64,
    #[serde(default)]
    scale: Option<f64>,
}

/// Parse a JS array of priors (see [`JsPrior`]).
pub(crate) fn parse_priors(priors: JsValue) -> Result<Vec<PriorSpec>, JsError> {
    let priors: Vec<JsPrior> =
        serde_wasm_bindgen::from_value(priors).map_err(|e| JsError::new(&e.to_string()))?;
    priors
        .into_iter()
        .map(|p| match (p.kind.as_str(), p.scale) {
            ("flat", _) => Ok(PriorSpec::flat()),
            ("normal", Some(scale)) => Ok(PriorSpec::normal(p.loc, scale)),
            ("laplace", Some(scale)) => Ok(PriorSpec::laplace(p.loc, scale)),
            ("normal" | "laplace", None) => {
                Err(JsError::new(&format!("A {} prior needs a scale", p.kind)))
            }
            (other, _) => Err(JsError::new(&format!(
                "Unknown prior kind '{}': expected flat, normal or laplace",
                other
            ))),
        })
        .collect()
}

/// Parse a covariance type for penalized/MAP fits: "laplace", "sandwich"
/// or "naive".
pub(crate) fn parse_vcov_type(
    vcov: &str,
) -> Result<anofox_regression::solvers::penalized_glm::VcovType, JsError> {
    use anofox_regression::solvers::penalized_glm::VcovType;
    match vcov {
        "laplace" => Ok(VcovType::Laplace),
        "sandwich" => Ok(VcovType::Sandwich),
        "naive" => Ok(VcovType::Naive),
        other => Err(JsError::new(&format!(
            "Unknown vcov type '{}': expected laplace, sandwich or naive",
            other
        ))),
    }
}
