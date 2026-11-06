//! Response transformations (`anofox_regression::transform`, RMS Ch 16):
//! Box–Cox (`MASS::boxcox`), Duan's smearing estimate
//! (`Hmisc::smearingEst`) and additive regression with optimal
//! transformations (`Hmisc::areg`).

use anofox_regression::transform::{
    areg as rust_areg, box_cox, box_cox_inverse, box_cox_transform, smearing_estimate,
    Areg as RustAreg, AregBasis, AregBootstrap, AregOptions, AregType, BoxCoxOptions,
    BoxCoxProfile, Smearing,
};
use faer::Col;
use js_sys::Function;
use serde::Deserialize;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, object, ToJs};
use crate::reduction::parse;
use crate::regressors::{fit_error, to_col, to_mat};

impl_to_js!(BoxCoxProfile {
    lambdas,
    log_likelihood,
    lambda_hat,
    max_log_likelihood,
    ci_lower,
    ci_upper
});
impl_to_js!(AregBootstrap {
    r2_optimism,
    r2_corrected,
    mad_optimism,
    mad_corrected,
    med_optimism,
    med_corrected
});

/// Box–Cox transform `(y^λ − 1)/λ` (`log y` at λ = 0).
#[wasm_bindgen(js_name = boxCoxTransform)]
pub fn box_cox_transform_js(y: f64, lambda: f64) -> f64 {
    box_cox_transform(y, lambda)
}

/// Inverse Box–Cox transform.
#[wasm_bindgen(js_name = boxCoxInverse)]
pub fn box_cox_inverse_js(z: f64, lambda: f64) -> f64 {
    box_cox_inverse(z, lambda)
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct BoxCoxOpts {
    lambdas: Option<Vec<f64>>,
    eps: Option<f64>,
    with_intercept: Option<bool>,
    confidence_level: Option<f64>,
}

/// Box–Cox profile log-likelihood of `y` (> 0) on `x` (`MASS::boxcox`);
/// options `{ lambdas?, eps?, withIntercept?, confidenceLevel? }` (default
/// grid −2 … 2 by 0.1). Returns `{ lambdas, logLikelihood, lambdaHat,
/// maxLogLikelihood, ciLower?, ciUpper? }` (limits undefined when the
/// interval leaves the grid).
#[wasm_bindgen(js_name = boxCox)]
pub fn box_cox_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o: BoxCoxOpts = parse(options)?;
    let mut opts = BoxCoxOptions::default();
    if let Some(v) = o.lambdas {
        opts = opts.lambdas(v);
    }
    if let Some(v) = o.eps {
        opts = opts.eps(v);
    }
    if let Some(v) = o.with_intercept {
        opts = opts.with_intercept(v);
    }
    if let Some(v) = o.confidence_level {
        opts = opts.confidence_level(v);
    }
    let x = to_mat(x, n_rows, n_cols)?;
    let y = Col::from_fn(y.len(), |i| y[i]);
    Ok(box_cox(&x, &y, &opts).map_err(fit_error)?.to_js())
}

/// Duan's smearing estimate on the original scale from the linear
/// predictors `lp` and residuals of a model of `g(y)`, with `inverse` =
/// `g⁻¹` (a JS function). `statistic` is "mean" (default, `E[Y | x]`),
/// "median", "fitted" (naive back-transform) or a number `q` (quantile).
#[wasm_bindgen(js_name = smearingEstimate)]
pub fn smearing_estimate_js(
    lp: &[f64],
    residuals: &[f64],
    inverse: &Function,
    statistic: JsValue,
) -> Result<Vec<f64>, JsError> {
    let statistic = if let Some(q) = statistic.as_f64() {
        Smearing::Quantile(q)
    } else {
        match statistic.as_string().as_deref() {
            None | Some("mean") => Smearing::Mean,
            Some("median") => Smearing::Median,
            Some("fitted") => Smearing::Fitted,
            Some(other) => {
                return Err(JsError::new(&format!(
                    "Unknown statistic '{}': expected mean, median, fitted or a quantile",
                    other
                )))
            }
        }
    };
    let err = RefCell::new(None::<String>);
    let out = smearing_estimate(
        lp,
        residuals,
        |z| match inverse.call1(&JsValue::NULL, &JsValue::from_f64(z)) {
            Ok(v) => v.as_f64().unwrap_or_else(|| {
                err.borrow_mut()
                    .get_or_insert_with(|| "inverse must return a number".into());
                f64::NAN
            }),
            Err(e) => {
                err.borrow_mut().get_or_insert_with(|| {
                    format!(
                        "inverse threw: {}",
                        String::from(js_sys::Error::from(e).message())
                    )
                });
                f64::NAN
            }
        },
        statistic,
    );
    if let Some(e) = err.into_inner() {
        return Err(JsError::new(&e));
    }
    out.map_err(fit_error)
}

// --------------------------------------------------------------- areg ----

fn parse_areg_type(name: &str) -> Result<AregType, JsError> {
    match name.to_ascii_lowercase().as_str() {
        "s" | "spline" => Ok(AregType::Spline),
        "l" | "linear" => Ok(AregType::Linear),
        "c" | "categorical" => Ok(AregType::Categorical),
        other => Err(JsError::new(&format!(
            "Unknown type '{}': expected spline (s), linear (l) or categorical (c)",
            other
        ))),
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct AregOpts {
    nk: Option<usize>,
    x_types: Option<Vec<String>>,
    y_type: Option<String>,
}

/// A fitted `Hmisc::areg` model.
#[wasm_bindgen]
pub struct Areg {
    inner: RustAreg,
}

/// A basis as `{ kind: "linear" | "spline" | "categorical", knots?, levels? }`.
#[derive(serde::Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AregBasisJs {
    kind: String,
    knots: Option<Vec<f64>>,
    levels: Option<Vec<f64>>,
}

impl AregBasisJs {
    fn from_rust(b: &AregBasis) -> Self {
        let (kind, knots, levels) = match b {
            AregBasis::Spline { knots } => ("spline", Some(knots.clone()), None),
            AregBasis::Categorical { levels } => ("categorical", None, Some(levels.clone())),
            _ => ("linear", None, None),
        };
        Self {
            kind: kind.to_string(),
            knots,
            levels,
        }
    }

    fn to_rust(&self) -> Result<AregBasis, JsError> {
        match self.kind.as_str() {
            "linear" => Ok(AregBasis::Linear),
            "spline" => Ok(AregBasis::Spline {
                knots: self.knots.clone().unwrap_or_default(),
            }),
            "categorical" => Ok(AregBasis::Categorical {
                levels: self.levels.clone().unwrap_or_default(),
            }),
            other => Err(JsError::new(&format!(
                "Unknown basis '{other}': expected linear, spline or categorical"
            ))),
        }
    }
}

#[wasm_bindgen]
impl Areg {
    /// Rebuild a fitted model from `xBases()`, `getResult().xCoefficients`,
    /// `yBasis()`, `getResult().yCoefficients` and `yRange()` of an earlier
    /// fit; it transforms, predicts and back-transforms as that fit.
    #[wasm_bindgen(js_name = fromParts)]
    pub fn from_parts(
        x_bases: JsValue,
        x_coefficients: Vec<f64>,
        y_basis: JsValue,
        y_coefficients: Vec<f64>,
        y_min: f64,
        y_max: f64,
    ) -> Result<Areg, JsError> {
        let xb: Vec<AregBasisJs> = serde_wasm_bindgen::from_value(x_bases)
            .map_err(|e| JsError::new(&format!("xBases: {e}")))?;
        let yb: AregBasisJs = serde_wasm_bindgen::from_value(y_basis)
            .map_err(|e| JsError::new(&format!("yBasis: {e}")))?;
        let x_bases = xb
            .iter()
            .map(AregBasisJs::to_rust)
            .collect::<Result<Vec<_>, _>>()?;
        let inner = RustAreg::from_parts(
            x_bases,
            faer::Col::from_fn(x_coefficients.len(), |j| x_coefficients[j]),
            yb.to_rust()?,
            faer::Col::from_fn(y_coefficients.len(), |j| y_coefficients[j]),
            (y_min, y_max),
        )
        .map_err(fit_error)?;
        Ok(Self { inner })
    }

    /// The basis of each predictor: `[{ kind, knots?, levels? }]`.
    #[wasm_bindgen(js_name = xBases)]
    pub fn x_bases(&self) -> Result<JsValue, JsError> {
        let b: Vec<AregBasisJs> = self
            .inner
            .x_bases()
            .iter()
            .map(AregBasisJs::from_rust)
            .collect();
        crate::regressors::to_js(&b)
    }

    /// The basis of the response: `{ kind, knots?, levels? }`.
    #[wasm_bindgen(js_name = yBasis)]
    pub fn y_basis(&self) -> Result<JsValue, JsError> {
        crate::regressors::to_js(&AregBasisJs::from_rust(&self.inner.y_basis()))
    }

    /// `[min, max]` of the fitted response.
    #[wasm_bindgen(js_name = yRange)]
    pub fn y_range(&self) -> Vec<f64> {
        let (lo, hi) = self.inner.y_range();
        vec![lo, hi]
    }

    /// `{ xCoefficients, yCoefficients, rSquared, xDf, yDf, transformedY,
    /// transformedX, linearPredictors, residuals, mad?, med? }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        let a = &self.inner;
        object(&[
            ("x_coefficients", a.x_coefficients.to_js()),
            ("y_coefficients", a.y_coefficients.to_js()),
            ("r_squared", a.r_squared.to_js()),
            ("x_df", a.x_df.to_js()),
            ("y_df", a.y_df.to_js()),
            ("transformed_y", a.transformed_y.to_js()),
            ("transformed_x", a.transformed_x.to_js()),
            ("linear_predictors", a.linear_predictors.to_js()),
            ("residuals", a.residuals.to_js()),
            ("mad", a.mad.to_js()),
            ("med", a.med.to_js()),
        ])
    }

    /// Knots of the response spline (undefined unless the response is a
    /// spline).
    #[wasm_bindgen(js_name = yKnots)]
    pub fn y_knots(&self) -> Option<Vec<f64>> {
        self.inner.y_knots().map(<[f64]>::to_vec)
    }

    /// The fitted transformation of the response at `y`.
    #[wasm_bindgen(js_name = transformY)]
    pub fn transform_y(&self, y: &[f64]) -> Vec<f64> {
        self.inner.transform_y(y)
    }

    /// Predicted transformed response for new rows.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let p = self.inner.x_df.len();
        Ok(self.inner.predict(&to_mat(x, n_rows, p)?))
    }

    /// Back-transform of transformed values to the response scale
    /// (undefined when the transformation is not monotone or categorical).
    #[wasm_bindgen(js_name = inverseY)]
    pub fn inverse_y(&self, t: &[f64]) -> Option<Vec<f64>> {
        self.inner.inverse_y(t)
    }

    /// Bootstrap overfitting correction of R², MAD and median absolute
    /// error from resampled row indices (arrays of `n` rows):
    /// `{ r2Optimism, r2Corrected, madOptimism, madCorrected, medOptimism,
    /// medCorrected }`.
    #[wasm_bindgen]
    pub fn bootstrap(&self, indices: JsValue) -> Result<JsValue, JsError> {
        let idx: Vec<Vec<usize>> =
            serde_wasm_bindgen::from_value(indices).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(self.inner.bootstrap(&idx).map_err(fit_error)?.to_js())
    }
}

/// Additive regression with optimal transformations of both sides
/// (`Hmisc::areg`); options `{ nk? (default 4), xTypes?: ("spline" |
/// "linear" | "categorical")[], yType? }` (also R's "s" / "l" / "c").
#[wasm_bindgen]
pub fn areg(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    options: JsValue,
) -> Result<Areg, JsError> {
    let o: AregOpts = parse(options)?;
    let mut opts = AregOptions::default();
    if let Some(v) = o.nk {
        opts = opts.nk(v);
    }
    if let Some(t) = o.x_types {
        let types = t
            .iter()
            .map(|s| parse_areg_type(s))
            .collect::<Result<Vec<_>, _>>()?;
        opts = opts.x_types(Some(types));
    }
    if let Some(t) = o.y_type {
        opts = opts.y_type(parse_areg_type(&t)?);
    }
    let x = to_mat(x, n_rows, n_cols)?;
    let _ = to_col(y, n_rows, "y")?;
    Ok(Areg {
        inner: rust_areg(&x, y, &opts).map_err(fit_error)?,
    })
}
