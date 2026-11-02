//! Resampling validation and calibration (`anofox_regression::validation`):
//! `rms::validate`, `rms::calibrate` (logistic, linear, Cox) and
//! `rms::bootcov`, plus the generic engine `predabResample` driven by JS
//! callbacks.
//!
//! Row indices passed to callbacks are `Uint32Array`s (0-based, training
//! rows may repeat). Seeds are non-negative integers up to 2^53.

use anofox_regression::validation::{
    bootcov, bootstrap_rows, calibrate_cph, calibrate_lrm, calibrate_ols, cluster_bootstrap_rows,
    predab_resample, rank_intervals, validate_lrm, validate_ols, BootCov as RustBootCov,
    Calibration, ResampleMethod, Resamples as RustResamples, SurvivalCalibration,
    ValidationTable as RustValidationTable,
};
use faer::Mat;
use js_sys::{Array, Function, Uint32Array};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, object, ToJs};
use crate::regressors::{fit_error, to_mat};
use crate::survival::{cox_options, events, usizes};

impl_to_js!(Calibration {
    predy,
    calibrated_orig,
    calibrated_corrected,
    optimism,
    n,
    mean_abs_error,
    mean_squared_error,
    quantile90_abs_error
});
impl_to_js!(SurvivalCalibration {
    predicted,
    cuts,
    index_orig,
    training,
    test,
    optimism,
    corrected,
    n,
    mean_predicted,
    km,
    km_corrected,
    std_err
});
impl_to_js!(RustValidationTable {
    names,
    index_orig,
    training,
    test,
    optimism,
    corrected,
    n,
    failed
});
impl_to_js!(RustBootCov {
    replicates,
    mean,
    vcov,
    failed
});

// ------------------------------------------------------------ helpers ----

fn seed(v: f64) -> Result<u64, JsError> {
    if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= 9_007_199_254_740_992.0 {
        Ok(v as u64)
    } else {
        Err(JsError::new(
            "seed must be a non-negative integer up to 2^53",
        ))
    }
}

fn samples(v: JsValue) -> Result<Vec<Vec<usize>>, JsError> {
    serde_wasm_bindgen::from_value(v).map_err(|e| {
        JsError::new(&format!(
            "samples: expected an array of row-index arrays ({})",
            e
        ))
    })
}

fn rows_js(rows: &[usize]) -> Uint32Array {
    let v: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
    Uint32Array::from(v.as_slice())
}

/// The first error raised by a JS callback, reported after the Rust call.
#[derive(Default)]
struct CallbackError(RefCell<Option<String>>);

impl CallbackError {
    fn set(&self, what: &str, e: JsValue) {
        let msg = e
            .dyn_ref::<js_sys::Error>()
            .map(|e| String::from(e.message()))
            .or_else(|| e.as_string())
            .unwrap_or_else(|| format!("{:?}", e));
        self.0
            .borrow_mut()
            .get_or_insert_with(|| format!("{} threw: {}", what, msg));
    }

    fn check(self) -> Result<(), JsError> {
        match self.0.into_inner() {
            Some(e) => Err(JsError::new(&e)),
            None => Ok(()),
        }
    }
}

/// Numbers from a JS array-like value (`number[]` or a typed array).
fn numbers(v: &JsValue) -> Option<Vec<f64>> {
    if !(Array::is_array(v) || v.is_instance_of::<js_sys::Float64Array>()) {
        return None;
    }
    Array::from(v).iter().map(|x| x.as_f64()).collect()
}

fn validate_xy(x: &[f64], n_rows: usize, n_cols: usize, y: &[f64]) -> Result<Mat<f64>, JsError> {
    let m = to_mat(x, n_rows, n_cols)?;
    if y.len() != n_rows {
        return Err(JsError::new(&format!(
            "Expected {} y values, got {}",
            n_rows,
            y.len()
        )));
    }
    Ok(m)
}

// ---------------------------------------------------------- resamples ----

/// Training / test splits of a resampling validation: optimism bootstrap,
/// B-fold cross-validation or Efron's .632.
#[wasm_bindgen]
pub struct Resamples {
    inner: RustResamples,
}

#[wasm_bindgen]
impl Resamples {
    /// Optimism bootstrap from explicit samples (arrays of `n` row
    /// indices); the test set is all rows.
    #[wasm_bindgen]
    pub fn bootstrap(n: usize, samples_: JsValue) -> Result<Resamples, JsError> {
        Ok(Self {
            inner: RustResamples::bootstrap(n, samples(samples_)?).map_err(fit_error)?,
        })
    }

    /// `b` seeded bootstrap samples of `n` rows.
    #[wasm_bindgen(js_name = bootstrapSeeded)]
    pub fn bootstrap_seeded(n: usize, b: usize, seed_: f64) -> Result<Resamples, JsError> {
        Ok(Self {
            inner: RustResamples::bootstrap_seeded(n, b, seed(seed_)?).map_err(fit_error)?,
        })
    }

    /// B-fold cross-validation over the permutation `perm` of `0..n`
    /// (rms's group boundaries).
    #[wasm_bindgen]
    pub fn crossvalidation(perm: &[u32], b: usize) -> Result<Resamples, JsError> {
        Ok(Self {
            inner: RustResamples::crossvalidation(&usizes(perm), b).map_err(fit_error)?,
        })
    }

    /// Seeded B-fold cross-validation.
    #[wasm_bindgen(js_name = crossvalidationSeeded)]
    pub fn crossvalidation_seeded(n: usize, b: usize, seed_: f64) -> Result<Resamples, JsError> {
        Ok(Self {
            inner: RustResamples::crossvalidation_seeded(n, b, seed(seed_)?).map_err(fit_error)?,
        })
    }

    /// Efron's .632 bootstrap from explicit samples; every row must be left
    /// out of at least one sample.
    #[wasm_bindgen]
    pub fn point632(n: usize, samples_: JsValue) -> Result<Resamples, JsError> {
        Ok(Self {
            inner: RustResamples::point632(n, samples(samples_)?).map_err(fit_error)?,
        })
    }

    /// `b` seeded .632 bootstrap samples of `n` rows.
    #[wasm_bindgen(js_name = point632Seeded)]
    pub fn point632_seeded(n: usize, b: usize, seed_: f64) -> Result<Resamples, JsError> {
        Ok(Self {
            inner: RustResamples::point632_seeded(n, b, seed(seed_)?).map_err(fit_error)?,
        })
    }

    /// "bootstrap", "crossvalidation" or "point632".
    #[wasm_bindgen]
    pub fn method(&self) -> String {
        match self.inner.method {
            ResampleMethod::Bootstrap => "bootstrap",
            ResampleMethod::Crossvalidation => "crossvalidation",
            _ => "point632",
        }
        .to_string()
    }

    /// Number of rows.
    #[wasm_bindgen]
    pub fn n(&self) -> usize {
        self.inner.n
    }

    /// The splits: `[[trainingRows, testRows], ...]`.
    #[wasm_bindgen]
    pub fn splits(&self) -> JsValue {
        self.inner.splits.to_js()
    }

    /// Weight of each split's test index (.632; 1 otherwise).
    #[wasm_bindgen]
    pub fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }
}

// --------------------------------------------------------- validation ----

/// Optimism-corrected indexes of a resampling validation.
#[wasm_bindgen]
pub struct ValidationTable {
    inner: RustValidationTable,
}

#[wasm_bindgen]
impl ValidationTable {
    /// `{ names, indexOrig, training, test, optimism, corrected, n, failed }`
    /// (one entry per index in each array).
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// One index: `{ orig, training, test, optimism, corrected, n }`
    /// (undefined for an unknown name).
    #[wasm_bindgen]
    pub fn get(&self, name: &str) -> JsValue {
        match self.inner.get(name) {
            Some((orig, training, test, optimism, corrected, n)) => object(&[
                ("orig", orig.to_js()),
                ("training", training.to_js()),
                ("test", test.to_js()),
                ("optimism", optimism.to_js()),
                ("corrected", corrected.to_js()),
                ("n", n.to_js()),
            ]),
            None => JsValue::UNDEFINED,
        }
    }
}

/// Generic resampling validation (`rms:::predab.resample`) of any modelling
/// process: `fit(rows)` fits on the rows and returns a model (`null` or
/// `undefined` for a failed fit, which is skipped); `measure(model, rows,
/// apparent)` returns the indexes (one per name) of the model on rows,
/// with `apparent` true on the rows it was fitted on.
#[wasm_bindgen(js_name = predabResample)]
pub fn predab_resample_js(
    resamples: &Resamples,
    names: Vec<String>,
    fit: &Function,
    measure: &Function,
) -> Result<ValidationTable, JsError> {
    let err = CallbackError::default();
    let k = names.len();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let table = predab_resample(
        &resamples.inner,
        &refs,
        |rows| match fit.call1(&JsValue::NULL, &rows_js(rows)) {
            Ok(m) if m.is_undefined() || m.is_null() => None,
            Ok(m) => Some(m),
            Err(e) => {
                err.set("fit", e);
                None
            }
        },
        |model, rows, apparent| {
            let r = measure.call3(
                &JsValue::NULL,
                model,
                &rows_js(rows),
                &JsValue::from_bool(apparent),
            );
            match r.map(|v| numbers(&v)) {
                Ok(Some(v)) if v.len() == k => v,
                Ok(_) => {
                    err.set(
                        "measure",
                        JsValue::from_str(&format!("expected an array of {} numbers", k)),
                    );
                    vec![f64::NAN; k]
                }
                Err(e) => {
                    err.set("measure", e);
                    vec![f64::NAN; k]
                }
            }
        },
    );
    err.check()?;
    Ok(ValidationTable {
        inner: table.map_err(fit_error)?,
    })
}

/// `rms::validate.lrm` for a binary logistic model of 0/1 `y` on the
/// row-major `nRows × nCols` predictors `x` (no intercept column):
/// Dxy, R2, Intercept, Slope, Emax, D, U, Q, B, g, gp.
#[wasm_bindgen(js_name = validateLrm)]
pub fn validate_lrm_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    resamples: &Resamples,
) -> Result<ValidationTable, JsError> {
    let x = validate_xy(x, n_rows, n_cols, y)?;
    Ok(ValidationTable {
        inner: validate_lrm(&x, y, &resamples.inner).map_err(fit_error)?,
    })
}

/// `rms::validate.ols` for a linear model: R-square, MSE, g, Intercept,
/// Slope.
#[wasm_bindgen(js_name = validateOls)]
pub fn validate_ols_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    resamples: &Resamples,
) -> Result<ValidationTable, JsError> {
    let x = validate_xy(x, n_rows, n_cols, y)?;
    Ok(ValidationTable {
        inner: validate_ols(&x, y, &resamples.inner).map_err(fit_error)?,
    })
}

/// `rms::calibrate` of a binary logistic model, on the grid `predy`
/// (default: 50 points over the predictions): `{ predy, calibratedOrig,
/// calibratedCorrected, optimism, n, meanAbsError, meanSquaredError,
/// quantile90AbsError }`.
#[wasm_bindgen(js_name = calibrateLrm)]
pub fn calibrate_lrm_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    resamples: &Resamples,
    predy: Option<Vec<f64>>,
) -> Result<JsValue, JsError> {
    let x = validate_xy(x, n_rows, n_cols, y)?;
    Ok(calibrate_lrm(&x, y, &resamples.inner, predy)
        .map_err(fit_error)?
        .to_js())
}

/// `rms::calibrate` of a linear model (as `calibrateLrm`).
#[wasm_bindgen(js_name = calibrateOls)]
pub fn calibrate_ols_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    resamples: &Resamples,
    predy: Option<Vec<f64>>,
) -> Result<JsValue, JsError> {
    let x = validate_xy(x, n_rows, n_cols, y)?;
    Ok(calibrate_ols(&x, y, &resamples.inner, predy)
        .map_err(fit_error)?
        .to_js())
}

/// `rms::calibrate.cph(cmethod = "KM")`: calibration of a Cox model's
/// predicted survival at time `u`, in groups of about `m` subjects (or at
/// the given `cuts` of predicted survival). `coxOptions` as in `coxph`
/// (`rms::cph` uses `{ eps: 1e-4, maxIter: 10 }`). Returns `{ predicted,
/// cuts, indexOrig, training, test, optimism, corrected, n, meanPredicted,
/// km, kmCorrected, stdErr }`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = calibrateCph)]
pub fn calibrate_cph_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    time: &[f64],
    event: &[u8],
    u: f64,
    resamples: &Resamples,
    m: usize,
    cuts: Option<Vec<f64>>,
    cox_opts: JsValue,
) -> Result<JsValue, JsError> {
    let x = validate_xy(x, n_rows, n_cols, time)?;
    let event = events(event, n_rows)?;
    Ok(calibrate_cph(
        &x,
        time,
        &event,
        u,
        &resamples.inner,
        m,
        cuts,
        &cox_options(cox_opts)?,
    )
    .map_err(fit_error)?
    .to_js())
}

// ------------------------------------------------------------ bootcov ----

/// Bootstrap replicates of a fit's coefficients (`rms::bootcov`).
#[wasm_bindgen]
pub struct BootCov {
    inner: RustBootCov,
}

#[wasm_bindgen]
impl BootCov {
    /// `{ replicates, mean, vcov, failed }` (matrices as arrays of rows).
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// Bootstrap covariance, row-major `k × k`.
    #[wasm_bindgen(js_name = getVcov)]
    pub fn get_vcov(&self) -> Vec<f64> {
        crate::regressors::mat_flat(&self.inner.vcov)
    }

    /// Percentile intervals of every coefficient: `[[lower, upper], ...]`.
    #[wasm_bindgen(js_name = percentileCi)]
    pub fn percentile_ci(&self, level: f64) -> JsValue {
        self.inner.percentile_ci(level).to_js()
    }

    /// Percentile interval `[lower, upper]` of the linear combination
    /// `cᵀβ`.
    #[wasm_bindgen(js_name = linearCombinationCi)]
    pub fn linear_combination_ci(&self, c: &[f64], level: f64) -> Result<Vec<f64>, JsError> {
        if c.len() != self.inner.replicates.ncols() {
            return Err(JsError::new(&format!(
                "Expected {} weights, got {}",
                self.inner.replicates.ncols(),
                c.len()
            )));
        }
        let (lo, hi) = self.inner.linear_combination_ci(c, level);
        Ok(vec![lo, hi])
    }
}

/// Bootstrap covariance of any fit: `fit(rows)` returns the coefficient
/// vector of the model refitted on the resampled rows (`null` or
/// `undefined` = failure, skipped), for each sample (from `bootstrapRows`,
/// `clusterBootstrapRows` or given explicitly).
#[wasm_bindgen(js_name = bootcov)]
pub fn bootcov_js(samples_: JsValue, fit: &Function) -> Result<BootCov, JsError> {
    let samples = samples(samples_)?;
    let err = CallbackError::default();
    let k = RefCell::new(None::<usize>);
    let out = bootcov(&samples, |rows| {
        match fit.call1(&JsValue::NULL, &rows_js(rows)) {
            Ok(v) if v.is_undefined() || v.is_null() => None,
            Ok(v) => match numbers(&v) {
                Some(c) if *k.borrow_mut().get_or_insert(c.len()) == c.len() => Some(c),
                _ => {
                    err.set(
                        "fit",
                        JsValue::from_str("expected arrays of numbers of one length"),
                    );
                    None
                }
            },
            Err(e) => {
                err.set("fit", e);
                None
            }
        }
    });
    err.check()?;
    Ok(BootCov {
        inner: out.map_err(fit_error)?,
    })
}

/// `b` seeded ordinary bootstrap samples of `n` rows.
#[wasm_bindgen(js_name = bootstrapRows)]
pub fn bootstrap_rows_js(n: usize, b: usize, seed_: f64) -> Result<JsValue, JsError> {
    Ok(bootstrap_rows(n, b, seed(seed_)?).to_js())
}

/// `b` seeded cluster bootstrap samples: clusters drawn with replacement,
/// all rows of each drawn cluster (one cluster code per row).
#[wasm_bindgen(js_name = clusterBootstrapRows)]
pub fn cluster_bootstrap_rows_js(
    clusters: &[u32],
    b: usize,
    seed_: f64,
) -> Result<JsValue, JsError> {
    Ok(cluster_bootstrap_rows(&usizes(clusters), b, seed(seed_)?).to_js())
}

/// Bootstrap intervals for the ranks of predictor importance: `original`
/// holds one statistic per predictor, `replicates` the same per bootstrap
/// fit (row-major `B × original.length`). Returns `[[rank, lower, upper],
/// ...]` (1 = least important, ties averaged).
#[wasm_bindgen(js_name = rankIntervals)]
pub fn rank_intervals_js(
    original: &[f64],
    replicates: &[f64],
    level: f64,
) -> Result<JsValue, JsError> {
    let m = original.len();
    if m == 0 || !replicates.len().is_multiple_of(m) {
        return Err(JsError::new(
            "replicates must hold original.length values per bootstrap fit",
        ));
    }
    let reps = to_mat(replicates, replicates.len() / m, m)?;
    Ok(rank_intervals(original, &reps, level).to_js())
}
