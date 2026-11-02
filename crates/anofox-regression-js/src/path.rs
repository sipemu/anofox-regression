//! Regularisation paths and cross-validation (`solvers::path`,
//! `solvers::cv`): glmnet-style ridge / elastic-net / lasso paths, k-fold
//! CV of a path, fold assignments (k-fold, group, leave-one-group-out,
//! stratified, time series) and nested CV.
//!
//! Path options are `{ alpha?: number (1 = lasso, 0 = ridge; default 1),
//! withIntercept?, standardize?, tol?, maxIter? }`. Group, stratum and time
//! keys are arrays of numbers or of strings; only their order and equality
//! matter. Splits are `{ train, test }` objects of 0-based row indices.

use anofox_regression::solvers::cv::{
    cv_path_splits, group_codes, group_kfold_ids, group_time_series_splits, logo_ids,
    nested_cv_path, nested_inner_splits, quantile_strata, splits_from_fold_ids,
    stratified_group_kfold_ids, stratified_kfold_ids, time_series_splits, CvMetric, CvSplit,
    GroupBalance, InnerFolds, LambdaRule, NestedCv, NestedCvFold, NestedCvOptions,
    TimeSeriesSplitOptions, TrainWindow,
};
use anofox_regression::solvers::path::{
    cv_path as rust_cv_path, fit_path as rust_fit_path, kfold_ids, lambda_grid,
    CvPath as RustCvPath, PathOptions, RegPath as RustRegPath,
};
use faer::Col;
use js_sys::Array;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, object, ToJs};
use crate::reduction::parse;
use crate::regressors::{fit_error, mat_flat, to_col, to_mat};

impl_to_js!(RustRegPath {
    lambdas,
    intercepts,
    coefficients,
    df,
    alpha
});
impl_to_js!(RustCvPath {
    lambdas,
    cv_mean,
    cv_se,
    df,
    lambda_min,
    lambda_1se,
    index_min,
    index_1se,
    n_folds,
    path
});
impl_to_js!(CvSplit { train, test });
impl_to_js!(NestedCvFold {
    split,
    lambda,
    index,
    test_metric,
    n_train,
    n_test,
    inner
});
impl_to_js!(NestedCv {
    folds,
    mean,
    se,
    metric
});

impl ToJs for CvMetric {
    fn to_js(&self) -> JsValue {
        JsValue::from_str(match self {
            CvMetric::Mae => "mae",
            _ => "mse",
        })
    }
}

// ------------------------------------------------------------ parsing ----

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct PathOpts {
    alpha: Option<f64>,
    with_intercept: Option<bool>,
    standardize: Option<bool>,
    tol: Option<f64>,
    max_iter: Option<usize>,
}

fn path_options(v: JsValue) -> Result<PathOptions, JsError> {
    let o: PathOpts = parse(v)?;
    let mut out = PathOptions::default();
    if let Some(a) = o.alpha {
        out = out.alpha(a);
    }
    if let Some(b) = o.with_intercept {
        out = out.with_intercept(b);
    }
    if let Some(b) = o.standardize {
        out = out.standardize(b);
    }
    if let Some(t) = o.tol {
        out = out.tol(t);
    }
    if let Some(m) = o.max_iter {
        out = out.max_iter(m);
    }
    Ok(out)
}

fn seed(v: Option<f64>) -> Result<Option<u64>, JsError> {
    v.map(|s| {
        if s.is_finite() && s >= 0.0 && s.fract() == 0.0 && s <= 9_007_199_254_740_992.0 {
            Ok(s as u64)
        } else {
            Err(JsError::new(
                "seed must be a non-negative integer up to 2^53",
            ))
        }
    })
    .transpose()
}

/// A number with a total order (for keys).
#[derive(PartialEq)]
struct NumKey(f64);
impl Eq for NumKey {}
impl PartialOrd for NumKey {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for NumKey {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Dense codes of a JS key array (numbers or strings), in key order.
fn key_codes(v: &JsValue, what: &str) -> Result<Vec<usize>, JsError> {
    let arr = Array::from(v);
    if !(Array::is_array(v) || js_sys::ArrayBuffer::is_view(v)) {
        return Err(JsError::new(&format!("{}: expected an array", what)));
    }
    let nums: Option<Vec<NumKey>> = arr.iter().map(|x| x.as_f64().map(NumKey)).collect();
    if let Some(n) = nums {
        return Ok(group_codes(&n).0);
    }
    let strs: Option<Vec<String>> = arr.iter().map(|x| x.as_string()).collect();
    match strs {
        Some(s) => Ok(group_codes(&s).0),
        None => Err(JsError::new(&format!(
            "{}: expected an array of numbers or of strings",
            what
        ))),
    }
}

fn parse_splits(v: JsValue) -> Result<Vec<CvSplit>, JsError> {
    #[derive(Deserialize)]
    struct S {
        train: Vec<usize>,
        test: Vec<usize>,
    }
    let s: Vec<S> = serde_wasm_bindgen::from_value(v)
        .map_err(|e| JsError::new(&format!("splits: expected [{{ train, test }}] ({})", e)))?;
    Ok(s.into_iter()
        .map(|s| CvSplit::new(s.train, s.test))
        .collect())
}

fn parse_metric(v: Option<&str>) -> Result<CvMetric, JsError> {
    match v {
        None | Some("mse") => Ok(CvMetric::Mse),
        Some("mae") => Ok(CvMetric::Mae),
        Some(other) => Err(JsError::new(&format!(
            "Unknown metric '{}': expected mse or mae",
            other
        ))),
    }
}

fn parse_balance(v: Option<&str>) -> Result<GroupBalance, JsError> {
    match v {
        None | Some("rows") => Ok(GroupBalance::Rows),
        Some("random") => Ok(GroupBalance::Random),
        Some(other) => Err(JsError::new(&format!(
            "Unknown balance '{}': expected rows or random",
            other
        ))),
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct TsOpts {
    n_splits: Option<usize>,
    test_size: Option<usize>,
    gap: Option<usize>,
    max_train_size: Option<usize>,
    min_train_size: Option<usize>,
}

fn ts_options(o: TsOpts) -> TimeSeriesSplitOptions {
    let mut out = TimeSeriesSplitOptions::default();
    if let Some(n) = o.n_splits {
        out = out.n_splits(n);
    }
    if o.test_size.is_some() {
        out = out.test_size(o.test_size);
    }
    if let Some(g) = o.gap {
        out = out.gap(g);
    }
    if let Some(m) = o.max_train_size {
        out = out.window(TrainWindow::Sliding(m));
    }
    if o.min_train_size.is_some() {
        out = out.min_train_size(o.min_train_size);
    }
    out
}

// --------------------------------------------------------------- paths ----

/// Coefficient path over a sequence of penalties.
#[wasm_bindgen]
pub struct RegPath {
    inner: RustRegPath,
}

#[wasm_bindgen]
impl RegPath {
    /// `{ lambdas, intercepts, coefficients (p rows of L), df, alpha }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// Number of penalties.
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the path is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Slope coefficients at penalty `l`.
    #[wasm_bindgen(js_name = coefficientsAt)]
    pub fn coefficients_at(&self, l: usize) -> Result<Vec<f64>, JsError> {
        if l >= self.inner.len() {
            return Err(JsError::new(&format!("No penalty {}", l)));
        }
        Ok(self.inner.coefficients_at(l).iter().copied().collect())
    }

    /// Predictions for new rows at every penalty, row-major
    /// `nRows × len()`.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let p = self.inner.coefficients.nrows();
        Ok(mat_flat(&self.inner.predict(&to_mat(x, n_rows, p)?)))
    }
}

/// Cross-validated path (`cv.glmnet`).
#[wasm_bindgen]
pub struct CvPath {
    inner: RustCvPath,
}

#[wasm_bindgen]
impl CvPath {
    /// `{ lambdas, cvMean, cvSe, df, lambdaMin, lambda1se, indexMin,
    /// index1se, nFolds, path }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// The path fitted on all rows.
    #[wasm_bindgen]
    pub fn path(&self) -> RegPath {
        RegPath {
            inner: self.inner.path.clone(),
        }
    }
}

fn xy(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
) -> Result<(faer::Mat<f64>, Col<f64>), JsError> {
    Ok((to_mat(x, n_rows, n_cols)?, to_col(y, n_rows, "y")?))
}

/// glmnet's default penalty grid: `nLambda` (default 100) values
/// log-spaced from `λ_max` down to `λ_max · lambdaMinRatio`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = lambdaGrid)]
pub fn lambda_grid_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    options: JsValue,
    n_lambda: Option<usize>,
    lambda_min_ratio: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let (x, y) = xy(x, n_rows, n_cols, y)?;
    lambda_grid(
        &x,
        &y,
        &path_options(options)?,
        n_lambda.unwrap_or(100),
        lambda_min_ratio,
    )
    .map_err(fit_error)
}

/// Fit the path at `lambdas` (glmnet's Gaussian objective).
#[wasm_bindgen(js_name = fitPath)]
pub fn fit_path_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    lambdas: &[f64],
    options: JsValue,
) -> Result<RegPath, JsError> {
    let (x, y) = xy(x, n_rows, n_cols, y)?;
    Ok(RegPath {
        inner: rust_fit_path(&x, &y, lambdas, &path_options(options)?).map_err(fit_error)?,
    })
}

/// Seeded random fold of each of `n` rows (`0..k`).
#[wasm_bindgen(js_name = kfoldIds)]
pub fn kfold_ids_js(n: usize, k: usize, seed_: f64) -> Result<Vec<usize>, JsError> {
    Ok(kfold_ids(n, k, seed(Some(seed_))?.unwrap_or(0)))
}

/// K-fold CV of the path with fold labels `foldIds` (one per row, at least
/// 3 distinct), as `cv.glmnet(foldid = )`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = cvPath)]
pub fn cv_path_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    lambdas: &[f64],
    fold_ids: &[u32],
    options: JsValue,
) -> Result<CvPath, JsError> {
    let (x, y) = xy(x, n_rows, n_cols, y)?;
    let folds: Vec<usize> = fold_ids.iter().map(|&f| f as usize).collect();
    Ok(CvPath {
        inner: rust_cv_path(&x, &y, lambdas, &folds, &path_options(options)?).map_err(fit_error)?,
    })
}

/// CV of the path over explicit splits, scored by `metric` ("mse" or
/// "mae").
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = cvPathSplits)]
pub fn cv_path_splits_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    lambdas: &[f64],
    splits: JsValue,
    options: JsValue,
    metric: Option<String>,
) -> Result<CvPath, JsError> {
    let (x, y) = xy(x, n_rows, n_cols, y)?;
    Ok(CvPath {
        inner: cv_path_splits(
            &x,
            &y,
            lambdas,
            &parse_splits(splits)?,
            &path_options(options)?,
            parse_metric(metric.as_deref())?,
        )
        .map_err(fit_error)?,
    })
}

// --------------------------------------------------------------- folds ----

/// Dense codes `0..m` of the distinct keys in key order: `{ codes,
/// nGroups }`.
#[wasm_bindgen(js_name = groupCodes)]
pub fn group_codes_js(keys: JsValue) -> Result<JsValue, JsError> {
    let codes = key_codes(&keys, "keys")?;
    let m = codes.iter().max().map_or(0, |&c| c + 1);
    Ok(object(&[("codes", codes.to_js()), ("n_groups", m.to_js())]))
}

/// Group k-fold: every group in one fold; `balance` "rows" (scikit-learn's
/// `GroupKFold`, default) or "random" (seeded shuffle of the groups).
#[wasm_bindgen(js_name = groupKfoldIds)]
pub fn group_kfold_ids_js(
    groups: JsValue,
    k: usize,
    balance: Option<String>,
    seed_: Option<f64>,
) -> Result<Vec<usize>, JsError> {
    group_kfold_ids(
        &key_codes(&groups, "groups")?,
        k,
        parse_balance(balance.as_deref())?,
        seed(seed_)?.unwrap_or(0),
    )
    .map_err(fit_error)
}

/// Leave-one-group-out: the fold of each row is its group's code.
#[wasm_bindgen(js_name = logoIds)]
pub fn logo_ids_js(groups: JsValue) -> Result<Vec<usize>, JsError> {
    Ok(logo_ids(&key_codes(&groups, "groups")?))
}

/// Stratified k-fold (scikit-learn's `StratifiedKFold`): blocks in row
/// order without a seed, a seeded order with one.
#[wasm_bindgen(js_name = stratifiedKfoldIds)]
pub fn stratified_kfold_ids_js(
    strata: JsValue,
    k: usize,
    seed_: Option<f64>,
) -> Result<Vec<usize>, JsError> {
    stratified_kfold_ids(&key_codes(&strata, "strata")?, k, seed(seed_)?).map_err(fit_error)
}

/// Strata of a continuous response: `nBins` quantile bins.
#[wasm_bindgen(js_name = quantileStrata)]
pub fn quantile_strata_js(y: &[f64], n_bins: usize) -> Vec<usize> {
    quantile_strata(&Col::from_fn(y.len(), |i| y[i]), n_bins)
}

/// Stratified group k-fold (`StratifiedGroupKFold`).
#[wasm_bindgen(js_name = stratifiedGroupKfoldIds)]
pub fn stratified_group_kfold_ids_js(
    groups: JsValue,
    strata: JsValue,
    k: usize,
    seed_: Option<f64>,
) -> Result<Vec<usize>, JsError> {
    stratified_group_kfold_ids(
        &key_codes(&groups, "groups")?,
        &key_codes(&strata, "strata")?,
        k,
        seed(seed_)?,
    )
    .map_err(fit_error)
}

/// Splits `{ train, test }` of fold labels (one split per distinct label,
/// in label order).
#[wasm_bindgen(js_name = splitsFromFoldIds)]
pub fn splits_from_fold_ids_js(fold_ids: &[u32]) -> JsValue {
    let f: Vec<usize> = fold_ids.iter().map(|&v| v as usize).collect();
    splits_from_fold_ids(&f).to_js()
}

/// Rolling-origin splits of `n` ordered rows (scikit-learn's
/// `TimeSeriesSplit`): options `{ nSplits? (5), testSize?, gap?,
/// maxTrainSize? (sliding window), minTrainSize? }`.
#[wasm_bindgen(js_name = timeSeriesSplits)]
pub fn time_series_splits_js(n: usize, options: JsValue) -> Result<JsValue, JsError> {
    Ok(time_series_splits(n, &ts_options(parse(options)?))
        .map_err(fit_error)?
        .to_js())
}

/// Rolling-origin splits over the distinct time keys (rows sharing a key
/// stay together).
#[wasm_bindgen(js_name = groupTimeSeriesSplits)]
pub fn group_time_series_splits_js(
    time_keys: JsValue,
    options: JsValue,
) -> Result<JsValue, JsError> {
    Ok(group_time_series_splits(
        &key_codes(&time_keys, "timeKeys")?,
        &ts_options(parse(options)?),
    )
    .map_err(fit_error)?
    .to_js())
}

// ---------------------------------------------------------- nested CV ----

/// Inner folds as passed from JS (keys are per row of the full data).
fn parse_inner(v: &JsValue) -> Result<InnerFolds, JsError> {
    let get =
        |k: &str| js_sys::Reflect::get(v, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED);
    let num = |k: &str| get(k).as_f64();
    let kind = get("type").as_string().unwrap_or_default();
    let k = || {
        num("k")
            .map(|k| k as usize)
            .ok_or_else(|| JsError::new("inner folds: k is required"))
    };
    Ok(match kind.as_str() {
        "kfold" => InnerFolds::Kfold {
            k: k()?,
            seed: seed(num("seed"))?.unwrap_or(0),
        },
        "group" => InnerFolds::Group {
            groups: key_codes(&get("groups"), "groups")?,
            k: k()?,
            balance: parse_balance(get("balance").as_string().as_deref())?,
            seed: seed(num("seed"))?.unwrap_or(0),
        },
        "logo" => InnerFolds::Logo {
            groups: key_codes(&get("groups"), "groups")?,
        },
        "stratified" => InnerFolds::Stratified {
            strata: key_codes(&get("strata"), "strata")?,
            k: k()?,
            seed: seed(num("seed"))?,
        },
        "timeSeries" => {
            let tk = get("timeKeys");
            InnerFolds::TimeSeries {
                opts: ts_options(parse(v.clone())?),
                time_keys: if tk.is_undefined() || tk.is_null() {
                    None
                } else {
                    Some(key_codes(&tk, "timeKeys")?)
                },
            }
        }
        other => {
            return Err(JsError::new(&format!(
                "Unknown inner folds '{}': expected kfold, group, logo, stratified or timeSeries",
                other
            )))
        }
    })
}

/// Inner splits (in full-data row indices) of every outer split; `inner`
/// is `{ type: "kfold", k, seed? } | { type: "group", groups, k, balance?,
/// seed? } | { type: "logo", groups } | { type: "stratified", strata, k,
/// seed? } | { type: "timeSeries", nSplits?, testSize?, gap?,
/// maxTrainSize?, minTrainSize?, timeKeys? }`.
#[wasm_bindgen(js_name = nestedInnerSplits)]
pub fn nested_inner_splits_js(
    n: usize,
    outer: JsValue,
    inner: JsValue,
) -> Result<JsValue, JsError> {
    Ok(
        nested_inner_splits(n, &parse_splits(outer)?, &parse_inner(&inner)?)
            .map_err(fit_error)?
            .to_js(),
    )
}

/// Nested CV of the path: on each outer split, an inner CV picks λ (`rule`
/// "min" or "1se") and the refit is scored on the outer test rows.
/// `nested` is `{ inner, rule?, metric?, nLambda?, lambdaMinRatio? }`;
/// `lambdas` = null uses each training set's glmnet grid. Returns `{ folds,
/// mean, se, metric }`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = nestedCvPath)]
pub fn nested_cv_path_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    lambdas: Option<Vec<f64>>,
    outer: JsValue,
    options: JsValue,
    nested: JsValue,
) -> Result<JsValue, JsError> {
    let (x, y) = xy(x, n_rows, n_cols, y)?;
    let get = |k: &str| {
        js_sys::Reflect::get(&nested, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED)
    };
    let mut opts = NestedCvOptions::new(parse_inner(&get("inner"))?);
    opts = opts.rule(match get("rule").as_string().as_deref() {
        None | Some("min") => LambdaRule::Min,
        Some("1se") => LambdaRule::OneSe,
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown rule '{}': expected min or 1se",
                other
            )))
        }
    });
    opts = opts.metric(parse_metric(get("metric").as_string().as_deref())?);
    if let Some(n) = get("nLambda").as_f64() {
        opts = opts.n_lambda(n as usize);
    }
    if let Some(r) = get("lambdaMinRatio").as_f64() {
        opts = opts.lambda_min_ratio(Some(r));
    }
    Ok(nested_cv_path(
        &x,
        &y,
        lambdas.as_deref(),
        &parse_splits(outer)?,
        &path_options(options)?,
        &opts,
    )
    .map_err(fit_error)?
    .to_js())
}
