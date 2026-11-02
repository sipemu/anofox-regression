//! Data reduction (`anofox_regression::reduction`, RMS §4.7): Spearman ρ²
//! screening, Hoeffding's D, hierarchical clustering, `varclus`, `redun`,
//! principal components, cluster scores and `transcan`.
//!
//! Inputs are row-major `nRows × nCols` matrices; matrices in results are
//! arrays of rows. Missing values are `NaN`.

use anofox_regression::reduction::{
    cluster_scores, hclust as rust_hclust, hoeffding_d, principal_components, redun, spearman2,
    spearman2_categorical, transcan, varclus as rust_varclus, Hclust as RustHclust, HoeffdingD,
    Linkage, Node, PrincipalComponents as RustPrincipalComponents, Redun, RedunOptions,
    RedunVariable, Similarity, SimilarityTransform, Spearman2, Transcan, TranscanOptions,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, object, ToJs};
use crate::regressors::{fit_error, mat_flat, to_mat};
use crate::survival::usizes;

impl_to_js!(Spearman2 {
    rho2,
    f,
    df1,
    df2,
    p_value,
    n,
    adjusted_rho2
});
impl_to_js!(HoeffdingD { d, p_value });
impl_to_js!(Redun {
    kept,
    removed,
    r_squared,
    r_squared_later,
    r_squared_first,
    too_few
});
impl_to_js!(Transcan {
    transformed,
    r_squared,
    r_squared_adj,
    iterations
});

/// R's `merge` convention: `-(i + 1)` for leaf `i`, `k + 1` for the
/// cluster formed at step `k`.
impl ToJs for Node {
    fn to_js(&self) -> JsValue {
        JsValue::from_f64(match *self {
            Node::Leaf(i) => -(i as f64 + 1.0),
            Node::Cluster(k) => k as f64 + 1.0,
        })
    }
}

pub(crate) fn parse<T: for<'de> Deserialize<'de> + Default>(v: JsValue) -> Result<T, JsError> {
    if v.is_undefined() || v.is_null() {
        return Ok(T::default());
    }
    serde_wasm_bindgen::from_value(v).map_err(|e| JsError::new(&e.to_string()))
}

fn unknown(what: &str, got: &str, expected: &str) -> JsError {
    JsError::new(&format!(
        "Unknown {} '{}': expected {}",
        what, got, expected
    ))
}

fn parse_linkage(name: Option<&str>) -> Result<Linkage, JsError> {
    Ok(match name.map(str::to_ascii_lowercase).as_deref() {
        None | Some("complete") => Linkage::Complete,
        Some("ward.d") => Linkage::WardD,
        Some("ward.d2") => Linkage::WardD2,
        Some("single") => Linkage::Single,
        Some("average") => Linkage::Average,
        Some("mcquitty") => Linkage::Mcquitty,
        Some("median") => Linkage::Median,
        Some("centroid") => Linkage::Centroid,
        Some(other) => {
            return Err(unknown(
                "linkage",
                other,
                "ward.D, ward.D2, single, complete, average, mcquitty, median or centroid",
            ))
        }
    })
}

// ---------------------------------------------------------- screening ----

/// Spearman ρ² of `y` on the ranks of `x` (`p = 2` adds their square,
/// default) (`Hmisc::spearman2`): `{ rho2, f, df1, df2, pValue, n,
/// adjustedRho2 }`.
#[wasm_bindgen(js_name = spearman2)]
pub fn spearman2_js(x: &[f64], y: &[f64], p: Option<usize>) -> Result<JsValue, JsError> {
    Ok(spearman2(x, y, p.unwrap_or(2)).map_err(fit_error)?.to_js())
}

/// `spearman2` of `y` on a categorical `x` (level codes).
#[wasm_bindgen(js_name = spearman2Categorical)]
pub fn spearman2_categorical_js(x: &[u32], y: &[f64]) -> Result<JsValue, JsError> {
    Ok(spearman2_categorical(&usizes(x), y)
        .map_err(fit_error)?
        .to_js())
}

/// Hoeffding's D between every pair of columns (`Hmisc::hoeffd`):
/// `{ d, pValue }` (`p × p`, D scaled by 30).
#[wasm_bindgen(js_name = hoeffdingD)]
pub fn hoeffding_d_js(x: &[f64], n_rows: usize, n_cols: usize) -> Result<JsValue, JsError> {
    Ok(hoeffding_d(&to_mat(x, n_rows, n_cols)?)
        .map_err(fit_error)?
        .to_js())
}

// --------------------------------------------------------- clustering ----

/// A hierarchical clustering (`stats::hclust`).
#[wasm_bindgen]
pub struct Hclust {
    inner: RustHclust,
}

#[wasm_bindgen]
impl Hclust {
    /// `{ merge, height, order }`: `merge` holds one `[a, b]` pair per
    /// step in R's convention (`-(i + 1)` = object `i`, `k + 1` = the
    /// cluster of step `k`); `order` is 0-based.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        let h = &self.inner;
        object(&[
            ("merge", h.merge.to_js()),
            ("height", h.height.to_js()),
            ("order", h.order.to_js()),
        ])
    }

    /// Group (0-based) of every object after cutting the tree at `height`
    /// (`cutree(h = )`).
    #[wasm_bindgen(js_name = cutHeight)]
    pub fn cut_height(&self, height: f64) -> Vec<usize> {
        self.inner.cut_height(height)
    }
}

/// Agglomerative clustering of the `n × n` dissimilarities `diss`
/// (row-major); `linkage` is "complete" (default), "single", "average",
/// "mcquitty", "median", "centroid", "ward.D" or "ward.D2".
#[wasm_bindgen]
pub fn hclust(diss: &[f64], n: usize, linkage: Option<String>) -> Result<Hclust, JsError> {
    let d = to_mat(diss, n, n)?;
    Ok(Hclust {
        inner: rust_hclust(&d, parse_linkage(linkage.as_deref())?).map_err(fit_error)?,
    })
}

/// Variable clustering (`Hmisc::varclus`): the similarity matrix and its
/// clustering tree.
#[wasm_bindgen]
pub struct Varclus {
    sim: faer::Mat<f64>,
    hclust: RustHclust,
}

#[wasm_bindgen]
impl Varclus {
    /// Similarity matrix (`p` rows of `p`).
    #[wasm_bindgen]
    pub fn sim(&self) -> JsValue {
        self.sim.to_js()
    }

    /// The clustering of `1 − similarity`.
    #[wasm_bindgen]
    pub fn hclust(&self) -> Hclust {
        Hclust {
            inner: self.hclust.clone(),
        }
    }
}

#[derive(Deserialize, Default)]
struct VarclusOpts {
    similarity: Option<String>,
    transform: Option<String>,
    linkage: Option<String>,
}

/// `Hmisc::varclus` of the columns of `x`; options `{ similarity?:
/// "spearman" | "pearson" | "hoeffding", transform?: "square" | "abs" |
/// "none", linkage? }` (defaults: Spearman, squared, complete).
#[wasm_bindgen]
pub fn varclus(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    options: JsValue,
) -> Result<Varclus, JsError> {
    let o: VarclusOpts = parse(options)?;
    let similarity = match o
        .similarity
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("spearman") => Similarity::Spearman,
        Some("pearson") => Similarity::Pearson,
        Some("hoeffding") => Similarity::Hoeffding,
        Some(other) => {
            return Err(unknown(
                "similarity",
                other,
                "spearman, pearson or hoeffding",
            ))
        }
    };
    let transform = match o
        .transform
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("square") => SimilarityTransform::Square,
        Some("abs") => SimilarityTransform::Abs,
        Some("none") => SimilarityTransform::None,
        Some(other) => return Err(unknown("transform", other, "square, abs or none")),
    };
    let v = rust_varclus(
        &to_mat(x, n_rows, n_cols)?,
        similarity,
        transform,
        parse_linkage(o.linkage.as_deref())?,
    )
    .map_err(fit_error)?;
    Ok(Varclus {
        sim: v.sim,
        hclust: v.hclust,
    })
}

// ------------------------------------------------------------- redun ----

pub(crate) fn variable_kinds<T>(
    kinds: Vec<String>,
    numeric: T,
    linear: T,
    categorical: T,
) -> Result<Vec<T>, JsError>
where
    T: Copy,
{
    kinds
        .iter()
        .map(|k| match k.to_ascii_lowercase().as_str() {
            "numeric" => Ok(numeric),
            "linear" => Ok(linear),
            "categorical" => Ok(categorical),
            other => Err(unknown("kind", other, "numeric, linear or categorical")),
        })
        .collect()
}

#[derive(Deserialize, Default)]
struct RedunOpts {
    r2: Option<f64>,
    adjusted: Option<bool>,
    nk: Option<usize>,
    tlinear: Option<bool>,
    rank: Option<bool>,
    qrank: Option<bool>,
    allcat: Option<bool>,
}

/// Redundancy analysis (`Hmisc::redun`): `kinds` gives "numeric" (spline),
/// "linear" or "categorical" per column; options `{ r2?, adjusted?, nk?,
/// tlinear?, rank?, qrank?, allcat? }` (defaults as Hmisc: r2 0.9, nk 3).
/// Returns `{ kept, removed, rSquared, rSquaredLater, rSquaredFirst,
/// tooFew }` with 0-based column indices.
#[wasm_bindgen(js_name = redun)]
pub fn redun_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    kinds: Vec<String>,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let kinds = variable_kinds(
        kinds,
        RedunVariable::Numeric,
        RedunVariable::Linear,
        RedunVariable::Categorical,
    )?;
    let o: RedunOpts = parse(options)?;
    let mut opts = RedunOptions::default();
    if let Some(v) = o.r2 {
        opts = opts.r2(v);
    }
    if let Some(v) = o.adjusted {
        opts = opts.adjusted(v);
    }
    if let Some(v) = o.nk {
        opts = opts.nk(v);
    }
    if let Some(v) = o.tlinear {
        opts = opts.tlinear(v);
    }
    if let Some(v) = o.rank {
        opts = opts.rank(v);
    }
    if let Some(v) = o.qrank {
        opts = opts.qrank(v);
    }
    if let Some(v) = o.allcat {
        opts = opts.allcat(v);
    }
    Ok(redun(&to_mat(x, n_rows, n_cols)?, &kinds, &opts)
        .map_err(fit_error)?
        .to_js())
}

// ---------------------------------------------- principal components ----

/// Principal components (`stats::prcomp`).
#[wasm_bindgen]
pub struct PrincipalComponents {
    inner: RustPrincipalComponents,
}

#[wasm_bindgen]
impl PrincipalComponents {
    /// `{ center, scale?, sdev, rotation, scores }` (rotation `p × p`,
    /// scores `n × p`, as arrays of rows).
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        let p = &self.inner;
        object(&[
            ("center", p.center.to_js()),
            ("scale", p.scale.to_js()),
            ("sdev", p.sdev.to_js()),
            ("rotation", p.rotation.to_js()),
            ("scores", p.scores.to_js()),
        ])
    }

    /// Proportion of variance of each component.
    #[wasm_bindgen(js_name = proportionOfVariance)]
    pub fn proportion_of_variance(&self) -> Vec<f64> {
        self.inner.proportion_of_variance()
    }

    /// Scores of new rows on the first `k` components, row-major
    /// `nRows × k`.
    #[wasm_bindgen]
    pub fn transform(&self, x: &[f64], n_rows: usize, k: usize) -> Result<Vec<f64>, JsError> {
        let p = self.inner.center.len();
        Ok(mat_flat(&self.inner.transform(&to_mat(x, n_rows, p)?, k)))
    }
}

/// Principal components of the columns of `x`, centred and (with `scale`)
/// scaled to unit variance.
#[wasm_bindgen(js_name = principalComponents)]
pub fn principal_components_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    scale: bool,
) -> Result<PrincipalComponents, JsError> {
    Ok(PrincipalComponents {
        inner: principal_components(&to_mat(x, n_rows, n_cols)?, scale).map_err(fit_error)?,
    })
}

/// Summary scores of variable clusters (RMS §4.7): the first `k` scaled
/// principal components of each cluster (array of column-index arrays),
/// side by side; row-major `nRows × columns`.
#[wasm_bindgen(js_name = clusterScores)]
pub fn cluster_scores_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    clusters: JsValue,
    k: usize,
) -> Result<Vec<f64>, JsError> {
    let clusters: Vec<Vec<usize>> =
        serde_wasm_bindgen::from_value(clusters).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(mat_flat(
        &cluster_scores(&to_mat(x, n_rows, n_cols)?, &clusters, k).map_err(fit_error)?,
    ))
}

// ----------------------------------------------------------- transcan ----

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct TranscanOpts {
    nk: Option<usize>,
    iter_max: Option<usize>,
    eps: Option<f64>,
    curtail: Option<bool>,
}

/// `Hmisc::transcan(method = "canonical")`: optimal transformations of
/// every column (missing values allowed); `categorical` flags each column
/// (1 = categorical); options `{ nk?, iterMax?, eps?, curtail? }`. Returns
/// `{ transformed, rSquared, rSquaredAdj, iterations }`.
#[wasm_bindgen(js_name = transcan)]
pub fn transcan_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    categorical: &[u8],
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o: TranscanOpts = parse(options)?;
    let mut opts = TranscanOptions::default();
    if o.nk.is_some() {
        opts = opts.nk(o.nk);
    }
    if let Some(v) = o.iter_max {
        opts = opts.iter_max(v);
    }
    if let Some(v) = o.eps {
        opts = opts.eps(v);
    }
    if let Some(v) = o.curtail {
        opts = opts.curtail(v);
    }
    let cat: Vec<bool> = categorical.iter().map(|&c| c != 0).collect();
    Ok(transcan(&to_mat(x, n_rows, n_cols)?, &cat, &opts)
        .map_err(fit_error)?
        .to_js())
}
