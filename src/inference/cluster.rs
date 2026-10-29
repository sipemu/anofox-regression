//! Cluster-robust (Huber–White) sandwich covariance (RMS §9.5).
//!
//! `V = A (Σ_g s_g s_gᵀ) A` with `A` the inverse information (bread) and
//! `s_g` the sum of the score contributions of the observations in cluster
//! `g`, as `sandwich::vcovCL` and `rms::robcov` compute it:
//!
//! * [`cluster_sandwich`] – from any bread and per-observation scores (any
//!   model), with one or more clustering dimensions;
//! * [`lm_cluster_vcov`] – linear models (OLS / WLS), including the
//!   leverage-adjusted HC2 (Bell–McCaffrey) and HC3 variants;
//! * [`glm_cluster_vcov`] – GLMs of any [`GlmFamily`] from the fitted means;
//! * `FittedOls::cluster_vcov`, `FittedWls::cluster_vcov` and
//!   `FittedAft::cluster_vcov` on the fits.
//!
//! Small-sample corrections follow `vcovCL(type, cadjust)`
//! ([`ClusterOptions`]): `cadjust` multiplies every cluster dimension's meat
//! by `G/(G − 1)`, [`ClusterType::HC1`] the total by `(n − 1)/(n − k)`. So
//! [`ClusterOptions::cr1`] is Stata's CR1, `G/(G−1)·(n−1)/(n−k)`, and
//! [`ClusterOptions::robcov`] is `rms::robcov` / `survreg(cluster =)` (no
//! correction). With several dimensions the meat is combined by
//! inclusion–exclusion (Cameron, Gelbach & Miller), e.g.
//! `V₁ + V₂ − V₁∩₂` for two-way clustering.
//!
//! Cluster labels are integer codes per dimension; see
//! [`crate::solvers::cv::group_codes`] to code arbitrary keys.

use crate::core::GlmFamily;
use faer::linalg::solvers::Solve;
use faer::{Col, Mat};
use std::collections::BTreeMap;
use thiserror::Error;

/// Variant of the clustered meat (`type` of `sandwich::vcovCL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClusterType {
    /// No small-sample correction of the meat.
    HC0,
    /// Multiply by `(n − 1)/(n − k)` (`k` = number of parameters).
    #[default]
    HC1,
    /// Bias-reduced linearization (Bell–McCaffrey, CR2): residuals of
    /// cluster `g` premultiplied by `(I − H_gg)^{−1/2}`, then
    /// `√((G − 1)/G)`. Linear models only.
    HC2,
    /// Jackknife-like: `(I − H_gg)^{−1}`, then `√((G − 1)/G)`. Linear
    /// models only.
    HC3,
}

/// Small-sample corrections of a cluster-robust covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterOptions {
    /// Meat variant.
    pub kind: ClusterType,
    /// Multiply each dimension's meat by `G/(G − 1)`.
    pub cadjust: bool,
}

impl Default for ClusterOptions {
    /// `vcovCL`'s default for linear models: HC1 with the cluster
    /// adjustment (Stata's CR1).
    fn default() -> Self {
        Self::cr1()
    }
}

impl ClusterOptions {
    /// Stata's CR1: `G/(G−1)·(n−1)/(n−k)`.
    pub fn cr1() -> Self {
        Self {
            kind: ClusterType::HC1,
            cadjust: true,
        }
    }
    /// `G/(G−1)` only (`vcovCL`'s default for GLMs).
    pub fn hc0() -> Self {
        Self {
            kind: ClusterType::HC0,
            cadjust: true,
        }
    }
    /// No correction, as `rms::robcov` and `survreg(cluster =)`.
    pub fn robcov() -> Self {
        Self {
            kind: ClusterType::HC0,
            cadjust: false,
        }
    }
}

/// Errors of the cluster-robust covariance.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ClusterError {
    /// Inconsistent dimensions.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    /// No clustering dimension given.
    #[error("no cluster variable")]
    NoClusters,
    /// The information matrix is singular.
    #[error("the information matrix is singular")]
    Singular,
    /// The fit keeps no residuals (e.g. a fit from sufficient statistics).
    #[error("the fit has no residuals for the rows of x")]
    NoResiduals,
    /// The variant is not available for this model.
    #[error("{0}")]
    Unsupported(&'static str),
}

/// Combined cluster codes of every non-empty subset of the dimensions,
/// with the inclusion–exclusion sign and the number of clusters.
fn cluster_terms(
    clusters: &[&[usize]],
    n: usize,
) -> Result<Vec<(Vec<usize>, f64, usize)>, ClusterError> {
    let d = clusters.len();
    if d == 0 {
        return Err(ClusterError::NoClusters);
    }
    if let Some(c) = clusters.iter().find(|c| c.len() != n) {
        return Err(ClusterError::Dimension(format!(
            "cluster has {} entries, expected {n}",
            c.len()
        )));
    }
    let mut out = Vec::new();
    for mask in 1..(1usize << d) {
        let dims: Vec<usize> = (0..d).filter(|&k| mask >> k & 1 == 1).collect();
        let mut codes: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
        let ids: Vec<Vec<usize>> = (0..n)
            .map(|i| dims.iter().map(|&k| clusters[k][i]).collect())
            .collect();
        for id in &ids {
            let next = codes.len();
            codes.entry(id.clone()).or_insert(next);
        }
        let g = codes.len();
        let code: Vec<usize> = ids.iter().map(|id| codes[id]).collect();
        let sign = if dims.len() % 2 == 1 { 1.0 } else { -1.0 };
        out.push((code, sign, g));
    }
    // Order as sandwich does (single dimensions first); the sum is
    // order-independent.
    out.sort_by_key(|t| t.0.len());
    Ok(out)
}

/// `Σ_g s_g s_gᵀ` of per-cluster score sums.
fn clustered_crossprod(ef: &Mat<f64>, code: &[usize], g: usize) -> Mat<f64> {
    let k = ef.ncols();
    let mut sums = Mat::<f64>::zeros(g, k);
    for i in 0..ef.nrows() {
        for j in 0..k {
            sums[(code[i], j)] += ef[(i, j)];
        }
    }
    sums.transpose() * &sums
}

/// Cluster-robust covariance from the bread `A` (the inverse information,
/// `k × k`) and the per-observation score contributions (`n × k`, the
/// estimating functions at the estimate).
///
/// `clusters` holds one code vector per clustering dimension. Only
/// [`ClusterType::HC0`] / [`ClusterType::HC1`] are available here; the
/// leverage-based variants need a linear model ([`lm_cluster_vcov`]).
pub fn cluster_sandwich(
    bread: &Mat<f64>,
    scores: &Mat<f64>,
    clusters: &[&[usize]],
    options: &ClusterOptions,
) -> Result<Mat<f64>, ClusterError> {
    if matches!(options.kind, ClusterType::HC2 | ClusterType::HC3) {
        return Err(ClusterError::Unsupported(
            "HC2 / HC3 need a linear model; use lm_cluster_vcov",
        ));
    }
    meat_and_sandwich(bread, scores, clusters, options, |_, ef| Ok(ef.clone()))
}

/// Shared assembly: `adjust(term, scores)` may modify the scores per
/// clustering term (HC2 / HC3).
fn meat_and_sandwich(
    bread: &Mat<f64>,
    scores: &Mat<f64>,
    clusters: &[&[usize]],
    options: &ClusterOptions,
    adjust: impl Fn(&(Vec<usize>, f64, usize), &Mat<f64>) -> Result<Mat<f64>, ClusterError>,
) -> Result<Mat<f64>, ClusterError> {
    let (n, k) = (scores.nrows(), scores.ncols());
    if bread.nrows() != k || bread.ncols() != k {
        return Err(ClusterError::Dimension(format!(
            "bread is {}x{}, scores have {k} columns",
            bread.nrows(),
            bread.ncols()
        )));
    }
    let mut meat = Mat::<f64>::zeros(k, k);
    for term in cluster_terms(clusters, n)? {
        let (code, sign, g) = &term;
        let adj = if options.cadjust {
            *g as f64 / (*g as f64 - 1.0)
        } else {
            1.0
        };
        let ef = adjust(&term, scores)?;
        let m = clustered_crossprod(&ef, code, *g);
        for a in 0..k {
            for b in 0..k {
                meat[(a, b)] += sign * adj * m[(a, b)];
            }
        }
    }
    if options.kind == ClusterType::HC1 {
        let f = (n as f64 - 1.0) / (n as f64 - k as f64);
        for a in 0..k {
            for b in 0..k {
                meat[(a, b)] *= f;
            }
        }
    }
    Ok(bread * &meat * bread)
}

/// Design with a leading intercept column (if any) and without the
/// excluded columns.
fn reduced_design(x: &Mat<f64>, with_intercept: bool, excluded: &[bool]) -> Mat<f64> {
    let kept: Vec<usize> = (0..x.ncols()).filter(|&j| !excluded[j]).collect();
    let off = usize::from(with_intercept);
    Mat::from_fn(x.nrows(), kept.len() + off, |i, j| {
        if j < off {
            1.0
        } else {
            x[(i, kept[j - off])]
        }
    })
}

/// `(Xᵀ W X)⁻¹`.
fn inverse_gram(x: &Mat<f64>, w: &[f64]) -> Result<Mat<f64>, ClusterError> {
    let k = x.ncols();
    let xw = Mat::from_fn(x.nrows(), k, |i, j| x[(i, j)] * w[i]);
    let g = x.transpose() * &xw;
    let llt = g
        .llt(faer::Side::Lower)
        .map_err(|_| ClusterError::Singular)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// Expand a reduced covariance (intercept first, kept columns) to the full
/// `[intercept,] p` layout with zero rows / columns for excluded columns.
fn expand(reduced: &Mat<f64>, with_intercept: bool, excluded: &[bool]) -> Mat<f64> {
    crate::inference::expand_reduced_factor(reduced, excluded, with_intercept)
}

/// Cluster-robust covariance of a linear model (OLS when `weights` is
/// `None`, WLS otherwise), as `sandwich::vcovCL(lm(...), cluster, type,
/// cadjust)`.
///
/// `x` is the design without intercept column, `residuals = y − ŷ`, and
/// `aliased` marks columns that are not part of the fit. The result is
/// `[intercept,] p` square (intercept first) with zero rows and columns for
/// aliased columns.
#[allow(clippy::too_many_arguments)]
pub fn lm_cluster_vcov(
    x: &Mat<f64>,
    residuals: &Col<f64>,
    weights: Option<&Col<f64>>,
    with_intercept: bool,
    aliased: &[bool],
    clusters: &[&[usize]],
    options: &ClusterOptions,
) -> Result<Mat<f64>, ClusterError> {
    let n = x.nrows();
    if residuals.nrows() != n {
        return Err(ClusterError::NoResiduals);
    }
    if aliased.len() != x.ncols() {
        return Err(ClusterError::Dimension(
            "aliased must have one entry per column".into(),
        ));
    }
    let w: Vec<f64> = (0..n).map(|i| weights.map_or(1.0, |w| w[i])).collect();
    let xr = reduced_design(x, with_intercept, aliased);
    let k = xr.ncols();
    let bread = inverse_gram(&xr, &w)?;
    // Working-weighted residuals w·e and scores w·e·x.
    let res: Vec<f64> = (0..n).map(|i| w[i] * residuals[i]).collect();
    let scores = Mat::from_fn(n, k, |i, j| res[i] * xr[(i, j)]);

    let leverage_adjust = |term: &(Vec<usize>, f64, usize), ef: &Mat<f64>| {
        let (code, _, g) = term;
        match options.kind {
            ClusterType::HC0 | ClusterType::HC1 => Ok(ef.clone()),
            ClusterType::HC2 | ClusterType::HC3 => {
                let power = if options.kind == ClusterType::HC2 {
                    -0.5
                } else {
                    -1.0
                };
                let mut out = Mat::<f64>::zeros(n, k);
                let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (i, &c) in code.iter().enumerate() {
                    members.entry(c).or_default().push(i);
                }
                for rows in members.values() {
                    let adj = cluster_residual_adjustment(&xr, &bread, &w, &res, rows, power)?;
                    for (a, &i) in rows.iter().enumerate() {
                        for j in 0..k {
                            out[(i, j)] = adj[a] * xr[(i, j)];
                        }
                    }
                }
                let s = ((*g as f64 - 1.0) / *g as f64).sqrt();
                Ok(Mat::from_fn(n, k, |i, j| s * out[(i, j)]))
            }
        }
    };
    let reduced = meat_and_sandwich(&bread, &scores, clusters, options, leverage_adjust)?;
    Ok(expand(&reduced, with_intercept, aliased))
}

/// `(I − H_gg)^p · res_g` for the rows of one cluster, with
/// `H_gg = X_g A X_gᵀ W_g`. Computed through the symmetric
/// `S = W_g^{½} X_g A X_gᵀ W_g^{½}`:
/// `(I − H_gg)^p = W_g^{−½} (I − S)^p W_g^{½}`.
fn cluster_residual_adjustment(
    x: &Mat<f64>,
    bread: &Mat<f64>,
    w: &[f64],
    res: &[f64],
    rows: &[usize],
    power: f64,
) -> Result<Vec<f64>, ClusterError> {
    let m = rows.len();
    let k = x.ncols();
    if rows.iter().any(|&i| w[i] <= 0.0) {
        return Err(ClusterError::Unsupported("HC2 / HC3 need positive weights"));
    }
    let sw: Vec<f64> = rows.iter().map(|&i| w[i].sqrt()).collect();
    let xg = Mat::from_fn(m, k, |a, j| x[(rows[a], j)] * sw[a]);
    let s = &xg * bread * xg.transpose();
    let ims = Mat::from_fn(m, m, |a, b| f64::from(u8::from(a == b)) - s[(a, b)]);
    let evd = ims
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| ClusterError::Singular)?;
    let (u, d) = (evd.U(), evd.S().column_vector());
    if (0..m).any(|i| d[i] <= 0.0) {
        return Err(ClusterError::Unsupported(
            "I − H_gg is not positive definite (a cluster has leverage 1)",
        ));
    }
    // v = W^{½} res_g; out = W^{−½} U D^p Uᵀ v.
    let v = Col::from_fn(m, |a| sw[a] * res[rows[a]]);
    let utv = u.transpose() * &v;
    let scaled = Col::from_fn(m, |a| d[a].powf(power) * utv[a]);
    let out = u * &scaled;
    Ok((0..m).map(|a| out[a] / sw[a]).collect())
}

/// Cluster-robust covariance of a GLM, as `sandwich::vcovCL(glm(...))`.
///
/// `mu` are the fitted means; the scores are `pᵢ (yᵢ − μᵢ) / (V(μᵢ) g′(μᵢ)) xᵢ`
/// and the bread `(Xᵀ W X)⁻¹` with working weights
/// `wᵢ = pᵢ / (V(μᵢ) g′(μᵢ)²)` (`pᵢ` prior weights). The dispersion cancels.
/// Only HC0 / HC1 are available. The result is `[intercept,] p` square.
#[allow(clippy::too_many_arguments)]
pub fn glm_cluster_vcov<F: GlmFamily + ?Sized>(
    family: &F,
    x: &Mat<f64>,
    with_intercept: bool,
    y: &[f64],
    mu: &[f64],
    prior_weights: Option<&[f64]>,
    clusters: &[&[usize]],
    options: &ClusterOptions,
) -> Result<Mat<f64>, ClusterError> {
    let n = x.nrows();
    if y.len() != n || mu.len() != n {
        return Err(ClusterError::Dimension(format!(
            "x has {n} rows, y {} and mu {}",
            y.len(),
            mu.len()
        )));
    }
    if matches!(options.kind, ClusterType::HC2 | ClusterType::HC3) {
        return Err(ClusterError::Unsupported(
            "HC2 / HC3 are available for linear models only",
        ));
    }
    let excluded = vec![false; x.ncols()];
    let xr = reduced_design(x, with_intercept, &excluded);
    let k = xr.ncols();
    let p = |i: usize| prior_weights.map_or(1.0, |w| w[i]);
    let mut ww = vec![0.0; n];
    let mut sc = vec![0.0; n];
    for i in 0..n {
        let v = family.variance(mu[i]);
        let gp = family.link_derivative(mu[i]);
        ww[i] = p(i) / (v * gp * gp);
        sc[i] = p(i) * (y[i] - mu[i]) / (v * gp);
    }
    let bread = inverse_gram(&xr, &ww)?;
    let scores = Mat::from_fn(n, k, |i, j| sc[i] * xr[(i, j)]);
    cluster_sandwich(&bread, &scores, clusters, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_clusters_are_hc0() {
        // With every observation its own cluster and no adjustment the
        // clustered sandwich is White's HC0.
        let x = Mat::from_fn(6, 1, |i, _| i as f64);
        let e = Col::from_fn(6, |i| [0.5, -1.0, 0.3, 0.8, -0.2, -0.4][i]);
        let codes: Vec<usize> = (0..6).collect();
        let v = lm_cluster_vcov(
            &x,
            &e,
            None,
            true,
            &[false],
            &[&codes],
            &ClusterOptions::robcov(),
        )
        .unwrap();
        let hc0 = crate::inference::compute_hc_standard_errors(
            &x,
            &e,
            &[false],
            true,
            crate::inference::HcType::HC0,
        )
        .unwrap();
        assert!((v[(1, 1)].sqrt() - hc0.std_errors[0]).abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let b = Mat::<f64>::identity(2, 2);
        let s = Mat::<f64>::zeros(3, 2);
        let c = [0usize, 1, 1];
        let o = ClusterOptions::default();
        assert_eq!(
            cluster_sandwich(&b, &s, &[], &o),
            Err(ClusterError::NoClusters)
        );
        assert!(cluster_sandwich(&b, &s, &[&c[..2]], &o).is_err());
        assert!(cluster_sandwich(&Mat::identity(3, 3), &s, &[&c], &o).is_err());
        let hc2 = ClusterOptions {
            kind: ClusterType::HC2,
            cadjust: true,
        };
        assert!(cluster_sandwich(&b, &s, &[&c], &hc2).is_err());
    }
}
