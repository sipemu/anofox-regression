//! Data reduction before looking at `Y` (RMS §4.7, Ch 8): generalized
//! Spearman ρ² screening (`Hmisc::spearman2`), Hoeffding's D
//! (`Hmisc::hoeffd`), variable clustering on a similarity matrix
//! (`Hmisc::varclus`) and the hierarchical clustering behind it
//! (`stats::hclust`, a port of R's Fortran with its tie-breaking, merge and
//! order conventions).
//!
//! ```rust,ignore
//! use anofox_regression::reduction::{varclus, Linkage, Similarity, SimilarityTransform};
//!
//! let vc = varclus(&x, Similarity::Spearman, SimilarityTransform::Square, Linkage::Complete)?;
//! // variables merged below 1 − ρ² = 0.3 are candidates for one summary score
//! let groups = vc.hclust.cut_height(0.3);
//! ```

use crate::solvers::RegressionError;
use crate::validation::average_ranks;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

fn invalid(msg: &str) -> RegressionError {
    RegressionError::NumericalError(msg.to_string())
}

// ================================================================ spearman2 ==

/// Result of [`spearman2`] (`Hmisc::spearman2`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spearman2 {
    /// Generalized Spearman ρ²: R² of `rank(y)` on the rank terms of `x`.
    pub rho2: f64,
    /// F statistic of `rho2`.
    pub f: f64,
    /// Numerator d.f. (`p`, or the number of levels − 1).
    pub df1: f64,
    /// Denominator d.f. `n − df1 − 1`.
    pub df2: f64,
    /// P-value of `F`.
    pub p_value: f64,
    /// Number of observations.
    pub n: usize,
    /// `1 − (1 − ρ²)(n − 1)/df2`.
    pub adjusted_rho2: f64,
}

/// R² of `y` on `[1, Z]` by least squares.
fn r_squared(z: &Mat<f64>, y: &[f64]) -> Result<f64, RegressionError> {
    let n = y.len();
    let design = Mat::from_fn(
        n,
        z.ncols() + 1,
        |i, j| {
            if j == 0 {
                1.0
            } else {
                z[(i, j - 1)]
            }
        },
    );
    let q = design.qr().compute_thin_Q();
    let yc = Col::from_fn(n, |i| y[i]);
    let fit = &q * (q.transpose() * &yc);
    let mean = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let sse: f64 = (0..n).map(|i| (y[i] - fit[i]).powi(2)).sum();
    if sst <= 0.0 {
        return Err(invalid("y is constant"));
    }
    Ok(1.0 - sse / sst)
}

fn spearman2_stats(rho2: f64, df1: usize, n: usize) -> Spearman2 {
    let df2 = (n - df1 - 1) as f64;
    let d1 = df1 as f64;
    let f = rho2 / d1 / ((1.0 - rho2) / df2);
    let p_value = FisherSnedecor::new(d1, df2).map_or(f64::NAN, |d| 1.0 - d.cdf(f));
    Spearman2 {
        rho2,
        f,
        df1: d1,
        df2,
        p_value,
        n,
        adjusted_rho2: 1.0 - (1.0 - rho2) * (n - 1) as f64 / df2,
    }
}

fn check_pair(x_len: usize, y: &[f64]) -> Result<(), RegressionError> {
    if x_len != y.len() {
        return Err(RegressionError::DimensionMismatch {
            x_rows: x_len,
            y_len: y.len(),
        });
    }
    if y.len() < 3 {
        return Err(RegressionError::InsufficientObservations {
            needed: 3,
            got: y.len(),
        });
    }
    Ok(())
}

/// Generalized Spearman ρ² of a numeric `x` with `y` (`spearman2(x, y,
/// p)`): with `p = 1` the squared Spearman correlation, with `p = 2` the R²
/// of `rank(y)` on `rank(x)` and its square (allowing non-monotone
/// association). `x` with fewer than three distinct values uses `p = 1`.
pub fn spearman2(x: &[f64], y: &[f64], p: usize) -> Result<Spearman2, RegressionError> {
    check_pair(x.len(), y)?;
    if !(1..=2).contains(&p) {
        return Err(invalid("p must be 1 or 2"));
    }
    let mut distinct = x.to_vec();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let p = if distinct.len() < 3 { 1 } else { p };
    let (rx, ry) = (average_ranks(x), average_ranks(y));
    let z = Mat::from_fn(x.len(), p, |i, j| rx[i].powi(j as i32 + 1));
    Ok(spearman2_stats(r_squared(&z, &ry)?, p, x.len()))
}

/// Generalized Spearman ρ² of a categorical `x` (level codes) with `y`
/// (`spearman2(factor(x), y)`): the R² of `rank(y)` on the level dummies.
pub fn spearman2_categorical(x: &[usize], y: &[f64]) -> Result<Spearman2, RegressionError> {
    check_pair(x.len(), y)?;
    let mut levels = x.to_vec();
    levels.sort_unstable();
    levels.dedup();
    if levels.len() < 2 {
        return Err(invalid("x needs two or more levels"));
    }
    let z = Mat::from_fn(x.len(), levels.len() - 1, |i, j| {
        f64::from(u8::from(x[i] == levels[j + 1]))
    });
    Ok(spearman2_stats(
        r_squared(&z, &average_ranks(y))?,
        levels.len() - 1,
        x.len(),
    ))
}

// ============================================================ Hoeffding's D ==

/// Hoeffding's D matrix ([`hoeffding_d`], `Hmisc::hoeffd`).
#[derive(Debug, Clone)]
pub struct HoeffdingD {
    /// `30 ×` Hoeffding's D for every pair of columns (1 on the diagonal),
    /// in `[−0.5, 1]`; large values mean dependence of any form.
    pub d: Mat<f64>,
    /// Asymptotic P-values (Blum, Kiefer and Rosenblatt); NaN on the
    /// diagonal.
    pub p_value: Mat<f64>,
}

/// Hoeffding's D (unscaled) of one pair, as Hmisc's `hoeff`.
fn hoeffding_pair(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let half = |a: f64, b: f64| -> f64 {
        if a == b {
            0.5
        } else if a < b {
            1.0
        } else {
            0.0
        }
    };
    let (mut q, mut r, mut s) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (mut rx, mut ry, mut rj) = (1.0, 1.0, 1.0);
        for j in (0..n).filter(|&j| j != i) {
            let (cx, cy) = (half(x[j], x[i]), half(y[j], y[i]));
            rx += cx;
            ry += cy;
            rj += cx * cy;
        }
        q += (rx - 1.0) * (rx - 2.0) * (ry - 1.0) * (ry - 2.0);
        r += (rx - 2.0) * (ry - 2.0) * (rj - 1.0);
        s += (rj - 1.0) * (rj - 2.0);
    }
    let z = n as f64;
    (q - 2.0 * (z - 2.0) * r + (z - 2.0) * (z - 3.0) * s)
        / (z * (z - 1.0) * (z - 2.0) * (z - 3.0) * (z - 4.0))
}

/// Hmisc's `phoeffd`: the P-value of an unscaled D on `n` pairs.
fn hoeffding_p(d: f64, n: f64) -> f64 {
    const TAB: [f64; 86] = [
        5297., 4918., 4565., 4236., 3930., 3648., 3387., 3146., 2924., 2719., 2530., 2355., 2194.,
        2045., 1908., 1781., 1663., 1554., 1453., 1359., 1273., 1192., 1117., 1047., 982., 921.,
        864., 812., 762., 716., 673., 633., 595., 560., 527., 496., 467., 440., 414., 390., 368.,
        347., 327., 308., 291., 274., 259., 244., 230., 217., 205., 194., 183., 173., 163., 154.,
        145., 137., 130., 123., 116., 110., 104., 98., 93., 87., 83., 78., 74., 70., 66., 63., 59.,
        56., 53., 50., 47., 45., 42., 25., 14., 8., 5., 3., 2., 1.,
    ];
    let z = 0.5 * std::f64::consts::PI.powi(4) * n * (d + 1.0 / 36.0 / n);
    if !(1.1..=8.5).contains(&z) {
        return (0.388_503_7 - 1.164_879 * z).exp().clamp(1e-8, 1.0);
    }
    // abscissae seq(1.1, 5, 0.05) then seq(5.5, 8.5, 0.5)
    let at = |k: usize| -> f64 {
        if k < 79 {
            1.1 + 0.05 * k as f64
        } else {
            5.5 + 0.5 * (k - 79) as f64
        }
    };
    let k = (0..TAB.len() - 1)
        .find(|&k| z <= at(k + 1))
        .unwrap_or(TAB.len() - 2);
    let (z0, z1) = (at(k), at(k + 1));
    (TAB[k] + (z - z0) * (TAB[k + 1] - TAB[k]) / (z1 - z0)) / 10000.0
}

/// Hoeffding's D between every pair of columns of `x` (`hoeffd(x)`): a
/// rank measure of dependence that also detects non-monotone association.
/// Needs five or more rows.
pub fn hoeffding_d(x: &Mat<f64>) -> Result<HoeffdingD, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if n < 5 {
        return Err(RegressionError::InsufficientObservations { needed: 5, got: n });
    }
    let cols: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| x[(i, j)]).collect())
        .collect();
    let mut d = Mat::<f64>::identity(p, p);
    let mut pv = Mat::from_fn(p, p, |_, _| f64::NAN);
    for i in 0..p {
        for j in i + 1..p {
            let raw = hoeffding_pair(&cols[i], &cols[j]);
            let pval = hoeffding_p(raw, n as f64);
            d[(i, j)] = 30.0 * raw;
            d[(j, i)] = 30.0 * raw;
            pv[(i, j)] = pval;
            pv[(j, i)] = pval;
        }
    }
    Ok(HoeffdingD { d, p_value: pv })
}

// ================================================================== hclust ==

/// Agglomeration method of [`hclust`] (`method =`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Linkage {
    /// `"ward.D"` (Lance–Williams Ward on the dissimilarities as given).
    WardD,
    /// `"single"`.
    Single,
    /// `"complete"` (default).
    #[default]
    Complete,
    /// `"average"` (UPGMA).
    Average,
    /// `"mcquitty"` (WPGMA).
    Mcquitty,
    /// `"median"` (WPGMC).
    Median,
    /// `"centroid"` (UPGMC).
    Centroid,
    /// `"ward.D2"` (Ward on squared dissimilarities).
    WardD2,
}

/// One side of a merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Node {
    /// An original object (0-based; R's `-(i + 1)`).
    Leaf(usize),
    /// The cluster formed at merge step `k` (0-based; R's `k + 1`).
    Cluster(usize),
}

/// A hierarchical clustering ([`hclust`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Hclust {
    /// The two nodes joined at each of the `n − 1` steps, in R's order
    /// (`merge`).
    pub merge: Vec<(Node, Node)>,
    /// Dissimilarity at each merge (`height`).
    pub height: Vec<f64>,
    /// Objects in dendrogram order (`order`, 0-based).
    pub order: Vec<usize>,
}

impl Hclust {
    /// Cluster label (0-based, numbered by first object) of every object
    /// after cutting the tree at `height` (merges at or below it are kept).
    pub fn cut_height(&self, height: f64) -> Vec<usize> {
        let n = self.merge.len() + 1;
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], i: usize) -> usize {
            let mut r = i;
            while p[r] != r {
                r = p[r];
            }
            p[i] = r;
            r
        }
        // a representative object of every merged cluster
        let mut rep = vec![0usize; self.merge.len()];
        for (k, &(a, b)) in self.merge.iter().enumerate() {
            let obj = |nd: Node| match nd {
                Node::Leaf(i) => i,
                Node::Cluster(c) => rep[c],
            };
            let (ra, rb) = (obj(a), obj(b));
            rep[k] = ra;
            if self.height[k] <= height {
                let (x, y) = (find(&mut parent, ra), find(&mut parent, rb));
                parent[y] = x;
            }
        }
        let mut label = vec![usize::MAX; n];
        let mut next = 0;
        (0..n)
            .map(|i| {
                let r = find(&mut parent, i);
                if label[r] == usize::MAX {
                    label[r] = next;
                    next += 1;
                }
                label[r]
            })
            .collect()
    }
}

/// Lance–Williams update of the dissimilarity between the merged cluster
/// `i ∪ j` and `k`, as R's Fortran.
fn lance_williams(linkage: Linkage, dik: f64, djk: f64, dij: f64, m: [f64; 3]) -> f64 {
    let [mi, mj, mk] = m;
    match linkage {
        Linkage::WardD | Linkage::WardD2 => {
            ((mi + mk) * dik + (mj + mk) * djk - mk * dij) / (mi + mj + mk)
        }
        Linkage::Single => dik.min(djk),
        Linkage::Complete => dik.max(djk),
        Linkage::Average => (mi * dik + mj * djk) / (mi + mj),
        Linkage::Mcquitty => (dik + djk) / 2.0,
        Linkage::Median => ((dik + djk) - dij / 2.0) / 2.0,
        Linkage::Centroid => (mi * dik + mj * djk - mi * mj * dij / (mi + mj)) / (mi + mj),
    }
}

/// Nearest later neighbour of `i` among active objects.
fn nearest(diss: &Mat<f64>, active: &[bool], i: usize) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for j in i + 1..active.len() {
        if active[j] && diss[(i, j)] < best.1 {
            best = (j, diss[(i, j)]);
        }
    }
    best
}

/// Agglomerative clustering of the symmetric dissimilarity matrix `diss`
/// (`hclust(as.dist(diss), method)`), with R's nearest-neighbour search,
/// tie-breaking and merge / order conventions.
pub fn hclust(diss: &Mat<f64>, linkage: Linkage) -> Result<Hclust, RegressionError> {
    let n = diss.nrows();
    if diss.ncols() != n || n < 2 {
        return Err(invalid(
            "hclust needs a square matrix of two or more objects",
        ));
    }
    if (0..n).any(|i| (0..n).any(|j| !diss[(i, j)].is_finite())) {
        return Err(invalid("dissimilarities must be finite"));
    }
    let mut d = if linkage == Linkage::WardD2 {
        Mat::from_fn(n, n, |i, j| diss[(i, j)].powi(2))
    } else {
        diss.to_owned()
    };
    let mut active = vec![true; n];
    let mut members = vec![1.0; n];
    let mut nn = vec![0usize; n];
    let mut disnn = vec![f64::INFINITY; n];
    for i in 0..n - 1 {
        (nn[i], disnn[i]) = nearest(&d, &active, i);
    }
    let mut pairs = Vec::with_capacity(n - 1);
    let mut crit = Vec::with_capacity(n - 1);
    for _ in 0..n - 1 {
        let (mut im, mut dmin) = (0, f64::INFINITY);
        for i in 0..n - 1 {
            if active[i] && disnn[i] < dmin {
                (im, dmin) = (i, disnn[i]);
            }
        }
        let (i2, j2) = (im.min(nn[im]), im.max(nn[im]));
        pairs.push((i2, j2));
        crit.push(if linkage == Linkage::WardD2 {
            dmin.sqrt()
        } else {
            dmin
        });
        active[j2] = false;
        let dij = d[(i2, j2)];
        let (mut best, mut jj) = (f64::INFINITY, 0);
        for k in (0..n).filter(|&k| active[k] && k != i2) {
            let v = lance_williams(
                linkage,
                d[(i2, k)],
                d[(j2, k)],
                dij,
                [members[i2], members[j2], members[k]],
            );
            d[(i2, k)] = v;
            d[(k, i2)] = v;
            if i2 < k {
                if v < best {
                    (best, jj) = (v, k);
                }
            } else if v < disnn[k] {
                (disnn[k], nn[k]) = (v, i2);
            }
        }
        members[i2] += members[j2];
        (disnn[i2], nn[i2]) = (best, jj);
        for i in 0..n - 1 {
            if active[i] && (nn[i] == i2 || nn[i] == j2) {
                (nn[i], disnn[i]) = nearest(&d, &active, i);
            }
        }
    }
    let (merge, order) = hcass2(&pairs);
    Ok(Hclust {
        merge,
        height: crit,
        order,
    })
}

/// R's `hcass2`: merge pairs of object indices → R's merge convention
/// (singletons before clusters, smaller first) and the dendrogram order.
fn hcass2(pairs: &[(usize, usize)]) -> (Vec<(Node, Node)>, Vec<usize>) {
    let n = pairs.len() + 1;
    // signed 1-based: −(object + 1) for leaves, step + 1 for clusters
    let mut ia: Vec<i64> = pairs.iter().map(|p| -(p.0 as i64 + 1)).collect();
    let mut ib: Vec<i64> = pairs.iter().map(|p| -(p.1 as i64 + 1)).collect();
    for i in 0..n.saturating_sub(2) {
        let k = pairs[i].0.min(pairs[i].1);
        for j in i + 1..n - 1 {
            if pairs[j].0 == k {
                ia[j] = i as i64 + 1;
            }
            if pairs[j].1 == k {
                ib[j] = i as i64 + 1;
            }
        }
    }
    for i in 0..n - 1 {
        if ia[i] > 0 && ib[i] < 0 {
            std::mem::swap(&mut ia[i], &mut ib[i]);
        }
        if ia[i] > 0 && ib[i] > 0 && ia[i] > ib[i] {
            std::mem::swap(&mut ia[i], &mut ib[i]);
        }
    }
    let mut order: Vec<i64> = vec![ia[n - 2], ib[n - 2]];
    for i in (1..n - 1).rev() {
        if let Some(j) = order.iter().position(|&o| o == i as i64) {
            order[j] = ia[i - 1];
            order.insert(j + 1, ib[i - 1]);
        }
    }
    let node = |v: i64| {
        if v < 0 {
            Node::Leaf((-v - 1) as usize)
        } else {
            Node::Cluster(v as usize - 1)
        }
    };
    let merge = ia
        .iter()
        .zip(&ib)
        .map(|(&a, &b)| (node(a), node(b)))
        .collect();
    let order = order.iter().map(|&o| (-o - 1) as usize).collect();
    (merge, order)
}

// ================================================================= varclus ==

/// Similarity measure of [`varclus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Similarity {
    /// Spearman rank correlation (default).
    #[default]
    Spearman,
    /// Pearson correlation.
    Pearson,
    /// `30 ×` Hoeffding's D (not transformed).
    Hoeffding,
}

/// Transformation of a correlation in [`varclus`] (`trans =`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimilarityTransform {
    /// `r²` (default).
    #[default]
    Square,
    /// `|r|`.
    Abs,
    /// `r`.
    None,
}

/// Result of [`varclus`].
#[derive(Debug, Clone)]
pub struct Varclus {
    /// The similarity matrix.
    pub sim: Mat<f64>,
    /// Hierarchical clustering of `1 − sim`.
    pub hclust: Hclust,
}

fn pearson_matrix(cols: &[Vec<f64>]) -> Mat<f64> {
    let p = cols.len();
    let centered: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| {
            let m = c.iter().sum::<f64>() / c.len() as f64;
            c.iter().map(|v| v - m).collect()
        })
        .collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
    Mat::from_fn(p, p, |i, j| {
        dot(&centered[i], &centered[j])
            / (dot(&centered[i], &centered[i]) * dot(&centered[j], &centered[j])).sqrt()
    })
}

/// Variable clustering (`varclus(x, similarity, trans, method)`): the
/// similarity of every pair of columns of `x`, clustered hierarchically on
/// `1 − similarity`.
pub fn varclus(
    x: &Mat<f64>,
    similarity: Similarity,
    transform: SimilarityTransform,
    linkage: Linkage,
) -> Result<Varclus, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if p < 2 {
        return Err(invalid("varclus needs two or more variables"));
    }
    let cols: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| x[(i, j)]).collect())
        .collect();
    let sim = match similarity {
        Similarity::Hoeffding => hoeffding_d(x)?.d,
        _ => {
            let r = match similarity {
                Similarity::Spearman => {
                    pearson_matrix(&cols.iter().map(|c| average_ranks(c)).collect::<Vec<_>>())
                }
                _ => pearson_matrix(&cols),
            };
            Mat::from_fn(p, p, |i, j| match transform {
                SimilarityTransform::Square => r[(i, j)].powi(2),
                SimilarityTransform::Abs => r[(i, j)].abs(),
                SimilarityTransform::None => r[(i, j)],
            })
        }
    };
    if (0..p).any(|i| (0..p).any(|j| !sim[(i, j)].is_finite())) {
        return Err(invalid(
            "part of the similarity matrix could not be computed",
        ));
    }
    let diss = Mat::from_fn(p, p, |i, j| 1.0 - sim[(i, j)]);
    Ok(Varclus {
        hclust: hclust(&diss, linkage)?,
        sim,
    })
}
