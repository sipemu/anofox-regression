//! Data reduction before looking at `Y` (RMS §4.7, Ch 8): generalized
//! Spearman ρ² screening (`Hmisc::spearman2`), Hoeffding's D
//! (`Hmisc::hoeffd`), variable clustering on a similarity matrix
//! (`Hmisc::varclus`) and the hierarchical clustering behind it
//! (`stats::hclust`, a port of R's Fortran with its tie-breaking, merge and
//! order conventions), and redundancy analysis (`Hmisc::redun`).
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
    let p_value = FisherSnedecor::new(d1, df2).map_or(f64::NAN, |d| d.sf(f));
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

// =================================================================== redun ==

/// How a variable enters [`redun`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedunVariable {
    /// Numeric: a restricted cubic spline (or ranks, see [`RedunOptions`]).
    #[default]
    Numeric,
    /// Numeric but always linear (`I(x)` in the formula).
    Linear,
    /// Categorical (level codes): dummy columns.
    Categorical,
}

/// Options of [`redun`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedunOptions {
    /// A variable is redundant when its R² from the others exceeds this.
    pub r2: f64,
    /// Use the adjusted R² (`type = "adjusted"`).
    pub adjusted: bool,
    /// Spline knots for numeric variables (default 3).
    pub nk: usize,
    /// Predict only the linear part of a spline variable (`tlinear`,
    /// default `true`).
    pub tlinear: bool,
    /// `rank = TRUE`: as in Hmisc this sets `nk = 0`, so numeric variables
    /// enter linearly.
    pub rank: bool,
    /// `qrank = TRUE`: numeric variables enter as their (scaled) ranks.
    pub qrank: bool,
    /// For a categorical variable, also require every dummy to be predicted
    /// with R² above the cutoff (`allcat`).
    pub allcat: bool,
}

impl Default for RedunOptions {
    fn default() -> Self {
        Self {
            r2: 0.9,
            adjusted: false,
            nk: 3,
            tlinear: true,
            rank: false,
            qrank: false,
            allcat: false,
        }
    }
}

/// Result of [`redun`].
#[derive(Debug, Clone, PartialEq)]
pub struct Redun {
    /// Variables kept (column indices of `x`), in their original order.
    pub kept: Vec<usize>,
    /// Variables declared redundant, in the order removed.
    pub removed: Vec<usize>,
    /// R² of each removed variable when it was removed.
    pub r_squared: Vec<f64>,
    /// At each removal after the first, the R² of the previously removed
    /// variables from the variables still kept (`r2later`).
    pub r_squared_later: Vec<Vec<f64>>,
    /// R² of every analysed variable from all the others at the first step
    /// (`rsq1`).
    pub r_squared_first: Vec<f64>,
    /// Constant variables, left out of the analysis (`toofew`).
    pub too_few: Vec<usize>,
}

/// Internal variable type, as Hmisc's `vtype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VType {
    Linear,
    Spline,
    Rank,
    QuadRank,
    Categorical,
}

/// Expansion columns of one variable.
fn redun_columns(z: &[f64], vtype: VType, nk: usize) -> Result<Mat<f64>, RegressionError> {
    let n = z.len();
    Ok(match vtype {
        VType::Linear => Mat::from_fn(n, 1, |i, _| z[i]),
        VType::Rank | VType::QuadRank => {
            let r: Vec<f64> = average_ranks(z).iter().map(|v| v / n as f64).collect();
            let m = if vtype == VType::QuadRank { 2 } else { 1 };
            Mat::from_fn(n, m, |i, j| r[i].powi(j as i32 + 1))
        }
        VType::Spline => {
            crate::splines::Rcs::from_data(z, nk)
                .map_err(|e| invalid(&e.to_string()))?
                .basis(z, "z")
                .columns
        }
        VType::Categorical => {
            let mut levels = z.to_vec();
            levels.sort_by(f64::total_cmp);
            levels.dedup();
            Mat::from_fn(n, levels.len() - 1, |i, j| {
                f64::from(u8::from(z[i] == levels[j + 1]))
            })
        }
    })
}

struct RedunData {
    blocks: Vec<Mat<f64>>,
    vtypes: Vec<VType>,
    n: usize,
}

impl RedunData {
    /// Columns of the variables `ix` side by side.
    fn design(&self, ix: &[usize]) -> Mat<f64> {
        let cols: Vec<(usize, usize)> = ix
            .iter()
            .flat_map(|&v| (0..self.blocks[v].ncols()).map(move |c| (v, c)))
            .collect();
        Mat::from_fn(self.n, cols.len(), |i, k| {
            self.blocks[cols[k].0][(i, cols[k].1)]
        })
    }

    fn adjust(&self, r2: f64, dof: usize, adjusted: bool) -> f64 {
        if !adjusted {
            return r2;
        }
        let n = self.n as f64;
        (1.0 - (1.0 - r2) * (n - 1.0) / (n - dof as f64 - 1.0)).max(0.0)
    }

    /// Hmisc's `fcan`: R² of variable `iy` from the variables `ix`.
    fn r2(&self, ix: &[usize], iy: usize, o: &RedunOptions) -> Result<f64, RegressionError> {
        let x = self.design(ix);
        let vt = self.vtypes[iy];
        let linear_y = o.tlinear && matches!(vt, VType::Spline | VType::QuadRank);
        let yb = &self.blocks[iy];
        let y = if linear_y || vt == VType::Linear || vt == VType::Rank {
            Mat::from_fn(self.n, 1, |i, _| yb[(i, 0)])
        } else {
            yb.clone()
        };
        let (_, _, r2) = crate::transform::canonical(&x, &y)?;
        let mut r2 = self.adjust(r2, x.ncols() + y.ncols() - 1, o.adjusted);
        if r2 > o.r2 && o.allcat && vt == VType::Categorical && yb.ncols() > 1 {
            for j in 0..yb.ncols() {
                let yj = Mat::from_fn(self.n, 1, |i, _| yb[(i, j)]);
                let (_, _, r2c) = crate::transform::canonical(&x, &yj)?;
                r2 = r2.min(self.adjust(r2c, x.ncols(), o.adjusted));
            }
        }
        Ok(r2)
    }
}

/// Index of the first maximum.
fn argmax(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// Redundancy analysis (`Hmisc::redun`, RMS §4.7.1): repeatedly predict
/// every remaining variable from the others with flexible expansions
/// (splines with `nk` knots, dummies, or ranks) and remove the best
/// predicted one while its R² exceeds `options.r2` and every variable
/// removed earlier stays predictable from those kept.
pub fn redun(
    x: &Mat<f64>,
    kinds: &[RedunVariable],
    options: &RedunOptions,
) -> Result<Redun, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if kinds.len() != p {
        return Err(invalid("kinds needs one entry per column"));
    }
    let nk = if options.rank || options.qrank {
        0
    } else {
        options.nk
    };
    let (mut vars, mut blocks, mut vtypes, mut too_few) = (vec![], vec![], vec![], vec![]);
    for (j, &kind) in kinds.iter().enumerate() {
        let z: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
        let mut distinct = z.clone();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        if distinct.len() < 2 {
            too_few.push(j);
            continue;
        }
        let vt = match kind {
            RedunVariable::Categorical => VType::Categorical,
            _ if nk == 0 || distinct.len() < 3 || kind == RedunVariable::Linear => {
                if options.qrank {
                    VType::Rank
                } else {
                    VType::Linear
                }
            }
            _ if options.qrank => VType::QuadRank,
            _ => VType::Spline,
        };
        blocks.push(redun_columns(&z, vt, nk)?);
        vtypes.push(vt);
        vars.push(j);
    }
    let data = RedunData { blocks, vtypes, n };
    let mut kept: Vec<usize> = (0..vars.len()).collect();
    let (mut removed, mut r_squared, mut later_all) = (vec![], vec![], vec![]);
    let mut first = Vec::new();
    while kept.len() >= 2 {
        let rsq = kept
            .iter()
            .map(|&j| {
                let others: Vec<usize> = kept.iter().copied().filter(|&k| k != j).collect();
                data.r2(&others, j, options)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if first.is_empty() {
            first.clone_from(&rsq);
        }
        let best = argmax(&rsq);
        if rsq[best] < options.r2 {
            break;
        }
        let cand = kept[best];
        let rest: Vec<usize> = kept.iter().copied().filter(|&k| k != cand).collect();
        let later = removed
            .iter()
            .map(|&j| data.r2(&rest, j, options))
            .collect::<Result<Vec<_>, _>>()?;
        if later.iter().any(|&r| r < options.r2) {
            break;
        }
        removed.push(cand);
        r_squared.push(rsq[best]);
        if !later.is_empty() {
            later_all.push(later);
        }
        kept = rest;
    }
    Ok(Redun {
        kept: kept.iter().map(|&k| vars[k]).collect(),
        removed: removed.iter().map(|&k| vars[k]).collect(),
        r_squared,
        r_squared_later: later_all,
        r_squared_first: first,
        too_few,
    })
}

// ===================================================== principal components ==

/// Principal components of a data matrix ([`principal_components`],
/// `stats::prcomp`).
#[derive(Debug, Clone)]
pub struct PrincipalComponents {
    /// Column means subtracted.
    pub center: Vec<f64>,
    /// Column standard deviations divided by (`scale = TRUE`), if any.
    pub scale: Option<Vec<f64>>,
    /// Standard deviations of the components, decreasing.
    pub sdev: Vec<f64>,
    /// Loadings (`p × min(n, p)`), one column per component. Each column's
    /// largest absolute loading is positive (`prcomp`'s signs are those of
    /// LAPACK and arbitrary).
    pub rotation: Mat<f64>,
    /// Component scores of the rows (`n × min(n, p)`).
    pub scores: Mat<f64>,
}

impl PrincipalComponents {
    /// Proportion of the total variance carried by each component.
    pub fn proportion_of_variance(&self) -> Vec<f64> {
        let total: f64 = self.sdev.iter().map(|s| s * s).sum();
        self.sdev.iter().map(|s| s * s / total).collect()
    }

    /// Scores of new rows on the first `k` components (`predict.prcomp`).
    pub fn transform(&self, x: &Mat<f64>, k: usize) -> Mat<f64> {
        let p = self.center.len();
        let k = k.min(self.rotation.ncols());
        Mat::from_fn(x.nrows(), k, |i, c| {
            (0..p)
                .map(|j| {
                    let s = self.scale.as_ref().map_or(1.0, |s| s[j]);
                    (x[(i, j)] - self.center[j]) / s * self.rotation[(j, c)]
                })
                .sum()
        })
    }
}

/// Principal components of the columns of `x` (`prcomp(x, scale. =
/// scale)`): the SVD of the centered (and, with `scale`, standardised) data.
pub fn principal_components(
    x: &Mat<f64>,
    scale: bool,
) -> Result<PrincipalComponents, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if n < 2 || p == 0 {
        return Err(RegressionError::InsufficientObservations { needed: 2, got: n });
    }
    let center: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64)
        .collect();
    let sd: Vec<f64> = (0..p)
        .map(|j| {
            ((0..n).map(|i| (x[(i, j)] - center[j]).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        })
        .collect();
    if scale && sd.iter().any(|s| s.is_nan() || *s <= 0.0) {
        return Err(invalid("cannot scale a constant column"));
    }
    let z = Mat::from_fn(n, p, |i, j| {
        (x[(i, j)] - center[j]) / if scale { sd[j] } else { 1.0 }
    });
    let svd = z.thin_svd().map_err(|_| invalid("SVD failed"))?;
    let (s, v) = (svd.S().column_vector(), svd.V());
    let k = s.nrows();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    let mut rotation = Mat::<f64>::zeros(p, k);
    for (c, &o) in order.iter().enumerate() {
        let big = (0..p)
            .max_by(|&a, &b| v[(a, o)].abs().total_cmp(&v[(b, o)].abs()))
            .unwrap_or(0);
        let sign = if v[(big, o)] < 0.0 { -1.0 } else { 1.0 };
        for j in 0..p {
            rotation[(j, c)] = sign * v[(j, o)];
        }
    }
    let scores = &z * &rotation;
    Ok(PrincipalComponents {
        center,
        scale: scale.then_some(sd),
        sdev: order
            .iter()
            .map(|&o| s[o] / ((n - 1) as f64).sqrt())
            .collect(),
        rotation,
        scores,
    })
}

/// Summary scores of variable clusters (RMS §4.7, Ch 8): for every cluster
/// (column indices of `x`, e.g. from [`Hclust::cut_height`] on a
/// [`varclus`]), the first `k` principal components of its standardised
/// variables (fewer if the cluster has fewer variables), side by side in
/// cluster order.
pub fn cluster_scores(
    x: &Mat<f64>,
    clusters: &[Vec<usize>],
    k: usize,
) -> Result<Mat<f64>, RegressionError> {
    let n = x.nrows();
    let mut blocks = Vec::with_capacity(clusters.len());
    for cl in clusters {
        if cl.is_empty() || cl.iter().any(|&j| j >= x.ncols()) {
            return Err(invalid("clusters must be non-empty and name columns of x"));
        }
        let sub = Mat::from_fn(n, cl.len(), |i, j| x[(i, cl[j])]);
        let pc = principal_components(&sub, true)?;
        let m = k.min(pc.scores.ncols());
        blocks.push(Mat::from_fn(n, m, |i, c| pc.scores[(i, c)]));
    }
    let total: usize = blocks.iter().map(|b| b.ncols()).sum();
    let mut out = Mat::<f64>::zeros(n, total);
    let mut c0 = 0;
    for b in &blocks {
        for c in 0..b.ncols() {
            for i in 0..n {
                out[(i, c0 + c)] = b[(i, c)];
            }
        }
        c0 += b.ncols();
    }
    Ok(out)
}

// ================================================================= transcan ==

/// Options of [`transcan`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranscanOptions {
    /// Spline knots; `None` is Hmisc's default (3, 4 or 5 for `n < 30`,
    /// `< 100`, `≥ 100`).
    pub nk: Option<usize>,
    /// Maximum iterations (`iter.max`, default 50).
    pub iter_max: usize,
    /// Convergence: largest change of a transformation (`eps`, default 0.1).
    pub eps: f64,
    /// Keep transformed values of missing entries within the observed range
    /// (`curtail`, default `true`).
    pub curtail: bool,
}

impl Default for TranscanOptions {
    fn default() -> Self {
        Self {
            nk: None,
            iter_max: 50,
            eps: 0.1,
            curtail: true,
        }
    }
}

/// Result of [`transcan`].
#[derive(Debug, Clone)]
pub struct Transcan {
    /// Transformed variables (`n × p`); missing entries hold the transformed
    /// value predicted from the other variables. The sign of each
    /// transformation is arbitrary (as in Hmisc).
    pub transformed: Mat<f64>,
    /// R² of each transformed variable from the others (`rsq`).
    pub r_squared: Vec<f64>,
    /// Adjusted R² (`rsq.adj`).
    pub r_squared_adj: Vec<f64>,
    /// Iterations used.
    pub iterations: usize,
}

/// `Hmisc::transcan(x, method = "canonical")` (RMS §4.7.3, §8.4): optimal
/// transformations of every variable (restricted cubic splines, or dummies
/// for `categorical` columns) so that each is best predicted from the
/// current transformations of all others, found by alternating canonical
/// correlations until the transformations change by less than `eps`. Use
/// the transformations as predictors in place of the originals, or reduce
/// them further with [`principal_components`] (`transcan`-based scoring).
/// Missing values (NaN) start at the median (the mode for categorical
/// variables) and are then predicted on the transformed scale.
pub fn transcan(
    x: &Mat<f64>,
    categorical: &[bool],
    options: &TranscanOptions,
) -> Result<Transcan, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if categorical.len() != p || p < 2 || n < 3 {
        return Err(invalid(
            "transcan needs two or more columns and one flag per column",
        ));
    }
    let nk = options.nk.unwrap_or(if n < 30 {
        3
    } else if n < 100 {
        4
    } else {
        5
    });
    let missing: Vec<Vec<bool>> = (0..p)
        .map(|j| (0..n).map(|i| x[(i, j)].is_nan()).collect())
        .collect();
    let (mut bases, mut nparm, mut fillin) = (Vec::new(), Vec::new(), Vec::new());
    let mut xt = x.to_owned();
    for j in 0..p {
        let obs: Vec<f64> = (0..n)
            .filter(|&i| !missing[j][i])
            .map(|i| x[(i, j)])
            .collect();
        if obs.len() < 3 {
            return Err(invalid("every column needs three or more observed values"));
        }
        let col: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
        if categorical[j] {
            let levels = distinct_sorted(&obs);
            let count = |l: f64| obs.iter().filter(|&&v| v == l).count();
            let mode =
                levels
                    .iter()
                    .copied()
                    .fold(levels[0], |m, l| if count(l) > count(m) { l } else { m });
            bases.push(Mat::from_fn(n, levels.len() - 1, |i, c| {
                f64::from(u8::from(col[i] == levels[c + 1]))
            }));
            nparm.push(levels.len() - 1);
            fillin.push(mode);
            for i in (0..n).filter(|&i| !missing[j][i]) {
                xt[(i, j)] = f64::from(u8::from(col[i] == mode));
            }
        } else {
            let rcs =
                crate::splines::Rcs::from_data(&obs, nk).map_err(|e| invalid(&e.to_string()))?;
            bases.push(rcs.basis(&col, "x").columns);
            nparm.push(rcs.n_columns());
            let mut s = obs.clone();
            s.sort_by(f64::total_cmp);
            fillin.push(crate::diagnostics::accuracy::quantile7(&s, 0.5));
        }
    }
    let varconst = ((n - 1) as f64).sqrt();
    let (mut r2s, mut r2a) = (vec![0.0; p], vec![0.0; p]);
    let mut last = false;
    let mut xtl = xt.clone();
    let mut iterations = 0;
    let mut dmax: f64 = 0.0;
    for iter in 1..=options.iter_max {
        iterations = iter;
        dmax = 0.0;
        if last {
            xtl = xt.clone();
        }
        for i in 0..p {
            let others: Vec<usize> = (0..p).filter(|&c| c != i).collect();
            let rows: Vec<usize> = (0..n).filter(|&r| !missing[i][r]).collect();
            let nn = rows.len();
            let xx = |r: usize, c: usize| -> f64 {
                let v = xt[(r, others[c])];
                if v.is_nan() {
                    fillin[others[c]]
                } else {
                    v
                }
            };
            let xs = Mat::from_fn(nn, p - 1, |r, c| xx(rows[r], c));
            let ys = Mat::from_fn(nn, bases[i].ncols(), |r, c| bases[i][(rows[r], c)]);
            let (xb, yb, r2) = crate::transform::canonical(&xs, &ys)?;
            let adj = varconst / ((nn - 1) as f64).sqrt();
            let affine = |coef: &Col<f64>, v: &dyn Fn(usize) -> f64, m: usize| {
                coef[0] * adj + (0..m).map(|c| coef[c + 1] * adj * v(c)).sum::<f64>()
            };
            let mut newy: Vec<f64> = (0..n)
                .map(|r| {
                    if missing[i][r] {
                        affine(&xb, &|c| xx(r, c), p - 1)
                    } else {
                        affine(&yb, &|c| bases[i][(r, c)], bases[i].ncols())
                    }
                })
                .collect();
            let k_other = others
                .iter()
                .map(|&c| nparm[c].saturating_sub(1))
                .sum::<usize>() as f64
                / (p - 1) as f64
                + (p - 1) as f64;
            let k = (nparm[i] - 1) as f64 + k_other;
            r2s[i] = r2;
            r2a[i] = (1.0 - (1.0 - r2) * (nn as f64 - 1.0) / (nn as f64 - k - 1.0)).max(0.0);
            if options.curtail && nn < n {
                let lo = rows.iter().map(|&r| newy[r]).fold(f64::INFINITY, f64::min);
                let hi = rows
                    .iter()
                    .map(|&r| newy[r])
                    .fold(f64::NEG_INFINITY, f64::max);
                for r in (0..n).filter(|&r| missing[i][r]) {
                    newy[r] = newy[r].clamp(lo, hi);
                }
            }
            if iter > 1 {
                let d = |s: f64| {
                    (0..n)
                        .map(|r| (s * xt[(r, i)] - newy[r]).abs())
                        .fold(0.0, f64::max)
                };
                dmax = dmax.max(d(1.0).min(d(-1.0)));
            }
            for (r, v) in newy.into_iter().enumerate() {
                if last {
                    xtl[(r, i)] = v;
                } else {
                    xt[(r, i)] = v;
                }
            }
        }
        if last {
            break;
        }
        last = iter == options.iter_max - 1 || (iter > 1 && dmax < options.eps);
    }
    if options.iter_max > 3 && iterations == options.iter_max && dmax >= options.eps {
        return Err(RegressionError::ConvergenceFailed { iterations });
    }
    Ok(Transcan {
        transformed: xtl,
        r_squared: r2s,
        r_squared_adj: r2a,
        iterations,
    })
}

fn distinct_sorted(v: &[f64]) -> Vec<f64> {
    let mut d = v.to_vec();
    d.sort_by(f64::total_cmp);
    d.dedup();
    d
}
