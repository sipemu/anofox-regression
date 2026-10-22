//! Cross-validation splits for the regularisation paths: group, stratified and
//! time-series fold generators, explicit train / test splits, the CV curve on
//! arbitrary splits and nested cross-validation (sipemu/anofox-regression#87).
//!
//! # Fold generators
//!
//! Each generator returns one fold id per row (`0..k`) and is deterministic:
//!
//! - [`group_kfold_ids`]: a group never spans two folds. [`GroupBalance::Rows`]
//!   is scikit-learn's `GroupKFold` (largest group first into the fold with the
//!   fewest rows); [`GroupBalance::Random`] is `GroupKFold(shuffle=True)`
//!   (a seeded permutation of the groups cut into `k` consecutive chunks).
//! - [`logo_ids`]: leave-one-group-out, one fold per distinct group.
//! - [`stratified_kfold_ids`]: scikit-learn's `StratifiedKFold`; every stratum's
//!   count per fold differs by at most one.
//! - [`quantile_strata`]: quantile bins of a continuous response, to stratify a
//!   regression CV.
//! - [`stratified_group_kfold_ids`]: scikit-learn's `StratifiedGroupKFold`.
//!
//! Group and stratum keys only need `Ord`, so a key made of several variables
//! (store × region) is a tuple. Keys are numbered in sorted order, as
//! `numpy.unique` does, so without a seed the folds equal scikit-learn's. With
//! a seed the shuffles use the crate's own generator and differ from
//! scikit-learn's `random_state`.
//!
//! # Splits
//!
//! A [`CvSplit`] is an explicit pair of training and test rows. Fold ids become
//! splits with [`splits_from_fold_ids`] (train = every other fold);
//! [`time_series_splits`] gives rolling-origin splits (scikit-learn's
//! `TimeSeriesSplit` with `test_size`, `gap` and `max_train_size`), where the
//! training rows precede the test rows and not every row is tested.
//! [`cv_path_splits`] computes the `cv.glmnet` curve on any splits and
//! [`nested_cv_path`] selects `λ` by an inner CV on each outer training set.
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::solvers::cv::{
//!     cv_path_splits, group_kfold_ids, splits_from_fold_ids, CvMetric, GroupBalance,
//! };
//! use anofox_regression::solvers::path::{lambda_grid, PathOptions};
//! use faer::{Col, Mat};
//!
//! let x = Mat::from_fn(60, 2, |i, j| ((i * (j + 3)) % 11) as f64);
//! let y = Col::from_fn(60, |i| 1.0 + x[(i, 0)] - 0.5 * x[(i, 1)] + (i % 4) as f64);
//! // Twelve stores of five rows each; a store is never split across folds.
//! let store: Vec<usize> = (0..60).map(|i| i / 5).collect();
//! let folds = group_kfold_ids(&store, 4, GroupBalance::Rows, 0).unwrap();
//!
//! let opts = PathOptions::ridge();
//! let lambdas = lambda_grid(&x, &y, &opts, 20, None).unwrap();
//! let cv = cv_path_splits(&x, &y, &lambdas, &splits_from_fold_ids(&folds), &opts, CvMetric::Mse)
//!     .unwrap();
//! assert_eq!(cv.n_folds, 4);
//! ```

use crate::solvers::path::{
    fit_path, kfold_ids, lambda_grid, largest_lambda_where, validate_lambdas, validate_xy, CvPath,
    PathOptions, SplitMix64,
};
use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};

fn invalid(msg: impl Into<String>) -> RegressionError {
    RegressionError::NumericalError(msg.into())
}

/// Dense codes `0..m` of the distinct keys in sorted key order (`numpy.unique`'s
/// `return_inverse`), and the number of distinct keys.
///
/// Codes keep the order of the keys, so they serve as time ranks for
/// [`group_time_series_splits`] and as group / stratum labels for
/// [`InnerFolds`].
pub fn group_codes<G: Ord>(keys: &[G]) -> (Vec<usize>, usize) {
    let mut uniq: Vec<&G> = keys.iter().collect();
    uniq.sort();
    uniq.dedup();
    let codes = keys
        .iter()
        .map(|k| uniq.binary_search(&k).expect("key is present"))
        .collect();
    (codes, uniq.len())
}

/// Seeded Fisher-Yates permutation in place.
fn shuffle<T>(v: &mut [T], rng: &mut SplitMix64) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
}

fn check_k(k: usize, n: usize, what: &str) -> Result<(), RegressionError> {
    if k < 2 {
        return Err(invalid(format!("{what}: k must be >= 2, got {k}")));
    }
    if k > n {
        return Err(invalid(format!(
            "{what}: cannot have k = {k} folds for {n} rows"
        )));
    }
    Ok(())
}

/// How [`group_kfold_ids`] assigns groups to folds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupBalance {
    /// Balance the folds by row count: groups are taken from the largest to
    /// the smallest (ties: the larger key first) and each goes into the fold
    /// with the fewest rows so far (the first such fold). This is
    /// scikit-learn's `GroupKFold()`; the seed is not used.
    #[default]
    Rows,
    /// Shuffle the groups with the seed and cut them into `k` consecutive
    /// chunks whose sizes differ by at most one group
    /// (`GroupKFold(shuffle=True)`, `numpy.array_split`). The folds have about
    /// the same number of groups, not of rows.
    Random,
}

/// Group k-fold: fold ids `0..k` such that all rows of a group share a fold.
///
/// Needs at least `k` distinct groups. See [`GroupBalance`] for the
/// assignment; `seed` is used by [`GroupBalance::Random`] only.
pub fn group_kfold_ids<G: Ord>(
    groups: &[G],
    k: usize,
    balance: GroupBalance,
    seed: u64,
) -> Result<Vec<usize>, RegressionError> {
    let (codes, m) = group_codes(groups);
    if k < 2 {
        return Err(invalid(format!("group k-fold: k must be >= 2, got {k}")));
    }
    if k > m {
        return Err(invalid(format!(
            "group k-fold: cannot have k = {k} folds with {m} groups"
        )));
    }
    let mut group_fold = vec![0usize; m];
    match balance {
        GroupBalance::Rows => {
            let mut size = vec![0usize; m];
            for &c in &codes {
                size[c] += 1;
            }
            // numpy's stable argsort, reversed: by size descending, ties by
            // the larger code first.
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&a, &b| size[b].cmp(&size[a]).then(b.cmp(&a)));
            let mut fold_rows = vec![0usize; k];
            for g in order {
                let mut lightest = 0;
                for f in 1..k {
                    if fold_rows[f] < fold_rows[lightest] {
                        lightest = f;
                    }
                }
                fold_rows[lightest] += size[g];
                group_fold[g] = lightest;
            }
        }
        GroupBalance::Random => {
            let mut perm: Vec<usize> = (0..m).collect();
            shuffle(&mut perm, &mut SplitMix64::new(seed));
            let (base, extra) = (m / k, m % k);
            let mut pos = 0;
            for f in 0..k {
                let len = base + usize::from(f < extra);
                for &g in &perm[pos..pos + len] {
                    group_fold[g] = f;
                }
                pos += len;
            }
        }
    }
    Ok(codes.iter().map(|&c| group_fold[c]).collect())
}

/// Leave-one-group-out: the fold of a row is the rank of its group among the
/// distinct groups (scikit-learn's `LeaveOneGroupOut` order).
pub fn logo_ids<G: Ord>(groups: &[G]) -> Vec<usize> {
    group_codes(groups).0
}

/// Stratified k-fold (scikit-learn's `StratifiedKFold`).
///
/// Strata are numbered by first appearance; the number of rows of each
/// stratum per fold follows a round robin over the rows sorted by stratum, so
/// each stratum's count per fold differs by at most one. Within a stratum the
/// rows get their folds in blocks, in row order (`seed = None`, exactly
/// `StratifiedKFold(shuffle=False)`), or in a seeded random order
/// (`Some(seed)`). Errors when `k` exceeds the number of rows or every
/// stratum has fewer than `k` rows.
pub fn stratified_kfold_ids<S: Ord>(
    strata: &[S],
    k: usize,
    seed: Option<u64>,
) -> Result<Vec<usize>, RegressionError> {
    let n = strata.len();
    check_k(k, n, "stratified k-fold")?;
    let (sorted_codes, m) = group_codes(strata);
    // Re-number by first appearance (scikit-learn's class_perm).
    let mut first = vec![usize::MAX; m];
    let mut next = 0;
    for &c in &sorted_codes {
        if first[c] == usize::MAX {
            first[c] = next;
            next += 1;
        }
    }
    let enc: Vec<usize> = sorted_codes.iter().map(|&c| first[c]).collect();
    let mut counts = vec![0usize; m];
    for &c in &enc {
        counts[c] += 1;
    }
    if counts.iter().all(|&c| c < k) {
        return Err(invalid(format!(
            "stratified k-fold: k = {k} is larger than the number of rows of every stratum"
        )));
    }
    // allocation[c][f]: rows of stratum c in fold f, from a round robin over
    // the rows sorted by stratum.
    let mut allocation = vec![vec![0usize; k]; m];
    let mut pos = 0;
    for (alloc, &cnt) in allocation.iter_mut().zip(&counts) {
        for r in pos..pos + cnt {
            alloc[r % k] += 1;
        }
        pos += cnt;
    }
    let mut rng = seed.map(SplitMix64::new);
    let mut out = vec![0usize; n];
    for (c, alloc) in allocation.iter().enumerate() {
        let mut folds: Vec<usize> = alloc
            .iter()
            .enumerate()
            .flat_map(|(f, &cnt)| std::iter::repeat_n(f, cnt))
            .collect();
        if let Some(rng) = rng.as_mut() {
            shuffle(&mut folds, rng);
        }
        let mut it = folds.into_iter();
        for (i, &e) in enc.iter().enumerate() {
            if e == c {
                out[i] = it.next().expect("allocation covers the stratum");
            }
        }
    }
    Ok(out)
}

/// Quantile strata of a continuous response: `n_bins` bins of (about) equal
/// size by rank, so that [`stratified_kfold_ids`] spreads small and large
/// responses over the folds. Tied values share a bin (the bin of the first of
/// them in sorted order). NaN values sort last.
pub fn quantile_strata(y: &Col<f64>, n_bins: usize) -> Vec<usize> {
    let n = y.nrows();
    let n_bins = n_bins.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
    let mut out = vec![0usize; n];
    let mut prev_bin = 0;
    for (r, &i) in order.iter().enumerate() {
        let tied = r > 0 && y[order[r - 1]].total_cmp(&y[i]).is_eq();
        let bin = if tied { prev_bin } else { r * n_bins / n };
        out[i] = bin;
        prev_bin = bin;
    }
    out
}

/// Stratified group k-fold (scikit-learn's `StratifiedGroupKFold`): groups
/// stay whole and the strata are balanced over the folds.
///
/// Groups are taken in decreasing order of the standard deviation of their
/// stratum counts (stable, so ties keep the group order: sorted keys, or a
/// seeded permutation with `Some(seed)`), and each goes into the fold that
/// minimises the mean over strata of the standard deviation across folds of
/// the stratum's share, ties going to the fold with fewer rows. With
/// `seed = None` the folds equal `StratifiedGroupKFold(shuffle=False)`.
pub fn stratified_group_kfold_ids<G: Ord, S: Ord>(
    groups: &[G],
    strata: &[S],
    k: usize,
    seed: Option<u64>,
) -> Result<Vec<usize>, RegressionError> {
    let n = groups.len();
    if strata.len() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: strata.len(),
        });
    }
    check_k(k, n, "stratified group k-fold")?;
    let (g_codes, n_groups) = group_codes(groups);
    let (s_codes, n_strata) = group_codes(strata);
    if k > n_groups {
        return Err(invalid(format!(
            "stratified group k-fold: cannot have k = {k} folds with {n_groups} groups"
        )));
    }
    let mut s_cnt = vec![0.0f64; n_strata];
    let mut per_group = vec![vec![0.0f64; n_strata]; n_groups];
    for (&g, &s) in g_codes.iter().zip(&s_codes) {
        per_group[g][s] += 1.0;
        s_cnt[s] += 1.0;
    }
    if s_cnt.iter().all(|&c| c < k as f64) {
        return Err(invalid(format!(
            "stratified group k-fold: k = {k} is larger than the number of rows of every stratum"
        )));
    }
    let mut perm: Vec<usize> = (0..n_groups).collect();
    if let Some(seed) = seed {
        shuffle(&mut perm, &mut SplitMix64::new(seed));
    }
    let std = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let spread: Vec<f64> = perm.iter().map(|&g| std(&per_group[g])).collect();
    let mut order: Vec<usize> = (0..n_groups).collect();
    order.sort_by(|&a, &b| spread[b].total_cmp(&spread[a]));

    let mut per_fold = vec![vec![0.0f64; n_strata]; k];
    let mut group_fold = vec![0usize; n_groups];
    let mut col = vec![0.0f64; k];
    for pos in order {
        let g = perm[pos];
        let counts = &per_group[g];
        let mut best = 0;
        let mut min_eval = f64::INFINITY;
        let mut min_rows = f64::INFINITY;
        for f in 0..k {
            let mut eval = 0.0;
            for s in 0..n_strata {
                for (h, c) in col.iter_mut().enumerate() {
                    let add = if h == f { counts[s] } else { 0.0 };
                    *c = (per_fold[h][s] + add) / s_cnt[s];
                }
                eval += std(&col);
            }
            eval /= n_strata as f64;
            let rows: f64 = per_fold[f].iter().sum();
            // numpy.isclose(eval, min_eval) with its default tolerances.
            let close = (eval - min_eval).abs() <= 1e-8 + 1e-5 * min_eval.abs();
            if eval < min_eval || (close && rows < min_rows) {
                min_eval = eval;
                min_rows = rows;
                best = f;
            }
        }
        for (a, c) in per_fold[best].iter_mut().zip(counts) {
            *a += c;
        }
        group_fold[g] = best;
    }
    Ok(g_codes.iter().map(|&g| group_fold[g]).collect())
}

/// An explicit cross-validation split: the rows to fit on and the rows to
/// predict, as row indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CvSplit {
    /// Training rows (increasing).
    pub train: Vec<usize>,
    /// Test rows (increasing).
    pub test: Vec<usize>,
}

/// One split per distinct fold id, in increasing id order: the fold's rows
/// are the test set, every other row is the training set. This is the
/// partition [`cv_path`](crate::solvers::path::cv_path) uses.
pub fn splits_from_fold_ids(fold_ids: &[usize]) -> Vec<CvSplit> {
    let mut labels: Vec<usize> = fold_ids.to_vec();
    labels.sort_unstable();
    labels.dedup();
    labels
        .iter()
        .map(|&label| {
            let (test, train): (Vec<usize>, Vec<usize>) =
                (0..fold_ids.len()).partition(|&i| fold_ids[i] == label);
            CvSplit { train, test }
        })
        .collect()
}

/// The training window of [`time_series_splits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainWindow {
    /// Every row before the gap (scikit-learn's default).
    #[default]
    Expanding,
    /// At most this many rows right before the gap (`max_train_size`).
    Sliding(usize),
}

/// Options of [`time_series_splits`] (scikit-learn's `TimeSeriesSplit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSeriesSplitOptions {
    /// Number of splits (test blocks), at least 2. Default 5.
    pub n_splits: usize,
    /// Rows per test block; default `n / (n_splits + 1)`.
    pub test_size: Option<usize>,
    /// Rows left out between the training rows and each test block
    /// (an embargo against leakage through autocorrelation). Default 0.
    pub gap: usize,
    /// Expanding or sliding training window. Default expanding.
    pub window: TrainWindow,
    /// Smallest admissible training set: an error is returned when the first
    /// split has fewer training rows. Default none.
    pub min_train_size: Option<usize>,
}

impl Default for TimeSeriesSplitOptions {
    fn default() -> Self {
        Self {
            n_splits: 5,
            test_size: None,
            gap: 0,
            window: TrainWindow::Expanding,
            min_train_size: None,
        }
    }
}

impl TimeSeriesSplitOptions {
    /// Options with `n_splits` splits and the other defaults.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Self::default()
        }
    }
}

/// Rolling-origin splits over `n` rows already in time order, exactly
/// scikit-learn's `TimeSeriesSplit(n_splits, test_size, gap, max_train_size)`:
/// the last `n_splits · test_size` rows form `n_splits` consecutive test
/// blocks, and each block is predicted from the rows before it, minus `gap`
/// rows (and only the last `max_train_size` of them for a sliding window).
pub fn time_series_splits(
    n: usize,
    opts: &TimeSeriesSplitOptions,
) -> Result<Vec<CvSplit>, RegressionError> {
    let n_splits = opts.n_splits;
    if n_splits < 2 {
        return Err(invalid(format!(
            "time-series split: n_splits must be >= 2, got {n_splits}"
        )));
    }
    if n_splits + 1 > n {
        return Err(invalid(format!(
            "time-series split: cannot have {} folds for {n} rows",
            n_splits + 1
        )));
    }
    let test_size = opts.test_size.unwrap_or(n / (n_splits + 1));
    if test_size == 0 {
        return Err(invalid("time-series split: test_size must be >= 1"));
    }
    if let TrainWindow::Sliding(0) = opts.window {
        return Err(invalid(
            "time-series split: a sliding window needs max_train_size >= 1",
        ));
    }
    let needed = opts.gap + test_size * n_splits;
    if needed >= n {
        return Err(invalid(format!(
            "time-series split: too many splits ({n_splits}) for {n} rows with test_size \
             {test_size} and gap {}",
            opts.gap
        )));
    }
    let splits: Vec<CvSplit> = (0..n_splits)
        .map(|s| {
            let test_start = n - (n_splits - s) * test_size;
            let train_end = test_start - opts.gap;
            let train_start = match opts.window {
                TrainWindow::Sliding(m) if m < train_end => train_end - m,
                _ => 0,
            };
            CvSplit {
                train: (train_start..train_end).collect(),
                test: (test_start..test_start + test_size).collect(),
            }
        })
        .collect();
    if let Some(min) = opts.min_train_size {
        let first = splits[0].train.len();
        if first < min {
            return Err(invalid(format!(
                "time-series split: the first split has {first} training rows, fewer than \
                 min_train_size {min}"
            )));
        }
    }
    Ok(splits)
}

/// Rolling-origin splits over the distinct values of a time key, for panel
/// data where several rows (series) share a time point: the splits of
/// [`time_series_splits`] over the sorted distinct keys, with every row of a
/// time point on the same side. `test_size`, `gap` and the window count time
/// points, not rows.
pub fn group_time_series_splits<T: Ord>(
    time_keys: &[T],
    opts: &TimeSeriesSplitOptions,
) -> Result<Vec<CvSplit>, RegressionError> {
    let (codes, m) = group_codes(time_keys);
    let splits = time_series_splits(m, opts)?;
    Ok(splits
        .into_iter()
        .map(|s| {
            let mut side = vec![0u8; m];
            s.train.iter().for_each(|&t| side[t] = 1);
            s.test.iter().for_each(|&t| side[t] = 2);
            CvSplit {
                train: (0..codes.len()).filter(|&i| side[codes[i]] == 1).collect(),
                test: (0..codes.len()).filter(|&i| side[codes[i]] == 2).collect(),
            }
        })
        .collect())
}

/// Prediction error measured on each test set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CvMetric {
    /// Mean squared error (`cv.glmnet`'s `type.measure = "mse"`; for the
    /// Gaussian family also its `"deviance"`).
    #[default]
    Mse,
    /// Mean absolute error (`type.measure = "mae"`).
    Mae,
}

impl CvMetric {
    fn loss(self, r: f64) -> f64 {
        match self {
            CvMetric::Mse => r * r,
            CvMetric::Mae => r.abs(),
        }
    }
}

fn validate_splits(splits: &[CvSplit], n: usize) -> Result<(), RegressionError> {
    if splits.len() < 2 {
        return Err(invalid(format!(
            "cross-validation needs at least 2 splits, got {}",
            splits.len()
        )));
    }
    let mut mark = vec![usize::MAX; n];
    for (s, split) in splits.iter().enumerate() {
        if split.test.is_empty() {
            return Err(invalid(format!("split {s} has no test rows")));
        }
        if split.train.len() < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: split.train.len(),
            });
        }
        for &i in split.train.iter().chain(&split.test) {
            if i >= n {
                return Err(invalid(format!(
                    "split {s}: row {i} out of range for {n} rows"
                )));
            }
        }
        for &i in &split.train {
            mark[i] = s;
        }
        if split.test.iter().any(|&i| mark[i] == s) {
            return Err(invalid(format!(
                "split {s}: a row is both a training and a test row"
            )));
        }
    }
    Ok(())
}

fn subset(x: &Mat<f64>, y: &Col<f64>, rows: &[usize]) -> (Mat<f64>, Col<f64>) {
    (
        Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]),
        Col::from_fn(rows.len(), |i| y[rows[i]]),
    )
}

/// Cross-validation of the path at `lambdas` on explicit splits.
///
/// For each split the path is fitted on the training rows (centred and
/// standardised on their own) with the same `lambdas` and options, and the
/// metric computed on the test rows. The aggregates are `cv.glmnet`'s
/// (`grouped = TRUE`): `cv_mean` is the split metrics averaged with the test
/// sizes as weights, `cv_se = sqrt(weighted.mean((m_s − cv_mean)², n_s) /
/// (S − 1))` over the `S` splits, and `lambda_min` / `lambda_1se` as in
/// [`cv_path`](crate::solvers::path::cv_path). Rows outside every test set
/// (time-series splits) do not count. `n_folds` is the number of splits and
/// `path` the fit on all rows.
///
/// Needs at least 2 splits, each with a test row and 2 training rows and no
/// row on both sides.
pub fn cv_path_splits(
    x: &Mat<f64>,
    y: &Col<f64>,
    lambdas: &[f64],
    splits: &[CvSplit],
    opts: &PathOptions,
    metric: CvMetric,
) -> Result<CvPath, RegressionError> {
    validate_xy(x, y)?;
    opts.validate()?;
    validate_lambdas(lambdas)?;
    validate_splits(splits, x.nrows())?;
    let nl = lambdas.len();
    let k = splits.len();
    let full = fit_path(x, y, lambdas, opts)?;

    let mut fold_err = vec![vec![0.0; nl]; k];
    let mut fold_n = vec![0.0; k];
    for (f, split) in splits.iter().enumerate() {
        let (xt, yt) = subset(x, y, &split.train);
        let fit = fit_path(&xt, &yt, lambdas, opts)?;
        let (xv, _) = subset(x, y, &split.test);
        let pred = fit.predict(&xv);
        for (l, err) in fold_err[f].iter_mut().enumerate() {
            let total: f64 = split
                .test
                .iter()
                .enumerate()
                .map(|(i, &row)| metric.loss(y[row] - pred[(i, l)]))
                .sum();
            *err = total / split.test.len() as f64;
        }
        fold_n[f] = split.test.len() as f64;
    }
    let n_tot: f64 = fold_n.iter().sum();
    let mut cv_mean = vec![0.0; nl];
    let mut cv_se = vec![0.0; nl];
    for l in 0..nl {
        let m: f64 = (0..k).map(|f| fold_n[f] * fold_err[f][l]).sum::<f64>() / n_tot;
        let var: f64 = (0..k)
            .map(|f| fold_n[f] * (fold_err[f][l] - m).powi(2))
            .sum::<f64>()
            / n_tot;
        cv_mean[l] = m;
        cv_se[l] = (var / (k as f64 - 1.0)).sqrt();
    }
    let cv_min = cv_mean.iter().copied().fold(f64::INFINITY, f64::min);
    let index_min = largest_lambda_where(lambdas, |l| cv_mean[l] <= cv_min);
    let bound = cv_mean[index_min] + cv_se[index_min];
    let index_1se = largest_lambda_where(lambdas, |l| cv_mean[l] <= bound);
    Ok(CvPath {
        lambdas: lambdas.to_vec(),
        cv_mean,
        cv_se,
        df: full.df.clone(),
        lambda_min: lambdas[index_min],
        lambda_1se: lambdas[index_1se],
        index_min,
        index_1se,
        n_folds: k,
        path: full,
    })
}

/// How the inner folds of [`nested_cv_path`] are built from each outer
/// training set. Key vectors (`groups`, `strata`, `time_keys`) have one entry
/// per row of the full data (for instance from [`group_codes`]); only the
/// entries of the outer training rows are used.
#[derive(Debug, Clone, PartialEq)]
pub enum InnerFolds {
    /// Seeded random k-fold ([`kfold_ids`]) over the training rows in order.
    Kfold { k: usize, seed: u64 },
    /// Group k-fold ([`group_kfold_ids`]): a group stays in one inner fold.
    Group {
        groups: Vec<usize>,
        k: usize,
        balance: GroupBalance,
        seed: u64,
    },
    /// Leave-one-group-out over the groups of the training rows.
    Logo { groups: Vec<usize> },
    /// Stratified k-fold ([`stratified_kfold_ids`]).
    Stratified {
        strata: Vec<usize>,
        k: usize,
        seed: Option<u64>,
    },
    /// Rolling-origin splits of the training rows: over the rows in order
    /// (`time_keys = None`, [`time_series_splits`]) or over the distinct time
    /// keys ([`group_time_series_splits`]).
    TimeSeries {
        opts: TimeSeriesSplitOptions,
        time_keys: Option<Vec<usize>>,
    },
}

/// Which `λ` of the inner CV curve [`nested_cv_path`] refits with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LambdaRule {
    /// `lambda_min`.
    #[default]
    Min,
    /// `lambda_1se`.
    OneSe,
}

/// Options of [`nested_cv_path`].
#[derive(Debug, Clone, PartialEq)]
pub struct NestedCvOptions {
    /// Inner folds, built from each outer training set only.
    pub inner: InnerFolds,
    /// Selection rule on the inner curve. Default [`LambdaRule::Min`].
    pub rule: LambdaRule,
    /// Metric of the inner curves and of the outer test sets. Default MSE.
    pub metric: CvMetric,
    /// Size of each outer fold's default grid (when no `lambdas` are given).
    /// Default 100.
    pub n_lambda: usize,
    /// `lambda_min_ratio` of that grid; `None` for glmnet's default.
    pub lambda_min_ratio: Option<f64>,
}

impl NestedCvOptions {
    /// Options with the given inner folds and the defaults otherwise.
    pub fn new(inner: InnerFolds) -> Self {
        Self {
            inner,
            rule: LambdaRule::Min,
            metric: CvMetric::Mse,
            n_lambda: 100,
            lambda_min_ratio: None,
        }
    }
}

/// One outer split of a nested cross-validation.
#[derive(Debug, Clone)]
pub struct NestedCvFold {
    /// Index of the outer split.
    pub split: usize,
    /// `λ` selected by the inner CV.
    pub lambda: f64,
    /// Index of `lambda` in `inner.lambdas`.
    pub index: usize,
    /// Metric of the refit at `lambda` on the outer test rows.
    pub test_metric: f64,
    /// Outer training rows.
    pub n_train: usize,
    /// Outer test rows.
    pub n_test: usize,
    /// The inner CV curve (its `path` is the fit on the outer training rows).
    pub inner: CvPath,
}

/// Result of [`nested_cv_path`].
#[derive(Debug, Clone)]
pub struct NestedCv {
    /// One entry per outer split, in split order.
    pub folds: Vec<NestedCvFold>,
    /// Outer test metric averaged with the test sizes as weights.
    pub mean: f64,
    /// Its standard error, `cv.glmnet`'s formula over the outer splits.
    pub se: f64,
    /// The metric.
    pub metric: CvMetric,
}

fn inner_local_splits(
    train: &[usize],
    inner: &InnerFolds,
    n: usize,
) -> Result<Vec<CvSplit>, RegressionError> {
    let keys_of = |keys: &[usize]| -> Result<Vec<usize>, RegressionError> {
        if keys.len() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: keys.len(),
            });
        }
        Ok(train.iter().map(|&i| keys[i]).collect())
    };
    let m = train.len();
    match inner {
        InnerFolds::Kfold { k, seed } => {
            check_k(*k, m, "inner k-fold")?;
            Ok(splits_from_fold_ids(&kfold_ids(m, *k, *seed)))
        }
        InnerFolds::Group {
            groups,
            k,
            balance,
            seed,
        } => Ok(splits_from_fold_ids(&group_kfold_ids(
            &keys_of(groups)?,
            *k,
            *balance,
            *seed,
        )?)),
        InnerFolds::Logo { groups } => Ok(splits_from_fold_ids(&logo_ids(&keys_of(groups)?))),
        InnerFolds::Stratified { strata, k, seed } => Ok(splits_from_fold_ids(
            &stratified_kfold_ids(&keys_of(strata)?, *k, *seed)?,
        )),
        InnerFolds::TimeSeries { opts, time_keys } => match time_keys {
            Some(keys) => group_time_series_splits(&keys_of(keys)?, opts),
            None => time_series_splits(m, opts),
        },
    }
}

/// The inner splits [`nested_cv_path`] uses for each outer split, as row
/// indices of the full data. They only ever contain rows of the outer
/// training set.
pub fn nested_inner_splits(
    n: usize,
    outer: &[CvSplit],
    inner: &InnerFolds,
) -> Result<Vec<Vec<CvSplit>>, RegressionError> {
    outer
        .iter()
        .map(|o| {
            let local = inner_local_splits(&o.train, inner, n)?;
            Ok(local
                .into_iter()
                .map(|s| CvSplit {
                    train: s.train.iter().map(|&i| o.train[i]).collect(),
                    test: s.test.iter().map(|&i| o.train[i]).collect(),
                })
                .collect())
        })
        .collect()
}

/// Nested cross-validation of the path: an honest estimate of the prediction
/// error of "choose `λ` by CV, then refit".
///
/// For each outer split, the inner folds are built from the outer training
/// rows only ([`nested_inner_splits`]), `λ` is chosen on the inner CV curve
/// ([`cv_path_splits`]) by `rule`, the path fitted on all outer training rows
/// is evaluated at that `λ` on the outer test rows. With `lambdas = None`
/// each outer split uses glmnet's default grid computed from its own training
/// rows, so the test rows do not influence the grid either. No outer test row
/// enters any inner fit, inner prediction, grid or refit.
pub fn nested_cv_path(
    x: &Mat<f64>,
    y: &Col<f64>,
    lambdas: Option<&[f64]>,
    outer: &[CvSplit],
    opts: &PathOptions,
    nested: &NestedCvOptions,
) -> Result<NestedCv, RegressionError> {
    validate_xy(x, y)?;
    opts.validate()?;
    if let Some(l) = lambdas {
        validate_lambdas(l)?;
    }
    let n = x.nrows();
    validate_splits(outer, n)?;
    let mut folds = Vec::with_capacity(outer.len());
    for (s, split) in outer.iter().enumerate() {
        let (xt, yt) = subset(x, y, &split.train);
        let grid = match lambdas {
            Some(l) => l.to_vec(),
            None => lambda_grid(&xt, &yt, opts, nested.n_lambda, nested.lambda_min_ratio)?,
        };
        let local = inner_local_splits(&split.train, &nested.inner, n)?;
        let inner = cv_path_splits(&xt, &yt, &grid, &local, opts, nested.metric)?;
        let index = match nested.rule {
            LambdaRule::Min => inner.index_min,
            LambdaRule::OneSe => inner.index_1se,
        };
        let (xv, _) = subset(x, y, &split.test);
        let pred = inner.path.predict(&xv);
        let test_metric = split
            .test
            .iter()
            .enumerate()
            .map(|(i, &row)| nested.metric.loss(y[row] - pred[(i, index)]))
            .sum::<f64>()
            / split.test.len() as f64;
        folds.push(NestedCvFold {
            split: s,
            lambda: grid[index],
            index,
            test_metric,
            n_train: split.train.len(),
            n_test: split.test.len(),
            inner,
        });
    }
    let n_tot: f64 = folds.iter().map(|f| f.n_test as f64).sum();
    let mean = folds
        .iter()
        .map(|f| f.n_test as f64 * f.test_metric)
        .sum::<f64>()
        / n_tot;
    let var = folds
        .iter()
        .map(|f| f.n_test as f64 * (f.test_metric - mean).powi(2))
        .sum::<f64>()
        / n_tot;
    let se = (var / (folds.len() as f64 - 1.0)).sqrt();
    Ok(NestedCv {
        folds,
        mean,
        se,
        metric: nested.metric,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_codes_sorted() {
        let (c, m) = group_codes(&["b", "a", "c", "a"]);
        assert_eq!(c, vec![1, 0, 2, 0]);
        assert_eq!(m, 3);
    }

    #[test]
    fn random_groups_whole_and_chunked() {
        let groups: Vec<usize> = (0..50).map(|i| (i * 7) % 11).collect();
        let f = group_kfold_ids(&groups, 3, GroupBalance::Random, 9).unwrap();
        assert_eq!(
            f,
            group_kfold_ids(&groups, 3, GroupBalance::Random, 9).unwrap()
        );
        let mut per_fold = [0usize; 3];
        for g in 0..11 {
            let fs: Vec<usize> = (0..50).filter(|&i| groups[i] == g).map(|i| f[i]).collect();
            assert!(fs.windows(2).all(|w| w[0] == w[1]));
            per_fold[fs[0]] += 1;
        }
        per_fold.sort_unstable();
        assert_eq!(per_fold, [3, 4, 4]);
    }

    #[test]
    fn quantile_strata_bins_and_ties() {
        let y = Col::from_fn(10, |i| {
            [5.0, 1.0, 3.0, 3.0, 9.0, 0.0, 7.0, 2.0, 8.0, 6.0][i]
        });
        let s = quantile_strata(&y, 5);
        // Sorted: 0 1 2 3 3 5 6 7 8 9 -> bins 0 0 1 1 1(tie) 2 3 3 4 4
        assert_eq!(s, vec![2, 0, 1, 1, 4, 0, 3, 1, 4, 3]);
    }

    #[test]
    fn splits_errors() {
        assert!(time_series_splits(10, &TimeSeriesSplitOptions::new(1)).is_err());
        let o = TimeSeriesSplitOptions {
            test_size: Some(3),
            gap: 1,
            ..TimeSeriesSplitOptions::new(3)
        };
        assert!(time_series_splits(10, &o).is_err());
        let o = TimeSeriesSplitOptions {
            min_train_size: Some(5),
            ..TimeSeriesSplitOptions::new(3)
        };
        assert!(time_series_splits(12, &o).is_err());
    }
}
