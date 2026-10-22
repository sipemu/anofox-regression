//! Validation of the CV fold generators, explicit splits and nested CV
//! (`solvers::cv`, sipemu/anofox-regression#87).
//!
//! The fold generators are checked against scikit-learn 1.9.1
//! (`GroupKFold`, `LeaveOneGroupOut`, `StratifiedKFold`,
//! `StratifiedGroupKFold`, `TimeSeriesSplit`; generator
//! `validation/python/generate_cv_splits_validation.py`), plus property tests
//! over many random designs: no group spans two folds, stratum counts per
//! fold differ by at most one, and no outer test row reaches an inner fit.

use anofox_regression::solvers::cv::{
    cv_path_splits, group_kfold_ids, group_time_series_splits, logo_ids, nested_cv_path,
    nested_inner_splits, quantile_strata, splits_from_fold_ids, stratified_group_kfold_ids,
    stratified_kfold_ids, time_series_splits, CvMetric, CvSplit, GroupBalance, InnerFolds,
    LambdaRule, NestedCvOptions, TimeSeriesSplitOptions, TrainWindow,
};
use anofox_regression::solvers::path::{cv_path, fit_path, kfold_ids, lambda_grid, PathOptions};
use faer::{Col, Mat};

include!("fixtures/cv_splits_validation.rs");

/// Small deterministic generator for the property tests.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % bound as u64) as usize
    }
}

fn data(n: usize) -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(n, 3, |i, j| (((i + 1) * (j + 2) * 7) % 13) as f64 / 3.0);
    let y = Col::from_fn(n, |i| {
        1.0 + x[(i, 0)] - 0.5 * x[(i, 1)] + 0.2 * x[(i, 2)] + ((i * 5) % 7) as f64 / 2.0
    });
    (x, y)
}

// ---------------------------------------------------------------------------
// scikit-learn reference fixtures
// ---------------------------------------------------------------------------

#[test]
fn group_kfold_matches_sklearn() {
    let f = group_kfold_ids(&GKF_GROUPS, 4, GroupBalance::Rows, 123).unwrap();
    assert_eq!(f, GKF_FOLDS.to_vec());
}

#[test]
fn logo_matches_sklearn() {
    assert_eq!(logo_ids(&GKF_GROUPS), LOGO_FOLDS.to_vec());
}

#[test]
fn stratified_kfold_matches_sklearn() {
    let f = stratified_kfold_ids(&SKF_STRATA, 5, None).unwrap();
    assert_eq!(f, SKF_FOLDS.to_vec());
}

#[test]
fn stratified_group_kfold_matches_sklearn() {
    let f = stratified_group_kfold_ids(&SGKF_GROUPS, &SGKF_STRATA, 3, None).unwrap();
    assert_eq!(f, SGKF_FOLDS.to_vec());
}

#[test]
fn time_series_splits_match_sklearn() {
    for &(n, k, ts, gap, mts, bounds) in TSS_CASES {
        let opts = TimeSeriesSplitOptions {
            n_splits: k,
            test_size: (ts > 0).then_some(ts),
            gap,
            window: if mts > 0 {
                TrainWindow::Sliding(mts)
            } else {
                TrainWindow::Expanding
            },
            min_train_size: None,
        };
        let splits = time_series_splits(n, &opts).unwrap();
        assert_eq!(splits.len(), bounds.len(), "case n = {n}");
        for (s, &(a, b, c, d)) in splits.iter().zip(bounds) {
            assert_eq!(s.train, (a..b).collect::<Vec<_>>(), "case n = {n}");
            assert_eq!(s.test, (c..d).collect::<Vec<_>>(), "case n = {n}");
        }
    }
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

fn assert_groups_whole<G: PartialEq>(groups: &[G], folds: &[usize]) {
    for i in 0..groups.len() {
        for j in 0..i {
            if groups[i] == groups[j] {
                assert_eq!(folds[i], folds[j], "a group spans two folds");
            }
        }
    }
}

#[test]
fn group_folds_never_split_a_group() {
    let mut rng = Lcg(1);
    for case in 0..200 {
        let n = 20 + rng.next(80);
        let n_groups = 4 + rng.next(20);
        let k = 2 + rng.next(n_groups.min(8) - 1);
        // Two-variable keys (store, region) as a tuple.
        let groups: Vec<(usize, char)> = (0..n)
            .map(|_| {
                let g = rng.next(n_groups);
                (g, ['a', 'b'][(g % 3).min(1)])
            })
            .collect();
        let distinct = {
            let mut d = groups.clone();
            d.sort();
            d.dedup();
            d.len()
        };
        if distinct < k {
            continue;
        }
        for balance in [GroupBalance::Rows, GroupBalance::Random] {
            let f = group_kfold_ids(&groups, k, balance, case).unwrap();
            assert_groups_whole(&groups, &f);
            // Every fold gets at least one group.
            for fold in 0..k {
                assert!(f.contains(&fold));
            }
        }
        let strata: Vec<usize> = (0..n).map(|_| rng.next(3)).collect();
        let f = stratified_group_kfold_ids(&groups, &strata, k, Some(case)).unwrap();
        assert_groups_whole(&groups, &f);
    }
}

#[test]
fn group_kfold_rows_is_balanced_greedily() {
    // The greedy assignment keeps the largest fold at most one largest group
    // above the smallest.
    let mut rng = Lcg(5);
    for _ in 0..100 {
        let n = 30 + rng.next(70);
        let groups: Vec<usize> = (0..n).map(|_| rng.next(12)).collect();
        let f = group_kfold_ids(&groups, 3, GroupBalance::Rows, 0).unwrap();
        let mut rows = [0usize; 3];
        f.iter().for_each(|&k| rows[k] += 1);
        let biggest = (0..12)
            .map(|g| groups.iter().filter(|&&h| h == g).count())
            .max()
            .unwrap();
        assert!(rows.iter().max().unwrap() - rows.iter().min().unwrap() <= biggest);
    }
}

#[test]
fn stratified_counts_differ_by_at_most_one() {
    let mut rng = Lcg(2);
    for case in 0..200 {
        let n = 10 + rng.next(90);
        let n_strata = 1 + rng.next(5);
        let k = 2 + rng.next(5);
        let strata: Vec<usize> = (0..n).map(|_| rng.next(n_strata)).collect();
        for seed in [None, Some(case)] {
            let Ok(f) = stratified_kfold_ids(&strata, k, seed) else {
                assert!(
                    k > n || (0..n_strata).all(|s| strata.iter().filter(|&&t| t == s).count() < k)
                );
                continue;
            };
            for s in 0..n_strata {
                let per: Vec<usize> = (0..k)
                    .map(|fold| (0..n).filter(|&i| strata[i] == s && f[i] == fold).count())
                    .collect();
                let (lo, hi) = (per.iter().min().unwrap(), per.iter().max().unwrap());
                assert!(hi - lo <= 1, "stratum {s}: {per:?}");
            }
        }
    }
}

#[test]
fn quantile_strata_balance_a_regression_cv() {
    let (_, y) = data(57);
    let strata = quantile_strata(&y, 4);
    let f = stratified_kfold_ids(&strata, 5, Some(3)).unwrap();
    for s in 0..4 {
        let per: Vec<usize> = (0..5)
            .map(|fold| (0..57).filter(|&i| strata[i] == s && f[i] == fold).count())
            .collect();
        assert!(per.iter().max().unwrap() - per.iter().min().unwrap() <= 1);
    }
}

#[test]
fn group_time_series_keeps_time_points_together() {
    // Three series observed on 12 dates, rows shuffled.
    let mut rows: Vec<(usize, usize)> = (0..12).flat_map(|t| (0..3).map(move |s| (t, s))).collect();
    let mut rng = Lcg(9);
    for i in (1..rows.len()).rev() {
        rows.swap(i, rng.next(i + 1));
    }
    let time: Vec<usize> = rows.iter().map(|r| r.0).collect();
    let opts = TimeSeriesSplitOptions {
        test_size: Some(2),
        gap: 1,
        ..TimeSeriesSplitOptions::new(3)
    };
    let splits = group_time_series_splits(&time, &opts).unwrap();
    let by_date = time_series_splits(12, &opts).unwrap();
    for (s, d) in splits.iter().zip(&by_date) {
        let mut tr: Vec<usize> = s.train.iter().map(|&i| time[i]).collect();
        let mut te: Vec<usize> = s.test.iter().map(|&i| time[i]).collect();
        tr.sort_unstable();
        tr.dedup();
        te.sort_unstable();
        te.dedup();
        assert_eq!(tr, d.train);
        assert_eq!(te, d.test);
        assert_eq!(s.train.len(), 3 * d.train.len());
        assert!(tr.iter().max().unwrap() + 1 < *te.iter().min().unwrap());
    }
}

// ---------------------------------------------------------------------------
// CV on explicit splits
// ---------------------------------------------------------------------------

#[test]
fn cv_path_is_cv_path_splits_on_fold_ids() {
    let (x, y) = data(40);
    let opts = PathOptions::elastic_net(0.5);
    let lambdas = lambda_grid(&x, &y, &opts, 12, None).unwrap();
    let folds = kfold_ids(40, 5, 11);
    let a = cv_path(&x, &y, &lambdas, &folds, &opts).unwrap();
    let b = cv_path_splits(
        &x,
        &y,
        &lambdas,
        &splits_from_fold_ids(&folds),
        &opts,
        CvMetric::Mse,
    )
    .unwrap();
    assert_eq!(a.cv_mean, b.cv_mean);
    assert_eq!(a.cv_se, b.cv_se);
    assert_eq!(a.index_min, b.index_min);
    assert_eq!(a.index_1se, b.index_1se);
}

#[test]
fn mae_metric_by_hand() {
    let (x, y) = data(30);
    let opts = PathOptions::ridge();
    let lambdas = [0.5, 0.05];
    let splits = time_series_splits(30, &TimeSeriesSplitOptions::new(3)).unwrap();
    let cv = cv_path_splits(&x, &y, &lambdas, &splits, &opts, CvMetric::Mae).unwrap();
    let mut num = [0.0; 2];
    let mut den = 0.0;
    let mut per = vec![];
    for s in &splits {
        let xt = Mat::from_fn(s.train.len(), 3, |i, j| x[(s.train[i], j)]);
        let yt = Col::from_fn(s.train.len(), |i| y[s.train[i]]);
        let fit = fit_path(&xt, &yt, &lambdas, &opts).unwrap();
        let xv = Mat::from_fn(s.test.len(), 3, |i, j| x[(s.test[i], j)]);
        let p = fit.predict(&xv);
        let mut m = [0.0; 2];
        for (l, ml) in m.iter_mut().enumerate() {
            *ml = s
                .test
                .iter()
                .enumerate()
                .map(|(i, &r)| (y[r] - p[(i, l)]).abs())
                .sum::<f64>()
                / s.test.len() as f64;
            num[l] += s.test.len() as f64 * *ml;
        }
        den += s.test.len() as f64;
        per.push(m);
    }
    for l in 0..2 {
        let mean = num[l] / den;
        assert!((cv.cv_mean[l] - mean).abs() < 1e-12);
        let var: f64 = per.iter().map(|m| 7.0 * (m[l] - mean).powi(2)).sum::<f64>() / den;
        assert!((cv.cv_se[l] - (var / 2.0).sqrt()).abs() < 1e-12);
    }
    assert_eq!(cv.n_folds, 3);
}

#[test]
fn invalid_splits_are_rejected() {
    let (x, y) = data(20);
    let opts = PathOptions::ridge();
    let overlap = vec![
        CvSplit {
            train: (0..10).collect(),
            test: (9..12).collect(),
        },
        CvSplit {
            train: (0..12).collect(),
            test: (12..15).collect(),
        },
    ];
    assert!(cv_path_splits(&x, &y, &[0.1], &overlap, &opts, CvMetric::Mse).is_err());
    let one = vec![CvSplit {
        train: (0..10).collect(),
        test: (10..15).collect(),
    }];
    assert!(cv_path_splits(&x, &y, &[0.1], &one, &opts, CvMetric::Mse).is_err());
    let out_of_range = vec![
        CvSplit {
            train: (0..10).collect(),
            test: vec![25],
        },
        one[0].clone(),
    ];
    assert!(cv_path_splits(&x, &y, &[0.1], &out_of_range, &opts, CvMetric::Mse).is_err());
}

// ---------------------------------------------------------------------------
// Nested CV
// ---------------------------------------------------------------------------

fn inner_variants(n: usize) -> Vec<InnerFolds> {
    let groups: Vec<usize> = (0..n).map(|i| (i * 7) % 9).collect();
    let strata: Vec<usize> = (0..n).map(|i| i % 3).collect();
    vec![
        InnerFolds::Kfold { k: 4, seed: 1 },
        InnerFolds::Group {
            groups: groups.clone(),
            k: 3,
            balance: GroupBalance::Rows,
            seed: 0,
        },
        InnerFolds::Logo { groups },
        InnerFolds::Stratified {
            strata,
            k: 3,
            seed: Some(2),
        },
        InnerFolds::TimeSeries {
            opts: TimeSeriesSplitOptions::new(3),
            time_keys: None,
        },
        InnerFolds::TimeSeries {
            opts: TimeSeriesSplitOptions::new(3),
            time_keys: Some((0..n).map(|i| i / 2).collect()),
        },
    ]
}

#[test]
fn nested_inner_splits_never_touch_outer_test_rows() {
    let n = 72;
    let outer_variants = vec![
        splits_from_fold_ids(&kfold_ids(n, 4, 3)),
        splits_from_fold_ids(
            &group_kfold_ids(
                &(0..n).map(|i| (i * 7) % 9).collect::<Vec<_>>(),
                3,
                GroupBalance::Rows,
                0,
            )
            .unwrap(),
        ),
        time_series_splits(n, &TimeSeriesSplitOptions::new(3)).unwrap(),
    ];
    for outer in &outer_variants {
        for inner in inner_variants(n) {
            let all = nested_inner_splits(n, outer, &inner).unwrap();
            for (o, inner_splits) in outer.iter().zip(&all) {
                assert!(!inner_splits.is_empty());
                for s in inner_splits {
                    for &i in s.train.iter().chain(&s.test) {
                        assert!(
                            o.train.contains(&i),
                            "inner split uses a row outside the outer training set"
                        );
                        assert!(!o.test.contains(&i), "outer test row in an inner split");
                    }
                }
            }
            if let InnerFolds::Group { groups, .. } | InnerFolds::Logo { groups } = &inner {
                // Inner folds respect the same groups.
                for inner_splits in &all {
                    for s in inner_splits {
                        for &i in &s.test {
                            assert!(s.train.iter().all(|&j| groups[j] != groups[i]));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn nested_selection_ignores_outer_test_responses() {
    // Perturbing y on an outer test set leaves that fold's selected lambda and
    // inner curve unchanged (it changes only its test metric).
    let n = 60;
    let (x, y) = data(n);
    let outer = splits_from_fold_ids(&kfold_ids(n, 4, 8));
    let opts = PathOptions::elastic_net(0.5);
    for inner in inner_variants(n) {
        let nested = NestedCvOptions::new(inner);
        let a = nested_cv_path(&x, &y, None, &outer, &opts, &nested).unwrap();
        let mut y2 = y.clone();
        for &i in &outer[1].test {
            y2[i] += 1000.0 * ((i % 3) as f64 - 1.0);
        }
        let b = nested_cv_path(&x, &y2, None, &outer, &opts, &nested).unwrap();
        assert_eq!(a.folds[1].lambda, b.folds[1].lambda);
        assert_eq!(a.folds[1].inner.cv_mean, b.folds[1].inner.cv_mean);
        assert_ne!(a.folds[1].test_metric, b.folds[1].test_metric);
    }
}

#[test]
fn nested_equals_hand_rolled_loop() {
    let n = 64;
    let (x, y) = data(n);
    let outer = time_series_splits(n, &TimeSeriesSplitOptions::new(3)).unwrap();
    let opts = PathOptions::ridge();
    for rule in [LambdaRule::Min, LambdaRule::OneSe] {
        let nested = NestedCvOptions {
            rule,
            n_lambda: 15,
            ..NestedCvOptions::new(InnerFolds::Kfold { k: 5, seed: 4 })
        };
        let res = nested_cv_path(&x, &y, None, &outer, &opts, &nested).unwrap();
        let mut num = 0.0;
        let mut den = 0.0;
        for (o, f) in outer.iter().zip(&res.folds) {
            let xt = Mat::from_fn(o.train.len(), 3, |i, j| x[(o.train[i], j)]);
            let yt = Col::from_fn(o.train.len(), |i| y[o.train[i]]);
            let grid = lambda_grid(&xt, &yt, &opts, 15, None).unwrap();
            let cv = cv_path(&xt, &yt, &grid, &kfold_ids(o.train.len(), 5, 4), &opts).unwrap();
            let lam = match rule {
                LambdaRule::Min => cv.lambda_min,
                LambdaRule::OneSe => cv.lambda_1se,
            };
            assert_eq!(f.lambda, lam);
            let fit = fit_path(&xt, &yt, &[lam], &opts).unwrap();
            let xv = Mat::from_fn(o.test.len(), 3, |i, j| x[(o.test[i], j)]);
            let p = fit.predict(&xv);
            let mse = o
                .test
                .iter()
                .enumerate()
                .map(|(i, &r)| (y[r] - p[(i, 0)]).powi(2))
                .sum::<f64>()
                / o.test.len() as f64;
            assert!((f.test_metric - mse).abs() < 1e-9 * (1.0 + mse));
            assert_eq!(f.n_train, o.train.len());
            num += o.test.len() as f64 * mse;
            den += o.test.len() as f64;
        }
        assert!((res.mean - num / den).abs() < 1e-9 * (1.0 + res.mean));
        assert!(res.se.is_finite() && res.se >= 0.0);
    }
}

#[test]
fn nested_with_fixed_lambdas() {
    let n = 48;
    let (x, y) = data(n);
    let outer = splits_from_fold_ids(&kfold_ids(n, 3, 1));
    let lambdas = [1.0, 0.3, 0.1, 0.01];
    let res = nested_cv_path(
        &x,
        &y,
        Some(&lambdas),
        &outer,
        &PathOptions::elastic_net(1.0),
        &NestedCvOptions::new(InnerFolds::Kfold { k: 3, seed: 2 }),
    )
    .unwrap();
    assert_eq!(res.folds.len(), 3);
    for f in &res.folds {
        assert_eq!(f.inner.lambdas, lambdas.to_vec());
        assert_eq!(f.lambda, lambdas[f.index]);
    }
}
