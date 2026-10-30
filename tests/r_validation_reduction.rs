#![allow(clippy::excessive_precision, clippy::approx_constant)]

//! #107: data reduction (RMS §4.7): `spearman2`, Hoeffding's D, `varclus`
//! and `stats::hclust` for every linkage, against Hmisc and stats.
//! Reference values come from
//! `tests/r_scripts/generate_reduction_validation.R`.

use anofox_regression::reduction::{
    hclust, hoeffding_d, spearman2, spearman2_categorical, varclus, Hclust, Linkage, Node,
    Similarity, SimilarityTransform,
};
use faer::Mat;

mod fixture {
    include!("fixtures/reduction_validation.rs");
}
use fixture::*;

const N: usize = 80;
const P: usize = 6;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *u, *v, tol);
    }
}

fn col(j: usize) -> Vec<f64> {
    (0..N).map(|i| X[i * P + j]).collect()
}

fn x() -> Mat<f64> {
    Mat::from_fn(N, P, |i, j| X[i * P + j])
}

fn flat(m: &Mat<f64>) -> Vec<f64> {
    let p = m.nrows();
    (0..p * p).map(|k| m[(k % p, k / p)]).collect()
}

/// R's `merge` (rows of `(a, b)`, negative leaves, 1-based), `height` and
/// 1-based `order`.
fn check_tree(tag: &str, h: &Hclust, merge: &[f64], height: &[f64], order: &[f64]) {
    let r = |nd: Node| match nd {
        Node::Leaf(i) => -(i as f64 + 1.0),
        Node::Cluster(c) => c as f64 + 1.0,
    };
    let m: Vec<f64> = h.merge.iter().flat_map(|&(a, b)| [r(a), r(b)]).collect();
    assert_eq!(m, merge, "{tag} merge");
    all_close(&format!("{tag} height"), &h.height, height, 1e-12);
    let o: Vec<f64> = h.order.iter().map(|&i| i as f64 + 1.0).collect();
    assert_eq!(o, order, "{tag} order");
}

#[test]
fn spearman2_matches_hmisc() {
    for (p, expected) in [(1, &SP1), (2, &SP2)] {
        for j in 0..P {
            let s = spearman2(&col(j), &Y, p).unwrap();
            let got = [
                s.rho2,
                s.f,
                s.df1,
                s.df2,
                s.p_value,
                s.n as f64,
                s.adjusted_rho2,
            ];
            all_close(
                &format!("p={p} x{j}"),
                &got,
                &expected[j * 7..j * 7 + 7],
                1e-10,
            );
        }
    }
    let levels: Vec<usize> = col(5).iter().map(|&v| v as usize).collect();
    let s = spearman2_categorical(&levels, &Y).unwrap();
    let got = [
        s.rho2,
        s.f,
        s.df1,
        s.df2,
        s.p_value,
        s.n as f64,
        s.adjusted_rho2,
    ];
    all_close("factor", &got, &SPF, 1e-10);
}

#[test]
fn hoeffding_d_matches_hmisc() {
    let h = hoeffding_d(&x()).unwrap();
    all_close("D", &flat(&h.d), &HOEFFD_D, 1e-12);
    let p: Vec<f64> = flat(&h.p_value)
        .iter()
        .map(|v| if v.is_nan() { -1.0 } else { *v })
        .collect();
    all_close("P", &p, &HOEFFD_P, 1e-10);
}

#[test]
fn varclus_matches_hmisc() {
    let cases = [
        (
            Similarity::Spearman,
            &VC_SPEARMAN_SIM,
            &VC_SPEARMAN_MERGE,
            &VC_SPEARMAN_HEIGHT,
            &VC_SPEARMAN_ORDER,
        ),
        (
            Similarity::Pearson,
            &VC_PEARSON_SIM,
            &VC_PEARSON_MERGE,
            &VC_PEARSON_HEIGHT,
            &VC_PEARSON_ORDER,
        ),
        (
            Similarity::Hoeffding,
            &VC_HOEFFDING_SIM,
            &VC_HOEFFDING_MERGE,
            &VC_HOEFFDING_HEIGHT,
            &VC_HOEFFDING_ORDER,
        ),
    ];
    for (s, sim, merge, height, order) in cases {
        let v = varclus(&x(), s, SimilarityTransform::Square, Linkage::Complete).unwrap();
        all_close(&format!("{s:?} sim"), &flat(&v.sim), sim, 1e-12);
        check_tree(&format!("{s:?}"), &v.hclust, merge, height, order);
    }
    let v = varclus(
        &x(),
        Similarity::Spearman,
        SimilarityTransform::Abs,
        Linkage::Average,
    )
    .unwrap();
    let m: Vec<f64> = v
        .hclust
        .merge
        .iter()
        .flat_map(|&(a, b)| {
            [a, b].map(|nd| match nd {
                Node::Leaf(i) => -(i as f64 + 1.0),
                Node::Cluster(c) => c as f64 + 1.0,
            })
        })
        .collect();
    assert_eq!(m, VC_ABS_AVG_MERGE);
    all_close(
        "abs average height",
        &v.hclust.height,
        &VC_ABS_AVG_HEIGHT,
        1e-12,
    );
}

#[test]
fn hclust_matches_r_for_every_linkage_with_ties() {
    let d = Mat::from_fn(8, 8, |i, j| HC_DIST[j * 8 + i]);
    let cases = [
        (
            Linkage::WardD,
            &HC_WARDD_MERGE,
            &HC_WARDD_HEIGHT,
            &HC_WARDD_ORDER,
        ),
        (
            Linkage::Single,
            &HC_SINGLE_MERGE,
            &HC_SINGLE_HEIGHT,
            &HC_SINGLE_ORDER,
        ),
        (
            Linkage::Complete,
            &HC_COMPLETE_MERGE,
            &HC_COMPLETE_HEIGHT,
            &HC_COMPLETE_ORDER,
        ),
        (
            Linkage::Average,
            &HC_AVERAGE_MERGE,
            &HC_AVERAGE_HEIGHT,
            &HC_AVERAGE_ORDER,
        ),
        (
            Linkage::Mcquitty,
            &HC_MCQUITTY_MERGE,
            &HC_MCQUITTY_HEIGHT,
            &HC_MCQUITTY_ORDER,
        ),
        (
            Linkage::Median,
            &HC_MEDIAN_MERGE,
            &HC_MEDIAN_HEIGHT,
            &HC_MEDIAN_ORDER,
        ),
        (
            Linkage::Centroid,
            &HC_CENTROID_MERGE,
            &HC_CENTROID_HEIGHT,
            &HC_CENTROID_ORDER,
        ),
        (
            Linkage::WardD2,
            &HC_WARDD2_MERGE,
            &HC_WARDD2_HEIGHT,
            &HC_WARDD2_ORDER,
        ),
    ];
    for (l, merge, height, order) in cases {
        check_tree(
            &format!("{l:?}"),
            &hclust(&d, l).unwrap(),
            merge,
            height,
            order,
        );
    }
}

#[test]
fn cut_height_groups_merged_objects() {
    let d = Mat::from_fn(8, 8, |i, j| HC_DIST[j * 8 + i]);
    let h = hclust(&d, Linkage::Complete).unwrap();
    let all = h.cut_height(f64::INFINITY);
    assert!(all.iter().all(|&g| g == 0));
    let none = h.cut_height(-1.0);
    assert_eq!(none, (0..8).collect::<Vec<_>>());
    // R: cutree(hclust(d, "complete"), h = height[3]) keeps the first three merges
    let k = h.cut_height(h.height[2]);
    let groups = k.iter().copied().max().unwrap() + 1;
    assert_eq!(groups, 8 - 3);
}

#[test]
fn rejects_bad_input() {
    assert!(spearman2(&[1.0, 2.0], &[1.0, 2.0], 1).is_err());
    assert!(spearman2(&col(0), &Y, 3).is_err());
    assert!(spearman2(&col(0), &Y[..10], 1).is_err());
    assert!(spearman2_categorical(&[1; N], &Y).is_err());
    assert!(hoeffding_d(&Mat::zeros(4, 2)).is_err());
    assert!(hclust(&Mat::zeros(1, 1), Linkage::Single).is_err());
    assert!(hclust(&Mat::from_fn(2, 2, |_, _| f64::NAN), Linkage::Single).is_err());
    let one = Mat::from_fn(N, 1, |i, _| X[i * P]);
    assert!(varclus(
        &one,
        Similarity::Spearman,
        SimilarityTransform::Square,
        Linkage::Complete
    )
    .is_err());
}
