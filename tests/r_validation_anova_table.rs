//! `inference::anova_table` against R 4.x `anova(lm(...))` (type I) and
//! car 3.1.5 `Anova(fit, type = 2)` (type II). Reference values from
//! `tests/r_scripts/generate_anova_table_validation.R`.

#![allow(clippy::excessive_precision)]

use anofox_regression::inference::{
    anova_table, AnovaTable, AnovaTableError, AnovaTableOptions, AnovaType,
};
use faer::{Col, Mat};

const TOL: f64 = 1e-9;

const X1: [f64; 12] = [
    1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9,
];
const X2: [f64; 12] = [
    0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5,
];
const X3: [f64; 12] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0];
const Y: [f64; 12] = [
    2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4,
];

const MPG: [f64; 32] = [
    21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4,
    14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4,
];
const WT: [f64; 32] = [
    2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78, 5.25,
    5.424, 5.345, 2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17,
    2.77, 3.57, 2.78,
];
const CYL: [f64; 32] = [
    6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0,
    4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 6.0, 8.0, 4.0,
];
const HP: [f64; 32] = [
    110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0, 123.0, 180.0, 180.0, 180.0,
    205.0, 215.0, 230.0, 66.0, 52.0, 65.0, 97.0, 150.0, 150.0, 245.0, 175.0, 66.0, 91.0, 113.0,
    264.0, 175.0, 335.0, 109.0,
];

fn close(name: &str, got: f64, want: f64) {
    assert!(
        (got - want).abs() <= TOL * want.abs().max(1.0),
        "{name}: got {got:e}, want {want:e}"
    );
}

/// Design with the given columns (x4 = 2·x1 − x2 is exactly collinear).
fn design(cols: &[&str]) -> Mat<f64> {
    Mat::from_fn(12, cols.len(), |i, j| match cols[j] {
        "x1" => X1[i],
        "x2" => X2[i],
        "x3" => X3[i],
        "x4" => 2.0 * X1[i] - X2[i],
        c => panic!("unknown column {c}"),
    })
}

fn y() -> Col<f64> {
    Col::from_fn(12, |i| Y[i])
}

/// mpg ~ wt + factor(cyl) + hp: columns wt, cyl == 6, cyl == 8, hp.
fn mtcars() -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(32, 4, |i, j| match j {
        0 => WT[i],
        1 => f64::from(u8::from(CYL[i] == 6.0)),
        2 => f64::from(u8::from(CYL[i] == 8.0)),
        _ => HP[i],
    });
    (x, Col::from_fn(32, |i| MPG[i]))
}

fn options(anova_type: AnovaType, with_intercept: bool) -> AnovaTableOptions {
    AnovaTableOptions {
        anova_type,
        with_intercept,
        ..Default::default()
    }
}

/// (df, sum_sq, F, p) per term, then (residual df, residual SS).
fn check(t: &AnovaTable, terms: &[(usize, f64, f64, f64)], residual: (usize, f64)) {
    assert_eq!(t.terms.len(), terms.len(), "number of terms");
    for (k, (row, &(df, ss, f, p))) in t.terms.iter().zip(terms).enumerate() {
        assert_eq!(row.term, k);
        assert_eq!(row.df, df, "df of term {k}");
        close(&format!("SS term {k}"), row.sum_sq, ss);
        close(&format!("MS term {k}"), row.mean_sq, ss / df as f64);
        close(&format!("F term {k}"), row.f_statistic, f);
        close(&format!("p term {k}"), row.p_value, p);
    }
    assert_eq!(t.residual_df, residual.0, "residual df");
    close("residual SS", t.residual_sum_sq, residual.1);
    close(
        "residual MS",
        t.residual_mean_sq,
        residual.1 / residual.0 as f64,
    );
}

#[test]
fn type_i_x1_x2_x3() {
    let t = anova_table(
        &design(&["x1", "x2", "x3"]),
        &y(),
        None,
        &options(AnovaType::I, true),
    )
    .unwrap();
    assert_eq!(t.anova_type, AnovaType::I);
    assert_eq!(t.n_observations, 12);
    check(
        &t,
        &[
            (
                1,
                4.19985688995136002e+02,
                1.33925719060596907e+03,
                3.40761192012848685e-10,
            ),
            (
                1,
                2.44185728190502864e+01,
                7.78663418520132211e+01,
                2.14185200229875218e-05,
            ),
            (
                1,
                1.46970239276738246e-01,
                4.68661087541797416e-01,
                5.12944752624746703e-01,
            ),
        ],
        (8, 2.50876794653673052e+00),
    );
}

#[test]
fn type_ii_x1_x2_x3() {
    let t = anova_table(
        &design(&["x1", "x2", "x3"]),
        &y(),
        None,
        &options(AnovaType::II, true),
    )
    .unwrap();
    check(
        &t,
        &[
            (
                1,
                2.97030757904135839e+02,
                9.47176508099688476e+02,
                1.35003073242217786e-09,
            ),
            (
                1,
                1.99150693446599405e+01,
                6.35054967826801757e+01,
                4.49040674179866078e-05,
            ),
            (
                1,
                1.46970239276738468e-01,
                4.68661087541798027e-01,
                5.12944752624746592e-01,
            ),
        ],
        (8, 2.50876794653673096e+00),
    );
}

#[test]
fn type_i_depends_on_term_order() {
    // anova(lm(y ~ x3 + x1 + x2)), as a term order over the x1, x2, x3 design.
    let terms = vec![vec![2], vec![0], vec![1]];
    let t = anova_table(
        &design(&["x1", "x2", "x3"]),
        &y(),
        Some(&terms),
        &options(AnovaType::I, true),
    )
    .unwrap();
    check(
        &t,
        &[
            (
                1,
                1.47515235849056523e+02,
                4.70398981468800343e+02,
                2.15269684509675983e-08,
            ),
            (
                1,
                2.77120926859746760e+02,
                8.83687715294043869e+02,
                1.77799860511455681e-09,
            ),
            (
                1,
                1.99150693446599547e+01,
                6.35054967826802113e+01,
                4.49040674179865265e-05,
            ),
        ],
        (8, 2.50876794653673141e+00),
    );
    assert_eq!(t.terms[0].columns, vec![2]);
}

#[test]
fn type_ii_is_invariant_to_term_order() {
    let x = design(&["x1", "x2", "x3"]);
    let opts = options(AnovaType::II, true);
    let a = anova_table(&x, &y(), None, &opts).unwrap();
    let terms = vec![vec![2], vec![0], vec![1]];
    let b = anova_table(&x, &y(), Some(&terms), &opts).unwrap();
    for (k, &j) in [2usize, 0, 1].iter().enumerate() {
        close("SS", b.terms[k].sum_sq, a.terms[j].sum_sq);
        close("F", b.terms[k].f_statistic, a.terms[j].f_statistic);
    }
}

#[test]
fn without_intercept() {
    let x = design(&["x1", "x2", "x3"]);
    let t1 = anova_table(&x, &y(), None, &options(AnovaType::I, false)).unwrap();
    check(
        &t1,
        &[
            (
                1,
                2.29252948576637846e+03,
                7.25456758744582567e+03,
                2.14741592057949608e-14,
            ),
            (
                1,
                2.62780351675297652e+01,
                8.31552149587269582e+01,
                7.66754255698895974e-06,
            ),
            (
                1,
                4.08372215726307997e-01,
                1.29226858725925675e+00,
                2.84990960542834015e-01,
            ),
        ],
        (9, 2.84410685036594302e+00),
    );
    let t2 = anova_table(&x, &y(), None, &options(AnovaType::II, false)).unwrap();
    check(
        &t2,
        &[
            (
                1,
                4.06054952569753084e+02,
                1.28493575150229140e+03,
                5.06410959502653182e-11,
            ),
            (
                1,
                1.99308667388766274e+01,
                6.30699935295362266e+01,
                2.34666013245371934e-05,
            ),
            (
                1,
                4.08372215726307886e-01,
                1.29226858725925675e+00,
                2.84990960542834015e-01,
            ),
        ],
        (9, 2.84410685036594257e+00),
    );
}

#[test]
fn aliased_term_has_zero_df() {
    // R's anova(lm(y ~ x1 + x2 + x4 + x3)) drops the aliased x4 and reports
    // the x1 + x2 + x3 table; here x4 keeps a row with df = 0.
    let t = anova_table(
        &design(&["x1", "x2", "x4", "x3"]),
        &y(),
        None,
        &options(AnovaType::I, true),
    )
    .unwrap();
    let x4 = &t.terms[2];
    assert_eq!((x4.df, x4.sum_sq), (0, 0.0));
    assert!(x4.mean_sq.is_nan() && x4.f_statistic.is_nan() && x4.p_value.is_nan());
    let mut kept = t.clone();
    kept.terms.remove(2);
    for (k, row) in kept.terms.iter_mut().enumerate() {
        row.term = k;
    }
    let reference = anova_table(
        &design(&["x1", "x2", "x3"]),
        &y(),
        None,
        &options(AnovaType::I, true),
    )
    .unwrap();
    for (a, b) in kept.terms.iter().zip(&reference.terms) {
        assert_eq!(a.df, b.df);
        close("SS", a.sum_sq, b.sum_sq);
        close("F", a.f_statistic, b.f_statistic);
    }
    assert_eq!(t.residual_df, 8);

    // Type II: x1, x2 and x4 are each in the span of the other two.
    let t2 = anova_table(
        &design(&["x1", "x2", "x4", "x3"]),
        &y(),
        None,
        &options(AnovaType::II, true),
    )
    .unwrap();
    assert!(t2.terms[..3].iter().all(|r| r.df == 0));
    assert_eq!(t2.terms[3].df, 1);
    close("SS x3", t2.terms[3].sum_sq, 1.46970239276738468e-01);
}

#[test]
fn mtcars_factor_term_type_i() {
    let (x, y) = mtcars();
    let terms = vec![vec![0], vec![1, 2], vec![3]];
    let t = anova_table(&x, &y, Some(&terms), &options(AnovaType::I, true)).unwrap();
    check(
        &t,
        &[
            (
                1,
                8.47725249956657080e+02,
                1.42361727695374839e+02,
                2.82849164659697284e-12,
            ),
            (
                2,
                9.52632898749658636e+01,
                7.99896342195708154e+00,
                1.87011943620145706e-03,
            ),
            (
                1,
                2.22810136837252628e+01,
                3.74173543017810761e+00,
                6.36126864018736193e-02,
            ),
        ],
        (27, 1.60777633984652510e+02),
    );
    assert_eq!(t.terms[1].columns, vec![1, 2]);
}

#[test]
fn mtcars_factor_term_type_ii() {
    let (x, y) = mtcars();
    let terms = vec![vec![0], vec![1, 2], vec![3]];
    let t = anova_table(&x, &y, Some(&terms), &options(AnovaType::II, true)).unwrap();
    check(
        &t,
        &[
            (
                1,
                1.16390091187040611e+02,
                1.95458309975507873e+01,
                1.44175576105983686e-04,
            ),
            (
                2,
                3.42701207568139239e+01,
                2.87755590594865440e+00,
                7.36449805084616049e-02,
            ),
            (
                1,
                2.22810136837252628e+01,
                3.74173543017810672e+00,
                6.36126864018736610e-02,
            ),
        ],
        (27, 1.60777633984652539e+02),
    );
}

#[test]
fn invalid_terms() {
    let r = anova_table(
        &design(&["x1", "x2"]),
        &y(),
        Some(&[vec![0]]),
        &AnovaTableOptions::default(),
    );
    assert!(matches!(r, Err(AnovaTableError::InvalidTerms(_))));
}
