#![allow(clippy::excessive_precision)]

//! #109: penalized maximum likelihood à la rms (RMS §9.10): the penalty
//! matrix, differential penalties, effective d.f., the sandwich covariance
//! and `pentrace`, against rms. Reference values come from
//! `tests/r_scripts/generate_pentrace_validation.R` (`lrm` at its default
//! convergence, hence 1e-5 on fitted quantities).

use anofox_regression::design::TermMap;
use anofox_regression::solvers::{
    base_penalty_matrix, effective_df_by_term, penalized_ols, penalty_grid, penalty_matrix,
    penalty_multipliers, pentrace, pentrace_ols, OrdinalLink, OrdinalRegressor, PenaltySpec,
    PentraceCriterion,
};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/pentrace_validation.rs");
}
use fixture::*;

const N: usize = 300;
const P: usize = 9;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

/// lrm(y ~ rcs(x1, 4) * x2 + g): x1 x1' x1'' x2 g=b g=c x1*x2 x1'*x2 x1''*x2
fn design() -> (TermMap, Vec<usize>, Mat<f64>, Col<f64>) {
    let mut t = TermMap::new();
    let x1 = t.add_main("x1", &[false, true, true]);
    let x2 = t.add_main("x2", &[false]);
    let g = t.add_main("g", &[false, false]);
    t.add_interaction(x1, x2, false).unwrap();
    let x = Mat::from_fn(N, P, |i, j| X[i * P + j]);
    let y = Col::from_fn(N, |i| Y[i]);
    (t, vec![g], x, y)
}

fn full_penalty() -> PenaltySpec {
    PenaltySpec {
        simple: 1.0,
        nonlinear: Some(4.0),
        interaction: Some(2.0),
        nonlinear_interaction: Some(8.0),
    }
}

#[test]
fn penalty_matrix_and_multipliers_match_rms() {
    let (t, cat, x, _) = design();
    let base = base_penalty_matrix(&x, &t, &cat).unwrap();
    for a in 0..P {
        for b in 0..P {
            close(
                &format!("P0 {a},{b}"),
                base[(a, b)],
                BASE_PENALTY[a * P + b],
                1e-12,
            );
        }
    }
    let mult = penalty_multipliers(&t, &full_penalty());
    assert_eq!(mult, MULTIPLIERS.to_vec());
    // fallbacks: interaction ← nonlinear ← simple
    let spec = PenaltySpec {
        simple: 2.0,
        nonlinear: Some(5.0),
        ..Default::default()
    };
    assert_eq!(spec.resolved(), [2.0, 5.0, 5.0, 5.0]);
    assert_eq!(PenaltySpec::uniform(3.0).resolved(), [3.0; 4]);
}

#[test]
fn penalized_lrm_matches_rms() {
    let (t, cat, x, y) = design();
    let pm = penalty_matrix(&x, &t, &cat, &full_penalty()).unwrap();
    let f = OrdinalRegressor::builder()
        .penalty_matrix(pm)
        .build()
        .fit(&x, &y)
        .unwrap();
    // binary lrm: P(Y = 1) = F(α + xβ), the one intercept of the cumulative model
    close("intercept", f.intercepts()[0], PEN_COEF[0], 1e-5);
    for j in 0..P {
        close(&format!("β{j}"), f.coefficients()[j], PEN_COEF[1 + j], 1e-5);
        close(
            &format!("effective df {j}"),
            f.effective_df_diagonal()[j],
            PEN_EFFECTIVE_DF[j],
            1e-5,
        );
    }
    close("LR χ²", f.lr_chisq(), PEN_LR_DF[0], 1e-5);
    close("d.f.", f.effective_df(), PEN_LR_DF[1], 1e-5);
    let (v, s) = (f.vcov(), f.vcov_sandwich());
    let d = P + 1;
    for a in 0..d {
        for b in 0..d {
            close(
                &format!("vcov {a},{b}"),
                v[(a, b)],
                PEN_VAR[a * d + b],
                1e-5,
            );
            close(
                &format!("sandwich {a},{b}"),
                s[(a, b)],
                PEN_SANDWICH[a * d + b],
                1e-5,
            );
        }
    }
    let by_term = effective_df_by_term(&t, &f.effective_df_diagonal());
    assert_eq!(by_term.len(), 4);
    close(
        "by term sums",
        by_term.iter().sum::<f64>(),
        f.effective_df(),
        1e-12,
    );
    close(
        "x1 term",
        by_term[0],
        PEN_EFFECTIVE_DF[..3].iter().sum(),
        1e-5,
    );
    // without a penalty the sandwich is the ordinary covariance
    let f0 = OrdinalRegressor::builder().build().fit(&x, &y).unwrap();
    assert_eq!(f0.vcov_sandwich(), f0.vcov());
}

#[test]
fn pentrace_matches_rms() {
    let (t, cat, x, y) = design();
    let grid = penalty_grid(&[0.0, 2.0], &[0.0, 4.0, 16.0], &[], &[]);
    assert_eq!(grid.len(), TRACE_PENALTY.len() / 2);
    for (i, spec) in grid.iter().enumerate() {
        assert_eq!(spec.simple, TRACE_PENALTY[2 * i], "row {i} simple");
        assert_eq!(
            spec.nonlinear,
            Some(TRACE_PENALTY[2 * i + 1]),
            "row {i} nonlinear"
        );
    }
    let tr = pentrace(
        &x,
        &y,
        &t,
        &cat,
        &grid,
        OrdinalLink::Logistic,
        PentraceCriterion::CorrectedAic,
    )
    .unwrap();
    for (i, row) in tr.rows.iter().enumerate() {
        let r = &TRACE_STATS[4 * i..4 * i + 4];
        close(&format!("row {i} df"), row.df, r[0], 1e-5);
        close(&format!("row {i} aic"), row.aic, r[1], 1e-5);
        close(&format!("row {i} bic"), row.bic, r[2], 1e-5);
        close(&format!("row {i} aic.c"), row.aic_c, r[3], 1e-5);
    }
    let best = tr.rows[tr.best].penalty;
    assert_eq!(
        (best.simple, best.nonlinear),
        (TRACE_BEST[0], Some(TRACE_BEST[1]))
    );
    close(
        "best fit d.f.",
        tr.best_fit.effective_df(),
        tr.rows[tr.best].df,
        1e-12,
    );

    let uniform: Vec<PenaltySpec> = (0..UNIFORM_STATS.len() / 5)
        .map(|i| PenaltySpec::uniform(UNIFORM_STATS[5 * i]))
        .collect();
    let tr = pentrace(
        &x,
        &y,
        &t,
        &cat,
        &uniform,
        OrdinalLink::Logistic,
        PentraceCriterion::CorrectedAic,
    )
    .unwrap();
    for (i, row) in tr.rows.iter().enumerate() {
        let r = &UNIFORM_STATS[5 * i..5 * i + 5];
        close(&format!("uniform {} df", r[0]), row.df, r[1], 1e-5);
        close(&format!("uniform {} aic", r[0]), row.aic, r[2], 1e-5);
        close(&format!("uniform {} bic", r[0]), row.bic, r[3], 1e-5);
        close(&format!("uniform {} aic.c", r[0]), row.aic_c, r[4], 1e-5);
    }
    assert_eq!(tr.rows[tr.best].penalty.simple, UNIFORM_BEST[0]);
    // the other criteria pick their own maxima
    for crit in [PentraceCriterion::Aic, PentraceCriterion::Bic] {
        let tr = pentrace(&x, &y, &t, &cat, &uniform, OrdinalLink::Logistic, crit).unwrap();
        let value = |r: &anofox_regression::solvers::PentraceRow| match crit {
            PentraceCriterion::Aic => r.aic,
            _ => r.bic,
        };
        let max = tr.rows.iter().map(value).fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(value(&tr.rows[tr.best]), max);
    }
}

#[test]
fn pentrace_rejects_bad_input() {
    let (t, cat, x, y) = design();
    assert!(pentrace(
        &x,
        &y,
        &t,
        &cat,
        &[],
        OrdinalLink::Logistic,
        PentraceCriterion::Aic
    )
    .is_err());
    let narrow = Mat::from_fn(N, 3, |i, j| X[i * P + j]);
    assert!(base_penalty_matrix(&narrow, &t, &cat).is_err());
    assert!(base_penalty_matrix(&x, &t, &[7]).is_err());
    assert!(base_penalty_matrix(&Mat::zeros(1, P), &t, &cat).is_err());
}

#[test]
fn penalized_ols_and_its_pentrace_match_rms() {
    let (t, cat, x, _) = design();
    let y = Col::from_fn(N, |i| Y_C[i]);
    let pm = penalty_matrix(&x, &t, &cat, &full_penalty()).unwrap();
    let f = penalized_ols(&x, &y, &pm).unwrap();
    let d = P + 1;
    for j in 0..d {
        close(
            &format!("ols coef {j}"),
            f.coefficients[j],
            OLS_COEF[j],
            1e-10,
        );
        close(
            &format!("ols effective df {j}"),
            f.effective_df_diagonal[j],
            OLS_EFFECTIVE_DF[j],
            1e-10,
        );
        for l in 0..d {
            close(
                &format!("ols var {j},{l}"),
                f.vcov[(j, l)],
                OLS_VAR[j * d + l],
                1e-10,
            );
            close(
                &format!("ols sandwich {j},{l}"),
                f.vcov_sandwich[(j, l)],
                OLS_SANDWICH[j * d + l],
                1e-10,
            );
        }
    }
    close("ols LR", f.lr_chisq, OLS_STATS[0], 1e-10);
    close("ols d.f.", f.effective_df, OLS_STATS[1], 1e-10);
    close("ols R²", f.r_squared, OLS_STATS[2], 1e-10);
    close("ols σ", f.sigma, OLS_STATS[3], 1e-10);
    close(
        "residuals",
        f.residuals.iter().map(|r| r * r).sum::<f64>(),
        f.residuals.nrows() as f64 * f.sigma * f.sigma - {
            let b = &f.coefficients;
            (0..P)
                .map(|r| {
                    (0..P)
                        .map(|c| b[r + 1] * pm[(r, c)] * b[c + 1])
                        .sum::<f64>()
                })
                .sum::<f64>()
        },
        1e-10,
    );

    let grid = penalty_grid(&[0.0, 2.0], &[0.0, 4.0, 16.0], &[], &[]);
    let tr = pentrace_ols(&x, &y, &t, &cat, &grid, PentraceCriterion::CorrectedAic).unwrap();
    for (i, row) in tr.rows.iter().enumerate() {
        let r = &OLS_TRACE_STATS[4 * i..4 * i + 4];
        close(&format!("ols row {i} df"), row.df, r[0], 1e-10);
        close(&format!("ols row {i} aic"), row.aic, r[1], 1e-10);
        close(&format!("ols row {i} bic"), row.bic, r[2], 1e-10);
        close(&format!("ols row {i} aic.c"), row.aic_c, r[3], 1e-10);
    }
    let best = tr.rows[tr.best].penalty;
    assert_eq!(
        (best.simple, best.nonlinear),
        (OLS_TRACE_BEST[0], Some(OLS_TRACE_BEST[1]))
    );
    // bad input
    assert!(penalized_ols(&x, &y, &Mat::identity(2, 2)).is_err());
    assert!(penalized_ols(&x, &Col::from_fn(5, |i| Y_C[i]), &pm).is_err());
    assert!(penalized_ols(&Mat::zeros(1, P), &Col::from_fn(1, |_| 1.0), &pm).is_err());
}
