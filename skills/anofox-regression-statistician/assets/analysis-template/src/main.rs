//! Worked example of the statistician's workflow with anofox-regression:
//! binary outcome, prespecified rcs(age, 4) * sex, model LR test and shrinkage,
//! rms::anova chunk tests, an LR test of nonlinearity, odds ratios over the
//! inter-quartile range, discrimination (lrm_stats) and bootstrap validation /
//! calibration (rms::validate / calibrate). Replace `simulate()` with
//! `Frame::read_csv(...)` for real data. Run with
//! `RAYON_NUM_THREADS=1 cargo run --release`: single-threaded faer is far faster
//! for the many tiny refits in bootstrap / CV loops.
mod data;

use anofox_regression::diagnostics::accuracy::lrm_stats;
use anofox_regression::effects::{Critical, ModelSpec};
use anofox_regression::inference::{
    anova, drop_intercept, heuristic_shrinkage, lr_test, model_lr_test, AnovaOptions,
    WaldDistribution,
};
use anofox_regression::prelude::*;
use anofox_regression::splines::Rcs;
use anofox_regression::validation::{calibrate_lrm, validate_lrm, Resamples};
use data::Rng;
use faer::{Col, Mat};

/// Ages, sex codes (0 = female, 1 = male) and a binary outcome with a U-shaped age effect.
fn simulate(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut u = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let (mut age, mut sex, mut y) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let a = 30.0 + 50.0 * u();
        let s = if u() < 0.5 { 0.0 } else { 1.0 };
        let lp = -1.0 + 0.002 * (a - 50.0).powi(2) + 0.7 * s;
        age.push(a);
        sex.push(s);
        y.push(if u() < 1.0 / (1.0 + (-lp).exp()) { 1.0 } else { 0.0 });
    }
    (age, sex, y)
}

/// Intercept-first coefficient vector of a fit.
fn full_beta(r: &RegressionResult) -> Col<f64> {
    let b = &r.coefficients;
    Col::from_fn(b.nrows() + 1, |j| if j == 0 { r.intercept.unwrap_or(0.0) } else { b[j - 1] })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (age, sex, y) = simulate(800, 7);
    let n = y.len();

    // 1. effective sample size and d.f. budget (RMS §4.4: p ≲ m / 15)
    let events = y.iter().filter(|&&v| v == 1.0).count();
    let m = events.min(n - events);

    // 2. prespecified model: rcs(age, 4) * sex. Keep the Rcs (knots) for new data.
    let mut spec = ModelSpec::new();
    let a = spec.rcs("age", Rcs::from_data(&age, 4)?, &age);
    let s = spec.categorical("sex", &sex);
    spec.interaction(a, s);
    let design = spec.design(&[&age, &sex])?;
    let x = design.matrix();
    let p = x.ncols();
    println!("n = {n}, m = {m}, candidate d.f. p = {p}, m/15 = {:.1}", m as f64 / 15.0);

    // 3. fit (BinomialRegressor implements LikelihoodFit, so LR tests work)
    let yc = Col::from_fn(n, |i| y[i]);
    let fit = BinomialRegressor::logistic().compute_inference(true).build().fit(&x, &yc)?;
    let model_lr = model_lr_test(&fit).expect("model LR test");
    println!(
        "Model LR χ² = {:.2}  d.f. = {}  P = {:.3e}",
        model_lr.statistic, model_lr.df, model_lr.p_value
    );
    println!("heuristic shrinkage γ = {:.3}", heuristic_shrinkage(model_lr.statistic, p as f64));

    // 4. chunk tests: rms::anova (Wald, per predictor incl. its interactions and nonlinearity)
    let r = fit.result();
    let beta = full_beta(r);
    let vcov = fit.vcov().expect("compute_inference(true)");
    let (b_slopes, v_slopes) = drop_intercept(&beta, &vcov);
    let opts = AnovaOptions::default().distribution(WaldDistribution::ChiSquare);
    let table = anova(&b_slopes, &v_slopes, design.terms(), &opts)?;
    for row in &table.rows {
        let t = &row.test;
        println!(
            "{:<45} χ² = {:>7.2}  d.f. = {:>2}  P = {:.3e}  χ²−d.f. = {:>7.2}",
            row.label, t.chi_square, t.df, t.p_value, t.chi_square_minus_df()
        );
    }
    // LR test of all nonlinear columns: refit without them
    let nonlinear: Vec<usize> = design
        .terms()
        .terms()
        .iter()
        .flat_map(|t| t.columns.iter().zip(&t.nonlinear).filter(|(_, &nl)| nl).map(|(&c, _)| c))
        .collect();
    let keep: Vec<usize> = (0..p).filter(|j| !nonlinear.contains(j)).collect();
    let x_lin = Mat::from_fn(n, keep.len(), |i, j| x[(i, keep[j])]);
    let linear = BinomialRegressor::logistic().build().fit(&x_lin, &yc)?;
    let lr_nl = lr_test(&linear, &fit);
    println!("All nonlinear (LR)  χ² = {:.2}  d.f. = {}  P = {:.3e}", lr_nl.statistic, lr_nl.df, lr_nl.p_value);

    // 5. effects on an interpretable scale: OR for age Q1 → Q3 within each sex (summary.rms / contrast)
    let lim = &spec.predictors()[a].limits;
    let (q1, q3) = (lim.low_effect, lim.high_effect);
    for (code, label) in [(0.0, "female"), (1.0, "male")] {
        let c = spec.contrast(&beta, &vcov, true, &[(a, q3), (s, code)], &[(a, q1), (s, code)], Critical::Normal, 0.95)?;
        println!(
            "OR age {q1:.0}→{q3:.0} ({label}): {:.2} [{:.2}, {:.2}]",
            c.estimate.exp(), c.lower.exp(), c.upper.exp()
        );
    }

    // 6. discrimination and overall fit (rms::lrm stats)
    let lp: Vec<f64> = (0..n)
        .map(|i| beta[0] + (0..p).map(|j| x[(i, j)] * beta[j + 1]).sum::<f64>())
        .collect();
    let st = lrm_stats(&y, &lp, None, p as f64)?;
    println!("C = {:.3}  Dxy = {:.3}  R² (Nagelkerke) = {:.3}  Brier = {:.3}", st.c, st.dxy, st.r2, st.brier);

    // 7. validation: optimism-corrected indexes and calibration (rms::validate / calibrate).
    // validate_lrm refits this fixed design; data-driven steps need predab_resample.
    let boot = Resamples::bootstrap_seeded(n, 200, 42)?;
    let v = validate_lrm(&x, &y, &boot)?;
    println!("{:<10} {:>9} {:>9} {:>9}", "index", "apparent", "optimism", "corrected");
    for name in ["Dxy", "R2", "Slope", "B"] {
        let (orig, _, _, opt, corr, nb) = v.get(name).expect("index");
        println!("{name:<10} {orig:>9.3} {opt:>9.3} {corr:>9.3}  (B ok = {nb})");
    }
    let cal = calibrate_lrm(&x, &y, &boot, None)?;
    println!(
        "calibration: mean |error| = {:.3}, 0.9 quantile = {:.3}",
        cal.mean_abs_error, cal.quantile90_abs_error
    );
    Ok(())
}
