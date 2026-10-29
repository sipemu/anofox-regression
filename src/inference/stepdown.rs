//! Fast backward step-down (RMS §5.5.2), as `rms::fastbw`.
//!
//! Uses only `β̂` and `V̂` of the full fit (Lawless & Singhal): at each step
//! the term with the smallest Wald criterion (χ² − k·d.f. for
//! [`StepRule::Aic`], the largest p-value for [`StepRule::P`]) is a
//! candidate for deletion; it is deleted while the residual χ² of all
//! deleted parameters ([`StepType::Residual`]) or its own χ²
//! ([`StepType::Individual`]) satisfies the rule. The coefficients of the
//! kept parameters are then approximated by `β_in − V_io V_oo⁻¹ β_out`
//! without refitting.
//!
//! Like `fastbw`, the procedure ignores term hierarchy (an interaction may
//! stay after its main effect is deleted).

use super::anova::{wald_chunk, WaldDistribution};
use crate::design::TermMap;
use faer::linalg::solvers::Solve;
use faer::{Col, Mat};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Stopping rule of [`fastbw`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepRule {
    /// Delete while `χ² − k·d.f. ≤ threshold` (rms: `k = 2`, `aics = 0`).
    Aic {
        /// Penalty per d.f. (2 for AIC).
        k: f64,
        /// Threshold (`aics`).
        threshold: f64,
    },
    /// Delete while the p-value exceeds `sls`.
    P {
        /// Significance level to stay (rms default 0.05).
        sls: f64,
    },
}

/// What the stopping rule is applied to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepType {
    /// The pooled test of all deleted parameters (rms default).
    Residual,
    /// The test of the term being deleted.
    Individual,
}

/// One deletion of [`fastbw`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepDeletion {
    /// Deleted term.
    pub term: usize,
    /// Its Wald χ² at this step.
    pub chisq: f64,
    /// Its d.f.
    pub df: usize,
    /// Its p-value.
    pub p_value: f64,
    /// Residual χ² of all deleted parameters so far.
    pub residual_chisq: f64,
    /// Residual d.f.
    pub residual_df: usize,
    /// Residual p-value.
    pub residual_p: f64,
    /// `residual χ² − k·residual d.f.`
    pub aic: f64,
}

/// Result of [`fastbw`].
#[derive(Debug, Clone, PartialEq)]
pub struct StepDown {
    /// Deletions in order.
    pub deleted: Vec<StepDeletion>,
    /// Terms kept.
    pub kept_terms: Vec<usize>,
    /// Approximate coefficients of the reduced model (full length, zero for
    /// deleted parameters).
    pub coefficients: Vec<f64>,
}

fn chisq_sf(x: f64, df: usize) -> f64 {
    ChiSquared::new(df as f64).map_or(f64::NAN, |d| d.sf(x))
}

/// Fast backward step-down on the full-model `beta` / `vcov`, whose first
/// `n_intercepts` entries are intercepts (never deleted) followed by the
/// slope columns of `terms`.
pub fn fastbw(
    beta: &Col<f64>,
    vcov: &Mat<f64>,
    n_intercepts: usize,
    terms: &TermMap,
    rule: StepRule,
    kind: StepType,
) -> StepDown {
    let pt = beta.nrows();
    let k_aic = match rule {
        StepRule::Aic { k, .. } => k,
        StepRule::P { .. } => 2.0,
    };
    let deletes = |chisq: f64, df: usize| match rule {
        StepRule::Aic { k, threshold } => chisq - k * df as f64 <= threshold,
        StepRule::P { sls } => chisq_sf(chisq, df) > sls,
    };
    let cols = |t: usize| -> Vec<usize> {
        terms.terms()[t]
            .columns
            .iter()
            .map(|c| c + n_intercepts)
            .collect()
    };
    let mut factors_in: Vec<usize> = (0..terms.terms().len()).collect();
    let mut out_params: Vec<usize> = Vec::new();
    let mut deleted = Vec::new();
    let mut coefficients: Vec<f64> = (0..pt).map(|j| beta[j]).collect();
    // Current approximate coefficients / covariance of the kept parameters.
    let (mut cur_b, mut cur_v, mut cur_idx) =
        (beta.clone(), vcov.clone(), (0..pt).collect::<Vec<_>>());
    while !factors_in.is_empty() {
        // Candidate: smallest criterion among the remaining terms.
        let mut best: Option<(usize, f64, f64, usize)> = None;
        for &t in &factors_in {
            let q: Vec<usize> = cols(t)
                .iter()
                .map(|c| cur_idx.iter().position(|v| v == c).expect("kept"))
                .collect();
            let chisq = wald_chunk(&cur_b, &cur_v, &q, WaldDistribution::ChiSquare)
                .map_or(f64::INFINITY, |w| w.chi_square);
            let df = q.len();
            let crit = match rule {
                StepRule::Aic { k, .. } => chisq - k * df as f64,
                StepRule::P { .. } => 1.0 - chisq_sf(chisq, df),
            };
            if best.is_none_or(|b| crit < b.1) {
                best = Some((t, crit, chisq, df));
            }
        }
        let (t, _, chisq, df) = best.expect("a remaining term");
        let mut out = out_params.clone();
        out.extend(cols(t));
        out.sort_unstable();
        let resid = wald_chunk(beta, vcov, &out, WaldDistribution::ChiSquare)
            .map_or(f64::NAN, |w| w.chi_square);
        let ok = match kind {
            StepType::Residual => deletes(resid, out.len()),
            StepType::Individual => deletes(chisq, df),
        };
        if !ok {
            break;
        }
        factors_in.retain(|&f| f != t);
        out_params = out;
        deleted.push(StepDeletion {
            term: t,
            chisq,
            df,
            p_value: chisq_sf(chisq, df),
            residual_chisq: resid,
            residual_df: out_params.len(),
            residual_p: chisq_sf(resid, out_params.len()),
            aic: resid - k_aic * out_params.len() as f64,
        });
        // β_in − V_io V_oo⁻¹ β_out and V_ii − V_io V_oo⁻¹ V_oi from the full fit.
        let kept: Vec<usize> = (0..pt).filter(|j| !out_params.contains(j)).collect();
        let o = out_params.len();
        let voo = Mat::from_fn(o, o, |a, b| vcov[(out_params[a], out_params[b])]);
        let vio = Mat::from_fn(kept.len(), o, |a, b| vcov[(kept[a], out_params[b])]);
        let bo = Col::from_fn(o, |a| beta[out_params[a]]);
        let lu = voo.partial_piv_lu();
        let w = lu.solve(vio.transpose()).transpose().to_owned();
        let shift = &w * &bo;
        cur_b = Col::from_fn(kept.len(), |a| beta[kept[a]] - shift[a]);
        let corr = &w * vio.transpose();
        cur_v = Mat::from_fn(kept.len(), kept.len(), |a, b| {
            vcov[(kept[a], kept[b])] - corr[(a, b)]
        });
        coefficients = vec![0.0; pt];
        for (a, &j) in kept.iter().enumerate() {
            coefficients[j] = cur_b[a];
        }
        cur_idx = kept;
    }
    StepDown {
        deleted,
        kept_terms: factors_in,
        coefficients,
    }
}
