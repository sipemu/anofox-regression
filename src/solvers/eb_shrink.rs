//! Empirical-Bayes shrinkage of per-group estimates toward a common mean.
//!
//! Moved from the `anofox-statistics` DuckDB extension in 0.5.17 (issue #58).
//! Rather than fitting a hierarchical model, take estimates that already exist —
//! one per group, each with a standard error — and shrink them toward their
//! precision-weighted mean by an amount the data itself determines.
//!
//! The model is the random-effects meta-analysis one:
//!
//! ```text
//!   theta_g ~ N(mu, tau^2)        (between-group variation)
//!   est_g   ~ N(theta_g, se_g^2)  (within-group sampling error)
//! ```
//!
//! `tau^2` is estimated by the DerSimonian–Laird moment estimator (or fixed, or
//! zero), after which each group's posterior mean is the precision-weighted blend
//! of its own estimate and the pooled mean. A group measured precisely (small
//! `se_g`) barely moves; a group measured poorly is pulled most of the way to `mu`.
//!
//! The inputs are estimates, not data, so this composes with any per-group fit
//! that yields an estimate and a standard error.
//!
//! # Agreement with R
//!
//! `mu`, `mu_se`, `tau_squared`, `q` and the shrunken estimates match
//! `metafor::rma(yi, sei = se, method = "DL")` and `blup()` (see
//! `tests/r_validation_eb_shrink.rs`). Two definitions to be aware of:
//!
//! * `shrunken_se` is the conditional posterior SD `sqrt(1 / (1/se^2 + 1/tau^2))`,
//!   which treats `mu` as known; `metafor::blup()`'s `se` additionally carries the
//!   uncertainty of `mu` and is therefore slightly larger.
//! * `i_squared` is always Higgins–Thompson `max(0, (Q - df) / Q)`. That equals
//!   metafor's `I2 / 100` for the DL and fixed-effect fits, but not when `tau^2` is
//!   supplied via [`EbShrinkOptions::tau_squared`] (metafor then derives `I^2` from
//!   the supplied `tau^2`).
//!
//! # Example
//!
//! ```
//! use anofox_regression::solvers::eb_shrink::{eb_shrink, EbShrinkOptions};
//!
//! let est = [0.10, 0.30, 0.35, 0.65, 1.00];
//! let se = [0.30, 0.10, 0.50, 0.20, 0.40];
//! let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
//! // metafor::rma(est, sei = se, method = "DL")
//! assert!((r.mu - 0.4223980166707431).abs() < 1e-12);
//! assert!((r.tau_squared - 0.02394587965990844).abs() < 1e-12);
//! ```

use thiserror::Error;

/// Errors from [`eb_shrink`].
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EbShrinkError {
    /// No estimates were supplied.
    #[error("Empty input: {field} cannot be empty")]
    EmptyInput {
        /// Name of the empty input.
        field: &'static str,
    },
    /// `estimates` and `standard_errors` have different lengths.
    #[error("Dimension mismatch: {estimates} estimates but {standard_errors} standard errors")]
    DimensionMismatch {
        /// Length of `estimates`.
        estimates: usize,
        /// Length of `standard_errors`.
        standard_errors: usize,
    },
    /// Fewer than two usable groups (finite estimate, finite positive SE).
    #[error("Insufficient data: {usable} usable groups, need at least 2")]
    InsufficientData {
        /// Number of usable groups.
        usable: usize,
    },
    /// An option outside its domain.
    #[error("Invalid value for {field}: {message}")]
    InvalidValue {
        /// Name of the offending option.
        field: &'static str,
        /// What is wrong with it.
        message: String,
    },
}

/// How `tau^2` (the between-group variance) is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TauMethod {
    /// DerSimonian–Laird moment estimator, `max(0, (Q - df) / C)` with
    /// `C = sum(w) - sum(w^2) / sum(w)`. What `metafor::rma(method = "DL")`
    /// computes.
    #[default]
    DerSimonianLaird,
    /// `tau^2 = 0`: complete pooling. Every group collapses onto the
    /// precision-weighted (fixed-effect) mean.
    None,
}

/// Options for [`eb_shrink`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EbShrinkOptions {
    /// Estimator for `tau^2`.
    pub method: TauMethod,
    /// A fixed `tau^2` instead of estimating one. Overrides `method` when set;
    /// must be finite and non-negative.
    pub tau_squared: Option<f64>,
}

/// One group's shrunken estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShrunkenGroup {
    /// The estimate as supplied.
    pub estimate: f64,
    /// The standard error as supplied.
    pub se: f64,
    /// Posterior mean: the shrunken estimate.
    pub shrunken: f64,
    /// Posterior standard deviation (conditional on `mu`).
    pub shrunken_se: f64,
    /// Shrinkage weight in `[0, 1]`: the share of the group's own estimate that
    /// survives, `tau^2 / (tau^2 + se^2)`. 1 means untouched, 0 means fully pooled.
    pub weight: f64,
}

/// Result of [`eb_shrink`].
#[derive(Debug, Clone, PartialEq)]
pub struct EbShrinkResult {
    /// Precision-weighted (random-effects) pooled mean.
    pub mu: f64,
    /// Standard error of the pooled mean, `1 / sqrt(sum(1 / (se^2 + tau^2)))`.
    pub mu_se: f64,
    /// Between-group variance used.
    pub tau_squared: f64,
    /// Higgins–Thompson `I^2 = max(0, (Q - df) / Q)`, in `[0, 1]`.
    pub i_squared: f64,
    /// Cochran's Q heterogeneity statistic about the fixed-effect mean.
    pub q: f64,
    /// Number of usable groups.
    pub n_groups: usize,
    /// One entry per input row, in input order.
    pub groups: Vec<ShrunkenGroup>,
}

fn usable(est: f64, se: f64) -> bool {
    est.is_finite() && se.is_finite() && se > 0.0
}

/// Shrink per-group estimates toward their common mean.
///
/// `estimates` and `standard_errors` are parallel arrays, one entry per group.
/// Rows with a non-finite estimate, or a non-positive or non-finite standard
/// error, are dropped from the `tau^2` and `mu` calculations but still appear in
/// `groups` with `NaN` outputs, so the result stays aligned with the input.
///
/// # Errors
///
/// [`EbShrinkError::EmptyInput`], [`EbShrinkError::DimensionMismatch`],
/// [`EbShrinkError::InsufficientData`] (fewer than two usable rows) and
/// [`EbShrinkError::InvalidValue`] (negative or non-finite fixed `tau^2`).
pub fn eb_shrink(
    estimates: &[f64],
    standard_errors: &[f64],
    options: &EbShrinkOptions,
) -> Result<EbShrinkResult, EbShrinkError> {
    if estimates.is_empty() {
        return Err(EbShrinkError::EmptyInput { field: "estimate" });
    }
    if estimates.len() != standard_errors.len() {
        return Err(EbShrinkError::DimensionMismatch {
            estimates: estimates.len(),
            standard_errors: standard_errors.len(),
        });
    }

    let rows: Vec<usize> = (0..estimates.len())
        .filter(|&i| usable(estimates[i], standard_errors[i]))
        .collect();
    if rows.len() < 2 {
        return Err(EbShrinkError::InsufficientData { usable: rows.len() });
    }

    // Fixed-effect (inverse-variance) weights.
    let w: Vec<f64> = rows
        .iter()
        .map(|&i| 1.0 / (standard_errors[i] * standard_errors[i]))
        .collect();
    let sum_w: f64 = w.iter().sum();
    let sum_wy: f64 = rows
        .iter()
        .zip(w.iter())
        .map(|(&i, &wi)| wi * estimates[i])
        .sum();
    let fixed_mean = sum_wy / sum_w;

    // Cochran's Q about the fixed-effect mean.
    let q: f64 = rows
        .iter()
        .zip(w.iter())
        .map(|(&i, &wi)| wi * (estimates[i] - fixed_mean).powi(2))
        .sum();

    let df = rows.len() as f64 - 1.0;

    let tau_squared = match options.tau_squared {
        Some(v) => {
            if !v.is_finite() || v < 0.0 {
                return Err(EbShrinkError::InvalidValue {
                    field: "tau_squared",
                    message: "tau_squared must be finite and non-negative".to_string(),
                });
            }
            v
        }
        None => match options.method {
            TauMethod::None => 0.0,
            TauMethod::DerSimonianLaird => {
                let sum_w2: f64 = w.iter().map(|x| x * x).sum();
                let c = sum_w - sum_w2 / sum_w;
                if c > 0.0 {
                    ((q - df) / c).max(0.0)
                } else {
                    0.0
                }
            }
        },
    };

    // Random-effects weights and pooled mean.
    let wr: Vec<f64> = rows
        .iter()
        .map(|&i| 1.0 / (standard_errors[i] * standard_errors[i] + tau_squared))
        .collect();
    let sum_wr: f64 = wr.iter().sum();
    let mu: f64 = rows
        .iter()
        .zip(wr.iter())
        .map(|(&i, &wi)| wi * estimates[i])
        .sum::<f64>()
        / sum_wr;
    let mu_se = (1.0 / sum_wr).sqrt();

    let i_squared = if q > df && q > 0.0 {
        ((q - df) / q).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let groups = estimates
        .iter()
        .zip(standard_errors.iter())
        .map(|(&est, &se)| {
            if !usable(est, se) {
                return ShrunkenGroup {
                    estimate: est,
                    se,
                    shrunken: f64::NAN,
                    shrunken_se: f64::NAN,
                    weight: f64::NAN,
                };
            }
            // Posterior precision is the sum of the two precisions. With tau^2 = 0
            // the group's own estimate carries no weight and everything collapses
            // onto the pooled mean.
            let (shrunken, shrunken_se, weight) = if tau_squared > 0.0 {
                let prec_within = 1.0 / (se * se);
                let prec_between = 1.0 / tau_squared;
                let post_prec = prec_within + prec_between;
                let weight = prec_within / post_prec;
                (
                    weight * est + (1.0 - weight) * mu,
                    (1.0 / post_prec).sqrt(),
                    weight,
                )
            } else {
                (mu, mu_se, 0.0)
            };
            ShrunkenGroup {
                estimate: est,
                se,
                shrunken,
                shrunken_se,
                weight,
            }
        })
        .collect();

    Ok(EbShrinkResult {
        mu,
        mu_se,
        tau_squared,
        i_squared,
        q,
        n_groups: rows.len(),
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.10, 0.30, 0.35, 0.65, 1.00],
            vec![0.30, 0.10, 0.50, 0.20, 0.40],
        )
    }

    #[test]
    fn matches_the_dersimonian_laird_formulas() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();

        let w: Vec<f64> = se.iter().map(|s| 1.0 / (s * s)).collect();
        let sw: f64 = w.iter().sum();
        let fixed: f64 = w.iter().zip(est.iter()).map(|(a, b)| a * b).sum::<f64>() / sw;
        let q: f64 = w
            .iter()
            .zip(est.iter())
            .map(|(a, b)| a * (b - fixed).powi(2))
            .sum();
        let sw2: f64 = w.iter().map(|a| a * a).sum();
        let c = sw - sw2 / sw;
        let tau2 = ((q - 4.0) / c).max(0.0);

        assert!((r.q - q).abs() < 1e-12, "Q {} vs {q}", r.q);
        assert!((r.tau_squared - tau2).abs() < 1e-12);

        let wr: Vec<f64> = se.iter().map(|s| 1.0 / (s * s + tau2)).collect();
        let swr: f64 = wr.iter().sum();
        let mu: f64 = wr.iter().zip(est.iter()).map(|(a, b)| a * b).sum::<f64>() / swr;
        assert!((r.mu - mu).abs() < 1e-12, "mu {} vs {mu}", r.mu);
        assert!((r.mu_se - (1.0 / swr).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn every_shrunken_value_lies_between_its_estimate_and_the_mean() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        for g in &r.groups {
            let lo = g.estimate.min(r.mu);
            let hi = g.estimate.max(r.mu);
            assert!(g.shrunken >= lo - 1e-12 && g.shrunken <= hi + 1e-12);
        }
    }

    #[test]
    fn noisier_groups_are_pulled_harder() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        let mut pairs: Vec<(f64, f64)> = r.groups.iter().map(|g| (g.se, g.weight)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for pair in pairs.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
    }

    #[test]
    fn shrinkage_always_reduces_the_standard_error() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        for g in &r.groups {
            assert!(g.shrunken_se <= g.se + 1e-12);
        }
    }

    #[test]
    fn homogeneous_groups_collapse_to_complete_pooling() {
        let est = vec![0.5; 6];
        let se = vec![0.2, 0.3, 0.1, 0.4, 0.25, 0.35];
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        assert_eq!(r.tau_squared, 0.0);
        assert_eq!(r.i_squared, 0.0);
        for g in &r.groups {
            assert!((g.shrunken - r.mu).abs() < 1e-12);
            assert_eq!(g.weight, 0.0);
        }
    }

    #[test]
    fn a_large_tau_leaves_the_estimates_essentially_alone() {
        let (est, se) = fixture();
        let opts = EbShrinkOptions {
            tau_squared: Some(1e6),
            ..Default::default()
        };
        let r = eb_shrink(&est, &se, &opts).unwrap();
        for g in &r.groups {
            assert!((g.shrunken - g.estimate).abs() < 1e-3);
            assert!(g.weight > 0.999);
        }
    }

    #[test]
    fn heterogeneity_shows_up_in_i_squared() {
        let est = vec![0.0, 1.0, 2.0, 3.0];
        let se = vec![0.05; 4];
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        assert!(r.i_squared > 0.95);
        assert!(r.tau_squared > 0.5);
    }

    #[test]
    fn the_none_method_pools_completely() {
        let (est, se) = fixture();
        let opts = EbShrinkOptions {
            method: TauMethod::None,
            ..Default::default()
        };
        let r = eb_shrink(&est, &se, &opts).unwrap();
        assert_eq!(r.tau_squared, 0.0);
        for g in &r.groups {
            assert!((g.shrunken - r.mu).abs() < 1e-12);
        }
    }

    #[test]
    fn unusable_rows_stay_in_place_as_nan() {
        let est = vec![0.1, f64::NAN, 0.3, 0.5];
        let se = vec![0.2, 0.2, -1.0, 0.3];
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        assert_eq!(r.n_groups, 2);
        assert_eq!(r.groups.len(), 4);
        assert!(r.groups[1].shrunken.is_nan());
        assert!(r.groups[2].shrunken.is_nan());
        assert!(r.groups[0].shrunken.is_finite());
        assert!(r.groups[3].shrunken.is_finite());
    }

    #[test]
    fn a_single_group_is_rejected() {
        let err = eb_shrink(&[0.5], &[0.2], &EbShrinkOptions::default());
        assert_eq!(err, Err(EbShrinkError::InsufficientData { usable: 1 }));
    }

    #[test]
    fn empty_input_is_rejected() {
        let err = eb_shrink(&[], &[], &EbShrinkOptions::default());
        assert!(matches!(err, Err(EbShrinkError::EmptyInput { .. })));
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let err = eb_shrink(&[0.1, 0.2], &[0.1], &EbShrinkOptions::default());
        assert_eq!(
            err,
            Err(EbShrinkError::DimensionMismatch {
                estimates: 2,
                standard_errors: 1
            })
        );
    }

    #[test]
    fn a_negative_or_non_finite_fixed_tau_is_rejected() {
        let (est, se) = fixture();
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let opts = EbShrinkOptions {
                tau_squared: Some(bad),
                ..Default::default()
            };
            assert!(matches!(
                eb_shrink(&est, &se, &opts),
                Err(EbShrinkError::InvalidValue { .. })
            ));
        }
    }

    #[test]
    fn the_pooled_mean_sits_inside_the_range_of_the_estimates() {
        let (est, se) = fixture();
        let r = eb_shrink(&est, &se, &EbShrinkOptions::default()).unwrap();
        let lo = est.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = est.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(r.mu > lo && r.mu < hi);
    }
}
