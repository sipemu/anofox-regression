//! Moment-heuristic AID: demand classification and per-observation anomaly flags.
//!
//! This is the lightweight AID used by the `anofox-statistics` DuckDB extension
//! (moved here in 0.5.17, issue #57). It complements, and does not replace, the
//! information-criterion classifier [`AidClassifier`](super::AidClassifier):
//!
//! * **demand type** — `intermittent` when the share of zeros among the finite
//!   values is at least [`AidOptions::intermittent_threshold`], else `regular`;
//! * **distribution** — chosen from moments, not by fitting:
//!   count data (all values non-negative integers) gives `poisson` when
//!   `variance / mean < 1.5` and `negative_binomial` otherwise (for intermittent
//!   count data: `negative_binomial` when `variance > 1.5 * mean`, else
//!   `geometric`); non-negative continuous data gives `lognormal` (skewness > 1),
//!   `gamma` (skewness > 0.5), `rectified_normal` (intermittent) or `normal`;
//!   anything with negative values gives `normal`;
//! * **anomalies** — per observation: `new_product` (zeros before the first
//!   non-zero), `obsolete_product` (zeros after the last non-zero), `stockout`
//!   (zeros strictly between the two), and `high_outlier` / `low_outlier` for
//!   non-zero values outside the z-score (`mean ± 3 sd`) or IQR
//!   (`Q1 - 1.5 IQR`, `Q3 + 1.5 IQR`) bounds, the lower bound floored at 0.
//!
//! # Conventions (kept bit-for-bit from the extension)
//!
//! * `mean` and `variance` (sample, `n - 1`) are over **all** finite values,
//!   zeros included.
//! * The skewness used for the continuous rule is the bias-corrected sample
//!   skewness of the **non-zero** values, standardized with the all-values
//!   `mean` and standard deviation above; it is 0 for fewer than 3 non-zero
//!   values or a zero standard deviation.
//! * IQR quartiles are order statistics at indices `n / 4` and `3n / 4` of the
//!   sorted finite values (no interpolation).
//! * A low outlier requires a strictly positive lower bound.
//! * Non-finite inputs are skipped in every statistic and get all-`false` flags;
//!   `n_observations` counts finite values only.
//!
//! # Example
//!
//! ```
//! use anofox_regression::solvers::aid::heuristic::{compute_aid, AidOptions};
//!
//! let y = [5.0, 6.0, 0.0, 0.0, 7.0, 8.0];
//! let r = compute_aid(&y, &AidOptions::default()).unwrap();
//! assert_eq!(r.demand_type, "intermittent");
//! assert_eq!(r.stockout_count, 2);
//! ```

use thiserror::Error;

/// Errors from the heuristic AID functions.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AidError {
    /// The input series is empty.
    #[error("Empty input: {field} cannot be empty")]
    EmptyInput {
        /// Name of the empty input.
        field: &'static str,
    },
    /// The input has no finite value.
    #[error("All rows filtered due to NULL/NaN values")]
    NoValidData,
}

/// Outlier detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutlierMethod {
    /// Outliers beyond `mean ± 3 * sd`.
    #[default]
    ZScore,
    /// Outliers beyond `Q1 - 1.5 * IQR` or `Q3 + 1.5 * IQR`.
    Iqr,
}

/// Options for [`compute_aid`] and [`compute_aid_anomalies`].
#[derive(Debug, Clone, PartialEq)]
pub struct AidOptions {
    /// Zero proportion at or above which demand is intermittent (default 0.3).
    pub intermittent_threshold: f64,
    /// Outlier detection method (default [`OutlierMethod::ZScore`]).
    pub outlier_method: OutlierMethod,
}

impl Default for AidOptions {
    fn default() -> Self {
        Self {
            intermittent_threshold: 0.3,
            outlier_method: OutlierMethod::ZScore,
        }
    }
}

/// Result of [`compute_aid`].
#[derive(Debug, Clone, PartialEq)]
pub struct AidResult {
    /// `"regular"` or `"intermittent"`.
    pub demand_type: String,
    /// Whether `zero_proportion >= intermittent_threshold`.
    pub is_intermittent: bool,
    /// Selected distribution: `poisson`, `negative_binomial`, `geometric`,
    /// `lognormal`, `gamma`, `rectified_normal` or `normal`.
    pub distribution: String,
    /// Mean of all finite values (zeros included).
    pub mean: f64,
    /// Sample variance (`n - 1`) of all finite values; 0 for a single value.
    pub variance: f64,
    /// Share of zeros among the finite values.
    pub zero_proportion: f64,
    /// Number of finite values.
    pub n_observations: usize,
    /// At least one stockout.
    pub has_stockouts: bool,
    /// At least one leading zero.
    pub is_new_product: bool,
    /// At least one trailing zero.
    pub is_obsolete_product: bool,
    /// Number of stockout observations (interior zeros).
    pub stockout_count: usize,
    /// Number of leading zeros.
    pub new_product_count: usize,
    /// Number of trailing zeros.
    pub obsolete_product_count: usize,
    /// Number of high outliers.
    pub high_outlier_count: usize,
    /// Number of low outliers.
    pub low_outlier_count: usize,
}

/// Per-observation anomaly flags.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AidAnomalyFlags {
    /// Zero strictly between the first and last non-zero value.
    pub stockout: bool,
    /// Zero before the first non-zero value.
    pub new_product: bool,
    /// Zero after the last non-zero value.
    pub obsolete_product: bool,
    /// Non-zero value above the upper outlier bound.
    pub high_outlier: bool,
    /// Non-zero value below a strictly positive lower outlier bound.
    pub low_outlier: bool,
}

/// Classify a demand series and count its anomalies.
///
/// `y` must be in time order. Errors when `y` is empty or has no finite value.
pub fn compute_aid(y: &[f64], options: &AidOptions) -> Result<AidResult, AidError> {
    if y.is_empty() {
        return Err(AidError::EmptyInput { field: "y" });
    }

    let valid_values: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    let n = valid_values.len();
    if n == 0 {
        return Err(AidError::NoValidData);
    }

    let sum: f64 = valid_values.iter().sum();
    let mean = sum / n as f64;
    let variance = if n > 1 {
        let sq_sum: f64 = valid_values.iter().map(|v| (v - mean).powi(2)).sum();
        sq_sum / (n - 1) as f64
    } else {
        0.0
    };

    let zero_count = valid_values.iter().filter(|&&v| v == 0.0).count();
    let zero_proportion = zero_count as f64 / n as f64;

    let is_intermittent = zero_proportion >= options.intermittent_threshold;
    let demand_type = if is_intermittent {
        "intermittent"
    } else {
        "regular"
    }
    .to_string();

    let flags = anomaly_flags(y, options);
    let count = |f: fn(&AidAnomalyFlags) -> bool| flags.iter().filter(|x| f(x)).count();
    let stockout_count = count(|f| f.stockout);
    let new_product_count = count(|f| f.new_product);
    let obsolete_product_count = count(|f| f.obsolete_product);
    let high_outlier_count = count(|f| f.high_outlier);
    let low_outlier_count = count(|f| f.low_outlier);

    let distribution = select_distribution(&valid_values, mean, variance, is_intermittent);

    Ok(AidResult {
        demand_type,
        is_intermittent,
        distribution,
        mean,
        variance,
        zero_proportion,
        n_observations: n,
        has_stockouts: stockout_count > 0,
        is_new_product: new_product_count > 0,
        is_obsolete_product: obsolete_product_count > 0,
        stockout_count,
        new_product_count,
        obsolete_product_count,
        high_outlier_count,
        low_outlier_count,
    })
}

/// Per-observation anomaly flags, one entry per element of `y` in input order.
///
/// Errors only when `y` is empty; an all-non-finite series yields all-`false`
/// flags.
pub fn compute_aid_anomalies(
    y: &[f64],
    options: &AidOptions,
) -> Result<Vec<AidAnomalyFlags>, AidError> {
    if y.is_empty() {
        return Err(AidError::EmptyInput { field: "y" });
    }
    Ok(anomaly_flags(y, options))
}

fn outlier_bounds(valid_values: &[f64], method: OutlierMethod) -> (f64, f64) {
    let n = valid_values.len();
    let mean: f64 = valid_values.iter().sum::<f64>() / n as f64;
    let std_dev = if n > 1 {
        let variance: f64 =
            valid_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    match method {
        OutlierMethod::ZScore => (mean + 3.0 * std_dev, (mean - 3.0 * std_dev).max(0.0)),
        OutlierMethod::Iqr => {
            let mut sorted = valid_values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1 = sorted.get(n / 4).copied().unwrap_or(0.0);
            let q3 = sorted.get((3 * n) / 4).copied().unwrap_or(mean);
            let iqr = q3 - q1;
            (q3 + 1.5 * iqr, (q1 - 1.5 * iqr).max(0.0))
        }
    }
}

fn anomaly_flags(y: &[f64], options: &AidOptions) -> Vec<AidAnomalyFlags> {
    let valid_values: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    if valid_values.is_empty() {
        return vec![AidAnomalyFlags::default(); y.len()];
    }

    let (high_threshold, low_threshold) = outlier_bounds(&valid_values, options.outlier_method);

    let is_nonzero = |v: &f64| v.is_finite() && *v != 0.0;
    let first_nonzero = y.iter().position(is_nonzero);
    let last_nonzero = y.iter().rposition(is_nonzero);

    y.iter()
        .enumerate()
        .map(|(i, &val)| {
            if !val.is_finite() {
                return AidAnomalyFlags::default();
            }
            let is_zero = val == 0.0;
            let (new_product, obsolete_product, stockout) = match (first_nonzero, last_nonzero) {
                (Some(first), Some(last)) => (
                    is_zero && i < first,
                    is_zero && i > last,
                    is_zero && i > first && i < last,
                ),
                _ => (false, false, false),
            };
            AidAnomalyFlags {
                stockout,
                new_product,
                obsolete_product,
                high_outlier: !is_zero && val > high_threshold,
                low_outlier: !is_zero && val < low_threshold && low_threshold > 0.0,
            }
        })
        .collect()
}

fn select_distribution(values: &[f64], mean: f64, variance: f64, is_intermittent: bool) -> String {
    let is_count_data = values.iter().all(|&v| v >= 0.0 && v == v.floor());

    let name = if is_count_data {
        if is_intermittent {
            if variance > mean * 1.5 {
                "negative_binomial"
            } else {
                "geometric"
            }
        } else {
            let dispersion_ratio = if mean > 0.0 { variance / mean } else { 1.0 };
            if dispersion_ratio < 1.5 {
                "poisson"
            } else {
                "negative_binomial"
            }
        }
    } else {
        let nonzero_values: Vec<f64> = values.iter().copied().filter(|&v| v > 0.0).collect();
        if nonzero_values.is_empty() || !values.iter().all(|&v| v >= 0.0) {
            "normal"
        } else {
            let skewness = skewness(&nonzero_values, mean, variance.sqrt());
            if skewness > 1.0 {
                "lognormal"
            } else if skewness > 0.5 {
                "gamma"
            } else if is_intermittent {
                "rectified_normal"
            } else {
                "normal"
            }
        }
    };
    name.to_string()
}

/// Bias-corrected sample skewness of `values`, standardized by the supplied
/// `mean` and `std_dev`.
fn skewness(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    if values.len() < 3 || std_dev == 0.0 {
        return 0.0;
    }
    let n = values.len() as f64;
    let m3: f64 = values.iter().map(|&v| ((v - mean) / std_dev).powi(3)).sum();
    (m3 / n) * (n * (n - 1.0)).sqrt() / (n - 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_demand() {
        let y = vec![10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 14.0, 10.0, 13.0, 11.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert_eq!(r.demand_type, "regular");
        assert!(!r.is_intermittent);
        assert_eq!(r.n_observations, 10);
        assert!(r.zero_proportion.abs() < 0.001);
    }

    #[test]
    fn intermittent_demand() {
        let y = vec![0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 3.0, 0.0, 0.0, 6.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert_eq!(r.demand_type, "intermittent");
        assert!(r.is_intermittent);
        assert!((r.zero_proportion - 0.6).abs() < 0.001);
    }

    #[test]
    fn stockout_detection() {
        let y = vec![5.0, 6.0, 0.0, 0.0, 7.0, 8.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert!(r.has_stockouts);
        assert_eq!(r.stockout_count, 2);
    }

    #[test]
    fn new_product_detection() {
        let y = vec![0.0, 0.0, 0.0, 5.0, 6.0, 7.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert!(r.is_new_product);
        assert_eq!(r.new_product_count, 3);
    }

    #[test]
    fn obsolete_product_detection() {
        let y = vec![5.0, 6.0, 7.0, 0.0, 0.0, 0.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert!(r.is_obsolete_product);
        assert_eq!(r.obsolete_product_count, 3);
    }

    #[test]
    fn zscore_high_outlier() {
        let mut y = vec![10.0; 20];
        y.push(100.0);
        let opts = AidOptions {
            outlier_method: OutlierMethod::ZScore,
            ..Default::default()
        };
        let r = compute_aid(&y, &opts).unwrap();
        assert!(r.high_outlier_count > 0);
    }

    #[test]
    fn anomaly_flags_by_position() {
        let y = vec![0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0];
        let f = compute_aid_anomalies(&y, &AidOptions::default()).unwrap();
        assert_eq!(f.len(), 7);
        assert!(f[0].new_product && f[1].new_product);
        assert!(f[3].stockout);
        assert!(f[5].obsolete_product && f[6].obsolete_product);
    }

    #[test]
    fn count_data_distribution() {
        let y = vec![2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 5.0, 2.0, 3.0, 4.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert!(r.distribution == "poisson" || r.distribution == "negative_binomial");
    }

    #[test]
    fn empty_input() {
        assert_eq!(
            compute_aid(&[], &AidOptions::default()),
            Err(AidError::EmptyInput { field: "y" })
        );
        assert!(compute_aid_anomalies(&[], &AidOptions::default()).is_err());
    }

    #[test]
    fn all_nan_input() {
        let y = [f64::NAN, f64::INFINITY];
        assert_eq!(
            compute_aid(&y, &AidOptions::default()),
            Err(AidError::NoValidData)
        );
        let f = compute_aid_anomalies(&y, &AidOptions::default()).unwrap();
        assert_eq!(f, vec![AidAnomalyFlags::default(); 2]);
    }

    #[test]
    fn hand_computed_moments_and_distribution() {
        // mean = 3, variance = 32 / 4 = 8 (sample), count data, 2/5 zeros => intermittent,
        // variance > 1.5 * mean => negative_binomial.
        let y = [0.0, 5.0, 0.0, 6.0, 4.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert_eq!(r.mean, 3.0);
        assert_eq!(r.variance, 8.0);
        assert_eq!(r.zero_proportion, 0.4);
        assert_eq!(r.distribution, "negative_binomial");
        assert_eq!(r.new_product_count, 1);
        assert_eq!(r.stockout_count, 1);
        assert_eq!(r.obsolete_product_count, 0);
    }

    #[test]
    fn iqr_bounds_use_order_statistics() {
        // sorted = [1,2,3,4,5,6,7,100]; q1 = sorted[2] = 3, q3 = sorted[6] = 7,
        // IQR = 4 => bounds (13, max(0, -3) = 0). Only 100 is a high outlier.
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0];
        let opts = AidOptions {
            outlier_method: OutlierMethod::Iqr,
            ..Default::default()
        };
        let f = compute_aid_anomalies(&y, &opts).unwrap();
        let high: Vec<usize> = (0..y.len()).filter(|&i| f[i].high_outlier).collect();
        assert_eq!(high, vec![7]);
        assert!(f.iter().all(|x| !x.low_outlier));
    }

    #[test]
    fn continuous_distributions() {
        // Strongly right-skewed positive data => lognormal.
        let y = [1.1, 1.2, 1.3, 1.25, 1.15, 1.2, 9.5];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert_eq!(r.distribution, "lognormal");
        // Negative values => normal.
        let y = [-1.5, 2.25, 0.5, 1.75];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert_eq!(r.distribution, "normal");
    }
}
