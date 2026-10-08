//! Issue #57: `solvers::aid::heuristic` must reproduce the `anofox-statistics`
//! extension's AID (`crates/anofox-stats-core/src/models/aid.rs`) bit-for-bit.
//!
//! `ext` below is a verbatim copy of the extension's algorithm (only its error
//! and option/result types are inlined). Every output field is compared with
//! exact equality (`to_bits` for floats) over a few thousand generated series
//! covering zeros, NaN/inf, fractional, negative and constant data, for both
//! outlier methods and several thresholds.

use anofox_regression::solvers::aid::heuristic::{
    compute_aid, compute_aid_anomalies, AidError, AidOptions, OutlierMethod,
};

#[allow(clippy::all)]
mod ext {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum OutlierMethod {
        ZScore,
        Iqr,
    }
    pub struct AidOptions {
        pub intermittent_threshold: f64,
        pub outlier_method: OutlierMethod,
    }
    #[derive(Debug, Clone)]
    pub struct AidResult {
        pub demand_type: String,
        pub is_intermittent: bool,
        pub distribution: String,
        pub mean: f64,
        pub variance: f64,
        pub zero_proportion: f64,
        pub n_observations: usize,
        pub has_stockouts: bool,
        pub is_new_product: bool,
        pub is_obsolete_product: bool,
        pub stockout_count: usize,
        pub new_product_count: usize,
        pub obsolete_product_count: usize,
        pub high_outlier_count: usize,
        pub low_outlier_count: usize,
    }
    #[derive(Debug, Clone, Default)]
    pub struct AidAnomalyFlags {
        pub stockout: bool,
        pub new_product: bool,
        pub obsolete_product: bool,
        pub high_outlier: bool,
        pub low_outlier: bool,
    }

    pub fn compute_aid(y: &[f64], options: &AidOptions) -> Result<AidResult, &'static str> {
        if y.is_empty() {
            return Err("empty");
        }
        let valid_values: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
        let n = valid_values.len();
        if n == 0 {
            return Err("novalid");
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
            "intermittent".to_string()
        } else {
            "regular".to_string()
        };
        let anomaly_flags = compute_anomaly_flags(y, options);
        let mut stockout_count = 0usize;
        let mut new_product_count = 0usize;
        let mut obsolete_product_count = 0usize;
        let mut high_outlier_count = 0usize;
        let mut low_outlier_count = 0usize;
        for flags in &anomaly_flags {
            if flags.stockout {
                stockout_count += 1;
            }
            if flags.new_product {
                new_product_count += 1;
            }
            if flags.obsolete_product {
                obsolete_product_count += 1;
            }
            if flags.high_outlier {
                high_outlier_count += 1;
            }
            if flags.low_outlier {
                low_outlier_count += 1;
            }
        }
        let has_stockouts = stockout_count > 0;
        let is_new_product = new_product_count > 0;
        let is_obsolete_product = obsolete_product_count > 0;
        let distribution = select_distribution(&valid_values, mean, variance, is_intermittent);
        Ok(AidResult {
            demand_type,
            is_intermittent,
            distribution,
            mean,
            variance,
            zero_proportion,
            n_observations: n,
            has_stockouts,
            is_new_product,
            is_obsolete_product,
            stockout_count,
            new_product_count,
            obsolete_product_count,
            high_outlier_count,
            low_outlier_count,
        })
    }

    pub fn compute_anomaly_flags(y: &[f64], options: &AidOptions) -> Vec<AidAnomalyFlags> {
        let n = y.len();
        if n == 0 {
            return vec![];
        }
        let valid_values: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
        if valid_values.is_empty() {
            return y.iter().map(|_| AidAnomalyFlags::default()).collect();
        }
        let mean: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
        let std_dev = if valid_values.len() > 1 {
            let variance: f64 = valid_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (valid_values.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let (high_threshold, low_threshold) = match options.outlier_method {
            OutlierMethod::ZScore => {
                let high = mean + 3.0 * std_dev;
                let low = (mean - 3.0 * std_dev).max(0.0);
                (high, low)
            }
            OutlierMethod::Iqr => {
                let mut sorted = valid_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q1_idx = sorted.len() / 4;
                let q3_idx = (3 * sorted.len()) / 4;
                let q1 = sorted.get(q1_idx).copied().unwrap_or(0.0);
                let q3 = sorted.get(q3_idx).copied().unwrap_or(mean);
                let iqr = q3 - q1;
                let high = q3 + 1.5 * iqr;
                let low = (q1 - 1.5 * iqr).max(0.0);
                (high, low)
            }
        };
        let mut first_nonzero_idx: Option<usize> = None;
        for (i, &val) in y.iter().enumerate() {
            if val.is_finite() && val != 0.0 {
                first_nonzero_idx = Some(i);
                break;
            }
        }
        let mut last_nonzero_idx: Option<usize> = None;
        for (i, &val) in y.iter().enumerate().rev() {
            if val.is_finite() && val != 0.0 {
                last_nonzero_idx = Some(i);
                break;
            }
        }
        y.iter()
            .enumerate()
            .map(|(i, &val)| {
                if !val.is_finite() {
                    return AidAnomalyFlags::default();
                }
                let is_zero = val == 0.0;
                let new_product = match first_nonzero_idx {
                    Some(first_idx) => is_zero && i < first_idx,
                    None => false,
                };
                let obsolete_product = match last_nonzero_idx {
                    Some(last_idx) => is_zero && i > last_idx,
                    None => false,
                };
                let stockout = match (first_nonzero_idx, last_nonzero_idx) {
                    (Some(first_idx), Some(last_idx)) => is_zero && i > first_idx && i < last_idx,
                    _ => false,
                };
                let high_outlier = !is_zero && val > high_threshold;
                let low_outlier = !is_zero && val < low_threshold && low_threshold > 0.0;
                AidAnomalyFlags {
                    stockout,
                    new_product,
                    obsolete_product,
                    high_outlier,
                    low_outlier,
                }
            })
            .collect()
    }

    fn select_distribution(
        values: &[f64],
        mean: f64,
        variance: f64,
        is_intermittent: bool,
    ) -> String {
        if values.is_empty() {
            return "unknown".to_string();
        }
        let is_count_data = values.iter().all(|&v| v >= 0.0 && v == v.floor());
        let nonzero_values: Vec<f64> = values.iter().copied().filter(|&v| v > 0.0).collect();
        if is_count_data {
            if is_intermittent {
                if variance > mean * 1.5 {
                    "negative_binomial".to_string()
                } else {
                    "geometric".to_string()
                }
            } else {
                let dispersion_ratio = if mean > 0.0 { variance / mean } else { 1.0 };
                if dispersion_ratio < 1.5 {
                    "poisson".to_string()
                } else {
                    "negative_binomial".to_string()
                }
            }
        } else {
            if nonzero_values.is_empty() {
                return "normal".to_string();
            }
            let all_positive = values.iter().all(|&v| v >= 0.0);
            if all_positive {
                let skewness = compute_skewness(&nonzero_values, mean, variance.sqrt());
                if skewness > 1.0 {
                    "lognormal".to_string()
                } else if skewness > 0.5 {
                    "gamma".to_string()
                } else if is_intermittent {
                    "rectified_normal".to_string()
                } else {
                    "normal".to_string()
                }
            } else {
                "normal".to_string()
            }
        }
    }

    fn compute_skewness(values: &[f64], mean: f64, std_dev: f64) -> f64 {
        if values.len() < 3 || std_dev == 0.0 {
            return 0.0;
        }
        let n = values.len() as f64;
        let m3: f64 = values.iter().map(|&v| ((v - mean) / std_dev).powi(3)).sum();
        (m3 / n) * (n * (n - 1.0)).sqrt() / (n - 2.0)
    }
}

/// Small deterministic generator (SplitMix64) so the test has no dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn series(rng: &mut Rng, case: usize) -> Vec<f64> {
    let n = 1 + (rng.next() % 40) as usize;
    let p_zero = rng.unif();
    (0..n)
        .map(|_| {
            if rng.unif() < p_zero {
                return 0.0;
            }
            let u = rng.unif();
            match case % 6 {
                0 => (u * 12.0).floor(),              // counts
                1 => (-(1.0 - u).ln() * 5.0).floor(), // overdispersed counts
                2 => (u * 3.0).exp(),                 // skewed continuous
                3 => u * 10.0 - 3.0,                  // negatives
                4 if u < 0.1 => f64::NAN,             // missing values
                4 => (u * 8.0).round(),
                _ if u < 0.05 => 1000.0, // spikes
                _ => 5.0 + u,
            }
        })
        .collect()
}

#[test]
fn heuristic_aid_matches_the_extension_bit_for_bit() {
    let mut rng = Rng(57);
    let methods = [
        (OutlierMethod::ZScore, ext::OutlierMethod::ZScore),
        (OutlierMethod::Iqr, ext::OutlierMethod::Iqr),
    ];
    let mut compared = 0;
    for case in 0..3000 {
        let y = series(&mut rng, case);
        for &threshold in &[0.0, 0.3, 0.5, 1.0] {
            for &(m, em) in &methods {
                let ours_opts = AidOptions {
                    intermittent_threshold: threshold,
                    outlier_method: m,
                };
                let ext_opts = ext::AidOptions {
                    intermittent_threshold: threshold,
                    outlier_method: em,
                };

                let flags = compute_aid_anomalies(&y, &ours_opts).unwrap();
                let eflags = ext::compute_anomaly_flags(&y, &ext_opts);
                assert_eq!(flags.len(), eflags.len());
                for (a, b) in flags.iter().zip(eflags.iter()) {
                    assert_eq!(
                        (
                            a.stockout,
                            a.new_product,
                            a.obsolete_product,
                            a.high_outlier,
                            a.low_outlier
                        ),
                        (
                            b.stockout,
                            b.new_product,
                            b.obsolete_product,
                            b.high_outlier,
                            b.low_outlier
                        ),
                        "flags differ for {y:?}"
                    );
                }

                match (compute_aid(&y, &ours_opts), ext::compute_aid(&y, &ext_opts)) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a.demand_type, b.demand_type);
                        assert_eq!(a.is_intermittent, b.is_intermittent);
                        assert_eq!(a.distribution, b.distribution, "series {y:?}");
                        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
                        assert_eq!(a.variance.to_bits(), b.variance.to_bits());
                        assert_eq!(a.zero_proportion.to_bits(), b.zero_proportion.to_bits());
                        assert_eq!(a.n_observations, b.n_observations);
                        assert_eq!(a.has_stockouts, b.has_stockouts);
                        assert_eq!(a.is_new_product, b.is_new_product);
                        assert_eq!(a.is_obsolete_product, b.is_obsolete_product);
                        assert_eq!(a.stockout_count, b.stockout_count);
                        assert_eq!(a.new_product_count, b.new_product_count);
                        assert_eq!(a.obsolete_product_count, b.obsolete_product_count);
                        assert_eq!(a.high_outlier_count, b.high_outlier_count);
                        assert_eq!(a.low_outlier_count, b.low_outlier_count);
                        compared += 1;
                    }
                    (Err(AidError::NoValidData), Err("novalid")) => {}
                    (a, b) => panic!("result kinds differ for {y:?}: {a:?} vs {b:?}"),
                }
            }
        }
    }
    assert!(compared > 20_000, "compared {compared}");
}
