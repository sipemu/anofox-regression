//! Automatic Identification of Demand (AID) - Demand pattern classifier.
//!
//! This module implements a demand pattern classification system that analyzes
//! time series data to identify whether demand is regular or intermittent,
//! and selects the best-fitting distribution. Based on the aid function from
//! the greybox R package.
//!
//! # Algorithm
//!
//! 1. Detect if data is fractional (any non-integer values)
//! 2. Calculate zero proportion to classify Regular vs Intermittent demand
//! 3. Select candidate distributions based on classification
//! 4. Fit each distribution using ALM and compute information criteria
//! 5. Select best distribution by IC
//! 6. Optionally detect anomalies (stockouts, lifecycle events)
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{AidClassifier, DemandType};
//! use faer::Col;
//!
//! // Regular demand data
//! let demand = Col::from_fn(100, |i| (10.0 + (i as f64 * 0.1).sin() * 2.0).round());
//!
//! let result = AidClassifier::builder()
//!     .intermittent_threshold(0.3)
//!     .build()
//!     .classify(&demand);
//!
//! match result.demand_type {
//!     DemandType::Regular => println!("Regular demand pattern"),
//!     DemandType::Intermittent => println!("Intermittent demand pattern"),
//! }
//! ```

use crate::solvers::alm::{AlmDistribution, AlmRegressor};
use crate::solvers::lm_dynamic::InformationCriterion;
use crate::solvers::traits::{FittedRegressor, Regressor};
use faer::{Col, Mat};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::LazyLock;
use statrs::function::gamma::ln_gamma;

/// Precomputed ln(k!) = ln_gamma(k+1) for k = 0..LN_FACT_SIZE.
/// Covers demand values up to 256, which handles virtually all practical
/// demand data without calling the expensive ln_gamma function.
const LN_FACT_SIZE: usize = 257;

static LN_FACTORIAL: LazyLock<[f64; LN_FACT_SIZE]> = LazyLock::new(|| {
    let mut table = [0.0; LN_FACT_SIZE];
    for k in 1..LN_FACT_SIZE {
        table[k] = table[k - 1] + (k as f64).ln();
    }
    table
});

/// Fast ln_gamma(k+1) = ln(k!) using a lookup table for small integers,
/// falling back to statrs::ln_gamma for large or non-integer values.
#[inline(always)]
fn fast_ln_gamma_p1(v: f64) -> f64 {
    let k = v as u32;
    if (k as usize) < LN_FACT_SIZE && v == k as f64 {
        LN_FACTORIAL[k as usize]
    } else {
        ln_gamma(v + 1.0)
    }
}

/// Demand pattern type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemandType {
    /// Regular demand - low proportion of zeros, consistent demand pattern
    Regular,
    /// Intermittent demand - high proportion of zeros, sporadic demand
    Intermittent,
}

/// Candidate distributions for demand modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DemandDistribution {
    // For regular count data
    /// Poisson distribution - for equi-dispersed count data
    Poisson,
    /// Negative Binomial - for overdispersed count data
    NegativeBinomial,
    /// Geometric distribution - for waiting time / first success
    Geometric,

    // For regular continuous/fractional data
    /// Normal distribution - for symmetric continuous data
    Normal,
    /// Gamma distribution - for positive skewed continuous data
    Gamma,
    /// Log-Normal distribution - for multiplicative processes
    LogNormal,

    // For intermittent patterns (detected via zero proportion)
    /// Rectified Normal - continuous with point mass at zero
    RectifiedNormal,
}

impl DemandDistribution {
    /// Convert to ALM distribution for fitting.
    pub fn to_alm_distribution(&self) -> AlmDistribution {
        match self {
            DemandDistribution::Poisson => AlmDistribution::Poisson,
            DemandDistribution::NegativeBinomial => AlmDistribution::NegativeBinomial,
            DemandDistribution::Geometric => AlmDistribution::Geometric,
            DemandDistribution::Normal => AlmDistribution::Normal,
            DemandDistribution::Gamma => AlmDistribution::Gamma,
            DemandDistribution::LogNormal => AlmDistribution::LogNormal,
            DemandDistribution::RectifiedNormal => AlmDistribution::RectifiedNormal,
        }
    }
}

/// Anomaly types that can be detected in demand patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnomalyType {
    /// No anomaly detected
    None,
    /// Unexpected zero (potential stockout)
    Stockout,
    /// Leading zeros indicating new product
    NewProduct,
    /// Trailing zeros indicating obsolete product
    ObsoleteProduct,
    /// Unusually high demand value
    HighOutlier,
    /// Unusually low non-zero demand
    LowOutlier,
}

/// Distribution parameters after fitting.
#[derive(Debug, Clone)]
pub struct DistributionParameters {
    /// Estimated mean
    pub mean: f64,
    /// Estimated variance
    pub variance: f64,
    /// Shape parameter (for Gamma, NegBinom) if applicable
    pub shape: Option<f64>,
    /// Estimated probability of zero (for intermittent models)
    pub zero_prob: Option<f64>,
    /// Scale parameter if applicable
    pub scale: Option<f64>,
}

/// Result of demand classification.
#[derive(Debug, Clone)]
pub struct DemandClassification {
    /// Primary demand type (Regular or Intermittent)
    pub demand_type: DemandType,
    /// Whether data contains fractional values
    pub is_fractional: bool,
    /// Best-fitting distribution
    pub distribution: DemandDistribution,
    /// Fitted distribution parameters
    pub parameters: DistributionParameters,
    /// Anomaly flags for each observation (if detection was enabled)
    pub anomalies: Vec<AnomalyType>,
    /// Information criteria values for each candidate distribution
    pub ic_values: HashMap<DemandDistribution, f64>,
    /// Zero proportion in the data
    pub zero_proportion: f64,
    /// Number of observations
    pub n_observations: usize,
}

impl DemandClassification {
    /// Check if any stockouts were detected.
    pub fn has_stockouts(&self) -> bool {
        self.anomalies.contains(&AnomalyType::Stockout)
    }

    /// Count the number of anomalies of each type.
    pub fn anomaly_counts(&self) -> HashMap<AnomalyType, usize> {
        let mut counts = HashMap::new();
        for anomaly in &self.anomalies {
            *counts.entry(*anomaly).or_insert(0) += 1;
        }
        counts
    }

    /// Check if product appears to be new (leading zeros).
    pub fn is_new_product(&self) -> bool {
        self.anomalies.contains(&AnomalyType::NewProduct)
    }

    /// Check if product appears to be obsolete (trailing zeros).
    pub fn is_obsolete_product(&self) -> bool {
        self.anomalies.contains(&AnomalyType::ObsoleteProduct)
    }
}

/// Automatic Identification of Demand (AID) classifier.
#[derive(Debug, Clone)]
pub struct AidClassifier {
    /// Significance level for anomaly detection (reserved for future use)
    #[allow(dead_code)]
    anomaly_alpha: f64,
    /// Threshold for classifying demand as intermittent (proportion of zeros)
    intermittent_threshold: f64,
    /// Whether to detect anomalies
    detect_anomalies: bool,
    /// Information criterion for distribution selection
    ic_type: InformationCriterion,
}

impl Default for AidClassifier {
    fn default() -> Self {
        Self {
            anomaly_alpha: 0.05,
            intermittent_threshold: 0.3,
            detect_anomalies: true,
            ic_type: InformationCriterion::AICc,
        }
    }
}

impl AidClassifier {
    /// Create a new classifier with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for configuring the classifier.
    pub fn builder() -> AidClassifierBuilder {
        AidClassifierBuilder::default()
    }

    /// Classify the demand pattern.
    ///
    /// # Arguments
    /// * `y` - Demand time series data
    ///
    /// # Returns
    /// Classification result containing demand type, best distribution, and anomalies.
    pub fn classify(&self, y: &Col<f64>) -> DemandClassification {
        let n = y.nrows();

        if n == 0 {
            return DemandClassification {
                demand_type: DemandType::Regular,
                is_fractional: false,
                distribution: DemandDistribution::Normal,
                parameters: DistributionParameters {
                    mean: 0.0,
                    variance: 0.0,
                    shape: None,
                    zero_prob: None,
                    scale: None,
                },
                anomalies: Vec::new(),
                ic_values: HashMap::new(),
                zero_proportion: 0.0,
                n_observations: 0,
            };
        }

        // Pass 1: cheap sums and flags (no transcendental functions)
        let n_f = n as f64;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut zero_count = 0u32;
        let mut is_fractional = false;
        let mut has_non_positive = false;
        let mut n_positive = 0usize;

        for &v in y.iter() {
            sum += v;
            sum_sq += v * v;
            if v == 0.0 {
                zero_count += 1;
            }
            if !is_fractional && v != v.round() {
                is_fractional = true;
            }
            if v <= 0.0 {
                has_non_positive = true;
            } else {
                n_positive += 1;
            }
        }

        let y_mean = sum / n_f;
        let zero_proportion = zero_count as f64 / n_f;
        // One-pass variance: Σ(y-mean)² = Σy² - n*mean²
        let sum_sq_dev = sum_sq - sum * sum / n_f;
        let y_var = sum_sq_dev / n_f;

        // Determine demand type early so we know which candidates we'll evaluate
        let demand_type = if zero_proportion > self.intermittent_threshold {
            DemandType::Intermittent
        } else {
            DemandType::Regular
        };

        // Pass 2 (conditional): only compute expensive sums needed by the
        // candidate distributions for this data type. This avoids calling
        // ln_gamma for continuous data and ln() for count data.
        let mut sum_ln_gamma_yp1 = 0.0;
        let mut sum_ln_y = 0.0;
        let mut sum_ln_y_sq = 0.0;

        if !is_fractional {
            // Count data path: Poisson/NegBin need Σ ln_Γ(y_i+1) = Σ ln(y_i!)
            // Use lookup table for small integers (typical demand data)
            for &v in y.iter() {
                sum_ln_gamma_yp1 += fast_ln_gamma_p1(v.round().max(0.0));
            }
        } else if !has_non_positive {
            // Continuous positive data: Gamma needs Σ ln(y), LogNormal needs Σ ln(y)²
            let needs_ln_sq = demand_type == DemandType::Regular; // LogNormal candidate
            for &v in y.iter() {
                let ln_v = v.ln();
                sum_ln_y += ln_v;
                if needs_ln_sq {
                    sum_ln_y_sq += ln_v * ln_v;
                }
            }
        }
        // else: fractional with zeros → candidates are RectifiedNormal(ALM), Gamma(skipped), Normal
        //       no expensive sums needed

        let log_mean_exp = if !has_non_positive && is_fractional && n_positive > 0 {
            Some((sum_ln_y / n_positive as f64).exp())
        } else {
            None
        };

        let stats = PrecomputedStats {
            n,
            y_mean,
            y_var,
            zero_proportion,
            has_non_positive,
            sum_y: sum,
            sum_sq_dev,
            sum_ln_gamma_yp1,
            sum_ln_y,
            sum_ln_y_sq,
            log_mean_exp,
        };

        // Select candidate distributions
        let candidates = self.get_candidate_distributions(demand_type, is_fractional);

        // Fit each distribution and compute IC
        let mut ic_values = HashMap::with_capacity(3);
        let mut best_dist = candidates[0];
        let mut best_ic = f64::INFINITY;
        let mut best_params = DistributionParameters {
            mean: y_mean,
            variance: y_var,
            shape: None,
            zero_prob: Some(zero_proportion),
            scale: None,
        };

        for dist in candidates {
            if let Some((ic, params)) = self.fit_distribution(y, dist, &stats) {
                ic_values.insert(dist, ic);
                if ic < best_ic {
                    best_ic = ic;
                    best_dist = dist;
                    best_params = params;
                }
            }
        }

        // Step 6: Detect anomalies if enabled
        let anomalies = if self.detect_anomalies {
            self.detect_anomalies_impl(y, best_dist, &best_params)
        } else {
            vec![AnomalyType::None; n]
        };

        DemandClassification {
            demand_type,
            is_fractional,
            distribution: best_dist,
            parameters: best_params,
            anomalies,
            ic_values,
            zero_proportion,
            n_observations: n,
        }
    }

    /// Get candidate distributions based on demand type and data type.
    #[inline]
    fn get_candidate_distributions(
        &self,
        demand_type: DemandType,
        is_fractional: bool,
    ) -> [DemandDistribution; 3] {
        match (demand_type, is_fractional) {
            (DemandType::Regular, false) => [
                DemandDistribution::Poisson,
                DemandDistribution::NegativeBinomial,
                DemandDistribution::Normal,
            ],
            (DemandType::Regular, true) => [
                DemandDistribution::Normal,
                DemandDistribution::Gamma,
                DemandDistribution::LogNormal,
            ],
            (DemandType::Intermittent, false) => [
                DemandDistribution::NegativeBinomial,
                DemandDistribution::Geometric,
                DemandDistribution::Poisson,
            ],
            (DemandType::Intermittent, true) => [
                DemandDistribution::RectifiedNormal,
                DemandDistribution::Gamma,
                DemandDistribution::Normal,
            ],
        }
    }

    /// Fit a distribution to the data and return IC and parameters.
    ///
    /// Uses closed-form MLE + O(1) log-likelihood formulas for intercept-only
    /// models. All expensive sums are precomputed in `stats`. Falls back to
    /// ALM only for RectifiedNormal which requires numerical optimization.
    fn fit_distribution(
        &self,
        y: &Col<f64>,
        dist: DemandDistribution,
        stats: &PrecomputedStats,
    ) -> Option<(f64, DistributionParameters)> {
        let n = stats.n;
        let n_f = n as f64;
        let mu = stats.y_mean;
        let var = stats.y_var;

        // Skip distributions that require positive data if we have zeros/negatives
        if stats.has_non_positive
            && matches!(
                dist,
                DemandDistribution::Gamma | DemandDistribution::LogNormal
            )
        {
            return None;
        }

        // RectifiedNormal requires numerical optimization — use ALM fallback
        if dist == DemandDistribution::RectifiedNormal {
            return self.fit_distribution_via_alm(y, dist);
        }

        // Compute LL, scale, shape, n_params using closed-form O(1) formulas.
        // Each formula mirrors the corresponding ll_* function in alm.rs
        // but for constant mu, expressed in terms of precomputed sums.
        let (ll, scale, shape, n_params, mu_out) = match dist {
            DemandDistribution::Poisson => {
                // ll_poisson: Σ [y_i * ln(λ) - λ - ln_Γ(y_i + 1)]
                // With constant λ = mu: sum_y * ln(λ) - n*λ - sum_ln_gamma_yp1
                let lambda = mu.max(1e-10);
                let ll = stats.sum_y * lambda.ln() - n_f * lambda - stats.sum_ln_gamma_yp1;
                (ll, 1.0, None, 1, mu)
            }

            DemandDistribution::NegativeBinomial => {
                // size = mu² / (var - mu), p = size/(size+mu)
                // ll_negbin: Σ [ln_Γ(y_i+size) - ln_Γ(size) - ln_Γ(y_i+1)
                //             + size*ln(p) + y_i*ln(1-p)]
                // With constant mu: only Σ ln_Γ(y_i+size) needs O(n)
                let size = if var > mu && mu > 0.0 {
                    mu * mu / (var - mu)
                } else {
                    1.0
                };
                let mu_c = mu.max(1e-10);
                let p = size / (size + mu_c);

                // O(n) for the size-dependent term.
                // Short-circuit y_i == 0 (very common in intermittent demand)
                // to avoid expensive ln_gamma calls.
                let ln_gamma_size = ln_gamma(size);
                let sum_ln_gamma_y_plus_size: f64 = y
                    .iter()
                    .map(|&v| {
                        if v <= 0.0 {
                            ln_gamma_size
                        } else {
                            ln_gamma(v.round() + size)
                        }
                    })
                    .sum();

                let ll = sum_ln_gamma_y_plus_size
                    - n_f * ln_gamma_size
                    - stats.sum_ln_gamma_yp1
                    + n_f * size * p.ln()
                    + stats.sum_y * (1.0 - p).ln();

                (ll, 1.0, Some(size), 2, mu)
            }

            DemandDistribution::Geometric => {
                // ll_geometric: Σ [ln(p) + k*ln(1-p)]
                // With constant λ = mu: n*ln(p) + sum_y*ln(1-p)
                let lambda = mu.max(1e-10);
                let p = 1.0 / (1.0 + lambda);
                let ll = n_f * p.ln() + stats.sum_y * (1.0 - p).ln();
                (ll, 1.0, None, 1, mu)
            }

            DemandDistribution::Normal => {
                // ll_normal: -n/2 * ln(2πσ²) - RSS/(2σ²)
                // RSS = sum_sq_dev, σ = sqrt(RSS/df)
                let df = (n - 1) as f64;
                let sigma = (stats.sum_sq_dev / df).sqrt();
                let sigma2 = sigma * sigma;
                let ll = -0.5 * n_f * (2.0 * PI * sigma2).ln()
                    - stats.sum_sq_dev / (2.0 * sigma2);
                (ll, sigma, None, 2, mu)
            }

            DemandDistribution::Gamma => {
                // shape = mu²/var, rate = shape/mu
                // ll_gamma: Σ [shape*ln(rate) + (shape-1)*ln(y_i) - rate*y_i - ln_Γ(shape)]
                // With constant mu: n*shape*ln(rate) + (shape-1)*sum_ln_y - rate*sum_y - n*ln_Γ(shape)
                let shape = if var > 0.0 { mu * mu / var } else { 1.0 };
                let rate = shape / mu;
                let ll = n_f * shape * rate.ln()
                    + (shape - 1.0) * stats.sum_ln_y
                    - rate * stats.sum_y
                    - n_f * ln_gamma(shape);
                (ll, 1.0, Some(shape), 2, mu)
            }

            DemandDistribution::LogNormal => {
                // MLE: mu_log = mean(ln(y)), mu_response = exp(mu_log)
                // σ = sqrt(Σ(ln(y) - mu_log)² / df)
                // ll_log_normal: Σ [-ln(y) - ln(σ) - 0.5*ln(2π) - (ln(y)-ln(mu))²/(2σ²)]
                // = -sum_ln_y - n*ln(σ) - n/2*ln(2π) - log_rss/(2σ²)
                let mu_resp = stats.log_mean_exp.unwrap_or(mu);
                let log_mu = mu_resp.ln();
                let df = (n - 1) as f64;
                // log_rss = Σ(ln(y) - log_mu)² = sum_ln_y_sq - 2*log_mu*sum_ln_y + n*log_mu²
                let log_rss = stats.sum_ln_y_sq
                    - 2.0 * log_mu * stats.sum_ln_y
                    + n_f * log_mu * log_mu;
                let sigma = (log_rss / df).sqrt();
                let sigma2 = sigma * sigma;
                let ll = -stats.sum_ln_y
                    - n_f * sigma.ln()
                    - 0.5 * n_f * (2.0 * PI).ln()
                    - log_rss / (2.0 * sigma2);
                (ll, sigma, None, 2, mu_resp)
            }

            DemandDistribution::RectifiedNormal => unreachable!(),
        };

        if !ll.is_finite() {
            return None;
        }

        let ic = self.ic_type.compute(ll, n_params, n);

        // Variance for output: use precomputed when mu_out == y_mean
        let out_var = if (mu_out - mu).abs() < 1e-15 {
            var
        } else {
            // LogNormal case: mu_out differs, need actual variance
            y.iter().map(|&v| (v - mu_out).powi(2)).sum::<f64>() / n_f
        };

        Some((
            ic,
            DistributionParameters {
                mean: mu_out,
                variance: out_var,
                shape,
                zero_prob: Some(stats.zero_proportion),
                scale: Some(scale),
            },
        ))
    }

    /// Fallback: fit distribution via ALM for distributions that need numerical optimization.
    fn fit_distribution_via_alm(
        &self,
        y: &Col<f64>,
        dist: DemandDistribution,
    ) -> Option<(f64, DistributionParameters)> {
        let n = y.nrows();
        let x = Mat::from_fn(n, 1, |_, _| 1.0);

        let model = AlmRegressor::builder()
            .distribution(dist.to_alm_distribution())
            .with_intercept(false)
            .compute_inference(false)
            .build();

        match model.fit(&x, y) {
            Ok(fitted) => {
                let result = fitted.result();
                let scale = fitted.scale();

                let ll = result.log_likelihood;
                let k = result.n_parameters;
                let ic = self.ic_type.compute(ll, k, n);

                let mean = result.fitted_values.iter().sum::<f64>() / n as f64;
                let variance: f64 = result
                    .residuals
                    .iter()
                    .map(|&r| r.powi(2))
                    .sum::<f64>()
                    / n as f64;

                let zero_count = y.iter().filter(|&&v| v == 0.0).count();
                let zero_prob = zero_count as f64 / n as f64;

                Some((
                    ic,
                    DistributionParameters {
                        mean,
                        variance,
                        shape: None,
                        zero_prob: Some(zero_prob),
                        scale: Some(scale),
                    },
                ))
            }
            Err(_) => None,
        }
    }

    /// Detect anomalies in the demand series.
    fn detect_anomalies_impl(
        &self,
        y: &Col<f64>,
        _dist: DemandDistribution,
        params: &DistributionParameters,
    ) -> Vec<AnomalyType> {
        let n = y.nrows();
        let mut anomalies = vec![AnomalyType::None; n];

        if n == 0 {
            return anomalies;
        }

        let mean = params.mean;
        let std_dev = params.variance.sqrt();

        // Detect leading zeros (new product)
        let mut leading_zeros = 0;
        for i in 0..n {
            if y[i] == 0.0 {
                leading_zeros += 1;
            } else {
                break;
            }
        }

        // If more than 20% are leading zeros, mark as new product
        if leading_zeros > 0 && (leading_zeros as f64 / n as f64) > 0.1 {
            for anomaly in anomalies.iter_mut().take(leading_zeros) {
                *anomaly = AnomalyType::NewProduct;
            }
        }

        // Detect trailing zeros (obsolete product)
        let mut trailing_zeros = 0;
        for i in (0..n).rev() {
            if y[i] == 0.0 {
                trailing_zeros += 1;
            } else {
                break;
            }
        }

        // If more than 20% are trailing zeros, mark as obsolete
        if trailing_zeros > 0 && (trailing_zeros as f64 / n as f64) > 0.1 {
            for anomaly in anomalies.iter_mut().skip(n - trailing_zeros) {
                if *anomaly == AnomalyType::None {
                    *anomaly = AnomalyType::ObsoleteProduct;
                }
            }
        }

        // Detect stockouts (unexpected zeros in the middle)
        if mean > 0.0 {
            let _z_threshold = 2.0; // ~95% confidence

            for i in leading_zeros..(n - trailing_zeros) {
                if y[i] == 0.0 && anomalies[i] == AnomalyType::None {
                    // Zero in a region where we expect positive demand
                    // Check if surrounding values are non-zero
                    let before_nonzero = if i > 0 {
                        (0..i).rev().take(3).any(|j| y[j] > 0.0)
                    } else {
                        false
                    };
                    let after_nonzero = ((i + 1)..n).take(3).any(|j| y[j] > 0.0);

                    if before_nonzero && after_nonzero {
                        anomalies[i] = AnomalyType::Stockout;
                    }
                }
            }
        }

        // Detect outliers (high/low values)
        if std_dev > 0.0 {
            let high_threshold = mean + 3.0 * std_dev;
            let low_threshold = (mean - 3.0 * std_dev).max(0.0);

            for i in 0..n {
                if anomalies[i] == AnomalyType::None {
                    if y[i] > high_threshold {
                        anomalies[i] = AnomalyType::HighOutlier;
                    } else if y[i] > 0.0 && y[i] < low_threshold && mean > 0.0 {
                        anomalies[i] = AnomalyType::LowOutlier;
                    }
                }
            }
        }

        anomalies
    }
}

/// Precomputed statistics for the input data, shared across distribution fits.
///
/// All O(n) sums are computed once so that per-distribution log-likelihood
/// evaluation is O(1) (except NegBin which needs one O(n) pass for
/// `ln_gamma(y_i + size)`).
struct PrecomputedStats {
    n: usize,
    y_mean: f64,
    y_var: f64,
    zero_proportion: f64,
    has_non_positive: bool,
    sum_y: f64,
    /// Σ (y_i - mean)² = n * variance
    sum_sq_dev: f64,
    /// Σ ln_Γ(round(y_i).max(0) + 1) — for Poisson & NegBin LL (count data only)
    sum_ln_gamma_yp1: f64,
    /// Σ ln(y_i) for y_i > 0 — for Gamma & LogNormal LL (continuous data only)
    sum_ln_y: f64,
    /// Σ (ln(y_i))² for y_i > 0 — for LogNormal LL (continuous regular only)
    sum_ln_y_sq: f64,
    /// exp(mean(ln(y))) for positive data — LogNormal MLE on response scale
    log_mean_exp: Option<f64>,
}

/// Builder for AidClassifier.
#[derive(Debug, Clone)]
pub struct AidClassifierBuilder {
    anomaly_alpha: f64,
    intermittent_threshold: f64,
    detect_anomalies: bool,
    ic_type: InformationCriterion,
}

impl Default for AidClassifierBuilder {
    fn default() -> Self {
        Self {
            anomaly_alpha: 0.05,
            intermittent_threshold: 0.3,
            detect_anomalies: true,
            ic_type: InformationCriterion::AICc,
        }
    }
}

impl AidClassifierBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the significance level for anomaly detection.
    ///
    /// # Arguments
    /// * `alpha` - Significance level (0.0 to 1.0, default 0.05)
    pub fn anomaly_alpha(mut self, alpha: f64) -> Self {
        self.anomaly_alpha = alpha.clamp(0.001, 0.5);
        self
    }

    /// Set the threshold for classifying demand as intermittent.
    ///
    /// If the proportion of zeros exceeds this threshold, demand is
    /// classified as intermittent.
    ///
    /// # Arguments
    /// * `threshold` - Zero proportion threshold (0.0 to 1.0, default 0.3)
    pub fn intermittent_threshold(mut self, threshold: f64) -> Self {
        self.intermittent_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set whether to detect anomalies.
    ///
    /// # Arguments
    /// * `detect` - Whether to detect anomalies (default true)
    pub fn detect_anomalies(mut self, detect: bool) -> Self {
        self.detect_anomalies = detect;
        self
    }

    /// Set the information criterion for distribution selection.
    ///
    /// # Arguments
    /// * `ic` - Information criterion type (default AICc)
    pub fn ic(mut self, ic: InformationCriterion) -> Self {
        self.ic_type = ic;
        self
    }

    /// Build the AidClassifier.
    pub fn build(self) -> AidClassifier {
        AidClassifier {
            anomaly_alpha: self.anomaly_alpha,
            intermittent_threshold: self.intermittent_threshold,
            detect_anomalies: self.detect_anomalies,
            ic_type: self.ic_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let classifier = AidClassifier::builder().build();
        assert_eq!(classifier.ic_type, InformationCriterion::AICc);
        assert!(classifier.detect_anomalies);
    }

    #[test]
    fn test_builder_custom() {
        let classifier = AidClassifier::builder()
            .intermittent_threshold(0.5)
            .detect_anomalies(false)
            .ic(InformationCriterion::BIC)
            .build();

        assert!(!classifier.detect_anomalies);
        assert_eq!(classifier.ic_type, InformationCriterion::BIC);
    }

    #[test]
    fn test_regular_count_demand() {
        // Regular Poisson-like demand
        let y = Col::from_fn(50, |i| ((i % 5) + 5) as f64);

        let result = AidClassifier::new().classify(&y);

        assert_eq!(result.demand_type, DemandType::Regular);
        assert!(!result.is_fractional);
        assert_eq!(result.zero_proportion, 0.0);
    }

    #[test]
    fn test_intermittent_demand() {
        // Intermittent demand with many zeros
        let y = Col::from_fn(50, |i| if i % 3 == 0 { 5.0 } else { 0.0 });

        let result = AidClassifier::builder()
            .intermittent_threshold(0.3)
            .build()
            .classify(&y);

        assert_eq!(result.demand_type, DemandType::Intermittent);
        // 33 zeros out of 50 = 66% zeros
        assert!(result.zero_proportion > 0.5);
    }

    #[test]
    fn test_fractional_demand() {
        // Fractional/continuous demand
        let y = Col::from_fn(30, |i| 5.5 + (i as f64 * 0.1).sin());

        let result = AidClassifier::new().classify(&y);

        assert!(result.is_fractional);
        assert_eq!(result.demand_type, DemandType::Regular);
    }

    #[test]
    fn test_stockout_detection() {
        // Demand with stockouts in the middle
        let mut y = Col::from_fn(30, |_| 10.0);
        y[10] = 0.0; // Stockout
        y[15] = 0.0; // Another stockout

        let result = AidClassifier::builder()
            .detect_anomalies(true)
            .build()
            .classify(&y);

        assert!(result.has_stockouts());
    }

    #[test]
    fn test_new_product_detection() {
        // New product: leading zeros then demand
        let y = Col::from_fn(30, |i| if i < 10 { 0.0 } else { 5.0 });

        let result = AidClassifier::new().classify(&y);

        assert!(result.is_new_product());
        // First 10 should be marked as NewProduct
        for i in 0..10 {
            assert_eq!(result.anomalies[i], AnomalyType::NewProduct);
        }
    }

    #[test]
    fn test_obsolete_product_detection() {
        // Obsolete product: demand then trailing zeros
        let y = Col::from_fn(30, |i| if i < 20 { 5.0 } else { 0.0 });

        let result = AidClassifier::new().classify(&y);

        assert!(result.is_obsolete_product());
        // Last 10 should be marked as ObsoleteProduct
        for i in 20..30 {
            assert_eq!(result.anomalies[i], AnomalyType::ObsoleteProduct);
        }
    }

    #[test]
    fn test_empty_data() {
        let y = Col::zeros(0);
        let result = AidClassifier::new().classify(&y);

        assert_eq!(result.n_observations, 0);
        assert!(result.anomalies.is_empty());
    }

    #[test]
    fn test_ic_values_populated() {
        let y = Col::from_fn(50, |i| (i + 1) as f64);

        let result = AidClassifier::new().classify(&y);

        // Should have at least one IC value
        assert!(!result.ic_values.is_empty());
    }

    #[test]
    fn test_demand_distribution_to_alm() {
        // Test conversions
        assert_eq!(
            DemandDistribution::Poisson.to_alm_distribution(),
            AlmDistribution::Poisson
        );
        assert_eq!(
            DemandDistribution::Normal.to_alm_distribution(),
            AlmDistribution::Normal
        );
        assert_eq!(
            DemandDistribution::Gamma.to_alm_distribution(),
            AlmDistribution::Gamma
        );
    }

    #[test]
    fn test_anomaly_counts() {
        let mut y = Col::from_fn(30, |_| 10.0);
        y[10] = 0.0;
        y[15] = 0.0;
        y[25] = 100.0; // High outlier

        let result = AidClassifier::new().classify(&y);
        let counts = result.anomaly_counts();

        // Should have some stockouts detected
        assert!(
            counts.get(&AnomalyType::Stockout).unwrap_or(&0) > &0
                || counts.get(&AnomalyType::HighOutlier).unwrap_or(&0) > &0
        );
    }
}
