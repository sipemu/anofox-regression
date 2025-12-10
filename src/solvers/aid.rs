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

        // Step 1: Check if fractional
        let is_fractional = self.detect_fractional(y);

        // Step 2: Calculate zero proportion and basic statistics
        let zero_count = y.iter().filter(|&&v| v == 0.0).count();
        let zero_proportion = zero_count as f64 / n as f64;

        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_var: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;

        // Step 3: Classify demand type
        let demand_type = if zero_proportion > self.intermittent_threshold {
            DemandType::Intermittent
        } else {
            DemandType::Regular
        };

        // Step 4: Select candidate distributions
        let candidates = self.get_candidate_distributions(demand_type, is_fractional);

        // Step 5: Fit each distribution and compute IC
        let mut ic_values: HashMap<DemandDistribution, f64> = HashMap::new();
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
            if let Some((ic, params)) = self.fit_distribution(y, dist) {
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

    /// Check if data contains fractional (non-integer) values.
    fn detect_fractional(&self, y: &Col<f64>) -> bool {
        y.iter().any(|&v| v != v.round())
    }

    /// Get candidate distributions based on demand type and data type.
    fn get_candidate_distributions(
        &self,
        demand_type: DemandType,
        is_fractional: bool,
    ) -> Vec<DemandDistribution> {
        match (demand_type, is_fractional) {
            // Regular count data
            (DemandType::Regular, false) => vec![
                DemandDistribution::Poisson,
                DemandDistribution::NegativeBinomial,
                DemandDistribution::Normal,
            ],
            // Regular fractional/continuous data
            (DemandType::Regular, true) => vec![
                DemandDistribution::Normal,
                DemandDistribution::Gamma,
                DemandDistribution::LogNormal,
            ],
            // Intermittent count data
            (DemandType::Intermittent, false) => vec![
                DemandDistribution::NegativeBinomial,
                DemandDistribution::Geometric,
                DemandDistribution::Poisson,
            ],
            // Intermittent fractional data
            (DemandType::Intermittent, true) => vec![
                DemandDistribution::RectifiedNormal,
                DemandDistribution::Gamma,
                DemandDistribution::Normal,
            ],
        }
    }

    /// Fit a distribution to the data and return IC and parameters.
    fn fit_distribution(
        &self,
        y: &Col<f64>,
        dist: DemandDistribution,
    ) -> Option<(f64, DistributionParameters)> {
        let n = y.nrows();

        // Create a simple intercept-only model (constant mean)
        let x = Mat::from_fn(n, 1, |_, _| 1.0);

        let alm_dist = dist.to_alm_distribution();

        // Skip distributions that require positive data if we have zeros/negatives
        let has_non_positive = y.iter().any(|&v| v <= 0.0);
        if has_non_positive
            && matches!(
                dist,
                DemandDistribution::Gamma | DemandDistribution::LogNormal
            )
        {
            return None;
        }

        // Fit the model
        let model = AlmRegressor::builder()
            .distribution(alm_dist)
            .with_intercept(false) // x is already an intercept column
            .compute_inference(false)
            .build();

        match model.fit(&x, y) {
            Ok(fitted) => {
                let result = fitted.result();
                let scale = fitted.scale();

                // Compute IC
                let ll = result.log_likelihood;
                let k = result.n_parameters;
                let ic = self.ic_type.compute(ll, k, n);

                // Extract parameters
                let mean = result.fitted_values.iter().sum::<f64>() / n as f64;
                let residuals = &result.residuals;
                let variance: f64 = residuals.iter().map(|&r| r.powi(2)).sum::<f64>() / n as f64;

                let zero_count = y.iter().filter(|&&v| v == 0.0).count();
                let zero_prob = zero_count as f64 / n as f64;

                // Shape parameter estimation (for Gamma, NegBinom)
                let shape = match dist {
                    DemandDistribution::Gamma => {
                        if variance > 0.0 {
                            Some(mean.powi(2) / variance)
                        } else {
                            Some(1.0)
                        }
                    }
                    DemandDistribution::NegativeBinomial => {
                        // Estimate size parameter: r = mean^2 / (variance - mean)
                        if variance > mean && mean > 0.0 {
                            Some(mean.powi(2) / (variance - mean))
                        } else {
                            Some(1.0)
                        }
                    }
                    _ => None,
                };

                Some((
                    ic,
                    DistributionParameters {
                        mean,
                        variance,
                        shape,
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
