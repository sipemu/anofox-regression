//! AID (Automatic Identification of Demand) classifier wrapper for WebAssembly.

use anofox_regression::solvers::{
    AidClassifier as RustAidClassifier, AnomalyType, DemandClassification, DemandDistribution,
    DemandType, InformationCriterion as RustInformationCriterion,
};
use faer::Col;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DistributionParametersJs {
    pub mean: f64,
    pub variance: f64,
    pub shape: Option<f64>,
    pub zero_prob: Option<f64>,
    pub scale: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DemandClassificationJs {
    /// "regular" or "intermittent".
    pub demand_type: String,
    /// Whether the data contains any fractional values.
    pub is_fractional: bool,
    /// Best-fitting distribution (camelCase string, e.g. "negativeBinomial").
    pub distribution: String,
    pub parameters: DistributionParametersJs,
    /// One anomaly label per observation (length = n_observations).
    pub anomalies: Vec<String>,
    /// Information criterion value for each candidate distribution.
    pub ic_values: HashMap<String, f64>,
    pub zero_proportion: f64,
    pub n_observations: usize,
}

fn demand_type_label(t: DemandType) -> &'static str {
    match t {
        DemandType::Regular => "regular",
        DemandType::Intermittent => "intermittent",
    }
}

fn distribution_label(d: DemandDistribution) -> &'static str {
    match d {
        DemandDistribution::Poisson => "poisson",
        DemandDistribution::NegativeBinomial => "negativeBinomial",
        DemandDistribution::Geometric => "geometric",
        DemandDistribution::Normal => "normal",
        DemandDistribution::Gamma => "gamma",
        DemandDistribution::LogNormal => "logNormal",
        DemandDistribution::RectifiedNormal => "rectifiedNormal",
    }
}

fn anomaly_label(a: AnomalyType) -> &'static str {
    match a {
        AnomalyType::None => "none",
        AnomalyType::Stockout => "stockout",
        AnomalyType::NewProduct => "newProduct",
        AnomalyType::ObsoleteProduct => "obsoleteProduct",
        AnomalyType::HighOutlier => "highOutlier",
        AnomalyType::LowOutlier => "lowOutlier",
    }
}

fn to_js(result: DemandClassification) -> DemandClassificationJs {
    DemandClassificationJs {
        demand_type: demand_type_label(result.demand_type).to_string(),
        is_fractional: result.is_fractional,
        distribution: distribution_label(result.distribution).to_string(),
        parameters: DistributionParametersJs {
            mean: result.parameters.mean,
            variance: result.parameters.variance,
            shape: result.parameters.shape,
            zero_prob: result.parameters.zero_prob,
            scale: result.parameters.scale,
        },
        anomalies: result
            .anomalies
            .iter()
            .map(|a| anomaly_label(*a).to_string())
            .collect(),
        ic_values: result
            .ic_values
            .into_iter()
            .map(|(d, v)| (distribution_label(d).to_string(), v))
            .collect(),
        zero_proportion: result.zero_proportion,
        n_observations: result.n_observations,
    }
}

/// Result of running the AID classifier on a demand series.
#[wasm_bindgen]
pub struct AidResult {
    inner: DemandClassification,
}

#[wasm_bindgen]
impl AidResult {
    /// Get the full classification result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let js = to_js(self.inner.clone());
        serde_wasm_bindgen::to_value(&js).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Demand type label ("regular" or "intermittent").
    #[wasm_bindgen(js_name = getDemandType)]
    pub fn get_demand_type(&self) -> String {
        demand_type_label(self.inner.demand_type).to_string()
    }

    /// Best-fitting distribution label.
    #[wasm_bindgen(js_name = getDistribution)]
    pub fn get_distribution(&self) -> String {
        distribution_label(self.inner.distribution).to_string()
    }

    /// Proportion of zero observations.
    #[wasm_bindgen(js_name = getZeroProportion)]
    pub fn get_zero_proportion(&self) -> f64 {
        self.inner.zero_proportion
    }

    /// Whether the data contains any fractional values.
    #[wasm_bindgen(js_name = isFractional)]
    pub fn is_fractional(&self) -> bool {
        self.inner.is_fractional
    }

    /// Whether any stockouts were detected in the series.
    #[wasm_bindgen(js_name = hasStockouts)]
    pub fn has_stockouts(&self) -> bool {
        self.inner.has_stockouts()
    }

    /// Whether the series looks like a new product (leading zeros).
    #[wasm_bindgen(js_name = isNewProduct)]
    pub fn is_new_product(&self) -> bool {
        self.inner.is_new_product()
    }

    /// Whether the series looks like an obsolete product (trailing zeros).
    #[wasm_bindgen(js_name = isObsoleteProduct)]
    pub fn is_obsolete_product(&self) -> bool {
        self.inner.is_obsolete_product()
    }
}

/// AID (Automatic Identification of Demand) classifier.
///
/// Inspects a univariate demand time series and reports the best-fitting
/// distribution, whether the pattern is regular vs intermittent, and per-
/// observation anomaly flags.
///
/// # Example (JavaScript)
/// ```javascript
/// const classifier = new AidClassifier();
/// classifier.setIc("aicc");
/// classifier.setIntermittentThreshold(0.3);
///
/// const result = classifier.classify(y);
/// console.log(result.getDistribution(), result.getZeroProportion());
/// ```
#[wasm_bindgen]
pub struct AidClassifier {
    anomaly_alpha: f64,
    intermittent_threshold: f64,
    detect_anomalies: bool,
    ic_type: String,
}

#[wasm_bindgen]
impl AidClassifier {
    /// Create a new AID classifier with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            anomaly_alpha: 0.05,
            intermittent_threshold: 0.3,
            detect_anomalies: true,
            ic_type: "aicc".to_string(),
        }
    }

    /// Set the significance level for anomaly detection (default: 0.05).
    #[wasm_bindgen(js_name = setAnomalyAlpha)]
    pub fn set_anomaly_alpha(&mut self, alpha: f64) {
        self.anomaly_alpha = alpha;
    }

    /// Set the intermittent-classification threshold on the zero proportion
    /// (default: 0.3).
    #[wasm_bindgen(js_name = setIntermittentThreshold)]
    pub fn set_intermittent_threshold(&mut self, threshold: f64) {
        self.intermittent_threshold = threshold;
    }

    /// Enable or disable per-observation anomaly detection (default: true).
    #[wasm_bindgen(js_name = setDetectAnomalies)]
    pub fn set_detect_anomalies(&mut self, detect: bool) {
        self.detect_anomalies = detect;
    }

    /// Set the information criterion for distribution selection.
    ///
    /// Accepted values: "aic", "aicc" (default), "bic". Unknown values fall
    /// back to AICc.
    #[wasm_bindgen(js_name = setIc)]
    pub fn set_ic(&mut self, ic: &str) {
        self.ic_type = ic.to_string();
    }

    fn parse_ic(&self) -> RustInformationCriterion {
        match self.ic_type.to_lowercase().as_str() {
            "aic" => RustInformationCriterion::AIC,
            "bic" => RustInformationCriterion::BIC,
            _ => RustInformationCriterion::AICc,
        }
    }

    /// Classify a demand series.
    ///
    /// # Arguments
    /// * `y` - Demand time series (Float64Array)
    #[wasm_bindgen]
    pub fn classify(&self, y: &[f64]) -> AidResult {
        let y_col = Col::from_fn(y.len(), |i| y[i]);

        let classifier = RustAidClassifier::builder()
            .anomaly_alpha(self.anomaly_alpha)
            .intermittent_threshold(self.intermittent_threshold)
            .detect_anomalies(self.detect_anomalies)
            .ic(self.parse_ic())
            .build();

        AidResult {
            inner: classifier.classify(&y_col),
        }
    }
}

impl Default for AidClassifier {
    fn default() -> Self {
        Self::new()
    }
}
