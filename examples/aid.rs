//! # Automatic Identification of Demand (AID)
//!
//! AID classifies demand patterns into regular vs intermittent categories
//! and identifies the best-fitting distribution for forecasting.
//!
//! ## When to Use
//! - Demand forecasting and inventory management
//! - Classifying SKU demand patterns
//! - Identifying intermittent/lumpy demand
//! - Detecting demand anomalies
//!
//! ## Key Features
//! - Regular vs Intermittent classification
//! - Distribution fitting (Poisson, NegBin, Gamma, etc.)
//! - Anomaly detection (stockouts, new products, obsolescence)
//! - Information criteria for model selection
//!
//! Run with: `cargo run --example aid`

use anofox_regression::solvers::{AidClassifier, DemandType};
use faer::Col;

fn main() {
    println!("=== Automatic Identification of Demand (AID) ===\n");

    regular_demand();
    intermittent_demand();
    distribution_classification();
    anomaly_detection();
}

/// Regular demand pattern
fn regular_demand() {
    println!("--- Regular Demand Pattern ---\n");

    // Generate regular demand (consistent, few zeros)
    let n = 50;
    let demand = Col::from_fn(n, |i| {
        let base = 10.0;
        let trend = (i as f64) * 0.1;
        let seasonal = ((i as f64) * 0.5).sin() * 2.0;
        let noise = ((i as f64 * 1.3).sin()) * 1.5;
        (base + trend + seasonal + noise).round().max(1.0)
    });

    let classifier = AidClassifier::builder().intermittent_threshold(0.3).build();

    let result = classifier.classify(&demand);

    println!(
        "Demand sample (first 10): {:?}",
        (0..10).map(|i| demand[i] as i64).collect::<Vec<_>>()
    );

    println!("\nClassification result:");
    println!("  Demand type: {:?}", result.demand_type);
    println!("  Distribution: {:?}", result.distribution);

    println!("  Parameters: {:?}", result.parameters);

    // Summary statistics
    let mean: f64 = demand.iter().sum::<f64>() / n as f64;
    let n_zeros = demand.iter().filter(|&&d| d == 0.0).count();
    let zero_pct = 100.0 * n_zeros as f64 / n as f64;

    println!("\nDemand statistics:");
    println!("  Mean: {:.2}", mean);
    println!("  Zero observations: {} ({:.1}%)", n_zeros, zero_pct);

    match result.demand_type {
        DemandType::Regular => println!("  -> Classified as REGULAR (few zeros)"),
        DemandType::Intermittent => println!("  -> Classified as INTERMITTENT (many zeros)"),
    }
    println!();
}

/// Intermittent demand pattern
fn intermittent_demand() {
    println!("--- Intermittent Demand Pattern ---\n");

    // Generate intermittent demand (many zeros, sporadic orders)
    let n = 50;
    let demand = Col::from_fn(n, |i| {
        // Only ~30% of periods have demand
        let has_demand = ((i as f64 * 0.7).sin() + 0.5) > 0.0;
        if has_demand {
            let size = 5.0 + ((i as f64 * 1.1).sin().abs()) * 10.0;
            size.round()
        } else {
            0.0
        }
    });

    let classifier = AidClassifier::builder().intermittent_threshold(0.3).build();

    let result = classifier.classify(&demand);

    println!(
        "Demand sample (first 15): {:?}",
        (0..15).map(|i| demand[i] as i64).collect::<Vec<_>>()
    );

    println!("\nClassification result:");
    println!("  Demand type: {:?}", result.demand_type);
    println!("  Distribution: {:?}", result.distribution);

    // Summary statistics
    let non_zero: Vec<f64> = demand.iter().cloned().filter(|&d| d > 0.0).collect();
    let n_zeros = demand.iter().filter(|&&d| d == 0.0).count();
    let zero_pct = 100.0 * n_zeros as f64 / n as f64;
    let mean_when_demand = if !non_zero.is_empty() {
        non_zero.iter().sum::<f64>() / non_zero.len() as f64
    } else {
        0.0
    };

    println!("\nDemand statistics:");
    println!("  Zero observations: {} ({:.1}%)", n_zeros, zero_pct);
    println!("  Mean when demand occurs: {:.2}", mean_when_demand);

    match result.demand_type {
        DemandType::Regular => println!("  -> Classified as REGULAR"),
        DemandType::Intermittent => println!("  -> Classified as INTERMITTENT (sporadic demand)"),
    }

    println!("\nNote: Intermittent demand requires different forecasting methods");
    println!("      (e.g., Croston's method, SBA) than regular demand.");
    println!();
}

/// Distribution classification
fn distribution_classification() {
    println!("--- Distribution Classification ---\n");

    // Test different demand patterns
    let patterns = [
        ("Poisson-like", generate_poisson_like(60)),
        ("Overdispersed", generate_overdispersed(60)),
        ("Continuous", generate_continuous(60)),
    ];

    for (name, demand) in patterns {
        let classifier = AidClassifier::builder().build();
        let result = classifier.classify(&demand);

        let mean: f64 = demand.iter().sum::<f64>() / demand.nrows() as f64;
        let variance: f64 =
            demand.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / (demand.nrows() - 1) as f64;

        println!("{} demand:", name);
        println!(
            "  Mean: {:.2}, Variance: {:.2}, Var/Mean: {:.2}",
            mean,
            variance,
            variance / mean.max(0.001)
        );
        println!("  Fitted distribution: {:?}", result.distribution);
        if let Some(ic) = result.ic_values.get(&result.distribution) {
            println!("  AIC for best distribution: {:.2}", ic);
        }
        println!();
    }
}

/// Anomaly detection
fn anomaly_detection() {
    println!("--- Anomaly Detection ---\n");

    // Generate demand with anomalies
    let n = 60;
    let demand = Col::from_fn(n, |i| {
        // Normal demand
        let base = 15.0 + ((i as f64 * 0.8).sin()) * 3.0;

        // Inject anomalies
        if i >= 50 && i < 55 {
            0.0 // Stockout period
        } else if i < 5 {
            ((i as f64) * 3.0).round() // New product ramp-up
        } else if i > 55 {
            (base * 0.3).round() // Declining (obsolescence)
        } else {
            base.round()
        }
    });

    let classifier = AidClassifier::builder().detect_anomalies(true).build();

    let result = classifier.classify(&demand);

    println!("Demand pattern with injected anomalies:");
    println!("  - New product ramp-up (t=0-4)");
    println!("  - Stockout period (t=50-54)");
    println!("  - Declining demand (t=56+)\n");

    println!("Demand sample:");
    println!(
        "  First 5:  {:?}",
        (0..5).map(|i| demand[i] as i64).collect::<Vec<_>>()
    );
    println!(
        "  Normal:   {:?}",
        (25..30).map(|i| demand[i] as i64).collect::<Vec<_>>()
    );
    println!(
        "  Stockout: {:?}",
        (50..55).map(|i| demand[i] as i64).collect::<Vec<_>>()
    );
    println!(
        "  Decline:  {:?}",
        (55..60).map(|i| demand[i] as i64).collect::<Vec<_>>()
    );

    println!("\nClassification: {:?}", result.demand_type);
    println!("Distribution: {:?}", result.distribution);

    if !result.anomalies.is_empty() {
        println!("\nDetected anomalies:");
        for anomaly in &result.anomalies {
            println!("  {:?}", anomaly);
        }
    } else {
        println!("\nNo anomalies detected by the classifier.");
        println!("(Anomaly detection depends on statistical thresholds)");
    }
}

// Helper functions to generate different demand patterns

fn generate_poisson_like(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| {
        let lambda = 8.0_f64;
        let noise = ((i as f64 * 1.1).sin()) * lambda.sqrt();
        (lambda + noise).round().max(0.0)
    })
}

fn generate_overdispersed(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| {
        let mu = 10.0;
        // High variance relative to mean
        let noise = ((i as f64 * 0.9).sin()) * (mu * 1.5);
        (mu + noise).round().max(0.0)
    })
}

fn generate_continuous(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| {
        let mu = 12.5;
        let noise = ((i as f64 * 0.7).sin()) * 3.0;
        // Non-integer values
        mu + noise + 0.5
    })
}
