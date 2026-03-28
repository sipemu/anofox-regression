use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use anofox_regression::solvers::{AidClassifier, AidClassifierBuilder};
use faer::Col;

/// Generate Poisson-like count demand data.
fn generate_poisson_like(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| {
        let base = 10.0;
        let variation = ((i as f64 * 0.3).sin() * 3.0).round();
        (base + variation).max(0.0)
    })
}

/// Generate intermittent demand data (many zeros).
fn generate_intermittent(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| {
        if i % 3 == 0 {
            (5.0 + (i as f64 * 0.2).sin() * 2.0).round().max(1.0)
        } else {
            0.0
        }
    })
}

/// Generate continuous/fractional demand data.
fn generate_continuous(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| {
        let base = 15.5;
        let variation = (i as f64 * 0.1).sin() * 3.0;
        (base + variation).max(0.1)
    })
}

/// Generate data with anomalies (stockouts + outliers).
fn generate_with_anomalies(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| {
        if i == n / 4 || i == n / 2 {
            0.0 // stockouts
        } else if i == 3 * n / 4 {
            100.0 // outlier
        } else {
            10.0 + (i as f64 * 0.1).sin() * 2.0
        }
    })
}

fn bench_classify_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("aid_classify");

    for &n in &[50, 100, 500, 1000] {
        let poisson_data = generate_poisson_like(n);
        group.bench_with_input(BenchmarkId::new("poisson_like", n), &poisson_data, |b, y| {
            b.iter(|| {
                let classifier = AidClassifier::new();
                black_box(classifier.classify(black_box(y)))
            })
        });

        let intermittent_data = generate_intermittent(n);
        group.bench_with_input(
            BenchmarkId::new("intermittent", n),
            &intermittent_data,
            |b, y| {
                b.iter(|| {
                    let classifier = AidClassifier::new();
                    black_box(classifier.classify(black_box(y)))
                })
            },
        );

        let continuous_data = generate_continuous(n);
        group.bench_with_input(
            BenchmarkId::new("continuous", n),
            &continuous_data,
            |b, y| {
                b.iter(|| {
                    let classifier = AidClassifier::new();
                    black_box(classifier.classify(black_box(y)))
                })
            },
        );
    }

    group.finish();
}

fn bench_classify_no_anomalies(c: &mut Criterion) {
    let mut group = c.benchmark_group("aid_classify_no_anomalies");

    for &n in &[50, 100, 500, 1000] {
        let poisson_data = generate_poisson_like(n);
        group.bench_with_input(BenchmarkId::new("poisson_like", n), &poisson_data, |b, y| {
            b.iter(|| {
                let classifier = AidClassifier::builder()
                    .detect_anomalies(false)
                    .build();
                black_box(classifier.classify(black_box(y)))
            })
        });
    }

    group.finish();
}

fn bench_anomaly_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("aid_anomaly_overhead");

    for &n in &[100, 500, 1000] {
        let data = generate_with_anomalies(n);

        group.bench_with_input(BenchmarkId::new("with_anomalies", n), &data, |b, y| {
            b.iter(|| {
                let classifier = AidClassifier::builder()
                    .detect_anomalies(true)
                    .build();
                black_box(classifier.classify(black_box(y)))
            })
        });

        group.bench_with_input(BenchmarkId::new("without_anomalies", n), &data, |b, y| {
            b.iter(|| {
                let classifier = AidClassifier::builder()
                    .detect_anomalies(false)
                    .build();
                black_box(classifier.classify(black_box(y)))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_classify_full,
    bench_classify_no_anomalies,
    bench_anomaly_detection,
);
criterion_main!(benches);
