//! Micro-benchmark for many small OLS fits (#129): 100K fits of 20×3, as
//! a per-group `GROUP BY` fit does, reporting allocations per fit and time.
//!
//! Run with: `cargo run --release --example many_small_fits`

use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn main() {
    let (n, p, fits) = (20, 3, 100_000);
    let x = Mat::from_fn(n, p, |i, j| ((i * 7 + j * 13) % 11) as f64 + 0.1 * j as f64);
    let y = Col::from_fn(n, |i| {
        1.0 + 0.5 * x[(i, 0)] - 0.2 * x[(i, 2)] + ((i * 3) % 5) as f64 * 0.1
    });
    for (label, inference) in [
        ("compute_inference(true) ", true),
        ("compute_inference(false)", false),
    ] {
        let model = OlsRegressor::builder().compute_inference(inference).build();
        let mut sink = 0.0;
        let a0 = ALLOCS.load(Ordering::Relaxed);
        let t0 = Instant::now();
        for _ in 0..fits {
            sink += model.fit(&x, &y).unwrap().result().coefficients[0];
        }
        let dt = t0.elapsed();
        let allocs = ALLOCS.load(Ordering::Relaxed) - a0;
        println!(
            "{label}: {:.2} µs/fit, {:.1} allocations/fit (checksum {sink:.6})",
            dt.as_secs_f64() * 1e6 / fits as f64,
            allocs as f64 / fits as f64
        );
    }
}
