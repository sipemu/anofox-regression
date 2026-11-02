//! WebAssembly bindings for anofox-regression.
//!
//! This crate provides JavaScript/TypeScript bindings for the anofox-regression
//! statistical regression library using wasm-bindgen.

use wasm_bindgen::prelude::*;

mod convert;
pub mod design;
pub mod diagnostics;
pub mod inference;
pub mod regressors;

/// Initialize the WASM module.
///
/// This is called automatically when the module is loaded.
#[wasm_bindgen(start)]
pub fn init() {
    // Future: set up panic hook for better error messages in the browser console
}

/// Get the library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
