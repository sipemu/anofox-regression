//! Conversion of Rust result types to plain JavaScript values.
//!
//! The core crate's result structs don't implement `serde::Serialize`, so
//! [`ToJs`] converts them field by field: numbers stay numbers, faer columns
//! and `Vec`s become arrays, matrices become arrays of rows, `Option::None`
//! becomes `undefined`, and structs become objects with camelCase keys
//! ([`impl_to_js!`]).

use faer::{Col, Mat};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

/// A value that can be handed to JavaScript.
pub trait ToJs {
    /// The JavaScript value.
    fn to_js(&self) -> JsValue;
}

macro_rules! to_js_number {
    ($($t:ty),*) => {$(
        impl ToJs for $t {
            fn to_js(&self) -> JsValue {
                JsValue::from_f64(*self as f64)
            }
        }
    )*};
}
to_js_number!(f64, f32, usize, u32, i32, i64, u64);

impl ToJs for bool {
    fn to_js(&self) -> JsValue {
        JsValue::from_bool(*self)
    }
}

impl ToJs for String {
    fn to_js(&self) -> JsValue {
        JsValue::from_str(self)
    }
}

impl ToJs for &str {
    fn to_js(&self) -> JsValue {
        JsValue::from_str(self)
    }
}

impl<T: ToJs> ToJs for Option<T> {
    fn to_js(&self) -> JsValue {
        match self {
            Some(v) => v.to_js(),
            None => JsValue::UNDEFINED,
        }
    }
}

impl<T: ToJs> ToJs for [T] {
    fn to_js(&self) -> JsValue {
        self.iter().map(ToJs::to_js).collect::<Array>().into()
    }
}

impl<T: ToJs> ToJs for Vec<T> {
    fn to_js(&self) -> JsValue {
        self.as_slice().to_js()
    }
}

impl<T: ToJs, const N: usize> ToJs for [T; N] {
    fn to_js(&self) -> JsValue {
        self.as_slice().to_js()
    }
}

impl<A: ToJs, B: ToJs> ToJs for (A, B) {
    fn to_js(&self) -> JsValue {
        Array::of2(&self.0.to_js(), &self.1.to_js()).into()
    }
}

impl<A: ToJs, B: ToJs, C: ToJs> ToJs for (A, B, C) {
    fn to_js(&self) -> JsValue {
        Array::of3(&self.0.to_js(), &self.1.to_js(), &self.2.to_js()).into()
    }
}

impl ToJs for Col<f64> {
    fn to_js(&self) -> JsValue {
        self.iter()
            .map(|v| JsValue::from_f64(*v))
            .collect::<Array>()
            .into()
    }
}

impl ToJs for Mat<f64> {
    fn to_js(&self) -> JsValue {
        (0..self.nrows())
            .map(|i| {
                (0..self.ncols())
                    .map(|j| JsValue::from_f64(self[(i, j)]))
                    .collect::<Array>()
            })
            .collect::<Array>()
            .into()
    }
}

impl<T: ToJs + ?Sized> ToJs for &T {
    fn to_js(&self) -> JsValue {
        (**self).to_js()
    }
}

/// `snake_case` → `camelCase`.
pub fn camel(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut upper = false;
    for ch in name.chars() {
        if ch == '_' {
            upper = true;
        } else if upper {
            out.extend(ch.to_uppercase());
            upper = false;
        } else {
            out.push(ch);
        }
    }
    out
}

/// A JS object from `(snake_case key, value)` pairs.
pub fn object(fields: &[(&str, JsValue)]) -> JsValue {
    let obj = Object::new();
    for (k, v) in fields {
        // Setting a property on a fresh plain object cannot fail.
        let _ = Reflect::set(&obj, &JsValue::from_str(&camel(k)), v);
    }
    obj.into()
}

/// Implement [`ToJs`] for a struct with public fields, as an object with
/// camelCase keys. Methods whose results are added as entries can follow
/// `;` as `name()`.
macro_rules! impl_to_js {
    ($t:ty { $($f:ident),* $(,)? } $(; $($m:ident()),* $(,)?)?) => {
        impl $crate::convert::ToJs for $t {
            fn to_js(&self) -> wasm_bindgen::JsValue {
                let s = self;
                $crate::convert::object(&[
                    $((stringify!($f), $crate::convert::ToJs::to_js(&s.$f)),)*
                    $($((stringify!($m), $crate::convert::ToJs::to_js(&s.$m())),)*)?
                ])
            }
        }
    };
}

/// `CamelCase` → `snake_case`.
pub fn snake(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// Implement [`ToJs`] for a fieldless enum as its `snake_case` `Debug` name.
macro_rules! impl_to_js_debug {
    ($($t:ty),* $(,)?) => {$(
        impl $crate::convert::ToJs for $t {
            fn to_js(&self) -> wasm_bindgen::JsValue {
                wasm_bindgen::JsValue::from_str(&$crate::convert::snake(&format!("{:?}", self)))
            }
        }
    )*};
}

pub(crate) use impl_to_js;
pub(crate) use impl_to_js_debug;
