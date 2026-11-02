//! Crate-private macros.

/// Fluent setters for the public fields of an options struct: for each
/// `field: Type`, `fn field(self, value: Type) -> Self`. Options structs are
/// `#[non_exhaustive]`, so callers outside the crate cannot use struct
/// literals; they write `Options::default().tau(0.5).seed(7)`.
macro_rules! setters {
    ($ty:ident { $($field:ident: $t:ty),* $(,)? }) => {
        impl $ty {
            $(
                #[doc = concat!("Set `", stringify!($field), "` (see the field's documentation).")]
                #[must_use]
                pub fn $field(mut self, value: $t) -> Self {
                    self.$field = value;
                    self
                }
            )*
        }
    };
}
