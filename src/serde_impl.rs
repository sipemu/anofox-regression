//! `serde` encodings for faer columns / matrices and non-finite floats
//! (feature `serde`, #321).
//!
//! Used through `#[serde(with = "...")]` on the fields of the fitted
//! models:
//!
//! * a finite `f64` is a plain number; NaN, `+∞` and `−∞` are the strings
//!   `"NaN"`, `"inf"` and `"-inf"`. JSON has no non-finite numbers
//!   (`serde_json` would write `null` and fail to read it back), and aliased
//!   coefficients and undefined statistics are NaN, so this keeps every
//!   format exact;
//! * a `Col<f64>` is an array of such numbers;
//! * a `Mat<f64>` is `{ "nrows", "ncols", "data" }` with `data` row-major.

use faer::{Col, Mat};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// One float on the wire.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Wire {
    Num(f64),
    Text(String),
}

fn to_wire(v: f64) -> Wire {
    if v.is_finite() {
        Wire::Num(v)
    } else if v.is_nan() {
        Wire::Text("NaN".into())
    } else if v > 0.0 {
        Wire::Text("inf".into())
    } else {
        Wire::Text("-inf".into())
    }
}

fn from_wire<E: serde::de::Error>(w: Wire) -> Result<f64, E> {
    match w {
        Wire::Num(v) => Ok(v),
        Wire::Text(s) => match s.as_str() {
            "NaN" => Ok(f64::NAN),
            "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            other => Err(E::custom(format!("invalid float {other:?}"))),
        },
    }
}

/// A list of floats (NaN-safe).
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct Floats(Vec<Wire>);

impl Floats {
    fn new(v: impl Iterator<Item = f64>) -> Self {
        Self(v.map(to_wire).collect())
    }
    fn values<E: serde::de::Error>(self) -> Result<Vec<f64>, E> {
        self.0.into_iter().map(from_wire).collect()
    }
}

/// `f64` (NaN-safe).
pub(crate) mod float {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        to_wire(*v).serialize(s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        from_wire(Wire::deserialize(d)?)
    }
}

/// `Option<f64>` (NaN-safe).
pub(crate) mod opt_float {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(v: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
        v.map(to_wire).serialize(s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
        Option::<Wire>::deserialize(d)?.map(from_wire).transpose()
    }
}

/// `Option<(f64, f64)>` (NaN-safe).
pub(crate) mod opt_pair {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(
        v: &Option<(f64, f64)>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        v.map(|(a, b)| (to_wire(a), to_wire(b))).serialize(s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<(f64, f64)>, D::Error> {
        Option::<(Wire, Wire)>::deserialize(d)?
            .map(|(a, b)| Ok((from_wire(a)?, from_wire(b)?)))
            .transpose()
    }
}

/// `Col<f64>` as an array.
pub(crate) mod col {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(v: &Col<f64>, s: S) -> Result<S::Ok, S::Error> {
        Floats::new(v.iter().copied()).serialize(s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Col<f64>, D::Error> {
        let v = Floats::deserialize(d)?.values()?;
        Ok(Col::from_fn(v.len(), |i| v[i]))
    }
}

/// `Option<Col<f64>>`.
pub(crate) mod opt_col {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(v: &Option<Col<f64>>, s: S) -> Result<S::Ok, S::Error> {
        v.as_ref()
            .map(|c| Floats::new(c.iter().copied()))
            .serialize(s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Col<f64>>, D::Error> {
        Option::<Floats>::deserialize(d)?
            .map(|f| {
                let v = f.values()?;
                Ok(Col::from_fn(v.len(), |i| v[i]))
            })
            .transpose()
    }
}

/// A matrix on the wire: dimensions and row-major data.
#[derive(Serialize, Deserialize)]
struct MatWire {
    nrows: usize,
    ncols: usize,
    data: Floats,
}

impl MatWire {
    fn new(m: &Mat<f64>) -> Self {
        let (r, c) = (m.nrows(), m.ncols());
        Self {
            nrows: r,
            ncols: c,
            data: Floats::new((0..r * c).map(|k| m[(k / c, k % c)])),
        }
    }
    fn into_mat<E: serde::de::Error>(self) -> Result<Mat<f64>, E> {
        let (r, c) = (self.nrows, self.ncols);
        let v = self.data.values::<E>()?;
        if v.len() != r * c {
            return Err(E::custom(format!(
                "matrix data has {} values, expected {r} × {c}",
                v.len()
            )));
        }
        Ok(Mat::from_fn(r, c, |i, j| v[i * c + j]))
    }
}

/// `Mat<f64>`.
pub(crate) mod mat {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(v: &Mat<f64>, s: S) -> Result<S::Ok, S::Error> {
        MatWire::new(v).serialize(s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Mat<f64>, D::Error> {
        MatWire::deserialize(d)?.into_mat()
    }
}

/// `Option<Mat<f64>>`.
pub(crate) mod opt_mat {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(v: &Option<Mat<f64>>, s: S) -> Result<S::Ok, S::Error> {
        v.as_ref().map(MatWire::new).serialize(s)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Mat<f64>>, D::Error> {
        Option::<MatWire>::deserialize(d)?
            .map(MatWire::into_mat)
            .transpose()
    }
}
