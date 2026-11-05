//! Variance Inflation Factor (VIF) for multicollinearity detection.

use faer::{Col, Mat};

/// Squared relative residual norm below which a column counts as exactly
/// explained by the others (a VIF above `1e14`), or a column as constant.
const COLLINEAR_TOL: f64 = 1e-14;

/// Compute Variance Inflation Factor for each predictor.
///
/// VIF measures how much the variance of a coefficient estimate is inflated
/// due to multicollinearity. For predictor j:
///
/// VIF_j = 1 / (1 - R²_j)
///
/// where R²_j is the R² from regressing x_j on all other predictors (with an
/// intercept), as `car::vif` reports it.
///
/// # Interpretation
/// - VIF = 1: No correlation with other predictors
/// - VIF > 5: Moderate multicollinearity (some sources say > 10)
/// - VIF > 10: High multicollinearity
///
/// # Degenerate designs
/// - A column that is exactly a linear combination of the intercept and the
///   other columns (R²_j = 1) gets `f64::INFINITY`. That includes a constant
///   column (aliased with the intercept) and every column when there are too
///   few rows for the auxiliary regression to leave residual degrees of
///   freedom (n ≤ p, e.g. a 2×2 or 4×5 matrix with generic values). Columns
///   that are not part of such a dependency keep their finite VIF, computed
///   on the span of the other columns (`car::vif` instead stops with
///   "there are aliased coefficients in the model").
/// - A single predictor gets 1 (nothing to be collinear with), or `∞` if it
///   is constant.
/// - With no rows every VIF is `NaN`.
///
/// The result is never a silent 1 for an undefined VIF.
///
/// # Returns
/// Vector of VIF values, one per predictor column.
pub fn variance_inflation_factor(x: &Mat<f64>) -> Col<f64> {
    let vif = generalized_vif(x, &vec![1; x.ncols()]);
    Col::from_fn(x.ncols(), |j| vif[j])
}

/// Generalized VIF (GVIF) of Fox & Monette (1992) for groups of columns,
/// e.g. the dummy variables of a factor, as `car::vif` reports it.
///
/// For a group of `df` consecutive columns, with `R` the correlation matrix
/// of all predictors and `R₁₁`, `R₂₂` its blocks for the group and the
/// remaining columns:
///
/// `GVIF = det(R₁₁) · det(R₂₂) / det(R)`
///
/// computed as `det(R₁₁) / det(R₁₁ − R₁₂ R₂₂⁻ R₂₁)`, the generalized variance
/// of the group over its generalized variance after projecting out the other
/// columns. A single-column group gives the ordinary VIF. To compare groups of
/// different sizes use `GVIF^(1/(2·df))` (the `GVIF^(1/(2*Df))` column of
/// `car::vif`), the factor by which the confidence-interval width is inflated;
/// [`generalized_vif_terms`] returns it with the GVIF.
///
/// # Degenerate designs
/// - A group gets `f64::INFINITY` when some combination of its columns is
///   exactly explained by the intercept and the other columns, in particular
///   when it contains a constant column (zero variance), or when its own
///   columns are collinear. Only the groups involved in the dependency get
///   `∞`: the others keep a finite GVIF computed on the span of the remaining
///   columns, so a constant column does not affect the other terms.
///   `car::vif` stops with "there are aliased coefficients in the model"
///   instead.
/// - A group that is the only term (no other columns) gets 1, or `∞` if it is
///   degenerate itself.
/// - `NaN` marks a GVIF that is undefined: an empty group, `group_sizes` that
///   do not sum to the number of columns (every group), or no rows.
pub fn generalized_vif(x: &Mat<f64>, group_sizes: &[usize]) -> Vec<f64> {
    if group_sizes.iter().sum::<usize>() != x.ncols() || x.nrows() == 0 {
        return vec![f64::NAN; group_sizes.len()];
    }
    let cols = standardized_columns(x);
    let mut start = 0;
    group_sizes
        .iter()
        .map(|&size| {
            let range = start..start + size;
            start += size;
            if size == 0 {
                return f64::NAN;
            }
            group_gvif(&cols, range)
        })
        .collect()
}

/// GVIF of one term, with the adjusted value `car::vif` reports.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GvifTerm {
    /// The generalized VIF (the `GVIF` column of `car::vif`).
    pub gvif: f64,
    /// Number of columns of the term (the `Df` column).
    pub df: usize,
    /// `GVIF^(1/(2·df))` (the `GVIF^(1/(2*Df))` column): the inflation of the
    /// confidence-interval width, comparable across terms of different sizes.
    /// For a single column it is `√VIF`. `∞` when `gvif` is `∞`, `NaN` when
    /// `gvif` is `NaN` or `df` is 0.
    pub adjusted_gvif: f64,
}

/// [`generalized_vif`] with the degrees of freedom and the adjusted GVIF
/// `GVIF^(1/(2·df))` of each term: the `GVIF`, `Df` and `GVIF^(1/(2*Df))`
/// columns of `car::vif`. Degenerate designs are handled as in
/// [`generalized_vif`].
pub fn generalized_vif_terms(x: &Mat<f64>, group_sizes: &[usize]) -> Vec<GvifTerm> {
    generalized_vif(x, group_sizes)
        .into_iter()
        .zip(group_sizes)
        .map(|(gvif, &df)| GvifTerm {
            gvif,
            df,
            adjusted_gvif: adjusted_gvif(gvif, df),
        })
        .collect()
}

fn adjusted_gvif(gvif: f64, df: usize) -> f64 {
    if df == 0 {
        f64::NAN
    } else {
        gvif.powf(1.0 / (2.0 * df as f64))
    }
}

/// Centred columns of `x` scaled to unit length; `None` for a constant
/// column (zero variance, aliased with the intercept).
fn standardized_columns(x: &Mat<f64>) -> Vec<Option<Vec<f64>>> {
    let n = x.nrows();
    (0..x.ncols())
        .map(|j| {
            let raw: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
            let mean = raw.iter().sum::<f64>() / n as f64;
            let centred: Vec<f64> = raw.iter().map(|v| v - mean).collect();
            let ss = dot(&centred, &centred);
            if ss <= COLLINEAR_TOL * COLLINEAR_TOL * dot(&raw, &raw) {
                return None;
            }
            let norm = ss.sqrt();
            Some(centred.into_iter().map(|v| v / norm).collect())
        })
        .collect()
}

/// GVIF of the columns `range` against all other columns, by sequential
/// Gram–Schmidt: `det(R₁₁) = Π aₖ` and `det(R₁₁·₂) = Π bₖ`, with `aₖ` (`bₖ`)
/// the squared residual length of the k-th group column after projecting on
/// the previous group columns (and on the other columns).
fn group_gvif(cols: &[Option<Vec<f64>>], range: std::ops::Range<usize>) -> f64 {
    let mut others = Vec::new();
    for (j, c) in cols.iter().enumerate() {
        if let (false, Some(c)) = (range.contains(&j), c) {
            extend_basis(&mut others, c);
        }
    }
    let mut own: Vec<Vec<f64>> = Vec::new();
    let mut gvif = 1.0;
    for c in &cols[range] {
        let Some(c) = c else {
            return f64::INFINITY;
        };
        let a = extend_basis(&mut own, c);
        let b = extend_basis(&mut others, c);
        if a < COLLINEAR_TOL || b < COLLINEAR_TOL * a {
            return f64::INFINITY;
        }
        gvif *= a / b;
    }
    gvif.max(1.0)
}

/// Orthogonalize `v` against `basis` (two passes of modified Gram–Schmidt),
/// append the normalized residual unless `v` lies in the span, and return the
/// squared residual length.
fn extend_basis(basis: &mut Vec<Vec<f64>>, v: &[f64]) -> f64 {
    let mut r = v.to_vec();
    for _ in 0..2 {
        for q in basis.iter() {
            let d = dot(&r, q);
            r.iter_mut().zip(q).for_each(|(ri, qi)| *ri -= d * qi);
        }
    }
    let ss = dot(&r, &r);
    if ss >= COLLINEAR_TOL * dot(v, v) {
        let norm = ss.sqrt();
        basis.push(r.into_iter().map(|ri| ri / norm).collect());
    }
    ss
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(u, v)| u * v).sum()
}

/// Identify predictors with high multicollinearity.
///
/// Returns indices of predictors with VIF > threshold.
/// Common threshold: 5 or 10.
pub fn high_vif_predictors(vif: &Col<f64>, threshold: f64) -> Vec<usize> {
    vif.iter()
        .enumerate()
        .filter(|(_, &v)| v > threshold)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vif_orthogonal_predictors() {
        // Orthogonal predictors should have VIF = 1
        let mut x: Mat<f64> = Mat::zeros(100, 2);
        for i in 0..100 {
            x[(i, 0)] = (i as f64 * 0.1).sin();
            x[(i, 1)] = (i as f64 * 0.1).cos();
        }

        let vif = variance_inflation_factor(&x);

        // VIF should be close to 1 for orthogonal predictors
        assert!(
            (vif[0] - 1.0).abs() < 0.5,
            "VIF[0] = {} should be near 1 for orthogonal predictor",
            vif[0]
        );
        assert!(
            (vif[1] - 1.0).abs() < 0.5,
            "VIF[1] = {} should be near 1 for orthogonal predictor",
            vif[1]
        );
    }

    #[test]
    fn test_vif_collinear_predictors() {
        // Highly collinear predictors should have high VIF
        let mut x: Mat<f64> = Mat::zeros(100, 2);
        for i in 0..100 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = i as f64 + 0.01 * (i as f64).sin(); // Almost identical
        }

        let vif = variance_inflation_factor(&x);

        // VIF should be very high for collinear predictors
        assert!(vif[0] > 10.0, "VIF[0] = {} should be > 10", vif[0]);
        assert!(vif[1] > 10.0, "VIF[1] = {} should be > 10", vif[1]);
    }

    #[test]
    fn test_vif_minimum_is_one() {
        let x = Mat::from_fn(50, 3, |i, j| ((i + j * 17) as f64).sin());

        let vif = variance_inflation_factor(&x);

        for j in 0..vif.nrows() {
            assert!(vif[j] >= 1.0, "VIF[{}] = {} should be >= 1", j, vif[j]);
        }
    }

    #[test]
    fn test_high_vif_detection() {
        // Create data with one collinear predictor
        let mut x: Mat<f64> = Mat::zeros(50, 3);
        for i in 0..50 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = (i as f64).sin(); // Independent
            x[(i, 2)] = i as f64 * 1.01 + 0.5; // Collinear with x0
        }

        let vif = variance_inflation_factor(&x);
        let high = high_vif_predictors(&vif, 5.0);

        // x0 and x2 should have high VIF
        assert!(
            high.contains(&0) || high.contains(&2),
            "At least one collinear predictor should be flagged"
        );
    }
}
