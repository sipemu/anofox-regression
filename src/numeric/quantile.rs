//! R-compatible sample quantiles and medians.

/// R's default (type 7) quantile of `sorted` (ascending) at probability `p`.
///
/// `h = (n − 1)·p` is rounded to an integer when within 4·eps of one (to
/// avoid spurious interpolation), otherwise neighbouring order statistics
/// are interpolated linearly. `p` is clamped to `[0, 1]`; NaN for an empty
/// slice or a non-finite `p`.
pub(crate) fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 || !p.is_finite() {
        return f64::NAN;
    }
    let h = (n - 1) as f64 * p.clamp(0.0, 1.0);
    let fuzz = 4.0 * f64::EPSILON;
    let lo = (h + fuzz).floor();
    let frac = h - lo;
    let lo = lo as usize;
    if frac.abs() < fuzz || lo + 1 >= n {
        sorted[lo.min(n - 1)]
    } else {
        (1.0 - frac) * sorted[lo] + frac * sorted[lo + 1]
    }
}

/// R-style median of `v` (mean of the two middle values for even length),
/// partially reordering `v` in place (`select_nth_unstable` with
/// `total_cmp`); NaN when `v` is empty.
pub(crate) fn median_in_place(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, hi, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *hi;
    if n % 2 == 1 {
        hi
    } else {
        let lo = v[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile7_matches_r() {
        // quantile(c(1,2,3,4), c(0, .1, .5, .9, 1), type = 7)
        let s = [1.0, 2.0, 3.0, 4.0];
        let want = [(0.0, 1.0), (0.1, 1.3), (0.5, 2.5), (0.9, 3.7), (1.0, 4.0)];
        for (p, q) in want {
            assert!((quantile7(&s, p) - q).abs() < 1e-12, "p = {p}");
        }
        // quantile(c(1, 2, 3, 5, 8), c(.025, .5, .975)) = 1.1, 3, 7.7
        let s = [1.0, 2.0, 3.0, 5.0, 8.0];
        assert!((quantile7(&s, 0.025) - 1.1).abs() < 1e-12);
        assert_eq!(quantile7(&s, 0.5), 3.0);
        assert!((quantile7(&s, 0.975) - 7.7).abs() < 1e-12);
        assert_eq!(quantile7(&[42.0], 0.3), 42.0);
    }

    #[test]
    fn quantile7_edge_cases() {
        assert!(quantile7(&[], 0.5).is_nan());
        assert!(quantile7(&[1.0, 2.0], f64::NAN).is_nan());
        assert!(quantile7(&[1.0, 2.0], f64::INFINITY).is_nan());
        assert_eq!(quantile7(&[1.0, 2.0], -0.5), 1.0);
        assert_eq!(quantile7(&[1.0, 2.0], 1.5), 2.0);
    }

    #[test]
    fn median_odd_even_and_empty() {
        assert_eq!(median_in_place(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_in_place(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_in_place(&mut [5.0]), 5.0);
        assert!(median_in_place(&mut []).is_nan());
    }
}
