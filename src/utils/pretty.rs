//! R's `pretty()`: "nice" equally spaced values covering a range.

/// `pretty(c(lo, hi), n)` with R's defaults (`min.n = n %/% 3`,
/// `shrink.sml = 0.75`, `high.u.bias = 1.5`, `u5.bias = 0.5 + 1.5 *
/// high.u.bias`): about `n + 1` round values from a multiple of the unit
/// at or below `lo` to one at or above `hi`, the unit being 1, 2 or 5
/// times a power of ten (a port of `R_pretty`).
pub fn pretty(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    if !(lo.is_finite() && hi.is_finite()) {
        return Vec::new();
    }
    let min_n = n / 3;
    let (h, h5) = (1.5, 0.5 + 1.5 * 1.5);
    let dx = hi - lo;
    let ndiv = n.max(1) as f64;
    let (mut cell, i_small) = if dx == 0.0 && hi == 0.0 {
        (1.0, true)
    } else {
        let c = lo.abs().max(hi.abs());
        let u =
            (1.0 + if h5 >= 1.5 * h + 0.5 {
                1.0 / (1.0 + h)
            } else {
                1.5 / (1.0 + h5)
            }) * ndiv
                * f64::EPSILON;
        (c, dx < c * u * 3.0)
    };
    if i_small {
        if cell > 10.0 {
            cell = 9.0 + cell / 10.0;
        }
        cell *= 0.75;
        if min_n > 1 {
            cell /= min_n as f64;
        }
    } else {
        cell = dx;
        if n > 1 {
            cell /= n as f64;
        }
    }
    let subsmall = 2f64.powi(-20) * f64::MIN_POSITIVE;
    cell = cell.clamp(subsmall, f64::MAX / 1.25);
    let base = 10f64.powf(cell.log10().floor());
    let mut unit = base;
    if 2.0 * base - cell < h * (cell - unit) {
        unit = 2.0 * base;
        if 5.0 * base - cell < h5 * (cell - unit) {
            unit = 5.0 * base;
            if 10.0 * base - cell < h * (cell - unit) {
                unit = 10.0 * base;
            }
        }
    }
    const EPS: f64 = 1e-10;
    let mut ns = (lo / unit + EPS).floor();
    let mut nu = (hi / unit - EPS).ceil();
    while ns * unit > lo + EPS * unit {
        ns -= 1.0;
    }
    while nu * unit < hi - EPS * unit {
        nu += 1.0;
    }
    let k = (0.5 + nu - ns) as i64;
    let mut ndiv_out = k;
    if k < min_n as i64 {
        let k = min_n as i64 - k;
        if lo == 0.0 && ns == 0.0 && hi != 0.0 {
            nu += k as f64;
        } else if hi == 0.0 && nu == 0.0 && lo != 0.0 {
            ns -= k as f64;
        } else if ns >= 0.0 {
            nu += (k / 2) as f64;
            ns -= (k / 2 + k % 2) as f64;
        } else {
            ns -= (k / 2) as f64;
            nu += (k / 2 + k % 2) as f64;
        }
        ndiv_out = min_n as i64;
    }
    let (l, u) = ((ns * unit).min(lo), (nu * unit).max(hi));
    let m = ndiv_out.max(0) as usize;
    if m == 0 {
        return vec![l];
    }
    // seq.int(l, u, length.out = m + 1), with tiny values set to zero
    let by = (u - l) / m as f64;
    let delta = (u - l) / m as f64;
    (0..=m)
        .map(|i| {
            let v = if i == m { u } else { l + i as f64 * by };
            if v.abs() < 1e-14 * delta {
                0.0
            } else {
                v
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::pretty;

    /// `pretty(c(lo, hi), n)` from R 4.x, printed with 17 digits.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_r_pretty() {
        let cases: [(f64, f64, usize, &[f64]); 8] = [
            (
                0.0,
                1.0,
                5,
                &[
                    0.00000000000000000e+00,
                    2.00000000000000011e-01,
                    4.00000000000000022e-01,
                    6.00000000000000089e-01,
                    8.00000000000000044e-01,
                    1.00000000000000000e+00,
                ],
            ),
            (1.0, 10.0, 5, &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
            (
                23.0,
                77.0,
                10,
                &[
                    20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
                ],
            ),
            (
                -0.37,
                0.41,
                5,
                &[
                    -4.00000000000000022e-01,
                    -2.00000000000000011e-01,
                    0.00000000000000000e+00,
                    2.00000000000000067e-01,
                    4.00000000000000022e-01,
                    6.00000000000000089e-01,
                ],
            ),
            (5.0, 5.0, 5, &[0.0, 5.0]),
            (
                0.0123,
                0.0987,
                10,
                &[
                    1.00000000000000002e-02,
                    2.00000000000000039e-02,
                    3.00000000000000058e-02,
                    4.00000000000000078e-02,
                    5.00000000000000097e-02,
                    6.00000000000000117e-02,
                    7.00000000000000067e-02,
                    8.00000000000000017e-02,
                    9.00000000000000105e-02,
                    1.00000000000000006e-01,
                ],
            ),
            (
                -1.4,
                1.0,
                10,
                &[
                    -1.40000000000000013e+00,
                    -1.20000000000000018e+00,
                    -1.00000000000000000e+00,
                    -8.00000000000000044e-01,
                    -5.99999999999999978e-01,
                    -3.99999999999999911e-01,
                    -1.99999999999999956e-01,
                    0.00000000000000000e+00,
                    2.00000000000000178e-01,
                    4.00000000000000133e-01,
                    6.00000000000000311e-01,
                    8.00000000000000488e-01,
                    1.00000000000000000e+00,
                ],
            ),
            (
                1234.0,
                98765.0,
                7,
                &[0.0, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5],
            ),
        ];
        for (lo, hi, n, expected) in cases {
            assert_eq!(
                pretty(lo, hi, n),
                expected.to_vec(),
                "pretty({lo}, {hi}, {n})"
            );
        }
    }
}
