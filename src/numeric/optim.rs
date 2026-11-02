//! Derivative-free minimisers: 1-D golden-section search and the
//! Nelder–Mead simplex.

/// `(√5 − 1)/2` written as a decimal literal (one ulp above the computed
/// `(5f64.sqrt() - 1.0) / 2.0`); the GLMM θ search uses this value.
pub(crate) const INV_PHI: f64 = 0.618_033_988_749_895;

/// Golden-section minimisation of a unimodal `f` on `[a, b]`.
///
/// The bracket shrinks by `ratio` (the inverse golden ratio) per step,
/// re-evaluating one interior point: `c = b − ratio·(b − a)`,
/// `d = a + ratio·(b − a)`, keeping the side with the smaller value. Before
/// each step `done(a, b)` is asked whether to stop, for at most `max_iter`
/// steps. Returns the midpoint of the final bracket.
pub(crate) fn golden_section(
    f: &dyn Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    ratio: f64,
    done: &dyn Fn(f64, f64) -> bool,
    max_iter: usize,
) -> f64 {
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..max_iter {
        if done(a, b) {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - ratio * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + ratio * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// Nelder–Mead simplex minimisation of `f` from `x0` (reflection 1,
/// expansion 2, contraction ½, shrink ½; initial steps of half of each
/// coordinate, or ½ near zero). Stops when the cost spread of the simplex is
/// below `tol·(1 + |f_best|)` or after `max_iter` iterations; returns the best
/// vertex.
#[allow(clippy::needless_range_loop)]
pub(crate) fn nelder_mead(
    x0: &[f64],
    f: &dyn Fn(&[f64]) -> f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = x0.len();
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut xi = x0.to_vec();
        let step = if x0[i].abs() > 1e-6 {
            0.5 * x0[i].abs()
        } else {
            0.5
        };
        xi[i] += step;
        simplex.push(xi);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|x| f(x)).collect();

    for _ in 0..max_iter {
        // Order vertices by cost (ascending).
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        if (fvals[n] - fvals[0]).abs() < tol * (1.0 + fvals[0].abs()) {
            break;
        }

        // Centroid of the best n vertices.
        let mut cent = vec![0.0; n];
        for vertex in simplex.iter().take(n) {
            for (k, ck) in cent.iter_mut().enumerate() {
                *ck += vertex[k];
            }
        }
        for ck in &mut cent {
            *ck /= n as f64;
        }

        // Reflection.
        let xr: Vec<f64> = (0..n)
            .map(|k| cent[k] + alpha * (cent[k] - simplex[n][k]))
            .collect();
        let fr = f(&xr);
        if fr < fvals[0] {
            // Expansion.
            let xe: Vec<f64> = (0..n)
                .map(|k| cent[k] + gamma * (xr[k] - cent[k]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contraction.
            let xc: Vec<f64> = (0..n)
                .map(|k| cent[k] + rho * (simplex[n][k] - cent[k]))
                .collect();
            let fc = f(&xc);
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // Shrink toward the best vertex.
                for i in 1..=n {
                    for k in 0..n {
                        simplex[i][k] = simplex[0][k] + sigma * (simplex[i][k] - simplex[0][k]);
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    let mut best = 0;
    for i in 1..=n {
        if fvals[i] < fvals[best] {
            best = i;
        }
    }
    simplex[best].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_golden_ratio_literal() {
        let computed = (5f64.sqrt() - 1.0) / 2.0;
        assert_eq!(INV_PHI.to_bits(), computed.to_bits() + 1);
    }

    #[test]
    fn golden_section_finds_quadratic_minimum() {
        let f = |x: f64| (x - 1.3).powi(2);
        let g = (5f64.sqrt() - 1.0) / 2.0;
        let x = golden_section(&f, 0.0, 4.0, g, &|a, b| b - a <= 1e-10, usize::MAX);
        assert!((x - 1.3).abs() < 1e-8);
        let x = golden_section(&f, -2.0, 5.0, INV_PHI, &|a, b| b - a < 1e-9, 200);
        assert!((x - 1.3).abs() < 1e-8);
        // The iteration cap stops early.
        let x0 = golden_section(&f, 0.0, 4.0, g, &|_, _| false, 0);
        assert_eq!(x0, 2.0);
    }

    #[test]
    fn nelder_mead_finds_rosenbrock_minimum() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let x = nelder_mead(&[-1.2, 1.0], &f, 1e-14, 5000);
        assert!((x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3);
    }
}
