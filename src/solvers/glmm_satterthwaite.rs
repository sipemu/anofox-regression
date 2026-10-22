//! Satterthwaite degrees of freedom for the fixed effects of a linear mixed
//! model, following `lmerTest` (Kuznetsova, Brockhoff & Christensen, *lmerTest
//! Package: Tests in Linear Mixed Effects Models*, JSS 2017).
//!
//! The variance parameters are `varpar = (θ, σ)`: the relative covariance
//! factor entries of every random-effect term and the residual standard
//! deviation. With `V(varpar) = σ² M⁻¹(θ)` the covariance of the fixed effects
//! and `A = 2 H⁻¹` the asymptotic covariance of `varpar` (`H` the Hessian of the
//! deviance in `varpar` at the optimum), the degrees of freedom of the
//! contrast `l` are
//!
//! ```text
//! df = 2 (lᵀ V l)² / (gᵀ A g),     g_k = lᵀ (∂V / ∂varpar_k) l
//! ```
//!
//! `lmerTest` takes `H` and `∂V/∂varpar` by Richardson-extrapolated finite
//! differences (`numDeriv::hessian` / `numDeriv::jacobian`); this module does
//! the same, and inverts `H` through its eigen decomposition, dropping
//! eigenvalues below `1e-8` (a Moore–Penrose inverse, as `lmerTest`).

/// The pieces of the penalized least-squares solution at a given `θ` that the
/// deviance and the fixed-effect covariance are built from.
pub(crate) struct LmmPieces {
    /// Penalized residual sum of squares `‖y − Xβ − ZΛu‖² + ‖u‖²`.
    pub prss: f64,
    /// `log|L_θ|²`.
    pub logdet_l2: f64,
    /// `log|R_X|²`, the log-determinant of the fixed-effects Schur complement.
    pub logdet_rx2: f64,
    /// `M⁻¹`, so that `vcov(β) = σ² M⁻¹`.
    pub m_inv: Vec<Vec<f64>>,
}

/// Eigenvalues below this are treated as zero when inverting the Hessian
/// (`lmerTest::as_lmerModLT`'s `tol`).
const EIGEN_TOL: f64 = 1e-8;

/// Satterthwaite degrees of freedom of every fixed effect (the unit contrasts
/// `l = e_j`), in design order. `pieces` solves the penalized least-squares
/// problem at a given `θ`. Entries are `NaN` where the denominator vanishes.
/// `None` if the problem cannot be solved at the optimum.
pub(crate) fn satterthwaite_df(
    theta: &[f64],
    sigma: f64,
    n: usize,
    p: usize,
    reml: bool,
    pieces: &dyn Fn(&[f64]) -> Option<LmmPieces>,
) -> Option<Vec<f64>> {
    let k = theta.len() + 1;
    let base = pieces(theta)?;

    // Deviance in varpar = (θ, σ) (lmerTest's devfun_vp).
    let two_pi = 2.0 * std::f64::consts::PI;
    let deviance = |vp: &[f64]| -> f64 {
        let s2 = vp[k - 1] * vp[k - 1];
        match pieces(&vp[..k - 1]) {
            Some(pc) => {
                let mut d = pc.logdet_l2 + pc.prss / s2 + n as f64 * (two_pi * s2).ln();
                if reml {
                    d += pc.logdet_rx2 - p as f64 * (two_pi * s2).ln();
                }
                d
            }
            None => f64::NAN,
        }
    };
    let mut x0 = theta.to_vec();
    x0.push(sigma);
    let h = hessian(&deviance, &x0);
    if h.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let a = scaled_pseudo_inverse(&h, 2.0);

    // ∂ diag(V) / ∂ varpar: numerically in θ, analytically in σ (V = σ² M⁻¹).
    let s2 = sigma * sigma;
    let diag_v = |th: &[f64]| -> Option<Vec<f64>> {
        pieces(th).map(|pc| (0..p).map(|j| s2 * pc.m_inv[j][j]).collect())
    };
    let mut grad: Vec<Vec<f64>> = Vec::with_capacity(k); // k × p
    for i in 0..k - 1 {
        grad.push(richardson_derivative(&diag_v, theta, i)?);
    }
    grad.push((0..p).map(|j| 2.0 * sigma * base.m_inv[j][j]).collect());

    Some(
        (0..p)
            .map(|j| {
                let var = s2 * base.m_inv[j][j];
                let g: Vec<f64> = (0..k).map(|r| grad[r][j]).collect();
                let denom: f64 = (0..k)
                    .map(|r| g[r] * (0..k).map(|c| a[r][c] * g[c]).sum::<f64>())
                    .sum();
                if denom > 0.0 && denom.is_finite() {
                    2.0 * var * var / denom
                } else {
                    f64::NAN
                }
            })
            .collect(),
    )
}

/// `numDeriv`'s base step: `d·|x| + eps` where `x` is (near) zero.
fn base_step(x: f64, d: f64) -> f64 {
    const EPS: f64 = 1e-4;
    let zero_tol = (f64::EPSILON / 7e-7).sqrt();
    (d * x).abs() + if x.abs() < zero_tol { EPS } else { 0.0 }
}

/// Number of Richardson levels (`numDeriv`'s `r = 4`, step ratio `v = 2`).
const LEVELS: usize = 4;

/// Richardson extrapolation of estimates whose error is even in the step
/// (`a[m]` taken at step `h / 2^m`).
fn extrapolate(mut a: Vec<f64>) -> f64 {
    for level in 1..a.len() {
        let f = 4f64.powi(level as i32);
        for m in 0..a.len() - level {
            a[m] = (f * a[m + 1] - a[m]) / (f - 1.0);
        }
    }
    a[0]
}

/// Derivative of the vector function `f` in coordinate `i` at `x`, by
/// Richardson-extrapolated central differences (`numDeriv::jacobian`).
fn richardson_derivative(
    f: &dyn Fn(&[f64]) -> Option<Vec<f64>>,
    x: &[f64],
    i: usize,
) -> Option<Vec<f64>> {
    let h0 = base_step(x[i], 1e-4);
    let mut levels: Vec<Vec<f64>> = Vec::with_capacity(LEVELS);
    for m in 0..LEVELS {
        let h = h0 / 2f64.powi(m as i32);
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let (fp, fm) = (f(&xp)?, f(&xm)?);
        levels.push(
            fp.iter()
                .zip(&fm)
                .map(|(a, b)| (a - b) / (2.0 * h))
                .collect(),
        );
    }
    let len = levels[0].len();
    Some(
        (0..len)
            .map(|j| extrapolate(levels.iter().map(|l| l[j]).collect()))
            .collect(),
    )
}

/// Hessian of the scalar function `f` at `x` by Richardson-extrapolated second
/// differences, with the relative step `d = 0.1` of `numDeriv::hessian`.
pub(crate) fn hessian(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<Vec<f64>> {
    let k = x.len();
    let f0 = f(x);
    let h0: Vec<f64> = x.iter().map(|&v| base_step(v, 0.1)).collect();
    let at = |steps: &[(usize, f64)]| -> f64 {
        let mut xs = x.to_vec();
        for &(i, s) in steps {
            xs[i] += s;
        }
        f(&xs)
    };
    let mut h = vec![vec![0.0; k]; k];
    for i in 0..k {
        let diag: Vec<f64> = (0..LEVELS)
            .map(|m| {
                let s = h0[i] / 2f64.powi(m as i32);
                (at(&[(i, s)]) - 2.0 * f0 + at(&[(i, -s)])) / (s * s)
            })
            .collect();
        h[i][i] = extrapolate(diag);
        for j in 0..i {
            let off: Vec<f64> = (0..LEVELS)
                .map(|m| {
                    let si = h0[i] / 2f64.powi(m as i32);
                    let sj = h0[j] / 2f64.powi(m as i32);
                    (at(&[(i, si), (j, sj)]) - at(&[(i, si), (j, -sj)]) - at(&[(i, -si), (j, sj)])
                        + at(&[(i, -si), (j, -sj)]))
                        / (4.0 * si * sj)
                })
                .collect();
            h[i][j] = extrapolate(off);
            h[j][i] = h[i][j];
        }
    }
    h
}

/// Newton polishing of a derivative-free minimum: up to `max_steps` Newton
/// steps `x ← x − H⁻¹∇f` with finite-difference derivatives, each taken only
/// if `H` is positive definite and the step lowers `f`. Nelder–Mead stops on
/// the spread of the simplex values, which leaves `x` accurate only to about
/// the square root of that tolerance; a few Newton steps from there recover
/// the optimum to the precision of `f` itself.
pub(crate) fn newton_polish(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_steps: usize) -> Vec<f64> {
    let k = x0.len();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    if !fx.is_finite() {
        return x;
    }
    let scalar = |v: &[f64]| Some(vec![f(v)]);
    for _ in 0..max_steps {
        let Some(grad) = (0..k)
            .map(|i| richardson_derivative(&scalar, &x, i).map(|g| g[0]))
            .collect::<Option<Vec<f64>>>()
        else {
            break;
        };
        let h = hessian(f, &x);
        let (values, vectors) = symmetric_eigen(&h);
        if values.iter().any(|&l| !(l > EIGEN_TOL && l.is_finite())) {
            break;
        }
        // step = −H⁻¹ g.
        let step: Vec<f64> = (0..k)
            .map(|r| {
                -(0..k)
                    .map(|e| {
                        let proj: f64 = (0..k).map(|c| vectors[c][e] * grad[c]).sum();
                        vectors[r][e] * proj / values[e]
                    })
                    .sum::<f64>()
            })
            .collect();
        if step.iter().any(|s| !s.is_finite()) {
            break;
        }
        let candidate: Vec<f64> = x.iter().zip(&step).map(|(a, s)| a + s).collect();
        let fc = f(&candidate);
        if fc.is_nan() || fc > fx {
            break;
        }
        let size = step
            .iter()
            .zip(&x)
            .map(|(s, a)| s.abs() / (1.0 + a.abs()))
            .fold(0.0, f64::max);
        x = candidate;
        fx = fc;
        if size < 1e-12 {
            break;
        }
    }
    x
}

/// `scale · H⁺` for a symmetric `H`: the eigen decomposition with eigenvalues
/// at or below [`EIGEN_TOL`] dropped.
fn scaled_pseudo_inverse(h: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    let k = h.len();
    let (values, vectors) = symmetric_eigen(h);
    let mut out = vec![vec![0.0; k]; k];
    for (e, &lambda) in values.iter().enumerate() {
        if lambda <= EIGEN_TOL {
            continue;
        }
        for r in 0..k {
            for c in 0..k {
                out[r][c] += scale * vectors[r][e] * vectors[c][e] / lambda;
            }
        }
    }
    out
}

/// Eigenvalues and eigenvectors (as columns) of a small symmetric matrix by
/// cyclic Jacobi rotations.
#[allow(clippy::needless_range_loop)]
fn symmetric_eigen(m: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = m.len();
    let mut a = m.to_vec();
    let mut v = vec![vec![0.0; k]; k];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..k)
            .flat_map(|r| (0..k).filter(move |&c| c != r).map(move |c| (r, c)))
            .map(|(r, c)| a[r][c] * a[r][c])
            .sum();
        let scale: f64 = (0..k)
            .map(|r| a[r][r] * a[r][r])
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..k {
            for q in p + 1..k {
                if a[p][q].abs() < f64::MIN_POSITIVE {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let t = if tau == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for r in 0..k {
                    let (arp, arq) = (a[r][p], a[r][q]);
                    a[r][p] = c * arp - s * arq;
                    a[r][q] = s * arp + c * arq;
                }
                for r in 0..k {
                    let (apr, aqr) = (a[p][r], a[q][r]);
                    a[p][r] = c * apr - s * aqr;
                    a[q][r] = s * apr + c * aqr;
                }
                for r in 0..k {
                    let (vrp, vrq) = (v[r][p], v[r][q]);
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }
    ((0..k).map(|i| a[i][i]).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_reconstructs_a_symmetric_matrix() {
        let m = vec![
            vec![4.0, 1.0, -2.0],
            vec![1.0, 3.0, 0.5],
            vec![-2.0, 0.5, 5.0],
        ];
        let (vals, vecs) = symmetric_eigen(&m);
        for r in 0..3 {
            for c in 0..3 {
                let rec: f64 = (0..3).map(|e| vecs[r][e] * vals[e] * vecs[c][e]).sum();
                assert!(
                    (rec - m[r][c]).abs() < 1e-12,
                    "({r},{c}) {rec} vs {}",
                    m[r][c]
                );
            }
        }
        // H⁺ of a full-rank matrix is its inverse.
        let inv = scaled_pseudo_inverse(&m, 1.0);
        for r in 0..3 {
            for c in 0..3 {
                let prod: f64 = (0..3).map(|j| m[r][j] * inv[j][c]).sum();
                let expect = if r == c { 1.0 } else { 0.0 };
                assert!((prod - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn finite_differences_are_accurate() {
        // f(x, y) = x³ y + exp(y): H = [[6xy, 3x²], [3x², exp(y)]].
        let f = |v: &[f64]| v[0].powi(3) * v[1] + v[1].exp();
        let h = hessian(&f, &[1.3, 0.7]);
        assert!((h[0][0] - 6.0 * 1.3 * 0.7).abs() < 1e-8);
        assert!((h[0][1] - 3.0 * 1.3 * 1.3).abs() < 1e-8);
        assert!((h[1][1] - 0.7f64.exp()).abs() < 1e-8);

        let g = |v: &[f64]| Some(vec![v[0].sin(), v[0] * v[0]]);
        let d = richardson_derivative(&g, &[0.4], 0).unwrap();
        assert!((d[0] - 0.4f64.cos()).abs() < 1e-10);
        assert!((d[1] - 0.8).abs() < 1e-10);
    }
}
