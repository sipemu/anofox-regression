//! R's default random number generator (Mersenne-Twister with
//! `sample.kind = "Rejection"`), so that algorithms that draw random numbers
//! can replay R exactly (e.g. to validate against R or to reproduce an R
//! analysis). `RRng::new(s)` is `set.seed(s)`.

const N: usize = 624;
const M: usize = 397;
const I2_32M1: f64 = 2.328_306_437_080_797e-10;

/// R's Mersenne-Twister stream.
#[derive(Debug, Clone)]
pub struct RRng {
    mt: [u32; N],
    mti: usize,
}

impl RRng {
    /// `set.seed(seed)` (`RNGkind("Mersenne-Twister", "Inversion",
    /// "Rejection")`, R's default since 3.6.0).
    pub fn new(seed: i32) -> Self {
        let mut s = seed as u32;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        // dummy[0] (mti) is seeded too, then reset to 624 by FixupSeeds
        s = s.wrapping_mul(69069).wrapping_add(1);
        let mut mt = [0u32; N];
        for v in mt.iter_mut() {
            s = s.wrapping_mul(69069).wrapping_add(1);
            *v = s;
        }
        Self { mt, mti: N }
    }

    fn genrand(&mut self) -> f64 {
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        let mag = |y: u32| if y & 1 == 1 { MATRIX_A } else { 0 };
        if self.mti >= N {
            for kk in 0..N - M {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag(y);
            }
            for kk in N - M..N - 1 {
                let y = (self.mt[kk] & UPPER) | (self.mt[kk + 1] & LOWER);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag(y);
            }
            let y = (self.mt[N - 1] & UPPER) | (self.mt[0] & LOWER);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag(y);
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        f64::from(y) * 2.328_306_436_538_696_3e-10
    }

    /// `unif_rand()`: uniform on (0, 1).
    pub fn unif_rand(&mut self) -> f64 {
        let x = self.genrand();
        if x <= 0.0 {
            0.5 * I2_32M1
        } else if 1.0 - x <= 0.0 {
            1.0 - 0.5 * I2_32M1
        } else {
            x
        }
    }

    /// `runif(1, a, b)`.
    pub fn runif(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unif_rand()
    }

    /// `R_unif_index(dn)`: a uniform integer in `0..dn` by rejection
    /// sampling on 16-bit chunks.
    pub fn unif_index(&mut self, dn: usize) -> usize {
        if dn == 0 {
            return 0;
        }
        let bits = (dn as f64).log2().ceil() as u32;
        loop {
            let mut v: u64 = 0;
            let mut n = 0;
            while n <= bits {
                let v1 = (self.unif_rand() * 65536.0).floor() as u64;
                v = 65536 * v + v1;
                n += 16;
            }
            let dv = v & ((1u64 << bits) - 1);
            if (dv as usize) < dn {
                return dv as usize;
            }
        }
    }

    /// `sample.int(n, size, replace)`, 0-based.
    pub fn sample(&mut self, n: usize, size: usize, replace: bool) -> Vec<usize> {
        if replace || size < 2 {
            return (0..size).map(|_| self.unif_index(n)).collect();
        }
        let mut x: Vec<usize> = (0..n).collect();
        let mut m = n;
        (0..size.min(n))
            .map(|_| {
                let j = self.unif_index(m);
                let v = x[j];
                m -= 1;
                x[j] = x[m];
                v
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::RRng;

    /// Values from R 4.x: `set.seed(42); runif(3)`, then `sample(10)`,
    /// `sample(1000, 5, TRUE)` and `sample(7, 3)` from the same stream.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn replays_r() {
        let mut r = RRng::new(42);
        let u: Vec<f64> = (0..3).map(|_| r.unif_rand()).collect();
        assert_eq!(u, R_UNIF.to_vec());
        assert_eq!(r.sample(10, 10, false), R_PERM.to_vec());
        assert_eq!(r.sample(1000, 5, true), R_BOOT.to_vec());
        assert_eq!(r.sample(7, 3, false), R_SUB.to_vec());
    }

    include!("r_rng_reference.rs");
}
