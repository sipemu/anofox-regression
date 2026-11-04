//! Data loading for the analysis template. Everything statistical comes from
//! anofox-regression itself (splines, design, tests, effects, validation).
#![allow(dead_code)]

/// A CSV loaded column-wise. Empty cells / "NA" / unparsable numbers become NaN (numeric view);
/// the raw strings are kept for categorical columns.
pub struct Frame {
    pub names: Vec<String>,
    pub raw: Vec<Vec<String>>, // column-major
}

impl Frame {
    pub fn read_csv(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let names: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();
        let mut raw = vec![Vec::new(); names.len()];
        for rec in rdr.records() {
            let rec = rec?;
            for (j, v) in rec.iter().enumerate() {
                raw[j].push(v.trim().to_string());
            }
        }
        Ok(Self { names, raw })
    }
    pub fn n(&self) -> usize {
        self.raw.first().map_or(0, |c| c.len())
    }
    fn idx(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no column {name}"))
    }
    pub fn num(&self, name: &str) -> Vec<f64> {
        self.raw[self.idx(name)]
            .iter()
            .map(|s| if s.is_empty() || s == "NA" { f64::NAN } else { s.parse().unwrap_or(f64::NAN) })
            .collect()
    }
    pub fn str(&self, name: &str) -> Vec<String> {
        self.raw[self.idx(name)].clone()
    }
    /// Fraction missing per column (empty / NA).
    pub fn missingness(&self) -> Vec<(String, f64)> {
        let n = self.n().max(1) as f64;
        self.names
            .iter()
            .zip(&self.raw)
            .map(|(nm, c)| (nm.clone(), c.iter().filter(|s| s.is_empty() || *s == "NA").count() as f64 / n))
            .collect()
    }
}

/// SplitMix64 — tiny deterministic RNG so the template needs no `rand` dependency.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn bootstrap(&mut self, n: usize) -> Vec<usize> {
        (0..n).map(|_| self.below(n)).collect()
    }
}

