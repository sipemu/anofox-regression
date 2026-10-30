//! Designs with a term map: which coefficient columns belong to which
//! predictor, which are nonlinear, and which are interactions.
//!
//! A [`Design`] stacks [`Basis`] blocks (e.g. from [`crate::splines`]) into a
//! design matrix (no intercept column) and records a [`TermMap`] alongside.
//! The term map is what [`mod@crate::inference::anova`] needs for the
//! `rms::anova.rms` table: the overall test of a predictor including its
//! interactions, its nonlinear terms, the interaction tests and the pooled
//! totals.
//!
//! Interactions are two-way between main-effect terms, in either of the two
//! forms rms produces:
//!
//! * [`Design::add_interaction`] – the full tensor product, as `a * b` in an
//!   rms formula (columns `a_i·b_j` with the first factor varying fastest);
//! * [`Design::add_restricted_interaction`] – `a %ia% b`, without the doubly
//!   nonlinear products (see [`crate::splines::restricted_interaction`]).
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::design::Design;
//! use anofox_regression::splines::{Basis, Rcs};
//!
//! let x1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.7).sin() * 2.0).collect();
//! let x2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).cos()).collect();
//! let mut d = Design::new();
//! let a = d.add("x1", Rcs::from_data(&x1, 4).unwrap().basis(&x1, "x1")).unwrap();
//! let b = d.add("x2", Basis::linear(&x2, "x2")).unwrap();
//! d.add_interaction(a, b).unwrap();
//! assert_eq!(d.column_names(), ["x1", "x1'", "x1''", "x2", "x1 * x2", "x1' * x2", "x1'' * x2"]);
//! assert_eq!(d.matrix().ncols(), 7);
//! ```

use crate::splines::Basis;
use faer::Mat;
use thiserror::Error;

/// Errors from building a design or term map.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DesignError {
    /// A block has a different number of rows than the design.
    #[error("block has {got} rows, the design has {expected}")]
    RowMismatch { expected: usize, got: usize },
    /// A term index is out of range or names an interaction where a main
    /// effect is required.
    #[error("term {0} is not a main-effect term of this design")]
    NotAMainEffect(usize),
    /// The two factors of an interaction are the same term.
    #[error("an interaction needs two different terms")]
    SelfInteraction,
    /// Other invalid input.
    #[error("{0}")]
    Invalid(String),
}

/// The two factors of a two-way interaction term.
#[derive(Debug, Clone, PartialEq)]
pub struct Interaction {
    /// Term indices of the two main effects.
    pub factors: (usize, usize),
    /// Per column: whether the component from the first / second factor is
    /// nonlinear.
    pub components: Vec<(bool, bool)>,
    /// `true` for a restricted (`%ia%`) interaction.
    pub restricted: bool,
}

/// One term of a design.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    /// Term name (`x1`, or `x1 * x2` for an interaction).
    pub name: String,
    /// Indices of its columns among the slope coefficients (no intercept).
    pub columns: Vec<usize>,
    /// Per column: nonlinear (or involving a nonlinear component).
    pub nonlinear: Vec<bool>,
    /// The factors, for an interaction term.
    pub interaction: Option<Interaction>,
}

/// Which slope columns belong to which term.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TermMap {
    terms: Vec<Term>,
    n_columns: usize,
}

impl TermMap {
    /// Empty term map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a main-effect term with one column per entry of `nonlinear`;
    /// returns its index.
    pub fn add_main(&mut self, name: &str, nonlinear: &[bool]) -> usize {
        let start = self.n_columns;
        self.n_columns += nonlinear.len();
        self.terms.push(Term {
            name: name.to_string(),
            columns: (start..self.n_columns).collect(),
            nonlinear: nonlinear.to_vec(),
            interaction: None,
        });
        self.terms.len() - 1
    }

    /// Append the interaction of main-effect terms `a` and `b`: the full
    /// tensor product (`restricted = false`, first factor fastest) or the
    /// rms `%ia%` product without doubly nonlinear columns. Returns its index.
    pub fn add_interaction(
        &mut self,
        a: usize,
        b: usize,
        restricted: bool,
    ) -> Result<usize, DesignError> {
        let pairs = self.pairs(a, b, restricted)?;
        let (ta, tb) = (&self.terms[a], &self.terms[b]);
        let components: Vec<(bool, bool)> = pairs
            .iter()
            .map(|&(i, j)| (ta.nonlinear[i], tb.nonlinear[j]))
            .collect();
        let name = format!("{} * {}", ta.name, tb.name);
        let start = self.n_columns;
        self.n_columns += pairs.len();
        self.terms.push(Term {
            name,
            columns: (start..self.n_columns).collect(),
            nonlinear: components.iter().map(|&(x, y)| x || y).collect(),
            interaction: Some(Interaction {
                factors: (a, b),
                components,
                restricted,
            }),
        });
        Ok(self.terms.len() - 1)
    }

    /// Column pairs `(i of a, j of b)` of an interaction, in column order.
    fn pairs(
        &self,
        a: usize,
        b: usize,
        restricted: bool,
    ) -> Result<Vec<(usize, usize)>, DesignError> {
        for t in [a, b] {
            if self.terms.get(t).is_none_or(|t| t.interaction.is_some()) {
                return Err(DesignError::NotAMainEffect(t));
            }
        }
        if a == b {
            return Err(DesignError::SelfInteraction);
        }
        let (na, nb) = (&self.terms[a].nonlinear, &self.terms[b].nonlinear);
        let mut pairs = Vec::new();
        if restricted {
            for (i, &nl_a) in na.iter().enumerate() {
                for (j, &nl_b) in nb.iter().enumerate() {
                    if nl_a && nl_b {
                        break;
                    }
                    pairs.push((i, j));
                }
            }
        } else {
            for j in 0..nb.len() {
                for i in 0..na.len() {
                    pairs.push((i, j));
                }
            }
        }
        Ok(pairs)
    }

    /// The terms, in design order.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// Number of slope columns.
    pub fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// Index of the term called `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.terms.iter().position(|t| t.name == name)
    }
}

/// A design matrix (without intercept column) built from basis blocks, with
/// its [`TermMap`].
#[derive(Debug, Clone, Default)]
pub struct Design {
    n_rows: Option<usize>,
    /// Column-major values.
    values: Vec<Vec<f64>>,
    names: Vec<String>,
    /// Basis of every term (for interactions, of the product).
    blocks: Vec<Basis>,
    terms: TermMap,
}

impl Design {
    /// Empty design.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a main-effect term made of `basis`; returns its term index.
    pub fn add(&mut self, name: &str, basis: Basis) -> Result<usize, DesignError> {
        self.check_rows(basis.columns.nrows())?;
        let t = self.terms.add_main(name, &basis.nonlinear);
        self.push_block(basis);
        Ok(t)
    }

    /// Append the full tensor-product interaction of main-effect terms `a`
    /// and `b` (rms `a * b`); returns its term index.
    pub fn add_interaction(&mut self, a: usize, b: usize) -> Result<usize, DesignError> {
        self.interaction(a, b, false)
    }

    /// Append the restricted interaction `a %ia% b` (no doubly nonlinear
    /// products); returns its term index.
    pub fn add_restricted_interaction(&mut self, a: usize, b: usize) -> Result<usize, DesignError> {
        self.interaction(a, b, true)
    }

    fn interaction(&mut self, a: usize, b: usize, restricted: bool) -> Result<usize, DesignError> {
        let pairs = self.terms.pairs(a, b, restricted)?;
        let (ba, bb) = (&self.blocks[a], &self.blocks[b]);
        let n = ba.columns.nrows();
        let basis = Basis {
            columns: Mat::from_fn(n, pairs.len(), |r, c| {
                let (i, j) = pairs[c];
                ba.columns[(r, i)] * bb.columns[(r, j)]
            }),
            names: pairs
                .iter()
                .map(|&(i, j)| format!("{} * {}", ba.names[i], bb.names[j]))
                .collect(),
            nonlinear: pairs
                .iter()
                .map(|&(i, j)| ba.nonlinear[i] || bb.nonlinear[j])
                .collect(),
        };
        let t = self.terms.add_interaction(a, b, restricted)?;
        self.push_block(basis);
        Ok(t)
    }

    fn check_rows(&mut self, rows: usize) -> Result<(), DesignError> {
        match self.n_rows {
            Some(n) if n != rows => Err(DesignError::RowMismatch {
                expected: n,
                got: rows,
            }),
            _ => {
                self.n_rows = Some(rows);
                Ok(())
            }
        }
    }

    fn push_block(&mut self, basis: Basis) {
        for j in 0..basis.columns.ncols() {
            self.values.push(
                (0..basis.columns.nrows())
                    .map(|i| basis.columns[(i, j)])
                    .collect(),
            );
        }
        self.names.extend(basis.names.iter().cloned());
        self.blocks.push(basis);
    }

    /// The design matrix (`n × columns`, no intercept column).
    pub fn matrix(&self) -> Mat<f64> {
        let n = self.n_rows.unwrap_or(0);
        Mat::from_fn(n, self.values.len(), |i, j| self.values[j][i])
    }

    /// Column names.
    pub fn column_names(&self) -> &[String] {
        &self.names
    }

    /// The term map.
    pub fn terms(&self) -> &TermMap {
        &self.terms
    }

    /// Basis block of term `t`.
    pub fn block(&self, t: usize) -> Option<&Basis> {
        self.blocks.get(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_and_restricted_orders_match_rms() {
        let mut m = TermMap::new();
        let a = m.add_main("x1", &[false, true, true]);
        let b = m.add_main("x2", &[false, true]);
        let full = m.add_interaction(a, b, false).unwrap();
        let ia = m.add_interaction(a, b, true).unwrap();
        // rms `rcs(x1,4)*rcs(x2,3)`: x1*x2, x1'*x2, x1''*x2, x1*x2', x1'*x2', x1''*x2'.
        let c = &m.terms()[full].interaction.as_ref().unwrap().components;
        assert_eq!(
            c,
            &[
                (false, false),
                (true, false),
                (true, false),
                (false, true),
                (true, true),
                (true, true)
            ]
        );
        // `%ia%`: x1*x2, x1*x2', x1'*x2, x1''*x2.
        let c = &m.terms()[ia].interaction.as_ref().unwrap().components;
        assert_eq!(
            c,
            &[(false, false), (false, true), (true, false), (true, false)]
        );
        assert_eq!(m.terms()[ia].columns, (11..15).collect::<Vec<_>>());
        assert_eq!(m.n_columns(), 15);
        assert_eq!(m.find("x1 * x2"), Some(full));
    }

    #[test]
    fn rejects_bad_interactions_and_rows() {
        let mut m = TermMap::new();
        let a = m.add_main("a", &[false]);
        let b = m.add_main("b", &[false]);
        let ab = m.add_interaction(a, b, false).unwrap();
        assert_eq!(
            m.add_interaction(a, a, false),
            Err(DesignError::SelfInteraction)
        );
        assert_eq!(
            m.add_interaction(ab, a, false),
            Err(DesignError::NotAMainEffect(ab))
        );
        assert_eq!(
            m.add_interaction(a, 9, false),
            Err(DesignError::NotAMainEffect(9))
        );

        let mut d = Design::new();
        d.add("a", Basis::linear(&[1.0, 2.0], "a")).unwrap();
        assert!(d.add("b", Basis::linear(&[1.0], "b")).is_err());
    }
}
