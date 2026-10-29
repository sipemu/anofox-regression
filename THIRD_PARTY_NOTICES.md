# Third Party Notices

This library contains independent Rust implementations inspired by algorithms and methodologies from several open-source projects. We gratefully acknowledge their contributions.

**Note**: This library is licensed under MIT. The attributions below are for methodological inspiration and academic courtesy. No source code was copied from these projects; all implementations are original Rust code.

## greybox

**Repository**: https://github.com/config-i1/greybox
**Author**: Ivan Svetunkov
**Original License**: LGPL-2.1

The Augmented Linear Model (ALM) methodology is inspired by the greybox R package. This includes:
- ALM distribution families (24 distributions)
- Maximum likelihood estimation approach
- AID (Automatic Identification of Demand) classification methodology

Our implementation is an independent clean-room Rust implementation validated against R's output, not a derivative work of greybox's source code.

## basin (MIT/Apache-2.0)

**Repository**: https://github.com/jolars/basin

Used for numerical optimization:
- L-BFGS optimization for ALM and Elastic Net
- More-Thuente line search

```
Copyright (c) 2026 Johan Larsson and basin contributors
License: MIT OR Apache-2.0
```

## faer (MIT)

**Repository**: https://github.com/sarah-ek/faer-rs

High-performance linear algebra library used for:
- Matrix operations
- QR decomposition for OLS
- Cholesky decomposition

```
Copyright (c) Sarah El-Kazdadi
License: MIT
```

## statrs (MIT)

**Repository**: https://github.com/statrs-dev/statrs

Statistical distributions and functions used for:
- Probability distributions (Normal, Gamma, Beta, etc.)
- Statistical functions (CDF, PDF, quantile)

```
Copyright (c) statrs developers
License: MIT
```

## R Statistical Computing

Several algorithms are validated against and inspired by R's implementation:
- `stats::lm()` - Linear model methodology
- `stats::glm()` - Generalized linear model framework
- `MASS::glm.nb()` - Negative binomial regression
- `glmnet::glmnet()` - Elastic net methodology
- `statmod::tweedie()` - Tweedie distribution
- `car::vif()` - Variance inflation factor

---

All licenses permit use in this MIT-licensed library with proper attribution.
