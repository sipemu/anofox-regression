# Python-based validation oracle

The estimators introduced in v0.5.5 (`TheilSen`, `RANSAC`, `PassiveAggressive`,
`Lars`, `LassoLars`, `BayesianRidge`, `ARDRegression`) have their canonical
reference implementations in scikit-learn rather than in R. This directory
holds a pinned, reproducible Python environment plus the oracle scripts that
emit Rust integration-test constants the same way the R scripts under
`tests/r_scripts/` do for the GLM-style estimators.

## Reproduce the environment

Requires [`uv`](https://github.com/astral-sh/uv) (`cargo install uv` or
[install instructions](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
cd validation/python
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Pinned versions

| Package      | Version |
|--------------|---------|
| Python       | 3.12    |
| scikit-learn | 1.5.2   |
| numpy        | 2.1.3   |

These are pinned because numerical defaults (default `tol`, default
`max_iter`, default RNG behavior) have shifted between sklearn releases.
The Rust integration tests embed reference values produced by exactly these
versions — bump them deliberately, then regenerate the fixtures.

## Re-generate the fixtures

Each script is invoked from this directory and writes a single block of
Rust source to stdout. The output is piped into the corresponding
`tests/fixtures/<name>_validation.rs` file, which is then included by the
Rust integration test at `tests/r_validation_<name>.rs`. (We keep the
`r_validation_` prefix for consistency with the R-validated tests even
though the oracle here is sklearn.)

```bash
.venv/bin/python generate_theil_sen_validation.py   > ../../tests/fixtures/theil_sen_validation.rs
.venv/bin/python generate_ransac_validation.py      > ../../tests/fixtures/ransac_validation.rs
.venv/bin/python generate_pa_validation.py          > ../../tests/fixtures/pa_validation.rs
.venv/bin/python generate_lars_validation.py        > ../../tests/fixtures/lars_validation.rs
.venv/bin/python generate_bayesian_validation.py    > ../../tests/fixtures/bayesian_validation.rs
```

Every script uses `numpy.random.default_rng(42)` so output is bit-stable
across runs on the same numpy version. The Rust tests assert agreement
with tolerances chosen per-estimator and documented inline.
