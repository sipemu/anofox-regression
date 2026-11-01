# Contributing

Thanks for helping. Bug reports, reference mismatches against R or
scikit-learn, and pull requests are all welcome.

## Reporting a problem

Open an issue with a minimal reproduction: the data (or code that generates
it), the call, what you got and what you expected. For a numerical
disagreement, include the reference call (for example the R code) and its
output. Security problems go through [SECURITY.md](SECURITY.md) instead.

## Development

The crate builds with stable Rust (MSRV 1.87, see `Cargo.toml`). Before you
push, run what CI runs:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny check            # licences, advisories (cargo install cargo-deny)
```

`.pre-commit-config.yaml` runs `fmt` and `clippy` on commit. The WASM
bindings are tested with `wasm-pack build crates/anofox-regression-js
--target web --out-dir pkg` and `npm ci && npm test` in `js/`.

## Validation is part of the change

Every estimate the crate reports is checked against an established
implementation, and new results need the same:

1. Write a generator in `tests/r_scripts/` (or `validation/python/` for
   scikit-learn) that prints the reference values as Rust `const`s with 17
   significant digits, from data made with a fixed seed.
2. Add the test that compares against them, naming the generator in its
   header, with the tightest tolerance that holds and a comment where it is
   loose (an optimiser that stops short, a weakly identified parameter, …).
3. Add the generator to the inventory in `validation/VALIDATION.md`; a
   generator whose output is redirected to `tests/fixtures/` also goes into
   `validation/r_fixtures.txt`, which the scheduled validation job
   regenerates and diffs.

If the crate and the reference disagree, find out why before you adjust a
tolerance: either the crate has a bug, or the reference does something the
docs should explain.

## Code

- Public items are documented (`#![deny(missing_docs)]`) and doc examples
  run; the README's code blocks run as doctests too.
- Bad input is an error, never a panic or `Ok` with NaN estimates; use the
  shared checks in `core` (`validate_xy`).
- No `unsafe` (`#![forbid(unsafe_code)]`).
- Keep functions small (AGENTS.md: cyclomatic complexity ≤ 8) and match the
  surrounding style and comment density.
- New code is covered by tests: CI fails below 94 % line coverage overall and
  codecov asks for 95 % on the changed lines.

## Pull requests

Keep a pull request to one topic, reference the issue, and add an entry to
the `Unreleased` section of `CHANGELOG.md`, marking behaviour changes. See
[STABILITY.md](STABILITY.md) for what counts as a breaking change.
