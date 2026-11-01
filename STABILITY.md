# Stability Policy

This document states what the version number promises. It applies from 1.0;
before 1.0 (`0.y.z`) a minor bump (`0.y`) may break the API, and the
`CHANGELOG.md` lists every breaking or behaviour change.

## What 1.0 covers

- **The public Rust API** of `anofox-regression`: every public item
  reachable from the crate root, with its documented behaviour. Breaking it
  requires a major release; additions come in minor releases. API changes are
  checked with `cargo-semver-checks` in CI.
- **The npm package** (`js/`, the WASM bindings) follows the same version
  line and the same promise. It covers a subset of the estimators; extending
  it is a minor change.
- **Results.** Estimates that are validated against a reference (R or
  scikit-learn, see `validation/VALIDATION.md`) agree with it within the
  documented tolerance. A change that brings a result closer to the reference
  is a bug fix, released in a minor or patch release and marked as a
  behaviour change in the changelog. Values beyond the documented tolerance
  (the last digits of an iterative fit, the exact iteration count) are not
  part of the promise.
- **Errors.** Which inputs are rejected is part of the behaviour; the exact
  error message is not.

Not covered: items marked `#[doc(hidden)]` or documented as internal, the
contents of `tests/`, `validation/` and `benches/`, and features named
`unstable-*` (none exist yet; new experimental modules will start behind
one).

## Dependencies in the public API

- **faer** types (`Mat`, `Col`, …) appear throughout the public API, and the
  crate re-exports it as `anofox_regression::faer` so callers can use the
  matching version. A faer upgrade that changes those types (a faer major,
  or `0.y` minor, release) is a **major release** of this crate.
- Other dependencies (`statrs`, `basin`, `rayon`) are implementation details
  unless a public signature names them; such uses are being removed before
  1.0 (#180).

## Minimum supported Rust version

The MSRV is set in `Cargo.toml` (`rust-version`, currently 1.87) and checked
in CI. It may be raised in a minor release, never in a patch release, and
the changelog says so.

## Numerical behaviour

Algorithms may change between minor releases when the result stays within
the documented tolerance of the reference, or moves closer to it. Random
draws that replay R (`set.seed`) are part of the behaviour for the functions
that document it.
