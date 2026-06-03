#!/usr/bin/env python3
"""Generate validation constants for Lars / LassoLars.

Oracle: sklearn.linear_model.Lars and sklearn.linear_model.LassoLars.

We disable column standardisation (`normalize=False`) so the comparison
is on the same scale as our default `standardize=False` setting.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import sklearn
from sklearn.linear_model import Lars, LassoLars


def emit_header() -> None:
    print(
        "// =============================================================================\n"
        "// LARS Validation Data Generated from sklearn.linear_model.Lars / LassoLars\n"
        f"// Generated on: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"// scikit-learn: {sklearn.__version__}\n"
        f"// numpy: {np.__version__}\n"
        "// =============================================================================\n"
    )


def emit_const_f64(name: str, value: float) -> None:
    print(f"const {name}: f64 = {value:.15e};")


def emit_array_f64(name: str, values: np.ndarray) -> None:
    body = ", ".join(f"{v:.15e}" for v in values)
    print(f"const {name}: [f64; {len(values)}] = [{body}];")


def emit_const_usize(name: str, value: int) -> None:
    print(f"const {name}: usize = {value};")


def case_lars_full_path() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 1: Plain LARS, all features active at the end (= OLS)\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(42)
    n, p = 60, 4
    X = rng.normal(size=(n, p))
    true_beta = np.array([1.5, -0.7, 0.0, 2.0])
    y = 0.3 + X @ true_beta + rng.normal(0.0, 0.05, n)

    model = Lars(
        fit_intercept=True,
        n_nonzero_coefs=p,  # full path
    ).fit(X, y)

    emit_const_usize("N_LARS_FULL", n)
    emit_const_usize("P_LARS_FULL", p)
    emit_array_f64("X_LARS_FULL_FLAT", X.flatten())
    emit_array_f64("Y_LARS_FULL", y)
    emit_const_f64("EXPECTED_INTERCEPT_LARS_FULL", float(model.intercept_))
    emit_array_f64("EXPECTED_COEFS_LARS_FULL", model.coef_)
    print()


def case_lars_truncated() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 2: Plain LARS truncated at n_nonzero_coefs = 2\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(7)
    n, p = 80, 6
    X = rng.normal(size=(n, p))
    true_beta = np.array([2.0, 0.0, 1.0, 0.0, -0.5, 0.0])
    y = X @ true_beta + rng.normal(0.0, 0.05, n)

    model = Lars(
        fit_intercept=True,
        n_nonzero_coefs=2,
    ).fit(X, y)

    emit_const_usize("N_LARS_TRUNC", n)
    emit_const_usize("P_LARS_TRUNC", p)
    emit_array_f64("X_LARS_TRUNC_FLAT", X.flatten())
    emit_array_f64("Y_LARS_TRUNC", y)
    emit_const_f64("EXPECTED_INTERCEPT_LARS_TRUNC", float(model.intercept_))
    emit_array_f64("EXPECTED_COEFS_LARS_TRUNC", model.coef_)
    print()


def case_lasso_lars() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 3: LassoLars with alpha=0.1\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(11)
    n, p = 100, 8
    X = rng.normal(size=(n, p))
    true_beta = np.array([3.0, 0.0, -1.5, 0.0, 0.0, 0.7, 0.0, -0.4])
    y = X @ true_beta + rng.normal(0.0, 0.1, n)

    model = LassoLars(
        alpha=0.1,
        fit_intercept=True,
        max_iter=500,
    ).fit(X, y)

    emit_const_usize("N_LASSO_LARS", n)
    emit_const_usize("P_LASSO_LARS", p)
    emit_array_f64("X_LASSO_LARS_FLAT", X.flatten())
    emit_array_f64("Y_LASSO_LARS", y)
    emit_const_f64("EXPECTED_INTERCEPT_LASSO_LARS", float(model.intercept_))
    emit_array_f64("EXPECTED_COEFS_LASSO_LARS", model.coef_)
    print()


def main() -> None:
    emit_header()
    case_lars_full_path()
    case_lars_truncated()
    case_lasso_lars()


if __name__ == "__main__":
    main()
