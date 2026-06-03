#!/usr/bin/env python3
"""Generate validation constants for TheilSenRegressor.

Oracle: sklearn.linear_model.TheilSenRegressor with deterministic settings
(random_state fixed and max_subpopulation high enough that the exhaustive
enumeration branch is taken).
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import sklearn
from sklearn.linear_model import TheilSenRegressor


def emit_header() -> None:
    print(
        "// =============================================================================\n"
        "// Theil-Sen Validation Data Generated from sklearn.linear_model.TheilSenRegressor\n"
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


def case_univariate() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 1: Univariate Theil-Sen (pairwise-slope median, exact branch)\n"
        "// sklearn: TheilSenRegressor(fit_intercept=True, random_state=42)\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(42)
    n = 30
    x = np.linspace(0.0, 10.0, n)
    y = 1.5 + 0.7 * x + rng.normal(0.0, 0.5, n)
    # Inject a couple of outliers
    y[2] = 50.0
    y[25] = -40.0

    model = TheilSenRegressor(
        fit_intercept=True,
        random_state=42,
        max_subpopulation=10_000,
    ).fit(x.reshape(-1, 1), y)

    emit_const_usize("N_THEIL_UNI", n)
    emit_array_f64("X_THEIL_UNI", x)
    emit_array_f64("Y_THEIL_UNI", y)
    emit_const_f64("EXPECTED_INTERCEPT_THEIL_UNI", float(model.intercept_))
    emit_const_f64("EXPECTED_COEF_THEIL_UNI", float(model.coef_[0]))
    print()


def case_multivariate_exact() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 2: Multivariate Theil-Sen, exhaustive enumeration branch\n"
        "// sklearn TheilSenRegressor with C(n, n_subsamples) <= max_subpopulation\n"
        "// (so sklearn enumerates every subset deterministically and our exact\n"
        "// enumeration branch should match within Weiszfeld convergence)\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(7)
    n, p = 14, 2  # C(14, 3) = 364 — small enough for exhaustive
    X = rng.normal(size=(n, p))
    y = 0.5 + 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(0.0, 0.4, n)
    # Inject one outlier so Theil-Sen actually does something useful
    y[3] = 40.0

    model = TheilSenRegressor(
        fit_intercept=True,
        random_state=42,
        max_subpopulation=10_000,
        max_iter=500,
        tol=1e-6,
    ).fit(X, y)

    # We pass X column-major-flattened by row-major iteration: emit row-major.
    emit_const_usize("N_THEIL_MULTI", n)
    emit_const_usize("P_THEIL_MULTI", p)
    flat = X.flatten()
    emit_array_f64("X_THEIL_MULTI_FLAT", flat)
    emit_array_f64("Y_THEIL_MULTI", y)
    emit_const_f64("EXPECTED_INTERCEPT_THEIL_MULTI", float(model.intercept_))
    emit_array_f64("EXPECTED_COEFS_THEIL_MULTI", model.coef_)
    print()


def main() -> None:
    emit_header()
    case_univariate()
    case_multivariate_exact()


if __name__ == "__main__":
    main()
