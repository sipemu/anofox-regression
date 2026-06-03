#!/usr/bin/env python3
"""Generate validation constants for PassiveAggressiveRegressor.

Oracle: sklearn.linear_model.PassiveAggressiveRegressor.

We disable shuffling (shuffle=False) so the per-epoch update order is
deterministic and does not depend on numpy's Mersenne-Twister sequence.
This lets us compare final weights directly without re-implementing
sklearn's RNG bit-for-bit.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import sklearn
from sklearn.linear_model import PassiveAggressiveRegressor


def emit_header() -> None:
    print(
        "// =============================================================================\n"
        "// Passive-Aggressive Validation Data Generated from sklearn\n"
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


def case_pa1_univariate() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 1: PA-I univariate, shuffle=False, max_iter=50\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(-3.0, 3.0, n)
    y = 1.0 + 0.8 * x + rng.normal(0.0, 0.1, n)

    model = PassiveAggressiveRegressor(
        C=1.0,
        epsilon=0.1,
        fit_intercept=True,
        max_iter=50,
        tol=1e-6,
        shuffle=False,
        loss="epsilon_insensitive",
        random_state=42,
        average=False,
    ).fit(x.reshape(-1, 1), y)

    emit_const_usize("N_PA1_UNI", n)
    emit_array_f64("X_PA1_UNI", x)
    emit_array_f64("Y_PA1_UNI", y)
    emit_const_f64("EXPECTED_INTERCEPT_PA1_UNI", float(model.intercept_[0]))
    emit_const_f64("EXPECTED_COEF_PA1_UNI", float(model.coef_[0]))
    print()


def case_pa1_multi() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 2: PA-I multivariate, shuffle=False, max_iter=100\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(7)
    n, p = 80, 3
    X = rng.normal(size=(n, p))
    y = -0.5 + 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0.0, 0.05, n)

    model = PassiveAggressiveRegressor(
        C=1.0,
        epsilon=0.1,
        fit_intercept=True,
        max_iter=100,
        tol=1e-6,
        shuffle=False,
        loss="epsilon_insensitive",
        random_state=42,
        average=False,
    ).fit(X, y)

    emit_const_usize("N_PA1_MULTI", n)
    emit_const_usize("P_PA1_MULTI", p)
    emit_array_f64("X_PA1_MULTI_FLAT", X.flatten())
    emit_array_f64("Y_PA1_MULTI", y)
    emit_const_f64("EXPECTED_INTERCEPT_PA1_MULTI", float(model.intercept_[0]))
    emit_array_f64("EXPECTED_COEFS_PA1_MULTI", model.coef_)
    print()


def case_pa2_multi() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 3: PA-II (squared epsilon-insensitive) multivariate\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(11)
    n, p = 100, 2
    X = rng.normal(size=(n, p))
    y = 2.0 - 0.5 * X[:, 0] + 1.2 * X[:, 1] + rng.normal(0.0, 0.1, n)

    model = PassiveAggressiveRegressor(
        C=0.5,
        epsilon=0.05,
        fit_intercept=True,
        max_iter=80,
        tol=1e-6,
        shuffle=False,
        loss="squared_epsilon_insensitive",
        random_state=42,
        average=False,
    ).fit(X, y)

    emit_const_usize("N_PA2_MULTI", n)
    emit_const_usize("P_PA2_MULTI", p)
    emit_array_f64("X_PA2_MULTI_FLAT", X.flatten())
    emit_array_f64("Y_PA2_MULTI", y)
    emit_const_f64("EXPECTED_INTERCEPT_PA2_MULTI", float(model.intercept_[0]))
    emit_array_f64("EXPECTED_COEFS_PA2_MULTI", model.coef_)
    print()


def main() -> None:
    emit_header()
    case_pa1_univariate()
    case_pa1_multi()
    case_pa2_multi()


if __name__ == "__main__":
    main()
