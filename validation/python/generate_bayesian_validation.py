#!/usr/bin/env python3
"""Generate validation constants for BayesianRidge and ARDRegression.

Oracle: scikit-learn's `BayesianRidge` and `ARDRegression`.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import sklearn
from sklearn.linear_model import ARDRegression, BayesianRidge


def emit_header() -> None:
    print(
        "// =============================================================================\n"
        "// Bayesian Regression Validation Data Generated from sklearn\n"
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


def case_bayesian_ridge() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 1: BayesianRidge defaults\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(42)
    n, p = 100, 4
    X = rng.normal(size=(n, p))
    true_beta = np.array([1.5, -0.7, 0.3, 2.0])
    y = 0.5 + X @ true_beta + rng.normal(0.0, 0.2, n)

    model = BayesianRidge(
        max_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=False,
        fit_intercept=True,
    ).fit(X, y)

    emit_const_usize("N_BR", n)
    emit_const_usize("P_BR", p)
    emit_array_f64("X_BR_FLAT", X.flatten())
    emit_array_f64("Y_BR", y)
    emit_const_f64("EXPECTED_INTERCEPT_BR", float(model.intercept_))
    emit_array_f64("EXPECTED_COEFS_BR", model.coef_)
    emit_const_f64("EXPECTED_ALPHA_BR", float(model.alpha_))
    emit_const_f64("EXPECTED_LAMBDA_BR", float(model.lambda_))
    print()


def case_ard() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 2: ARDRegression sparse-recovery\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(7)
    n, p = 120, 6
    X = rng.normal(size=(n, p))
    true_beta = np.array([2.0, 0.0, 1.0, 0.0, 0.0, -0.5])
    y = X @ true_beta + rng.normal(0.0, 0.1, n)

    model = ARDRegression(
        max_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        threshold_lambda=10_000.0,
        fit_intercept=True,
    ).fit(X, y)

    emit_const_usize("N_ARD", n)
    emit_const_usize("P_ARD", p)
    emit_array_f64("X_ARD_FLAT", X.flatten())
    emit_array_f64("Y_ARD", y)
    emit_const_f64("EXPECTED_INTERCEPT_ARD", float(model.intercept_))
    emit_array_f64("EXPECTED_COEFS_ARD", model.coef_)
    emit_const_f64("EXPECTED_ALPHA_ARD", float(model.alpha_))
    emit_array_f64("EXPECTED_LAMBDAS_ARD", model.lambda_)
    print()


def main() -> None:
    emit_header()
    case_bayesian_ridge()
    case_ard()


if __name__ == "__main__":
    main()
