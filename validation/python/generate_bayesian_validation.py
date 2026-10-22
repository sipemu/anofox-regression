#!/usr/bin/env python3
"""Generate validation constants for BayesianRidge and ARDRegression.

Oracle: scikit-learn's `BayesianRidge` and `ARDRegression`.

Besides the coefficients and hyperparameters, each case emits the posterior
covariance `sigma_`, `n_iter_` and the posterior predictive standard deviation
of a few query rows from `predict(X, return_std=True)`, and the posterior
SDs / 90% central credible intervals of the coefficients and the intercept
(see `emit_credible`). scikit-learn >= 1.8 is
required: earlier releases did not center `x` in `return_std` and, with more
features than rows, dropped the null-space part of `sigma_`.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.linear_model import ARDRegression, BayesianRidge

CREDIBLE_LEVEL = 0.9


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


def emit_posterior(suffix: str, model, X_query: np.ndarray) -> None:
    """sigma_, n_iter_ and the predictive std of the query rows."""
    emit_array_f64(f"EXPECTED_SIGMA_{suffix}", np.asarray(model.sigma_).flatten())
    emit_const_usize(f"EXPECTED_N_ITER_{suffix}", int(model.n_iter_))
    mean, std = model.predict(X_query, return_std=True)
    emit_const_usize(f"N_QUERY_{suffix}", X_query.shape[0])
    emit_array_f64(f"X_QUERY_{suffix}_FLAT", X_query.flatten())
    emit_array_f64(f"EXPECTED_PRED_{suffix}", mean)
    emit_array_f64(f"EXPECTED_STD_{suffix}", std)


def emit_credible(suffix: str, model, X: np.ndarray, kept: np.ndarray) -> None:
    """Posterior SDs and 90% credible intervals of coefficients and intercept.

    The coefficient SDs are `sqrt(diag(sigma_))` (0 for features ARD pruned).
    The intercept SD is computed independently of the closed form used in
    Rust (`1/(n alpha) + x̄' Σ x̄`): invert the full posterior precision of
    `(b0, β_kept)` under a flat prior on `b0`, `alpha [1 X]'[1 X] + diag(0,
    lambda)`, and take its `(0, 0)` element.
    """
    z = norm.ppf(0.5 + CREDIBLE_LEVEL / 2.0)
    p = X.shape[1]
    sd = np.zeros(p)
    sd[kept] = np.sqrt(np.diag(model.sigma_))
    emit_array_f64(f"EXPECTED_POST_SD_{suffix}", sd)
    emit_array_f64(f"EXPECTED_CRED_LO_{suffix}", model.coef_ - z * sd)
    emit_array_f64(f"EXPECTED_CRED_HI_{suffix}", model.coef_ + z * sd)
    if model.fit_intercept:
        lam = np.broadcast_to(np.asarray(model.lambda_, dtype=float), (p,))[kept]
        design = np.column_stack([np.ones(X.shape[0]), X[:, kept]])
        precision = model.alpha_ * design.T @ design + np.diag(np.concatenate([[0.0], lam]))
        b0_sd = float(np.sqrt(np.linalg.inv(precision)[0, 0]))
        emit_const_f64(f"EXPECTED_INTERCEPT_SD_{suffix}", b0_sd)
        emit_const_f64(f"EXPECTED_INTERCEPT_CRED_LO_{suffix}", float(model.intercept_ - z * b0_sd))
        emit_const_f64(f"EXPECTED_INTERCEPT_CRED_HI_{suffix}", float(model.intercept_ + z * b0_sd))


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
        compute_score=True,
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
    emit_const_f64("EXPECTED_SCORE_BR", float(model.scores_[-1]))
    emit_posterior("BR", model, rng.normal(size=(5, p)) * 2.0)
    emit_credible("BR", model, X, np.ones(p, dtype=bool))
    print()


def case_bayesian_ridge_wide() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 3: BayesianRidge with more features than rows (p > n)\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(11)
    n, p = 8, 12
    X = rng.normal(size=(n, p))
    y = 1.0 + X[:, 0] - 2.0 * X[:, 1] + rng.normal(0.0, 0.3, n)

    model = BayesianRidge(compute_score=True).fit(X, y)

    emit_const_usize("N_BRW", n)
    emit_const_usize("P_BRW", p)
    emit_array_f64("X_BRW_FLAT", X.flatten())
    emit_array_f64("Y_BRW", y)
    emit_const_f64("EXPECTED_INTERCEPT_BRW", float(model.intercept_))
    emit_array_f64("EXPECTED_COEFS_BRW", model.coef_)
    emit_const_f64("EXPECTED_ALPHA_BRW", float(model.alpha_))
    emit_const_f64("EXPECTED_LAMBDA_BRW", float(model.lambda_))
    emit_const_f64("EXPECTED_SCORE_BRW", float(model.scores_[-1]))
    emit_posterior("BRW", model, rng.normal(size=(3, p)))
    emit_credible("BRW", model, X, np.ones(p, dtype=bool))
    print()


def case_bayesian_ridge_no_intercept() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 4: BayesianRidge without an intercept\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(5)
    n, p = 40, 3
    X = rng.normal(loc=1.0, size=(n, p))
    y = X @ np.array([0.8, -1.2, 0.4]) + rng.normal(0.0, 0.5, n)

    model = BayesianRidge(fit_intercept=False, compute_score=True).fit(X, y)

    emit_const_usize("N_BRN", n)
    emit_const_usize("P_BRN", p)
    emit_array_f64("X_BRN_FLAT", X.flatten())
    emit_array_f64("Y_BRN", y)
    emit_array_f64("EXPECTED_COEFS_BRN", model.coef_)
    emit_const_f64("EXPECTED_ALPHA_BRN", float(model.alpha_))
    emit_const_f64("EXPECTED_LAMBDA_BRN", float(model.lambda_))
    emit_const_f64("EXPECTED_SCORE_BRN", float(model.scores_[-1]))
    emit_posterior("BRN", model, rng.normal(size=(3, p)))
    emit_credible("BRN", model, X, np.ones(p, dtype=bool))
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
    kept = model.lambda_ < model.threshold_lambda
    emit_array_f64("EXPECTED_KEPT_ARD", kept.astype(float))
    emit_posterior("ARD", model, rng.normal(size=(5, p)))
    emit_credible("ARD", model, X, kept)
    print()


def main() -> None:
    emit_header()
    case_bayesian_ridge()
    case_ard()
    case_bayesian_ridge_wide()
    case_bayesian_ridge_no_intercept()


if __name__ == "__main__":
    main()
