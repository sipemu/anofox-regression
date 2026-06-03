#!/usr/bin/env python3
"""Generate validation constants for RansacRegressor.

We use scikit-learn's RANSACRegressor with an OLS base estimator and
construct fixtures where the consensus inlier set is unique (clear gap
between inlier and outlier residuals). The final fit is then OLS on that
inlier set, so both implementations should reach identical coefficients
regardless of subsample-ordering differences.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, RANSACRegressor


def emit_header() -> None:
    print(
        "// =============================================================================\n"
        "// RANSAC Validation Data Generated from sklearn.linear_model.RANSACRegressor\n"
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


def emit_bool_array(name: str, mask: np.ndarray) -> None:
    body = ", ".join("true" if b else "false" for b in mask)
    print(f"const {name}: [bool; {len(mask)}] = [{body}];")


def case_univariate_clean_split() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 1: Univariate clean inlier/outlier split\n"
        "// 50 inliers on y = 1.5 + 0.7x + N(0, 0.05); 15 outliers far above the line.\n"
        "// Residual threshold of 0.5 produces a unique consensus set, so final fit\n"
        "// is OLS on that set and matches sklearn to machine precision.\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(42)
    n_in, n_out = 50, 15
    n = n_in + n_out
    x_in = np.linspace(0.0, 10.0, n_in)
    y_in = 1.5 + 0.7 * x_in + rng.normal(0.0, 0.05, n_in)
    x_out = rng.uniform(0.0, 10.0, n_out)
    y_out = rng.uniform(20.0, 40.0, n_out)
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])

    model = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True),
        min_samples=2,
        residual_threshold=0.5,
        max_trials=200,
        random_state=42,
    ).fit(x.reshape(-1, 1), y)

    inlier_mask = model.inlier_mask_.astype(bool)
    final = model.estimator_  # OLS refit on inliers

    emit_const_usize("N_RANSAC_UNI", n)
    emit_array_f64("X_RANSAC_UNI", x)
    emit_array_f64("Y_RANSAC_UNI", y)
    emit_const_f64("EXPECTED_INTERCEPT_RANSAC_UNI", float(final.intercept_))
    emit_const_f64("EXPECTED_COEF_RANSAC_UNI", float(final.coef_[0]))
    emit_bool_array("EXPECTED_INLIER_MASK_RANSAC_UNI", inlier_mask)
    emit_const_usize("EXPECTED_N_INLIERS_RANSAC_UNI", int(inlier_mask.sum()))
    print()


def case_multivariate_clean_split() -> None:
    print(
        "// --------------------------------------------------------------------------\n"
        "// Test 2: Multivariate clean inlier/outlier split (p = 2)\n"
        "// 80 inliers on y = 0.5 + 2*x1 - 1*x2 + N(0, 0.02); 20 large outliers.\n"
        "// --------------------------------------------------------------------------"
    )
    rng = np.random.default_rng(7)
    n_in, n_out, p = 80, 20, 2
    n = n_in + n_out
    X_in = rng.normal(size=(n_in, p))
    y_in = 0.5 + 2.0 * X_in[:, 0] - 1.0 * X_in[:, 1] + rng.normal(0.0, 0.02, n_in)
    X_out = rng.normal(size=(n_out, p))
    y_out = rng.uniform(50.0, 100.0, n_out)
    X = np.vstack([X_in, X_out])
    y = np.concatenate([y_in, y_out])

    model = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True),
        min_samples=3,
        residual_threshold=0.3,
        max_trials=500,
        random_state=42,
    ).fit(X, y)

    inlier_mask = model.inlier_mask_.astype(bool)
    final = model.estimator_

    emit_const_usize("N_RANSAC_MULTI", n)
    emit_const_usize("P_RANSAC_MULTI", p)
    emit_array_f64("X_RANSAC_MULTI_FLAT", X.flatten())
    emit_array_f64("Y_RANSAC_MULTI", y)
    emit_const_f64("EXPECTED_INTERCEPT_RANSAC_MULTI", float(final.intercept_))
    emit_array_f64("EXPECTED_COEFS_RANSAC_MULTI", final.coef_)
    emit_bool_array("EXPECTED_INLIER_MASK_RANSAC_MULTI", inlier_mask)
    emit_const_usize("EXPECTED_N_INLIERS_RANSAC_MULTI", int(inlier_mask.sum()))
    print()


def main() -> None:
    emit_header()
    case_univariate_clean_split()
    case_multivariate_clean_split()


if __name__ == "__main__":
    main()
