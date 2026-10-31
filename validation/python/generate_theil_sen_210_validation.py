#!/usr/bin/env python3
"""#210: TheilSenRegressor without intercept and with n_subsamples < n.

Every case takes sklearn's exhaustive-enumeration branch
(C(n, n_subsamples) <= max_subpopulation), so the result is deterministic.

    validation/python/.venv/bin/python validation/python/generate_theil_sen_210_validation.py \
        > tests/fixtures/theil_sen_210_validation.rs
"""
import numpy as np
import sklearn
from sklearn.linear_model import TheilSenRegressor


def arr(name, values):
    values = np.ravel(values)
    body = ", ".join(f"{v:.17e}" for v in values)
    print(f"const {name}: [f64; {len(values)}] = [{body}];")


def fit(X, y, **kw):
    return TheilSenRegressor(random_state=0, max_subpopulation=10_000, max_iter=1000, tol=1e-12, **kw).fit(X, y)


print(f"// scikit-learn {sklearn.__version__}, numpy {np.__version__}")
rng = np.random.default_rng(210)
n = 12
X = rng.normal(size=(n, 2))
y = 0.8 + 1.5 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(0.0, 0.3, n)
y[5] += 6.0
arr("X_FLAT", X)
arr("Y", y)
# no intercept, univariate (n_subsamples = 1) and bivariate (n_subsamples = 2)
m = fit(X[:, :1], y, fit_intercept=False)
arr("NOINT_UNI", m.coef_)
m = fit(X, y, fit_intercept=False)
arr("NOINT_MULTI", m.coef_)
# n_subsamples = 5 > p + 1 = 3: least squares on each of the C(12, 5) = 792 subsets
m = fit(X, y, n_subsamples=5)
arr("SUBSAMPLES5", np.r_[m.intercept_, m.coef_])
m = fit(X, y, fit_intercept=False, n_subsamples=4)
arr("NOINT_SUBSAMPLES4", m.coef_)
