#!/usr/bin/env python3
"""Generate validation constants for the CV fold generators (solvers::cv).

Oracles: sklearn.model_selection.GroupKFold, LeaveOneGroupOut,
StratifiedKFold, StratifiedGroupKFold and TimeSeriesSplit, all without
shuffling (the shuffled variants use numpy's generator, which the crate does
not reproduce).

Usage: python generate_cv_splits_validation.py > ../../tests/fixtures/cv_splits_validation.rs
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import sklearn
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


def emit_header() -> None:
    print(
        "// =============================================================================\n"
        "// CV split validation data generated from sklearn.model_selection\n"
        f"// Generated on: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"// scikit-learn: {sklearn.__version__}\n"
        f"// numpy: {np.__version__}\n"
        "// Generator: validation/python/generate_cv_splits_validation.py\n"
        "// =============================================================================\n"
    )


def emit_array(name: str, ty: str, values) -> None:
    body = ", ".join(str(int(v)) for v in values)
    print(f"const {name}: [{ty}; {len(values)}] = [{body}];")


def test_fold(splitter, n, **kw) -> np.ndarray:
    """Fold id per row: the index of the split whose test set holds the row."""
    out = np.full(n, -1)
    for f, (_, test) in enumerate(splitter.split(np.zeros((n, 1)), **kw)):
        out[test] = f
    assert (out >= 0).all()
    return out


def case_group_kfold() -> None:
    print("\n// GroupKFold(n_splits=4): 14 groups of uneven size (with ties), labels not in order")
    rng = np.random.default_rng(87)
    sizes = np.array([7, 3, 3, 5, 1, 8, 2, 5, 4, 6, 2, 3, 1, 9])
    labels = 100 + 7 * rng.permutation(len(sizes))
    groups = np.repeat(labels, sizes)
    groups = groups[rng.permutation(len(groups))]
    emit_array("GKF_GROUPS", "i64", groups)
    emit_array("GKF_FOLDS", "usize", test_fold(GroupKFold(n_splits=4), len(groups), groups=groups))

    print("\n// LeaveOneGroupOut on the same groups")
    emit_array("LOGO_FOLDS", "usize", test_fold(LeaveOneGroupOut(), len(groups), groups=groups))


def case_stratified_kfold() -> None:
    print("\n// StratifiedKFold(n_splits=5): 3 strata (sizes 17, 9, 12), appearance order 2, 0, 1")
    rng = np.random.default_rng(204)
    strata = np.concatenate([np.full(17, 0), np.full(9, 1), np.full(12, 2)])
    strata = strata[rng.permutation(len(strata))]
    # Make the first appearance order differ from the sorted order.
    first = [2, 0, 1]
    for pos, s in enumerate(first):
        j = np.where(strata == s)[0][-1]
        strata[[pos, j]] = strata[[j, pos]]
    emit_array("SKF_STRATA", "i64", strata)
    emit_array("SKF_FOLDS", "usize", test_fold(StratifiedKFold(n_splits=5), len(strata), y=strata))


def case_stratified_group_kfold() -> None:
    print("\n// StratifiedGroupKFold(n_splits=3): 12 groups with mixed strata")
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(12), rng.integers(2, 7, size=12))
    strata = rng.integers(0, 3, size=len(groups))
    perm = rng.permutation(len(groups))
    groups, strata = groups[perm], strata[perm]
    emit_array("SGKF_GROUPS", "i64", groups)
    emit_array("SGKF_STRATA", "i64", strata)
    emit_array(
        "SGKF_FOLDS",
        "usize",
        test_fold(StratifiedGroupKFold(n_splits=3), len(groups), y=strata, groups=groups),
    )


def case_time_series() -> None:
    print(
        "\n// TimeSeriesSplit: (n, n_splits, test_size (0 = default), gap, max_train_size (0 = none),\n"
        "// then per split: train_start, train_end, test_start, test_end (half-open))"
    )
    configs = [
        (23, 4, None, 0, None),
        (30, 3, 5, 2, None),
        (40, 5, 4, 1, 10),
        (17, 2, None, 3, 6),
        (50, 6, 3, 0, 20),
    ]
    rows = []
    for n, k, ts, gap, mts in configs:
        tss = TimeSeriesSplit(n_splits=k, test_size=ts, gap=gap, max_train_size=mts)
        bounds = []
        for train, test in tss.split(np.zeros((n, 1))):
            assert (np.diff(train) == 1).all() and (np.diff(test) == 1).all()
            bounds.append((train[0], train[-1] + 1, test[0], test[-1] + 1))
        rows.append((n, k, ts or 0, gap, mts or 0, bounds))
    print("#[allow(clippy::type_complexity)]")
    print("const TSS_CASES: &[(usize, usize, usize, usize, usize, &[(usize, usize, usize, usize)])] = &[")
    for n, k, ts, gap, mts, bounds in rows:
        b = ", ".join(f"({a}, {b_}, {c}, {d})" for a, b_, c, d in bounds)
        print(f"    ({n}, {k}, {ts}, {gap}, {mts}, &[{b}]),")
    print("];")


if __name__ == "__main__":
    emit_header()
    case_group_kfold()
    case_stratified_kfold()
    case_stratified_group_kfold()
    case_time_series()
