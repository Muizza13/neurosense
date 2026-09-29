"""Temporal-dependence diagnostics and time-aware split generators.

Added in the 2026-09 audit. Phase 1 windows are drawn from one continuous
recording, so neighbouring windows are correlated and a chronological split
without a buffer can still leak information at the boundary. These utilities
let Phase 1 (a) inspect the strength of that correlation and (b) enforce a
justified temporal gap in the leakage-safe protocols.

Nothing here is Phase 1 specific in principle. Phase 2 uses discrete trials
with subject-level grouping, so it does not need a temporal gap.
"""
from __future__ import annotations

import numpy as np


def feature_lag_autocorr(features, lags=(1, 2, 3, 4, 5)):
    """Mean absolute autocorrelation of features across neighbouring windows.

    Parameters
    ----------
    features : ndarray of shape (n_windows, n_features)
        Time-ordered feature matrix. In Phase 1 this is the fold-safe band
        power extracted after the training-fitted clipper.
    lags : iterable of int
        Positive integer lags to evaluate.

    Returns
    -------
    dict with keys "lags" (list[int]) and "mean_abs_autocorr" (list[float]).
        `mean_abs_autocorr[i]` is the mean over features of
        `|corr(x[t], x[t + lags[i]])|`, using the sample Pearson correlation
        on the demeaned column.

    Rationale
    ---------
    A value at lag 1 near zero means the chronological split does not need a
    temporal gap, because adjacent windows are already independent under the
    feature representation. A value near 1 means neighbouring windows are
    near-duplicates and even a chronological split can leak short-range
    memory across the boundary.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be 2-D (n_windows, n_features)")
    lags = [int(l) for l in lags]
    if any(l <= 0 for l in lags):
        raise ValueError("lags must be positive integers")
    n_win = features.shape[0]
    if n_win < max(lags) + 2:
        raise ValueError(
            f"need at least {max(lags) + 2} windows to compute lag "
            f"{max(lags)} autocorrelation, got {n_win}"
        )

    xs = features - features.mean(axis=0, keepdims=True)
    denom = (xs ** 2).sum(axis=0)
    denom = np.where(denom > 0, denom, np.nan)

    out = []
    for lag in lags:
        num = (xs[lag:] * xs[:-lag]).sum(axis=0)
        # Column-wise Pearson correlation between x[t] and x[t + lag]
        per_feature = num / denom
        out.append(float(np.nanmean(np.abs(per_feature))))
    return {"lags": lags, "mean_abs_autocorr": out}


def choose_temporal_gap(autocorr_report, threshold=0.30):
    """Turn a lag report into a windows-of-gap recommendation.

    Rule: the recommended gap is the smallest lag at which mean absolute
    autocorrelation first drops below `threshold`. If autocorrelation is
    already below the threshold at lag 1 the recommendation is 0. If it
    never drops below the threshold across the reported lags the
    recommendation is `max(lags)` and the caller should treat the
    chronological split as suspect.

    Returns
    -------
    dict:
        {"recommended_gap_windows": int,
         "threshold": float,
         "reason": "already_below" | "crossed_at_lag_k" | "never_below"}
    """
    lags = autocorr_report["lags"]
    ac = autocorr_report["mean_abs_autocorr"]
    for lag, value in zip(lags, ac):
        if value < threshold:
            if lag == lags[0]:
                return {"recommended_gap_windows": 0,
                        "threshold": float(threshold),
                        "reason": "already_below"}
            # Gap = number of windows to drop between train tail and test head.
            return {"recommended_gap_windows": int(lag - 1),
                    "threshold": float(threshold),
                    "reason": f"crossed_at_lag_{lag}"}
    return {"recommended_gap_windows": int(max(lags)),
            "threshold": float(threshold),
            "reason": "never_below"}


def expanding_window_splits_by_group(group_ids, n_init_groups, step=1, gap_groups=0):
    """Walk-forward splits at the granularity of groups (label blocks).

    Yields pairs `(train_idx, test_idx)` where indices refer to rows of a
    feature matrix ordered by `group_ids`. Splitting at the group boundary
    keeps a label block whole and matches the leave-one-block-out logic
    already used elsewhere in the project.

    Parameters
    ----------
    group_ids : sequence of int
        Group id per row, in time order. Must be non-decreasing.
    n_init_groups : int
        Number of initial groups used for the first training set.
    step : int
        How many groups advance between successive test blocks.
    gap_groups : int
        Number of groups skipped between the end of training and the start
        of the test block. A gap of `k` drops `k` groups from the tail of
        the training window before the test group.

    Yields
    ------
    (train_idx, test_idx) : ndarray, ndarray
        Both are int arrays into the original row order.
    """
    g = np.asarray(group_ids)
    if g.ndim != 1:
        raise ValueError("group_ids must be 1-D")
    if np.any(np.diff(g) < 0):
        raise ValueError("group_ids must be non-decreasing (time-ordered)")
    unique_groups = list(dict.fromkeys(g.tolist()))
    if n_init_groups < 1 or n_init_groups >= len(unique_groups):
        raise ValueError(
            f"n_init_groups must be in [1, {len(unique_groups) - 1}], "
            f"got {n_init_groups}"
        )
    if step < 1:
        raise ValueError("step must be at least 1")
    if gap_groups < 0:
        raise ValueError("gap_groups must be non-negative")

    for start in range(n_init_groups, len(unique_groups), step):
        test_group = unique_groups[start]
        train_end = start - gap_groups
        if train_end < 1:
            continue
        train_groups = set(unique_groups[:train_end])
        train_idx = np.where(np.isin(g, list(train_groups)))[0]
        test_idx = np.where(g == test_group)[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        yield train_idx, test_idx


def expanding_window_splits_by_index(n, n_init, step, gap=0):
    """Row-level walk-forward splits with an explicit gap.

    Kept for completeness. In Phase 1 the group-level generator above is
    preferred because a row-level test fold would be single-class and would
    reproduce the AUC-undefined problem the LOBO protocol already has.
    """
    n = int(n); n_init = int(n_init); step = int(step); gap = int(gap)
    if n_init < 1 or n_init >= n:
        raise ValueError(f"n_init must be in [1, {n - 1}], got {n_init}")
    if step < 1:
        raise ValueError("step must be at least 1")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    start = n_init
    while start < n:
        train_end = start - gap
        if train_end < 1:
            start += step
            continue
        train_idx = np.arange(0, train_end)
        test_end = min(start + step, n)
        test_idx = np.arange(start, test_end)
        yield train_idx, test_idx
        start += step
