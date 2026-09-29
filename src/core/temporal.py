"""Temporal-dependence diagnostics and time-aware split generators.

Phase 1 can measure a lag correlation of its band-power features and drop
windows between the chronological train and test segments when that value
is above a preset threshold. A small correlation is not evidence that the
windows are independent.

Phase 2 uses discrete trials and subject-level grouping, so it does not
apply this gap.
"""
from __future__ import annotations

import numpy as np


def _pearson_abs_mean(earlier, later):
    """Mean |Pearson r| across columns of two paired matrices.

    Each column is centered on the paired rows only. The denominator is the
    product of the two paired residual norms, not the energy of the full
    series. A column with zero variance on either side is omitted. Fewer
    than three pairs returns None. The result is a correlation. It is not
    a test of independence.
    """
    earlier = np.asarray(earlier, dtype=float)
    later = np.asarray(later, dtype=float)
    if earlier.shape != later.shape or earlier.ndim != 2:
        raise ValueError("paired windows must share shape (n_pairs, n_features)")
    if earlier.shape[0] < 3:
        return None
    a = earlier - earlier.mean(axis=0, keepdims=True)
    b = later - later.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / den
    corr = corr[np.isfinite(corr)]
    if corr.size == 0:
        return None
    return float(np.mean(np.abs(corr)))


def feature_lag_autocorr(features, lags=(1, 2, 3, 4, 5), starts=None,
                         win_samples=None):
    """Mean absolute Pearson correlation of features at a time lag.

    Estimator. For lag k and feature column x, collect pairs (x_i, x_j),
    center each side on those pairs, and compute the Pearson correlation.
    The reported value is the mean of the absolute correlations over
    features that have non-zero variance on both sides. Lags with fewer
    than three pairs are null.

    Pairing. If ``starts`` is omitted, row t is paired with row t+k. That
    mode assumes the rows are equally spaced. If ``starts`` and
    ``win_samples`` are given, row i is paired with row j only when
    ``starts[j] - starts[i] == k * win_samples``. A kept window that follows
    a discarded window is not a lag-1 neighbour of the previous kept window.

    A value near zero does not show that the windows are independent.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be 2-D (n_windows, n_features)")
    lags = [int(lag) for lag in lags]
    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must be positive integers")
    n_win = features.shape[0]
    if starts is None:
        if n_win < max(lags) + 2:
            raise ValueError(
                f"need at least {max(lags) + 2} windows to compute lag "
                f"{max(lags)} autocorrelation, got {n_win}"
            )
        pair_index = {
            lag: (np.arange(0, n_win - lag), np.arange(lag, n_win))
            for lag in lags
        }
        gap_count = None
    else:
        starts = np.asarray(starts)
        if starts.shape != (n_win,):
            raise ValueError("starts must have one entry per window")
        if win_samples is None or int(win_samples) <= 0:
            raise ValueError("win_samples must be a positive integer when starts is set")
        win_samples = int(win_samples)
        if np.any(np.diff(starts) <= 0):
            raise ValueError("starts must be strictly increasing")
        start_to_row = {int(s): i for i, s in enumerate(starts)}
        pair_index = {}
        for lag in lags:
            left, right = [], []
            step = lag * win_samples
            for row, start in enumerate(starts):
                other = start_to_row.get(int(start) + step)
                if other is None:
                    continue
                left.append(row)
                right.append(other)
            pair_index[lag] = (np.asarray(left, dtype=int), np.asarray(right, dtype=int))
        gaps = np.diff(starts.astype(int))
        gap_count = int(np.sum(gaps != win_samples))

    values = []
    n_pairs = []
    for lag in lags:
        left, right = pair_index[lag]
        n_pairs.append(int(len(left)))
        values.append(_pearson_abs_mean(features[left], features[right])
                      if len(left) else None)

    report = {
        "estimator": (
            "mean absolute Pearson correlation on paired windows; "
            "each side centered on the pairs; lags with fewer than 3 pairs are null"
        ),
        "lags": lags,
        "mean_abs_autocorr": values,
        "n_pairs": n_pairs,
        "not_an_independence_test": True,
    }
    if gap_count is not None:
        report["win_samples"] = int(win_samples)
        report["n_index_adjacent_pairs"] = int(n_win - 1)
        report["n_index_adjacent_pairs_with_time_gap"] = gap_count
    return report


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
        if value is None:
            return {"recommended_gap_windows": None,
                    "threshold": float(threshold),
                    "reason": f"undefined_at_lag_{lag}"}
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
