"""Tests for the temporal diagnostics added in the 2026-09 audit."""
import numpy as np
import pytest

from src.core.temporal import (
    choose_temporal_gap,
    expanding_window_splits_by_group,
    expanding_window_splits_by_index,
    feature_lag_autocorr,
)


def test_autocorr_of_iid_features_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 6))
    r = feature_lag_autocorr(x, lags=(1, 2, 3))
    assert r["lags"] == [1, 2, 3]
    for v in r["mean_abs_autocorr"]:
        assert v < 0.05


def test_autocorr_detects_ar1_dependence():
    rng = np.random.default_rng(1)
    n, k = 3000, 4
    x = np.zeros((n, k))
    x[0] = rng.standard_normal(k)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal(k)
    r = feature_lag_autocorr(x, lags=(1, 5, 20))
    # Lag 1 near 0.9, lag 5 = 0.9^5 ≈ 0.59, lag 20 near 0.12
    assert r["mean_abs_autocorr"][0] > 0.85
    assert 0.4 < r["mean_abs_autocorr"][1] < 0.7
    assert r["mean_abs_autocorr"][2] < 0.2


def test_choose_gap_already_below():
    ac = {"lags": [1, 2, 3], "mean_abs_autocorr": [0.1, 0.05, 0.02]}
    out = choose_temporal_gap(ac, threshold=0.3)
    assert out["recommended_gap_windows"] == 0
    assert out["reason"] == "already_below"


def test_choose_gap_crosses_at_specific_lag():
    ac = {"lags": [1, 2, 3, 4], "mean_abs_autocorr": [0.8, 0.5, 0.2, 0.1]}
    out = choose_temporal_gap(ac, threshold=0.3)
    assert out["recommended_gap_windows"] == 2  # crossed at lag 3, gap = lag - 1
    assert out["reason"] == "crossed_at_lag_3"


def test_choose_gap_never_below_returns_worst_case():
    ac = {"lags": [1, 2, 3], "mean_abs_autocorr": [0.9, 0.9, 0.9]}
    out = choose_temporal_gap(ac, threshold=0.3)
    assert out["recommended_gap_windows"] == 3
    assert out["reason"] == "never_below"


def test_expanding_group_splits_are_forward_and_disjoint():
    g = np.array([0]*3 + [1]*4 + [2]*2 + [3]*5 + [4]*3 + [5]*4)
    folds = list(expanding_window_splits_by_group(g, n_init_groups=2, step=1))
    assert len(folds) == 4  # groups 2, 3, 4, 5 as successive test blocks
    for train_idx, test_idx in folds:
        assert set(g[train_idx]).isdisjoint(set(g[test_idx]))
        # Every training row's group must appear strictly before the test group
        max_train_group = g[train_idx].max()
        test_group = g[test_idx][0]
        assert max_train_group < test_group


def test_expanding_group_splits_respect_gap():
    g = np.array([0]*2 + [1]*2 + [2]*2 + [3]*2 + [4]*2 + [5]*2)
    folds = list(expanding_window_splits_by_group(
        g, n_init_groups=2, step=1, gap_groups=1
    ))
    # First test group is 2 -> train covers groups {0} only (dropped group 1)
    tr, te = folds[0]
    assert set(g[tr]) == {0}
    assert set(g[te]) == {2}


def test_expanding_group_splits_rejects_unsorted_groups():
    g = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        list(expanding_window_splits_by_group(g, n_init_groups=1))


def test_expanding_index_splits_advance_and_respect_gap():
    folds = list(expanding_window_splits_by_index(n=100, n_init=40, step=10, gap=5))
    for tr, te in folds:
        assert tr[-1] < te[0] - 5 + 1        # gap of 5 rows enforced
        assert te[-1] >= te[0]
    starts = [te[0] for _, te in folds]
    assert starts == sorted(starts)
