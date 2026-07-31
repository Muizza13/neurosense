"""Statistical reporting helpers. Resampling is always over subjects.

With ten subjects in Phase 2 a percentile interval is wide. That width is the
honest answer, not a defect to be engineered away by resampling epochs instead.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot=10000, alpha=0.05, random_state=42):
    """Percentile bootstrap over subject-level scores.

    Parameters
    ----------
    values : 1-D array
        One score per subject. Never one score per epoch.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return None

    rng = np.random.default_rng(random_state)
    means = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "level": float(1 - alpha),
        "n_subjects": int(n),
        "method": "percentile bootstrap, resampling subjects",
    }


def subject_permutation_test(
    X, y, groups, model_factory, evaluate_fn, n_perm=200, random_state=42
):
    """Label-shuffling null with the subject grouping preserved.

    Labels are permuted within each subject, so the null keeps subject identity
    and class balance intact and destroys only the label-to-epoch mapping.
    Returns the observed mean subject balanced accuracy, the null distribution,
    and a one-sided p-value with the standard (r + 1) / (n + 1) correction.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    observed = evaluate_fn(X, y, groups, model_factory)
    obs = observed["subject_mean"]["balanced_accuracy"]["mean"]

    null = []
    for _ in range(n_perm):
        y_perm = y.copy()
        for g in np.unique(groups):
            m = groups == g
            y_perm[m] = rng.permutation(y[m])
        res = evaluate_fn(X, y_perm, groups, model_factory)
        null.append(res["subject_mean"]["balanced_accuracy"]["mean"])

    null = np.asarray(null, dtype=float)
    p = (int((null >= obs).sum()) + 1) / (n_perm + 1)
    return {
        "observed": float(obs),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_percentile_95": float(np.percentile(null, 95)),
        "p_value": float(p),
        "n_permutations": int(n_perm),
        "scheme": "labels shuffled within subject, subject grouping preserved",
    }


def paired_subject_delta(scores_a, scores_b, n_boot=10000, random_state=42):
    """Bootstrap CI on a paired per-subject difference (e.g. calibration gain).

    Both inputs must be ordered by the same subjects.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired inputs must have the same length")
    diff = b - a
    ci = bootstrap_ci(diff, n_boot=n_boot, random_state=random_state)
    return {
        "mean_delta": float(np.nanmean(diff)),
        "n_improved": int(np.nansum(diff > 0)),
        "n_subjects": int(np.sum(~np.isnan(diff))),
        "ci": ci,
    }
