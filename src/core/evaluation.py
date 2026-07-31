"""Leakage-aware evaluation shared by all NeuroSense phases.

The unit of inference is the subject, not the epoch. ``evaluate_loso`` fits one
model per held-out subject and stores that subject's metrics as a row, so the
headline number is a mean over subjects with a subject-level bootstrap interval.

Pooling every held-out prediction into one array and scoring it once (what
``cross_val_predict`` gives you) is kept only as a descriptive appendix. It
treats correlated epochs as independent observations and hides the between-
subject spread entirely.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from .statistics import bootstrap_ci


def _safe_auc(y_true, proba):
    """AUC is undefined if a held-out subject has only one class present."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _fold_metrics(y_true, y_pred, proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n_test": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": _safe_auc(y_true, proba),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_loso(
    X,
    y,
    groups,
    model_factory,
    *,
    tune=False,
    param_grid=None,
    inner_splits=5,
    random_state=42,
    subject_labels=None,
    n_boot=10000,
):
    """Leave-one-subject-out evaluation with subject-level inference.

    Parameters
    ----------
    X, y, groups : ndarray
        Feature matrix, binary labels, subject id per row.
    model_factory : callable
        Zero-argument callable returning an unfitted estimator. Must be a
        Pipeline if scaling is needed, so that scaling is fitted inside the
        fold. A pre-fitted or shared estimator instance is rejected.
    tune : bool
        If True, tune with a group-aware inner loop over the training subjects
        only. The held-out subject never influences hyperparameter selection.
    param_grid : dict, optional
        Required when ``tune`` is True.
    subject_labels : dict, optional
        Maps group value to a human-readable subject id for the report.

    Returns
    -------
    dict with keys ``folds``, ``subject_mean``, ``subject_bootstrap_ci``,
    ``pooled_descriptive_metrics``, ``config``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    if tune and not param_grid:
        raise ValueError("param_grid is required when tune=True")

    probe = model_factory()
    if hasattr(probe, "classes_"):
        raise ValueError("model_factory must return an unfitted estimator")

    logo = LeaveOneGroupOut()
    folds, pooled_true, pooled_pred, pooled_proba = [], [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = groups[test_idx][0]

        model = model_factory()
        if tune:
            inner = GridSearchCV(
                model,
                param_grid,
                scoring="balanced_accuracy",
                cv=_inner_group_cv(groups[train_idx], inner_splits, random_state),
                n_jobs=1,
            )
            inner.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
            model = inner.best_estimator_
            chosen = inner.best_params_
        else:
            model.fit(X[train_idx], y[train_idx])
            chosen = None

        y_pred = model.predict(X[test_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]

        row = _fold_metrics(y[test_idx], y_pred, proba)
        row["subject_id"] = (
            subject_labels.get(held_out, str(held_out))
            if subject_labels
            else str(held_out)
        )
        if chosen is not None:
            row["chosen_params"] = {k: str(v) for k, v in chosen.items()}
        folds.append(row)

        pooled_true.append(y[test_idx])
        pooled_pred.append(y_pred)
        pooled_proba.append(proba)

    metrics = ["balanced_accuracy", "macro_f1", "roc_auc"]
    subject_mean, subject_ci = {}, {}
    for m in metrics:
        vals = np.array([f[m] for f in folds if f[m] is not None], dtype=float)
        n_valid = int(len(vals))
        subject_mean[m] = {
            "mean": float(vals.mean()) if n_valid else None,
            "std": float(vals.std(ddof=1)) if n_valid > 1 else None,
            "median": float(np.median(vals)) if n_valid else None,
            "n_subjects": n_valid,
            "n_above_chance": int((vals > 0.5).sum()) if n_valid else 0,
        }
        subject_ci[m] = (
            bootstrap_ci(vals, n_boot=n_boot, random_state=random_state)
            if n_valid > 1
            else None
        )

    pt = np.concatenate(pooled_true)
    pp = np.concatenate(pooled_pred)
    pr = np.concatenate(pooled_proba)
    pooled = _fold_metrics(pt, pp, pr)
    pooled["_warning"] = (
        "Descriptive only. Pools correlated epochs across subjects and treats "
        "them as independent. Not the inferential result."
    )

    return {
        "folds": folds,
        "subject_mean": subject_mean,
        "subject_bootstrap_ci": subject_ci,
        "pooled_descriptive_metrics": pooled,
        "config": {
            "n_subjects": int(len(np.unique(groups))),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "tuned": bool(tune),
            "param_grid": {k: [str(v) for v in vs] for k, vs in (param_grid or {}).items()},
            "random_state": random_state,
            "n_boot": n_boot,
        },
    }


def _inner_group_cv(train_groups, n_splits, random_state):
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(train_groups))
    return GroupKFold(n_splits=min(n_splits, n_groups))


def evaluate_naive_split(X, y, model_factory, n_splits=5, random_state=42):
    """Deliberately leaky random epoch split, for the inflation comparison only.

    Ignores subject identity entirely. Reported alongside the LOSO result to
    quantify the gap, never as a headline.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    proba = cross_val_predict(
        model_factory(), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = _fold_metrics(y, pred, proba)
    out["_warning"] = (
        "Leaky by construction: epochs from the same subject appear in both "
        "train and test. Included to quantify inflation."
    )
    return out
