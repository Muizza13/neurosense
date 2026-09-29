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

import hashlib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

from .statistics import bootstrap_ci


def _index_hash(idx):
    """Deterministic short hash of an index array, for fold provenance."""
    arr = np.asarray(idx, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def per_split_baselines(y_train, y_test, random_state=42):
    """Baselines computed on this split's test set alone.

    Every leakage-aware split reports these three, so a headline number is
    always comparable to a same-split baseline rather than to a constant
    that carries over from a different protocol.
    """
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    if y_test.size == 0:
        return {
            "n_test": 0,
            "positive_rate_train": (float(np.mean(y_train == 1))
                                    if y_train.size else None),
            "positive_rate_test": None,
            "majority_class_accuracy": None,
            "dummy_stratified_balanced_accuracy": None,
            "dummy_uniform_balanced_accuracy": None,
        }

    dummy_shape = (
        np.zeros((max(1, len(y_train)), 1))
        if len(y_train) > 0
        else np.zeros((1, 1))
    )
    strat = DummyClassifier(strategy="stratified", random_state=random_state)
    strat.fit(dummy_shape, y_train if len(y_train) > 0 else np.array([0, 1]))
    uni = DummyClassifier(strategy="uniform", random_state=random_state)
    uni.fit(dummy_shape, y_train if len(y_train) > 0 else np.array([0, 1]))

    strat_pred = strat.predict(np.zeros((len(y_test), 1)))
    uni_pred = uni.predict(np.zeros((len(y_test), 1)))
    pos = float(np.mean(y_test == 1))
    return {
        "n_test": int(len(y_test)),
        "positive_rate_train": float(np.mean(y_train == 1)) if len(y_train) else None,
        "positive_rate_test": float(pos),
        "majority_class_accuracy": float(max(pos, 1.0 - pos)),
        "dummy_stratified_balanced_accuracy": float(
            balanced_accuracy_score(y_test, strat_pred)
        ),
        "dummy_uniform_balanced_accuracy": float(
            balanced_accuracy_score(y_test, uni_pred)
        ),
    }


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


def merged_block_groups(block_ids, blocks_per_superblock):
    """Partition an ordered sequence of block ids into contiguous super-blocks.

    Pre-specify ``blocks_per_superblock`` before evaluating. The mapping is
    deterministic: the k-th distinct block (in the order it first appears in
    ``block_ids``) is assigned to super-block ``k // blocks_per_superblock``.

    Returns an array of super-block ids the same length as ``block_ids``.
    Choosing folds by trying different values of ``blocks_per_superblock`` and
    picking the one with best scores would be cheating; pick it once, up
    front, on a structural criterion.
    """
    if blocks_per_superblock < 1:
        raise ValueError("blocks_per_superblock must be >= 1")
    block_ids = np.asarray(block_ids)
    order = []
    seen = set()
    for b in block_ids:
        b_int = int(b)
        if b_int not in seen:
            seen.add(b_int)
            order.append(b_int)
    to_super = {b: i // blocks_per_superblock for i, b in enumerate(order)}
    return np.array([to_super[int(b)] for b in block_ids], dtype=int)


def _extract_fold_coefficients(fitted_estimator):
    """Return the linear coefficient vector of a fitted estimator, or None.

    Round 4 (Task 8) helper: called after every fit inside evaluate_loso so
    downstream figure/table code can consume the saved per-fold coefficients
    from JSON instead of independently retraining models. Signs are
    preserved (not just magnitudes).

    Works for a bare linear estimator (``LogisticRegression`` etc.) or for a
    scikit-learn Pipeline whose final step is a linear estimator. Returns
    ``None`` for anything else (RandomForest, SVM-RBF, DummyClassifier).
    """
    if fitted_estimator is None:
        return None
    if hasattr(fitted_estimator, "named_steps"):
        final = list(fitted_estimator.named_steps.values())[-1]
    else:
        final = fitted_estimator
    coef = getattr(final, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef)
    if coef.ndim == 2 and coef.shape[0] == 1:
        return [float(v) for v in coef[0].tolist()]
    if coef.ndim == 1:
        return [float(v) for v in coef.tolist()]
    return None


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
    sample_metadata=None,
    include_baselines=False,
    persist_predictions=False,
    dummy_strategy=None,
    capture_coefficients=False,
    feature_names=None,
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
    sample_metadata : dict[str, sequence], optional
        Extra per-row information (e.g. window start-sample indices, block
        ids) that will be sliced by test indices and attached to each fold
        row under a "provenance" key. Added in the 2026-09 audit so that a
        fold in the JSON can be re-located in the raw recording.
    include_baselines : bool
        If True, attach a "baselines" dict to each fold with majority-class,
        stratified-dummy, and uniform-dummy scores computed on that fold's
        test set alone. No baseline is shared across protocols.
    persist_predictions : bool
        If True, attach ``y_true``, ``y_pred``, ``y_proba`` lists to each fold
        row. Useful for downstream diagnostics and for reproducing the pooled
        result from disk without re-fitting.
    dummy_strategy : str, optional
        If provided (e.g. ``"prior"``, ``"stratified"``), replaces the real
        model with a ``DummyClassifier(strategy=...)`` at each fold. Used to
        expose whether pooled-metric anomalies (e.g. below-chance pooled AUC
        on single-class folds) are artefacts of the split rather than signal.
    capture_coefficients : bool
        If True and the fitted estimator (or its Pipeline final step) is
        linear, attach ``coefficients`` (signed, in the fitted feature-order)
        to every fold row. Round 4 (Task 8): the figure code reads these
        instead of retraining.
    feature_names : sequence of str, optional
        Names for the coefficient vector, stored in ``config`` if provided.
        Order must match the coefficients returned by the fitted estimator.

    Returns
    -------
    dict with keys ``folds``, ``subject_mean``, ``subject_bootstrap_ci``,
    ``pooled_descriptive_metrics``, ``config``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    sample_metadata = sample_metadata or {}

    if tune and not param_grid:
        raise ValueError("param_grid is required when tune=True")

    if dummy_strategy is None:
        probe = model_factory()
        if hasattr(probe, "classes_"):
            raise ValueError("model_factory must return an unfitted estimator")

    logo = LeaveOneGroupOut()
    folds, pooled_true, pooled_pred, pooled_proba = [], [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = groups[test_idx][0]

        if dummy_strategy is not None:
            model = DummyClassifier(strategy=dummy_strategy,
                                    random_state=random_state)
            model.fit(X[train_idx], y[train_idx])
            chosen = None
        elif tune:
            model = model_factory()
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
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            chosen = None

        y_pred = model.predict(X[test_idx])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X[test_idx])
            # DummyClassifier may drop a column if training was single-class.
            if proba.shape[1] == 2:
                proba = proba[:, 1]
            else:
                # Only one class was in training; positive-class prob is 0 or 1
                # depending on which class the classifier learned.
                only_cls = int(model.classes_[0])
                proba = np.full(len(test_idx), float(only_cls))
        else:
            proba = y_pred.astype(float)

        row = _fold_metrics(y[test_idx], y_pred, proba)
        row["subject_id"] = (
            subject_labels.get(held_out, str(held_out))
            if subject_labels
            else str(held_out)
        )
        if chosen is not None:
            row["chosen_params"] = {k: str(v) for k, v in chosen.items()}
        if sample_metadata or include_baselines:
            provenance = {
                "n_train": int(len(train_idx)),
                "train_indices_hash": _index_hash(train_idx),
                "test_indices_hash": _index_hash(test_idx),
            }
            for key, arr in sample_metadata.items():
                arr = np.asarray(arr)
                provenance[f"test_{key}"] = [
                    _jsonable(v) for v in arr[test_idx].tolist()
                ]
            row["provenance"] = provenance
        if include_baselines:
            row["baselines"] = per_split_baselines(
                y[train_idx], y[test_idx], random_state=random_state
            )
        if persist_predictions:
            row["y_true"] = [int(v) for v in y[test_idx].tolist()]
            row["y_pred"] = [int(v) for v in np.asarray(y_pred).tolist()]
            row["y_proba"] = [float(v) for v in np.asarray(proba).tolist()]
        if capture_coefficients and dummy_strategy is None:
            coef = _extract_fold_coefficients(model)
            if coef is not None:
                row["coefficients"] = coef
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
            "dummy_strategy": dummy_strategy,
            "predictions_persisted": bool(persist_predictions),
            "coefficients_captured": bool(
                capture_coefficients and dummy_strategy is None
            ),
            "feature_names": (
                [str(n) for n in feature_names] if feature_names is not None
                else None
            ),
        },
    }


def _jsonable(value):
    """Convert numpy scalars to plain Python scalars for JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _inner_group_cv(train_groups, n_splits, random_state):
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(train_groups))
    return GroupKFold(n_splits=min(n_splits, n_groups))


def evaluate_naive_split(X, y, model_factory, n_splits=5, random_state=42,
                         include_baselines=False):
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
    if include_baselines:
        # The naive split does not have a single held-out set. Report the
        # baseline the classifier is being compared to on the pooled dummy,
        # using the same y as train and test for the majority calculation.
        out["baselines"] = per_split_baselines(y, y, random_state=random_state)
    return out


def evaluate_within_subject_shuffled_cv(
    X,
    y,
    groups,
    model_factory,
    *,
    n_splits=5,
    random_state=42,
    capture_coefficients=False,
    feature_names=None,
):
    """Per-subject stratified K-fold CV with optional coefficient capture.

    One row per subject, containing the subject's per-fold-averaged
    predictions (pooled by ``cross_val_predict``) plus, optionally, the
    mean-across-folds signed coefficient vector. Round 4 (Task 8): callers
    can persist ``coefficients`` in the results JSON and never refit.

    Trials from the same recording run appear in both train and test, so the
    per-subject scores are mildly optimistic relative to leave-one-run-out.
    Kept for direct comparison to published within-subject numbers.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    rows = []
    for s in np.unique(groups):
        mask = groups == s
        Xs, ys = X[mask], y[mask]
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)

        proba = cross_val_predict(
            model_factory(), Xs, ys, cv=cv, method="predict_proba"
        )[:, 1]
        pred = (proba >= 0.5).astype(int)

        row = {
            "subject_id": str(int(s)),
            "n_test": int(len(ys)),
            "balanced_accuracy": float(balanced_accuracy_score(ys, pred)),
            "macro_f1": float(f1_score(ys, pred, average="macro",
                                       zero_division=0)),
            "roc_auc": float(roc_auc_score(ys, proba)),
            "y_true": [int(v) for v in ys.tolist()],
            "y_pred": [int(v) for v in pred.tolist()],
            "y_proba": [float(v) for v in proba.tolist()],
        }

        if capture_coefficients:
            fold_coefs = []
            for tr_idx, _te_idx in cv.split(Xs, ys):
                est = model_factory()
                est.fit(Xs[tr_idx], ys[tr_idx])
                c = _extract_fold_coefficients(est)
                if c is not None:
                    fold_coefs.append(c)
            if fold_coefs:
                fold_coefs = np.asarray(fold_coefs)
                row["coefficients_mean"] = [float(v) for v in
                                            fold_coefs.mean(axis=0).tolist()]
                row["coefficients_per_fold"] = [
                    [float(v) for v in c.tolist()] for c in fold_coefs
                ]

        rows.append(row)

    return {
        "folds": rows,
        "config": {
            "n_splits": int(n_splits),
            "n_subjects": int(len(np.unique(groups))),
            "n_features": int(X.shape[1]),
            "random_state": random_state,
            "coefficients_captured": bool(capture_coefficients),
            "feature_names": (
                [str(n) for n in feature_names] if feature_names is not None
                else None
            ),
        },
    }


def evaluate_expanding_window(
    X,
    y,
    groups,
    model_factory,
    *,
    n_init_groups,
    step_groups=1,
    gap_groups=0,
    random_state=42,
    sample_metadata=None,
    include_baselines=True,
):
    """Walk-forward evaluation at the granularity of groups (label blocks).

    Fits a fresh estimator per fold, so any preprocessing inside the pipeline
    is fitted on training rows only. The temporal gap is applied at the group
    level, matching the leave-one-block-out grouping used elsewhere.

    Returns a dict with ``folds``, ``subject_mean``, ``subject_bootstrap_ci``,
    ``pooled_descriptive_metrics``, and ``config``. The "subject" naming is
    reused for parity with ``evaluate_loso`` even though each fold here is a
    label block rather than a person.
    """
    from .temporal import expanding_window_splits_by_group

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    sample_metadata = sample_metadata or {}

    probe = model_factory()
    if hasattr(probe, "classes_"):
        raise ValueError("model_factory must return an unfitted estimator")

    folds, pooled_true, pooled_pred, pooled_proba = [], [], [], []
    for train_idx, test_idx in expanding_window_splits_by_group(
        groups, n_init_groups=n_init_groups, step=step_groups,
        gap_groups=gap_groups,
    ):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]

        row = _fold_metrics(y[test_idx], y_pred, proba)
        row["subject_id"] = str(groups[test_idx][0])
        provenance = {
            "n_train": int(len(train_idx)),
            "train_indices_hash": _index_hash(train_idx),
            "test_indices_hash": _index_hash(test_idx),
            "gap_groups": int(gap_groups),
        }
        for key, arr in sample_metadata.items():
            arr = np.asarray(arr)
            provenance[f"test_{key}"] = [
                _jsonable(v) for v in arr[test_idx].tolist()
            ]
        row["provenance"] = provenance
        if include_baselines:
            row["baselines"] = per_split_baselines(
                y[train_idx], y[test_idx], random_state=random_state
            )
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
            "n_folds": n_valid,
            "n_above_chance": int((vals > 0.5).sum()) if n_valid else 0,
        }
        subject_ci[m] = (
            bootstrap_ci(vals, random_state=random_state)
            if n_valid > 1
            else None
        )

    if folds:
        pt = np.concatenate(pooled_true)
        pp = np.concatenate(pooled_pred)
        pr = np.concatenate(pooled_proba)
        pooled = _fold_metrics(pt, pp, pr)
        pooled["_warning"] = (
            "Descriptive only. Pools correlated epochs across folds. Not the "
            "inferential result."
        )
    else:
        pooled = None

    return {
        "folds": folds,
        "subject_mean": subject_mean,
        "subject_bootstrap_ci": subject_ci,
        "pooled_descriptive_metrics": pooled,
        "config": {
            "n_folds": len(folds),
            "n_init_groups": int(n_init_groups),
            "step_groups": int(step_groups),
            "gap_groups": int(gap_groups),
            "n_features": int(X.shape[1]),
            "random_state": random_state,
        },
    }
