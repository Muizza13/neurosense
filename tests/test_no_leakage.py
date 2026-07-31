"""Regression tests for the failure modes this project is about.

These encode the thesis as executable checks: if someone later reintroduces a
naive split, a globally fitted scaler, or an epoch-level confidence interval,
one of these fails.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.evaluation import evaluate_loso, evaluate_naive_split
from src.core.features import ArtifactClipper
from src.core.statistics import subject_permutation_test


def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=5000, random_state=42)),
    ])


def subject_fingerprint_data(n_subjects=8, n_per=60, seed=0):
    """Data where the label is only decodable via a subject-specific code.

    Every subject carries its label in one private feature column and noise
    everywhere else. Within a subject the problem is easy. For a held-out
    subject it is impossible, because during training that subject's column
    contained nothing but noise, so the model assigns it no weight.

    This is the synthetic version of the Phase 2 motor imagery result: honest
    within-subject decoding, chance across subjects.
    """
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in range(n_subjects):
        labels = rng.integers(0, 2, n_per)
        feats = rng.standard_normal((n_per, n_subjects))
        feats[:, s] += 4.0 * labels          # private channel for this subject
        X.append(feats); y.append(labels); g.append(np.full(n_per, s))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def test_naive_split_inflates_relative_to_loso():
    """The core claim of the whole project, as a unit test."""
    X, y, g = subject_fingerprint_data()
    naive = evaluate_naive_split(X, y, make_pipeline)
    loso = evaluate_loso(X, y, g, make_pipeline, n_boot=500)
    honest = loso["subject_mean"]["balanced_accuracy"]["mean"]
    assert naive["balanced_accuracy"] > honest + 0.15
    assert honest < 0.60          # subject-specific code does not transfer
    assert "_warning" in naive


def test_no_training_row_belongs_to_the_held_out_subject():
    from sklearn.model_selection import LeaveOneGroupOut

    X, y, g = subject_fingerprint_data(n_subjects=5, n_per=20)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, g):
        assert set(g[train_idx]).isdisjoint(set(g[test_idx]))
        assert len(set(g[test_idx])) == 1


def test_scaler_is_fitted_inside_the_fold():
    """A pipeline sees only training rows; a pre-scaled matrix does not.

    Given a held-out subject with a wildly different scale, global scaling and
    fold-internal scaling must not produce identical results.
    """
    rng = np.random.default_rng(3)
    X, y, g = subject_fingerprint_data(n_subjects=6, n_per=40)
    X = X.copy()
    X[g == 5] *= 40.0             # held-out subject on a different scale

    inside = evaluate_loso(X, y, g, make_pipeline, n_boot=200)
    X_global = StandardScaler().fit_transform(X)   # the leak
    outside = evaluate_loso(
        X_global, y, g,
        lambda: LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=5000, random_state=42),
        n_boot=200,
    )
    a = [f["balanced_accuracy"] for f in inside["folds"]]
    b = [f["balanced_accuracy"] for f in outside["folds"]]
    assert a != b


def test_clipping_thresholds_never_come_from_held_out_data():
    rng = np.random.default_rng(7)
    train = rng.standard_normal((30, 3, 128))
    test = rng.standard_normal((10, 3, 128)) * 50.0

    train_only = ArtifactClipper(lo_q=0.01, hi_q=0.99).fit(train)
    contaminated = ArtifactClipper(lo_q=0.01, hi_q=0.99).fit(
        np.concatenate([train, test])
    )
    assert np.all(contaminated.hi_ > train_only.hi_)


def test_confidence_interval_is_not_computed_over_epochs():
    X, y, g = subject_fingerprint_data(n_subjects=6, n_per=100)
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=2000)
    ci = res["subject_bootstrap_ci"]["balanced_accuracy"]
    assert ci["n_subjects"] == 6
    assert ci["n_subjects"] != len(y)
    # 600 correlated epochs would give an implausibly tight interval
    assert (ci["hi"] - ci["lo"]) > 0.02


def test_shuffled_labels_land_at_chance_under_loso():
    X, y, g = subject_fingerprint_data(n_subjects=6, n_per=40)
    rng = np.random.default_rng(11)
    y_shuffled = rng.permutation(y)
    res = evaluate_loso(X, y_shuffled, g, make_pipeline, n_boot=500)
    ci = res["subject_bootstrap_ci"]["balanced_accuracy"]
    assert ci["lo"] < 0.5 < ci["hi"]


def test_permutation_null_sits_at_chance():
    X, y, g = subject_fingerprint_data(n_subjects=5, n_per=30)

    def _eval(Xa, ya, ga, factory):
        return evaluate_loso(Xa, ya, ga, factory, n_boot=100)

    out = subject_permutation_test(X, y, g, make_pipeline, _eval, n_perm=25)
    assert 0.40 < out["null_mean"] < 0.60
    assert out["p_value"] <= 1.0
