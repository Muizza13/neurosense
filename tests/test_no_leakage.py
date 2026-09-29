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
from src.core.features import (
    STANDARD_BANDS,
    ArtifactClipper,
    EpochBandPower,
    flatten_epochs,
)
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


# ---------------------------------------------------------------------------
# Regression guard added in the 2026-09 audit.
#
# The claim under test:
#   For a pipeline that fits data-dependent preprocessing (artifact clipping,
#   scaling), mutating held-out rows cannot change either the fitted
#   preprocessing parameters or the transformed training features.
#
# If a future refactor moves clipping outside the pipeline, or globally fits a
# scaler on X before splitting, this test detects it.
# ---------------------------------------------------------------------------
def _phase1_pipeline(channels, n_times, sfreq=128.0):
    return Pipeline([
        ("bandpower", EpochBandPower(sfreq, STANDARD_BANDS, channels, n_times)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=5000, random_state=42)),
    ])


def _synthetic_phase1_windows(n_windows=40, n_channels=4, n_times=128, seed=0):
    """Windows with two label classes and a single feature-carrying channel."""
    rng = np.random.default_rng(seed)
    channels = [f"CH{i}" for i in range(n_channels)]
    labels = rng.integers(0, 2, n_windows)
    epochs = rng.standard_normal((n_windows, n_channels, n_times))
    epochs[:, 0, :] += 0.5 * labels[:, None]
    return flatten_epochs(epochs), labels, channels, n_times


def test_held_out_mutation_cannot_change_training_preprocessing():
    """The regression guard the 2026-09 audit was asked to add.

    Fit the Phase 1 preprocessing on the training rows. Snapshot the fitted
    clipper thresholds, the fitted scaler statistics, and the transformed
    training features. Then mutate several held-out rows to catastrophic
    values, refit on the same training rows, and require every snapshot to
    be bit-identical.
    """
    X, y, channels, n_times = _synthetic_phase1_windows(n_windows=40)
    train_idx = np.arange(0, 30)
    test_idx = np.arange(30, 40)

    pipe = _phase1_pipeline(channels, n_times)
    pipe.fit(X[train_idx], y[train_idx])

    clip_lo = pipe.named_steps["bandpower"].clipper_.lo_.copy()
    clip_hi = pipe.named_steps["bandpower"].clipper_.hi_.copy()
    scaler_mean = pipe.named_steps["scaler"].mean_.copy()
    scaler_scale = pipe.named_steps["scaler"].scale_.copy()
    train_features = pipe.named_steps["bandpower"].transform(X[train_idx]).copy()

    X_mutated = X.copy()
    X_mutated[test_idx] *= 1e6                 # electrode pop on every test row
    X_mutated[test_idx[0]] = np.full_like(X_mutated[test_idx[0]], 1e12)

    pipe2 = _phase1_pipeline(channels, n_times)
    pipe2.fit(X_mutated[train_idx], y[train_idx])

    np.testing.assert_array_equal(clip_lo, pipe2.named_steps["bandpower"].clipper_.lo_)
    np.testing.assert_array_equal(clip_hi, pipe2.named_steps["bandpower"].clipper_.hi_)
    np.testing.assert_array_equal(scaler_mean, pipe2.named_steps["scaler"].mean_)
    np.testing.assert_array_equal(scaler_scale, pipe2.named_steps["scaler"].scale_)
    np.testing.assert_array_equal(
        train_features,
        pipe2.named_steps["bandpower"].transform(X_mutated[train_idx]),
    )


def test_globally_fitted_clipper_would_be_contaminated_by_test_rows():
    """Negative control: verifies the guard above is not vacuous.

    If clipping is fit on the whole recording, mutating any subset shifts the
    fitted thresholds. This test would fail if `ArtifactClipper` somehow
    ignored large values, which would make the positive test above
    meaningless.
    """
    X, y, channels, n_times = _synthetic_phase1_windows(n_windows=40)
    all_epochs = X.reshape(len(X), len(channels), n_times)

    global_fit_clean = ArtifactClipper(lo_q=0.001, hi_q=0.999).fit(all_epochs)

    contaminated = all_epochs.copy()
    contaminated[30:] *= 1e6
    contaminated[30] = np.full_like(contaminated[30], 1e12)
    global_fit_dirty = ArtifactClipper(lo_q=0.001, hi_q=0.999).fit(contaminated)

    assert np.any(global_fit_dirty.hi_ > global_fit_clean.hi_ * 10)
