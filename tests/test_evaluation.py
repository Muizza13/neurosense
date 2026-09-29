import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.evaluation import (
    evaluate_loso,
    loro_subject_summary,
    merged_block_groups,
    per_split_baselines,
)
from src.core.statistics import bootstrap_ci, paired_subject_delta


def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=5000, random_state=42)),
    ])


def toy_data(n_subjects=6, n_per=40, effect=2.0, seed=0):
    """Label-driven signal plus a large per-subject offset."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in range(n_subjects):
        offset = rng.normal(0, 10)  # subject fingerprint, unrelated to label
        labels = rng.integers(0, 2, n_per)
        feats = rng.standard_normal((n_per, 4)) + offset
        feats[:, 0] += effect * labels
        X.append(feats); y.append(labels); g.append(np.full(n_per, s))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def test_one_fold_per_subject():
    X, y, g = toy_data(n_subjects=6)
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=200)
    assert len(res["folds"]) == 6
    assert {f["subject_id"] for f in res["folds"]} == {str(i) for i in range(6)}


def test_every_fold_reports_the_required_metrics():
    X, y, g = toy_data()
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=200)
    for fold in res["folds"]:
        for key in ("subject_id", "n_test", "balanced_accuracy", "macro_f1",
                    "roc_auc", "confusion_matrix"):
            assert key in fold
        assert np.array(fold["confusion_matrix"]).shape == (2, 2)


def test_test_rows_are_exactly_one_subject():
    X, y, g = toy_data(n_subjects=5, n_per=30)
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=200)
    assert sum(f["n_test"] for f in res["folds"]) == len(y)
    assert all(f["n_test"] == 30 for f in res["folds"])


def test_headline_is_subject_level_not_pooled():
    X, y, g = toy_data()
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=500)
    mean = res["subject_mean"]["balanced_accuracy"]
    per_subject = [f["balanced_accuracy"] for f in res["folds"]]
    assert mean["mean"] == pytest.approx(np.mean(per_subject))
    assert mean["n_subjects"] == len(per_subject)
    assert "_warning" in res["pooled_descriptive_metrics"]


def test_bootstrap_ci_resamples_subjects():
    X, y, g = toy_data(n_subjects=8)
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=2000)
    ci = res["subject_bootstrap_ci"]["balanced_accuracy"]
    assert ci["n_subjects"] == 8            # not the 320 epochs
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_auc_is_none_when_a_subject_has_one_class():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 3))
    y = np.concatenate([rng.integers(0, 2, 40), np.ones(20, dtype=int)])
    g = np.concatenate([np.zeros(20), np.ones(20), np.full(20, 2)]).astype(int)
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=200)
    single = [f for f in res["folds"] if f["subject_id"] == "2"][0]
    assert single["roc_auc"] is None
    # and that subject is excluded from the AUC mean rather than counted as 0
    assert res["subject_mean"]["roc_auc"]["n_subjects"] == 2


def test_reproducible_across_runs():
    X, y, g = toy_data()
    a = evaluate_loso(X, y, g, make_pipeline, random_state=42, n_boot=500)
    b = evaluate_loso(X, y, g, make_pipeline, random_state=42, n_boot=500)
    assert a["subject_bootstrap_ci"] == b["subject_bootstrap_ci"]


def test_tuning_requires_a_grid():
    X, y, g = toy_data()
    with pytest.raises(ValueError):
        evaluate_loso(X, y, g, make_pipeline, tune=True)


def test_prefitted_estimator_is_rejected():
    X, y, g = toy_data()
    fitted = make_pipeline().fit(X, y)
    with pytest.raises(ValueError):
        evaluate_loso(X, y, g, lambda: fitted, n_boot=100)


def test_paired_delta_is_computed_over_subjects():
    base = np.array([0.50, 0.55, 0.60, 0.45])
    after = np.array([0.60, 0.58, 0.72, 0.44])
    out = paired_subject_delta(base, after, n_boot=2000)
    assert out["n_subjects"] == 4
    assert out["n_improved"] == 3
    assert out["mean_delta"] == pytest.approx(np.mean(after - base))


def test_bootstrap_needs_more_than_one_subject():
    assert bootstrap_ci(np.array([0.6])) is None


def test_persist_predictions_stores_arrays_on_every_fold():
    X, y, g = toy_data(n_subjects=4)
    res = evaluate_loso(X, y, g, make_pipeline, n_boot=200,
                        persist_predictions=True)
    for fold in res["folds"]:
        assert len(fold["y_true"]) == fold["n_test"]
        assert len(fold["y_pred"]) == fold["n_test"]
        assert len(fold["y_proba"]) == fold["n_test"]
        assert all(v in (0, 1) for v in fold["y_true"])
    assert res["config"]["predictions_persisted"] is True


def test_dummy_strategy_bypasses_the_model_factory():
    X, y, g = toy_data(n_subjects=5)

    def would_fail():
        raise AssertionError("factory should be unused when dummy_strategy is set")

    res = evaluate_loso(
        X, y, g, would_fail, n_boot=200, dummy_strategy="prior",
        persist_predictions=True,
    )
    # DummyClassifier(strategy="prior") always predicts training majority.
    for fold in res["folds"]:
        assert len(set(fold["y_pred"])) == 1


def test_merged_block_groups_partitions_contiguous_blocks():
    block_ids = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4])
    # size 2, five native blocks: leftovers join the last complete group.
    # {0: [0,1], 1: [2,3,4]}
    supers = merged_block_groups(block_ids, blocks_per_superblock=2)
    assert list(supers) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_merged_block_remainder_is_not_its_own_fold():
    block_ids = np.repeat(np.arange(19), 2)
    supers = merged_block_groups(block_ids, blocks_per_superblock=4)
    assert set(supers.tolist()) == {0, 1, 2, 3}
    for native in range(12, 19):
        assert set(supers[block_ids == native].tolist()) == {3}
    for native in range(4):
        assert set(supers[block_ids == native].tolist()) == {0}


def test_most_frequent_dummy_is_fit_on_training_labels():
    # Test labels are all class 1. A test-set majority score would be 1.
    # The training majority is class 0, so every prediction is wrong.
    baselines = per_split_baselines(
        np.array([0, 0, 0, 0, 1]),
        np.array([1, 1, 1]),
        random_state=42,
    )
    assert baselines["training_majority_class"] == 0
    assert baselines["most_frequent"]["balanced_accuracy"] == 0.0
    assert baselines["most_frequent"]["roc_auc"] is None
    assert "majority_class_accuracy" not in baselines


def test_loro_bootstrap_resamples_subjects_not_runs():
    rows = []
    for subject in (0, 1, 2):
        for run in (4, 8, 12):
            rows.append({
                "subject_id": str(subject),
                "held_out_run": run,
                "balanced_accuracy": 0.4 + 0.1 * subject,
                "macro_f1": 0.4,
                "roc_auc": 0.6,
            })
    summary = loro_subject_summary(
        rows, random_state=42, n_boot=200,
        limitation="test",
    )
    assert summary["n_run_folds"] == 9
    assert summary["subject_mean"]["balanced_accuracy"]["n_subjects"] == 3
    assert summary["subject_bootstrap_ci"]["balanced_accuracy"]["n_subjects"] == 3
    assert "fold_bootstrap_ci" not in summary
    # Subject means are 0.4, 0.5, 0.6. The mean of those is 0.5, which is
    # also the mean of the nine run rows because each subject has three runs.
    assert summary["subject_mean"]["balanced_accuracy"]["mean"] == pytest.approx(0.5)


def test_merged_block_groups_respects_first_seen_order():
    # Block ids are not required to be sorted; the mapping tracks the order
    # in which each new id first appears.
    block_ids = np.array([5, 5, 3, 3, 3, 8, 8, 1, 1])
    supers = merged_block_groups(block_ids, blocks_per_superblock=2)
    # 5 -> 0, 3 -> 0, 8 -> 1, 1 -> 1
    assert list(supers) == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_merged_block_groups_rejects_zero_size():
    with pytest.raises(ValueError):
        merged_block_groups(np.array([0, 1, 2]), blocks_per_superblock=0)
