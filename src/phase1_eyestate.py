"""Phase 1: UCI EEG Eye State, rerun after the 2026-09 audit.

Note on the "confirmatory" framing: this file is a rerun on the same UCI
recording that has been inspected many times in this project. It is not an
untouched confirmatory study of Phase 1. The audit's job is to make the
existing evaluation honest, not to provide fresh evidence about the eye-
state paradigm.


Changes from the pre-audit pipeline (still true):

1. Artifact clipping is fitted inside every fold via `EpochBandPower` in the
   Pipeline. Nothing data-dependent is fitted on the whole recording.
2. Leave-one-block-out reuses the shared `evaluate_loso` on contiguous label
   blocks. Phase 1 has one subject, so the block is the only leakage-safe
   grouping available.

Audit additions (2026-09):

3. Every leakage-safe protocol reports per-split baselines: majority-class
   accuracy, and balanced accuracy of a stratified and a uniform dummy
   classifier. No baseline is shared across protocols.
4. Temporal dependence of the fold-safe features is measured, and the choice
   of temporal gap for the chronological and expanding-window protocols is
   derived from the measurement rather than picked by hand.
5. An expanding-window (walk-forward across label blocks) evaluation is added
   as an additional leakage-safe protocol.
6. Every fold row carries a `provenance` block with test-window start-sample
   indices, block ids, and index hashes, so a fold can be re-located in the
   raw recording.

Run:  bash scripts/download_data.sh  (or just the UCI half)
      python src/phase1_eyestate.py
"""
import sys
import warnings

sys.path.insert(0, ".")

# Single-class LOBO test blocks trigger benign warnings from sklearn about
# predicted classes not present in y_true. They are expected here and are
# handled in reporting; silence them so the console output is readable.
warnings.filterwarnings(
    "ignore", message="y_pred contains classes not in y_true"
)
warnings.filterwarnings(
    "ignore", message="A single label was found in 'y_true' and 'y_pred'"
)

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.core.evaluation import (
    evaluate_expanding_window,
    evaluate_loso,
    evaluate_naive_split,
    merged_block_groups,
    per_split_baselines,
)
from src.core.features import (
    STANDARD_BANDS,
    ArtifactClipper,
    EpochBandPower,
    band_power,
    flatten_epochs,
    label_blocks,
    make_continuous_windows,
)
from src.core.results import format_ci, save_results
from src.core.temporal import (
    choose_temporal_gap,
    feature_lag_autocorr,
)

ARFF = "data/raw/EEG Eye State.arff"
SFREQ = 128.0
WIN_SEC = 1.0
RANDOM_STATE = 42

PRIMARY_NAME = "LogisticRegression(C=1.0, balanced)"
CHRONO_TRAIN_FRAC = 0.70
AUTOCORR_LAGS = (1, 2, 3, 4, 5)
AUTOCORR_THRESHOLD = 0.30            # gap chosen at first lag with |ac| < 0.30

# ---------------------------------------------------------------------------
# Prespecified merged-block grouping rule (2026-09 audit, Task 4)
#
# The single continuous recording alternates eyes-open and eyes-closed
# stretches, so every native contiguous-label block is single-class. Per-fold
# AUC is undefined and pooled AUC from separately-trained folds is a known
# splitting artefact (each fold model sees a different class prior).
#
# To produce a leakage-safe temporal fold whose held-out set can contain both
# classes we merge consecutive native blocks into super-blocks of a fixed
# size. Rule (declared before any evaluation):
#
#     MERGED_BLOCK_SIZE = 4  native blocks per super-block.
#
# Chosen for two structural reasons:
#   (a) with N = 19 native blocks we get ceil(19 / 4) = 5 super-blocks,
#       enough folds for a percentile bootstrap over folds;
#   (b) 4 blocks span at least one full open/closed transition regardless of
#       where the block boundaries fall, so each super-block covers both
#       classes.
#
# The size is not tuned on results. We report merged-LOBO scores at this
# fixed size and separately show DummyClassifier(strategy="prior") on the
# same splits so the reader can see how much of any pooled artefact
# survives when nothing is being learned.
# ---------------------------------------------------------------------------
MERGED_BLOCK_SIZE = 4


def make_primary(channels, n_times):
    return Pipeline([
        ("bandpower", EpochBandPower(SFREQ, STANDARD_BANDS, channels, n_times)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=5000, random_state=RANDOM_STATE)),
    ])


def load():
    df = pd.DataFrame(arff.loadarff(ARFF)[0])
    df["eyeDetection"] = df["eyeDetection"].astype(int)
    channels = [c for c in df.columns if c != "eyeDetection"]
    x = df[channels].values
    y = df["eyeDetection"].values

    blocks_per_sample = label_blocks(y)
    epochs, labels, starts = make_continuous_windows(x, y, SFREQ, WIN_SEC)
    block_ids = blocks_per_sample[starts]
    return flatten_epochs(epochs), labels, block_ids, starts, channels, epochs.shape[2]


def _training_fitted_features(X, split, channels, n_times):
    """Extract features with a clipper fit on the training half of the split.

    Used by the temporal-dependence diagnostic. Everything else in Phase 1
    lets `EpochBandPower` in the Pipeline handle this per fold. Here we
    just need one representative feature matrix to measure autocorrelation
    of what the model actually consumes.
    """
    epochs = X.reshape(len(X), len(channels), n_times)
    clipper = ArtifactClipper().fit(epochs[:split])
    clipped = clipper.transform(epochs)
    feats, _ = band_power(clipped, SFREQ, STANDARD_BANDS, channels)
    return feats


def temporal_diagnostic(X, split, channels, n_times):
    """Return the autocorrelation report and the recommended gap."""
    feats = _training_fitted_features(X, split, channels, n_times)
    report = feature_lag_autocorr(feats, lags=AUTOCORR_LAGS)
    recommendation = choose_temporal_gap(report, threshold=AUTOCORR_THRESHOLD)
    return {
        "lag_report": report,
        "gap_choice": recommendation,
        "note": (
            "Autocorrelation is measured on band-power features extracted with "
            "a clipper fitted on the first "
            f"{int(CHRONO_TRAIN_FRAC * 100)}% of windows (the chronological "
            "training half), so the values reflect what the model actually "
            "consumes rather than the raw voltages."
        ),
    }


def chronological_holdout(X, y, factory, starts, block_ids,
                          frac=CHRONO_TRAIN_FRAC, gap_windows=0):
    """Train on the first `frac` of time; test on the last (1 - frac).

    A `gap_windows` argument drops that many windows between the training
    tail and the test head so short-range temporal memory cannot leak across
    the boundary.
    """
    n = len(y)
    split = int(frac * n)
    if gap_windows < 0:
        raise ValueError("gap_windows must be non-negative")
    if split + gap_windows >= n:
        raise ValueError(
            f"gap ({gap_windows}) leaves no test windows at split={split} "
            f"of {n}"
        )
    train_idx = np.arange(0, split)
    test_idx = np.arange(split + gap_windows, n)

    model = factory()
    model.fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)

    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, roc_auc_score
    )
    yte = y[test_idx]
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "gap_windows": int(gap_windows),
        "train_fraction": float(frac),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "roc_auc": (float(roc_auc_score(yte, proba))
                    if len(np.unique(yte)) > 1 else None),
        "baselines": per_split_baselines(y[train_idx], yte,
                                         random_state=RANDOM_STATE),
        "provenance": {
            "train_start_sample": int(starts[train_idx[0]]),
            "train_end_sample_exclusive": int(starts[train_idx[-1]]
                                              + int(SFREQ * WIN_SEC)),
            "test_start_sample": int(starts[test_idx[0]]),
            "test_end_sample_exclusive": int(starts[test_idx[-1]]
                                             + int(SFREQ * WIN_SEC)),
            "test_block_ids": [int(b) for b in block_ids[test_idx]],
        },
    }


def _log_lobo_or_expanding(name, result):
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        ci = result["subject_bootstrap_ci"].get(m)
        if ci is None:
            print(f"  {m:20s} n/a (single-class folds only)")
            continue
        print(f"  {m:20s} {format_ci(ci)}")


def _describe_baselines(split_name, baselines):
    print(f"  baselines ({split_name}):")
    print(f"    positive-rate test  : {baselines['positive_rate_test']:.3f}")
    print(f"    majority-class acc  : {baselines['majority_class_accuracy']:.3f}")
    print(f"    stratified dummy balAcc: "
          f"{baselines['dummy_stratified_balanced_accuracy']:.3f}")
    print(f"    uniform dummy    balAcc: "
          f"{baselines['dummy_uniform_balanced_accuracy']:.3f}")


def main():
    import os
    for d in ("reports/figures", "reports/results", "models"):
        os.makedirs(d, exist_ok=True)
    X, y, blocks, starts, channels, n_times = load()
    factory = lambda: make_primary(channels, n_times)

    print(f"Phase 1: {len(y)} windows of {WIN_SEC}s, {len(channels)} channels, "
          f"{len(np.unique(blocks))} label blocks, 1 subject")
    print(f"Primary model (prespecified): {PRIMARY_NAME}")
    print(f"Class balance: open {int((y == 0).sum())} / "
          f"closed {int((y == 1).sum())}\n")

    # ------------------------------------------------------------------
    # Temporal dependence diagnostic and gap choice
    # ------------------------------------------------------------------
    chrono_split = int(CHRONO_TRAIN_FRAC * len(y))
    diagnostic = temporal_diagnostic(X, chrono_split, channels, n_times)
    gap_windows = diagnostic["gap_choice"]["recommended_gap_windows"]
    print("=== TEMPORAL DEPENDENCE (fold-safe band-power features) ===")
    for lag, ac in zip(diagnostic["lag_report"]["lags"],
                       diagnostic["lag_report"]["mean_abs_autocorr"]):
        print(f"  lag {lag}: mean |autocorr| = {ac:.3f}")
    print(f"  chosen gap: {gap_windows} windows "
          f"({diagnostic['gap_choice']['reason']}, "
          f"threshold={diagnostic['gap_choice']['threshold']})\n")

    # ------------------------------------------------------------------
    # Naive random window split (descriptive contrast, not a valid estimate)
    # ------------------------------------------------------------------
    print("=== NAIVE RANDOM WINDOW SPLIT (leaky, descriptive only) ===")
    naive = evaluate_naive_split(X, y, factory, random_state=RANDOM_STATE,
                                 include_baselines=True)
    print(f"  balanced_accuracy    {naive['balanced_accuracy']:.3f}")
    print(f"  macro_f1             {naive['macro_f1']:.3f}")
    print(f"  roc_auc              {naive['roc_auc']:.3f}")
    _describe_baselines("naive random", naive["baselines"])

    # ------------------------------------------------------------------
    # Chronological 70/30 holdout with the diagnostic-chosen gap
    # ------------------------------------------------------------------
    print("\n=== CHRONOLOGICAL 70/30 HOLDOUT (leakage-safe, temporal gap) ===")
    chrono = chronological_holdout(
        X, y, factory, starts, blocks,
        frac=CHRONO_TRAIN_FRAC, gap_windows=gap_windows,
    )
    print(f"  balanced_accuracy    {chrono['balanced_accuracy']:.3f}")
    print(f"  macro_f1             {chrono['macro_f1']:.3f}")
    if chrono["roc_auc"] is None:
        print("  roc_auc              n/a")
    else:
        print(f"  roc_auc              {chrono['roc_auc']:.3f}")
    print(f"  gap                  {chrono['gap_windows']} windows")
    _describe_baselines("chronological", chrono["baselines"])

    # ------------------------------------------------------------------
    # Expanding-window across label blocks (walk-forward, leakage-safe)
    # ------------------------------------------------------------------
    print("\n=== EXPANDING-WINDOW (walk-forward across label blocks) ===")
    n_unique_blocks = len(np.unique(blocks))
    # Start with a third of the blocks in the initial training window, walk
    # one block at a time. Gap is measured in blocks; if the recommended gap
    # in windows is 0 we still keep 0 blocks (adjacent-block boundary is
    # already leakage-safe under the block grouping).
    gap_groups = 1 if gap_windows > 0 else 0
    n_init_groups = max(3, n_unique_blocks // 3)
    expanding = evaluate_expanding_window(
        X, y, blocks, factory,
        n_init_groups=n_init_groups,
        step_groups=1,
        gap_groups=gap_groups,
        random_state=RANDOM_STATE,
        sample_metadata={"start_sample": starts, "block_id": blocks},
        include_baselines=True,
    )
    _log_lobo_or_expanding("expanding-window", expanding)
    print(f"  n_folds              {expanding['config']['n_folds']}")
    print(f"  n_init_groups        {expanding['config']['n_init_groups']}")
    print(f"  gap_groups           {expanding['config']['gap_groups']}")

    # ------------------------------------------------------------------
    # Leave-one-merged-block-out (Task 4 primary group protocol)
    # ------------------------------------------------------------------
    print(f"\n=== LEAVE-ONE-MERGED-BLOCK-OUT "
          f"(super-block size = {MERGED_BLOCK_SIZE} native blocks, "
          f"pre-specified) ===")
    super_blocks = merged_block_groups(blocks, MERGED_BLOCK_SIZE)
    unique_super = np.unique(super_blocks)
    class_summary = []
    for s in unique_super:
        mask = super_blocks == s
        n0 = int((y[mask] == 0).sum())
        n1 = int((y[mask] == 1).sum())
        class_summary.append({
            "super_block_id": int(s),
            "n_test": int(mask.sum()),
            "n_open": n0,
            "n_closed": n1,
            "is_both_classes": bool(n0 > 0 and n1 > 0),
        })
        print(f"  super-block {int(s)}: n={int(mask.sum())}  "
              f"open={n0} closed={n1}  both_classes="
              f"{'yes' if n0 > 0 and n1 > 0 else 'NO'}")
    n_both = sum(1 for c in class_summary if c["is_both_classes"])
    print(f"  {n_both}/{len(class_summary)} folds contain both classes")

    phase1_feature_names = [f"{ch}_{b}" for ch in channels for b in STANDARD_BANDS]
    merged_lobo = evaluate_loso(
        X, y, super_blocks, factory,
        random_state=RANDOM_STATE,
        sample_metadata={"start_sample": starts,
                         "native_block_id": blocks,
                         "super_block_id": super_blocks},
        include_baselines=True,
        persist_predictions=True,
        capture_coefficients=True,
        feature_names=phase1_feature_names,
    )
    _log_lobo_or_expanding("merged-LOBO", merged_lobo)

    # DummyClassifier(strategy="prior") on the same splits: reproduces the
    # pooled-metric artefact so it is not read as electrode drift.
    dummy_merged = evaluate_loso(
        X, y, super_blocks,
        # factory unused when dummy_strategy is set; supply anything valid.
        factory,
        random_state=RANDOM_STATE,
        sample_metadata={"start_sample": starts,
                         "native_block_id": blocks,
                         "super_block_id": super_blocks},
        include_baselines=False,
        persist_predictions=True,
        dummy_strategy="prior",
    )
    print("  DummyClassifier(strategy='prior') on same merged splits:")
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        ci = dummy_merged["subject_bootstrap_ci"].get(m)
        if ci is None:
            print(f"    {m:20s} n/a")
        else:
            print(f"    {m:20s} {format_ci(ci)}")

    # ------------------------------------------------------------------
    # Leave-one-native-block-out (retained diagnostic; single-class folds)
    # ------------------------------------------------------------------
    print("\n=== LEAVE-ONE-NATIVE-BLOCK-OUT (diagnostic only, single-class "
          "folds; do not read pooled AUC as signal) ===")
    lobo = evaluate_loso(
        X, y, blocks, factory,
        random_state=RANDOM_STATE,
        sample_metadata={"start_sample": starts, "block_id": blocks},
        include_baselines=True,
        persist_predictions=True,
    )
    _log_lobo_or_expanding("native-LOBO", lobo)
    dummy_lobo = evaluate_loso(
        X, y, blocks, factory,
        random_state=RANDOM_STATE,
        sample_metadata={"start_sample": starts, "block_id": blocks},
        include_baselines=False,
        persist_predictions=True,
        dummy_strategy="prior",
    )
    print("  DummyClassifier(strategy='prior') on the same native splits:")
    pooled_dummy_auc = dummy_lobo["pooled_descriptive_metrics"].get("roc_auc")
    pooled_real_auc = lobo["pooled_descriptive_metrics"].get("roc_auc")
    print(f"    pooled AUC (real primary model): "
          f"{pooled_real_auc:.3f}" if pooled_real_auc is not None else
          "    pooled AUC (real primary model): n/a")
    print(f"    pooled AUC (dummy prior):        "
          f"{pooled_dummy_auc:.3f}" if pooled_dummy_auc is not None else
          "    pooled AUC (dummy prior):        n/a")
    print("  Interpretation: the dummy is learning nothing, so any non-0.5 "
          "pooled AUC here is a splitting artefact rather than electrode "
          "drift.")

    # ------------------------------------------------------------------
    # Secondary models under leave-one-merged-block-out (Task 4 primary
    # group protocol; single-class native-LOBO retained above only as a
    # labelled diagnostic).
    # ------------------------------------------------------------------
    print("\n=== SECONDARY MODELS (leave-one-merged-block-out) ===")
    secondary = {}
    for sname, model in [
        ("SVM-RBF", lambda: SVC(kernel="rbf", probability=True,
                                random_state=RANDOM_STATE)),
        ("RandomForest(300)", lambda: RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE)),
    ]:
        f = lambda m=model: Pipeline([
            ("bandpower", EpochBandPower(SFREQ, STANDARD_BANDS,
                                         channels, n_times)),
            ("scaler", StandardScaler()),
            ("model", m()),
        ])
        res = evaluate_loso(X, y, super_blocks, f, random_state=RANDOM_STATE)
        secondary[sname] = {
            "subject_mean": res["subject_mean"],
            "subject_bootstrap_ci": res["subject_bootstrap_ci"],
        }
        ci = res["subject_bootstrap_ci"]["balanced_accuracy"]
        print(f"  {sname:20s} "
              f"balAcc={format_ci(ci) if ci else 'n/a'}")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    path = save_results("phase1_results", {
        "phase": "1_eye_state",
        "dataset": "UCI EEG Eye State, single continuous recording, 1 subject",
        "primary_model": PRIMARY_NAME,
        "primary_model_prespecified": True,
        "grouping_unit": (
            "contiguous label block (single subject, so no LOSO)"
        ),
        "cross_subject": None,
        "cross_subject_note": "Not available. The dataset contains one subject.",
        "temporal_dependence": diagnostic,
        "temporal_gap_windows_used": int(gap_windows),
        "naive_random_split": naive,
        "chronological_holdout": chrono,
        "expanding_window": expanding,
        "leave_one_merged_block_out": {
            "super_block_size_native_blocks": MERGED_BLOCK_SIZE,
            "grouping_rule_prespecified": True,
            "grouping_rule_description": (
                "Contiguous native blocks partitioned into fixed-size "
                "super-blocks of MERGED_BLOCK_SIZE = 4. Rule declared in "
                "source before evaluation; not tuned on results."
            ),
            "fold_class_summary": class_summary,
            "n_folds_with_both_classes": int(n_both),
            "n_folds_total": int(len(class_summary)),
            "primary_model": merged_lobo,
            "dummy_prior_diagnostic": dummy_merged,
        },
        "leave_one_block_out_diagnostic": {
            "note": (
                "Retained only as a diagnostic. Each native block is single-"
                "class, so per-fold AUC is undefined and pooled AUC from "
                "separately-trained folds is a splitting artefact, not "
                "electrode drift. The paired dummy_prior_diagnostic "
                "reproduces the artefact with a classifier that learns "
                "nothing."
            ),
            "primary_model": lobo,
            "dummy_prior_diagnostic": dummy_lobo,
        },
        "secondary_models": secondary,
        "audit": {
            "version": "2026-09",
            "notes": (
                "Naive split retained as descriptive only. Chronological "
                "holdout and expanding-window run with a temporal gap "
                "chosen from a measured autocorrelation threshold. Every "
                "leakage-safe split reports its own baselines; no baseline "
                "is shared across protocols. Leave-one-merged-block-out "
                "(super-block size 4 native blocks, declared before "
                "evaluation) is now the primary group protocol; "
                "leave-one-native-block-out is retained only as a labelled "
                "diagnostic with a DummyClassifier(strategy='prior') "
                "comparison on the same splits."
            ),
        },
    })
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
