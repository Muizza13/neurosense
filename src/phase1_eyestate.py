"""Phase 1: UCI EEG Eye State, rerun through the shared core.

Changes from the original train.py:

1. Artifact clipping is fitted inside the fold. The original called
   clip_artifacts() on the entire 14980-sample recording before windowing and
   before the chronological split, so the winsorisation thresholds were
   computed partly from held-out future samples.

2. Band power extraction also happens inside the fold, via EpochBandPower.

3. Leave-one-block-out reuses the same evaluate_loso as Phase 2, with the
   contiguous label block as the grouping unit. Phase 1 has one subject, so the
   block is the only leakage-safe grouping available. Its cross-subject cell
   stays empty by construction.

Run:  bash scripts/download_data.sh  (or just the UCI half)
      python src/phase1_eyestate.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.core.evaluation import evaluate_loso, evaluate_naive_split
from src.core.features import (
    STANDARD_BANDS,
    EpochBandPower,
    flatten_epochs,
    label_blocks,
    make_continuous_windows,
)
from src.core.results import format_ci, save_results

ARFF = "data/raw/EEG Eye State.arff"
SFREQ = 128.0
WIN_SEC = 1.0
RANDOM_STATE = 42

PRIMARY_NAME = "LogisticRegression(C=1.0, balanced)"


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
    return flatten_epochs(epochs), labels, block_ids, channels, epochs.shape[2]


def chronological_holdout(X, y, factory, frac=0.70):
    """Train on the first 70 percent of time, test on the last 30 percent."""
    split = int(frac * len(y))
    model = factory()
    model.fit(X[:split], y[:split])
    proba = model.predict_proba(X[split:])[:, 1]
    pred = (proba >= 0.5).astype(int)
    yte = y[split:]
    return {
        "n_train": int(split),
        "n_test": int(len(yte)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "roc_auc": (float(roc_auc_score(yte, proba))
                    if len(np.unique(yte)) > 1 else None),
        "majority_baseline_accuracy": float(max(np.mean(yte), 1 - np.mean(yte))),
    }


def main():
    X, y, blocks, channels, n_times = load()
    factory = lambda: make_primary(channels, n_times)

    print(f"Phase 1: {len(y)} windows of {WIN_SEC}s, {len(channels)} channels, "
          f"{len(np.unique(blocks))} label blocks, 1 subject")
    print(f"Primary model (prespecified): {PRIMARY_NAME}")
    print(f"Class balance: open {int((y == 0).sum())} / closed {int((y == 1).sum())}\n")

    print("=== NAIVE RANDOM WINDOW SPLIT (leaky, for contrast) ===")
    naive = evaluate_naive_split(X, y, factory, random_state=RANDOM_STATE)
    print(f"  balanced_accuracy    {naive['balanced_accuracy']:.3f}")
    print(f"  macro_f1             {naive['macro_f1']:.3f}")
    print(f"  roc_auc              {naive['roc_auc']:.3f}")

    print("\n=== CHRONOLOGICAL 70/30 HOLDOUT ===")
    chrono = chronological_holdout(X, y, factory)
    print(f"  balanced_accuracy    {chrono['balanced_accuracy']:.3f}")
    print(f"  macro_f1             {chrono['macro_f1']:.3f}")
    print(f"  roc_auc              {chrono['roc_auc']:.3f}")
    print(f"  majority baseline    {chrono['majority_baseline_accuracy']:.3f}")

    print("\n=== LEAVE-ONE-BLOCK-OUT (same evaluator as Phase 2) ===")
    lobo = evaluate_loso(X, y, blocks, factory, random_state=RANDOM_STATE)
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        ci = lobo["subject_bootstrap_ci"][m]
        n_ok = lobo["subject_mean"][m]["n_subjects"]
        print(f"  {m:20s} {format_ci(ci)}   ({n_ok} blocks scored)")
    print("  note: each held-out block is single-class, so per-block AUC is "
          "undefined and balanced accuracy collapses to that block's recall.")

    print("\n=== SECONDARY MODELS (leave-one-block-out) ===")
    secondary = {}
    for sname, model in [
        ("SVM-RBF", lambda: SVC(kernel="rbf", probability=True,
                                random_state=RANDOM_STATE)),
        ("RandomForest(300)", lambda: RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE)),
    ]:
        f = lambda m=model: Pipeline([
            ("bandpower", EpochBandPower(SFREQ, STANDARD_BANDS, channels, n_times)),
            ("scaler", StandardScaler()),
            ("model", m()),
        ])
        res = evaluate_loso(X, y, blocks, f, random_state=RANDOM_STATE)
        secondary[sname] = {
            "subject_mean": res["subject_mean"],
            "subject_bootstrap_ci": res["subject_bootstrap_ci"],
        }
        print(f"  {sname:20s} "
              f"balAcc={format_ci(res['subject_bootstrap_ci']['balanced_accuracy'])}")

    path = save_results("phase1_results", {
        "phase": "1_eye_state",
        "dataset": "UCI EEG Eye State, single continuous recording, 1 subject",
        "primary_model": PRIMARY_NAME,
        "primary_model_prespecified": True,
        "grouping_unit": "contiguous label block (single subject, so no LOSO)",
        "cross_subject": None,
        "cross_subject_note": "Not available. The dataset contains one subject.",
        "naive_random_split": naive,
        "chronological_holdout": chrono,
        "leave_one_block_out": lobo,
        "secondary_models": secondary,
    })
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
