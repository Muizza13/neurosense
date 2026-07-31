"""Phase 2: PhysioNet motor imagery (left vs right fist), corrected reporting.

Changes from the original evaluate_physionet.py:

1. Cross-subject evaluation is an explicit leave-one-subject-out loop that
   stores one metric row per held-out subject. The previous version pooled all
   437 held-out predictions with cross_val_predict and scored them once, which
   makes the trial the unit of inference and hides the between-subject spread.

2. The headline model is prespecified: logistic regression, C=1.0,
   class_weight="balanced". The original C=0.5 was not documented as having
   been chosen before results were seen, so it is reported only in the
   secondary comparison.

3. All headline metrics come from one named model. The previous README paired
   RandomForest balanced accuracy with LogisticRegression AUC.

4. Confidence intervals resample subjects, not trials.

Run:  python src/phase2_motor_imagery.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.evaluation import evaluate_loso, evaluate_naive_split
from src.core.results import format_ci, save_results
from src.core.statistics import bootstrap_ci

FEATURES = "data/processed/physionet_features.npz"
RANDOM_STATE = 42

# Prespecified primary model. Fixed before this rerun, not selected on results.
PRIMARY = ("LogisticRegression(C=1.0, balanced)", lambda: Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=5000, random_state=RANDOM_STATE)),
]))

SECONDARY = [
    ("LogisticRegression(C=0.5)", lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=5000,
                                     random_state=RANDOM_STATE)),
    ])),
    ("RandomForest(300)", lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300,
                                         random_state=RANDOM_STATE)),
    ])),
]


def within_subject(X, y, g, factory, n_splits=5):
    """Per-subject trial CV. One metric row per subject, same as cross-subject.

    Limitation: the committed feature file does not record which run each trial
    came from, so folds are stratified over trials rather than left-one-run-out.
    Trials within a run are temporally adjacent, so this is mildly optimistic.
    """
    rows = []
    for s in np.unique(g):
        Xs, ys = X[g == s], y[g == s]
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
        proba = cross_val_predict(factory(), Xs, ys, cv=cv,
                                  method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "subject_id": str(s),
            "n_test": int(len(ys)),
            "balanced_accuracy": float(balanced_accuracy_score(ys, pred)),
            "macro_f1": float(f1_score(ys, pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(ys, proba)),
        })
    summary, cis = {}, {}
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        vals = np.array([r[m] for r in rows])
        summary[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "n_subjects": int(len(vals)),
            "n_above_chance": int((vals > 0.5).sum()),
        }
        cis[m] = bootstrap_ci(vals, random_state=RANDOM_STATE)
    return {
        "folds": rows,
        "subject_mean": summary,
        "subject_bootstrap_ci": cis,
        "_limitation": (
            "Stratified trial folds, not leave-one-run-out. Run identity is "
            "absent from the committed feature file."
        ),
    }


def main():
    d = np.load(FEATURES)
    X, y, g = d["X"], d["y"].astype(int), d["g"]
    name, factory = PRIMARY

    print(f"Phase 2: {X.shape[0]} trials, {X.shape[1]} features, "
          f"{len(np.unique(g))} subjects")
    print(f"Primary model (prespecified): {name}\n")

    print("=== WITHIN-SUBJECT (per-subject trial CV) ===")
    within = within_subject(X, y, g, factory)
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        print(f"  {m:20s} {format_ci(within['subject_bootstrap_ci'][m])}"
              f"   {within['subject_mean'][m]['n_above_chance']}/10 above chance")

    print("\n=== CROSS-SUBJECT (leave-one-subject-out) ===")
    cross = evaluate_loso(X, y, g, factory, random_state=RANDOM_STATE)
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        print(f"  {m:20s} {format_ci(cross['subject_bootstrap_ci'][m])}"
              f"   {cross['subject_mean'][m]['n_above_chance']}/10 above chance")
    print(f"  pooled (descriptive)  balAcc="
          f"{cross['pooled_descriptive_metrics']['balanced_accuracy']:.3f}"
          f"  AUC={cross['pooled_descriptive_metrics']['roc_auc']:.3f}")

    print("\n  per-subject cross-subject balanced accuracy:")
    for f in cross["folds"]:
        print(f"    S{f['subject_id']}: {f['balanced_accuracy']:.3f} "
              f"(n={f['n_test']})")

    print("\n=== NAIVE RANDOM TRIAL SPLIT (leaky, for contrast) ===")
    naive = evaluate_naive_split(X, y, factory, random_state=RANDOM_STATE)
    print(f"  balanced_accuracy    {naive['balanced_accuracy']:.3f}")
    print(f"  roc_auc              {naive['roc_auc']:.3f}")

    print("\n=== SECONDARY MODELS (cross-subject, not headline) ===")
    secondary = {}
    for sname, sfactory in SECONDARY:
        res = evaluate_loso(X, y, g, sfactory, random_state=RANDOM_STATE)
        secondary[sname] = {
            "subject_mean": res["subject_mean"],
            "subject_bootstrap_ci": res["subject_bootstrap_ci"],
        }
        print(f"  {sname:30s} "
              f"balAcc={format_ci(res['subject_bootstrap_ci']['balanced_accuracy'])}")

    path = save_results("phase2_results", {
        "phase": "2_motor_imagery",
        "dataset": "PhysioNet EEG Motor Movement/Imagery, 10 subjects, "
                   "left vs right fist imagery",
        "primary_model": name,
        "primary_model_prespecified": True,
        "within_subject": within,
        "cross_subject": cross,
        "naive_random_split": naive,
        "secondary_models": secondary,
    })
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
