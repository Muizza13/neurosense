"""Phase 2: PhysioNet motor imagery (left vs right fist), 2026-09 audit rerun.

Audit changes from the previous version:

1. Cross-subject: keeps the explicit leave-one-subject-out loop with per-
   subject rows and subject-level bootstrap CIs. The pooled scores are
   retained only as a secondary descriptive appendix, and the report is
   careful to distinguish standard deviation (spread across subjects) from
   the bootstrap confidence interval (uncertainty of the mean).
2. Each held-out subject's ``y_true``, ``y_pred``, ``y_proba`` are persisted
   in the JSON, so a reader can rescore or diagnose without re-fitting.
3. Within-subject: the shuffled stratified-trial CV is retained but labelled
   accurately (trials from the same run appear in both train and test).
4. A new leave-one-run-out (LORO) within-subject evaluation is added. Each
   subject has three imagery runs (R04, R08, R12); we train on two and test
   on the third. This tests transfer across runs of the same session; it
   does not establish transfer to a new recording session.
5. The primary model is prespecified: LogisticRegression(C=1.0, balanced).
   Recorded in the JSON with ``primary_model_prespecified=True``.
6. "8 of 10 subjects above 0.5" is reported as a descriptive count only; no
   significance claim is attached.

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

PRIMARY = (
    "LogisticRegression(C=1.0, balanced)",
    lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ]),
)

SECONDARY = [
    ("LogisticRegression(C=0.5)", lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.5, max_iter=5000, random_state=RANDOM_STATE)),
    ])),
    ("RandomForest(300)", lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE)),
    ])),
]


def _within_subject_trial_cv(X, y, g, factory, n_splits=5):
    """Shuffled stratified trial CV, per subject. Accurately labelled.

    Trials from the same recording run appear in both train and test, so this
    is mildly optimistic. Kept because it is the direct comparison to a lot
    of published within-subject numbers on this dataset.
    """
    rows = []
    for s in np.unique(g):
        Xs, ys = X[g == s], y[g == s]
        cv = StratifiedKFold(n_splits, shuffle=True,
                             random_state=RANDOM_STATE)
        proba = cross_val_predict(factory(), Xs, ys, cv=cv,
                                  method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "subject_id": str(int(s)),
            "n_test": int(len(ys)),
            "balanced_accuracy": float(balanced_accuracy_score(ys, pred)),
            "macro_f1": float(f1_score(ys, pred, average="macro",
                                       zero_division=0)),
            "roc_auc": float(roc_auc_score(ys, proba)),
            "y_true": [int(v) for v in ys.tolist()],
            "y_pred": [int(v) for v in pred.tolist()],
            "y_proba": [float(v) for v in proba.tolist()],
        })
    return _summarise(rows, unit="subject", limitation=(
        "Shuffled stratified trial CV within subject. Trials from the same "
        "recording run appear in both train and test. Mildly optimistic "
        "relative to leave-one-run-out."
    ))


def _within_subject_leave_one_run_out(X, y, g, run, factory):
    """Leave-one-run-out within subject.

    For each subject with at least two runs, iterate over its runs, train on
    the others, and test on that one. One row per (subject, held-out run).
    This tests generalisation across recording runs of the same session; it
    does not establish transfer to a new recording session.
    """
    rows = []
    for s in np.unique(g):
        subj_mask = g == s
        Xs, ys, rs = X[subj_mask], y[subj_mask], run[subj_mask]
        unique_runs = np.unique(rs)
        if len(unique_runs) < 2:
            continue
        for r in unique_runs:
            tr = rs != r
            te = rs == r
            if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
                continue
            model = factory()
            model.fit(Xs[tr], ys[tr])
            proba = model.predict_proba(Xs[te])[:, 1]
            pred = (proba >= 0.5).astype(int)
            row = {
                "subject_id": str(int(s)),
                "held_out_run": int(r),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(ys[te], pred)),
                "macro_f1": float(f1_score(ys[te], pred, average="macro",
                                           zero_division=0)),
                "y_true": [int(v) for v in ys[te].tolist()],
                "y_pred": [int(v) for v in pred.tolist()],
                "y_proba": [float(v) for v in proba.tolist()],
            }
            row["roc_auc"] = (
                float(roc_auc_score(ys[te], proba))
                if len(np.unique(ys[te])) > 1 else None
            )
            rows.append(row)
    return _summarise(rows, unit="fold", limitation=(
        "Leave-one-run-out within subject. Tests transfer across recording "
        "runs of the same session; does NOT establish transfer to a new "
        "recording session."
    ))


def _summarise(rows, *, unit, limitation):
    metrics = ("balanced_accuracy", "macro_f1", "roc_auc")
    summary, cis = {}, {}
    for m in metrics:
        vals = np.array([r[m] for r in rows if r.get(m) is not None],
                        dtype=float)
        n_valid = int(len(vals))
        summary[m] = {
            "mean": float(vals.mean()) if n_valid else None,
            "std_across_" + unit + "s":
                float(vals.std(ddof=1)) if n_valid > 1 else None,
            "median": float(np.median(vals)) if n_valid else None,
            "n_" + unit + "s": n_valid,
            "n_above_half_descriptive_count":
                int((vals > 0.5).sum()) if n_valid else 0,
        }
        cis[m] = (
            bootstrap_ci(vals, random_state=RANDOM_STATE)
            if n_valid > 1 else None
        )
    return {
        "folds": rows,
        unit + "_mean": summary,
        unit + "_bootstrap_ci": cis,
        "_limitation": limitation,
        "_note_std_vs_ci": (
            "std_across_" + unit + "s is the spread of the per-" + unit +
            " metric. The bootstrap CI is the uncertainty around the mean, "
            "computed by resampling " + unit + "s (never trials)."
        ),
    }


def _print_summary(name, res, unit):
    key_mean = unit + "_mean"
    key_ci = unit + "_bootstrap_ci"
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        ci = res[key_ci].get(m)
        entry = res[key_mean][m]
        count = entry["n_above_half_descriptive_count"]
        n = entry["n_" + unit + "s"]
        if ci is None:
            print(f"  {m:20s} mean={entry['mean']}   {count}/{n} above 0.5 "
                  "(descriptive count, not a significance claim)")
        else:
            print(f"  {m:20s} mean={entry['mean']:.3f} "
                  f"CI={format_ci(ci)}   {count}/{n} above 0.5 "
                  "(descriptive count, not a significance claim)")


def main():
    d = np.load(FEATURES)
    has_run = "run" in d.files
    X = d["X"]
    y = d["y"].astype(int)
    g = d["g"]
    run = d["run"].astype(int) if has_run else None
    name, factory = PRIMARY

    print(
        f"Phase 2: {X.shape[0]} trials, {X.shape[1]} features, "
        f"{len(np.unique(g))} subjects, "
        + (f"{len(np.unique(run))} imagery runs per subject" if has_run
           else "runs UNKNOWN (rerun src/physionet_features.py)")
    )
    print(f"Primary model (prespecified): {name}\n")

    print("=== WITHIN-SUBJECT (shuffled trial CV) ===")
    within_shuffled = _within_subject_trial_cv(X, y, g, factory)
    _print_summary("within-shuffled", within_shuffled, unit="subject")

    within_loro = None
    if has_run:
        print("\n=== WITHIN-SUBJECT (leave-one-run-out) ===")
        within_loro = _within_subject_leave_one_run_out(
            X, y, g, run, factory)
        _print_summary("within-LORO", within_loro, unit="fold")
        print("  note: tests transfer across runs of the same session; "
              "does NOT establish transfer to a new recording session.")

    print("\n=== CROSS-SUBJECT (leave-one-subject-out) ===")
    cross = evaluate_loso(
        X, y, g, factory,
        random_state=RANDOM_STATE,
        sample_metadata=({"run": run} if has_run else None),
        include_baselines=True,
        persist_predictions=True,
    )
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        ci = cross["subject_bootstrap_ci"].get(m)
        entry = cross["subject_mean"][m]
        count = entry["n_above_chance"]
        n = entry["n_subjects"]
        line = (
            f"  {m:20s} mean={entry['mean']:.3f} "
            f"CI={format_ci(ci) if ci else 'n/a'}  "
            f"std_across_subjects="
            f"{entry['std']:.3f}" if entry["std"] is not None else "std=n/a"
        )
        print(line
              + f"   {count}/{n} above 0.5 (descriptive count, not "
                "statistical significance)")
    print(
        "  pooled (descriptive only):  balAcc="
        f"{cross['pooled_descriptive_metrics']['balanced_accuracy']:.3f}"
        f"  macroF1={cross['pooled_descriptive_metrics']['macro_f1']:.3f}"
        f"  AUC={cross['pooled_descriptive_metrics']['roc_auc']:.3f}"
    )
    print("  Note: std_across_subjects is the spread between subjects. The "
          "bootstrap CI is the uncertainty of the mean across subjects.")

    print("\n  per-subject cross-subject balanced accuracy:")
    for f in cross["folds"]:
        print(f"    S{f['subject_id']}: {f['balanced_accuracy']:.3f} "
              f"(n={f['n_test']})")

    print("\n=== NAIVE RANDOM TRIAL SPLIT (leaky, descriptive contrast) ===")
    naive = evaluate_naive_split(X, y, factory,
                                 random_state=RANDOM_STATE,
                                 include_baselines=True)
    print(f"  balanced_accuracy    {naive['balanced_accuracy']:.3f}")
    print(f"  macro_f1             {naive['macro_f1']:.3f}")
    print(f"  roc_auc              {naive['roc_auc']:.3f}")

    print("\n=== SECONDARY MODELS (cross-subject, not headline) ===")
    secondary = {}
    for sname, sfactory in SECONDARY:
        res = evaluate_loso(X, y, g, sfactory, random_state=RANDOM_STATE)
        secondary[sname] = {
            "subject_mean": res["subject_mean"],
            "subject_bootstrap_ci": res["subject_bootstrap_ci"],
        }
        ci = res["subject_bootstrap_ci"]["balanced_accuracy"]
        print(f"  {sname:30s} "
              f"balAcc={format_ci(ci) if ci else 'n/a'}")

    payload = {
        "phase": "2_motor_imagery",
        "dataset": (
            "PhysioNet EEG Motor Movement/Imagery, 10 subjects, "
            "left vs right fist imagery, imagery runs R04/R08/R12"
        ),
        "primary_model": name,
        "primary_model_prespecified": True,
        "within_subject_shuffled": within_shuffled,
        "within_subject_leave_one_run_out": within_loro,
        "cross_subject": cross,
        "naive_random_split": naive,
        "secondary_models": secondary,
        "audit": {
            "version": "2026-09",
            "notes": (
                "Explicit per-subject LOSO with persisted per-fold "
                "predictions, subject-level bootstrap CI, std across "
                "subjects reported separately from CI. Added leave-one-"
                "run-out within subject (does not establish transfer to a "
                "new session). '8/10 above 0.5' is a descriptive count, "
                "not statistical significance. Pooled cross-subject "
                "scores are retained only as a descriptive appendix."
            ),
            "run_ids_available": bool(has_run),
        },
    }
    path = save_results("phase2_results", payload)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
