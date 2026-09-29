"""Phase 2 extension: CSP + LDA baseline (round 4, Task 9).

The band-power + logistic-regression pipeline reported in
``src/phase2_motor_imagery.py`` is the audited primary. Common Spatial
Patterns (CSP) with Linear Discriminant Analysis is the standard
motor-imagery baseline, so it is added here as a separately labelled
extension. Results land in ``reports/results/phase2_csp_lda_results.json``;
the primary band-power JSON is not touched.

Design constraints (Task 9):

- CSP and every learned transformation are fitted inside training folds
  only. This module never fits CSP on the whole dataset.
- The same 10 subjects, the same trial eligibility (imagery runs R04,
  R08, R12), and the same evaluation splits are used, so the CSP result
  is directly comparable to the band-power result.
- Signal filtering is documented: a 4th-order Butterworth zero-phase
  bandpass 8-30 Hz is applied to each epoch before CSP, matching the
  frequency range that the mu (8-13) and beta (13-30) band-power
  features cover in Phase 2.
- Nothing here is used to replace the band-power result; both live
  side by side and are labelled accordingly in the README and the
  report.

Skip behaviour: this script needs MNE and the raw EDFs under
``data/physionet/``. If either is missing the script prints a clear
message and exits with code 0 so CI doesn't fail on a machine without the
raw data cache.

Run:  bash scripts/download_data.sh
      python -m src.phase2_csp_lda
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.core.evaluation import loro_subject_summary
from src.core.results import format_ci, save_results
from src.core.statistics import bootstrap_ci

# Constants shared with src/physionet_features.py so we evaluate on the
# same eligible trials as the primary band-power pipeline.
from src.physionet_features import (
    EPOCH_TMAX,
    EPOCH_TMIN,
    EXPECTED_FS,
    EXPECTED_SUBJECTS,
    IMAGERY_RUNS,
    MOTOR,
    _validate_layout,
    load_subject,
)

RANDOM_STATE = 42
BANDPASS_HZ = (8.0, 30.0)          # documented signal filtering
BANDPASS_ORDER = 4                  # 4th-order Butterworth (zero-phase)
CSP_COMPONENTS = 6                  # standard motor-imagery choice

RESULTS_NAME = "phase2_csp_lda_results"


def _skip(reason: str, exit_code: int = 0) -> None:
    print(f"CSP+LDA extension skipped: {reason}")
    print("This is expected on machines without the raw PhysioNet EDFs; "
          "the primary band-power results in phase2_results.json are the "
          "audited numbers.")
    sys.exit(exit_code)


def _load_all_epochs():
    """Load per-trial epochs for every expected subject, mu+beta filtered.

    Returns a dict with parallel arrays:
        epochs:   (n_trials, n_channels, n_times)
        y:        binary labels (0=left, 1=right)
        g:        subject index
        run:      PhysioNet run id (4, 8, or 12)
    """
    import mne  # noqa: F401  (import here so a missing MNE is a clean skip)
    from scipy.signal import butter, filtfilt

    _validate_layout()

    lo, hi = BANDPASS_HZ
    nyq = EXPECTED_FS / 2.0
    b, a = butter(BANDPASS_ORDER, [lo / nyq, hi / nyq], btype="bandpass")

    def bp(x):
        # zero-phase filter, per-channel across the last axis
        return filtfilt(b, a, x, axis=-1)

    all_ep, all_y, all_g, all_run = [], [], [], []
    for s_idx, subj in enumerate(EXPECTED_SUBJECTS):
        subj_dir = f"data/physionet/{subj}"
        data, yy, rr, _used = load_subject(
            subj_dir, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX
        )
        # load_subject already restricts to the MOTOR channel list, so
        # data is (n_trials, len(MOTOR), n_times) in the same order.
        if data.shape[1] != len(MOTOR):
            raise RuntimeError(
                f"{subj}: expected {len(MOTOR)} motor channels, "
                f"got {data.shape[1]}"
            )
        filt = bp(data)
        all_ep.append(filt)
        all_y.append(yy)
        all_g.append(np.full(len(yy), s_idx))
        all_run.append(rr)

        print(f"  {subj}: {len(yy)} trials, "
              f"{filt.shape[1]} motor channels, {filt.shape[2]} samples")

    return {
        "epochs": np.vstack(all_ep),
        "y": np.concatenate(all_y).astype(int),
        "g": np.concatenate(all_g).astype(int),
        "run": np.concatenate(all_run).astype(int),
    }


def _fit_predict(epochs_train, y_train, epochs_test):
    """Fit CSP+LDA on train, return (y_pred, y_proba) on test.

    CSP is fitted inside the fold; no learned parameter comes from outside
    ``epochs_train``. MNE's CSP prints per-fit MNE log lines; those are
    silenced globally in ``main()`` via ``mne.set_log_level("ERROR")`` so
    the console shows only the actual results.
    """
    from mne.decoding import CSP
    csp = CSP(n_components=CSP_COMPONENTS, reg=None, log=True,
              norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    Xtr = csp.fit_transform(epochs_train, y_train)
    Xte = csp.transform(epochs_test)
    lda.fit(Xtr, y_train)
    pred = lda.predict(Xte)
    proba = lda.predict_proba(Xte)[:, 1]
    return pred, proba


def _fold_metrics(y_true, pred, proba):
    ba = float(balanced_accuracy_score(y_true, pred))
    mf1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
    auc = (float(roc_auc_score(y_true, proba))
           if len(np.unique(y_true)) > 1 else None)
    return {
        "n_test": int(len(y_true)),
        "balanced_accuracy": ba,
        "macro_f1": mf1,
        "roc_auc": auc,
    }


def _summary(rows, unit):
    metrics = ("balanced_accuracy", "macro_f1", "roc_auc")
    mean, cis = {}, {}
    for m in metrics:
        vals = np.array([r[m] for r in rows if r.get(m) is not None],
                        dtype=float)
        n_valid = int(len(vals))
        mean[m] = {
            "mean": float(vals.mean()) if n_valid else None,
            "std_across_" + unit + "s":
                float(vals.std(ddof=1)) if n_valid > 1 else None,
            "median": float(np.median(vals)) if n_valid else None,
            "n_" + unit + "s": n_valid,
            "n_above_half_descriptive_count":
                int((vals > 0.5).sum()) if n_valid else 0,
        }
        cis[m] = (bootstrap_ci(vals, random_state=RANDOM_STATE)
                  if n_valid > 1 else None)
    return {
        "unit": unit,
        "folds": rows,
        unit + "_mean": mean,
        unit + "_bootstrap_ci": cis,
    }


def within_shuffled(epochs, y, g, n_splits=5):
    """Per-subject stratified K-fold CV, CSP+LDA inside each training fold."""
    rows = []
    for s in np.unique(g):
        mask = g == s
        E, ys = epochs[mask], y[mask]
        cv = StratifiedKFold(n_splits, shuffle=True,
                             random_state=RANDOM_STATE)
        y_true_all, y_pred_all, y_proba_all = [], [], []
        for tr, te in cv.split(np.zeros(len(ys)), ys):
            pred, proba = _fit_predict(E[tr], ys[tr], E[te])
            y_true_all.extend(ys[te].tolist())
            y_pred_all.extend(pred.tolist())
            y_proba_all.extend(proba.tolist())
        row = _fold_metrics(
            np.asarray(y_true_all),
            np.asarray(y_pred_all),
            np.asarray(y_proba_all),
        )
        row["subject_id"] = str(int(s))
        rows.append(row)
    return _summary(rows, unit="subject")


def within_loro(epochs, y, g, run):
    """Leave-one-run-out within subject, one row per (subject, held-out run)."""
    rows = []
    for s in np.unique(g):
        subj_mask = g == s
        E, ys, rs = epochs[subj_mask], y[subj_mask], run[subj_mask]
        unique_runs = np.unique(rs)
        if len(unique_runs) < 2:
            continue
        for r in unique_runs:
            tr = rs != r
            te = rs == r
            if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
                continue
            pred, proba = _fit_predict(E[tr], ys[tr], E[te])
            row = _fold_metrics(ys[te], pred, proba)
            row["subject_id"] = str(int(s))
            row["held_out_run"] = int(r)
            rows.append(row)
    return loro_subject_summary(rows, random_state=RANDOM_STATE, limitation=(
        "Leave-one-run-out within subject. The interval resamples subject "
        "means of the run-level folds. It tests transfer across recording "
        "runs of the same session; it does NOT establish transfer to a new "
        "recording session."
    ))


def cross_loso(epochs, y, g):
    """Leave-one-subject-out, one row per held-out subject."""
    rows = []
    for s in np.unique(g):
        tr = g != s
        te = g == s
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        pred, proba = _fit_predict(epochs[tr], y[tr], epochs[te])
        row = _fold_metrics(y[te], pred, proba)
        row["subject_id"] = str(int(s))
        rows.append(row)
    return _summary(rows, unit="subject")


def _print_summary(name, block):
    unit = block["unit"]
    for m in ("balanced_accuracy", "macro_f1", "roc_auc"):
        ci = block[unit + "_bootstrap_ci"].get(m)
        mean = block[unit + "_mean"][m]
        n = mean["n_" + unit + "s"]
        above = mean["n_above_half_descriptive_count"]
        print(f"  {m:20s} mean={mean['mean']} CI="
              f"{format_ci(ci) if ci else 'n/a'} "
              f"{above}/{n} above 0.5")


def main():
    import os
    for d in ("reports/figures", "reports/results", "models"):
        os.makedirs(d, exist_ok=True)

    try:
        import mne
    except ImportError:
        _skip("MNE is not installed. Add mne==1.8.0 (already in "
              "requirements.txt) and rerun to include the CSP extension.")

    if not Path("data/physionet").exists():
        _skip("no data/physionet directory. Run "
              "bash scripts/download_data.sh first, then rerun.")

    mne.set_log_level("ERROR")

    print("CSP + LDA extension (round 4, Task 9)")
    print(f"  bandpass:      {BANDPASS_HZ[0]}-{BANDPASS_HZ[1]} Hz "
          f"(Butterworth order {BANDPASS_ORDER}, zero-phase)")
    print(f"  CSP components: {CSP_COMPONENTS}")
    print(f"  channels:       {len(MOTOR)} motor electrodes")
    print(f"  epoch window:   {EPOCH_TMIN}-{EPOCH_TMAX} s\n")

    print("Loading and filtering epochs")
    bundle = _load_all_epochs()
    E, y, g, run = bundle["epochs"], bundle["y"], bundle["g"], bundle["run"]
    print(f"\nTotal: {E.shape[0]} trials, "
          f"{len(np.unique(g))} subjects, runs={sorted(set(run.tolist()))}\n")

    print("=== WITHIN-SUBJECT (shuffled trial CV, CSP+LDA in-fold) ===")
    ws = within_shuffled(E, y, g)
    _print_summary("within-shuffled", ws)

    print("\n=== WITHIN-SUBJECT (leave-one-run-out, CSP+LDA in-fold) ===")
    wl = within_loro(E, y, g, run)
    _print_summary("within-LORO", wl)

    print("\n=== CROSS-SUBJECT (leave-one-subject-out, CSP+LDA in-fold) ===")
    cs = cross_loso(E, y, g)
    _print_summary("cross-LOSO", cs)

    payload = {
        "phase": "2_motor_imagery_csp_lda_extension",
        "dataset": (
            "PhysioNet EEG Motor Movement/Imagery, 10 subjects, "
            "imagery runs R04/R08/R12, left vs right fist"
        ),
        "primary_model": (
            f"CSP({CSP_COMPONENTS} components, log-variance) + LDA"
        ),
        "extension_of": "phase2_results.json",
        "note": (
            "Separately labelled extension of Phase 2. The audited primary "
            "remains the band-power LogReg pipeline; CSP+LDA is reported "
            "here to give the standard motor-imagery baseline on the same "
            "eligible trials and evaluation splits. Not a replacement."
        ),
        "signal_filtering": {
            "bandpass_hz": list(BANDPASS_HZ),
            "filter_type": (
                f"Butterworth order {BANDPASS_ORDER}, zero-phase filtfilt"
            ),
            "channel_set": list(MOTOR),
            "epoch_window_s": [EPOCH_TMIN, EPOCH_TMAX],
        },
        "in_fold_fit": (
            "CSP and LDA are fitted on training epochs only, inside every "
            "evaluation fold. The extension never fits CSP on the whole "
            "dataset."
        ),
        "within_subject_shuffled": ws,
        "within_subject_leave_one_run_out": wl,
        "cross_subject": cs,
    }
    path = save_results(RESULTS_NAME, payload)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
