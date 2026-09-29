"""Extract PhysioNet motor-imagery features and cache subject + run IDs.

Task 5 (2026-09 audit): the previous cache only stored ``X, y, g`` (features,
labels, subject index). Without a run identifier we cannot test transfer
across recording runs. This script re-extracts features and also stores
``run``, a per-trial int in {4, 8, 12} (the imagery runs R04, R08, R12), so
downstream code can do leave-one-run-out within subject.

Signal path is unchanged: mu (8-13 Hz) and beta (13-30 Hz) band power via
Welch on the sensorimotor strip, 0.5-3.5 s window relative to the cue. To
match the previously cached epoch counts (437 trials across 10 subjects)
we concatenate a subject's runs and epoch the concatenated recording, then
tag each event with its run by looking up which run's sample range the
event fell into.

Usage:
    bash scripts/download_data.sh   # to fetch the EDFs
    python -m src.physionet_features
"""
from __future__ import annotations

import glob
import warnings

import mne
import numpy as np
from scipy.signal import welch

warnings.filterwarnings("ignore")

FS = 160
BANDS = {"mu": (8, 13), "beta": (13, 30)}
MOTOR = [
    "FC3", "FCZ", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CPZ", "CP4",
]
IMAGERY_RUNS = (4, 8, 12)
CACHE_PATH = "data/processed/physionet_features.npz"
_integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _clean(name):
    return name.replace(".", "").upper()


def _run_id_from_filename(path):
    """Parse the ``R\\d\\d`` field from a PhysioNet EDF filename."""
    stem = path.rsplit("/", 1)[-1]
    r_idx = stem.rfind("R")
    return int(stem[r_idx + 1:r_idx + 3])


def load_subject(subj_dir, tmin=0.5, tmax=3.5):
    """Concatenate the imagery runs and epoch the joined recording.

    Returns:
        data: (n_epochs, n_channels, n_times)
        y:    (n_epochs,) 0=left, 1=right
        run:  (n_epochs,) int in {4, 8, 12}
    """
    raw_by_run = []
    for path in sorted(glob.glob(f"{subj_dir}/*R*.edf")):
        rid = _run_id_from_filename(path)
        if rid not in IMAGERY_RUNS:
            continue
        r = mne.io.read_raw_edf(path, preload=True, verbose=False)
        r.rename_channels({c: _clean(c) for c in r.ch_names})
        raw_by_run.append((rid, r))
    if not raw_by_run:
        return None, None, None

    # Sample-index ranges of each run inside the concatenated raw. Because
    # ``mne.concatenate_raws`` splices the runs in the order it is given,
    # a cumulative-length lookup tells us which run any event index falls
    # into. Store (run_id, [start_sample_exclusive, end_sample_exclusive)).
    cum = 0
    run_ranges = []
    for rid, r in raw_by_run:
        n = r.n_times
        run_ranges.append((rid, cum, cum + n))
        cum += n

    raw = mne.concatenate_raws([r for _, r in raw_by_run], verbose=False)
    raw.pick(MOTOR)
    events, eid = mne.events_from_annotations(raw, verbose=False)
    want = {k: v for k, v in eid.items() if k in ("T1", "T2")}
    if len(want) < 2:
        return None, None, None
    # ``reject_by_annotation=False`` because ``mne.concatenate_raws`` adds
    # ``BAD boundary`` annotations at each splice; the original loader used
    # an older MNE that did not reject on those, so we match the original
    # trial counts by keeping the boundary-adjacent epochs.
    ep = mne.Epochs(
        raw, events, event_id=want, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False,
        reject_by_annotation=False,
    )
    y = (ep.events[:, 2] == want["T2"]).astype(int)  # 1 = right fist

    # Look up each epoch's run by its event sample index.
    event_samples = ep.events[:, 0]
    run_ids = np.empty(len(y), dtype=int)
    for i, s in enumerate(event_samples):
        for rid, lo, hi in run_ranges:
            if lo <= s < hi:
                run_ids[i] = rid
                break
        else:
            run_ids[i] = -1
    return ep.get_data(), y, run_ids


def epoch_features(data):
    feats = []
    for ep in data:
        row = []
        for ch in range(ep.shape[0]):
            fr, psd = welch(ep[ch], fs=FS, nperseg=min(FS, ep.shape[1]))
            for lo, hi in BANDS.values():
                m = (fr >= lo) & (fr < hi)
                row.append(_integrate(psd[m], fr[m]))
        feats.append(row)
    return np.array(feats)


def feature_names():
    return [f"{ch}_{b}" for ch in MOTOR for b in BANDS]


def load_all(root="data/physionet"):
    X, y, g, run = [], [], [], []
    subj_dirs = sorted(glob.glob(f"{root}/S0*"))
    for subj_idx, subj_dir in enumerate(subj_dirs):
        data, yy, rr = load_subject(subj_dir)
        if data is None:
            print(f"  {subj_dir.split('/')[-1]}: skipped (no imagery runs)")
            continue
        feats = epoch_features(data)
        X.append(feats)
        y.append(yy)
        g.append(np.full(len(yy), subj_idx))
        run.append(rr)
        run_counts = ", ".join(
            f"R{int(rid):02d}={int((rr == rid).sum())}"
            for rid in sorted(set(rr.tolist()))
        )
        print(
            f"  {subj_dir.split('/')[-1]}: {len(yy)} trials "
            f"(left {int((yy == 0).sum())}/right {int((yy == 1).sum())}) "
            f"[{run_counts}]"
        )
    return (
        np.vstack(X),
        np.concatenate(y),
        np.concatenate(g),
        np.concatenate(run),
    )


def main():
    X, y, g, run = load_all()
    np.savez_compressed(CACHE_PATH, X=X, y=y, g=g, run=run)
    print(
        f"\nsaved {CACHE_PATH}: X={X.shape}, "
        f"{len(np.unique(g))} subjects, runs={sorted(set(run.tolist()))}"
    )


if __name__ == "__main__":
    main()
