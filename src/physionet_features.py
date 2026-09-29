"""Extract PhysioNet motor-imagery features from raw EDF files.

Round 2 of the 2026-09 audit (Tasks 5 and 7) rebuilt this step so a reader
can reproduce Phase 2 from raw data without silent skips.

What this script does:

1. Validates that every expected subject (S001..S010) and every expected
   imagery run (R04, R08, R12) is present on disk. Missing or truncated
   files raise a hard error rather than getting silently skipped.
2. Only R04/R08/R12 are ever loaded, even if extra files exist in the
   subject directory. Other runs (rest, real movement) belong to
   different task conditions and are not part of Phase 2.
3. The sampling rate on every raw file is read and asserted equal to the
   expected 160 Hz. The rate is not assumed from the constant alone.
4. A manifest is written alongside the feature cache
   (``data/processed/physionet_features_manifest.json``) recording every
   subject id, run id, per-run trial count, channel order, frequency
   bands, epoch timing, preprocessing settings, and the SHA-256 of every
   source EDF. A rerun on unchanged EDFs produces a byte-identical
   manifest.

Cached-vs-raw reproduction paths:

- **Cached (default)**: the two phase scripts read
  ``data/processed/physionet_features.npz`` directly, so no EDF is needed
  after the first extraction. This is the fast path.
- **From raw**: run ``bash scripts/download_data.sh && python -m
  src.physionet_features``. The manifest lets you verify byte-for-byte
  that the cache came from the expected raw files.

Usage:
    bash scripts/download_data.sh   # fetch the EDFs (~60 MB total)
    python -m src.physionet_features
"""
from __future__ import annotations

import glob
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

# ``mne`` is only needed by ``load_subject`` (Path B: rebuild the feature
# cache from raw EDFs). It is deliberately imported inside that function
# so this module can be imported by tests and by the cached-features
# reproduction path (Path A) without MNE installed. Round 4 CI does not
# install MNE.

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Expected structure. Change these constants only if you are extending the
# study; downstream code assumes them.
# ---------------------------------------------------------------------------
EXPECTED_FS = 160.0
EXPECTED_SUBJECTS = tuple(f"S{i:03d}" for i in range(1, 11))
IMAGERY_RUNS = (4, 8, 12)                       # explicitly allow-listed
BANDS = {"mu": (8, 13), "beta": (13, 30)}
MOTOR = [
    "FC3", "FCZ", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CPZ", "CP4",
]
EPOCH_TMIN = 0.5
EPOCH_TMAX = 3.5
DATA_ROOT = "data/physionet"
CACHE_PATH = "data/processed/physionet_features.npz"
MANIFEST_PATH = "data/processed/physionet_features_manifest.json"
MIN_EDF_BYTES = 500_000                          # truncated files are smaller

_integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _clean(name):
    return name.replace(".", "").upper()


def _run_id_from_filename(path):
    """Parse the ``R\\d\\d`` field from a PhysioNet EDF filename."""
    stem = Path(path).name
    r_idx = stem.rfind("R")
    return int(stem[r_idx + 1:r_idx + 3])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_edf(subj, run):
    return f"{DATA_ROOT}/{subj}/{subj}R{run:02d}.edf"


def _validate_layout():
    """Refuse to proceed unless every expected EDF is present and non-empty."""
    missing = []
    truncated = []
    for subj in EXPECTED_SUBJECTS:
        for run in IMAGERY_RUNS:
            path = _expected_edf(subj, run)
            if not Path(path).exists():
                missing.append(path)
                continue
            size = Path(path).stat().st_size
            if size < MIN_EDF_BYTES:
                truncated.append((path, size))
    if missing or truncated:
        parts = []
        if missing:
            parts.append("Missing EDFs:\n  " + "\n  ".join(missing))
        if truncated:
            parts.append(
                "Truncated EDFs (< " f"{MIN_EDF_BYTES // 1000} KB):\n  "
                + "\n  ".join(f"{p} ({s} bytes)" for p, s in truncated)
            )
        parts.append(
            "Rerun scripts/download_data.sh. This script refuses to skip "
            "expected files silently."
        )
        raise RuntimeError("\n\n".join(parts))


def load_subject(subj_dir, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX):
    """Concatenate the imagery runs of a subject and epoch the joined recording.

    Returns:
        data: (n_epochs, n_channels, n_times)
        y:    (n_epochs,) 0=left, 1=right
        run:  (n_epochs,) int in IMAGERY_RUNS
        source_paths: list[str] of EDFs used for this subject

    Note: ``mne`` is imported inside this function on purpose. The module
    is importable and testable without MNE installed; only the raw-EDF
    path (Path B) needs it.
    """
    import mne  # noqa: PLC0415 - required only on Path B
    raw_by_run = []
    used_paths = []
    for path in sorted(glob.glob(f"{subj_dir}/*R*.edf")):
        rid = _run_id_from_filename(path)
        if rid not in IMAGERY_RUNS:
            continue
        r = mne.io.read_raw_edf(path, preload=True, verbose=False)
        actual_fs = float(r.info["sfreq"])
        if not np.isclose(actual_fs, EXPECTED_FS):
            raise RuntimeError(
                f"{path}: sampling rate {actual_fs} Hz, expected "
                f"{EXPECTED_FS} Hz. Refusing to run with mismatched sfreq."
            )
        r.rename_channels({c: _clean(c) for c in r.ch_names})
        raw_by_run.append((rid, r))
        used_paths.append(path)
    if not raw_by_run:
        raise RuntimeError(f"No imagery runs found under {subj_dir}")

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
        raise RuntimeError(
            f"{subj_dir}: fewer than 2 annotation classes present. "
            f"Cannot form a left-vs-right dataset from this subject."
        )
    ep = mne.Epochs(
        raw, events, event_id=want, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False,
        reject_by_annotation=False,   # keep boundary-adjacent epochs
    )
    y = (ep.events[:, 2] == want["T2"]).astype(int)   # 1 = right fist

    event_samples = ep.events[:, 0]
    run_ids = np.empty(len(y), dtype=int)
    for i, s in enumerate(event_samples):
        for rid, lo, hi in run_ranges:
            if lo <= s < hi:
                run_ids[i] = rid
                break
        else:
            run_ids[i] = -1
    return ep.get_data(), y, run_ids, used_paths


def epoch_features(data):
    feats = []
    for ep in data:
        row = []
        for ch in range(ep.shape[0]):
            fr, psd = welch(ep[ch], fs=EXPECTED_FS,
                            nperseg=min(int(EXPECTED_FS), ep.shape[1]))
            for lo, hi in BANDS.values():
                m = (fr >= lo) & (fr < hi)
                row.append(_integrate(psd[m], fr[m]))
        feats.append(row)
    return np.array(feats)


def feature_names():
    return [f"{ch}_{b}" for ch in MOTOR for b in BANDS]


def load_all(root=DATA_ROOT):
    """Load every expected subject. Raises on missing or truncated files."""
    _validate_layout()

    X, y, g, run, trials = [], [], [], [], []
    per_subject_meta = []
    for subj_idx, subj in enumerate(EXPECTED_SUBJECTS):
        subj_dir = f"{root}/{subj}"
        data, yy, rr, used = load_subject(subj_dir)
        feats = epoch_features(data)

        # Stable per-trial identifiers: subject index, run id, trial index
        # within that (subject, run). Deterministic on the same inputs.
        trial_ids = []
        for rid in sorted(set(rr.tolist())):
            idxs = np.where(rr == rid)[0]
            for k, _ in enumerate(idxs):
                trial_ids.append(f"{subj}_R{int(rid):02d}_T{k:03d}")
        # Sort trial_ids so they align with concatenation order of feats
        trial_ids = np.empty(len(yy), dtype=object)
        counters = {rid: 0 for rid in IMAGERY_RUNS}
        for i, rid in enumerate(rr.tolist()):
            trial_ids[i] = f"{subj}_R{int(rid):02d}_T{counters[int(rid)]:03d}"
            counters[int(rid)] += 1

        X.append(feats)
        y.append(yy)
        g.append(np.full(len(yy), subj_idx))
        run.append(rr)
        trials.append(trial_ids)

        run_counts = {int(rid): int((rr == rid).sum())
                      for rid in sorted(set(rr.tolist()))}
        per_subject_meta.append({
            "subject_id": subj,
            "subject_index": subj_idx,
            "n_trials": int(len(yy)),
            "n_left": int((yy == 0).sum()),
            "n_right": int((yy == 1).sum()),
            "run_trial_counts": run_counts,
            "source_files": [
                {"path": p, "sha256": _sha256(p), "bytes": Path(p).stat().st_size}
                for p in used
            ],
        })
        print(
            f"  {subj}: {len(yy)} trials "
            f"(left {int((yy == 0).sum())}/right {int((yy == 1).sum())}) "
            f"[{', '.join(f'R{k:02d}={v}' for k, v in run_counts.items())}]"
        )
    return (
        np.vstack(X),
        np.concatenate(y),
        np.concatenate(g),
        np.concatenate(run),
        np.concatenate(trials),
        per_subject_meta,
    )


def _write_manifest(X, y, g, run, trials, per_subject_meta):
    manifest = {
        "cache_path": CACHE_PATH,
        "n_trials": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_subjects": int(len(np.unique(g))),
        "sampling_rate_hz": EXPECTED_FS,
        "epoch_window_seconds": {"tmin": EPOCH_TMIN, "tmax": EPOCH_TMAX},
        "bands_hz": {k: list(v) for k, v in BANDS.items()},
        "channel_order": list(MOTOR),
        "feature_names": feature_names(),
        "imagery_runs": list(IMAGERY_RUNS),
        "preprocessing": {
            "welch_fs": EXPECTED_FS,
            "welch_nperseg": int(EXPECTED_FS),
            "integration_rule": "trapezoid over PSD",
            "channel_reference": "raw EDF, no re-reference applied",
            "boundary_handling": "reject_by_annotation=False in mne.Epochs",
        },
        "per_subject": per_subject_meta,
        "npz_keys": ["X", "y", "g", "run", "trial_id"],
    }
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X, y, g, run, trial_ids, per_subject_meta = load_all()
    np.savez_compressed(
        CACHE_PATH, X=X, y=y, g=g, run=run, trial_id=trial_ids.astype(str)
    )
    _write_manifest(X, y, g, run, trial_ids, per_subject_meta)
    print(
        f"\nsaved {CACHE_PATH}: X={X.shape}, {len(np.unique(g))} subjects, "
        f"runs={sorted(set(run.tolist()))}"
    )
    print(f"saved {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
