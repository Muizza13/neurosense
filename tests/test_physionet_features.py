"""Unit tests for the physionet_features extractor.

The full end-to-end path needs the raw EDFs on disk and MNE installed, so
this module tests only what is testable without those inputs:

- pure helpers (SHA-256, run-id parsing, expected-file layout)
- the manifest schema (all required keys, sorted keys, deterministic on
  identical inputs)
- structural invariants of the committed feature cache

These tests run in a clean environment without downloading anything.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from src import physionet_features as pf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_run_id_from_filename_parses_all_expected_runs():
    for r in pf.IMAGERY_RUNS:
        path = f"data/physionet/S001/S001R{r:02d}.edf"
        assert pf._run_id_from_filename(path) == r


def test_sha256_matches_hashlib_reference(tmp_path):
    payload = b"neurosense-audit-round-3\n" * 1024
    f = tmp_path / "x.bin"
    f.write_bytes(payload)
    assert pf._sha256(str(f)) == hashlib.sha256(payload).hexdigest()


def test_expected_edf_uses_two_digit_run_zero_padding():
    assert pf._expected_edf("S002", 4).endswith("/S002/S002R04.edf")
    assert pf._expected_edf("S010", 12).endswith("/S010/S010R12.edf")


def test_feature_names_covers_every_channel_and_band_in_declared_order():
    names = pf.feature_names()
    assert len(names) == len(pf.MOTOR) * len(pf.BANDS)
    # first block should be the first channel's bands, in dict order
    assert names[0] == f"{pf.MOTOR[0]}_{list(pf.BANDS)[0]}"
    assert names[-1] == f"{pf.MOTOR[-1]}_{list(pf.BANDS)[-1]}"


# ---------------------------------------------------------------------------
# Layout validation
# ---------------------------------------------------------------------------

def test_validate_layout_lists_missing_files(tmp_path, monkeypatch):
    """No EDFs on disk => the extractor refuses to run, and the error names
    every missing file so the reader knows what to download."""
    monkeypatch.setattr(pf, "DATA_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        pf._validate_layout()
    msg = str(excinfo.value)
    # every expected subject x run pair shows up
    assert msg.count(".edf") == len(pf.EXPECTED_SUBJECTS) * len(pf.IMAGERY_RUNS)
    assert "Missing EDFs" in msg
    assert "refuses to skip" in msg


def test_validate_layout_reports_truncated_files(tmp_path, monkeypatch):
    """A file smaller than MIN_EDF_BYTES must trip the check, not be
    silently accepted."""
    monkeypatch.setattr(pf, "DATA_ROOT", str(tmp_path))
    # write one full-sized dummy for each expected file
    for subj in pf.EXPECTED_SUBJECTS:
        (tmp_path / subj).mkdir()
        for run in pf.IMAGERY_RUNS:
            f = tmp_path / subj / f"{subj}R{run:02d}.edf"
            f.write_bytes(b"\0" * pf.MIN_EDF_BYTES)
    # now truncate one
    victim = tmp_path / pf.EXPECTED_SUBJECTS[0] / f"{pf.EXPECTED_SUBJECTS[0]}R04.edf"
    victim.write_bytes(b"\0" * 100)
    with pytest.raises(RuntimeError) as excinfo:
        pf._validate_layout()
    assert "Truncated EDFs" in str(excinfo.value)
    assert str(victim) in str(excinfo.value)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_write_manifest_records_every_required_field(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "MANIFEST_PATH", str(tmp_path / "manifest.json"))

    X = np.zeros((3, len(pf.feature_names())))
    y = np.array([0, 1, 0])
    g = np.array([0, 0, 1])
    run = np.array([4, 8, 12])
    trials = np.array(["S001_R04_T000", "S001_R08_T000", "S002_R12_T000"])
    per_subject_meta = [
        {"subject_id": "S001", "n_trials": 2, "n_left": 1, "n_right": 1,
         "run_trial_counts": {4: 1, 8: 1}, "source_files": []},
        {"subject_id": "S002", "n_trials": 1, "n_left": 1, "n_right": 0,
         "run_trial_counts": {12: 1}, "source_files": []},
    ]

    pf._write_manifest(X, y, g, run, trials, per_subject_meta)
    manifest = json.loads(Path(pf.MANIFEST_PATH).read_text())

    required = {
        "cache_path", "n_trials", "n_features", "n_subjects",
        "sampling_rate_hz", "epoch_window_seconds", "bands_hz",
        "channel_order", "feature_names", "imagery_runs",
        "preprocessing", "per_subject", "npz_keys",
    }
    assert required.issubset(manifest.keys())
    assert manifest["sampling_rate_hz"] == pf.EXPECTED_FS
    assert manifest["imagery_runs"] == list(pf.IMAGERY_RUNS)
    assert manifest["channel_order"] == list(pf.MOTOR)
    assert manifest["preprocessing"]["welch_fs"] == pf.EXPECTED_FS


def test_write_manifest_is_deterministic(tmp_path, monkeypatch):
    """Same inputs => bit-identical JSON. This is what makes the manifest
    useful as an integrity check between the cached and raw paths."""
    monkeypatch.setattr(pf, "MANIFEST_PATH", str(tmp_path / "a.json"))
    args = (
        np.zeros((1, len(pf.feature_names()))),
        np.array([0]), np.array([0]), np.array([4]), np.array(["S001_R04_T000"]),
        [{"subject_id": "S001", "n_trials": 1, "n_left": 1, "n_right": 0,
          "run_trial_counts": {4: 1}, "source_files": []}],
    )
    pf._write_manifest(*args)
    a = Path(pf.MANIFEST_PATH).read_bytes()

    monkeypatch.setattr(pf, "MANIFEST_PATH", str(tmp_path / "b.json"))
    pf._write_manifest(*args)
    b = Path(pf.MANIFEST_PATH).read_bytes()

    assert a == b


# ---------------------------------------------------------------------------
# Committed cache shape
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/processed/physionet_features.npz").exists(),
    reason="feature cache not present in this checkout",
)
def test_committed_cache_has_expected_keys_and_shapes():
    d = np.load("data/processed/physionet_features.npz", allow_pickle=True)
    assert set(d.files) >= {"X", "y", "g", "run"}
    n = d["X"].shape[0]
    assert d["y"].shape == (n,)
    assert d["g"].shape == (n,)
    assert d["run"].shape == (n,)
    # only the three allow-listed imagery runs should appear
    assert set(int(r) for r in np.unique(d["run"])).issubset(set(pf.IMAGERY_RUNS))
    if "trial_id" in d.files:
        ids = d["trial_id"]
        assert ids.shape == (n,)
        # per-trial ids should be unique
        assert len(set(map(str, ids.tolist()))) == n
