"""Deterministic result serialisation.

Every phase writes one JSON file under reports/results/. Files are sorted and
indented so that a rerun producing identical numbers produces a byte-identical
file, which makes an unintended change visible in a git diff.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("reports/results")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _clean(obj):
    """Convert numpy scalars and arrays into plain JSON types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_results(name, payload, results_dir=RESULTS_DIR, provenance=True):
    """Write reports/results/<name>.json and return the path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = _clean(payload)
    if provenance:
        payload["_provenance"] = {
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
    path = results_dir / f"{name}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_results(name, results_dir=RESULTS_DIR):
    with open(Path(results_dir) / f"{name}.json") as fh:
        return json.load(fh)


def format_ci(ci, digits=3):
    """Render a bootstrap interval for tables and README text."""
    if not ci:
        return "n/a"
    return f"{ci['point']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"
