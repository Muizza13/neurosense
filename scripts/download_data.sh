#!/usr/bin/env bash
# Downloads the datasets used in this project. Data is NOT stored in the repo.
#
# Two datasets:
#   1. UCI EEG Eye State (Phase 1).
#   2. PhysioNet EEG Motor Movement/Imagery, imagery runs R04/R08/R12 only,
#      subjects S001..S010.
#
# Each file is written to a temporary path and moved into place only after
# the download succeeds. An EDF is accepted only when its byte length matches
# the length declared in its header. A failed file is retried, then the
# script exits. A partial file is never left at the destination.

set -euo pipefail

cd "$(dirname "$0")/.."

MAX_ATTEMPTS=3
PY=""
for candidate in python3 /usr/bin/python3 python; do
  if "$candidate" -c 'import numpy' >/dev/null 2>&1; then
    PY=$candidate
    break
  fi
done
if [ -z "$PY" ]; then
  echo "a python with numpy is required to check EDF headers" >&2
  exit 1
fi

download_atomic() {
  local url="$1"
  local dest="$2"
  local kind="$3"
  local tmp="${dest}.partial"
  local attempt
  rm -f "$tmp"
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    if curl -fL --retry 0 --max-time 300 -o "$tmp" "$url"; then
      if [ "$kind" = "edf" ]; then
        if "$PY" -m src.physionet_features --check-edf "$tmp"; then
          mv -f "$tmp" "$dest"
          return 0
        fi
        echo "  integrity check failed, attempt ${attempt}/${MAX_ATTEMPTS}: $dest" >&2
      else
        mv -f "$tmp" "$dest"
        return 0
      fi
    else
      echo "  download failed, attempt ${attempt}/${MAX_ATTEMPTS}: $url" >&2
    fi
    rm -f "$tmp"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
      sleep 5
    fi
  done
  rm -f "$tmp" "$dest"
  echo "FAILED after ${MAX_ATTEMPTS} attempts: $dest" >&2
  exit 1
}

echo "[1/2] UCI EEG Eye State (Phase 1)"
mkdir -p data/raw
if [ -s "data/raw/EEG Eye State.arff" ]; then
  echo "  ARFF already present, skipping download"
else
  uci_url="https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip"
  uci_ok=0
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    uci_tmp="$(mktemp)"
    if curl -sfL --max-time 300 -o "$uci_tmp" "$uci_url" && unzip -t "$uci_tmp" >/dev/null; then
      unzip -o "$uci_tmp" -d data/raw >/dev/null
      rm -f "$uci_tmp"
      uci_ok=1
      break
    fi
    rm -f "$uci_tmp"
    echo "  UCI download failed, attempt ${attempt}/${MAX_ATTEMPTS}" >&2
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
      sleep 5
    fi
  done
  if [ "$uci_ok" -ne 1 ]; then
    echo "FAILED after ${MAX_ATTEMPTS} attempts: UCI EEG Eye State" >&2
    exit 1
  fi
fi

echo "[2/2] PhysioNet Motor Movement/Imagery, imagery runs R04/R08/R12 (Phase 2)"
base="https://physionet.org/files/eegmmidb/1.0.0"
for s in $(seq -w 1 10); do
  mkdir -p "data/physionet/S0$s"
  for r in 04 08 12; do
    dest="data/physionet/S0$s/S0${s}R$r.edf"
    if [ -f "$dest" ] && "$PY" -m src.physionet_features --check-edf "$dest"; then
      continue
    fi
    download_atomic "$base/S0$s/S0${s}R$r.edf" "$dest" edf
  done
done

echo ""
echo "Validating every EDF against its header..."
"$PY" - << 'PY'
from src.physionet_features import EXPECTED_SUBJECTS, IMAGERY_RUNS, _expected_edf, assert_edf_complete

failures = []
for subj in EXPECTED_SUBJECTS:
    for run in IMAGERY_RUNS:
        path = _expected_edf(subj, run)
        try:
            assert_edf_complete(path)
        except ValueError as exc:
            failures.append(str(exc))
if failures:
    raise SystemExit("EDF integrity failures:\n  " + "\n  ".join(failures))
print(f"OK: {len(EXPECTED_SUBJECTS) * len(IMAGERY_RUNS)} EDFs match their headers.")
PY
echo "done."
