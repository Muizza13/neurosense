#!/usr/bin/env bash
# Downloads the datasets used in this project. Data is NOT stored in the repo.
#
# Two datasets:
#   1. UCI EEG Eye State (Phase 1). ~200 KB.
#   2. PhysioNet EEG Motor Movement/Imagery, imagery runs R04/R08/R12 only,
#      subjects S001..S010. ~60 MB total.
#
# The Phase 2 script does not need this data if data/processed/physionet_features.npz
# is already present (see "Cached vs. raw reproduction" in README.md).
# Rerun this script to force a fresh raw-data reproduction. It will fail loudly
# on any missing or truncated EDF rather than silently skipping.

set -euo pipefail

MIN_EDF_BYTES=500000

echo "[1/2] UCI EEG Eye State (Phase 1)"
mkdir -p data/raw
curl -sL -o data/raw/uci.zip "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip"
unzip -o data/raw/uci.zip -d data/raw >/dev/null && rm -f data/raw/uci.zip

echo "[2/2] PhysioNet Motor Movement/Imagery, imagery runs R04/R08/R12 (Phase 2)"
base="https://physionet.org/files/eegmmidb/1.0.0"
for s in $(seq -w 1 10); do
  mkdir -p "data/physionet/S0$s"
  for r in 04 08 12; do
    f="data/physionet/S0$s/S0${s}R$r.edf"
    if [ -s "$f" ] && [ "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")" -ge $MIN_EDF_BYTES ]; then
      continue                  # already present and non-truncated
    fi
    # 300 s per file: PhysioNet is often throttled; a 30 s ceiling had
    # produced truncated EDFs. Retry up to 3 times.
    for attempt in 1 2 3; do
      if curl -sfL --max-time 300 -o "$f" "$base/S0$s/S0${s}R$r.edf"; then
        break
      fi
      echo "  retry $attempt for S0${s}R$r"
      sleep 5
    done
  done
done

echo ""
echo "Validating downloads..."
missing=""
truncated=""
for s in $(seq -w 1 10); do
  for r in 04 08 12; do
    f="data/physionet/S0$s/S0${s}R$r.edf"
    if [ ! -s "$f" ]; then
      missing="$missing\n  $f"
    else
      sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
      if [ "$sz" -lt $MIN_EDF_BYTES ]; then
        truncated="$truncated\n  $f ($sz bytes)"
      fi
    fi
  done
done

if [ -n "$missing" ] || [ -n "$truncated" ]; then
  echo ""
  if [ -n "$missing" ]; then
    printf "MISSING EDFs:%b\n" "$missing" >&2
  fi
  if [ -n "$truncated" ]; then
    printf "TRUNCATED EDFs (< $MIN_EDF_BYTES bytes):%b\n" "$truncated" >&2
  fi
  echo ""
  echo "Rerun this script. Do not proceed with a partial download." >&2
  exit 1
fi

echo "OK: 30 EDFs present (S001..S010, R04/R08/R12), all above $MIN_EDF_BYTES bytes."
echo "done."
