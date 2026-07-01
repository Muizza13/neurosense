#!/usr/bin/env bash
# Downloads the datasets used in this project. Data is NOT stored in the repo.
set -e

echo "[1/2] UCI EEG Eye State (Phase 1)"
mkdir -p data/raw
curl -sL -o data/raw/uci.zip "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip"
unzip -o data/raw/uci.zip -d data/raw >/dev/null && rm -f data/raw/uci.zip

echo "[2/2] PhysioNet Motor Movement/Imagery, imagery runs R04/R08/R12 (Phase 2)"
base="https://physionet.org/files/eegmmidb/1.0.0"
for s in $(seq -w 1 10); do
  mkdir -p "data/physionet/S0$s"
  for r in 04 08 12; do
    curl -sfL --max-time 30 -o "data/physionet/S0$s/S0${s}R$r.edf" "$base/S0$s/S0${s}R$r.edf" || echo "skip S0${s}R$r"
  done
done
echo "done."
