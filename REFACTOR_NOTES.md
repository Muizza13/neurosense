# Refactor notes: shared core, and what the numbers did

Phases 1 and 2 now run through `src/core/`. The conclusions of both phases
survive. Several reported numbers changed, and two published claims did not
survive.

## Completion criteria

| Criterion | Status |
|---|---|
| No feature function contains a hardcoded sampling frequency | done, `band_power` takes `sfreq` |
| Windowing separated from feature extraction | done, `make_continuous_windows` / `make_recording_epochs` / `band_power` |
| Fold-level preprocessing cannot see held-out data | done, `EpochBandPower` clips and extracts inside the Pipeline |
| Phase 2 produces one metric row per held-out subject | done |
| Bootstrap CIs resample subjects, not trials | done, `statistics.bootstrap_ci` |
| README metrics come from one named model | pending, see below |
| Hyperparameters prespecified or nested-tuned | done, C = 1.0 prespecified; nested path available via `tune=True` |
| Results saved to deterministic JSON | done, `reports/results/*.json`, sorted keys |
| Phases 1 and 2 reproduce through refactored code | done |
| PREREGISTRATION.md committed before Phase 3 results | done, uncommitted results |

## Phase 2, corrected

Primary model, prespecified: logistic regression, C = 1.0, balanced.
Intervals are percentile bootstrap over the 10 subjects.

| Protocol | balanced accuracy | macro F1 | ROC AUC |
|---|---|---|---|
| Within-subject (trial CV) | 0.580 [0.493, 0.675] | 0.578 [0.490, 0.674] | 0.624 [0.536, 0.721] |
| Cross-subject (LOSO) | 0.482 [0.429, 0.539] | 0.431 [0.380, 0.487] | 0.520 [0.439, 0.608] |
| Naive random trial split | 0.540 | - | 0.525 |

Per-subject cross-subject balanced accuracy, which the pooled number hid
entirely: 0.637, 0.442, 0.384, 0.608, 0.443, 0.387, 0.522, 0.546, 0.365, 0.482.

### Two claims that did not survive

1. **"Cross-subject AUC = 0.48, below chance."** That was the pooled figure.
   Pooling probabilities across subjects whose decision scores sit on different
   scales manufactures apparent below-chance performance. The per-subject mean
   AUC is 0.520 with an interval of [0.439, 0.608]. The honest statement is
   chance, not below chance.

2. **"Within-subject decoding is real."** The per-subject balanced accuracy
   interval is [0.493, 0.675]. It includes 0.5. The AUC interval [0.536, 0.721]
   does exclude 0.5, and 8 of 10 subjects sit above chance on balanced accuracy.
   So the defensible claim is weaker and more specific: within-subject ranking
   is above chance, within-subject thresholded accuracy is not clearly so.

The overall conclusion is unchanged. Motor imagery decodes modestly within
subject and not at all across subjects.

### Also worth noting

The naive random trial split reaches only 0.540, barely above the LOSO 0.482.
Phase 2 uses discrete trials, so random splitting leaks far less than it does in
Phase 1's continuous windows. The inflation story is a Phase 1 phenomenon and
should be presented as such rather than as a general claim.

## Phase 1, refactored

The original `train.py` called `clip_artifacts()` on all 14980 samples before
windowing and before the chronological split, so winsorisation thresholds were
computed partly from held-out future samples. Clipping is now fitted inside the
fold.

| Protocol | balanced accuracy | macro F1 | ROC AUC |
|---|---|---|---|
| Naive random window split | 0.533 | 0.533 | 0.566 |
| Chronological 70/30 holdout | 0.416 | 0.330 | 0.491 |
| Leave-one-block-out | 0.482 [0.334, 0.630] | 0.369 [0.239, 0.515] | undefined |

Majority baseline accuracy on the chronological test segment is 0.767.

Two structural facts that deserve to be visible in the report:

- The recording yields **100 usable one-second windows**. Every Phase 1 number
  rests on 100 observations from one person, which is why the leave-one-block-out
  interval is nearly half a unit wide.
- Every contiguous label block is single-class, so per-block AUC is undefined and
  per-block balanced accuracy degenerates into the recall of whichever class the
  block contains. Any previously reported per-block AUC range must have come from
  pooling. Chronological holdout is the more interpretable leakage-safe protocol
  here, and leave-one-block-out should be reported as a supporting check.

## Still to do

- Rewrite the README results table from `reports/results/*.json`, one model
  throughout.
- Track A versus Track B feature harmonisation before the three-panel figure.
- Retire `src/train.py`, `src/features.py`, `src/physionet.py`,
  `src/evaluate_physionet.py`, or keep them tagged as the pre-refactor versions.
