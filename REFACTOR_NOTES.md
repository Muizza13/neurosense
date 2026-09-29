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

## Still to do (as of the shared-core rebuild)

- Rewrite the README results table from `reports/results/*.json`, one model
  throughout.
- Track A versus Track B feature harmonisation before the three-panel figure.
- Retire `src/train.py`, `src/features.py`, `src/physionet.py`,
  `src/evaluate_physionet.py`, or keep them tagged as the pre-refactor versions.

## Audit 2026-09

Second-pass audit against the criteria in the audit brief.

### What was already correct

The shared-core rebuild had already put artifact clipping inside
`EpochBandPower`, which sits inside every Pipeline consumed by
`evaluate_loso` and `evaluate_naive_split`. The Phase 1
`chronological_holdout` also runs through that Pipeline. A repository-wide
grep for a hardcoded `F1 = 0.38` baseline found only Phase 2 macro-F1 CI
lower bounds (`0.380`, `0.384`, `0.387`) coming from the results JSON, not a
constant shared across protocols. No preprocessing leakage or shared
hardcoded baseline was present in the code at the start of the audit.

### What the audit added

- **Regression guard against future leakage**
  (`tests/test_no_leakage.py::test_held_out_mutation_cannot_change_training_preprocessing`).
  Snapshots the fitted clipper thresholds, scaler statistics, and
  transformed training features; mutates several held-out rows to `1e12`;
  refits on the same training rows; asserts every snapshot is
  bit-identical. A negative-control test verifies the guard is not
  vacuous, by showing that a global clipper *would* have been contaminated.
- **Temporal-dependence diagnostic**
  (`src/core/temporal.py::feature_lag_autocorr` and
  `choose_temporal_gap`). Measures lag-1..5 mean absolute autocorrelation of
  fold-safe band-power features. On this recording the values sit in
  [0.07, 0.10], well below the 0.30 threshold, so the chosen temporal gap
  is 0 windows and the choice is recorded in the results JSON with the
  reason (`already_below`).
- **Expanding-window evaluation across label blocks**
  (`src/core/evaluation.py::evaluate_expanding_window`). Walk-forward,
  starting with a third of the blocks in the training window and testing
  each subsequent block. Runs alongside the retained chronological holdout,
  not in place of it.
- **Per-split baselines everywhere**
  (`src/core/evaluation.py::per_split_baselines`). Every leakage-safe split
  reports its own majority-class accuracy, stratified-dummy balanced
  accuracy, and uniform-dummy balanced accuracy. No baseline is shared
  across protocols.
- **Fold traceability.** Every Phase 1 fold row carries a `provenance`
  block with test-window start-sample indices, block ids, and index hashes.
  A fold in the JSON can now be re-located in the raw recording without
  rerunning the code.
- **Preserved history.** Pre-audit JSONs and figures were copied verbatim
  to `reports/results/archive/pre-audit-e56156d/` and
  `reports/figures/archive/pre-audit-e56156d/` before any rerun.
  `reports/AUDIT_LOG.md` records the pre-audit commit hash, environment,
  and the exact set of changes.

### Phase 1 audit rerun, four protocols

Primary model unchanged (logistic regression, C = 1.0, balanced). Numbers
below come from `reports/results/phase1_results.json`.

| Protocol | balAcc | majority-class | stratified dummy | uniform dummy |
|---|---|---|---|---|
| Naive random window split | 0.533 | 0.550 | 0.552 | 0.536 |
| Chronological 70/30 (gap = 0) | 0.416 | 0.767 | 0.366 | 0.339 |
| Expanding-window (13 folds) | 0.380 [0.230, 0.545] | single-class folds | | |
| Leave-one-block-out (19 folds) | 0.482 [0.334, 0.630] | single-class folds | | |

The chronological logistic regression at 0.416 beats the two dummy
classifiers by a small margin but sits well below the 0.767 majority-class
baseline. The expanding-window mean of 0.380 sits close to a stratified
dummy. Neither is evidence of real generalisation.

For the leave-one-block-out and expanding-window protocols each held-out
block is single-class by construction, so majority-class accuracy is
trivially 1.0 and stratified/uniform baselines are undefined at the fold
level. The relevant baseline is chance (0.5), which every fold-level
interval brackets.

### Phase 2, unchanged

The Phase 2 script was rerun for reproducibility. The resulting JSON is
byte-identical to the pre-audit archive except for the provenance stamps
(commit hash and timestamp). No number moved.

### What the audit did not change

- The primary model, feature bands, channel sets, window lengths, or the
  shared-core primitives in `src/core/features.py`.
- Phase 2 protocol (within-subject trial CV and leave-one-subject-out).
- The withdrawn-claims list from the shared-core rebuild above.
