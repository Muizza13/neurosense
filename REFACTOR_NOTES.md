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
Intervals are percentile bootstrap over the 10 subjects (or 30 LORO folds).
Numbers below reflect the post-audit re-extraction with run IDs (383 trials
across 10 subjects; the pre-audit cache had 437 trials because an older MNE
version retained more edge epochs near the concatenation boundary).

| Protocol | balanced accuracy | macro F1 | ROC AUC |
|---|---|---|---|
| Within-subject shuffled trial CV | 0.607 [0.522, 0.700] | 0.605 [0.518, 0.698] | 0.632 [0.534, 0.735] |
| Within-subject leave-one-run-out (30 folds) | 0.588 [0.529, 0.648] | 0.556 [0.490, 0.622] | 0.657 [0.598, 0.718] |
| Cross-subject (LOSO) | 0.474 [0.432, 0.516] | 0.427 [0.380, 0.474] | 0.511 [0.447, 0.583] |
| Naive random trial split | 0.504 | - | 0.521 |

Per-subject cross-subject balanced accuracy, which the pooled number hid
entirely: 0.545, 0.471, 0.381, 0.600, 0.487, 0.353, 0.500, 0.496, 0.449, 0.461
(range 0.353 to 0.600, std across subjects 0.072).

### Two claims that did not survive

1. **"Cross-subject AUC = 0.48, below chance."** That was the pooled figure.
   Pooling probabilities across subjects whose decision scores sit on different
   scales manufactures apparent below-chance performance. The per-subject mean
   AUC is 0.511 with an interval of [0.447, 0.583]. The honest statement is
   chance, not below chance.

2. **"Within-subject decoding is real."** After the round-2 re-extraction, the
   per-subject balanced accuracy interval is [0.522, 0.700] on shuffled trial
   CV and [0.529, 0.648] on leave-one-run-out; both narrowly exclude 0.5. The
   defensible claim is: within-subject decoding is modestly above chance and
   consistent across recording runs of the same session; it does not establish
   transfer to a new recording session.

The overall conclusion is unchanged. Motor imagery decodes modestly within
subject and not at all across subjects.

### Also worth noting

The naive random trial split reaches only 0.504, barely above the LOSO 0.474.
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

### Phase 1 audit rerun, five leakage-aware protocols

Primary model unchanged (logistic regression, C = 1.0, balanced). Numbers
below come from `reports/results/phase1_results.json`.

| Protocol | balAcc | AUC | majority-class | stratified dummy | uniform dummy |
|---|---|---|---|---|---|
| Naive random window split | 0.533 | 0.566 | 0.550 | 0.552 | 0.536 |
| Chronological 70/30 (gap = 0) | 0.416 | 0.491 | 0.767 | 0.366 | 0.339 |
| Expanding-window (13 folds) | 0.380 [0.230, 0.545] | undefined | single-class folds | | |
| **Leave-one-merged-block-out (primary, 5 folds)** | 0.456 [0.270, 0.605] | 0.533 [0.396, 0.629] | mixed | mixed | mixed |
| Leave-one-native-block-out (diagnostic, 19 folds) | 0.482 [0.334, 0.630] | undefined | single-class folds | | |

None of the leakage-aware protocols shows evidence of decoding.

**Task 4: correcting the LOBO analysis.** Native leave-one-block-out
produced single-class test folds by construction, so per-fold AUC was
undefined and the pooled AUC across separately-trained models was open to a
splitting artefact. The audit added `DummyClassifier(strategy="prior")` on
the same native splits: it scores pooled AUC = 0.000 while learning nothing,
which reproduces the below-chance pooled-AUC artefact and rules out the
"electrode drift" reading. A pre-specified grouping rule (declared in
`src/phase1_eyestate.py` as `MERGED_BLOCK_SIZE = 4` native blocks per
super-block, chosen for structural reasons before evaluation and not tuned
on results) partitions the 19 native blocks into 5 super-blocks. 4 of the
5 super-blocks contain both classes, giving well-defined per-fold AUC
under leave-one-merged-block-out. This is the primary Phase 1 group
protocol; native LOBO is retained only as a labelled diagnostic.

### Phase 2 audit rerun

Primary model unchanged (logistic regression, C = 1.0, balanced). Cross-
subject leave-one-subject-out now runs as an explicit loop with:

- per-subject `y_true`, `y_pred`, `y_proba` persisted in the JSON, so a
  reader can rescore or diagnose without re-fitting;
- `std_across_subjects` reported alongside the bootstrap CI, so the
  reader can distinguish spread from uncertainty of the mean;
- "N of 10 above 0.5" labelled as a descriptive count, not a
  significance claim;
- pooled cross-subject scores kept only as a labelled descriptive
  appendix.

A new **within-subject leave-one-run-out (LORO)** evaluation was added,
requiring the run identifier to be preserved during feature extraction
(`src/physionet_features.py`, now caches `X, y, g, run`). For each
subject, the model trains on two of the three imagery runs (R04, R08,
R12) and tests on the third. This tests transfer across recording runs of
the same session; it does not establish transfer to a new recording
session. The shuffled trial CV is retained and labelled as such
(trials from the same run appear in both train and test).

### What the audit did not change

- The primary model, feature bands, channel sets, window lengths, or the
  shared-core primitives in `src/core/features.py`.
- The withdrawn-claims list from the shared-core rebuild above.
- The Phase 2 cross-subject leave-one-subject-out point estimates
  themselves (byte-identical up to provenance stamps).
