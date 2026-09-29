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
Intervals are percentile bootstrap over subjects. Leave-one-run-out
resamples 10 subject means, not 30 run rows. The cache has 450 trials
after every EDF was required to match its header length. An earlier
383-trial cache was built from files that failed that check. The
pre-audit archive still has 437 trials.

| Protocol | balanced accuracy | macro F1 | ROC AUC |
|---|---|---|---|
| Within-subject shuffled trial CV | 0.607 [0.546, 0.688] | 0.606 [0.544, 0.687] | 0.644 [0.577, 0.729] |
| Within-subject leave-one-run-out (10 subject means) | 0.574 [0.498, 0.654] | 0.543 [0.458, 0.632] | 0.633 [0.545, 0.729] |
| Cross-subject (LOSO) | 0.470 [0.423, 0.524] | 0.421 [0.377, 0.473] | 0.518 [0.450, 0.593] |
| Naive random trial split | 0.516 | 0.515 | 0.551 |

Per-subject cross-subject balanced accuracy: 0.638, 0.395, 0.361, 0.542,
0.464, 0.393, 0.499, 0.546, 0.402, 0.464 (range 0.361 to 0.638, std across
subjects 0.087).

### Two claims that did not survive

1. **"Cross-subject AUC = 0.48, below chance."** That was the pooled figure.
   Pooling probabilities across subjects whose decision scores sit on different
   scales manufactures apparent below-chance performance. The per-subject mean
   AUC is 0.511 with an interval of [0.447, 0.583]. The honest statement is
   chance, not below chance.

2. **"Within-subject decoding is real."** On this cache the shuffled-trial
   balanced-accuracy interval is [0.546, 0.688]. Leave-one-run-out, resampling
   subject means, is [0.498, 0.654] and uses three runs from the same recording.
   The earlier [0.522, 0.700] and [0.529, 0.648] intervals resampled a different
   cache and, for leave-one-run-out, a different unit. They are not this result.

The per-subject cross-subject AUC mean is 0.518 [0.450, 0.593]. The pooled
descriptive AUC is 0.476.

### Also worth noting

The naive random trial split reaches 0.516. Leave-one-subject-out is 0.470.
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

On the chronological test segment, a most-frequent dummy fitted on the training prefix scores balanced accuracy 0.500, macro F1 0.189, and AUC 0.500. The model scores 0.416, 0.330, and 0.491.

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

Dummy columns are balanced accuracy of classifiers fitted on training labels.

| Protocol | balAcc | AUC | most-frequent balAcc | stratified balAcc | uniform balAcc |
|---|---|---|---|---|---|
| Naive random window split | 0.533 | 0.566 | 0.500 | 0.581 | 0.444 |
| Chronological 70/30 (gap = 0) | 0.416 | 0.491 | 0.500 | 0.366 | 0.339 |
| Expanding-window (13 folds) | 0.380 [0.230, 0.545] | undefined | 0 or 1 per fold | | |
| **Leave-one-merged-block-out (primary, 4 folds)** | 0.517 [0.401, 0.631] | 0.518 [0.396, 0.617] | 0.500 | | |
| Leave-one-native-block-out (diagnostic, 19 folds) | 0.482 [0.334, 0.630] | undefined | 0 or 1 per fold | | |

On the merged splits the dummy prior is balAcc 0.500, macro F1 0.351 [0.310, 0.394], AUC 0.500. The model's macro F1 is 0.512 [0.392, 0.624]. The balanced-accuracy and AUC intervals include 0.500. The macro F1 intervals do not.

**Task 4: correcting the LOBO analysis.** Native leave-one-block-out
produced single-class test folds by construction, so per-fold AUC was
undefined and the pooled AUC across separately-trained models was open to a
splitting artefact. The audit added `DummyClassifier(strategy="prior")` on
the same native splits: it scores pooled AUC = 0.000 while learning nothing,
which reproduces the below-chance pooled-AUC artefact and rules out the
"electrode drift" reading. A pre-specified grouping rule (declared in
`src/phase1_eyestate.py` as `MERGED_BLOCK_SIZE = 4` native blocks per
super-block. If the block count is not a multiple of 4, the leftover
blocks join the last complete super-block. Nineteen native blocks become
four groups, and all four contain both classes. That count was measured
after grouping. Per-fold AUC is defined under leave-one-merged-block-out. This is the primary Phase 1 group
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

### Round 3 (Tasks 6 and 7)

**Task 6 (interpretation and explanation claims):**

- Rewrote Figure 3 so both panels use the **same** attribution method:
  signed standardized logistic-regression coefficients extracted from
  within real evaluation folds. Phase 1 uses the 5 merged-LOBO folds;
  Phase 2 uses per-subject models fitted on that subject's own trials.
  White dots on the figure are per-fold coefficients so between-fold
  spread is visible. Signs are preserved (not just magnitudes).
- Replaced the earlier "same interpretability method, two very different
  situations" claim — which was inaccurate, since Phase 1 was
  permutation importance and Phase 2 was |coef| on a model fit to all
  subjects at once.
- Labelled the Phase 1 attribution figure as a **descriptive ranking**,
  not physiological evidence, because the underlying model overlaps a
  same-splits dummy prior.
- Rewrote the Phase 2 interpretation from "confirms known physiology" to
  "consistent with expected sensorimotor patterns", and dropped the
  contrast rhetoric where it implied one attribution proved physiology.
- Removed the claim that frontal importance proved ocular artifact and
  the claim that poor cross-subject performance proved subject-identity
  learning; both would require dedicated experiments the repo does not
  run.
- Softened the "widely reported high accuracies on this dataset are an
  artifact" language: on this recording under the leakage-aware
  protocols implemented here, the merged-block balanced-accuracy and
  AUC intervals include the dummy prior, and the macro F1 intervals do
  not. That is not evidence that every published high-accuracy result
  on this dataset used a leaky evaluation.
- Documented explanation-faithfulness and model-randomisation tests as
  future work (not implemented).

**Task 7 (raw-data reproduction):**

- `scripts/download_data.sh` validates every expected EDF (10 subjects
  x 3 imagery runs = 30 files) by comparing the file length with the
  length declared in the EDF header. It writes to a temporary file,
  moves the file into place only after that check, retries up to 3
  times, and exits non-zero after exhausted retries. A size floor is
  not used. No silent skips.
- `src/physionet_features.py` rewritten to (a) validate the raw-data
  layout up front, (b) allow only R04, R08, R12, (c) read and validate
  the sampling rate on every EDF against `EXPECTED_FS = 160.0`, and (d)
  write `data/processed/physionet_features_manifest.json` recording
  subject ids, run ids, per-trial ids, channel order, frequency bands,
  epoch timing, Welch parameters, and the SHA-256 of every source EDF.
- The npz cache now stores `X, y, g, run, trial_id`. `trial_id` is a
  stable per-trial identifier of the form `S001_R04_T000`.
- Phase 1, Phase 2, and `make_figures.py` create `reports/figures/`,
  `reports/results/`, and `models/` on startup so a fresh clone runs
  without manual mkdir.
- `requirements.txt` documents the supported Python (3.9) and pins
  `mne==1.8.0`. Two documented reproduction paths in README (cached vs
  full raw-data rebuild) with the manifest as the byte-level integrity
  check between them.
- Added a licensing note in README for the committed derived feature
  cache: it contains no raw EEG samples, only per-trial band-power
  features and integer labels, so it does not constitute
  redistribution of the underlying recordings.

### Round 4 (Tasks 8-11)

**Task 8 (remove duplicated evaluation logic):**

- Added `evaluate_within_subject_shuffled_cv` and
  `_extract_fold_coefficients` to `src/core/evaluation.py`. Both are
  small; the goal is to share what is genuinely shared without
  building a framework. Dataset-specific loading (ARFF for Phase 1,
  EDF+MNE for Phase 2, `src/physionet_features.py`) stays in the phase
  scripts and does not leak into `src/core/`.
- `evaluate_loso` and `evaluate_within_subject_shuffled_cv` now accept
  `capture_coefficients=True`. Signed per-fold coefficients (Phase 1
  merged-LOBO) and per-subject mean coefficients (Phase 2 within-subject
  shuffled CV) are persisted to the phase-results JSON with feature
  names in the fitted-vector order.
- `src/make_figures.py` no longer fits any model. All three figures are
  built from `reports/results/phase1_results.json` and
  `phase2_results.json`. Round 3's Figure 3 had refit both phases
  independently; round 4 removes that duplication.
- Added `src/make_tables.py`, which regenerates the phase 1 / phase 2
  markdown tables from the same JSONs. The README and the LaTeX report
  can cite the same numbers without independent rounding.
- Feature spaces are deliberately not unified across the two phases
  (14 EEG channels x 4 bands for Phase 1, 13 sensorimotor channels x 2
  bands for Phase 2). The two datasets are not comparable to each
  other; forcing a common feature space would imply a controlled
  comparison the audit is careful not to claim.

**Task 9 (CSP + LDA extension):**

- `src/phase2_csp_lda.py` implements CSP (6 log-variance components) +
  LDA on the same 10 subjects, the same imagery runs (R04, R08, R12),
  and the same evaluation splits (within-subject shuffled trial CV,
  leave-one-run-out, leave-one-subject-out) as the audited band-power
  primary. CSP and LDA are fitted inside training folds only. No
  learned transformation crosses a fold boundary.
- Signal filtering is documented in-source and in the JSON: 4th-order
  Butterworth zero-phase bandpass 8-30 Hz per epoch, 13-channel motor
  strip, epoch window 0.5-3.5 s. Sampling rate is read from and
  asserted against `EXPECTED_FS = 160.0` on every EDF via the shared
  physionet loader.
- Results land in `reports/results/phase2_csp_lda_results.json`. The
  primary `phase2_results.json` is not touched. The README labels the
  extension as such and does not fold its numbers into the audited
  band-power tables.
- Within-subject the CSP extension is meaningfully stronger than the
  band-power primary (about 5 balAcc points, 6-10 AUC points).
  Cross-subject the point estimate rises to 0.541 balAcc (from 0.474)
  and 0.654 AUC (from 0.511), but the balanced-accuracy CI still just
  brackets 0.5 (0.478 lower bound); readable as "CSP lifts the
  cross-subject baseline without cleanly clearing chance on 10
  subjects", not as "cross-subject decoding is solved".
- No EEGNet or transformers. Not in the audit's scope.

**Task 10 (tests and CI):**

- Added `tests/test_round4_extras.py` (14 tests) covering the items
  Task 10 lists explicitly: LORO run separation, feature-free
  pooled-AUC diagnostic (dummy prior on single-class folds),
  per-protocol dummy baselines on every fold, deterministic bootstrap
  output for a fixed seed, auto-creation of missing output
  directories, result-schema invariants on the phase 1 / phase 2
  JSONs, coefficient capture behaviour, and a small end-to-end smoke
  test on synthetic subject-fingerprint data.
- Every fixture is synthetic (`_synthetic_multi_run`,
  `subject_fingerprint_data`) and clearly separated from the real
  research artefacts. The schema tests read the committed JSONs
  read-only and skip when they are absent, so tests still pass in a
  fresh checkout without the raw-data cache.
- CI workflow at `.github/workflows/tests.yml` runs on push and PR.
  It installs the pinned dependencies (without the MNE line, which
  is only required for the raw-EDF Path B), runs `pytest tests/`,
  and rebuilds the three figures and two tables from the committed
  JSONs to guard against `make_figures.py` or `make_tables.py`
  quietly refitting a model. CI never downloads the raw dataset.
- Test count grew from 58 to 73 across rounds 3 and 4 (49 pre-round-3
  + 9 manifest tests in round 3 + 15 new tests in round 4).

**Task 11 (research artifacts):**

- README gains the CSP+LDA extension section, updates the repo
  structure block, and adds a one-paragraph "how to read the four
  rounds" summary linking historical, corrected, and extension
  results.
- LaTeX report gains a Phase 2 CSP+LDA subsection and a Reproducibility
  section update.
- PDF rebuilt via `tectonic -X compile`.

### What the audit did not change

- The primary model, feature bands, channel sets, window lengths, or the
  shared-core primitives in `src/core/features.py`.
- The withdrawn-claims list from the shared-core rebuild above.
- The Phase 2 cross-subject protocol is still leave-one-subject-out
  with a subject-level bootstrap. The point estimates in this file are
  from the 450-trial cache, not from the withdrawn 383-trial cache.

### Framing note

This audit is a rerun on data that has been inspected many times in this
project. It is not an untouched confirmatory study of either paradigm.
Where the rerun changes a number or a claim, the change is documented
here and in `reports/AUDIT_LOG.md`; where the rerun agrees with the
pre-audit result, that agreement should be read as internal consistency
rather than as independent replication.
