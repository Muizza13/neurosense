# Audit log

The 2026-09 audit had three rounds. Round 1 (2026-09-29) added leakage-safe
protocols, per-split baselines, a temporal-dependence diagnostic, an
expanding-window evaluation, a regression guard, and fold provenance to
Phase 1; Phase 2 was unchanged. Round 2 (later the same day) corrected the
leave-one-block-out analysis, added leave-one-run-out to Phase 2, persisted
per-fold labels and predictions in the Phase 2 JSON, and tightened
reporting language around baselines, standard deviation vs. confidence
interval, and "N of 10 above 0.5" as a descriptive count. Round 3 (Tasks 6
and 7) rebuilt Figure 3 so both panels use the same attribution method
(signed standardized coefficients from within real evaluation folds),
corrected the interpretation language around ocular artifact,
subject-identity learning, and "confirms physiology", hardened raw-data
reproduction with a validated download script and a feature manifest, and
documented explanation-faithfulness and randomisation tests as future
work.

This is a rerun on data that has been inspected many times in this project.
It is not an untouched confirmatory study of either paradigm.

All pre-audit research artifacts are preserved unmodified in the `archive/`
subfolders so a reviewer can diff the old and new numbers directly.

## Provenance of the pre-audit state

- Branch inherited from: `refactor/shared-core`
- Head commit at start of audit: `e56156de914906b9cba098c8bd8e93c55be1b594`
  (short: `e56156d`, message: "Retire pre-refactor pipeline superseded by
  src/core")
- Audit branch: `audit/repair-eval-2026-09`
- Audit start (UTC): 2026-09-29

## Environment (audit runs)

- Platform: `macOS-15.7.9-arm64-arm-64bit`
- Python: `3.9.6`
- numpy: `2.0.2`
- scipy: `1.13.1`
- scikit-learn: `1.6.1`
- pandas: `2.3.3`
- matplotlib: `3.9.4`
- mne: `1.8.0` installed for round 2 to re-extract Phase 2 features with
  run IDs preserved

Note: `requirements.txt` currently pins `mne==2.4.1`, which does not exist on
PyPI (MNE has no 2.x). The audit uses `mne==1.8.0` in the local Python 3.9
environment. Pin should be corrected in a follow-up.

## Preserved as historical artifacts

Copied verbatim from the pre-audit tree, no edits:

- `reports/results/archive/pre-audit-e56156d/phase1_results.json`
- `reports/results/archive/pre-audit-e56156d/phase2_results.json`
- `reports/figures/archive/pre-audit-e56156d/fig1_phase1_leakage.png`
- `reports/figures/archive/pre-audit-e56156d/fig2_phase2_generalization.png`
- `reports/figures/archive/pre-audit-e56156d/fig3_interpretability_contrast.png`

These files record the state the current README and report were written
against. If a rerun produces different numbers, the archive is the ground
truth for what the earlier version claimed.

## What the audit changes

### Round 1

1. Adds a regression test that fabricates a held-out epoch with extreme
   values, refits the Phase 1 pipeline on the training epochs, and asserts
   that the clipper thresholds and the transformed training features are
   byte-identical to a baseline without the mutation.
2. Threads per-window start-sample timestamps and per-fold provenance
   (`train_indices_hash`, `test_start_samples`, `test_block_ids`) into every
   Phase 1 fold, so a fold in the JSON can be re-located in the raw
   recording.
3. Adds a temporal-dependence diagnostic on the Phase 1 band-power features
   and reports lag-1 through lag-5 mean absolute autocorrelation in the
   Phase 1 results JSON.
4. Chooses a temporal gap from the diagnostic and applies it uniformly to
   the chronological holdout and the expanding-window evaluation. The chosen
   gap and its justification are recorded in the JSON.
5. Adds an expanding-window (walk-forward across label blocks) evaluation to
   Phase 1. Reported alongside the existing chronological holdout, not in
   place of it.
6. Computes majority-class, stratified-dummy, and uniform-dummy baselines
   independently for every Phase 1 split. No baseline is shared across
   protocols.

### Round 2 (Tasks 4 and 5)

7. Replaces leave-one-native-block-out as the primary Phase 1 group
   protocol with a pre-specified leave-one-merged-block-out. Merged-block
   size is declared in `src/phase1_eyestate.py` as
   `MERGED_BLOCK_SIZE = 4` native blocks per super-block, chosen for
   structural reasons before evaluation and not tuned on results. 4 of 5
   super-blocks contain both classes; per-fold AUC is defined for those.
8. Adds a `DummyClassifier(strategy="prior")` diagnostic that runs on the
   same splits as the primary model for both merged-LOBO and native-LOBO.
   Native-LOBO's pooled AUC of 0.437 (real model) vs 0.000 (dummy prior)
   demonstrates that pooled AUC on separately-trained single-class folds
   is a splitting artefact rather than electrode drift.
9. Retains native leave-one-block-out only as a clearly labelled
   diagnostic.
10. Fixes `scripts/download_data.sh` (curl `--max-time` bumped from 30 s
    to 300 s) after a prior download had produced truncated EDFs.
11. Rewrites `src/physionet_features.py` to preserve run IDs (R04, R08,
    R12) and cache them alongside `X, y, g` in
    `data/processed/physionet_features.npz`.
12. Adds `evaluate_loso(persist_predictions=True, dummy_strategy=...)` and
    persists `y_true`, `y_pred`, `y_proba` per fold in the Phase 2 JSON.
13. Adds a within-subject leave-one-run-out evaluation to Phase 2, labelled
    as testing transfer across runs of the same session (not to a new
    session).
14. Reports `std_across_subjects` separately from the bootstrap CI of the
    mean, and re-labels "N of 10 above 0.5" as a descriptive count rather
    than a significance claim.
15. Retains the shuffled trial CV within Phase 2 with an accurate
    description of its limitation (trials from the same run appear in
    both train and test).

### Round 3 (Tasks 6 and 7)

16. Rewrites Figure 3 so both panels use the same attribution method:
    signed standardized logistic-regression coefficients extracted from
    within real evaluation folds. Phase 1 uses the 5 merged-LOBO folds;
    Phase 2 uses per-subject within-subject models. Signs are preserved
    (not just magnitudes); per-fold values are plotted as white dots so
    between-fold spread is visible.
17. Corrects the earlier "same interpretability method" claim, which was
    inaccurate: Phase 1 was RandomForest permutation importance and
    Phase 2 was |coef| on a model fit to all subjects at once.
18. Removes claims that Phase 1 frontal importance proves ocular artifact
    and that Phase 2 poor cross-subject performance proves subject-identity
    learning; both would require dedicated experiments not implemented
    here.
19. Replaces "confirms known physiology" with "consistent with expected
    sensorimotor patterns" in the Phase 2 interpretation. Labels the
    Phase 1 attribution ranking as descriptive rather than physiological
    evidence, since the underlying model overlaps a same-splits dummy
    prior.
20. Softens the "widely reported high accuracies on this dataset are an
    artifact" language: the finding is scoped to this recording and to
    the leakage-aware protocols implemented here.
21. Documents explanation-faithfulness (permutation-of-top-features vs
    held-out score drop) and model-randomisation tests as future work.
22. Hardens `scripts/download_data.sh`: 3-retry loop, 300 s timeout, hard
    validation that all 10 subjects x 3 imagery runs are present and >=
    500 KB each. Exits non-zero if any file is missing or truncated. No
    silent skips.
23. Rewrites `src/physionet_features.py` to (a) validate all expected
    files up front, (b) allow only imagery runs R04, R08, R12, (c) read
    and validate `raw.info["sfreq"]` against `EXPECTED_FS = 160.0` on
    every EDF, (d) write `data/processed/physionet_features_manifest.json`
    with per-subject SHA-256 checksums, run trial counts, channel order,
    frequency bands, epoch timing (0.5-3.5 s), Welch parameters, and
    per-trial identifiers of the form `S001_R04_T000`.
24. Adds `trial_id` to the npz cache (`X, y, g, run, trial_id`), so
    per-fold predictions in the Phase 2 JSON can be linked back to
    specific trials.
25. Adds `os.makedirs("reports/figures", "reports/results", "models",
    exist_ok=True)` at the top of every runnable script so a fresh clone
    runs without manual mkdir.
26. Updates `requirements.txt` to document the supported Python (3.9,
    audit used 3.9.6) and pins `mne==1.8.0` (round-2 note about the
    `mne==2.4.1` typo is now fixed here).
27. Adds a "cached vs raw-data reproduction" section to the README with
    the two paths explicitly documented, and a licensing note for the
    committed derived feature cache (band-power features only, no raw
    EEG samples, so not redistribution of the PhysioNet recordings).

## What the audit does not change

- The prespecified primary model, logistic regression with `C = 1.0`,
  `class_weight="balanced"`, `max_iter=5000`, `random_state=42`.
- Feature bands, channel sets, window lengths, or the shared-core primitives
  in `src/core/features.py` and `src/core/evaluation.py`.
- The Phase 2 cross-subject LOSO point estimates (byte-identical up to
  provenance stamps after re-extraction from the newly downloaded EDFs).

Where an audit rerun changes a headline number, the change is documented in
`REFACTOR_NOTES.md` under the "Audit 2026-09" section.
