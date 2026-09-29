# Audit log

This audit began on 2026-09-29 and repairs the Phase 1 evaluation of the
NeuroSense repository. The Phase 2 protocol is unchanged. All pre-audit
research artifacts are preserved unmodified in the `archive/` subfolders so
that a reviewer can diff the old and new numbers directly.

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
- mne: not installed in the audit environment; Phase 2 loads the cached
  `data/processed/physionet_features.npz`, so MNE is not on the reproduction
  path for this audit

Note: `requirements.txt` currently pins `mne==2.4.1`, which does not exist on
PyPI (MNE has no 2.x). This is an unrelated bug in the pinned deps and does not
affect the audit runs, but it is flagged in `REFACTOR_NOTES.md` for a later
follow-up.

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

## What the audit does not change

- Phase 2 protocol (within-subject CV, leave-one-subject-out on 10 subjects).
- The prespecified primary model, logistic regression with `C = 1.0`,
  `class_weight="balanced"`, `max_iter=5000`, `random_state=42`.
- Feature bands, channel sets, window lengths, or the shared-core primitives
  in `src/core/features.py` and `src/core/evaluation.py`.

Where an audit rerun changes a headline number, the change is documented in
`REFACTOR_NOTES.md` under a new "Audit 2026-09" section.
