# NeuroSense: Leakage-Aware, Explainable EEG Decoding

**An honest look at when EEG machine learning actually generalizes, and when it only appears to.**

This repository reports logistic-regression scores on two public EEG datasets under standard train/test rules: a random split, a chronological holdout, walk-forward blocks, leave-one-block-out, leave-one-subject-out, and leave-one-run-out. Those rules are not a new method. The numbers below are what this code produced on these files.

A full write-up is in [`reports/NeuroSense_Report.pdf`](reports/NeuroSense_Report.pdf).

## Core finding

> On the UCI eye-state recording, the leakage-safe scores do not beat the same-split dummy. On the PhysioNet subset, within-subject band-power logistic regression is above 0.5. Leave-one-subject-out is not. Coefficient lists are the fitted weights of that model. They are not a physiological result.

## How to read the numbers

Both phases run through a shared core (`src/core/`) with one band definition, preprocessing fitted inside each fold, and one prespecified model throughout: logistic regression, C = 1.0, `class_weight="balanced"`, `max_iter=5000`, `random_state=42`. Other models appear only in secondary tables.

**The unit of inference is the subject, or in Phase 1 the label block, never the epoch.** Every interval below is a 95 percent percentile bootstrap resampling subjects or blocks. Pooled epoch-level figures are recorded in the results JSON as descriptive only, because pooling correlated epochs and treating them as independent both understates uncertainty and distorts threshold-free metrics.

**Every leakage-safe split reports its own baselines.** Majority-class accuracy, a stratified dummy, and a uniform dummy are computed on that split's test set alone. Nothing is shared across protocols. Where a protocol produces single-class test folds by construction (leave-one-block-out and expanding-window on Phase 1), the meaningful baseline is chance (0.5) and the majority-class number is trivially 1.0.

All numbers are produced by the code in this repo and stored in `reports/results/`. Nothing is rounded up.

## Results at a glance

| Setting                 | Evaluation                                                   | Result                                                                                          | Reading                              |
| ----------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------ |
| Phase 1 (eye state)     | Naive random window split                                    | balAcc = 0.533, majority 0.55                                                                   | leaked, near dummy                   |
| Phase 1 (eye state)     | Chronological holdout (gap = 0 windows)                      | balAcc = 0.416, majority 0.767, stratified dummy 0.366                                          | below majority                       |
| Phase 1 (eye state)     | Expanding-window (13 folds across blocks)                    | balAcc = 0.380 [0.230, 0.545]                                                                   | at level of a stratified dummy, wide |
| Phase 1 (eye state)     | Leave-one-merged-block-out (5 folds, super-block = 4 blocks) | balAcc = 0.456 [0.270, 0.605], AUC = 0.533 [0.396, 0.629]; dummy-prior balAcc 0.400             | at level of dummy prior              |
| Phase 1 (eye state)     | Leave-one-native-block-out (19 single-class folds)           | diagnostic only; pooled AUC 0.437 real vs 0.000 dummy prior on same splits, i.e. split artefact | do not read as drift                 |
| Phase 1 (eye state)     | Cross-subject                                                | not available                                                                                   | one subject                          |
| Phase 2 (motor imagery) | Naive random trial split                                     | balAcc = 0.504                                                                                  | near dummy                           |
| Phase 2 (motor imagery) | Within-subject shuffled trial CV                             | balAcc = 0.607 [0.522, 0.700], AUC = 0.632 [0.534, 0.735]; 8/10 subjects above 0.5              | modest, ranking above chance         |
| Phase 2 (motor imagery) | Within-subject leave-one-run-out (30 folds)                  | balAcc = 0.588 [0.529, 0.648], AUC = 0.657 [0.598, 0.718]; tests transfer across runs of same session, NOT across sessions | modest, added in 2026-09 audit       |
| Phase 2 (motor imagery) | Cross-subject (leave-one-subject-out)                        | balAcc = 0.474 [0.432, 0.516], AUC = 0.511 [0.447, 0.583]                                       | chance                               |

The 2026-09 audit added (a) temporal-gap chronological holdout and expanding-window on Phase 1, (b) leave-one-merged-block-out with a DummyClassifier(prior) diagnostic that replaces single-class native LOBO as the primary group protocol, (c) leave-one-run-out within subject on Phase 2, (d) persisted per-fold labels, predictions, and probabilities in Phase 2 JSON, (e) per-split baselines throughout, (f) a hardened raw-data reproduction path with a manifest of SHA-256 source-file checksums, (g) signed per-fold coefficients captured and consumed by the figure code so figures do not refit models, and (h) a CSP + LDA extension for Phase 2 as a separately labelled comparison to the audited band-power primary. The primary model is the same as before (LogisticRegression, C = 1.0, balanced). See [`REFACTOR_NOTES.md`](REFACTOR_NOTES.md) and [`reports/AUDIT_LOG.md`](reports/AUDIT_LOG.md) for the full change list.

**How to read the four rounds.** Round 1 fixed preprocessing leakage and Phase 1 baselines. Round 2 corrected the leave-one-block-out analysis and rebuilt Phase 2 reporting. Round 3 corrected interpretation language, unified the attribution method across both figures, and hardened raw-data reproduction. Round 4 removed duplicated evaluation logic (figures and tables now read the same JSON, no refit), added the CSP+LDA extension, added the CI workflow, and expanded test coverage. Numbers reported here are the round-3/round-4 numbers; historical pre-audit numbers are preserved verbatim under `reports/results/archive/pre-audit-<commit>/` and `reports/figures/archive/pre-audit-<commit>/`.

"Above 0.5" is reported as a descriptive count, not a significance claim. Standard deviation across subjects is spread; the bootstrap CI is uncertainty of the mean, computed by resampling subjects or blocks (never trials).

---

## Phase 1: UCI EEG Eye State (the cautionary result)

**Dataset.** A single continuous 117-second recording from **one person**, 14 channels at 128 Hz. The eyes-open/closed label runs in only 24 contiguous blocks (median 3.9 s), 19 of which contain at least one complete window. Several channels contain single-sample electrode pops up to around 700,000 against a 4,000 baseline, clipped before feature extraction using thresholds fitted on training data only. One-second non-overlapping windows yield **100 windows** and 56 band-power features (delta, theta, alpha, beta). Gamma is not included.

Every Phase 1 number rests on 100 observations from one person. That single fact drives the width of every interval below and is the main reason this phase is a cautionary tale rather than a result.

**Temporal-dependence diagnostic.** Before choosing the temporal gap between train and test in the chronological and expanding-window protocols, mean absolute autocorrelation of the fold-safe band-power features is measured at lags 1 through 5. Values on this recording sit in [0.07, 0.10], well below the 0.30 threshold, so **the chosen gap is 0 windows** and the choice is recorded in the results JSON under `temporal_dependence`. The autocorrelation is measured on features extracted with a clipper fitted on the training half of the chronological split, so the values reflect what the model actually consumes.

**Five leakage-aware protocols, per-split baselines.**

| Protocol                                          | Balanced accuracy    | Macro F1             | ROC AUC              | Majority | Stratified dummy | Uniform dummy |
| ------------------------------------------------- | -------------------- | -------------------- | -------------------- | -------- | ---------------- | ------------- |
| Naive random window split (leaky)                 | 0.533                | 0.533                | 0.566                | 0.550    | 0.552            | 0.536         |
| Chronological 70/30 (gap = 0)                     | 0.416                | 0.330                | 0.491                | 0.767    | 0.366            | 0.339         |
| Expanding-window (13 folds)                       | 0.380 [0.230, 0.545] | 0.283 [0.165, 0.432] | undefined            | see note | see note         | see note      |
| **Leave-one-merged-block-out (primary, 5 folds)** | 0.456 [0.270, 0.605] | 0.433 [0.255, 0.582] | 0.533 [0.396, 0.629] | mixed    | mixed            | mixed         |
| Leave-one-native-block-out (diagnostic, 19 folds) | 0.482 [0.334, 0.630] | 0.369 [0.239, 0.515] | undefined            | see note | see note         | see note      |

Note: for native leave-one-block-out and expanding-window each held-out block is single-class by construction, so majority-class accuracy is trivially 1.0 and the dummy baselines degenerate. The meaningful reference for those protocols is chance (0.5), which every reported interval brackets. The merged-block variant partitions consecutive native blocks into super-blocks of size 4 (declared in source before evaluation; not tuned on results); 4 of 5 super-blocks contain both classes, so per-fold AUC is well-defined.

**On the merged-LOBO number vs a dummy prior.** On the same merged splits, `DummyClassifier(strategy="prior")` scores balAcc = 0.400 [0.200, 0.500] and AUC = 0.500 [0.500, 0.500]. The primary model at balAcc = 0.456 and AUC = 0.533 is within a fold of that dummy and its CIs overlap. This is not evidence of decoding.

**Reading the native-LOBO diagnostic.** Every native block is single-class, so per-fold AUC is undefined and only the pooled AUC across separately-trained models is a number. On this recording the pooled AUC of the real model is 0.437 and the pooled AUC of `DummyClassifier(strategy="prior")` on the same splits is 0.000. Because the dummy is learning nothing, the sub-chance dummy pooled AUC is a splitting artefact rather than evidence of electrode drift; the same artefact affects any pooled-AUC number on this recording. The native-LOBO row is retained only as a diagnostic for that reason.

The chronological logistic regression at 0.416 beats a stratified dummy by a small margin but sits well below the 0.767 majority-class baseline. The expanding-window mean of 0.380 sits at the level a stratified dummy would reach on a two-class split. Neither is evidence of real generalisation.

Secondary models under leave-one-merged-block-out: SVM-RBF and RandomForest (300); see `reports/results/phase1_results.json` for per-fold rows.

**The leakage trap.** Because the label runs in long blocks, neighboring samples are near-identical and share a label. A random shuffled split scatters those neighbors across train and test, so the model scores well by recognizing near-duplicates it has already seen. Holding everything else constant and changing only how the split is drawn produces the performance gap between the first row and the leakage-safe rows below.

![Phase 1 leakage](reports/figures/fig1_phase1_leakage.png)

**Why native-LOBO and expanding-window AUCs are undefined.** Every contiguous native label block is single-class by construction, so a held-out block contains only eyes-open or only eyes-closed windows. AUC cannot be computed on a single-class test set, and balanced accuracy degenerates into the recall of whichever class the block contains. Native LOBO is retained only as a labelled diagnostic and paired with a DummyClassifier(prior) run on the same splits to make the pooled-metric artefact obvious. **Leave-one-merged-block-out is the primary group protocol; chronological holdout is the primary temporal protocol.**

**Coefficients.** Signed standardized logistic-regression coefficients are stored for each merged-LOBO training fold. The largest mean coefficient in the saved JSON is `F4_beta` (+1.430). The merged-LOBO balanced accuracy is 0.456 [0.270, 0.605], and the same-splits dummy prior is 0.400 [0.200, 0.500]. The two intervals overlap. An earlier README treated a frontal ranking as evidence of eye-movement artifact. That sentence is removed. No eye-artifact regressor or blink annotation is in this repository, and no Berger-effect contrast is computed here.

**Takeaway.** On this one recording, under these splits, the prespecified model does not beat the same-split dummy. The file has one subject.

---

## Phase 2: PhysioNet Motor Imagery (the honest result)

**Dataset.** PhysioNet EEG Motor Movement/Imagery, imagined left versus right fist. 10 subjects, 383 trials across imagery runs R04, R08, and R12; features are mu (8 to 13 Hz) and beta (13 to 30 Hz) band power over a 13-channel sensorimotor strip at 160 Hz (26 features). The same band-power approach as Phase 1, applied to data with many independent trials and multiple subjects.

| Protocol                                    | Balanced accuracy    | Macro F1             | ROC AUC              |
| ------------------------------------------- | -------------------- | -------------------- | -------------------- |
| Naive random trial split (leaky)            | 0.504                | -                    | 0.521                |
| Within-subject shuffled trial CV            | 0.607 [0.522, 0.700] | 0.605 [0.518, 0.698] | 0.632 [0.534, 0.735] |
| Within-subject leave-one-run-out (30 folds) | 0.588 [0.529, 0.648] | 0.556 [0.490, 0.622] | 0.657 [0.598, 0.718] |
| Cross-subject (leave-one-subject-out)       | 0.474 [0.432, 0.516] | 0.427 [0.380, 0.474] | 0.511 [0.447, 0.583] |

Descriptive counts above 0.5 (not significance claims): 8 of 10 subjects on within-subject balanced accuracy, 8 of 10 on within-subject AUC, 19 of 30 folds on leave-one-run-out balanced accuracy, 25 of 30 folds on leave-one-run-out AUC, 2 of 10 subjects on cross-subject balanced accuracy.

Leave-one-run-out balanced accuracy is 0.588 and shuffled-trial balanced accuracy is 0.607. Leave-one-run-out uses the three imagery runs from the same recording. It does not test a later session.

**Within-subject shuffled trial CV vs leave-one-run-out.** The shuffled trial CV is retained because it is the direct comparison to a lot of published within-subject numbers on this dataset. It is optimistic: trials from the same recording run appear in both train and test. The leave-one-run-out (LORO) evaluation added in the 2026-09 audit trains on two of a subject's three imagery runs (R04, R08, R12) and tests on the third, so trials in the test set come from a run the model has not seen. This tests transfer across recording runs of the same session; it does not establish transfer to a new recording session.

**Standard deviation is spread, not uncertainty.** Every per-subject block reports `std_across_subjects` alongside the bootstrap CI. The two answer different questions: `std_across_subjects` is how much subjects differ; the CI is how well the sample mean estimates the population mean. Subject-level bootstrap CIs are computed by resampling subjects, never trials.

**Per-subject cross-subject balanced accuracy**

| S0    | S1    | S2    | S3    | S4    | S5    | S6    | S7    | S8    | S9    |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.545 | 0.471 | 0.381 | 0.600 | 0.487 | 0.353 | 0.500 | 0.496 | 0.449 | 0.461 |

A spread from 0.353 to 0.600 that a single pooled figure hides completely. Standard deviation across subjects: 0.072. Secondary models cross-subject: LogisticRegression (C = 0.5) 0.472 [0.427, 0.514], RandomForest (300) 0.487 [0.444, 0.531].

**Scores.** Within-subject AUC intervals exclude 0.5. The leave-one-subject-out intervals include 0.5. The cross-subject result is the score. This repository does not include a subject-identity classifier, so it does not identify what the cross-subject model failed to use.

![Phase 2 generalization](reports/figures/fig2_phase2_generalization.png)

**Coefficients, same estimator both sides.** Both panels of Figure 3 use signed standardized logistic-regression coefficients from the saved evaluation folds. Phase 1 uses the merged-LOBO folds. Phase 2 uses the per-subject shuffled-trial fits. White dots are the per-fold values. The five largest mean absolute coefficients in the Phase 2 JSON are `C6_beta`, `C1_mu`, `C3_beta`, `C5_mu`, and `C4_mu`. Those names are channels in the motor montage used to build the features. The plot does not test whether the weights match a physiological pattern, and an earlier sentence that said they confirm known physiology is removed.

![Interpretability contrast](reports/figures/fig3_interpretability_contrast.png)

The Phase 1 weights come from a model whose score overlaps the same-split dummy. The Phase 2 weights come from within-subject models whose AUC interval excludes 0.5. No faithfulness test (permute a feature, rescore the held-out fold) is implemented.

---

## CSP + LDA extension (round 4, Task 9)

Round 4 of the audit adds Common Spatial Patterns with Linear Discriminant Analysis as a separately labelled **extension** to Phase 2. It is not a replacement for the band-power result above; both live side by side and are labelled accordingly.

- Same 10 subjects, same eligible trials (imagery runs R04, R08, R12), same evaluation splits, so the numbers are directly comparable to the audited band-power primary.
- Signal filtering: 4th-order Butterworth zero-phase bandpass 8-30 Hz, applied per epoch before CSP. Channel set is the same 13-electrode sensorimotor strip. Epoch window 0.5-3.5 s.
- 6 CSP components with log-variance features, LDA classifier.
- **CSP and every learned transformation are fitted inside training folds only.** The extension never fits CSP on the whole dataset. Same rule as everywhere else in the repo.
- Written to `reports/results/phase2_csp_lda_results.json`. The primary `phase2_results.json` is unchanged.

| Protocol | Model | balAcc | AUC |
|---|---|---|---|
| Within-subject shuffled trial CV | LogReg band-power (primary) | 0.607 [0.522, 0.700] | 0.632 [0.534, 0.735] |
| Within-subject shuffled trial CV | CSP+LDA (extension) | 0.655 [0.543, 0.773] | 0.693 [0.569, 0.819] |
| Within-subject LORO (30 folds) | LogReg band-power (primary) | 0.588 [0.529, 0.648] | 0.657 [0.598, 0.718] |
| Within-subject LORO (30 folds) | CSP+LDA (extension) | 0.640 [0.577, 0.705] | 0.758 [0.675, 0.836] |
| Cross-subject LOSO | LogReg band-power (primary) | 0.474 [0.432, 0.516] | 0.511 [0.447, 0.583] |
| Cross-subject LOSO | CSP+LDA (extension) | 0.541 [0.478, 0.618] | 0.654 [0.556, 0.767] |

CSP plus LDA is the usual motor-imagery baseline (Ramoser, Müller-Gerking, and Pfurtscheller, 2000), not a method introduced here. On these 10 subjects the within-subject CSP balanced-accuracy point estimates are higher than the band-power point estimates (0.655 vs 0.607 shuffled; 0.640 vs 0.588 leave-one-run-out). Cross-subject CSP balanced accuracy is 0.541 [0.478, 0.618], so the interval still includes 0.5. The cross-subject CSP AUC interval is 0.654 [0.556, 0.767]. The JSON records 3 of 10 subjects above 0.5 on balanced accuracy and 9 of 10 above 0.5 on AUC.

---

## Two corrections to earlier versions of this README

**1. Cross-subject AUC is chance, not below chance.** An earlier version reported 0.48 and called it below chance. That figure came from pooling every held-out subject's predicted probabilities into one array and scoring it once. Subjects' decision scores sit on different scales, so pooling manufactures apparent below-chance performance. The subject-level mean is 0.511 [0.447, 0.583]. The conclusion is unchanged, chance either way, but "below chance" was a reporting artifact rather than a finding. The pooled figures remain in `reports/results/phase2_results.json` under `pooled_descriptive_metrics`, tagged as descriptive.

**2. Within-subject decoding is weaker than previously stated.** The per-subject balanced accuracy interval is [0.522, 0.700] and **narrowly excludes 0.5**. The AUC interval [0.534, 0.735] also excludes it, and 8 of 10 subjects sit above 0.5 on both. Leave-one-run-out gives balanced accuracy 0.588 [0.529, 0.648] and AUC 0.657 [0.598, 0.718] across 30 folds. The defensible claim is narrow: within-subject decoding is modestly above chance and consistent across runs of the same session; it is not evidence of transfer to a new recording session.

An earlier version also paired a balanced accuracy from one model with an AUC from another. Every headline metric here comes from the single prespecified model.

**3. The Phase 1 frontal-artifact claim is withdrawn.** It was produced by a different model (RandomForest), a different metric (F1), and globally fitted artifact clipping. Under the prespecified pipeline the top features are mixed, and the underlying model overlaps a same-splits dummy prior, which makes the attribution a descriptive ranking rather than physiological evidence. See the Phase 1 explainability section above.

**4. The two attribution panels used different methods before Round 3.** Round 3 of the audit (Task 6) replaced the previous Figure 3 — Phase 1 permutation importance paired with Phase 2 |coef| from a model fit to all subjects at once — with signed standardized coefficients from real evaluation folds on both sides. The "same attribution method" phrasing in the earlier README was inaccurate; it is correct now.

Full detail in [`REFACTOR_NOTES.md`](REFACTOR_NOTES.md).

---

## A caveat on the inflation story

Phase 2's naive random trial split reaches 0.504 against a cross-subject 0.474. Essentially no gap. Phase 2 uses discrete trials, and random splitting of discrete trials leaks far less than random splitting of Phase 1's continuous windows.

**The dramatic inflation result belongs to Phase 1 and is not a general claim about EEG machine learning.** How much a naive split inflates depends on how much temporal and session structure the epochs share.

## Unified finding

On the UCI file, the random-split balanced accuracy is 0.533 and the chronological holdout is 0.416. On the PhysioNet subset, the random-split balanced accuracy is 0.504 and leave-one-subject-out is 0.474. The large gap between a random split and a blocked split is in Phase 1. Phase 2 does not show that gap. Leave-one-subject-out band-power balanced accuracy includes 0.5. No further cause is identified.

## Repository structure

```
src/
  core/
    features.py             # windowing, artifact clipping, band power; sfreq is always explicit
    evaluation.py           # LOSO / expanding-window / merged-block / within-subject CV,
                            #   coefficient capture, per-split baselines, dummy diagnostic
    statistics.py           # subject-level bootstrap, grouped permutation test
    temporal.py             # feature autocorrelation, gap chooser, walk-forward split generators
    results.py              # deterministic JSON serialisation
  phase1_eyestate.py        # Phase 1: naive / chronological / expanding-window / merged-LOBO + dummy
  phase2_motor_imagery.py   # Phase 2 primary: within-subject shuffled + LORO / cross-subject LOSO
  phase2_csp_lda.py         # Phase 2 extension (round 4, Task 9): CSP + LDA on the same splits
  physionet_features.py     # raw EDF -> feature cache + manifest (validated, no silent skips)
  make_figures.py           # rebuilds three figures from saved JSON only (no model refit)
  make_tables.py            # rebuilds phase1/phase2 markdown tables from the same JSONs
tests/
  test_features.py          # feature extraction, windowing, clipping
  test_evaluation.py        # per-subject reporting, merged_block_groups, persist_predictions, dummy
  test_temporal.py          # autocorrelation diagnostic and expanding-window splits
  test_no_leakage.py        # the project thesis as executable regression tests
  test_physionet_features.py# manifest schema, layout validation, cache invariants
  test_round4_extras.py     # LORO run separation, dummy diagnostic, deterministic bootstrap,
                            #   coefficient capture, result schema, figure smoke test
.github/workflows/
  tests.yml                 # CI: pytest + rebuild figures/tables from JSON on push and PR
reports/
  AUDIT_LOG.md              # 2026-09 audit: pre-audit commit, environment, changes (rounds 1-4)
  NeuroSense_Report.pdf     # write-up (start here)
  NeuroSense_Report.tex     # its LaTeX source
  results/
    phase1_results.json                 # 5 Phase 1 protocols, per-split baselines, dummy prior,
                                        #   per-fold merged-LOBO coefficients
    phase2_results.json                 # LOSO + LORO + shuffled CV, per-fold preds, per-subject coefs
    phase2_csp_lda_results.json         # CSP+LDA extension (round 4), separate from the primary
    tables/                             # markdown tables regenerated by make_tables.py
    archive/pre-audit-<commit>/         # verbatim pre-audit JSONs, never rewritten
  figures/
    fig1_phase1_leakage.png
    fig2_phase2_generalization.png
    fig3_interpretability_contrast.png
    archive/pre-audit-<commit>/         # verbatim pre-audit figures
scripts/
  download_data.sh          # fetches datasets, hard-fails on missing/truncated files
data/processed/
  physionet_features.npz              # committed band-power cache (see licensing note above)
  physionet_features_manifest.json    # SHA-256 of source EDFs, channel order, preprocessing
models/                                # created automatically by the phase scripts
PREREGISTRATION.md          # analysis plan for Phase 3, committed before results
REFACTOR_NOTES.md           # shared-core rebuild + 2026-09 audit notes (rounds 1-4)
```

## Reproduce

Supported Python: **3.9** (audit runs used 3.9.6). The `numpy==2.0.2` pin means Python 3.8 is out. Newer Python versions are likely fine but not verified.

### Path A: cached-features reproduction (fast, no EDF download)

```bash
pip install -r requirements.txt       # MNE is only needed for path B
python src/phase1_eyestate.py          # reads data/raw/EEG Eye State.arff
python src/phase2_motor_imagery.py     # reads data/processed/physionet_features.npz
python src/make_figures.py             # figures from JSON only, no model fit
python src/make_tables.py              # markdown tables from the same JSONs
python -m pytest tests/ -q             # 73 tests
```

`data/processed/physionet_features.npz` and its manifest `data/processed/physionet_features_manifest.json` are committed. The manifest records the SHA-256 of every source EDF the cache was built from, so a Path-B rerun can verify byte-for-byte agreement.

### Path B: full raw-data reproduction (rebuild the cache from EDFs)

```bash
pip install -r requirements.txt        # includes mne==1.8.0
bash scripts/download_data.sh          # ~60 MB of PhysioNet EDFs, refuses to skip missing files
python -m src.physionet_features       # rebuild data/processed/physionet_features.npz + manifest
python src/phase1_eyestate.py
python src/phase2_motor_imagery.py
python -m src.phase2_csp_lda           # CSP+LDA extension (round 4, needs MNE + raw EDFs)
python src/make_figures.py
python src/make_tables.py              # picks up the CSP results automatically
python -m pytest tests/ -q
```

`scripts/download_data.sh` hard-fails on any missing or truncated EDF (below 500 KB per file) rather than silently skipping. `src/physionet_features.py` validates that all 10 expected subjects and all 3 imagery runs per subject (R04, R08, R12) are present, and asserts the sampling rate on every EDF equals the expected 160 Hz.

### What's in the manifest

`data/processed/physionet_features_manifest.json` records, per feature cache:

- subject ids, per-run trial counts, and per-trial ids (`S001_R04_T000`, ...)
- channel order, frequency bands, epoch window (0.5-3.5 s)
- preprocessing (Welch parameters, integration rule, boundary handling)
- source EDF SHA-256 checksums and file sizes

Rerunning `python -m src.physionet_features` on unchanged EDFs produces a byte-identical manifest.

### Reproducibility notes

Every number in the tables above has been reproduced on two independent machines and matches to three decimal places, with one exception: the RandomForest bootstrap interval varies in the third decimal across scikit-learn versions because of tie-breaking in tree construction. Point estimates are identical. Versions are pinned in `requirements.txt`.

The Path-B trial count is 383 (post-audit re-extraction with MNE 1.8). The pre-audit cache had 437 trials because an older MNE version retained more boundary-adjacent epochs after `mne.concatenate_raws` in the feature extractor. The point estimates on the trimmed cache are close to the pre-audit numbers and do not change any substantive conclusion.

`tests/test_no_leakage.py` encodes the project's thesis as executable checks. If a global scaler, a naive split, or an epoch-level confidence interval is reintroduced, a test fails. The 2026-09 audit added `test_held_out_mutation_cannot_change_training_preprocessing`, which mutates held-out rows to 1e12, refits the Phase 1 pipeline, and requires the fitted clipper thresholds, scaler statistics, and transformed training features to be bit-identical to a baseline without the mutation.

### Committed derived artefacts

`data/processed/physionet_features.npz` is a derived feature cache built from the public PhysioNet EEG Motor Movement/Imagery Database (PhysioNet is distributed under the ODC-BY license). The cache contains only per-trial band-power features and label/subject/run/trial-id integers — no raw EEG samples — so it does not constitute redistribution of the underlying recordings. The cache is committed for convenience of reproduction; delete it and rerun Path B to rebuild from raw EDFs.

## Limitations

Phase 1 is a single subject and a single session on consumer-grade hardware, so its results are not subject-generalizable and its high-frequency content is unreliable. Its 100 usable windows make every interval wide, and its single-class label blocks make leave-one-block-out only partially interpretable. Phase 2 uses 10 of the 109 available subjects, so its intervals are wider than they need to be. The two phases also differ in hardware, channel count, and sampling rate, so any comparison between them confounds paradigm with recording setup.

## Future work

Scale Phase 2 to all 109 subjects to tighten the estimates; add common spatial pattern (CSP) features, which are the standard for motor imagery; attempt subject-adaptive transfer, both unsupervised alignment and small-sample calibration, to move cross-subject decoding above chance; and explore deep models only after establishing these leakage-safe baselines. Add explanation-faithfulness checks (permutation of top features vs held-out score drop) and model-randomisation tests to strengthen the attribution reading beyond a descriptive ranking. A third phase on mental arithmetic is planned, with its analysis prespecified in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Data and licensing

Datasets are downloaded by `scripts/download_data.sh` and are not redistributed here. UCI EEG Eye State: UCI Machine Learning Repository (dataset 264). PhysioNet EEG Motor Movement/Imagery Database v1.0.0 (Schalk et al., 2004; Goldberger et al., 2000). Code in this repository is released under the MIT License.
