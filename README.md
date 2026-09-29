# NeuroSense: Leakage-Aware, Explainable EEG Decoding

**An honest look at when EEG machine learning actually generalizes, and when it only appears to.**

Most introductory EEG classification projects report a high accuracy and stop there. This project asks whether that accuracy is real. Across two datasets and two paradigms, it shows that the headline numbers commonly reported on a popular EEG benchmark are an artifact of how the data is split, quantifies the honest performance under leakage-aware evaluation, and uses model explanations as a validity check rather than decoration.

A full write-up is in [`reports/NeuroSense_Report.pdf`](reports/NeuroSense_Report.pdf).

## Core finding

> Naive evaluation overstates EEG decoding. Under leakage-aware evaluation, EEG band-power features mostly capture session-specific and subject-specific structure that does not generalize across time or across people. Where decoding genuinely works, its explanations match known physiology. Where it does not, the explanations are not evidence of anything, including not evidence of what went wrong.

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

The 2026-09 audit added (a) temporal-gap chronological holdout and expanding-window on Phase 1, (b) leave-one-merged-block-out with a DummyClassifier(prior) diagnostic that replaces single-class native LOBO as the primary group protocol, (c) leave-one-run-out within subject on Phase 2, (d) persisted per-fold labels, predictions, and probabilities in Phase 2 JSON, and (e) per-split baselines throughout. The primary model is the same as before (LogisticRegression, C = 1.0, balanced). See [`REFACTOR_NOTES.md`](REFACTOR_NOTES.md) and [`reports/AUDIT_LOG.md`](reports/AUDIT_LOG.md) for the full change list.

"Above 0.5" is reported as a descriptive count, not a significance claim. Standard deviation across subjects is spread; the bootstrap CI is uncertainty of the mean, computed by resampling subjects or blocks (never trials).

---

## Phase 1: UCI EEG Eye State (the cautionary result)

**Dataset.** A single continuous 117-second recording from **one person**, 14 channels at 128 Hz. The eyes-open/closed label runs in only 24 contiguous blocks (median 3.9 s), 19 of which contain at least one complete window. Several channels contain single-sample electrode pops up to around 700,000 against a 4,000 baseline, clipped before feature extraction using thresholds fitted on training data only. One-second non-overlapping windows yield **100 windows** and 56 band-power features (delta, theta, alpha, beta; gamma is excluded because 30 to 45 Hz on consumer hardware is dominated by muscle activity, not cortex).

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

**Explainability, and its limits.** A pre-modeling check found no clean Berger effect in this recording: occipital alpha does not rise on eye closure, which is the first sign that this is not a decodable eye-state signal.

An earlier version of this project went further and reported that the most class-separating features were frontal, reading that as evidence of eye-movement and blink artifact. That claim does not survive the rebuild, and the audit's re-runs do not restore it either. Signed standardized coefficients extracted from within each merged-LOBO fold (see `src/make_figures.py`; same method as Phase 2) put both frontal and posterior/temporal channels at the top of the mean-|coef| ranking; the single strongest by mean coefficient is `F4_beta`, but per-fold spread is very wide and the model's merged-LOBO balanced accuracy (0.456) overlaps the same-splits DummyClassifier(prior). A high frontal coefficient here is not itself evidence of ocular artifact, and a low central coefficient is not evidence of anything either.

The Phase 1 attribution figure is therefore a **descriptive ranking**, not a physiological finding. An explanation inherits the credibility of the evaluation beneath it, and when the evaluation overlaps a dummy the ranking should be read as descriptive text rather than mined for a mechanistic story. Explanation-faithfulness checks (permutation of labels within blocks, model-randomisation tests) are not implemented in this repo; see [`REFACTOR_NOTES.md`](REFACTOR_NOTES.md) for the future-work list.

**Takeaway.** On this recording and under the leakage-aware protocols implemented here, the widely reported high accuracies on this dataset appear to be an artifact of evaluation design. This is not a claim that every published high-accuracy result on this dataset was wrong — some published pipelines use different features, filters, or evaluation designs — only that under a leakage-safe design the model does not clear a same-split dummy. A single continuous recording from one person on consumer-grade hardware also cannot support subject-generalizable eye-state decoding on its own, which is what motivates moving to Phase 2's multi-subject data.

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

Leave-one-run-out is close to the shuffled-trial CV, which is reassuring: the within-subject signal is consistent across recording runs of the same session. The AUC is actually slightly higher on LORO than on shuffled CV (0.657 vs 0.632), which is within noise given the fold overlap between the two protocols. Leave-one-run-out still does not establish transfer to a new recording session, because all three runs share the electrode montage, cap placement, and skin condition of a single day.

**Within-subject shuffled trial CV vs leave-one-run-out.** The shuffled trial CV is retained because it is the direct comparison to a lot of published within-subject numbers on this dataset. It is optimistic: trials from the same recording run appear in both train and test. The leave-one-run-out (LORO) evaluation added in the 2026-09 audit trains on two of a subject's three imagery runs (R04, R08, R12) and tests on the third, so trials in the test set come from a run the model has not seen. This tests transfer across recording runs of the same session; it does not establish transfer to a new recording session.

**Standard deviation is spread, not uncertainty.** Every per-subject block reports `std_across_subjects` alongside the bootstrap CI. The two answer different questions: `std_across_subjects` is how much subjects differ; the CI is how well the sample mean estimates the population mean. Subject-level bootstrap CIs are computed by resampling subjects, never trials.

**Per-subject cross-subject balanced accuracy**

| S0    | S1    | S2    | S3    | S4    | S5    | S6    | S7    | S8    | S9    |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.545 | 0.471 | 0.381 | 0.600 | 0.487 | 0.353 | 0.500 | 0.496 | 0.449 | 0.461 |

A spread from 0.353 to 0.600 that a single pooled figure hides completely. Standard deviation across subjects: 0.072. Secondary models cross-subject: LogisticRegression (C = 0.5) 0.472 [0.427, 0.514], RandomForest (300) 0.487 [0.444, 0.531].

**Two honest evaluations.** Within a subject, decoding is above chance on AUC but modest, with clear between-subject variability. Across subjects, performance collapses to chance. The chance-level cross-subject result is _consistent with_ features that carry subject-specific information (montage, cap placement, individual mu-rhythm topography) and do not transfer to a new person without calibration; it does not by itself _prove_ that the model is learning subject identity, which would require an explicit subject-id decoding experiment that is not implemented here.

![Phase 2 generalization](reports/figures/fig2_phase2_generalization.png)

**Explainability, same method both sides (Task 6 correction).** Both Phase 1 and Phase 2 now report signed standardized logistic-regression coefficients extracted from within each real evaluation fold (Phase 1: merged-LOBO folds; Phase 2: per-subject models). The white dots on Figure 3 are the per-fold coefficients so the between-fold spread is visible. The Phase 2 mean coefficients place `C6_beta`, `C1_mu`, `C3_beta`, `C5_mu`, and `C4_mu` at the top of the |coef| ranking, sitting on the central sensorimotor strip — **consistent with expected sensorimotor patterns** for imagined left-vs-right fist movement, and consistent enough across subjects that the signs of the top mu features do not flip randomly. That is a validity check on the model, not a proof of a specific physiological mechanism; the audit deliberately stopped short of "confirms known physiology" language.

![Interpretability contrast](reports/figures/fig3_interpretability_contrast.png)

Same method, two very different situations. On the right the ranking is stable across subjects and lands on the expected motor strip. On the left the model overlaps a same-splits dummy prior, so the ranking is presented as **descriptive** — a description of what the coefficients happen to be on this recording, not evidence about ocular artifact or about eye-state physiology in either direction.

**Attribution caveat.** These are model-specific coefficients from one prespecified classifier fit inside real evaluation folds. They report what the model uses; they do not by themselves establish faithfulness (that permuting the top feature would degrade held-out score by a matching amount) or robustness (that a differently regularised model would rank the same features). Randomisation tests and permutation-importance-vs-coefficient agreement checks would strengthen the interpretation and are listed in [`REFACTOR_NOTES.md`](REFACTOR_NOTES.md) as future work.

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

Across both datasets, naive evaluation overstates performance on Phase 1 (dramatically) and marginally on Phase 2. Under leakage-aware evaluation, Phase 1 features from this single-subject continuous recording do not clear a same-split dummy, and Phase 2 features carry structure that supports modest within-subject decoding but does not transfer across people. Whether that non-transfer reflects subject-identity leakage, montage variability, or genuine between-subject differences in motor-imagery neurophysiology is not answered by the present evaluation. Model explanations are usable as descriptive rankings; they warrant a physiological reading only when paired with faithfulness and randomisation checks, which are listed as future work.

## Repository structure

```
src/
  core/
    features.py           # windowing, artifact clipping, band power; sfreq is always explicit
    evaluation.py         # LOSO / expanding-window / merged-block grouping / per-split baselines
    statistics.py         # subject-level bootstrap, grouped permutation test
    temporal.py           # feature autocorrelation, gap chooser, walk-forward split generators
    results.py            # deterministic JSON serialisation
  phase1_eyestate.py      # Phase 1: naive / chronological / expanding-window / merged-LOBO + dummy
  phase2_motor_imagery.py # Phase 2: within-subject shuffled + LORO / cross-subject LOSO
  physionet_features.py   # raw EDF -> feature cache + manifest (validated, no silent skips)
  make_figures.py         # regenerates the three figures from data
tests/
  test_features.py        # feature extraction, windowing, clipping
  test_evaluation.py      # per-subject reporting, merged_block_groups, persist_predictions, dummy
  test_temporal.py        # autocorrelation diagnostic and expanding-window splits
  test_no_leakage.py      # the project thesis as executable regression tests
reports/
  AUDIT_LOG.md            # 2026-09 audit: pre-audit commit, environment, changes (rounds 1-3)
  NeuroSense_Report.pdf   # write-up (start here)
  NeuroSense_Report.tex   # its LaTeX source
  results/
    phase1_results.json   # current, five Phase 1 protocols with per-split baselines + dummy prior
    phase2_results.json   # current, LOSO + LORO + shuffled CV, per-fold preds persisted
    archive/pre-audit-<commit>/  # verbatim pre-audit JSONs, never rewritten
  figures/
    fig1_phase1_leakage.png
    fig2_phase2_generalization.png
    fig3_interpretability_contrast.png
    archive/pre-audit-<commit>/  # verbatim pre-audit figures
scripts/
  download_data.sh        # fetches datasets, hard-fails on missing/truncated files
data/processed/
  physionet_features.npz            # committed feature cache (see licensing note above)
  physionet_features_manifest.json  # SHA-256 of source EDFs, channel order, preprocessing
models/                              # created automatically by the phase scripts
PREREGISTRATION.md        # analysis plan for Phase 3, committed before results
REFACTOR_NOTES.md         # shared-core rebuild + 2026-09 audit notes (rounds 1-3)
```

## Reproduce

Supported Python: **3.9** (audit runs used 3.9.6). The `numpy==2.0.2` pin means Python 3.8 is out. Newer Python versions are likely fine but not verified.

### Path A: cached-features reproduction (fast, no EDF download)

```bash
pip install -r requirements.txt       # MNE is only needed for path B
python src/phase1_eyestate.py          # reads data/raw/EEG Eye State.arff
python src/phase2_motor_imagery.py     # reads data/processed/physionet_features.npz
python src/make_figures.py
python -m pytest tests/ -q             # 49 tests
```

`data/processed/physionet_features.npz` and its manifest `data/processed/physionet_features_manifest.json` are committed. The manifest records the SHA-256 of every source EDF the cache was built from, so a Path-B rerun can verify byte-for-byte agreement.

### Path B: full raw-data reproduction (rebuild the cache from EDFs)

```bash
pip install -r requirements.txt        # includes mne==1.8.0
bash scripts/download_data.sh          # ~60 MB of PhysioNet EDFs, refuses to skip missing files
python -m src.physionet_features       # rebuild data/processed/physionet_features.npz + manifest
python src/phase1_eyestate.py
python src/phase2_motor_imagery.py
python src/make_figures.py
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
