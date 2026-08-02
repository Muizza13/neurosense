# NeuroSense: Leakage-Aware, Explainable EEG Decoding

**An honest look at when EEG machine learning actually generalizes, and when it only appears to.**

Most introductory EEG classification projects report a high accuracy and stop there. This project asks whether that accuracy is real. Across two datasets and two paradigms, it shows that the headline numbers commonly reported on a popular EEG benchmark are an artifact of how the data is split, quantifies the honest performance under leakage-aware evaluation, and uses model explanations as a validity check rather than decoration.

A full write-up is in [`reports/NeuroSense_Report.pdf`](reports/NeuroSense_Report.pdf).

## Core finding

> Naive evaluation overstates EEG decoding. Under leakage-aware evaluation, EEG band-power features mostly capture session-specific and subject-specific structure that does not generalize across time or across people. Where decoding genuinely works, its explanations match known physiology. Where it does not, the explanations are not evidence of anything, including not evidence of what went wrong.

## How to read the numbers

Both phases run through a shared core (`src/core/`) with one band definition, preprocessing fitted inside each fold, and one prespecified model throughout: logistic regression, C = 1.0, `class_weight="balanced"`, `max_iter=5000`, `random_state=42`. Other models appear only in secondary tables.

**The unit of inference is the subject, or in Phase 1 the label block, never the epoch.** Every interval below is a 95 percent percentile bootstrap resampling subjects or blocks. Pooled epoch-level figures are recorded in the results JSON as descriptive only, because pooling correlated epochs and treating them as independent both understates uncertainty and distorts threshold-free metrics.

All numbers are produced by the code in this repo and stored in `reports/results/`. Nothing is rounded up.

## Results at a glance

| Setting                 | Evaluation                            | Result                                                                       | Reading             |
| ----------------------- | ------------------------------------- | ---------------------------------------------------------------------------- | ------------------- |
| Phase 1 (eye state)     | Naive random window split             | balAcc = 0.533                                                               | leaked              |
| Phase 1 (eye state)     | Chronological holdout                 | balAcc = 0.416, majority baseline 0.767                                      | below chance        |
| Phase 1 (eye state)     | Leave-one-block-out                   | balAcc = 0.482 [0.334, 0.630]                                                | chance, very wide   |
| Phase 1 (eye state)     | Cross-subject                         | not available                                                                | one subject         |
| Phase 2 (motor imagery) | Naive random trial split              | balAcc = 0.540                                                               | mildly leaked       |
| Phase 2 (motor imagery) | Within-subject (trial CV)             | balAcc = 0.580 [0.493, 0.675], AUC = 0.624 [0.536, 0.721], 8/10 above chance | modest, real on AUC |
| Phase 2 (motor imagery) | Cross-subject (leave-one-subject-out) | balAcc = 0.482 [0.429, 0.539], AUC = 0.520 [0.439, 0.608]                    | chance              |

---

## Phase 1: UCI EEG Eye State (the cautionary result)

**Dataset.** A single continuous 117-second recording from **one person**, 14 channels at 128 Hz. The eyes-open/closed label runs in only 24 contiguous blocks (median 3.9 s), 19 of which contain at least one complete window. Several channels contain single-sample electrode pops up to around 700,000 against a 4,000 baseline, clipped before feature extraction using thresholds fitted on training data only. One-second non-overlapping windows yield **100 windows** and 56 band-power features (delta, theta, alpha, beta; gamma is excluded because 30 to 45 Hz on consumer hardware is dominated by muscle activity, not cortex).

Every Phase 1 number rests on 100 observations from one person. That single fact drives the width of every interval below and is the main reason this phase is a cautionary tale rather than a result.

| Protocol                          | Balanced accuracy    | Macro F1             | ROC AUC   |
| --------------------------------- | -------------------- | -------------------- | --------- |
| Naive random window split (leaky) | 0.533                | 0.533                | 0.566     |
| Chronological 70/30 holdout       | 0.416                | 0.330                | 0.491     |
| Leave-one-block-out               | 0.482 [0.334, 0.630] | 0.369 [0.239, 0.515] | undefined |

Majority-class accuracy on the chronological test segment is **0.767**. Balanced accuracy of 0.416 sits below chance and far below that baseline.

Secondary models under leave-one-block-out: SVM-RBF 0.534 [0.376, 0.686], RandomForest (300) 0.487 [0.328, 0.648].

**The leakage trap.** Because the label runs in long blocks, neighboring samples are near-identical and share a label. A random shuffled split scatters those neighbors across train and test, so the model scores well by recognizing near-duplicates it has already seen. Holding everything else constant and changing only how the split is drawn produces the entire performance gap.

![Phase 1 leakage](reports/figures/fig1_phase1_leakage.png)

**Why leave-one-block-out AUC is undefined.** Every contiguous label block is single-class by construction, so a held-out block contains only eyes-open or only eyes-closed windows. AUC cannot be computed on a single-class test set, and balanced accuracy degenerates into the recall of whichever class the block contains. Leave-one-block-out is therefore reported as a supporting check; **chronological holdout is the more interpretable leakage-safe protocol for this phase.**

**Explainability, and its limits.** A pre-modeling check found no clean Berger effect in this recording: occipital alpha does not rise on eye closure, which is the first sign that this is not a decodable eye-state signal.

An earlier version of this project went further and reported that the most class-separating features were frontal, reading that as evidence of eye-movement and blink artifact. That claim does not survive the rebuild. Under the prespecified pipeline the top ten features are 6 frontal and 4 posterior or temporal, and the single strongest is `T7_alpha`.

More fundamentally, the model these attributions describe scores **0.416 balanced accuracy, below chance**. Permutation importance on a model that does not work is not evidence about physiology or about artifact. The honest statement is that Phase 1 produces no trustworthy explanation at all, which is itself the point: an explanation inherits the credibility of the evaluation beneath it, and here there is none to inherit.

**Takeaway.** The widely reported high accuracies on this dataset are an artifact of evaluation design. A single continuous recording from one person cannot support generalizable eye-state decoding. This motivates moving to data built for generalization.

---

## Phase 2: PhysioNet Motor Imagery (the honest result)

**Dataset.** PhysioNet EEG Motor Movement/Imagery, imagined left versus right fist. 10 subjects, 437 trials, features are mu (8 to 13 Hz) and beta (13 to 30 Hz) band power over a 13-channel sensorimotor strip at 160 Hz (26 features). The same band-power approach as Phase 1, applied to data with many independent trials and multiple subjects.

| Protocol                              | Balanced accuracy    | Macro F1             | ROC AUC              |
| ------------------------------------- | -------------------- | -------------------- | -------------------- |
| Naive random trial split (leaky)      | 0.540                | -                    | 0.525                |
| Within-subject (per-subject trial CV) | 0.580 [0.493, 0.675] | 0.578 [0.490, 0.674] | 0.624 [0.536, 0.721] |
| Cross-subject (leave-one-subject-out) | 0.482 [0.429, 0.539] | 0.431 [0.380, 0.487] | 0.520 [0.439, 0.608] |

Subjects above chance: 8 of 10 on within-subject balanced accuracy, 9 of 10 on within-subject AUC, 4 of 10 cross-subject.

**Per-subject cross-subject balanced accuracy**

| S0    | S1    | S2    | S3    | S4    | S5    | S6    | S7    | S8    | S9    |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.637 | 0.442 | 0.384 | 0.608 | 0.443 | 0.387 | 0.522 | 0.546 | 0.365 | 0.482 |

A spread from 0.365 to 0.637 that a single pooled figure hides completely. Secondary models cross-subject: LogisticRegression (C = 0.5) 0.489 [0.437, 0.547], RandomForest (300) 0.487 [0.452, 0.523].

**Two honest evaluations.** Within a subject, decoding is above chance on AUC but modest, with clear between-subject variability. Across subjects, performance collapses to chance: the features are subject-specific and do not transfer to a new person without calibration.

![Phase 2 generalization](reports/figures/fig2_phase2_generalization.png)

**Explainability as a validity check, again.** The within-subject model keys on the C3 versus C4 mu rhythm, the lateralized sensorimotor pattern expected from contralateral desynchronization during motor imagery. Here the explanation _confirms_ known physiology. The contrast with Phase 1 is the centerpiece of the project:

![Interpretability contrast](reports/figures/fig3_interpretability_contrast.png)

Same interpretability method, two very different situations. On the right it corroborates known physiology on a model that genuinely decodes. On the left it describes a model that performs below chance, so the ranking it produces should not be read as a finding.

---

## Two corrections to earlier versions of this README

**1. Cross-subject AUC is chance, not below chance.** An earlier version reported 0.48 and called it below chance. That figure came from pooling every held-out subject's predicted probabilities into one array and scoring it once. Subjects' decision scores sit on different scales, so pooling manufactures apparent below-chance performance. The subject-level mean is 0.520 [0.439, 0.608]. The conclusion is unchanged, chance either way, but "below chance" was a reporting artifact rather than a finding. The pooled figures remain in `reports/results/phase2_results.json` under `pooled_descriptive_metrics`, tagged as descriptive.

**2. Within-subject decoding is weaker than previously stated.** The per-subject balanced accuracy interval is [0.493, 0.675] and **includes 0.5**. The AUC interval [0.536, 0.721] does exclude it, and 8 of 10 subjects sit above chance. The defensible claim is narrower than "real decoding": within-subject _ranking_ is above chance, within-subject _thresholded accuracy_ is not clearly so.

An earlier version also paired a balanced accuracy from one model with an AUC from another. Every headline metric here comes from the single prespecified model.

**3. The Phase 1 frontal-artifact claim is withdrawn.** It was produced by a different model (RandomForest), a different metric (F1), and globally fitted artifact clipping. Under the prespecified pipeline the top features are mixed, and the underlying model scores below chance, which makes the attribution uninterpretable either way. See the Phase 1 explainability section above.

Full detail in [`REFACTOR_NOTES.md`](REFACTOR_NOTES.md).

---

## A caveat on the inflation story

Phase 2's naive random trial split reaches 0.540 against a cross-subject 0.482. That is a small gap. Phase 2 uses discrete trials, and random splitting of discrete trials leaks far less than random splitting of Phase 1's continuous windows.

**The dramatic inflation result belongs to Phase 1 and is not a general claim about EEG machine learning.** How much a naive split inflates depends on how much temporal and session structure the epochs share.

## Unified finding

Across both datasets, naive evaluation overstates performance. Leakage-aware evaluation reveals that EEG band-power features capture session and time structure (Phase 1) and subject identity (Phase 2), neither of which is the intended target. Honest within-subject motor-imagery decoding is achievable but modest, and its explanations align with sensorimotor physiology. Model explanations are trustworthy as evidence only when the evaluation underneath them is leakage-free, and where the evaluation shows the model failing, the explanation should be reported as uninterpretable rather than mined for a story.

## Repository structure

```
src/
  core/
    features.py           # windowing, artifact clipping, band power; sfreq is always explicit
    evaluation.py         # leave-one-group-out with per-subject metric rows
    statistics.py         # subject-level bootstrap, grouped permutation test
    results.py            # deterministic JSON serialisation
  phase1_eyestate.py      # Phase 1: naive / chronological / leave-one-block-out
  phase2_motor_imagery.py # Phase 2: within-subject and cross-subject
  make_figures.py         # regenerates the three figures from data
tests/
  test_features.py        # feature extraction, windowing, clipping
  test_evaluation.py      # per-subject reporting structure
  test_no_leakage.py      # the project thesis as executable regression tests
reports/
  NeuroSense_Report.pdf   # write-up (start here)
  NeuroSense_Report.tex   # its LaTeX source
  results/                # phase1_results.json, phase2_results.json
  figures/                # the three figures above
scripts/
  download_data.sh        # fetches both datasets (data is not stored in the repo)
data/processed/           # cached PhysioNet feature matrix (small)
PREREGISTRATION.md        # analysis plan for Phase 3, committed before results
REFACTOR_NOTES.md         # what changed in the shared-core rebuild and why
```

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/download_data.sh          # downloads UCI + PhysioNet (not committed to the repo)
python src/phase1_eyestate.py          # Phase 1
python src/phase2_motor_imagery.py     # Phase 2
python src/make_figures.py             # figures
python -m pytest tests/ -q             # 33 tests
```

Every number above has been reproduced on two independent machines and matches to three decimal places, with one exception: the RandomForest bootstrap interval varies in the third decimal across scikit-learn versions because of tie-breaking in tree construction. Point estimates are identical. Versions are pinned in `requirements.txt`.

`tests/test_no_leakage.py` encodes the project's thesis as executable checks. If a global scaler, a naive split, or an epoch-level confidence interval is reintroduced, a test fails.

## Limitations

Phase 1 is a single subject and a single session on consumer-grade hardware, so its results are not subject-generalizable and its high-frequency content is unreliable. Its 100 usable windows make every interval wide, and its single-class label blocks make leave-one-block-out only partially interpretable. Phase 2 uses 10 of the 109 available subjects, so its intervals are wider than they need to be. The two phases also differ in hardware, channel count, and sampling rate, so any comparison between them confounds paradigm with recording setup.

## Future work

Scale Phase 2 to all 109 subjects to tighten the estimates; add common spatial pattern (CSP) features, which are the standard for motor imagery; attempt subject-adaptive transfer, both unsupervised alignment and small-sample calibration, to move cross-subject decoding above chance; and explore deep models only after establishing these leakage-safe baselines. A third phase on mental arithmetic is planned, with its analysis prespecified in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Data and licensing

Datasets are downloaded by `scripts/download_data.sh` and are not redistributed here. UCI EEG Eye State: UCI Machine Learning Repository (dataset 264). PhysioNet EEG Motor Movement/Imagery Database v1.0.0 (Schalk et al., 2004; Goldberger et al., 2000). Code in this repository is released under the MIT License.
