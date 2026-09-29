# Phase 3 preregistration

Committed before any EEGMAT result is observed. Nothing below is edited after
the first Phase 3 run. Deviations are recorded in a "Deviations" section
appended at the bottom, with dates, rather than by rewriting the plan.

## Dataset

PhysioNet EEG During Mental Arithmetic Tasks (EEGMAT), 36 subjects, 23 EEG
channels, 500 Hz. Two recordings per subject: rest (`_1`) and mental arithmetic
(`_2`). Roughly 60 usable seconds per condition, so approximately 15
non-overlapping 4-second epochs per condition per subject.

## Research question

Does mental-arithmetic condition separation transfer to unseen subjects, and if
so, is that transfer attributable to task-related spectral change rather than to
eye condition or low-frequency drift?

## Known confound, stated in advance

Rest is always recorded before arithmetic, for every subject. Condition and
recording order are therefore perfectly confounded by design. No analysis
performed after data collection can separate them. Phase 3 can demonstrate
cross-subject discrimination of two fixed-order recording conditions. It cannot
establish that the discriminating signal is mental workload. All reporting uses
the phrasing "condition discrimination", never "workload detection" and never
"stress detection".

The source paper documents that rest was recorded with eyes closed. It does not
document the eye condition during arithmetic. If arithmetic was eyes-open, then
occipital alpha blocking alone can separate the conditions, and the result would
be an eye-state effect rather than a cognitive one. This is treated as unresolved
and is the reason for the O1/O2 ablation below.

## Hypotheses

- **H1.** Cross-subject condition discrimination on EEGMAT exceeds the
  cross-subject result obtained for Phase 2 motor imagery.
- **H2.** Discrimination survives removal of occipital channels and of the delta
  band, indicating it does not rest solely on eye condition or drift.
- **H3.** Unsupervised per-subject feature centring improves cross-subject
  balanced accuracy relative to zero-shot.
- **H4.** Supervised calibration with a small number of labelled epochs from the
  held-out subject improves on both zero-shot and unsupervised alignment.

H2 is the one that can falsify the interesting reading of H1. If discrimination
collapses without occipital channels or without delta, H1 is reported as
supported but uninformative about cognition.

## Primary analysis

- Epochs: 4 seconds, non-overlapping.
- Features: Welch band power, delta 1 to 4, theta 4 to 8, alpha 8 to 13,
  beta 13 to 30 Hz. Gamma excluded.
- Model: logistic regression, C = 1.0, `class_weight="balanced"`,
  `max_iter=5000`, `random_state=42`. Fixed here, not selected on results.
- Pipeline: `StandardScaler` then the model, fitted inside each fold.
- Evaluation: leave-one-subject-out via `src/core/evaluation.evaluate_loso`.
- **Primary endpoint:** mean per-subject balanced accuracy with a 95 percent
  percentile bootstrap interval resampling the 36 subjects.
- Pooled epoch-level metrics are reported as a descriptive appendix only.
- Null: labels shuffled within subject, 200 permutations, subject grouping
  preserved.

A result is called "transfers" only if the lower bound of the subject-level
bootstrap interval exceeds 0.5.

## Prespecified ablations

Each rerun uses the identical protocol, changing only the feature set.

| Ablation | Purpose |
|---|---|
| all bands, all channels | reference |
| minus delta | drift and impedance artifact |
| minus alpha | eye-condition contribution |
| minus O1, O2 | eye-condition contribution, spatial |
| minus delta and minus O1, O2 | combined |
| frontal channels, theta and beta only | task-related spectral change |

## Secondary analyses

1. **Temporal drift control.** Within each recording alone, first 20 s versus
   final 20 s, 20 s buffer discarded, LOSO. Reported as weak evidence only: it
   probes drift within a recording, while the confound of interest lies between
   two separate recordings.
2. **Calibration curve.** For each held-out subject, 0, 1, 2, 3, 5 labelled
   epochs per condition, 50 resamples per level with fixed seeds, evaluated only
   on that subject's remaining epochs. Three lines: zero-shot, unsupervised
   centring, supervised retraining. Not more than 5 epochs per condition, since
   approximately 15 exist.

## What will not be done

- No three-class relaxed/focused/stressed construction from merged datasets.
- No EEGNet as a headline comparison at this sample size.
- No epoch-level confidence intervals.
- No claim that a highly ranked feature identifies a neural mechanism.
- No subgroup claim from the 24 versus 12 good and poor counter split. If
  reported at all, it is labelled exploratory and underpowered.

## Reporting commitment

All four hypotheses are reported with their outcome regardless of direction. A
null result for H1 is reported as the headline with the same prominence a
positive result would receive.

## Deviations

None recorded yet.
