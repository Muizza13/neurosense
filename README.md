## Reproduce

```bash
pip install -r requirements.txt
bash scripts/download_data.sh      # downloads UCI + PhysioNet (not committed to the repo)
python src/train.py                # Phase 1
python src/evaluate_physionet.py   # Phase 2
python src/make_figures.py         # figures
```

Tested with Python 3.12 on Linux (3.10+ should work). After the datasets are downloaded, the analysis runs in about 2 to 3 minutes.

## Limitations

Phase 1 is a single subject and single session on consumer-grade hardware, so its results are not subject-generalizable and its high-frequency content is unreliable. Phase 2 uses 10 of the 109 available subjects, so the within-subject confidence interval is wider than it needs to be. Windowing produces a small effective sample in Phase 1 (100 windows), which is why variance across folds is reported rather than a single number.

## Future work

Scale Phase 2 to all 109 subjects to tighten the within-subject estimate; add common spatial pattern (CSP) features, which are the standard for motor imagery; attempt subject-adaptive transfer to move cross-subject decoding above chance; and explore deep models only after establishing these leakage-safe baselines.

## Data and licensing

Datasets are downloaded by `scripts/download_data.sh` and are not redistributed here. UCI EEG Eye State: UCI Machine Learning Repository (dataset 264). PhysioNet EEG Motor Movement/Imagery Database v1.0.0 (Schalk et al., 2004; Goldberger et al., 2000). Code in this repository is released under the MIT License.
