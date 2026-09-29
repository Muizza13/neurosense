| Protocol | balAcc | AUC |
|---|---|---|
| Naive random split (leaky) | 0.533 | 0.566 |
| Chronological 70/30 (gap=0) | 0.416 | 0.491 |
| Expanding-window (walk-forward) | 0.380 [0.230, 0.545] | undefined (single-class folds) |
| **Merged-LOBO (primary group protocol)** | 0.456 [0.270, 0.605] | 0.533 [0.396, 0.629] |
| Same splits, DummyClassifier(prior) | 0.400 [0.200, 0.500] | 0.500 [0.500, 0.500] |
| Native LOBO (diagnostic, single-class folds) | 0.482 [0.334, 0.630] | undefined |
