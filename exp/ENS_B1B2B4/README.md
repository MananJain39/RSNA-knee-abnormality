# Ensemble — B1 + B2 + B4 (3 runs × 4 folds = 12 models)

**Method for evaluation:** rank-mean across runs (each run = mean of its 4 folds on gold; OOF uses each study's
held-out prediction from each run). AUC only depends on ranking, so averaging ranks is the safe way to combine.
**On Kaggle:** `infer.py` averages the 12 models' probabilities (near-identical ranking to rank-mean here).

| | OOF (4,349, vs LLM labels) | Gold-58 (experts) |
|---|---|---|
| B1 ResNet34d | 0.886 | 0.905 |
| B2 EffNetV2-B0 | 0.882 | 0.905 |
| B4 ResNet34d 24-slice | 0.885 | 0.901 |
| **Ensemble** | **0.893** | **0.908** |

Kaggle: B1 alone → public 0.917 / private 0.905. Ensemble submitted as submission #2 (ref 57654093).
