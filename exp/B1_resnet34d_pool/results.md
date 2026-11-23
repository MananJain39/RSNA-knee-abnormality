# B1 — Results

| Metric | Value | How to read it |
|---|---|---|
| **Gold-58 macro AUC, 4-fold ensemble** | **0.9052** | Honest test vs expert labels; ensemble = average of the 4 fold models (what Kaggle gets). |
| Gold-58 macro AUC, single fold models | 0.8880, 0.8911, 0.8987, 0.8859 (mean 0.8909) | Ensembling adds ~+0.014. |
| OOF macro AUC vs LLM labels | 0.8858 | Every one of 4,349 studies scored by the model that didn't train on it; vs *noisy* labels (binarised at 0.5). |
| Best val AUC per fold | 0.8898, 0.8842, 0.8844, 0.8888 | Spread across folds is small ⇒ stable training. |
| Training time | ~22 min (4 folds in parallel, 1 L40S each) | |

## Per-label AUC
| Label | Gold-58 (ensemble) | OOF (vs LLM labels) | LLM labels vs gold (exp 01) |
|---|---|---|---|
| ACL | 0.968 | 0.920 | 0.974 |
| MCL | 0.909 | 0.880 | 0.974 |
| Medial Meniscus | 0.936 | 0.910 | 0.920 |
| Lateral Meniscus | 0.816 | 0.870 | 0.869 |
| Medial OA | 0.978 | 0.911 | 0.923 |
| Lateral OA | 0.847 | 0.872 | 0.841 |
| PF OA | 0.869 | 0.864 | 0.887 |
| Effusion | 0.979 | 0.917 | 0.827 |
| Synovitis | 0.760 | 0.802 | 0.777 |
| Baker's | 0.966 | 0.944 | 0.930 |
| Contusion | 0.937 | 0.867 | 0.814 |
| Fracture | 0.897 | 0.873 | 0.777 |
| **Macro** | **0.905** | **0.886** | 0.876 |

## Interpretation
- The image model beats its own training labels against the experts (0.905 vs 0.876). Training on ~3,300 noisy
  labels averages out individual report errors — especially for findings radiologists describe inconsistently
  (Effusion 0.83 → 0.98, Fracture 0.78 → 0.90, Contusion 0.81 → 0.94).
- Weakest: Synovitis (0.76; hard to see without contrast, and labels are weak) and Lateral Meniscus (0.82).
- Gold-58 is small: a single label's AUC can move ±0.05 by chance; macro ±~0.02. Reference: 18th-place public
  write-up had local CV 0.854 → LB 0.903.
