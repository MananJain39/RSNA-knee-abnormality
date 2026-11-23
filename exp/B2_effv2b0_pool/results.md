### B2_effv2b0_pool
- folds: 4 | OOF studies: 4349
- **OOF macro AUC vs LLM labels: 0.8823**
- gold-58 macro AUC per fold: 0.8859, 0.8963, 0.8938, 0.8964 (mean 0.8931)
- **gold-58 macro AUC, 4-fold ensemble: 0.9049**

#### Per-label AUC
|                  |   B2_effv2b0_pool gold(ens) |   B2_effv2b0_pool OOF |
|:-----------------|----------------------------:|----------------------:|
| ACL              |                       0.944 |                 0.919 |
| MCL              |                       0.941 |                 0.886 |
| Medial Meniscus  |                       0.931 |                 0.904 |
| Lateral Meniscus |                       0.815 |                 0.851 |
| Medial OA        |                       0.983 |                 0.911 |
| Lateral OA       |                       0.845 |                 0.868 |
| PF OA            |                       0.837 |                 0.854 |
| Effusion         |                       0.969 |                 0.911 |
| Synovitis        |                       0.787 |                 0.805 |
| Baker's          |                       0.958 |                 0.941 |
| Contusion        |                       0.951 |                 0.861 |
| Fracture         |                       0.897 |                 0.876 |
| MACRO            |                       0.905 |                 0.882 |
