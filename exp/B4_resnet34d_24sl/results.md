### B4_resnet34d_24sl
- folds: 4 | OOF studies: 4349
- **OOF macro AUC vs LLM labels: 0.8847**
- gold-58 macro AUC per fold: 0.8802, 0.8923, 0.8964, 0.8857 (mean 0.8886)
- **gold-58 macro AUC, 4-fold ensemble: 0.9012**

#### Per-label AUC
|                  |   B4_resnet34d_24sl gold(ens) |   B4_resnet34d_24sl OOF |
|:-----------------|------------------------------:|------------------------:|
| ACL              |                         0.961 |                   0.926 |
| MCL              |                         0.907 |                   0.87  |
| Medial Meniscus  |                         0.942 |                   0.912 |
| Lateral Meniscus |                         0.819 |                   0.86  |
| Medial OA        |                         0.98  |                   0.907 |
| Lateral OA       |                         0.83  |                   0.869 |
| PF OA            |                         0.848 |                   0.87  |
| Effusion         |                         0.961 |                   0.916 |
| Synovitis        |                         0.755 |                   0.802 |
| Baker's          |                         0.967 |                   0.94  |
| Contusion        |                         0.942 |                   0.87  |
| Fracture         |                         0.901 |                   0.875 |
| MACRO            |                         0.901 |                   0.885 |
