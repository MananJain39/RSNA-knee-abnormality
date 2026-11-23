### B1_resnet34d_pool
- folds: 4 | OOF studies: 4349
- **OOF macro AUC vs LLM labels: 0.8858**
- gold-58 macro AUC per fold: 0.8880, 0.8911, 0.8987, 0.8859 (mean 0.8909)
- **gold-58 macro AUC, 4-fold ensemble: 0.9052**

### B2_effv2b0_pool
- folds: 4 | OOF studies: 4349
- **OOF macro AUC vs LLM labels: 0.8823**
- gold-58 macro AUC per fold: 0.8859, 0.8963, 0.8938, 0.8964 (mean 0.8931)
- **gold-58 macro AUC, 4-fold ensemble: 0.9049**

### B4_resnet34d_24sl
- folds: 4 | OOF studies: 4349
- **OOF macro AUC vs LLM labels: 0.8847**
- gold-58 macro AUC per fold: 0.8802, 0.8923, 0.8964, 0.8857 (mean 0.8886)
- **gold-58 macro AUC, 4-fold ensemble: 0.9012**

### Rank-mean ensemble of 3 runs
- OOF: 0.8934
- **gold-58: 0.9076**

#### Per-label AUC
|                  |   B1_resnet34d_pool gold(ens) |   B1_resnet34d_pool OOF |   B2_effv2b0_pool gold(ens) |   B2_effv2b0_pool OOF |   B4_resnet34d_24sl gold(ens) |   B4_resnet34d_24sl OOF |   ENSEMBLE gold |   ENSEMBLE OOF |
|:-----------------|------------------------------:|------------------------:|----------------------------:|----------------------:|------------------------------:|------------------------:|----------------:|---------------:|
| ACL              |                         0.968 |                   0.92  |                       0.944 |                 0.919 |                         0.961 |                   0.926 |           0.961 |          0.933 |
| MCL              |                         0.909 |                   0.88  |                       0.941 |                 0.886 |                         0.907 |                   0.87  |           0.925 |          0.891 |
| Medial Meniscus  |                         0.936 |                   0.91  |                       0.931 |                 0.904 |                         0.942 |                   0.912 |           0.942 |          0.917 |
| Lateral Meniscus |                         0.816 |                   0.87  |                       0.815 |                 0.851 |                         0.819 |                   0.86  |           0.829 |          0.875 |
| Medial OA        |                         0.978 |                   0.911 |                       0.983 |                 0.911 |                         0.98  |                   0.907 |           0.982 |          0.916 |
| Lateral OA       |                         0.847 |                   0.872 |                       0.845 |                 0.868 |                         0.83  |                   0.869 |           0.838 |          0.877 |
| PF OA            |                         0.869 |                   0.864 |                       0.837 |                 0.854 |                         0.848 |                   0.87  |           0.856 |          0.874 |
| Effusion         |                         0.979 |                   0.917 |                       0.969 |                 0.911 |                         0.961 |                   0.916 |           0.973 |          0.92  |
| Synovitis        |                         0.76  |                   0.802 |                       0.787 |                 0.805 |                         0.755 |                   0.802 |           0.769 |          0.811 |
| Baker's          |                         0.966 |                   0.944 |                       0.958 |                 0.941 |                         0.967 |                   0.94  |           0.967 |          0.948 |
| Contusion        |                         0.937 |                   0.867 |                       0.951 |                 0.861 |                         0.942 |                   0.87  |           0.947 |          0.874 |
| Fracture         |                         0.897 |                   0.873 |                       0.897 |                 0.876 |                         0.901 |                   0.875 |           0.903 |          0.885 |
| MACRO            |                         0.905 |                   0.886 |                       0.905 |                 0.882 |                         0.901 |                   0.885 |           0.908 |          0.893 |
