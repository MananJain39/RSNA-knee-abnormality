# Execution Flow

This file documents the execution flow, entry points, function call hierarchies, and session changes.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Phase 0: Label Extraction                                      │
│   ├─ scripts/download_labels.py                                 │
│   ├─ scripts/process_labels.py                                  │
│   └─ Output: data/processed/labels_final.csv                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         KAGGLE PLATFORM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Phase 1: DICOM Preprocessing                                   │
│   ├─ notebooks/01_dicom_preprocessing.ipynb                     │
│   ├─ Uses: src/imaging/geometry.py                              │
│   ├─ Uses: src/imaging/preprocessing.py                         │
│   ├─ Uses: src/imaging/slot_builder.py                          │
│   └─ Output: Kaggle dataset (processed volumes)                 │
│                                                                   │
│   Phase 2: Classical Model                                       │
│   ├─ notebooks/02_baseline_training.ipynb                       │
│   ├─ Uses: src/classical/dinov2_backbone.py                     │
│   ├─ Uses: src/classical/attention_head.py                      │
│   └─ Output: Trained model + submission                         │
│                                                                   │
│   Phase 3-5: Fine-tuning & Ensemble                              │
│   └─ notebooks/03_finetuning.ipynb                              │
│                                                                   │
│   Phase 6-10: Quantum ML                                         │
│   └─ notebooks/04_quantum_*.ipynb                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Entry Points

### Phase 0: Label Extraction (Local)
1. `scripts/download_labels.py` - Downloads LLM label datasets from Kaggle
2. `scripts/process_labels.py` - Validates, imputes, computes confidence weights
3. Output: `data/processed/labels_final.csv`

### Phase 1: DICOM Preprocessing (Kaggle)
1. `notebooks/01_dicom_preprocessing.ipynb` - Kaggle notebook
2. Entry: Load DICOM → Process → Save as Kaggle dataset
3. Uses: `src/imaging/geometry.py`, `src/imaging/preprocessing.py`, `src/imaging/slot_builder.py`

### Phase 2: Classical Model (Kaggle)
1. `notebooks/02_baseline_training.ipynb` - Kaggle notebook
2. Entry: Load preprocessed volumes → Train DINOv2 → Save model

---

## Execution Order

```
Phase 0 (Local)
  │
  ├─ Download LLM labels
  ├─ Validate against gold 58
  ├─ Apply selective imputation
  ├─ Compute confidence weights
  └─ Save labels_final.csv
  │
  ▼
Phase 1 (Kaggle)
  │
  ├─ Load DICOM series
  ├─ Physical position sorting
  ├─ Laterality normalization
  ├─ Physical crop (130mm → 336px)
  ├─ Percentile normalization
  ├─ Slot assignment (5 slots)
  ├─ Slice sampling (central band)
  ├─ 2.5D group creation
  └─ Save as Kaggle dataset
  │
  ▼
Phase 2 (Kaggle)
  │
  ├─ Load preprocessed volumes
  ├─ Load labels_final.csv
  ├─ 2.5D input (groups of 3)
  ├─ Frozen DINOv2 backbone
  ├─ Per-diagnosis attention head
  ├─ 5-fold CV with report grouping
  ├─ Confidence-weighted BCE loss
  └─ Rank-mean ensemble
  │
  ▼
Phase 3-5 (Kaggle)
  │
  ├─ Partial unfreezing
  ├─ Weight EMA
  ├─ Multi-resolution inference
  ├─ Specialist models
  └─ Ensemble
  │
  ▼
Phase 6-10 (Kaggle - Quantum)
  │
  ├─ Feature extraction
  ├─ Quantum encoding
  ├─ VQC training
  ├─ Quantum kernels
  ├─ Hybrid models
  └─ Benchmarking
```

---

## Function Call Hierarchies

### scripts/process_labels.py

```
main()
  │
  ├─ load_gold_labels()
  │   └─ Reads train.csv, filters 58 gold studies
  │
  ├─ load_pilkwang()
  │   └─ Reads pilkwang dataset
  │
  ├─ load_stevenleehans()
  │   └─ Reads stevenleehans v4_blend dataset
  │
  ├─ load_barun2104()
  │   └─ Reads barun2104 pseudo labels
  │
  ├─ validate_against_gold() [for each dataset]
  │   ├─ Merges predictions with gold
  │   └─ Computes macro_auc()
  │
  ├─ selective_imputation()
  │   └─ Imputes Synovitis from Effusion
  │       ├─ If effusion > 0.5: synovitis = 0.63
  │       └─ If effusion ≤ 0.5: synovitis = 0.22
  │
  ├─ compute_confidence_weights()
  │   └─ weight = 0.3 if label == 0.5 else (abs(label - 0.5) * 2 + 0.5)
  │
  └─ Save to data/processed/labels_final.csv
```

### notebooks/01_dicom_preprocessing.ipynb

```
Cell 1: Load metadata
  └─ pd.read_csv(train_series.csv)

Cell 2: Geometry utilities
  ├─ compute_slice_normal()
  │   └─ cross(row_cosine, col_cosine)
  │
  ├─ compute_physical_position()
  │   └─ dot(position, normal)
  │
  ├─ sort_by_physical_position()
  │   └─ argsort by physical position
  │
  ├─ determine_laterality()
  │   ├─ compute_image_center()
  │   └─ sign of x-coordinate (with dead zone)
  │
  └─ normalize_laterality()
      └─ fliplr() for left knees (coronal/axial)

Cell 3: Preprocessing functions
  ├─ physical_crop()
  │   └─ Center crop to target_extent_mm
  │
  ├─ resize()
  │   └─ scipy.ndimage.zoom()
  │
  ├─ percentile_normalize_series()
  │   └─ clip to 1st-99th percentile, scale to [0, 1]
  │
  ├─ sample_central_band()
  │   └─ linspace(low, high, n_samples)
  │
  └─ create_25d_groups()
      └─ Stack groups of 3 adjacent slices

Cell 4: Slot definitions
  └─ assign_slot()
      └─ Match plane + fluid to slot name

Cell 5: process_series()
  ├─ Load DICOM files
  ├─ sort_by_physical_position()
  ├─ determine_laterality()
  ├─ normalize_laterality()
  ├─ physical_crop()
  ├─ resize()
  └─ percentile_normalize_series()

Cell 6: Process all studies
  ├─ For each study:
  │   ├─ For each series:
  │   │   ├─ process_series()
  │   │   ├─ sample_central_band()
  │   │   └─ create_25d_groups()
  │   └─ Store in slots dict
  └─ Save as pickle

Cell 7: Compute slot fill rates
  └─ Count slots per study, compute rates
```

---

## Session Changes

### Session 1: 2026-08-15 (Commits 1-9)

**What Changed**:
1. Wired YAML config into src/config.py
2. Added Kaggle label downloader
3. Added label processing pipeline
4. Implemented DICOM geometry utilities
5. Implemented DICOM loader
6. Completed image preprocessing pipeline
7. Added slot builder
8. Created DICOM preprocessing notebook

**Why**:
- Establish foundation for Phase 0 (labels) and Phase 1 (DICOM preprocessing)
- All preprocessing logic implemented locally, will run on Kaggle

**Key Decisions**:
- Use stevenleehans labels (0.89 AUC)
- 5 slots (not 6)
- Physical crop 130mm
- 2.5D input (not 3D)

**Files Modified**:
- src/config.py (YAML loading)
- scripts/download_labels.py (new)
- scripts/process_labels.py (new)
- src/imaging/geometry.py (new)
- src/imaging/dicom_loader.py (implemented stubs)
- src/imaging/preprocessing.py (completed stubs)
- src/imaging/slot_builder.py (new)
- notebooks/01_dicom_preprocessing.ipynb (new)
- tests/test_geometry.py (new)
- tests/test_dicom_loader.py (new)
- tests/test_preprocessing.py (new)
- tests/test_slot_builder.py (new)

**Test Coverage**: 73 tests passing, 2 skipped (torch not installed)

**Next Steps**:
- Run preprocessing notebook on Kaggle
- Implement Phase 2 (classical model)

---

## Quiz Questions (For Major Changes)

Before making major changes, quiz yourself on:

1. **Previous Decisions**: What were the key decisions in the last session?
2. **Rejected Approaches**: What did we try but reject, and why?
3. **Current State**: What's the current execution flow?
4. **Impact**: How will this change affect downstream phases?
5. **Testing**: What tests need to be updated or added?

---

*Last updated: 2026-08-15*
