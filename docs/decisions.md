# Decision Log

This file logs all meaningful architectural, design, and implementation decisions made throughout the project.

---

## Phase 0: Label Extraction

### Decision 1: Use stevenleehans v4_blend labels
**Date**: 2026-08-15  
**Context**: Multiple public LLM label datasets were available on Kaggle.  
**Options Considered**:
- pilkwang/rsna-knee-llm-labels (Macro AUC: 0.8700)
- stevenleehans/rsna-knee-llm-report-labels v4_blend (Macro AUC: 0.8927)
- barun2104/rsna-knee-stratified-folds-and-llm-soft-labels (Macro AUC: 1.0000, but pseudo labels = gold labels for 58 studies)

**Decision**: Use stevenleehans v4_blend  
**Rationale**: 
- Highest AUC among non-trivial datasets (0.8927 vs 0.8700)
- Contains graded probabilities (not binary), enabling confidence weighting
- barun2104 was excluded because its pseudo labels for gold studies exactly match gold labels (AUC=1.0 is trivial)
- stevenleehans provides fine-grained probabilities (0.01 to 1.0 in 0.005 increments), allowing nuanced training signals

**Rejected Approaches**:
- pilkwang: Lower AUC (0.8700), fewer probability levels (0.08, 0.28, 0.68, 0.82, 0.94)
- barun2104: Trivial AUC=1.0 (pseudo labels for gold studies = gold labels), not a real improvement

**Impact**: Better label quality → higher model performance ceiling

---

### Decision 2: Apply selective imputation for Synovitis from Effusion
**Date**: 2026-08-15  
**Context**: Some labels have "not addressed" rates near 0% (stevenleehans), but community research shows certain findings benefit from imputation.  
**Options Considered**:
- No imputation (use raw LLM labels)
- Impute all "not addressed" from correlated findings
- Selective imputation (only for findings where silence is uninformative)

**Decision**: Selective imputation for Synovitis only  
**Rationale**:
- Community research (RESEARCH_SUMMARY.md §3) shows:
  - Baker's: silence = absent (3% positive when silent vs 44% when spoken)
  - Synovitis: silence = uninformative (34% positive when silent vs 76% when spoken)
- P(synovitis | effusion) = 0.63 vs P(synovitis | no effusion) = 0.22
- Imputing synovitis from effusion improves AUC by +0.056
- Imputing Baker's hurts AUC by -0.029

**Rejected Approaches**:
- No imputation: Misses +0.056 AUC improvement on Synovitis
- Impute all: Hurts Baker's by -0.029 AUC (silence = absent for Baker's)

**Impact**: +0.056 AUC improvement on Synovitis label

---

### Decision 3: Compute confidence weights based on label certainty
**Date**: 2026-08-15  
**Context**: Not all labels are equally certain. "Not addressed" (0.5) should have less influence than explicit 0.0 or 1.0.  
**Formula**:
```
weight = 0.3 if label == 0.5 else (abs(label - 0.5) * 2 + 0.5)
```

This gives:
- label = 0.5 → weight = 0.3 (low confidence)
- label = 0.0 or 1.0 → weight = 1.0 (high confidence)
- label = 0.25 or 0.75 → weight = 0.5 (medium confidence)

**Rationale**: Reports that explicitly address a finding should pull harder on that label than reports that are silent. This prevents training every silence as a confident negative.

**Rejected Approaches**:
- Uniform weights: Treats all labels equally, even uncertain ones
- Binary confidence (0 or 1): Loses nuance from graded probabilities

**Impact**: More stable training, better handling of uncertain labels

---

## Phase 1: DICOM Preprocessing

### Decision 4: Use 5 slots instead of 6
**Date**: 2026-08-15  
**Context**: Community research compared 5-slot vs 6-slot designs.  
**Options Considered**:
- 5 slots: Sagittal_Fluid, Sagittal_Structural, Coronal_Fluid, Coronal_Structural, Axial_Fluid
- 6 slots: Add Axial_Structural

**Decision**: Use 5 slots  
**Rationale**:
- Axial_Structural exists for only 19.4% of studies (857/4407)
- A 6th slot would be padded zeros for 80% of the corpus
- That encoding budget is better spent as slices in slots that always fill
- 5-slot design achieves 92.9% mean fill rate vs 6-slot's ~77%

**Rejected Approaches**:
- 6 slots: Poor fill rate (77% vs 92.9%), wastes encoding budget on Axial_Structural

**Impact**: Better memory efficiency, higher fill rates

---

### Decision 5: Physical crop to 130mm before resize
**Date**: 2026-08-15  
**Context**: MRI scans have varying physical scales (PixelSpacing varies 3.4× across corpus).  
**Options Considered**:
- Resize directly to 336px (loses small pathology)
- Crop to 130mm, then resize to 336px (consistent physical scale)
- Crop to 160mm, then resize (risks no-op if FOV < 160mm)

**Decision**: Crop to 130mm  
**Rationale**:
- 130mm is smaller than the smallest field of view in the dataset
- Ensures the crop is never a no-op
- At 336px resolution: 0.387 mm/pixel
- Can resolve 1-3mm meniscal tears (which occupy 2.5-7.7 pixels)
- If we used 160mm, some studies with smaller FOV would skip the crop entirely

**Rejected Approaches**:
- Direct resize: Inconsistent physical scale, loses small pathology
- 160mm crop: Risks no-op for studies with FOV < 160mm

**Impact**: Consistent physical scale across all studies, small pathology preserved

---

### Decision 6: Normalize laterality using image center, not corner
**Date**: 2026-08-15  
**Context**: 5 of 12 labels distinguish medial from lateral. Left/right knees must be normalized.  
**Options Considered**:
- Use ImagePositionPatient corner (unreliable near origin)
- Use image center geometry (robust)
- Use Laterality DICOM tag (missing for some studies)

**Decision**: Use image center with dead zone  
**Rationale**:
- Image center = IPP + row_cosine × col_spacing × (Rows/2) + col_cosine × row_spacing × (Columns/2)
- Laterality = sign of center's x-coordinate in patient space
- Dead zone: within 25mm of origin, sign is no better than chance
- In dead zone: fall back to Laterality tag if present, else default to "R"
- Without this correction, agreement with tag drops from 96.9% to 58.8%

**Rejected Approaches**:
- Corner-based: Unreliable near origin (within 25mm)
- Tag-only: Missing for some studies

**Impact**: Correct medial/lateral orientation for 5 labels

---

### Decision 7: Per-series percentile normalization (1st-99th)
**Date**: 2026-08-15  
**Context**: MRI has no absolute intensity scale. Same tissue appears different on different sequences.  
**Options Considered**:
- Per-slice min/max normalization (one bright vessel compresses everything)
- Per-series min/max (better, but outliers still problematic)
- Per-series 1st-99th percentile (robust to outliers)

**Decision**: Per-series 1st-99th percentile  
**Rationale**:
- Normalization must be over the whole sampled stack (preserves inter-slice contrast)
- Percentiles are robust to outliers (one bright vessel, one dark artifact)
- 1st-99th captures the dynamic range without extreme values
- Per-slice normalization destroys relative contrast between slices

**Rejected Approaches**:
- Per-slice min/max: Destroys inter-slice contrast, sensitive to outliers
- Per-series min/max: Still sensitive to extreme outliers

**Impact**: Preserves inter-slice contrast, robust to outliers

---

### Decision 8: 2.5D input (groups of 3 adjacent slices) instead of 3D
**Date**: 2026-08-15  
**Context**: DINOv2 expects 3-channel RGB-like input.  
**Options Considered**:
- Stack N slices as channels, average first conv weights (destroys pretrained interface)
- True 3D CNN (memory-intensive, no pretrained weights)
- 2.5D: groups of 3 adjacent slices as 3-channel input (preserves pretrained weights)

**Decision**: 2.5D with group_size=3  
**Rationale**:
- DINOv2 was trained on 2D images with 3-channel input
- Stacking 12 slices and averaging first conv weights destroys the input interface
- 2.5D preserves the pretrained representation
- Groups of 3 provide local context (adjacent slices) without memory blowup
- At inference: average logits across all groups (acts as ensemble)

**Rejected Approaches**:
- Stacking N slices: Destroys pretrained interface
- True 3D CNN: No pretrained weights, memory-intensive

**Impact**: Leverages pretrained DINOv2 weights, memory-efficient

---

### Decision 9: Sample from central band (12-88% of stack)
**Date**: 2026-08-15  
**Context**: Outer slices are mostly soft tissue outside the joint.  
**Options Considered**:
- Sample all slices (includes irrelevant outer slices)
- Sample central 50% (may miss pathology near edges)
- Sample central 12-88% (sagittal) or 20-80% (coronal/axial)

**Decision**: Sample central band with plane-specific bounds  
**Rationale**:
- Sagittal: 12-88% (joint spans most of the stack)
- Coronal/Axial: 20-80% (joint is more centralized)
- Outer slices are mostly soft tissue, fat, muscle — not joint pathology
- Sampling from central band focuses on diagnostically relevant regions

**Rejected Approaches**:
- All slices: Includes noise from outer slices
- Central 50%: May miss pathology near edges

**Impact**: Focuses model on joint pathology, reduces noise

---

## Phase 2: Classical Model (Pending)

*Decisions will be logged as Phase 2 is implemented.*

---

## Phase 2a: Hierarchical Clustering + DINOv2 (2026-08-17)

### Decision 10: Study-level 80/20 split with report-grouping
**Context**: The plan.txt proposed splitting data 80/20, but train.csv has 131 studies sharing verbatim reports with other studies (top duplicate is a Turkish normal-knee template shared by 37 studies). A naive random split would leak identical report content across train/val and let the model memorize report text rather than learn from MRI.  
**Options Considered**:
- Random study-level split (leaks via duplicate reports)
- Stratified split by gold/not-gold (forgets report duplicates)
- GroupShuffleSplit on normalized report text + reserve gold-to-val (chosen)

**Decision**: GroupShuffleSplit on whitespace-normalized, case-folded report text, with all 58 gold studies moved to val.  
**Rationale**:
- Normalized grouping puts duplicate reports on the same side → no leak
- Gold studies (the only ground truth) reserved to val → benchmarking is meaningful
- Test: `test_real_split_no_overlap_no_leak` confirms 0 reports span both splits, all 58 gold on val
- Real-data split at seed=42, val_frac=0.20: train=3489 (no gold), val=918 (58 gold + 860 unlabeled)

**Rejected Approaches**:
- Random: would emit ~3-15 leaked report groups per run
- K-fold: not yet needed; this 80/20 holds the gold 58 fixed for benchmarking

**Impact**: Safe benchmarking on gold 58, no false AUC from report memorization.

---

### Decision 11: Cluster on raw acquisition features (PixelSpacing / slice count / FOV / syntax)
**Context**: `plan.txt` proposed "sorting the resolution issue" via clustering. Three resolutions of the word "resolution" were considered.  
**Options Considered**:
- Post-crop 336×336 constant: constant input — nothing to cluster on
- Report-text length: no clear signal vs labels
- Raw acquisition features: chosen

**Decision**: Cluster on `pixel_spacing`, `rows`, `cols`, `n_slices`, `slice_thickness`, `fov_extent`, `fluid_sensitive`, `fat_suppression`, `transfer_syntax_family`.  
**Rationale**:
- Resolution varies 3.4× in PixelSpacing across the corpus
- Slice count varies 10-60+ per series
- Transfer-syntax family correlates with vendor/pipeline age → hint at MRI physics
- One-hot the categorical `transfer_syntax_family` before scaling
- KMeans as default (cheap, deterministic), GMM as opt-in for soft assignments

**Rejected Approaches**:
- Post-crop clustering: nothing varies after `physical_crop` + `resize`
- Per-finding clustering (12 specialists): already in the existing Phase 3-5 plan

**Impact**: Each specialist sees a homogeneous acquisition-physics slice of the corpus, reducing the "small noise" you described.

---

### Decision 12: Per-cluster DINOv2-small specialists, rank-mean ensemble with main DINOv2-base
**Context**: The plan called for "different small models for different resolutions" feeding into "the main DINOv2 final solve".  
**Options Considered**:
- One mega-DINOv2-base on the full 80%
- Per-cluster DINOv2-base specialists, max-mean ensemble
- Per-cluster DINOv2-small + main DINOv2-base, rank-mean ensemble (chosen)

**Decision**: DINOv2-base on full 80% + K=4 DINOv2-small specialists routed by cluster → rank-mean across branches.  
**Rationale**:
- DINOv2-small keeps total param count reasonable (4× small + 1× base ≈ 1.7× base)
- Specialist routing lives in `cluster_specialist.specialist_router`
- All branches share `PerDiagnosisHead` so logits are in the same space
- Rank-mean is rank-invariant — required by the macro-AUC metric
- AUC sanity: synthetic 400-row test confirms rank-mean ≥ min(single branches)

**Rejected Approaches**:
- Mega-model only: contradicts the "noise reduction" goal of the plan
- Per-cluster base: param budget doubles; unsustainable on Kaggle 2× T4 × 16h

**Impact**: The "feed into the main DINOv2" idea is now realized as a rank-mean blend, not a learned fusion layer — sidestepping the cold-start risk of training a fusion layer with only 58 gold labels.

---

### Decision 13: Dataset emits 5 slots × 10 groups of 3 (50 groups per study)

**Context**: `KneeMRIDataset.__getitem__` shape contract.
**Decision**: 5 slots × 10 groups × 3 adjacent slices per group = 50 group-tensors per study.
**Rationale**: Matches `configs/classical:n_groups_per_slot=10` and the existing `sample_central_band` convention from `src/imaging/preprocessing.py`. Group-of-3 lets us reuse the pretrained 3-channel DINOv2 input interface.
**Impact**: Loading 1 study → 50 groups, ~5 KB each at 336². Batched with `collate_batch` → `(B, 5, 10, 3, 336, 336)` shape with a slot mask for missing slots.

---

### Decision 14: Confidence-weighted BCE with NaN→0.5 fallback

**Context**: BCE loss for the 12-diagnosis classifier.
**Decision**: Use `confidence_weighted_bce` with NaN labels mapped to 0.5 and weight 0.
**Rationale**: Reuses `process_labels.compute_confidence_weights()` output (`_conf` columns). Silent reports contribute 0 to the loss — they don't dominate training. The fallback formula `abs(label-0.5)*2 + 0.5` is applied when `_conf` columns are missing.
**Impact**: Stable training signal. Handles the 25% "not addressed" cells in the report-derived labels.

---

### Decision 15: Single `train_classical.py --mode {main,specialist,ensemble}` entry point

**Context**: How to invoke training on Kaggle.
**Decision**: One CLI script with `--mode` flag.
**Rationale**: Matches the Kaggle notebook pattern. The user picks the mode per run; same code path for both branches.
**Impact**: Simpler Kaggle-side orchestration. Easy to switch from `main` to `specialist_3` without changing scripts.

---

### Decision 16: EMA decay 0.999 main / 0.995 specialists, AdamW + cosine LR

**Context**: Training stability choices.
**Decision**: Standard EMA decay rates and cosine schedule.
**Rationale**: 0.999 / 0.995 are well-validated defaults from the vision-transformer literature. AdamW with cosine LR matches the DINOv2 paper recommendations.
**Impact**: Training is stable, head learns faster than backbone (10× LR multiplier), warmup prevents early-epoch collapse.

---

### Decision 17: `RSNA_DICOM_DIR` env var override for DICOM directory

**Date**: 2026-08-21
**Context**: `src/config.py:DICOM_DIR` defaults to `data/raw/train_series` (a non-existent local path because the competition's 570 GB of DICOMs are not on disk). The Kaggle CLI couldn't download them either.
**Options Considered**:
- Hardcoded `data/raw/train_series` — broken on local machines
- Symlink to wherever DICOMs live — fragile, requires setup
- Env var override — explicit, no setup needed
**Decision**: `os.environ.get("RSNA_DICOM_DIR", default)`. Any script that reads `src.config.DICOM_DIR` picks up the override automatically.
**Rationale**: Same pattern as `KAGGLE_DICOM_DIR`-style conventions. Zero code changes needed in scripts. CI-friendly.
**Impact**: `RSNA_DICOM_DIR=data/local_dicoms python scripts/...` works for any script.

---

### Decision 18: Extract-only Kaggle kernel pattern for partial data download

**Date**: 2026-08-21
**Context**: Need 50 studies of real DICOMs locally to validate the pipeline. `kaggle competitions download` is 570 GB; impractical.
**Options Considered**:
- Full 570 GB download — too big for the local 920 GB disk
- Pre-existing public dataset (e.g. `barun2104/rsna-knee-mri-processed-3d-volumes`) — preprocessed, doesn't test our preprocessing pipeline
- Push an extract-only kernel that copies N studies from `/kaggle/input/competitions/.../train_series/` to `/kaggle/working/`, then `kaggle kernels output <slug>/<version>` to download
**Decision**: Extract-only kernel + per-version output download.
**Rationale**:
- Kernel runs entirely inside Kaggle's compute — no local disk pressure
- No submission slot used (no `--file` flag at submit time)
- `kaggle kernels output <slug>/<version>` returns the file tree as a directory, bypassing the broken-binary-zip download path
- 1 study / 25 files / 13 MB is enough to validate that `load_series`, `get_series_metadata`, and the slot builder all work on real DICOMs
**Rejected Approaches**:
- Single zip via `ZIP_STORED` — Kaggle download returned 0 bytes for binary zips (>50 MB threshold observed)
- Per-study zips — same 0-byte issue
- Inline base64 — works for tiny files, scales terribly
**Impact**: `data/local_dicoms/` now has 1 study / 2 series / 25 real DICOMs. `RSNA_DICOM_DIR=data/local_dicoms` validates the full DICOM → numpy pipeline. Documented limitation: only 1 study downloaded (not 50) due to Kaggle's output truncation; more would require fixing the download bug or running directly on Kaggle.

---

### Decision 19: Training strategy — joint first, incremental as research track, bagged ensemble conditional

**Date**: 2026-09-07
**Context**: Data arrives in extraction-sized chunks (Decision 18 transport bug). Two user-proposed training strategies — sequential fine-tuning on successive sample batches, and hierarchical ensembling ("tree") of per-chunk models — were evaluated alongside joint training.
**Options Considered**:
- Track A: joint training on all on-disk studies at once
- Track B: continual/incremental fine-tuning, one chunk after another (optionally with replay)
- Track C: bagged chunk ensembles, flat rank-mean blend of all per-chunk models
- Wait-for-full-corpus / community preprocessed volumes / kernel-only training / feature caching

**Decision**: Track A first (baseline on all 172 studies, local GPU), then a feature-cache accelerator, then Track B (± replay) as the research experiment on new chunks, then Track C only once ≥3 chunks exist. Full why/why-not analysis: `.md Files/RESEARCH_SUMMARY.md` §15.
**Rationale**: Joint training is the upper bound and cheapest path to a real number; the constraint is transport, not training memory. Sequential fine-tuning ends below joint (catastrophic forgetting) — valuable as a studied phenomenon, not as the primary strategy. A hierarchy of averages collapses mathematically to a flat average, so the ensemble is flattened; the tree is kept only as a paper ablation.
**Rejected Approaches**:
- Incremental as primary: sacrifices leaderboard score for no benefit
- Hierarchical ("tree") ensembling as primary: mean-of-means is the flat mean; learned meta-stacking would overfit on 58 gold labels
**Impact**: All tracks share one fixed gold-51 evaluation; `RESULTS.md` becomes the paper's ablation table. Terminology fixed: Track B is continual/incremental learning (not federated); Track C is bagging.

---

### Decision 20: Kaggle-first execution — code-as-dataset + single gated training kernel

**Date**: 2026-09-08
**Context**: Local machine has 7 GB RAM (probes/12 GB collates stall) but a Kaggle GPU is free and the preprocessed volumes already live there as a private dataset. The >50 MB download bug corrupts local pulls only, not Kaggle-side dataset storage.
**Options Considered**:
- Fix local memory handling (mmap, sampled probes) and train locally
- Train only on Kaggle with code cloned from GitHub
- Train on Kaggle with the existing repo-snapshot dataset pattern

**Decision**: All real compute (validation gate B2, smoke train, full training, gold eval) runs in a single Kaggle GPU script kernel (`train_sample200`). Code ships via a refreshed version of `mananjain39/rsna-knee-repo-snapshot` (kept for determinism; no GitHub push needed). `labels_final.csv` pushed as `mananjain39/rsna-knee-labels-final`. Probes gained a `--max-studies` sampled contract mode for any local debugging.
**Rationale**: 16 GB VRAM + data co-located beats a 7 GB laptop for every training-shaped task; kernel versions double as immutable experiment records for the paper. Sampled probe mode keeps the same contract guarantees locally when iteration speed is needed.
**Rejected Approaches**:
- Local-first training: RAM ceiling makes every substantive run swap-drown regardless of mmap fixes
- GitHub clone in kernel: requires push access mid-experiment and is non-deterministic relative to a versioned snapshot
**Impact**: Each experiment is reproducible via (repo-snapshot version × dataset version × kernel version); all three are recorded in RESULTS.md entries going forward. Track A baseline expected to complete in one ~9 h kernel session.

---

### Decision 21: Kaggle CLI upload quirks — payload-as-`.bundle` pattern

**Date**: 2026-09-08
**Context**: `kaggle datasets create/version` (CLI 2.2.4) silently skips subdirectories and ignores `.zip` files; probes then mount only top-level files. Kernel mounts are namespaced (`/kaggle/input/datasets/<owner>/<slug>`).
**Decision**: Ship any multi-directory payload (code snapshots) as a single archive with a non-archive extension (`.bundle`), uploaded as one top-level file, and unzip it inside the kernel with `unzip -q -o`. Set both `dataset_sources` paths with the `datasets/` prefix.
**Rationale**: Avoids both failure modes deterministically; the bundle is verifiable by byte size after upload.
**Impact**: `rsna-knee-repo-snapshot` now always carries a complete tree; extract/training kernels unpack it. Also pinned `torch==2.5.1 torchvision==0.20.1` (cu124) inside GPU kernels — Kaggle's image torch has no sm60 (P100) kernels and a mismatched torchvision ABI.

---

## Phase 6-10: Quantum ML (Pending)

*Decisions will be logged as quantum phases are implemented.*
