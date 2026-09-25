# Next Steps — RSNA Knee (2026-09-01)

Locked plan to get the project from "code compiles, 3-study sample" to
"first real gold-58 macro AUC". Follow the phases in order; do not skip
the Validation Gate (Phase B).

---

## Context recap

- All Phase 0–5 code from `.md Files/PLAN.md` is implemented and unit/smoke-tested.
- Data side is at 3 real preprocessed studies (`data/processed/volumes_sample3`).
- All `experiments/*.jsonl` show `val_macro_auc = 0.5` (degenerate — nothing has
  trained on real MRI yet).
- `KneeModel` forward/backward has **not** been demonstrated on real volumes
  (only on the synthetic end-to-end smoke test).

The single thing standing between us and a meaningful AUC is a larger,
gold-inclusive real sample that survives the Kaggle transport bug.

---

## Phase A — Real 150–200 study sample as a private Kaggle Dataset

Bypass the >50 MB ZIP download bug (`.md Files/DECISION.md` Decision 18) by
running the existing preprocessing path **inside** the kernel and pushing the
result as a private Kaggle Dataset instead of as a downloadable artifact.

### A1. Rewrite the extraction kernel

File: `kaggle_kernels/dicom_sample_extract/extract.ipynb`

Behaviour:
1. Mount the competition and read `train.csv` from
   `/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv`.
2. Build a deterministic study list:
   - all 58 gold studies (`ACL.notna()`)
   - plus ~100–142 non-gold studies (random, seed=42, stable across re-runs)
   - cap at 200 total.
3. Clone this repo into the kernel and `pip install -e .[dev]` (no `.[classical]`
   yet — preprocessing does not need torch).
4. Reuse the local pipeline:
   - `src/imaging/dicom_loader.load_series` / `get_series_metadata`
   - `src/imaging.pipeline.preprocess_series`
   - `src/imaging.slot_builder.build_slots_for_study`
   - write `<study_uid>/<slot>.npy` (float32, `(N, 336, 336)`).
5. Build a `series_metadata.csv` (PixelSpacing, rows, cols, n_slices,
   slice_thickness, fov_extent, fluid_sensitive, fat_suppression,
   transfer_syntax_family) — same schema as
   `scripts/preprocess_volumes.py:_metadata_row`.
6. Write `manifest.json`:
   ```json
   {
     "created_at": "<iso>",
     "kernel_version": "<n>",
     "n_studies": 158,
     "gold_uids": ["...", ...],
     "study_uids": ["...", ...],
     "preprocessing_failures": []
   }
   ```
7. Push the resulting folder as a **private** Kaggle dataset, slug
   `rsna-knee-processed-sample200`, via either:
   - `kagglesdk` from inside the kernel (preferred), or
   - `subprocess.run(['kaggle', 'datasets', 'create', ...])` using the
     kernel's bundled auth.
8. Leave the kernel output as a tiny text log + the dataset URL.
9. `kernel-metadata.json`: flip `enable_internet` to `true` (needed later
   in Phase B for DINOv2 weights — keep it `true` for both kernels).

### A2. Local auth

There is no `~/.kaggle/kaggle.json` on this machine. Set:

```bash
export KAGGLE_API_TOKEN="<KGAT>"
```

Treat as a secret — never log or print it. Add to `.env` (already
git-ignored) if you want it persisted across shells.

### A3. Download the dataset locally

```bash
mkdir -p data/processed/volumes_sample200
kaggle datasets download mananjain39/rsna-knee-processed-sample200 \
    -p data/processed/volumes_sample200 --unzip
```

Expected layout:

```
data/processed/volumes_sample200/
  manifest.json
  series_metadata.csv
  <StudyInstanceUID>/
    Sagittal_Fluid.npy
    Sagittal_Structural.npy
    Coronal_Fluid.npy
    Coronal_Structural.npy
    Axial_Fluid.npy
```

### A4. Build a leak-safe, gold-reserved dev split

`scripts/build_classical_split.py` currently calls `make_split()` which reads
the full `train.csv`. Restrict it to the sampled UIDs first, then split:

```bash
# Add a --study-uids-csv argument to build_classical_split.py so it
# filters before GroupShuffleSplit + gold-to-val reservation
python scripts/build_classical_split.py \
    --tag sample200 \
    --study-uids-csv data/processed/volumes_sample200/manifest.json \
    --output-dir data/processed/splits/sample200
```

Expected outcome:
- all 58 gold UIDs land on the val side (Decision 10 guarantee)
- no duplicate `Report` text appears in both train and val
- `train.csv` ≈ 100 studies, `val.csv` ≈ 58 + a handful of non-gold

### DoD — Phase A
- ≥ 150 study directories, each with 5 slot `.npy` files.
- `data/processed/splits/sample200/{train,val}.csv` exists.
- `manifest.json` lists every downloaded UID plus its gold flag.
- `preprocessing_failures` is empty (or every failure is acknowledged in
  `.md Files/DECISION.md`).

---

## Phase B — Validation Gate (per `.md Files/PLAN.md` §"Validation Gate")

This is the gate between "code compiles" and "any AUC is meaningful". Run
both before touching model hyperparameters.

### B1. CPU contract (no torch needed)

```bash
pip install -e ".[classical]"     # one-time; gives us torch + transformers
python scripts/probe_sample_architecture.py \
    --volumes-dir data/processed/volumes_sample200 \
    --split-dir data/processed/splits/sample200
```

Pass criteria:
- `batch_shape` is `(B, 5, 10, 3, 336, 336)` with `B = len(train)+len(val)`.
- every sample has `slot_fill_counts > 0` for at least one slot.
- finite values, weights sum > 0.

### B2. GPU + model contract (Kaggle T4)

```bash
python scripts/probe_sample_architecture.py \
    --volumes-dir data/processed/volumes_sample200 \
    --split-dir data/processed/splits/sample200 \
    --with-model --device cuda
```

Pass criteria:
- `loss` is finite.
- `head_gradient_norm > 0`.
- one optimizer step changes an attention-head parameter.

### B3. Smoke training pass (1–2 epochs, frozen backbone)

```bash
python scripts/train_classical.py --mode main \
    --data-dir data/processed/volumes_sample200 \
    --labels-csv data/processed/labels_final.csv \
    --split-tag sample200 \
    --output-dir data/processed/checkpoints/sample200_main \
    --max-epochs 2 --device cuda
```

Pass criteria:
- `val_macro_auc > 0.5` and strictly increasing across the 2 epochs
  (degenerate baseline was always exactly 0.5).
- non-empty `best.pt`.

### DoD — Phase B
- All three probe outputs are checked into `experiments/probe_*.json`.
- One `main.jsonl` line shows `best_val_macro_auc > 0.5`.

---

## Phase C — First real baseline number (`.md Files/BASELINE_RUNBOOK.md`)

Run on Kaggle with the processed dataset attached. Goal: one immutable
baseline artifact with a real macro AUC.

### C1. Train (Kaggle T4 ×2, ≤16 h)

```bash
python scripts/train_classical.py --mode main \
    --data-dir /kaggle/input/rsna-knee-processed-sample200 \
    --labels-csv /kaggle/input/rsna-knee-processed-sample200/labels_final.csv \
    --split-tag sample200 \
    --output-dir /kaggle/working/checkpoints/main \
    --max-epochs 20 --device cuda
```

### C2. Evaluate against gold

```bash
python scripts/evaluate_classical.py \
    --main-checkpoint /kaggle/working/checkpoints/main/best.pt \
    --data-dir /kaggle/input/rsna-knee-processed-sample200 \
    --split-tag sample200 \
    --output-dir /kaggle/working/metrics/main \
    --device cuda
```

Produces:
- `gold_metrics.json` (per-label + macro AUC)
- `gold_per_label_metrics.csv`
- `gold_predictions.csv`

### C3. Save the immutable baseline artifact

Bundle and version:
```
runs/baseline_001/
  best.pt
  gold_metrics.json
  gold_per_label_metrics.csv
  main.jsonl
  config_snapshot.yaml   # copy of configs/default.yaml at training time
  README.md              # one-paragraph "what & why"
```

Compare every future experiment against this same split and gold eval
(`.md Files/BASELINE_RUNBOOK.md` rule).

### DoD — Phase C
- One immutable baseline bundle checked in with a real, non-degenerate AUC.
- `.md Files/DECISION.md` updated with the resulting number and any decisions
  taken during the run.

---

## Phase D — Decision point (out of scope for this iteration, defined for context)

Only after Phase C lands a real number, pick from:
- partial-unfreeze backbone + longer schedule (if under-trained)
- cluster specialists + rank-mean ensemble (Decision 12, if acquisition
  heterogeneity dominates)
- per-diagnosis head tweaks (if specific labels underperform)
- begin Phase 6–10 Quantum ML (`.md Files/PLAN.md` §6) only when classical
  is competitive

---

## Out of scope for this iteration

- Full 4,407-study preprocessing — `BASELINE_RUNBOOK.md` forbids starting
  specialists until a representative slice produces a real gold number.
- Phase 6–10 (Quantum ML) — blocked on classical baseline.
- New model components or head architectures — `.md Files/PLAN.md`
  "Validation Gate" rule.

---

## Risks and fallbacks

1. **`kaggle datasets create` from inside a kernel can be brittle.**
   Fallback: pre-create the dataset slug locally with
   `kaggle datasets create -p <empty-dir> --title rsna-knee-processed-sample200`
   and have the kernel call `kaggle datasets version -m "..."` with the new
   files placed in `/kaggle/working/upload/`.

2. **DICOM transfer-syntax variety may surface preprocessing failures.**
   Requirement of A1: the kernel writes `preprocessing_failures` into the
   manifest, and we do not proceed past Phase A until it is empty or each
   failure is acknowledged.

3. **DINOv2 weights need internet inside the kernel.**
   `kernel-metadata.json.enable_internet` must be `true` for the Phase B
   kernel; flip it in the same commit.

4. **`.[classical]` extra is not currently installed.**
   One-time `pip install -e ".[classical]"` before B1. If torch pulls in
   large deps, do it in a fresh venv to avoid touching `.venv`.

5. **Failure of the ZIP download path is exactly the bug we are routing
   around.** Do not reintroduce a download-and-zip step "for convenience" —
   it will break past 50 MB.

---

## Checklist

- [ ] A1. Rewrite extract.ipynb, repush kernel, confirm dataset appears
- [ ] A2. Set `KAGGLE_API_TOKEN`
- [ ] A3. `kaggle datasets download ... --unzip`
- [ ] A4. `build_classical_split.py` learns `--study-uids-csv`, runs, deposits splits
- [ ] B1. Probe (CPU) prints finite batch
- [ ] B2. Probe (GPU + DINOv2) shows non-zero gradient on attention head
- [ ] B3. 1–2 epoch main training shows `val_macro_auc > 0.5`
- [ ] C1. Full main training on Kaggle
- [ ] C2. Evaluate against the 58 gold studies
- [ ] C3. Bundle and version `runs/baseline_001/`

---

## Phase E — LLM label extraction quality (critical, biggest impact)

**Problem identified:** Our LLM prompt is generic (one-liner per finding, no clinical rules). Top-2.3% 
teams use clinically-grounded prompts with explicit rules per finding, iterated against gold-58 by reading 
errors and refining. Their prompt reached 0.833 mean per-class accuracy against expert labels; the gap 
between that and the ~0.88-0.92 ceiling (inherent in report-vs-image disagreement) is "probably the real 
limiting factor on this whole approach right now, ahead of anything on the image side."

### E1. Rewrite `src/labels/llm_extractor.py` SYSTEM_PROMPT

Current prompt: one-liner per finding, no clinical rules, no negation handling.

New prompt structure:
- **Explicit clinical rules per finding**: e.g., Synovitis prompt must include "Hoffa fat pad impingement, 
  plica syndrome, friction/impingement syndromes with inflammatory component" (radiologists describe 
  synovitis in indirect ways before using the word).
- **Universal rules**: explicit negation ("no effusion", "sin rotura") = 0; unmentioned finding = 0 
  (radiologists report positives, stay silent on negatives); hedged mentions ("possible small tear") 
  are judgment calls; Impression/Conclusion section weighted most heavily; only current findings count 
  (not chronic or post-operative).
- **Temperature 0, reasoning off, fixed JSON output** with all 12 keys or retry.

### E2. Iterate against gold-58

1. Run current prompt on 58 gold studies → compute per-label accuracy + per-class errors.
2. Read errors: identify systematic patterns (e.g., Synovitis false negatives from indirect descriptions, 
   Effusion false positives from "trace" vs "some" disagreement).
3. Revise prompt to fix specific patterns (not random noise).
4. Re-run, re-measure. Target: 0.83+ mean per-class accuracy.
5. Lock prompt for full 4,349-study run.

### E3. Full extraction run

- All 4,349 + 58 gold reports (gpt-4o-mini, temp 0, ~$2-5 API cost).
- JSONL checkpoint output → resume-safe across interruptions.
- Concurrency: ThreadPool ~10 workers → ~20-40 min total.

### E4. Validate and compare

- `scripts/validate_labels.py` vs gold-58 → per-label agreement, not-addressed rates.
- Compare vs stevenleehans (0.8927 AUC) on same subset.
- If our extraction < 0.83 accuracy, iterate more before full run.

**DoD:** Prompt reaches 0.83+ mean per-class accuracy on gold-58; full extraction complete; 
validation report logged in RESEARCH_SUMMARY.md.

---

## Phase F — Label fusion with reliability weights (critical)

**Problem identified:** We pick one label source (stevenleehans) or blend naively. Top teams fuse multiple 
sources with per-label reliability weights. When sources disagree, output is a soft label (reliability-weighted 
average), not a hard vote.

### F1. New `scripts/blend_labels.py`

Inputs:
- Our LLM extraction (Phase E output)
- stevenleehans v4_blend (existing)
- Optional: pilkwang, barun2104 (existing in data/external/)

Logic:
- For each label (12 columns), measure each source's per-label accuracy against gold-58.
- Reliability weight = source's accuracy on that specific label (not blanket weight).
- When all sources agree → hard 0 or 1.
- When sources disagree → soft label = reliability-weighted average of votes.

**Critical bug to avoid:** Top team's write-up documents a fusion bug where `int()` truncation made 
one-directional asymmetric errors (positive probability couldn't override negative primary). Ensure 
symmetric fusion with no truncation.

### F2. Validate fusion candidates

- Candidates: ours alone, stevenleehans alone, mean-blend, reliability-weighted blend.
- Validate each vs gold-58 with `scripts/validate_labels.py`.
- Pick winner → `labels_final_v2.csv` (same schema: gold override + Synovitis imputation + conf weights).

### F3. Push as labels dataset v2

- New Kaggle dataset version: `mananjain39/rsna-knee-labels-final` v2.
- Required by train kernel for all future runs.

**DoD:** Fusion logic implemented and audited for symmetry; winner validated vs gold-58; 
labels_final_v2.csv pushed as dataset v2.

---

## Phase G — Augmentation (critical, prevents overfitting)

**Problem identified:** We have zero augmentation (no rotation, no brightness, no jitter). Top teams use 
"random rotation, brightness-gamma jitter, and slight rescale to every slice in a series." With 116-1000 
training studies, augmentation is crucial for generalization.

### G1. New `src/classical/augmentation.py`

Augmentations (applied identically to all slices in a series):
- **Random rotation**: ±10° (small, since knee MRI is already oriented).
- **Brightness-gamma jitter**: gamma in [0.8, 1.2] (simulate intensity variation).
- **Slight rescale**: 0.95-1.05× (simulate size variation).
- **NEVER horizontal flip**: would swap left/right anatomy without updating laterality-specific labels 
  (medial vs lateral meniscus, MCL, etc.).

### G2. Integrate into `src/classical/dataset.py`

- `KneeMRIDataset.__getitem__`: after loading slots, apply augmentation if `self.augment=True`.
- Augmentation applied per-series (all slices in a series get same transform).
- At eval time: `self.augment=False` (deterministic).

### G3. Config flag

- `configs/default.yaml`: `augmentation: {enabled: true, rotation_deg: 10, gamma_range: [0.8, 1.2], rescale_range: [0.95, 1.05]}`.
- `scripts/train_classical.py`: `--no-augmentation` flag for ablation.

**DoD:** Augmentation implemented; training with augmentation shows finite loss + no NaN; 
ablation (with vs without) logged in RESEARCH_SUMMARY.md.

---

## Phase H — Hyperparameter proxy search (critical, finds best configuration cheaply)

**Problem identified:** We used default config (DINOv2-base, 10 groups/slot, no search). Top teams did 
20 proxy trials on 10% data, 2 epochs each, ranked by blended score (0.7×CV-AUC + 0.3×gold58-AUC). 
Proxy predicted 0.6811; full run scored 0.8788 — proxy was rough guide, not predictor, but selected the 
right configuration.

### H1. New `scripts/proxy_search.py`

Trials (20 cheap runs):
- **Backbone**: DINOv2-small (22M params), DINOv2-base (86M), efficientnet_b3 (12M).
- **Groups per slot**: 5, 10, 15.
- **Augmentation**: none, medium (Phase G).
- **Pooling**: max, gated attention.
- **Data**: 10% of training studies (random, seed=42).
- **Epochs**: 2 (fast, rough signal).

Metric per trial:
- Blended score = 0.7 × cross-validation macro-AUC + 0.3 × gold-58 macro-AUC.
- Rank trials by blended score.

### H2. Take winner to full training

- Best trial → full 5-fold training (or single large run if data is limited).
- Log proxy prediction vs full-run outcome in RESEARCH_SUMMARY.md.

**DoD:** 20 proxy trials complete; winner selected; full training run complete; proxy vs full comparison logged.

---

## Phase I — Cross-fitted gold-58 evaluation (important, more robust than single split)

**Problem identified:** We have a single train/val split (116 train / 84 val with 58 gold). Top teams use 
5-fold GroupKFold, gold excluded from fold pool. Each gold study scored only by the fold that held it out. 
Top team caught leakage (0.9357 → 0.8568) by re-scoring each gold study against the one fold where it was 
held out.

### I1. New `scripts/cross_validate.py`

- 5-fold GroupKFold over study UIDs (each study is its own group).
- Gold-58 excluded from train folds entirely (never in any fold's training set).
- For each fold: train on 4 folds, validate on held-out fold.
- Cross-fitted gold-58 evaluation: each gold study scored only by the fold that held it out.
- Report: mean macro-AUC across folds + per-fold breakdown.

### I2. Compare vs single-split result

- Single-split result (Phase 0 re-eval) vs cross-fitted result.
- If cross-fitted is significantly different → single-split was high-variance.
- Log comparison in RESEARCH_SUMMARY.md.

**DoD:** Cross-fitted gold-58 evaluation complete; comparison with single-split logged.

---

## Phase J — uint8 quantization (important, solves 1,000-study dataset size issue)

**Problem identified:** 200 studies = 12 GB float32 → 1,000 studies ≈ 60 GB > 20 GB/dataset cap. Top teams 
quantize to uint8 + compression: "39GB for 24,371 series." uint8 halves size (30 GB for 1,000 studies) + 
compression → ~15 GB. Fits in one dataset.

### J1. Modify `src/imaging/preprocessing.py`

- After [0,1] normalization: multiply by 255, cast to uint8.
- Save as `.npz` (compressed) instead of `.npy` (uncompressed).
- Loader: read uint8, convert back to float32 (divide by 255).

### J2. Verify bit-exact parity

- Top team's write-up: "preprocessing was checked bit-exact against the cached training tensors."
- Our loader must reproduce exact same windowing, cropping, resizing, quantization at train time.
- Test: load same study via old (float32) and new (uint8) paths → assert max absolute difference < 1e-3.

### J3. Re-extract 200 studies as uint8

- New dataset version: `rsna-knee-processed-sample200` v2 (uint8 + npz).
- Size: ~6 GB (down from 12 GB).

### J4. Extract 1,000 studies (Phase 2 from earlier plan)

- With uint8 + compression: 1,000 studies ≈ 15 GB → fits in one dataset.
- No chunking needed.

**DoD:** uint8 path implemented and tested; 200-study dataset re-extracted as v2; 1,000-study extraction complete.

---

## Phase K — Simpler model alternatives (important, proxy search will tell us if it wins)

**Problem identified:** We use DINOv2-base (86M params) + PerDiagnosisHead (learned attention queries). 
Top teams used efficientnet_b3 (12M params) + max-pool + linear head. Proxy trials showed max-pool beat 
gated attention. With 116-1000 training studies, simpler model might generalize better.

### K1. Add efficientnet_b3 as alternative backbone

- New `src/classical/efficientnet.py`: efficientnet_b3 (pretrained ImageNet weights) + single input channel.
- Same interface as DINOv2Backbone: `encode(pixel_values) → (cls, patches)`.
- Reuse existing PerDiagnosisHead.

### K2. Proxy search comparison (Phase H)

- Proxy trials include efficientnet_b3 vs DINOv2-small vs DINOv2-base.
- Winner → full training.

### K3. If efficientnet wins

- Full training with efficientnet_b3 + max-pool + linear head.
- Compare vs DINOv2 results in RESEARCH_SUMMARY.md.

**DoD:** efficientnet_b3 backbone implemented; proxy search includes it; winner selected; full training complete.

---

## Phase L — Inference robustness (nice-to-have, crash safety)

**Problem identified:** Top teams' inference kernels have: decode fallback chain (pydicom → pylibjpeg → gdcm), 
runtime self-monitoring (degrades from 32→16 slices, then 5→3 models if behind), crash-safe submission 
writes every 25 studies, neutral 0.5 for failed DICOMs.

### L1. Decode fallback chain

- `src/imaging/dicom_loader.py`: try pydicom default handler → pylibjpeg → gdcm.
- Log which handler worked (for debugging).

### L2. Runtime self-monitoring

- Inference kernel: track projected total runtime every 25 studies.
- If projection > budget: degrade from 32→16 slices per series.
- If still behind: degrade from 5→3 models in ensemble.

### L3. Crash-safe submission

- Write `submission.csv` every 25 studies (not just at end).
- Failed DICOMs → neutral 0.5 prediction (don't crash whole run).

**DoD:** Fallback chain implemented; runtime monitoring + degradation implemented; crash-safe writes implemented.

---

## Execution order (revised 2026-09-21)

**Priority 1 (critical, biggest impact):**
1. Phase E: LLM prompt engineering (label quality is the bottleneck)
2. Phase F: Label fusion (more robust than single source)
3. Phase G: Augmentation (prevents overfitting)
4. Phase H: Proxy search (finds best configuration cheaply)

**Priority 2 (important, more robust):**
5. Phase I: Cross-fitted gold-58 evaluation
6. Phase J: uint8 quantization (solves dataset size issue)
7. Phase K: Simpler model alternatives (proxy search decides)

**Priority 3 (nice-to-have):**
8. Phase L: Inference robustness (crash safety)

**Parallelizable:**
- Phase E (LLM extraction) runs locally while Phase J (uint8 re-extraction) runs on Kaggle.
- Phase H (proxy search) can start as soon as Phase G (augmentation) is done.
- Phase I (cross-validation) can run after Phase 0 (re-eval baseline) is done.

**Dependencies:**
- Phase F (fusion) depends on Phase E (our extraction).
- Phase H (proxy search) depends on Phase G (augmentation).
- Phase K (simpler models) is part of Phase H (proxy search).

---

## Updated checklist (2026-09-21)

- [ ] E1. Rewrite LLM prompt with clinical rules
- [ ] E2. Iterate prompt against gold-58 (target 0.83+ accuracy)
- [ ] E3. Full LLM extraction run (4,349 reports)
- [ ] E4. Validate extraction vs gold-58
- [ ] F1. Implement label fusion with reliability weights
- [ ] F2. Validate fusion candidates vs gold-58
- [ ] F3. Push labels_final_v2.csv as dataset v2
- [ ] G1. Implement augmentation (rotation, brightness, rescale)
- [ ] G2. Integrate augmentation into KneeMRIDataset
- [ ] G3. Add config flag + ablation test
- [ ] H1. Implement proxy search (20 trials, 10% data, 2 epochs)
- [ ] H2. Take winner to full training
- [ ] I1. Implement cross-fitted gold-58 evaluation (5-fold CV)
- [ ] I2. Compare vs single-split result
- [ ] J1. Implement uint8 quantization in preprocessing
- [ ] J2. Verify bit-exact parity
- [ ] J3. Re-extract 200 studies as uint8
- [ ] J4. Extract 1,000 studies (uint8 + compression)
- [ ] K1. Add efficientnet_b3 as alternative backbone
- [ ] K2. Proxy search comparison
- [ ] L1. Decode fallback chain
- [ ] L2. Runtime self-monitoring + degradation
- [ ] L3. Crash-safe submission writes

*Last updated: 2026-09-21*
