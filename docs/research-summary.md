# RSNA Knee Abnormality Detection — Research Summary

> Competition: [RSNA 2026 Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
> Metric: Macro-averaged AUC across 12 binary labels
> Status: Active (as of August 13, 2026)

---

## Table of Contents

1. [Competition Mechanics](#1-competition-mechanics)
2. [Report-to-Label Extraction](#2-report-to-label-extraction)
3. [Handling "Not Addressed"](#3-handling-not-addressed)
4. [DINOv2 / DINOv3 Approaches](#4-dinov2--dinov3-approaches)
5. [MRI Series / Slice Selection](#5-mri-series--slice-selection)
6. [2.5D vs 3D Approaches](#6-25d-vs-3d-approaches)
7. [Multimodal Fusion](#7-multimodal-fusion)
8. [Cross-Validation Strategies](#8-cross-validation-strategies)
9. [External Datasets](#9-external-datasets)
10. [Computational Efficiency](#10-computational-efficiency)
11. [Public Leaderboard Landscape](#11-public-leaderboard-landscape)
12. [Critical Preprocessing Pitfalls](#12-critical-preprocessing-pitfalls)
13. [Recommended Implementation Order](#13-recommended-implementation-order)
14. [Key Source Links](#14-key-source-links)

---

## 1. Competition Mechanics

### What is being predicted

12 binary findings per knee MRI study:

| # | Label | Gold prevalence (58 studies) | Clinical notes |
|---|-------|------------------------------|----------------|
| 1 | ACL | 24/58 = 41% | High-grade partial or full-thickness tear |
| 2 | MCL | 9/58 = 16% | High-grade acute tear; low-grade sprains are negative |
| 3 | Medial Meniscus | 26/58 = 45% | Signal reaching meniscal surface on ≥2 images |
| 4 | Lateral Meniscus | 23/58 = 40% | Same criteria, lateral side |
| 5 | Medial OA | 15/58 = 26% | ≥1 cm area of >50% cartilage loss |
| 6 | Lateral OA | 11/58 = 19% | Same criteria, lateral compartment |
| 7 | PF OA | 21/58 = 36% | Same criteria, patellofemoral compartment |
| 8 | Effusion | 35/58 = 60% | Moderate or large fluid |
| 9 | Synovitis | 27/58 = 47% | Synovial lining inflammation/thickening |
| 10 | Baker's | 12/58 = 21% | Moderate/large popliteal cyst |
| 11 | Contusion | 19/58 = 33% | Bone marrow edema without fracture line |
| 12 | Fracture | 18/58 = 31% | Acute cortical break |

### Data shape

- **Training**: 4,407 studies, only **58** carry expert labels (1.3%)
- **Test**: ~1,300 studies, no reports
- **Series**: 24,371 training series (mean 5.5 per study, range 3–14)
- **Planes**: Sagittal 9,864 / Coronal 8,609 / Axial 5,898
- **Fluid-sensitive**: 14,010 (57.5%) / Non-fluid: 10,361 (42.5%)
- **Reports**: 9 languages (English, Spanish, French, German, Dutch, Bulgarian, Turkish, Greek, others), avg 1,098 chars
- **Duplicate reports**: 131 studies share verbatim reports (must be grouped in CV splits)
- **DICOM**: ~570 GB total, mixed transfer syntaxes (Explicit VR, JPEG Lossless, JPEG 2000, Implicit VR)

### Metric properties that constrain every design decision

**Score = (1/12) × Σ AUC_i**

Three consequences:

1. **Only rank order matters.** AUC is invariant under any strictly increasing transform. Calibration is worthless. Thresholds are worthless. Model combination must use rank averaging, not probability averaging.

2. **Every label costs the same.** A finding left at chance forfeits (M − 0.5)/12 of the final score. At M=0.85, that is 0.029 — larger than gaps between neighboring leaderboard positions. Rare findings deserve MORE attention than common ones.

3. **Prevalence drift is survivable; thresholds are not.** AUC is invariant to positive rate in expectation. The competition warns prevalence may differ across train/public/private. This is fine for AUC but fatal for any accuracy-like metric.

### Key design constraint

`train.csv` has a `Report` column. `test.csv` does NOT. Text is available during training but ABSENT at inference. This means:
- Reports can ONLY produce training targets, never model inputs at inference
- No text branch in any fusion model
- The pipeline shape is forced: read reports → targets → train imaging model → discard text

---

## 2. Report-to-Label Extraction

### The problem

Only 58 of 4,407 studies have expert labels. The remaining 4,349 need labels derived from their radiology reports. This is the binding constraint — the quality of extracted labels is a ceiling nothing downstream can pass.

### Two approaches compared

| Method | Macro AUC vs 58 gold | Notes |
|--------|---------------------|-------|
| Regex/lexicon (multilingual) | 0.8136 | Fast, deterministic, fails on silence |
| LLM reading reports | 0.8780 | GPT-4o or local Qwen3-14B, much better coverage |
| LLM + selective imputation | 0.8873 | Best published; fills only high-uncertainty cells |
| LLM (Qwen3-14B, closed vocab) | 0.881 | Two-layer: LLM picks from fixed states → deterministic mapping to probabilities |

### Public label sets

Several teams have published LLM-derived label sets as Kaggle datasets:

- **Pilkwang Kim** — `rsna-knee-llm-labels` (first, 2026-08-06), also `rsna-knee-baseline-v1`
- **stevenleehans** — LLM labels with "not addressed" analysis (2026-08-09)
- **barun2104** — Stratified Folds & LLM Soft Labels (2026-08-07)
- **ixin73** — LLM Report Labels (GPT-5.6-Sol) (2026-08-08)

### Best practices for extraction

**Graded targets, not binary.** A report saying "small joint effusion" may sit against a negative annotation because the annotator only marked significant effusions. A rule of "term present → positive" is wrong by construction. The solution: grade the mention (trace/unqualified/marked) and emit a probability.

**Two-layer LLM extraction** (from [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee)):
1. LLM reads report into a closed vocabulary of discrete findings (complete_tear, degeneration_no_tear, absent_explicit, severity grades)
2. Deterministic Python layer maps those states onto probabilities

This works better than asking the LLM for calibrated numbers directly, because LLMs are good at classification from a fixed set but bad at calibration.

**Multilingual handling** (from [romanrozen](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline)):
- Do NOT identify language first. Carry all 9 languages' cues simultaneously and test each clause against the union.
- Turkish dotted/dotless i must be pre-folded before casefolding
- Greek micro sign (U+00B5) must be mapped to real mu
- NFKD decomposition strips Latin accents and Greek tonos

**Commercial LLM APIs are permitted.** Host confirmed: "Submitting Competition Data, including report text, to an external LLM or API for inference or other computational processing will not, by itself, be considered prohibited PRIVATE SHARING." ([discussion/733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965))

### HARD vs SOFT targets

From [FHZ982](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105):

| Arm | Definition | 58-study AUC |
|-----|-----------|-------------|
| HARD | Explicit affirmation → 1, explicit negation/no mention → 0 | 0.7867 |
| SOFT | Affirmed → 1, hedged → 0.3-0.8, negated → 0.04-0.16, not mentioned → 0.28 | 0.7956 |

SOFT wins all 3 seeds (+0.0143 rank-mean ensemble), but the 95% bootstrap interval [-0.0041, +0.0330] crosses zero. With only 58 studies, the difference is inconclusive. The effect was much larger (+0.056) in a weaker ResNet-18 pipeline but did not transfer to the stronger DINOv2 pipeline.

**Takeaway**: SOFT targets are worth trying but the benefit is pipeline-dependent and not yet proven at scale.

---

## 3. Handling "Not Addressed"

This is arguably the most important finding from the community so far.

### The distribution of silence

25.4% of all label cells are "the report does not address this" (mapped to 0.5). But they are wildly uneven:

| Finding | Gold AUC | "Not addressed" rate | Interpretation of silence |
|---------|----------|---------------------|--------------------------|
| Synovitis | 0.678 | 83.7% | Nobody looked — silence is uninformative |
| Baker's | 0.946 | 48.2% | Absent — silence IS the label |
| Fracture | 0.793 | 42.9% | Mostly absent |
| ACL | 0.993 | 8.3% | Almost always addressed |
| Medial Meniscus | 0.954 | 5.5% | Almost always addressed |

### The critical insight

"Not addressed" is not one thing. For some findings it means absent; for others it means nobody looked.

**Silence ratios** (from [stevenleehans](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)):

| Finding | Gold+ when report SILENT | Gold+ when report SPEAKS |
|---------|------------------------|-------------------------|
| Synovitis | 0.34 | 0.76 |
| PF OA | 0.21 | 0.41 |
| Baker's | 0.03 | 0.44 |
| Medial OA | 0.00 | 0.36 |

- When a radiologist does NOT mention a Baker's cyst, there almost certainly is not one (3% vs 44%). The silence IS the label. Overwriting it with a correlation-based guess throws away real information.
- But silence about synovitis still leaves a 34% chance it is present. The silence is genuinely uninformative and imputing helps.

### Selective imputation results

| Configuration | Macro AUC vs gold |
|---------------|-------------------|
| v1 baseline (LLM labels) | 0.8780 |
| v2 — synovitis-only imputation (fill from Effusion) | 0.8873 |
| v3 — learned imputation, all twelve | 0.8805 |

The blanket version (v3) is WORSE than the targeted one (v2). Per label:
- Synovitis: +0.056 (helps)
- Baker's: -0.029 (hurts)
- PF OA: +0.023 (helps)
- Contusion: -0.031 (hurts)

Across the nine readable findings, the "silence ratio" correlates +0.59 with how much imputation helped.

**Practical recommendation**:
1. Ask your labeller for "I don't know" as a first-class answer (0.5)
2. For each finding, compute the silence ratio: P(positive | report silent)
3. Only impute findings where silence is genuinely uninformative (high silence ratio)
4. Never override explicit statements — only fill undecided cells

### Confidence weighting

Reports that explicitly address a finding should pull harder on that label than reports that are silent. Both the lexicon and LLM extractors can emit a confidence score per (report, finding) pair. This becomes a per-finding sample weight in the loss function.

A study whose report never mentions synovitis should pull on the synovitis head far less than one that names it. A loss that cannot express this trains every silence as a confident negative.

---

## 4. DINOv2 / DINOv3 Approaches

### Why DINOv2 dominates

DINOv2 is self-supervised — trained to agree with itself across augmented views, never told what an object is. This makes its features unusually good under linear probing, which is why a frozen baseline was viable in an afternoon.

It is Apache 2.0, so it redistributes cleanly as a Kaggle dataset (important when internet is off).

### The resolution argument

From [wguesdon](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) (83 votes):

Every public imaging kernel initially ran DINOv2 at 126 pixels. That number is not a hyperparameter — it is a measurement, and the measurement says the model cannot see the disease.

**The math:**
- Every series is acquired over a 150 mm field of view
- Four acquisition matrices appear (960, 896, 800, 640 px at 0.1562, 0.1674, 0.1875, 0.2344 mm/px)
- At 126 px input: 1.19 mm/pixel, one 14-pixel DINOv2 patch token covers 16.7 mm of knee
- A medial meniscus is 9-12 mm wide. A meniscal tear is 1-3 mm.
- At 126 px: a 1 mm tear occupies less than one pixel — it is GONE before the patch embedding ever sees it

**Measured on the frozen backbone:**
- A 1-3 mm feature moves the pooled embedding 2-5× further at 336 px than at 126 px
- A 10 mm feature gains nothing from higher resolution
- 448 px beats 336 on no lesion width tested

| Input size | Patch tokens | mm per token | Can resolve meniscal tear? |
|-----------|-------------|-------------|---------------------------|
| 126 px | 81 | 16.7 mm | No — below sampling limit |
| 224 px | 256 | 9.3 mm | Marginal |
| 336 px | 576 | 6.25 mm | Yes — 1-3 mm features detectable |
| 448 px | 1024 | 4.7 mm | No additional gain |

### DINOv2 ViT-S/14 vs ViT-B/14

| Variant | Params | Common use |
|---------|--------|-----------|
| ViT-S/14 | 21M (5M trainable) | wguesdon, pilkwang, hida1211 — most public notebooks |
| ViT-B/14 | 86M | sadamtorres (0.883 LB), romantamrazov |

### DINOv3 status

DINOv3 is NOT registered on Kaggle's models tab. Community consensus:
- It has been used in other competitions this year
- One user's V2 scored 0.775 while "pretty much the same" V3 scored 0.763 — a small loss, not a gain
- Hypothesis: the gram anchoring that makes DINOv3 powerful is only helpful with massive datasets and long training, which this competition does not have
- Pretrained weights are better from V3, but if fine-tuning is needed, these advantages are lost
- License risk: weights are gated behind approval process

**Takeaway**: DINOv2 is the safe choice. DINOv3 is not proven better here and has licensing complications.

### Fine-tuning strategy

From [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee):

| System | OOF (soft labels) | Gold-58 | Public LB | Runtime |
|--------|-------------------|---------|-----------|---------|
| Frozen DINOv2 + trained head | 0.682 | 0.771 | 0.776 | ~2h |
| 5-fold fine-tune @224px | 0.717 | 0.824 | 0.866 | ~3h |
| 5-fold fine-tune @336px | 0.722 | — | 0.883 | ~4h |

**Domain adaptation does most of the work, not resolution:**

| Finding | Frozen | Fine-tuned @224 | Δ |
|---------|--------|----------------|---|
| Medial Meniscus | 0.679 | 0.850 | +0.171 |
| MCL | 0.708 | 0.825 | +0.118 |
| ACL | 0.727 | 0.840 | +0.113 |
| Contusion | 0.676 | 0.775 | +0.099 |

A backbone trained on natural images does not know what to look for in an MRI. Once it does, 224px is enough to find a meniscal tear. Resolution was a real but secondary axis — worth +0.017 on top, not the +0.09 that adaptation bought.

**Partial unfreezing recipe** (from [FHZ982](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105)):
- DINOv2, 12 blocks, last six trainable (10.7M trainable parameters), feature dimension 768
- 10 epochs per run
- Weight EMA for epoch selection

### Encoder input design

**Do NOT stack 18 slices as channels and average the first convolution's pretrained weights.** This destroys the input interface the encoder learned.

**Correct approach**: Feed groups of 3 adjacent slices through the untouched backbone (as 3-channel RGB-like input), then aggregate feature vectors afterwards. This is the 2.5D approach.

---

## 5. MRI Series / Slice Selection

### The slot design

The winning approach organizes series into "slots" defined by plane × contrast:

**5-slot design** (from [wguesdon](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution)):

| Slot | Plane | Contrast | Slices | Fill rate |
|------|-------|----------|--------|-----------|
| Sagittal_Fluid | Sagittal | Fluid-sensitive | 12 | 94.2% |
| Sagittal_Structural | Sagittal | Non-fluid | 12 | 96.8% |
| Coronal_Fluid | Coronal | Fluid-sensitive | 8 | 96.4% |
| Coronal_Structural | Coronal | Non-fluid | 6 | 77.3% |
| Axial_Fluid | Axial | Fluid-sensitive | 8 | 100.0% |

Mean fill: 4.65 of 5 = 92.9% per slot.

**Why not 6 slots?** An Axial non-fluid slot exists for only 19.4% of studies. A sixth of the encoding budget buys a feature that is a padded zero for 80% of the corpus. That budget is worth more as slices in the slots that always fill.

**6-slot design** (from [romantamrazov](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2)):

| Slot | Definition |
|------|-----------|
| SAG_FLUID_FS | Sagittal, fluid-sensitive, fat-suppressed |
| COR_FLUID_FS | Coronal, fluid-sensitive, fat-suppressed |
| AX_FLUID_FS | Axial, fluid-sensitive, fat-suppressed |
| SAG_FLUID_NOFS | Sagittal, fluid-sensitive, NOT fat-suppressed |
| COR_T1 | Coronal, non-fluid, non-fat-suppressed |
| SAG_T1 | Sagittal, non-fluid, non-fat-suppressed |

**Key finding**: `Fluid_Sensitive` and `Fat_Suppression` are IDENTICAL on all 24,371 rows. They are one column wearing two names. Any design that spends a dimension on "fluid-sensitive AND fat-suppressed" is spending it on a copy.

### Per-label slot priors

Different findings are visible on different sequences:

| Label | Best slots |
|-------|-----------|
| ACL | Sagittal (fluid + structural), Axial |
| MCL | Coronal (fluid + structural) |
| Medial/Lateral Meniscus | Sagittal + Coronal (all contrasts) |
| Medial/Lateral OA | Coronal + Sagittal structural |
| PF OA | Sagittal fluid, Axial fluid, Sagittal structural |
| Effusion | Sagittal fluid, Axial fluid |
| Synovitis | Sagittal fluid, Axial fluid |
| Baker's | Sagittal fluid only |
| Contusion | Sagittal + Coronal + Axial fluid |
| Fracture | All slots |

### Slice ordering

**Do NOT sort .dcm files by filename.** The filename is a SOP Instance UID, assigned to be unique rather than ordered. Measured Spearman ρ ≈ 0.01 between filename rank and physical position.

**Correct approach**: Sort by physical position:
```
n = cross(r_x, r_y)    # slice normal from ImageOrientationPatient
k = p · n              # projected position from ImagePositionPatient
```

`InstanceNumber` is the fallback. It usually tracks k up to sign, but interleaved and multi-echo acquisitions number slices in an order that is not the order they occupy in space.

### Slice sampling

- Sample from the central 60-76% of each stack (outer slices are mostly soft tissue outside the joint)
- For 2.5D: take groups of 3 adjacent slices as 3-channel input
- Spread samples evenly across the band using `np.linspace(lo, hi, n_slices)`
- Each decoded slice appears in up to 3 groups (overlapping windows), providing context without extra reads

### Physical scale normalization

Crop to a constant physical extent before resizing:
- L = 130 mm (below smallest FOV, contains the joint)
- Resize to P = 336 px → 0.387 mm/pixel
- Effective scale: s_eff = L/P mm/pixel (independent of acquisition)

**The crop must be smaller than the smallest field of view** or it silently becomes a no-op. L = 130 mm works; L = 160 mm risks it.

### Intensity normalization

Per-series (not per-slice) 1st-99th percentile normalization:
- Over the whole sampled stack, preserving inter-slice contrast
- Percentiles rather than min/max because one bright vessel would compress everything else
- MR has no absolute scale — same tissue, different number on different sequence/coil/day

---

## 6. 2.5D vs 3D Approaches

### Why 2.5D dominates

Every top-scoring public notebook uses 2.5D. The reasons:

1. **DINOv2 expects 3-channel input.** Stacking N slices as channels and averaging the first conv's pretrained weights destroys the input interface. Groups of 3 preserve it.

2. **Memory efficiency.** A true 3D volume at 336³ would be enormous. 2.5D keeps the backbone's 2D architecture intact.

3. **Pretrained weights work.** DINOv2 was trained on 2D images. Fine-tuning it on 2D slices (with neighborhood context from adjacent slices as channels) leverages the full pretrained representation.

### The 2.5D recipe

```
For each slot:
  1. Sort series by physical position
  2. Sample CACHE_SLICES evenly from central band
  3. Form N_GROUP groups of GROUP=3 consecutive slices
  4. Each group → 3-channel input to DINOv2
  5. Extract CLS token + mean patch tokens per group
  6. Aggregate across groups at inference (average logits)
  7. During training: randomly sample one group per step (acts as augmentation)
```

### True 3D approaches

No public notebook uses true 3D CNNs. The Medical Slice Transformer (Müller-Franzes et al., 2025) with DINOv2 showed inter-slice modelling improved knee-meniscus AUC over a 3D ResNet, suggesting that if you do 3D, transformer-based approaches may work better than 3D CNNs.

### Aggregation: slots → study

**Per-diagnosis attention over slot embeddings** (from pilkwang/wguesdon):

```
For each slot s:
  h_s = φ(x_s) + e_s    # project + learned slot identity

For each diagnosis o:
  α_{o,s} = softmax(<h_s, q_o> / √H, masked by presence)
  c_o = Σ_s α_{o,s} · h_s
  logit_o = <c_o, w_o> + b_o
```

The masked softmax renormalizes over whatever slots the study actually contains, so a missing axial series shifts attention to present sequences rather than feeding a zero vector.

**Alternative** (from sadamtorres): Attention pooling over slices within each series, then mean within each plane, then concatenate 3 plane vectors → MLP → 12 logits.

---

## 7. Multimodal Fusion

### The fundamental constraint

Test has no reports. Text can ONLY be used for:
1. Training target derivation
2. Confidence weighting of training samples
3. Auxiliary losses during training (dropped at inference)

### What works

**Report-supervised training** (the dominant paradigm):
1. Extract labels from reports (LLM or lexicon)
2. Extract confidence weights from reports
3. Train a pure imaging model against these targets
4. Discard text before inference

**Confidence weighting**: A study whose report never mentions synovitis should pull on the synovitis head far less than one that names it. The per-(report, finding) confidence becomes a sample weight in the loss.

**Gold study weighting**: The 58 expert-labeled studies get higher weight (e.g., 5×) since their labels are ground truth rather than report-derived.

### What does NOT work

**Text branch at inference**: Impossible — no reports in test.

**Report-teacher distillation**: [Blacklions](https://www.kaggle.com/code/blacklions/report-teacher-anatomy-aware-hierarchical-multimod) attempted anatomy-aware hierarchical multimodal training with a report teacher, but this is complex and the gains are unclear.

### The ceiling insight

From [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee):

> Those 58 studies were annotated by radiologists reading the images, not the reports. Report and image disagree irreducibly maybe 10-15% of the time. The report reader therefore has a ceiling around 0.88-0.90 — and the imaging model, which sees what the annotators saw, CAN AND SHOULD beat its own teacher.

This is why OOF on soft labels systematically understates real gains. When the model gets better at seeing the knee, it departs from the labels precisely on the studies where the report was wrong.

---

## 8. Cross-Validation Strategies

### The challenges

1. **Only 58 labeled studies** — too few for reliable per-label AUC (SE ≈ 0.07)
2. **131 duplicate reports** — same target vector for multiple studies, must not split across folds
3. **265 scanner fingerprints** — site memorization is real (0.053 AUC gap between random and grouped folds)
4. **Report-image disagreement** — OOF on soft labels understates real performance

### What the top teams do

**5-fold with report grouping** (dominant approach):
- Group studies by identical report text before splitting
- Ensures all studies sharing a report are in the same fold
- Prevents target leakage from duplicate reports

**GroupKFold on scanner fingerprints** (from [zhukovoleksiy](https://www.kaggle.com/code/zhukovoleksiy/rsna-metadata-probe)):
- Cluster studies on Manufacturer + Model + Software + Frequency + Coil
- 265 distinct fingerprints, top 20 covering 45.5% of studies
- Stricter than site holdout (separates individual scanners and software revisions)
- The gap between random and grouped folds measures site memorization

**Out-of-fold (OOF) estimation** (from hida1211):
- The annotated reference for a fold is the annotated studies held out of that fold, pooled across folds
- Previous approaches scored annotated studies including those just trained on — incorrect

### Validation hierarchy

From [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee), the corrected protocol:

1. **OOF** (n=4,407): Selects epochs and detects breakage. Low variance. Right tool for "did this run go wrong."
2. **Gold-58** (n=58): Decides whether a direction is worth pursuing. Noisy (bootstrap CI ≈ ±0.04) but measures against the same ground truth the leaderboard does.
3. **Leaderboard**: Confirmation step, not exploration tool.

**Critical finding**: Both scored systems from sadamtorres landed exactly 0.044 above their gold-58 macro AUC (0.771 → 0.776; 0.824 → 0.866). A third independent system reports the same offset (0.857 gold-58 → 0.903 LB). Three independent systems, one constant.

**This turns the 58-study local check into a leaderboard estimator.** Whatever your local anchor is, measure its offset before spending submissions to find one.

### What NOT to do

- Do NOT tune on gold-58 AUC. The Hanley-McNeil interval at n=58 is ±0.07 per label. Any change smaller than ~0.02 macro AUC is indistinguishable from resampling noise.
- Do NOT use random splits without report grouping — duplicate reports cause target leakage.
- Do NOT trust OOF alone — it systematically understates gains as the model approaches the teacher's ceiling.

---

## 9. External Datasets

### Rules clarification

From [discussion/733652](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652) and [discussion/733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965):

- External data and models are acceptable unless specifically prohibited
- Must be "reasonably accessible to all" and "minimal cost"
- Public datasets (MRNet, fastMRI+, OAI, SKM-TEA) are all free but require click-through agreements — whether this satisfies "equally accessible" is not definitively ruled on
- Commercial LLM APIs are explicitly permitted for label extraction
- Winners must open-source under CC-BY-NC 4.0; external datasets with incompatible licenses are exempted from that requirement

### Datasets in use

| Dataset | Size | Use case |
|---------|------|----------|
| MRNet (Stanford) | ~1,370 studies | Pretraining, external validation |
| fastMRI+ (NYU) | ~1,500 studies | Pretraining |
| Osteoarthritis Initiative (OAI) | ~4,300 subjects | Pretraining, OA labels |
| SKM-TEA | ~48 studies | Pretraining (small but high-quality) |
| RadImageNet | ~1.3M images | Pretrained weights |

### RadImageNet

From [prvsiyan](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee): A RadImageNet ResNet-50 at 20% rank vote improved the ensemble from 0.899 → 0.906. This is the highest verified public score.

### Domain adaptation vs external data

[sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee) achieved 0.883 with NO external predictions blended in — just 5 fine-tuned DINOv2 models. The title claim is that domain adaptation of the self-supervised backbone does more work than bolting on external predictions.

---

## 10. Computational Efficiency

### Efficiency prize

The competition includes an efficiency track. The scoring formula weights AUC against runtime:
- Roughly +0.044 AUC needed per extra hour to break even
- Stage B (336px, +0.017 AUC, +1h) is worth it for the main leaderboard
- Stage A (224px) is better for the efficiency prize

### Runtime budgets

| Component | Typical time | Notes |
|-----------|-------------|-------|
| DICOM header reading | ~90 min for full corpus | Latency-bound on network mount |
| Pixel decoding | ~60-90 min | CPU-bound, parallelize aggressively |
| Training (10 epochs, DINOv2) | ~3-4h per fold | GPU-bound |
| Inference (5 folds) | ~3-4h | Decode once, score 5× |

### Optimization strategies

1. **Cache decoded volumes in memory** (uint8, not float32 — 4× smaller, intensity already normalized to [0,1])
2. **Thread pinning BEFORE numpy/torch import** — OMP/MKL thread counts are read at import time
3. **Read once, train many times** — decode all slots into a numpy array, reuse every epoch
4. **Decode once, score N times** — at inference, pass each study through all ensemble models before moving to the next
5. **Bounded chunk submission** to thread pools — prevents unbounded memory growth
6. **Write a valid submission before any real work** — if the run dies, you still have a scoreable file

### Memory budget

Cache size: `N_studies × N_slots × S × P²` bytes

- Coverage (S) is the cheap axis — linear
- Resolution (P) is the expensive axis — quadratic
- At 336px, 5 slots, 12 slices, 4407 studies: ~9.5 GB uint8

### Kaggle submission constraints

- Internet off during submission
- All dependencies must be bundled as datasets (wheels)
- GPU runtime ~9h budget
- Mixed transfer syntaxes in test set — bundle pylibjpeg, python-gdcm as insurance

---

## 11. Public Leaderboard Landscape

### Verified scores

| Score | Team/Notebook | Approach | Evidence |
|-------|--------------|----------|----------|
| **0.906** | [prvsiyan](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) V52 | RadImageNet ResNet-50, 20% rank vote | Completed Kaggle row 55399732 |
| **0.903** | Yash Bishnoi (external ref) | Five-fold B3 MIL recipe | Inspected, not reproduced |
| **0.899** | [prvsiyan](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) V46 / [salemali7](https://www.kaggle.com/code/salemali7/rsna-knee-90-reports-llm-30-epochs) | Specialist ensemble / LLM labels + 30 epochs | Completed Kaggle row |
| **0.898** | prvsiyan V49 | Five-fold EfficientNet-B3 rank contribution | Completed Kaggle row |
| **0.894** | prvsiyan V14 | Fracture max + OOF-selected target-wise rank blend | Completed Kaggle row |
| **0.891** | [pilkwang](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) V15 | 20-member ensemble, lexicon + LLM labels | Original public anchor |
| **0.883** | [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee) | 5-fold DINOv2 ViT-B/14 @336px, no external blends | Completed Kaggle row |
| **0.866** | sadamtorres Stage A | Same @224px | Completed Kaggle row |
| **0.776** | sadamtorres frozen | Frozen DINOv2 + trained head | Completed Kaggle row |
| **0.598** | [zhukovoleksiy](https://www.kaggle.com/code/zhukovoleksiy/rsna-metadata-probe) | Metadata-only probe (no pixels) | No shortcut found |

### Score ladder

```
0.500  ─── random baseline (all 0.5)
0.598  ─── metadata only (no shortcut)
0.776  ─── frozen DINOv2 + head (no fine-tuning)
0.866  ─── fine-tuned DINOv2 @224px, 5-fold
0.883  ─── fine-tuned DINOv2 @336px, 5-fold
0.891  ─── 20-member ensemble, lexicon + LLM labels
0.899  ─── specialist ensemble, LLM soft labels
0.906  ─── + RadImageNet rank blend (current frontier)
```

### Per-finding performance (from frozen DINOv2 baseline on gold-58)

| Finding | Frozen | Fine-tuned | Gap to frontier |
|---------|--------|-----------|-----------------|
| ACL | 0.727 | 0.840 | ~0.866 |
| MCL | 0.708 | 0.825 | ~0.760 |
| Medial Meniscus | 0.679 | 0.850 | ~0.850 |
| Lateral Meniscus | 0.700 | — | ~0.748 |
| Medial OA | 0.880 | — | ~0.953 |
| Lateral OA | 0.800 | — | ~0.725 |
| PF OA | 0.820 | — | ~0.815 |
| Effusion | 0.870 | — | ~0.880 |
| Synovitis | 0.650 | — | ~0.713 |
| Baker's | 0.900 | — | ~0.909 |
| Contusion | 0.676 | 0.775 | ~0.782 |
| Fracture | 0.660 | — | ~0.838 |

**Weakest labels** (most room for improvement): MCL, Synovitis, Lateral Meniscus, Lateral OA, Contusion

---

## 12. Critical Preprocessing Pitfalls

These are bugs that silently destroy performance. All were discovered and documented by the community.

### 12.1 Slice order

**Bug**: Sorting .dcm files by filename.
**Effect**: Spearman ρ ≈ 0.01 with physical position. Three channels of "2.5D" input become three unrelated views. The "middle of the stack" is a random subset.
**Fix**: Sort by projected physical position: `k = ImagePositionPatient · cross(ImageOrientationPatient[:3], ImageOrientationPatient[3:])`

### 12.2 Physical scale

**Bug**: Resizing without a physical crop.
**Effect**: PixelSpacing varies 3.4× across the corpus. A fixed-pixel resize hands the encoder images whose physical scale differs by a factor of several. Can erase 1-3 mm pathology before the first convolution.
**Fix**: Crop to 130 mm, resize to 336 px. Verify the crop actually fits (if L/s exceeds image width, the crop silently becomes a no-op).

### 12.3 Laterality

**Bug**: Not normalizing left/right knees.
**Effect**: 5 of 12 labels distinguish medial from lateral. If half your studies are left knees and you do nothing, those 5 labels are learned as noise. Nothing crashes. You lose a third of the score without a symptom.
**Fix**: Derive laterality from geometry (sign of image center's x-coordinate in patient space). NOT from ImagePositionPatient corner — use the CENTER: `center = IPP + row_cosine × col_spacing × (Columns/2) + col_cosine × row_spacing × (Rows/2)`. Without this correction, agreement with the tag drops from 96.9% to 58.8%.

**Per-plane normalization**:
- Coronal/Axial: flip last axis horizontally
- Sagittal: reverse slice order (not pixel flip)

**Dead zone**: Within 25 mm of the origin, the sign is no better than chance. Use the tag where present; use geometry outside the dead zone.

### 12.4 No horizontal flip augmentation

Having normalized laterality, random horizontal flip would destroy exactly the information that 5 labels depend on. **Never flip horizontally.** Use small rotation, scale, and shift instead.

### 12.5 Encoder input

**Bug**: Stacking N slices as channels and averaging the first convolution's pretrained weights.
**Effect**: Destroys the input interface the encoder learned.
**Fix**: Feed groups of 3 adjacent slices as 3-channel input through the untouched backbone. Aggregate feature vectors afterwards.

### 12.6 Fluid_Sensitive == Fat_Suppression

**Bug**: Treating these as independent flags and building 4 contrast classes.
**Reality**: They are identical on all 24,371 rows. There are 2 contrast classes, not 4.

### 12.7 Intensity normalization

**Bug**: Per-slice min/max normalization.
**Effect**: One bright vessel compresses the whole dynamic range. Slices lose their relative contrast.
**Fix**: Per-series 1st-99th percentile over the whole sampled stack.

---

## 13. Recommended Implementation Order

### Phase 0: Infrastructure (Day 1) — No GPU needed

**Goal**: Get the data pipeline and label extraction working.

1. **Validate data schema** — run existing `scripts/validate_labels.py`
2. **Implement multilingual regex extractor** — already partially in `src/labels/regex_extractor.py`, extend to all 9 languages with proper Unicode handling (Turkish i, Greek mu, NFKD normalization)
3. **Implement LLM extractor** — use `src/labels/llm_extractor.py` (already exists with GPT-4o). Consider local Qwen3-14B for cost/speed.
4. **Emit graded targets** — not binary. Trace/unqualified/marked → probabilities.
5. **Compute "not addressed" rates** — per finding. Identify which findings have informative vs uninformative silence.
6. **Selective imputation** — for findings where silence is uninformative (Synovitis, PF OA), fill from correlated findings. Never override explicit statements.
7. **Compute confidence weights** — per (study, finding) pair, from the extractor's certainty.
8. **Publish labels as Kaggle dataset** — for use in notebooks.

**Existing code**: `src/labels/`, `scripts/extract_labels.py`, `src/data/loader.py`

### Phase 1: DICOM Pipeline (Days 2-3) — No GPU needed

**Goal**: Correctly load, sort, crop, and normalize all DICOM series.

1. **Implement DICOM loader** — replace stubs in `src/imaging/dicom_loader.py`
   - Read with `stop_before_pixels=True` for headers
   - Handle all transfer syntaxes (bundle pylibjpeg, python-gdcm)
   - Physical position sorting (NOT filename)
   - Laterality from geometry (image center, not corner)
2. **Physical crop + resize** — 130 mm → 336 px
3. **Per-series intensity normalization** — 1st-99th percentile
4. **Slot assignment** — 5 slots (plane × contrast), measure fill rates
5. **Slice sampling** — central band (12-88% for sagittal, 20-80% for coronal/axial)
6. **Cache decoded volumes** — uint8, in memory
7. **Verify on a few studies** — check that left/right normalization works, slices are ordered correctly

**Key files to create**: `src/imaging/dicom_loader.py` (full implementation), `src/imaging/slot_builder.py`, `src/imaging/geometry.py`

### Phase 2: Baseline Model (Days 4-5) — GPU required

**Goal**: Get a working training loop with a frozen DINOv2 backbone.

1. **Load DINOv2 ViT-S/14** (Apache 2.0, bundle as Kaggle dataset)
2. **2.5D input** — groups of 3 adjacent slices as 3-channel input
3. **Per-diagnosis attention head** over slot embeddings
4. **5-fold CV with report grouping** — group identical reports before splitting
5. **Binary cross-entropy loss** with per-finding confidence weights
6. **Gold study weighting** — 5× weight for the 58 expert-labeled studies
7. **Rank-mean ensemble** across folds for submission
8. **Target**: ~0.78-0.80 on gold-58, ~0.77-0.80 on LB

**Key files to create**: `src/experiments/model.py`, `src/experiments/train.py`, `src/experiments/dataset.py`

### Phase 3: Fine-tuning (Days 6-8) — GPU required

**Goal**: Push toward 0.86+ on LB.

1. **Partial unfreezing** — last 6 blocks of DINOv2 trainable
2. **Weight EMA** for epoch selection
3. **No vertical flip** augmentation (knee has fixed superior-inferior orientation)
4. **Small rotation, scale, shift** augmentation only
5. **Pairwise ranking term** — small ranking loss on confident pairs inside each batch
6. **Laterality fallback** — measured before trusted (geometry vs tag agreement)
7. **Target**: ~0.82-0.84 on gold-58, ~0.86-0.88 on LB

### Phase 4: Frontier (Days 9-12) — GPU required

**Goal**: Push toward 0.90+ on LB.

1. **ViT-B/14 backbone** — larger capacity, domain-adapted
2. **RadImageNet pretrained weights** — or domain-adapted weights
3. **Specialist models** for weak labels (Synovitis, MCL, Fracture)
4. **Target-wise rank blending** — per-label model selection
5. **Multi-resolution inference** — 224px + 336px ensemble
6. **EfficientNet-B3 branch** — 5-fold, as an independent architecture for diversity
7. **Target**: 0.90+ on LB

### Phase 5: Hardening (Ongoing)

1. **Monitor per-label AUC** — any label at chance costs 0.029 of the final score
2. **Fracture label problem** — 97% of studies share one value. Fix the label derivation, not the model.
3. **Synovitis from Effusion** — P(syn | eff) = 0.63 vs P(syn | no eff) = 0.22
4. **Submission safety** — always write a valid submission before real work begins
5. **Transfer syntax census** — check first 200 test studies for unexpected codecs

---

## 14. Key Source Links

### Discussions (sorted by votes)

| Title | Author | Votes | Link |
|-------|--------|-------|------|
| Knee Abnormality Detection AI Challenge Overview | Po-Hao "Howard" Chen | 38 | [discussion/733343](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) |
| I got a perfect usecase for this competition | Harsh_2035 | 26 | [discussion/733497](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733497) |
| 0.932 LB within one day. Tested for DICOM metadata shortcut | Oleksii Zhukov | 17 | [discussion/733517](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517) |
| Rules clarification: external datasets, LLM API | Fernando Faria | 16 | [discussion/733652](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652) |
| Possible inconsistencies between MRI reports and labels | Cho Royou | 13 | [discussion/733826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826) |
| Welcome to Knee Abnormality Detection Challenge! | Po-Hao "Howard" Chen | 11 | [discussion/733375](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733375) |
| Use of Commercially Hosted LLMs (HOST ANSWER) | Po-Hao "Howard" Chen | 7 | [discussion/733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) |
| "Not addressed" is a label too — LLM findings | stevenleehans | 5 | [discussion/733932](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932) |
| Using Dino3 | Ryuhki Kimura | 5 | [discussion/733313](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733313) |
| Data/Reporting Inconsistencies | avg-HU | 5 | [discussion/733491](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491) |
| Strong-pipeline replication: SOFT vs HARD targets | FHZ982 | 2 | [discussion/734105](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105) |

### Notebooks (sorted by votes)

| Title | Author | Votes | Score | Link |
|-------|--------|-------|-------|------|
| RSNA Knee baseline v1 | Pilkwang Kim | 300 | 0.891 | [code/pilkwang/rsna-knee-baseline-v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) |
| RSNA Knee: read the report, then the knee | prvsiyan | 198 | 0.906 | [code/prvsiyan/rsna-knee-read-the-report-then-the-knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) |
| RSNA Knee Abnormalities - Efficiency LB | Ryan Holbrook | 135 | — | [code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) |
| RSNA Knee - Data structure, EDA, baseline | Roman Rozen | 93 | — | [code/romanrozen/rsna-knee-data-structure-eda-baseline](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline) |
| RSNA Knee DINOv2 at meniscus resolution | Will (wguesdon) | 83 | — | [code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) |
| [0.899] Let me Cook | AADIGUPTA | 79 | 0.899 | [code/aadigupta7686/0899-let-me-cook](https://www.kaggle.com/code/aadigupta7686/0899-let-me-cook) |
| RSNA Knee - DINOsaur V2 | Roman Tamrazov | 76 | — | [code/romantamrazov/rsna-knee-dinosaur-v2](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2) |
| rsna-knee-enhanced | Sakhawat Hossen | 74 | — | [code/sakhawathossen/rsna-knee-enhanced](https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced) |
| RSNA Knee Solution | KAIWALYA RAUT | 62 | — | [code/kaiwalyaatulraut/rsna-knee-solution](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-solution) |
| RSNA Knee Public 4-fold DINOv2 v4 | hida1211 | 39 | — | [code/hida1211/rsna-knee-public-4-fold-dinov2-v4](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4) |
| Domain adaptation beats resolution: DINOv2 on knee | Sadam Torres | 15 | 0.883 | [code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee) |
| RSNA Knee +90% reports LLM 30 epochs | Salem Ali | 26 | — | [code/salemali7/rsna-knee-90-reports-llm-30-epochs](https://www.kaggle.com/code/salemali7/rsna-knee-90-reports-llm-30-epochs) |
| RSNA Knee: EDA to 2.5D | Karnakbayev Artur | 29 | — | [code/karnakbaevarthursna-knee-eda-to-2-5d](https://www.kaggle.com/code/karnakbaevarthur/rsna-knee-eda-to-2-5d) |
| RSNA Knee DINO Protocol Fusion | Jiachen Li | 6 | — | [code/llccqq624/rsna-knee-dino-protocol-fusion](https://www.kaggle.com/code/llccqq624/rsna-knee-dino-protocol-fusion) |
| RSNA Knee baseline, dual-grouped folds | Stephen Gardner | 3 | — | [code/flight0234/rsna-knee-baseline-dual-grouped-folds](https://www.kaggle.com/code/flight0234/rsna-knee-baseline-dual-grouped-folds) |

### Public label datasets

| Dataset | Author | Method | Link |
|---------|--------|--------|------|
| RSNA Knee LLM Labels | Pilkwang Kim | LLM (first published) | [datasets/pilkwang/rsna-knee-llm-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) |
| RSNA Knee LLM Report Labels | stevenleehans | LLM + not-addressed analysis | [datasets/stevenleehans/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) |
| Stratified Folds & LLM Soft Labels | barun2104 | LLM + fold assignment | [datasets/barun2104/stratified-folds-llm-soft-labels](https://www.kaggle.com/datasets/barun2104/stratified-folds-llm-soft-labels) |

---

## Summary: What Matters Most

Ranked by expected impact on final score:

1. **Label quality** (0.81 regex → 0.88 LLM → 0.89 with selective imputation) — the binding constraint
2. **Physical preprocessing** (slice order, physical crop, laterality) — silent bugs that destroy 0.05-0.10 AUC
3. **Domain adaptation** (fine-tuning DINOv2 on knee MRI) — +0.09 over frozen baseline
4. **Resolution** (336px > 224px > 126px) — +0.017 on top of adaptation
5. **Ensembling** (5-fold rank-mean, multi-architecture) — +0.01-0.02
6. **Slot design** (5 contrast-defined slots vs 6 plane-only slots) — better fill rates
7. **External pretrained weights** (RadImageNet) — +0.007 from 20% rank vote
8. **SOFT vs HARD targets** — inconclusive at +0.014 with CI crossing zero
9. **DINOv3 over DINOv2** — not proven better, licensing complications
10. **Efficiency optimizations** — only matters for the efficiency prize track

---

## 15. Training Strategy Decision (2026-09-07)

Context: data arrives in extraction-sized chunks because of the Kaggle >50 MB
transport bug (`DECISION.md` Decision 18). 172 studies are currently on disk
(51 gold, 121 non-gold). Three candidate training paradigms were evaluated for
both the competition score and the research-paper contribution.

### The candidates

| Track | Paradigm | Knowledge handling | Forgetting risk | Inference cost |
|---|---|---|---|---|
| A — Joint | One model trained on all data at once | All data seen together | None | 1× |
| B — Incremental | One model fine-tuned chunk after chunk | Accumulates but overwrites | High (replay mitigates) | 1× |
| C — Bagged chunks | Every chunk trains its own model; rank-mean blend | Preserved per member | None by construction | k× |

### Verdict: A is mandatory and first; B is the research experiment; C is conditional

**Track A (joint) — YES, execute first. Why:**
- It is the upper bound every other number is compared against
  (`BASELINE_RUNBOOK.md` rule: same split, same gold eval).
- One training run — the cheapest path to the first real number.
- Nothing forces chunked *training*: 172 studies fit trivially in local GPU
  memory. The only real constraint is *transport* (how data arrives), not
  *training*.
- Community evidence (§11 score ladder): frozen caps at 0.776, fine-tuned
  joint training reaches 0.883. Sequential-only strategies start from a
  worse position than joint.

**Track B (incremental fine-tuning) — YES, but as the paper's experiment, not the primary strategy. Why yes:**
- Under genuine transport constraints it is the honest setting, and
  quantifying catastrophic forgetting on knee MRI is a real contribution
  (forgetting curves on a fixed gold-51 eval).
- Replay (mixing a slice of previously-seen studies into each new chunk) is
  the standard mitigation and gives a second curve for the paper.

Why not as primary: naive sequential fine-tuning ends below joint training
(well-documented continual-learning result). Adopting it as the main strategy
sacrifices leaderboard score for no benefit.

**Track C (bagged chunk ensembles / "tree of models") — CONDITIONAL, only once ≥3 extraction chunks exist. Why:**
- "Clubbing models" = averaging predictions, not accumulating knowledge. Each
  member only saw its own chunk, so the blend cannot exceed the members'
  collective knowledge (though it usually beats any single member).
- The tree hierarchy is mathematically redundant: a mean of means is the flat
  mean. Flatten to one rank-mean over all chunk models (reuses
  `src/classical/ensemble.py`, Decision 12). The hierarchy can be reported as
  a negative-result ablation in the paper.
- Strength: no forgetting by construction; robustness story.
- Cost: k× inference and k training runs; with few chunks the ensemble gain
  is small. Greedy ensemble selection on val controls inference cost.

### Alternatives considered and rejected

| Alternative | Why not |
|---|---|
| Wait until all 4,407 studies are extracted, then train | ~570 GB of DICOM, weeks of kernel time; `BASELINE_RUNBOOK.md` forbids full-corpus work before a representative baseline exists |
| Train on community preprocessed volumes (e.g. barun2104 3D) | Does not exercise our preprocessing pipeline; 3D format conflicts with the 2.5D slot design; provenance/license issues for the paper |
| Train only inside Kaggle kernels (no download) | 12–16 h kernel limit + slow debugging kills iteration; right home for final long runs, not for experimentation |
| Feature caching (freeze backbone, cache embeddings once, train heads on cached features) | YES as an *accelerator*, not as the final model — fine-tuning adds ~+0.09 over frozen (§11). Cached features make Track B/C rounds nearly free. Recommended for the ablation experiments once the joint baseline exists. |

### Decision

Execute in order: **A (joint baseline on all 172 studies, local GPU) →
feature-cache accelerator → B (incremental ± replay on new extraction chunks)
→ C (flat bagged blend once ≥3 chunks exist)**. All tracks evaluated on the
identical gold-51 split; every run logged to `RESULTS.md`. Quantum phases
(6–10) start only after the classical baseline is competitive.

**Terminology note (for the paper):** Track B is * continual / incremental
learning* — not federated learning (which means distributed clients sharing
weight updates). Track C is *bagging / ensembling*. Using "incremental
federated learning" for either would be incorrect.

---

## 16. Stage 1 Execution Log (2026-09-09, completed)

Track A's sample-200 run is complete. This section records the reproducible
engineering result and must not be read as evidence of clinical performance.

### Data forensics (resolved)

- The local `volumes_sample200` download was truncated by the >50 MB Kaggle
  download bug (Decision 18 family): 822/839 slot files verified intact via
  magic/size/full-read check; **17 truncated files** quarantined locally;
  1 study lost its only file (a gold study) — 181 usable studies locally.
- **On Kaggle's side the dataset stores individual `.npy` files intact** —
  only the download path truncates. The completed training kernel revalidated
  all 200 studies there successfully.
- The extract kernel (v7) also had a 13 GiB single-zip payload variant not
  visible locally; the training kernel auto-detects either layout
  (individual files or zip) and unzips if needed.

### Infrastructure shipped for Stage 1

1. **Repo-snapshot dataset refreshed** (`mananjain39/rsna-knee-repo-snapshot`) —
   code travels to kernels as a private Kaggle dataset (avoids GitHub push
   dependency; deterministic per version).
2. **Labels dataset created** (`mananjain39/rsna-knee-labels-final`) —
   `labels_final.csv` (4,407 rows, 12 graded labels + conf weights).
3. **Probe gained `--max-studies`** — memory-safe sampled contract check
   (contract is per-study; 8 studies prove the same shapes/masks/finiteness
   at 1/20th the RAM — fixes the local 7 GB-RAM stall).
4. **Stage-1 kernel `train-sample200-v18`** (script kernel, GPU + internet):
   repository install → integrity scan → report-grouped, gold-reserved split
   → B2 GPU probe (8 studies) → B3 2-epoch smoke train → C1 20-epoch global
   main-branch training → gold-58 evaluation → checkpoint publication. Small
   JSON/log artifacts are written to `/kaggle/working/out`.
5. **Stage-1 temporary clustering fallback:** the processed sample did not
   contain the acquisition metadata needed for real clustering. The kernel
   therefore assigned every usable study to synthetic cluster `0`. This is
   schema support for the global main branch only, not a specialist-model or
   hierarchical-clustering result.

### Kaggle platform learnings (2026-09-08, during Stage 1 debugging)

- Kaggle CLI 2.2.4 `kaggle datasets create/version` **skips subdirectories**
  unless directory handling is enabled, and **silently ignores `.zip` uploads**.
  The repository snapshot is therefore published as a ZIP-format payload with
  a non-archive extension (`rsna-knee-repo-snapshot.bundle`) and unzipped in
  the kernel. The final snapshot was verified to contain the current trainer,
  gold-only validation logic, metric import, and `src/experiments` package.
- Kernel mounts are namespaced: `/kaggle/input/competitions/<slug>` and
  `/kaggle/input/datasets/<owner>/<slug>`.
- Kaggle's Python GPU image ships a torch build with **no kernels for Tesla
  P100 (sm60)** and a mismatched torchvision ABI → probe hit
  `no kernel image available` then `torchvision::nms does not exist`.
  Fix: pin `torch==2.5.1 torchvision==0.20.1 --index-url .../whl/cu124`.
- DINOv2-base forward/backward on real 2.5D volumes **passes on P100** with
  this pin. The final probe used 8 studies with tensors shaped
  `(8, 5, 10, 3, 336, 336)`, finite loss `0.9853`, and non-zero head-gradient
  norm `2.6230`.

### Completed Stage-1 results (kernel v18)

- **Integrity:** 200/200 studies usable; 925 slot files scanned; 0 corrupt
  `.npy` files. Slot histogram: 3 slots × 9 studies, 4 slots × 57 studies,
  and 5 slots × 134 studies.
- **Split:** `make_split(..., val_frac=0.20)` first makes a report-grouped
  split, then moves all gold-report groups to validation. The realized split
  is therefore **116 train / 84 validation**, not a conventional 75/25 or
  80/20 split. All **58 gold** studies are in validation and **zero gold**
  studies are in training; the remaining 26 validation studies are non-gold.
  No normalized report group crosses the train/validation boundary.
- **Smoke training:** two epochs completed in 407 seconds. Gold-only
  validation macro AUC was 0.4905 then 0.4897. These finite, scoreable values
  prove that the training/evaluation path runs, but they are near chance and
  are not a performance success.
- **Full training:** the 20-epoch global DINOv2-base run completed in 3,806
  seconds (about 63 minutes). It produced
  `/kaggle/tmp/ckpts/main/best.pt` (692.9 MB), then copied it as
  `/kaggle/working/out/best_main.pt`. The trainer's internal best validation
  macro AUC was 0.5218.
- **Held-out expert evaluation (invalidated on 2026-09-14):** the published
  checkpoint was evaluated on exactly 58 unseen gold studies and reported
  macro AUC **0.5000** with every per-label AUC also 0.5000. This is not a
  valid measurement of model performance: a later inference audit found that
  main-only prediction rank-normalized one study at a time. A one-row rank is
  always zero, so every emitted prediction became identical. The checkpoint
  must be re-evaluated with the corrected inference path before reporting a
  gold-58 model score.
- **Verification:** Python compilation and the local automated suite passed
  (`160 passed`), and all nine Kaggle stages (repository install, integrity,
  split, synthetic-cluster map, GPU probe, smoke train, full train, gold
  evaluation, checkpoint publication) reported `ok`.

### Interpretation and next gate

The sample-200 run is a successful **data, training, and checkpointing
validation**, but the original final-evaluation result is invalid. The model
trainer itself used sigmoid probabilities across the full gold validation set
and recorded an internal best AUC of 0.5218; the later predictor bug destroyed
that ordering only during per-study evaluation. Therefore neither failure nor
success of the model can be claimed until corrected gold-58 inference runs.

### Inference root cause and correction (2026-09-14)

- `ClassicalPredictor.predict_studies` invoked `rank_mean_logits` separately
  for each study, even when only the main model was present.
- `rank_mean_logits` correctly ranks **studies** within a branch, but an input
  with one study has rank zero for every label. This made every main-only
  prediction identical and forced every AUC to 0.5000.
- Main-only inference now applies sigmoid directly to the logits; sigmoid
  preserves their ordering and therefore preserves AUC. A regression test
  verifies that two studies with different logits receive different scores.
- Main-plus-specialist rank blending remains a separate future path and is
  not part of this global main-branch Stage-1 result.

### Revised next gate

1. Re-evaluate the existing `best_main.pt` with corrected main-only inference
   and inspect per-label prediction ranges, UID alignment, and gold-58 AUC.
2. If the original checkpoint cannot be recovered from Kaggle output, rerun
   only the sample-200 global main branch with the corrected snapshot; do not
   change labels or model hyperparameters in that rerun.
3. Keep gold-58 locked for final reporting. Add a report-grouped non-gold
   development split for checkpoint selection rather than selecting epochs on
   gold AUC.
4. Only after a valid corrected baseline is recorded should we compare frozen
   DINOv2-small against partially fine-tuned DINOv2-small on a larger,
   gold-reserved sample.

### Gap analysis — what we figured out (2026-09-21)

We audited our approach against what works in this competition and identified eight critical gaps.
The analysis revealed that label quality, not model architecture, is the real bottleneck.

**1. Label quality is the bottleneck, not the model.** Our LLM prompt is generic (one-liner per finding,
no clinical rules). Top-2.3% teams use clinically-grounded prompts with explicit rules per finding,
iterated against gold-58 by reading errors and refining. Their prompt reached 0.833 mean per-class
accuracy against expert labels; the gap between that and the ~0.88-0.92 ceiling (inherent in report-vs-image
disagreement) is "probably the real limiting factor on this whole approach right now, ahead of anything
on the image side." We need to rewrite our prompt with explicit clinical rules (e.g., Synovitis includes
Hoffa fat pad impingement, plica syndrome), iterate against gold-58, and target 0.83+ accuracy before
the full extraction run.

**2. Label fusion with reliability weights is more robust than single source.** We pick one label source
(stevenleehans, 0.8927 AUC) or blend naively. Top teams fuse multiple sources with per-label reliability
weights (measured against gold-58). When sources disagree, output is a soft label (reliability-weighted
average), not a hard vote. This is more robust than any single source. We also need to audit fusion logic
for symmetry bugs (top team documented a bug where `int()` truncation made one-directional asymmetric errors).

**3. Zero augmentation = overfitting risk.** We have no augmentation (no rotation, no brightness, no jitter).
Top teams use "random rotation, brightness-gamma jitter, and slight rescale to every slice in a series."
With 116-1000 training studies, augmentation is crucial for generalization. Critical: NEVER horizontally
flip (would swap left/right anatomy without updating laterality-specific labels: medial vs lateral meniscus,
MCL, etc.).

**4. No hyperparameter search = using default configuration.** We used default config (DINOv2-base,
10 groups/slot, no search). Top teams did 20 proxy trials on 10% data, 2 epochs each, ranked by blended
score (0.7×CV-AUC + 0.3×gold58-AUC). Proxy predicted 0.6811; full run scored 0.8788 — proxy was rough
guide, not predictor, but selected the right configuration (efficientnet_b3, K=3 series, T=12 slices,
max-pool, medium augmentation). Proxy trials are cheap (2 epochs on 10% data) and find configurations
that full training wouldn't.

**5. Single train/val split is high-variance.** We have a single split (116 train / 84 val with 58 gold).
Top teams use 5-fold GroupKFold, gold excluded from fold pool. Each gold study scored only by the fold
that held it out. Top team caught leakage (0.9357 → 0.8568) by re-scoring each gold study against the
one fold where it was held out. Cross-fitted evaluation is more robust than single split.

**6. float32 volumes exceed Kaggle dataset size cap.** 200 studies = 12 GB float32 → 1,000 studies ≈ 60 GB >
20 GB/dataset cap. Top teams quantize to uint8 + compression: "39GB for 24,371 series." uint8 halves size
(30 GB for 1,000 studies) + compression → ~15 GB. Fits in one dataset. Loader must reproduce exact same
windowing, cropping, resizing, quantization at train time (bit-exact parity).

**7. DINOv2-base (86M params) may be too large for 116-1000 training studies.** Top teams used efficientnet_b3
(12M params) + max-pool + linear head. Proxy trials showed max-pool beat gated attention. With limited data,
simpler model might generalize better. Proxy search will tell us which backbone wins.

**8. No inference robustness (crash safety, degradation).** Top teams' inference kernels have: decode fallback
chain (pydicom → pylibjpeg → gdcm), runtime self-monitoring (degrades from 32→16 slices, then 5→3 models
if behind), crash-safe submission writes every 25 studies, neutral 0.5 for failed DICOMs. Our inference
script is simple with no fallbacks, no degradation, no crash safety.

**What we're doing RIGHT:**
- Per-series percentile windowing (1st-99th) — matches top teams
- No horizontal flip — matches top teams (critical for laterality-specific labels)
- Weighted BCE loss — similar to soft labels approach
- Gold-58 reserved for validation — matches top teams after leak fix
- Confidence weights — similar to reliability-weighted fusion idea

**Recommended priority order:**
1. LLM prompt engineering (critical, $0 cost, biggest impact on label quality)
2. Label fusion (critical, more robust than single source)
3. Augmentation (critical, prevents overfitting)
4. Proxy hyperparameter search (critical, finds best configuration cheaply)
5. Cross-fitted gold-58 evaluation (important, more robust than single split)
6. uint8 quantization (important, solves 1,000-study dataset size issue)
7. Simpler model alternatives (important, proxy search will tell us if it wins)
8. Inference robustness (nice-to-have, crash safety)

Detailed action plans for each gap are in `.md Files/NEXT_STEPS.md` Phases E-L.
### Phase E1: Clinically-grounded LLM prompt (2026-09-21)

**What we did:** Rewrote `src/labels/llm_extractor.py` SYSTEM_PROMPT from generic one-liners to explicit clinical rules per finding.

**Before:** Generic prompt with one-liner definitions (e.g., "ACL - anterior cruciate ligament tear/injury"). No clinical rules, no negation handling, no iteration against gold.

**After:** Clinically-grounded prompt with:
- **Universal rules**: explicit negation = 0.0, unmentioned = 0.0, hedged mentions use judgment, Impression section weighted most heavily, current findings only (not chronic/post-operative)
- **Finding-specific rules**: e.g., Synovitis includes Hoffa fat pad impingement, plica syndrome, friction/impingement syndromes (radiologists describe synovitis indirectly); Contusion requires traumatic pattern (degenerative marrow edema is OA, not contusion); Fracture includes avulsion and insufficiency fractures
- **Temperature 0, fixed JSON output** with all 12 keys

**Why the Kaggle-200 run wasn't sufficient:** The generic prompt likely missed systematic patterns (e.g., Synovitis false negatives from indirect descriptions, Effusion false positives from "trace" vs "some" disagreement). Top-2.3% teams iterated their prompt against gold-58 by reading errors and refining specific patterns, reaching 0.833 mean per-class accuracy. Our generic prompt has no clinical rules and no iteration — label quality is the bottleneck, not the model. The gap between 0.833 and the ~0.88-0.92 ceiling (inherent in report-vs-image disagreement) is "probably the real limiting factor on this whole approach right now, ahead of anything on the image side."

**Next step:** Iterate prompt against gold-58 (Phase E2): run on 58 gold studies, compute per-label accuracy, read errors, refine prompt, target 0.83+ accuracy before full extraction run.


### Phase G: Data augmentation (2026-09-21)

**What we did:** Implemented data augmentation for knee MRI volumes to prevent overfitting.

**Before:** Zero augmentation (no rotation, no brightness, no jitter). With only 116 training studies, the model had no regularization and was at high risk of overfitting to the training set.

**After:** New `src/classical/augmentation.py` module with three augmentations applied identically to every slice in a series (preserving anatomical consistency):
- **Random rotation**: ±10° (small, since knee MRI is already oriented)
- **Brightness-gamma jitter**: gamma in [0.8, 1.2] (simulates intensity variation across scanners/sequences)
- **Slight rescale**: 0.95-1.05× (simulates size variation)
- **CRITICAL: NEVER horizontal flip** — would swap left/right anatomy without updating laterality-specific labels (medial vs lateral meniscus, MCL, etc.)

**Integration:** Modified `KneeMRIDataset` to accept `augment=True/False` flag. Training dataset uses augmentation; validation dataset never augmented (deterministic evaluation). Added `--no-augmentation` flag to `train_classical.py` for ablation studies. Added augmentation config to `configs/default.yaml`.

**Why the Kaggle-200 run wasn't sufficient:** The Kaggle-200 run trained with zero augmentation on only 116 training studies. This is a recipe for overfitting — the model memorizes the training set instead of learning generalizable features. Top-2.3% teams use "random rotation, brightness-gamma jitter, and slight rescale to every slice in a series" as standard practice. With limited data (116-1000 studies), augmentation is crucial for generalization. The Kaggle-200 model likely overfit to the training set, which is why its validation AUC was near chance (0.5218) despite the model architecture being sound.

**Next step:** Ablation study comparing training with vs without augmentation to quantify the regularization effect.


### Phase E3: Resumable, concurrent LLM extraction runner (2026-09-21)

**What:** `LLMExtractor.extract_batch_parallel` (ThreadPool, default 10 workers, per-result callback)
+ `scripts/extract_labels.py` rewritten with `--concurrency`, `--resume`, `--include-gold`, and
incremental JSONL checkpointing (`<output>.jsonl`), one line per completed report.

**Why:** 4,349 multilingual reports at ~3s each is ~3.6 hours sequential; 10 workers cuts it to
~20-40 min. JSONL checkpointing means an interrupted run (laptop sleep, API hiccup) resumes from
exactly where it stopped instead of losing paid-for API work. `--include-gold` extracts the 58
gold studies too — required to measure our extractor's agreement against expert labels (Phase E2)
without conflating it with the fusion step.

**Expected effect:** extraction runs become restartable and cheap to iterate; gold-agreement
measurement becomes a one-flag operation.

**Observed:** compile-verified; live run pending (venv torch download in flight).

### Phase F: Reliability-weighted label fusion (2026-09-21)

**What:** New `src/labels/fusion.py` (`per_label_reliability`, `fuse_labels`, `apply_gold_override`)
+ CLI `scripts/blend_labels.py` + `tests/test_fusion.py` (6 tests).

**Why:** A single label source caps every downstream lever. Fusion design (from the gap analysis):
(1) per-label reliability = each source's per-label AUC against gold-58 (rank-based, matches the
competition metric — a source weak on Synovitis specifically loses Synovitis say, not blanket
weight); (2) per-cell fusion = reliability-weighted average of graded values, with NaN = abstain;
(3) hard-vote snapping when 2+ present sources binarized at 0.5 agree (exact 0/1); (4) gold
override last. Symmetry is asserted by a renaming-invariance regression test — the top-team
`int()`-truncation bug made fusion one-directional (a confident positive could never override a
primary-negative); our implementation has no truncation and the test guards the class of bug.

**Expected effect:** fused labels should beat the best single source on gold-58 macro AUC; the CLI
prints a per-candidate comparison table (each source alone / plain mean / weighted fuse) so the
winner is chosen on evidence, not assumption.

**Observed:** compile-verified; validation run pending (needs our extraction + venv).

### Phase G: Vectorized slot augmentation (2026-09-21)

**What:** New `src/classical/augmentation.py` — one sampled transform (rotation ±10°, gamma jitter
[0.8, 1.2], rescale [0.95, 1.05]) applied to the whole slot stack in a single vectorized bilinear
pass. `KneeMRIDataset` gained `augment=`/`augmentation_config=`; `train_classical.py` gained
`--no-augmentation` (train set augments, validation never does); `configs/default.yaml` gained an
`augmentation` block. `tests/test_augmentation.py` (6 tests) covers shape/range, determinism,
identity-transform exactness, and within-slot transform consistency.

**Why (gap):** zero augmentation on 116 training studies invites memorization. One transform per
slot (a slot = one MRI series) matches the reference policy; slices within a series stay
anatomically consistent. NEVER horizontal flip — five labels are laterality-specific. The first
draft looped per-slice in Python (~150 full-res passes per study); vectorizing over the stack
makes augmentation a negligible fraction of step time.

**Expected effect:** reduced train/val gap once retraining runs; ablation via `--no-augmentation`.

**Observed:** compile-verified; live training effect pending (needs Phase J re-extraction + venv).

### Phase I: Cross-fitted gold-58 CV scaffolding (2026-09-21)

**What:** New `scripts/cross_validate.py` — generates K report-grouped folds over the non-gold
pool (GroupKFold), with gold-58 appended to every fold's val side and never in any train side.
Per fold: leak assertions (no group crosses train/dev, no UID overlap), splits persisted under
`data/processed/splits/<tag>/fold<k>/`. `--run` drives one `train_classical.py` run per fold;
`--study-uids-csv` restricts to an extracted sample.

**Why (gap):** a single split is high-variance and the Stage-1 experience showed how easily an
evaluation path can measure the wrong thing. Cross-fitted scoring (each fold's checkpoint scored
on gold it never trained on) is the robust estimator; the generator asserts the invariants that
the leak audit taught us to check rather than trust.

**Expected effect:** mean ± std gold-58 AUC across folds replaces any single-split number in
results reporting.

**Observed:** compile-verified; fold generation run pending venv; fold training deferred until
after Phase J re-extraction (per next-gate ordering).

### Phase J: uint8 quantized slot storage (2026-09-21)

**What:** `src/imaging/preprocessing.py` gained `to_uint8`/`from_uint8` (round-to-nearest LSB,
max round-trip error 0.5/255 ≈ 0.002). `scripts/preprocess_volumes.py` gained
`--storage-format uint8|float32` (default uint8 → `np.savez_compressed` .npz). The loader
`build_default_volume_index` reads both `.npz` (uint8, restored to float32 [0,1] on load) and
legacy `.npy`, with `.npz` winning when both exist. Tests: round-trip error bound, range
rejection, 4x size gain, npz loading, npz-over-npy precedence.

**Why (gap):** float32 .npy gives ~60 GB per 1,000 studies vs Kaggle's 20 GB/dataset cap — the
hard blocker for scaling training data. uint8 + compression puts 1,000 studies in ~8-15 GB (one
dataset, no chunking), and the loader is format-blind so the existing sample200 dataset keeps
working unchanged. Quantization error (≤ 0.002) is far below the 1st-99th percentile windowing
already applied upstream, and the reference pipeline made the same trade deliberately.

**Expected effect:** re-extraction (next) ships sample200-v2 at ~6 GB; the 1,000-study corpus
fits a single private dataset.

**Observed:** compile-verified; parity/round-trip tests pending venv install completion.

### Environment note (2026-09-21)

The project venv broke during the workspace relocation (`.venv` shebangs pointed at the old
absolute path, and the system's python3.12 disappeared, making the cpython-312 site-packages
unusable). Rebuilt with python3.14 (5.3 GB deps reinstalling from pip cache). Code work above
was compile-verified against python3.14 while the install completes.
