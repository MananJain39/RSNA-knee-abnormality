# Team session — Learning image ML from one real project (knee MRI)
*~45–60 min. Each "slide" = a heading; speaker notes under it. Details & numbers: `LEARNING_JOURNAL.md`.*

---

## 1. The task (3 min)
- Knee MRI → 12 yes/no findings (ACL tear, meniscus tears, arthritis, fluid, fracture…). Multi-label.
- Metric: macro ROC-AUC → only ranking matters; every finding counts 1/12.
- Twist: 4,407 scans, only **58** expert-labelled; others have free-text reports in 9 languages; test = images only.
- Ask the room: *"How would you get labels?"*

## 2. Medical images aren't photos (4 min)
- A study = several 3D series (planes: sagittal/coronal/axial × contrasts), each = 20–45 2D slices; DICOM files.
- Brightness has no absolute meaning; orientation and left/right live in metadata.
- Show `exp/montage_smoke.png`: 4 patients × 5 series after our canonicalisation — they all line up.

## 3. Labels from text with an LLM (8 min)
- Closed vocabulary (negated / silent / below threshold / equivocal / positive) using the *expert definitions*,
  then deterministic mapping to soft labels. LLMs classify well, calibrate badly.
- Bake-off on the 58 gold reports: Haiku 0.859, Sonnet 0.867, Opus 0.871 → difference < noise → pick cheaper.
- "Silence ratio": unmentioned synovitis is still positive 31% of the time → impute; unmentioned Baker's cyst ~3% → don't.
- Result 0.876 vs experts. **Labels set the ceiling.**

## 4. Preprocessing is where silent bugs live (6 min)
- Sort slices by physical position, canonical orientation, mirror left knees, resize, percentile-normalise, uint8.
- Found a reversed left/right sign in the previous code. Verified our flip logic with a self-supervised classifier
  (99.1% on held-out tagged knees).
- Test **parity**: inference preprocessing reproduces training arrays bit-exactly.

## 5. Model: 2.5D CNN (5 min)
- Pretrained ResNet34d on 3 neighbouring slices as RGB → per-series max+mean pooling → concat 5 series → linear(12).
- GPU-side augmentation; no horizontal flips (medial↔lateral!). Series dropout for missing scans.

## 6. War story: the model that learned nothing (8 min) ← the most useful slide
- First model: fancy transformer + per-label attention head → AUC 0.51, loss stuck at "predict the average".
- Debug protocol: (1) `pred_std` ≈ 0 → outputs don't vary; (2) **overfit 48 studies** → couldn't (0.83);
  (3) change one thing at a time, 4 variants in parallel on 4 GPUs → the head was the culprit.
- Simple pooling head: 0.77 after 3 short epochs; overfit test 0.985.

## 7. Validation discipline (5 min)
- 4-fold CV, duplicates grouped; checkpoint chosen on the fold's LLM-label AUC.
- Gold-58 never used for training/selection → honest estimate. Mention noise: ±0.02 macro at n=58.

## 8. Results (5 min)
| | Macro AUC |
|---|---|
| LLM labels vs experts | 0.876 |
| Single image model vs experts | 0.891 |
| 4-fold ensemble vs experts | **0.905** |
| Out-of-fold (4,349 studies, vs LLM labels) | 0.886 → 0.891 with 2-model ensemble |
- **The model beats its labels**: noisy labels at scale average out (Effusion 0.83 → 0.98).
- More slices (24 vs 16) didn't help; ensembling different models did.
- Kaggle late submission: see journal §12.4 for the leaderboard score.

## 9. Engineering that made it fast (3 min)
- Data next to compute: 265 GB downloaded to a 4-GPU box at 550 MB/s; decoded straight from the zip on 46 cores in 5 min.
- Long jobs as services (survive disconnects); smoke test 5 samples before every big run.
- Cost: ~$30 LLM labels + ~$5/h GPU.

## 10. Takeaways (3 min)
1. Understand the metric. 2. Measure label quality. 3. LLMs as fixed-vocabulary classifiers.
4. Look at your data; test train/inference parity. 5. Simplest model first; overfit test when stuck.
6. Keep an untouched test set. 7. Noisy labels at scale beat few clean ones. 8. Ensembles = cheapest gain.
