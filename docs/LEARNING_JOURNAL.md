# Learning Journal — Detecting Knee Problems in MRI Scans with Deep Learning

> **Who this is for:** someone who knows basic ML (what training/validation is, what a classifier is,
> what overfitting means) but has never worked with medical images, deep-learning vision models, or
> noisy labels. Every concept is explained the first time it appears. Read top to bottom.
>
> **How to use it:** each step has *What we did*, *Why*, *What we observed*, and *Lesson*. The
> "Lesson" boxes are the transferable ideas you'll reuse on any future image project.

---

## Table of contents
0. [The problem in plain English](#0)
1. [Background: what an MRI scan actually is](#1)
2. [How the competition is scored (AUC) — and why that changes everything](#2)
3. [Auditing the previous attempt](#3)
4. [Exploratory data analysis on the tables](#4)
5. [Making labels from radiology reports with an LLM](#5)
6. [Compute infrastructure — why one big GPU machine](#6)
7. [Getting the data (265 GB) onto the machine](#7)
8. [Preprocessing: from raw DICOM files to clean arrays](#8)
9. [The model architecture, piece by piece](#9)
10. [Training: loss, optimiser, learning rate, augmentation](#10)
11. [Validation without fooling ourselves](#11)
12. [Results and experiments log](#12)
13. [Glossary](#13)

---

<a id="0"></a>
## 0. The problem in plain English

A hospital takes an MRI scan of a patient's knee. A radiologist (a doctor specialised in reading
scans) looks at it and writes a report: *"Complete tear of the ACL. Small joint effusion. Medial
meniscus intact."* The goal is to build a computer program that looks **only at the scan** and says,
for each of 12 possible problems, how likely it is to be present.

The 12 problems ("findings" or "labels"):

| Label | What it is, in simple terms |
|---|---|
| ACL | Tear of the anterior cruciate ligament — the main ligament inside the knee that stops the shin sliding forward. Classic sports injury. |
| MCL | Tear of the medial collateral ligament — the ligament on the inner side of the knee. |
| Medial Meniscus | Tear of the inner C-shaped cartilage "shock absorber" between thigh bone and shin bone. |
| Lateral Meniscus | Same, outer side. |
| Medial OA | Osteoarthritis (wear-and-tear cartilage loss) in the inner compartment. |
| Lateral OA | Osteoarthritis in the outer compartment. |
| PF OA | Osteoarthritis behind the kneecap (patello-femoral joint). |
| Effusion | Too much fluid inside the joint (a "swollen knee"). |
| Synovitis | Inflamed joint lining. |
| Baker's | A fluid-filled cyst at the back of the knee. |
| Contusion | Bone bruise — bleeding/swelling inside the bone without a crack. |
| Fracture | A break in the bone. |

**This is "multi-label classification":** a knee can have several problems at once (an ACL tear very
often comes with a lateral bone bruise and an effusion). So we train one model with 12 outputs, each an
independent yes/no probability — *not* one softmax over 12 classes.

**The twist:** we have 4,407 training scans, but only **58** of them were labelled by expert
radiologists. For the other 4,349 we only have the *report text* (in ~9 languages). The test set (~1,300
scans, hidden on Kaggle) has **no reports**, only images. So the pipeline is forced to be:

```
   report text  ──(language model reads it)──►  training labels
        +                                           │
   MRI images  ─────────────────────────────►  train an IMAGE model
                                                    │
   test MRI images (no text) ───────────────►  predictions
```

---

<a id="1"></a>
## 1. Background: what an MRI scan actually is

**Pixels with no fixed meaning.** A photo pixel means a colour. An MRI pixel's brightness depends on
the tissue *and* on the machine settings. Two hospitals scanning the same knee give different
brightness values. ⇒ We normalise every scan's brightness independently (Step 8).

**3D, built from 2D slices.** The scanner takes a stack of thin 2D images ("slices"), each ~3–4 mm
thick, ~20–45 slices per stack. One stack = one **series**.

**One exam = several series ("a study").** Each study has 3–14 series, which differ in:
- **Plane** (the direction of the slices):
  - *Sagittal* — slices from the side, like cutting the knee left-to-right. Best view for ACL and menisci.
  - *Coronal* — slices from the front. Best for MCL, menisci, inner/outer compartments.
  - *Axial* — horizontal slices from top to bottom. Best for the kneecap, Baker's cyst, effusion.
- **Contrast** (how the scanner is tuned):
  - *Fluid-sensitive / fat-suppressed* (names like "PD FS", "T2 FS", "STIR"): water is bright,
    fat is dark. Tears, fluid, bone bruises "light up". Great for detecting abnormalities.
  - *Structural* (T1 / PD without fat suppression): shows anatomy detail well.

**DICOM** is the file format: one `.dcm` file per slice, containing pixels + hundreds of metadata tags
(patient position, orientation, pixel size, scanner model, which knee — left/right…). Our data:
**819,078 DICOM files, 265 GB**.

**Lesson:** in medical imaging, the *metadata* is as important as the pixels — it tells you slice order,
orientation, and which side of the body you're looking at.

---

<a id="2"></a>
## 2. How it is scored (AUC) — and why that changes everything

**ROC-AUC, intuitively.** Take every pair (one patient who has an ACL tear, one who doesn't). AUC is the
fraction of pairs where the model gave the sick patient the higher score.
- AUC 0.5 = random guessing. AUC 1.0 = perfect ranking. 0.85 = "85% of pairs ranked correctly".
- **Macro AUC** = compute AUC separately for each of the 12 findings, then average them.

**Three consequences we design around:**
1. **Only ranking matters.** If the model outputs 0.9 vs 0.8 or 0.09 vs 0.08, AUC is identical.
   So: no threshold tuning, no calibration work. When combining models we can average scores freely.
2. **Each finding weighs 1/12.** A rare finding (MCL, ~10–16% positive) counts as much as a common one.
   A model that ignores rare findings loses a lot.
3. **Small test sets are noisy.** With 58 gold patients and ~9 MCL positives, a single AUC can move
   ±0.05 by luck. We must not over-interpret small differences.

---

<a id="3"></a>
## 3. Auditing the previous attempt

**What we did:** read the previous folder (`rsna-knee/`): docs, code, git history, experiment logs.

**What we observed:**
- Very good research notes (community findings, label definitions).
- The model had never trained on real data at scale. Every logged validation AUC was exactly
  **0.0, 0.5 or 0.52** — these are red flags: 0.5 exactly means the model outputs constants;
  0.0 means a bug (e.g. labels misaligned with predictions).
- A bug in left/right knee detection: DICOM uses "LPS" coordinates where **+x = the patient's LEFT**.
  The old code treated negative x as left — reversed.
- Over-engineered plans (per-scanner specialist models, "quantum ML") before any baseline existed.

**Lesson — "baseline first":** get the simplest end-to-end pipeline working and producing a sane number
before adding complexity. A suspiciously round metric (0.5, 0.0, 1.0) almost always means a bug.

---

<a id="4"></a>
## 4. Exploratory data analysis (EDA) on the tables

**What we did:** `scripts/eda.py` → output saved in `exp/eda_csv.txt`. We studied the CSV files before
touching images (cheap, fast, and it shapes every later decision).

| What we found | Number | Why it matters / what we did about it |
|---|---|---|
| Gold studies | 58 of 4,407 (1.3%) | Too few to train on; we use them only as an honest test. |
| Gold is enriched with positives | ACL 41% in gold vs ~18% overall | Gold is not representative; AUC is still OK because AUC doesn't depend on the positive rate. |
| Languages | EN 40%, ES 17%, TR 12%, EL 7%, DE 6%, NL 3%, … | Need a multilingual text reader → LLM. |
| Identical report text in different studies | 177 rows | If duplicates fall into both train and validation, validation is "leaky" (too optimistic). We keep them in the same fold. |
| Series per study | median 5 (3–14) | Need a rule to choose which series to use. |
| Axial structural series present | only 19% of studies | Dropped that slot entirely. |
| Coronal structural present | 77% | Kept, but the model must handle it being missing. |
| More than one candidate series for a slot | 13.6% of slots | Take the one with most slices (more coverage). |
| `Fluid_Sensitive` vs `Fat_Suppression` columns | identical in 100% of rows | Effectively one flag. |

**Lesson:** EDA isn't plotting for its own sake — each finding should translate into a design decision.

---

<a id="5"></a>
## 5. Making labels from radiology reports with an LLM

Folder: `exp/01_labels/` (README + results). Scripts: `llm_labels.py`, `score_labels.py`, `build_labels.py`.

### 5.1 Why not keywords?
Reports say the same thing in endless ways and languages ("rotura completa del LCA",
"ön çapraz bağda tam kat yırtık", "ACL not visualised, consistent with complete tear"). And negations
("no evidence of meniscal tear") contain the same keywords. Keyword rules reach ~0.81 AUC vs experts;
LLMs reach ~0.87–0.89 (community results).

### 5.2 The label definitions matter
The experts used specific thresholds. Example: "small effusion" is **negative** for them (only
moderate/large counts); a low-grade MCL sprain is **negative** (only high-grade tears count). If our labels
said "effusion = yes" for every mention, the model would learn the wrong concept. So we wrote the expert
definitions into the LLM instructions.

### 5.3 Closed vocabulary instead of probabilities
LLMs are good at *choosing from fixed options*, bad at producing calibrated numbers. So for each of the
12 findings, the LLM picks one of 5 **states**:

| State | Meaning | Mapped to |
|---|---|---|
| `negated` | report explicitly says normal/absent | 0.02 |
| `not_mentioned` | report is silent | 0.10 |
| `below_threshold` | something is there but below the expert definition (small effusion, sprain) | 0.30 |
| `equivocal` | "possible", "cannot exclude", size not stated | 0.60 |
| `positive` | clearly meets the definition | 0.95 |

These numbers are **soft labels**: instead of forcing 0 or 1, the target carries our uncertainty. The
model is trained to output ~0.6 for an "equivocal" report rather than being punished for not saying 1.
(Technically, binary cross-entropy works fine with targets between 0 and 1.)

The LLM also returns a short **evidence quote** for each state, so we can audit its decisions. Output is
forced to valid JSON with a schema, so nothing has to be parsed by hand.

### 5.4 Choosing the LLM scientifically (a "bake-off")
We ran three Claude models on the 58 gold reports and measured agreement with experts:

| Model | Macro AUC vs experts | Cost for all 4,349 reports |
|---|---|---|
| Haiku 5.5 (small) | 0.859 | ~$1.5 |
| Sonnet 5.5 (medium) | 0.867 | ~$28 |
| Opus 5.5 (large) | 0.871 | ~$52 |

The Sonnet→Opus gain (+0.004) is far smaller than the noise from only 58 samples (~±0.02), so paying
2× for Opus isn't justified. **Lesson:** compare options on a held-out set and weigh the gain against the
measurement noise before spending money.

### 5.5 Two measured improvements
1. **Ensemble of two readers** (70% Sonnet + 30% Haiku). Different models make different mistakes;
   averaging cancels some. Biggest gain on Contusion (0.76 → 0.81).
2. **"Silence" handling for synovitis.** On the gold set, when a report doesn't mention synovitis,
   31% of those knees still have it — radiologists often just don't comment on it. So silence is
   *uninformative*. Synovitis strongly co-occurs with effusion, so for silent reports we fill the synovitis
   score from the effusion score. For Baker's cyst, silence means "absent" (only 3% positive when silent),
   so we leave it alone. Synovitis: 0.705 → 0.777.

**Final label quality: 0.876 macro AUC vs experts** (best published community label set ≈ 0.887).

**Why this number matters so much:** the image model learns to imitate these labels. If labels agree
with experts at ~0.88, the image model will typically land a few points *below* that on expert labels.
Label quality is a ceiling.

**Honest caveat:** we chose the blend weights and the imputation rule by looking at gold-58, so 0.876 is
slightly optimistic (this is "tuning on the test set" — small here because we tried only ~8 variants).

---

<a id="6"></a>
## 6. Compute infrastructure — why one big GPU machine

- **The data (265 GB) never touches the laptop.** Everything runs on one cloud machine.
- **Machine:** AWS `g6e.12xlarge` = 4× NVIDIA L40S GPUs (48 GB memory each), 48 CPU cores, 384 GB RAM,
  3.5 TB NVMe SSD. Bought as a "spot" instance (spare capacity, ~$5/hr instead of ~$10.5/hr; the risk is
  AWS can reclaim it, so we back up results to S3 storage).
- **Why one machine with 4 GPUs instead of several small ones?** Data stays in one place (no copying),
  and the 4 GPUs train 4 cross-validation folds at the same time.
- **Security:** the machine has *no* open network ports; we send it commands via AWS Systems Manager.

**Lesson:** for image ML, put compute next to the data. Moving hundreds of GB is often slower than the
training itself.

---

<a id="7"></a>
## 7. Getting the data onto the machine

- Kaggle provides one 265 GB zip. Downloaded with `aria2c` using 16 parallel connections →
  **~550 MB/s, ~8 minutes**. (A normal single-connection download is often 5–10× slower.)
- We had a hiccup: the first download process was killed when our remote session closed. Fix: run long
  jobs as **system services** (`systemd-run`) so they survive disconnects, and use resumable downloads.
- **We never unzip.** Our preprocessing reads each DICOM file directly out of the zip. That saves ~30
  minutes and 265 GB of extra disk.

**Lesson:** long-running jobs must be detached from your terminal session and resumable.

---

<a id="8"></a>
## 8. Preprocessing: from raw DICOM files to clean arrays

Script: `scripts/preprocess.py`. Neural networks want fixed-size arrays of numbers that look
consistent across patients. Raw DICOMs are none of that. Each step below fixes one source of chaos:

**8.1 Pick one series per "slot".** We define 5 slots = (plane, contrast):
`sag_fs, sag_t1, cor_fs, cor_t1, ax_fs`. When several series compete for one slot, take the one with
the most slices. Missing slots are allowed (filled with zeros + a "missing" flag).

**8.2 Sort slices by physical position.** Files in a folder are not in anatomical order. Each DICOM
records where the slice is in 3D space (`ImagePositionPatient`) and its orientation
(`ImageOrientationPatient`). We compute each slice's position along the stack direction and sort.

**8.3 Canonical orientation.** Scanners store images with different axis directions (one hospital's
image may be the mirror or transpose of another's). Using the orientation metadata, we rotate/flip every
image so that, e.g., sagittal images always show the front of the knee on the left and the head up.
*Why:* otherwise the model must waste capacity learning that "the same thing, mirrored" is the same.

**8.4 Left/right knee normalisation (laterality).** A left knee is a mirror image of a right knee.
"Medial" (inner side) is on opposite sides of the image. We mirror left knees to look like right knees,
so medial is always on the same side — this matters because 4 labels are medial-vs-lateral.
*Consequence:* we must **never** use random horizontal flips as augmentation (that would swap medial and
lateral and corrupt the labels). *Known weakness (see Results):* the left/right tag is missing in ~48% of
studies.

**8.5 Resize to 256×256.** Images arrive at 256–512 px; networks need one size. We pad to square, then
resize with area-averaging (which avoids jagged artefacts).

**8.6 Intensity normalisation.** Clip brightness to the 0.5th–99.5th percentile of that series and
rescale to 0–255. *Why:* MRI brightness has no absolute meaning (see §1); percentiles remove the effect
of a few extremely bright pixels.

**8.7 Store as uint8 and fixed depth of 32 slices.** uint8 (0–255 integers) is 4× smaller than float32,
with no meaningful loss after normalisation. Every series is resampled to exactly 32 slices so batches
have a fixed shape. Final dataset: 4,407 × 5 × 32 × 256 × 256 bytes = **46 GB**, kept fully in RAM.

**8.8 Smoke test first.** Before running on 4,407 studies we ran 5, checked array shapes, checked that
the left/right decision agreed with series names ("RT. PD_TSE_SAG", "LT_pd_tse_fs_cor"), and **looked at
the images** (`exp/montage_smoke.png`): one row per patient, one column per slot — all patients oriented
the same way. Then the full run: 4,407 studies, 46 CPU cores, **4.7 minutes, 0 failures**.

**Lesson:** always *look* at your preprocessed data. Many silent bugs (wrong orientation, all-black images,
wrong slice order) are obvious to the eye and invisible in summary statistics.

---

<a id="9"></a>
## 9. The model architecture, piece by piece

Script: `scripts/train.py` (class `KneeNet`). The model must take up to 5 series × 32 slices and output 12
probabilities. We build it from standard blocks:

**9.1 Backbone = a pretrained 2D CNN ("transfer learning").** A CNN (convolutional neural network)
turns an image into a feature vector (e.g. 640 numbers) describing its content. Training one from
scratch needs millions of images; we have ~4k studies. So we start from a CNN pretrained on ImageNet
(14M everyday photos). Its early layers already detect edges, textures and shapes, which transfer
surprisingly well to medical images; we then *fine-tune* all its weights on our data.
Candidates we benchmarked (training speed on one L40S): ResNet34d ~2,660 img/s, EfficientNetV2-B0
~1,890 img/s, ConvNeXt-Nano ~1,520 img/s. We use small models because with few, noisy labels big models
don't help much and train 3–5× slower.

**9.2 "2.5D" input.** A pretrained CNN expects 3-channel (RGB) images. We feed 3 *neighbouring slices*
as the 3 channels. The CNN then sees a sliver of 3D context (useful for e.g. a tear that spans slices)
while still using ImageNet weights. Fully 3D CNNs exist but are slower and have fewer good pretrained
weights.

**9.3 Slice sampling.** Each series has 32 slices; we feed 16 (every 2nd slice, with a random starting
offset during training). This halves compute, still covers the whole knee, and the random offset acts as
augmentation.

**9.4 Transformer over slices and series.** After the CNN, each slice is a feature vector ("token").
We add two learned embeddings to each token: *which series it came from* and *where in the stack it
is*. A small **transformer** (2 layers) lets every token look at every other token — so evidence from the
sagittal view and the coronal view can be combined. Missing series are masked out (ignored).

**9.5 Per-finding attention head.** We create 12 learned "query" vectors, one per finding. Each query
uses **attention** to compute a weighted average of the tokens — effectively learning *which slices to
look at* for that finding (the ACL query should focus on central sagittal slices; Baker's on posterior
axial ones). Each query's result becomes one logit (score) → sigmoid → probability.

```
5 series × 16 slices (×3 neighbours)  →  CNN  →  80 tokens  →  + series/position embedding
   →  Transformer (2 layers)  →  12 per-finding attention queries  →  12 probabilities
```

---

<a id="10"></a>
## 10. Training: loss, optimiser, learning rate, augmentation

- **Loss = binary cross-entropy (BCE)** for each of the 12 outputs, against the soft labels.
- **Optimiser = AdamW** with weight decay 0.02 (a mild penalty on large weights → less overfitting).
- **Two learning rates:** 3e-4 for the pretrained CNN (it's already good; change it gently),
  1e-3 for the new transformer/head (random start; needs to learn fast).
- **Schedule:** linear *warm-up* for the first ~300 steps (large updates at the start can wreck pretrained
  weights), then *cosine decay* to zero (smaller steps later → settle into a good minimum).
- **Mixed precision (bfloat16)** — ~2× faster on modern GPUs with no accuracy loss.
- **Gradient clipping** at norm 2.0 — stops rare huge updates from destabilising training.
- **Augmentation** (random changes to training images so the model can't memorise them), done on the GPU:
  - small rotation (±12°), zoom (±12%), shift (±8%) — the same transform for all slices of one series;
  - brightness, contrast and gamma jitter — simulates different scanners;
  - **series dropout**: 30% of the time hide one series — the model learns to cope when a series is
    missing (test studies have missing series too);
  - **no horizontal flip** (would swap medial/lateral, §8.4).
- **Batch:** 6 studies per step (≈ 480 slice-images through the CNN).

---

<a id="11"></a>
## 11. Validation without fooling ourselves

**Cross-validation (CV).** Split the 4,349 report-labelled studies into 4 *folds*. Train 4 models; each
model trains on 3 folds and is checked on the 4th. Every study gets exactly one "out-of-fold" prediction
from a model that never saw it. Studies with identical reports are kept in the same fold (§4).

**Two validation numbers per model:**
1. **Val AUC vs LLM labels** (~1,090 studies per fold): large sample, but measured against *our noisy
   labels*. Used to pick the best epoch (checkpoint).
2. **Gold-58 AUC vs expert labels**: small sample but the *true* target. Never used for training or for
   choosing checkpoints — otherwise it would stop being an honest estimate.
   Each of the 4 fold models predicts the 58 gold studies; averaging the 4 = our **ensemble** estimate
   (and the final Kaggle submission is exactly such an average).

**Lesson — leakage:** any information from the evaluation set that influences training or model
choice makes the evaluation optimistic. Keep one set that you look at as rarely as possible.

---

<a id="12"></a>
## 12. Results and experiments log

| ID | What | Status | Val AUC (LLM labels) | Gold-58 AUC (experts) |
|---|---|---|---|---|
| 01 | Report → label extraction (LLM) | ✅ done | — | **0.876** (labels themselves) |
| A1 | ConvNeXt-Nano + attention head | ❌ collapsed (see 12.1) | 0.51 | 0.46 |
| B1 | **ResNet34d + pool head**, 16 slices, 4 folds × 10 epochs | ✅ done | **0.886** (OOF) | **0.905** (4-fold ens.), 0.891 single |
| B2 | EfficientNetV2-B0 + pool head (ensemble member) | ⏳ fold 1 re-running (OOM) | folds 0/2/3: 0.873–0.893 | per fold 0.887–0.896 |
| B4 | ResNet34d + pool, **24 slices**, seed 1 | ✅ done | 0.885 | 0.901 (ens.), 0.889 single |
| B1+B4 | rank-mean ensemble of 8 models | ✅ | **0.891** | 0.905 |
| B5 | ConvNeXt-Nano + pool, lr 1e-4 (ensemble member) | ⏳ running | | |

### 12.2 What B1 tells us (first working model)
- **Gold-58 0.905 with the 4-fold ensemble, 0.891 for a single model.** The image model scores *higher* vs the
  experts than the labels it was trained on (0.876). Why: each individual report label is noisy, but the model sees
  ~3,300 of them; errors in different reports point in random directions and average out, while the true visual
  pattern is consistent. This is the "learning from noisy labels" effect and the most important idea in this project.
- Largest gains over the labels: Effusion 0.83 → 0.98, Fracture 0.78 → 0.90, Contusion 0.81 → 0.94 — exactly the
  findings radiologists describe inconsistently in text but which are very visible on fluid-sensitive images.
- Weakest: Synovitis 0.76, Lateral Meniscus 0.82.
- **Ensembling the 4 folds adds +0.014** over a single model — free accuracy since we train the folds anyway.

### 12.3 What B4 tells us (more slices)
24 slices per series instead of 16 → OOF 0.885 vs 0.886, gold 0.901 vs 0.905: **no gain, 50% more compute**.
Lesson: 16 evenly spaced slices already cover the knee; more resolution along the slice axis isn't the bottleneck.
But B1+B4 together (different seeds/sampling) → OOF **0.891** (+0.005): ensembling different models helps more than
making one model bigger.

### 12.4 Kaggle submission
- Weights + code uploaded as a private Kaggle dataset; inference notebook runs **offline** (Kaggle code competitions
  forbid internet), re-runs our exact preprocessing (verified bit-identical to training), averages all fold models,
  writes `submission.csv` every 25 studies (crash safety), predicts 0.5 if a study fails.
- B1 submitted as a late submission (competition closed 2026-10-22, so it is scored but not ranked).
  **Score: public LB 0.917, private LB 0.905.** The private score equals our gold-58 estimate (0.905) — our
  validation was honest, which is the main thing a validation set is for.
- Ensemble B1+B2+B4 (12 models; gold-58 0.908, OOF 0.893) submitted as #2.

### 12.4b Where that puts us — honest context
Final public leaderboard (6,425 teams): **median 0.942, top-100 ≥ 0.961, winner 0.966**. Our 0.917 would be
rank ~4,220 (≈ 66th percentile). The often-quoted "18th place = 0.903" blog post was written *early* in the
competition; the field improved a lot afterwards. So: a sound, honest pipeline that beats the previous attempt
by a mile (no working model → 0.905), but **not competitive with the top** yet.

**Why the gap is probably this big — and what we would try next (in order of expected gain):**
1. **Image resolution.** We train at 224 px crops of 256 px images. Meniscus and cartilage findings are a few
   pixels wide at that size; top solutions typically use 384–512 px. Biggest single lever, ~3× compute.
2. **Training length / model size.** 10 epochs of ResNet34d; validation AUC was still rising at epoch 9. Top teams
   train 20–40 epochs and use larger backbones (ConvNeXt-S/B, EfficientNet-B4+, ViTs) or 3D CNNs.
3. **Gold-label / test-label mismatch.** Public LB (0.917) > our gold (0.905) > OOF (0.893): the hidden test
   labels may follow the report-style definitions more closely than gold-58 does → label quality (more
   readers, better prompt per finding) feeds straight into LB.
4. **Per-finding views / crops** (e.g. a meniscus-focused crop of the coronal series), test-time augmentation.
Each of these is a ~1-hour experiment on the same pipeline; the infrastructure doesn't need to change.

### 12.5 Compute and cost
| Item | Cost |
|---|---|
| LLM labels (Sonnet + Haiku on 4,407 reports + gold bake-off with Opus) | ~$30 |
| GPU box g6e.12xlarge spot ~$5.3/h | ~$5/h while running |
| S3, data transfer | <$1 |

### 12.6 Comparison with the previous attempt (the benchmark)
| | Previous attempt | This project |
|---|---|---|
| Labels | public label set, gold overwritten (couldn't be evaluated) | own LLM labels, 0.876 vs experts, measured |
| Data used for training | ≤200 studies | all 4,349 |
| Best validation AUC | 0.0 / 0.5 (degenerate) | 0.886 OOF, 0.905 gold |
| Kaggle submission | none | submitted |
| Compute | laptop + free Kaggle T4 | 4× L40S |

**Could they have done it with their resources?** Very likely yes, mostly. The decisive parts were *process*, not
hardware: measured labels, a simple head, an overfit test, CV discipline. Compute-wise, B1 = ~1.5 L40S-GPU-hours;
a Kaggle T4 is ~6–8× slower → ~10–12 T4-hours, within Kaggle's 30 h/week free quota (as 1–2 folds per 9-h session).
The real blocker for them was data access: preprocessing 265 GB inside Kaggle's 20 GB output limit needs uint8 +
cropping (≈26 GB for our version — would need splitting into 2 datasets or fewer slices). We can verify this by
re-running B1 on a T4-sized budget (planned follow-up).

### 12.7 Teaching summary — the 8 ideas to take away
1. **Understand the metric first** (AUC = ranking; macro = every label counts equally).
2. **Labels are the ceiling** — measure label quality against the best ground truth you have.
3. **Use LLMs as classifiers over a fixed vocabulary**, not as probability generators.
4. **Look at your data** after preprocessing (montage) and **test parity** between training and inference code.
5. **Simplest model first**; when it fails, check `pred_std`, run the **overfit test**, change one thing at a time.
6. **Keep an untouched test set** (gold-58) — never select models on it.
7. **Noisy labels at scale beat few clean labels** — the model ends up better than its labels.
8. **Ensembles of different models/folds** are the cheapest reliable gain.
| Lat | Self-supervised left/right classifier | ✅ diagnostic | 99.1% acc on tagged | not applied |

*(This table and the per-experiment write-ups below are updated as runs finish.)*

### 12.1 First real training run (A1) failed — and how we debugged it

**What happened.** A1 (ConvNeXt-Nano + transformer + per-finding attention head, learning rate 3e-4)
trained for 2 epochs (~11 min). Validation AUC was **0.51**, gold AUC **0.46** — i.e. random. The training
loss stalled at ~0.53.

**How to read that.** If a model predicted the *same* probability for every patient (each finding's
average rate), the BCE loss would be about 0.52–0.55 for our label distribution. A loss stuck there +
AUC ≈ 0.5 means **the model collapsed to a constant** — it learned the average and nothing else.

**Debugging method: change one thing at a time, in parallel.** We stopped A1 and ran four short
diagnostics, one per GPU (each on 1,200 training studies, 3 epochs):

| Run | Change | Val AUC after 3 ep | Spread of predictions across patients (`pred_std`) |
|---|---|---|---|
| D0 | lower LR 1e-4 | 0.63 | 0.0004 |
| D1 | much lower LR 3e-5 | 0.66 | 0.039 |
| D2 | different backbone (ResNet34d) | 0.66 | 0.0000 |
| D3 | **overfit test**: 48 studies, 25 epochs, check *training* AUC | train AUC only **0.83** | 0.044 |

**Interpretation.**
- The *overfit test* is the single most useful sanity check in deep learning: a correct model+pipeline
  can always memorise a tiny dataset (training AUC → ~1.0). Ours couldn't ⇒ something structural blocks
  learning, not just a bad hyper-parameter.
- `pred_std ≈ 0` across all variants ⇒ outputs barely vary between patients regardless of backbone/LR.
- Common factor = the head. In the attention head, the same learned query vector is *added* back
  (residual) and then LayerNorm'd; since the query is identical for every patient and the attention
  output starts tiny, the normalised result is ~the same for everyone. Plus the final weights started at
  exactly zero. Signal from the image is drowned out → very slow learning.

**Fix being tested (E-runs):** a simpler "pool" head — for each series, take the max and the mean of slice
features (max = "is there a lesion on *any* slice?", mean = overall appearance), concatenate the 5
series, one linear layer → 12 outputs. Also a repaired attention head (no query residual), and a new
overfit test.

**Lesson:** when a model doesn't learn, (1) check whether predictions vary at all between inputs,
(2) run the overfit-a-tiny-subset test, (3) change one component at a time. Fancy heads are a common
source of silent failure; start with the simplest pooling and add complexity only when it measurably helps.

### Known issues / next experiments
- **Laterality is a coin-flip for ~48% of studies** (no left/right tag; position fallback unreliable
  because knees are scanned at the machine's centre). Planned fix: train a small classifier on the 52% of
  studies with a reliable tag to predict left vs right from the image itself, then re-normalise the rest.

---

<a id="13"></a>
## 13. Glossary

- **AUC / ROC-AUC** — probability that a random positive is ranked above a random negative (§2).
- **Augmentation** — random, label-preserving changes to training inputs to reduce overfitting.
- **Backbone** — the pretrained feature extractor part of a network.
- **BCE** — binary cross-entropy, the standard loss for yes/no outputs.
- **Checkpoint** — saved model weights at some point in training.
- **CNN** — convolutional neural network; the standard architecture for images.
- **DICOM** — medical image file format (pixels + metadata).
- **Ensemble** — averaging the predictions of several models; usually more accurate than any single one.
- **Epoch** — one pass over the whole training set.
- **Fine-tuning** — continuing to train a pretrained model on your own data.
- **Fold** — one of the K parts in K-fold cross-validation.
- **Gold labels** — labels from experts (here: 58 studies), treated as ground truth.
- **Laterality** — left vs right side of the body.
- **Leakage** — evaluation info sneaking into training → over-optimistic results.
- **Logit** — the raw model score before the sigmoid turns it into a probability.
- **Multi-label** — several labels can be true at once (vs multi-class: exactly one).
- **Series / Study / Slice** — one 3D stack / one patient exam / one 2D image.
- **Soft labels** — targets between 0 and 1 expressing uncertainty.
- **Spot instance** — discounted cloud machine that can be reclaimed.
- **Transfer learning** — reusing a model trained on one task/data as a starting point for another.
- **Transformer / attention** — layers that let every element look at every other and weight them.
