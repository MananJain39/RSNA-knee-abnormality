# RSNA Knee Abnormality Detection — Data & Project Explained

> A comprehensive guide to understanding the competition data, structure, and challenge.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [What You're Predicting](#2-what-youre-predicting)
3. [The Data Files](#3-the-data-files)
4. [The 12 Knee Abnormalities](#4-the-12-knee-abnormalities)
5. [MRI Image Structure](#5-mri-image-structure)
6. [The Core Challenge](#6-the-core-challenge)
7. [The Evaluation Metric](#7-the-evaluation-metric)
8. [Why This Is Hard](#8-why-this-is-hard)
9. [The Solution Pipeline](#9-the-solution-pipeline)

---

## 1. The Big Picture

### What is this competition?

This is a **medical imaging Kaggle competition** where you build AI to detect 12 different knee abnormalities from MRI scans.

### The unique twist

**Almost none of the training data comes pre-labeled.** You have to read radiology reports to create your own labels, then train an imaging model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE COMPETITION IN ONE IMAGE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   TRAINING DATA                    TEST DATA                                 │
│   ─────────────                    ─────────                                 │
│                                                                              │
│   4,407 knee MRI studies           ~1,300 knee MRI studies                   │
│   ├── 58 have expert labels        ├── No labels                             │
│   ├── 4,349 have reports only      ├── No reports (!!)                       │
│   └── All have MRI images          └── Only MRI images                       │
│                                                                              │
│   YOUR JOB:                                                                 │
│   1. Read the 4,349 reports → extract labels                                │
│   2. Train an imaging model on all 4,407 studies                            │
│   3. Predict 12 abnormalities for each test study                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why is this interesting?

In the real world, radiologists write reports but don't label every finding in a machine-readable way. This competition simulates that reality: you have the reports, you have the images, but you don't have clean labels for most of your data.

---

## 2. What You're Predicting

For each knee MRI study, you must predict **12 probabilities** — one for each abnormality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION OUTPUT FORMAT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   StudyInstanceUID          ACL   MCL   MedialMeniscus   ...   Fracture     │
│   ──────────────────        ───   ───   ──────────────         ─────────    │
│   1.2.826.0.1.368...        0.85  0.12  0.73             ...   0.05         │
│   1.2.826.0.1.368...        0.23  0.91  0.15             ...   0.67         │
│   ...                                                                       │
│                                                                             │
│   Each number is a probability from 0.0 to 1.0                              │
│   Higher = more confident the abnormality is PRESENT                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Data Files

### 3.1 `train.csv` — The Training Studies

**4,407 rows**, one per knee MRI study.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              train.csv STRUCTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Columns:                                                                   │
│   ─────────                                                                  │
│   • StudyInstanceUID    Unique identifier for the study                      │
│   • Report              Free-text radiology report (any of 9 languages)      │
│   • ACL                 ┐                                                    │
│   • MCL                 │                                                    │
│   • Medial Meniscus     │                                                    │
│   • Lateral Meniscus    │  12 label columns                                  │
│   • Medial OA           │  Each is:                                          │
│   • Lateral OA          │    • 0 = abnormality absent                        │
│   • PF OA               │    • 1 = abnormality present                       │
│   • Effusion            │    • NaN = not labeled (need to extract)           │
│   • Synovitis           │                                                    │
│   • Baker's             │                                                    │
│   • Contusion           │                                                    │
│   • Fracture            ┘                                                    │
│                                                                              │
│   Example rows:                                                              │
│   ─────────────                                                                │
│   Study 1 (GOLD):  UID=1.2.826...  Report="..."  ACL=1  MCL=0  ...         │
│   Study 2 (GOLD):  UID=1.2.826...  Report="..."  ACL=0  MCL=0  ...         │
│   ...                                                                        │
│   Study 59:        UID=1.2.826...  Report="..."  ACL=NaN MCL=NaN ...        │
│   Study 60:        UID=1.2.826...  Report="..."  ACL=NaN MCL=NaN ...        │
│   ...                                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### The Two Types of Studies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GOLD STUDIES (58 total)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   These have been labeled by TWO expert radiologists who independently       │
│   reviewed the images and reached consensus.                                 │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  StudyInstanceUID: 1.2.826.0.1.3680043.8.498.100956877...         │   │
│   │                                                                      │   │
│   │  Report (Spanish, 1199 chars):                                       │   │
│   │  "Antecedentes Clínicos: Esguince rodilla.                           │   │
│   │   Hallazgos: No hay alteraciones de señal significativas...          │   │
│   │   Amputación marginal del cuerpo del menisco lateral...              │   │
│   │   Fina úlcera condral focal... Fenómenos condrales reparativos..."   │   │
│   │                                                                      │   │
│   │  Labels:                                                             │   │
│   │    ACL=0  MCL=0  MedialMeniscus=0  LateralMeniscus=0                 │   │
│   │    MedialOA=0  LateralOA=0  PF_OA=1  Effusion=1                      │   │
│   │    Synovitis=0  Baker's=0  Contusion=0  Fracture=0                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   These are your GROUND TRUTH — the only clean labels you have.              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     UNLABELED STUDIES (4,349 total)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   These have reports but NO expert labels. You must extract labels from      │
│   the report text yourself.                                                  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  StudyInstanceUID: 1.2.826.0.1.3680043.8.498.100048732...         │   │
│   │                                                                      │   │
│   │  Report (Spanish, 322 chars):                                        │   │
│   │  "Técnica: RMN de la rodilla.                                        │   │
│   │   Resultados: Rotura de menisco interno.                             │   │
│   │   Signo de necrosis avascular subcondral en el cóndilo femoral       │   │
│   │   medial. Artrosis femorotibial medial. Derrame."                    │   │
│   │                                                                      │   │
│   │  Labels:                                                             │   │
│   │    ACL=NaN  MCL=NaN  MedialMeniscus=NaN  LateralMeniscus=NaN         │   │
│   │    MedialOA=NaN  LateralOA=NaN  PF_OA=NaN  Effusion=NaN              │   │
│   │    Synovitis=NaN  Baker's=NaN  Contusion=NaN  Fracture=NaN           │   │
│   │                                                                      │   │
│   │  → You need to parse this report and extract:                        │   │
│   │    MedialMeniscus=1 (rotura de menisco interno)                      │   │
│   │    MedialOA=1 (artrosis femorotibial medial)                         │   │
│   │    Effusion=1 (derrame)                                              │   │
│   │    Contusion=1? (necrosis avascular subcondral — maybe?)             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   This is where your NLP/LLM skills come in.                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 `train_series.csv` — The MRI Series Metadata

**24,371 rows**, one per MRI series.

Each study contains multiple MRI series — different "camera angles" and settings.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           train_series.csv STRUCTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Columns:                                                                   │
│   ─────────                                                                  │
│   • StudyInstanceUID    Links to train.csv                                   │
│   • SeriesInstanceUID   Unique ID for this series                            │
│   • Anatomical_Plane    Sagittal / Coronal / Axial                           │
│   • Fluid_Sensitive     1 = fluid shows bright (T2/PD/STIR)                  │
│   • Fat_Suppression     1 = fat signal suppressed                            │
│                         ⚠️  IDENTICAL to Fluid_Sensitive!                    │
│                                                                              │
│   Example:                                                                   │
│   ─────────                                                                  │
│   Study A has 5 series:                                                      │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ Series 1: Sagittal, Fluid=1    (side view, fluid bright)             │  │
│   │ Series 2: Sagittal, Fluid=0    (side view, anatomy detail)           │  │
│   │ Series 3: Coronal,  Fluid=1    (front view, fluid bright)            │  │
│   │ Series 4: Coronal,  Fluid=0    (front view, anatomy detail)          │  │
│   │ Series 5: Axial,    Fluid=1    (cross-section, fluid bright)         │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   Statistics:                                                                │
│   ───────────                                                                │
│   • Mean series per study: 5.5                                               │
│   • Range: 3 to 14 series per study                                          │
│   • Every study has all 3 planes                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Series Fill Rates by Plane and Contrast

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HOW OFTEN DOES EACH SLOT EXIST?                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Plane × Contrast              Studies with this slot    Fill rate          │
│   ──────────────────            ──────────────────────    ─────────          │
│                                                                              │
│   Sagittal + Fluid-sensitive    4,150 / 4,407             94.2%  ████████▋  │
│   Sagittal + Non-fluid          4,266 / 4,407             96.8%  █████████▏ │
│   Coronal  + Fluid-sensitive    4,248 / 4,407             96.4%  █████████▏ │
│   Coronal  + Non-fluid          3,406 / 4,407             77.3%  ███████▎   │
│   Axial    + Fluid-sensitive    4,407 / 4,407            100.0%  ██████████  │
│   Axial    + Non-fluid            857 / 4,407             19.4%  █▊         │
│                                                                              │
│   ⚠️  Axial non-fluid is almost never present (only 1 in 5 studies)          │
│   ⚠️  Most teams DROP this slot and redistribute its budget                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 `train_series/` — The Actual DICOM Images (~570 GB)

**NOT DOWNLOADED** — this is the raw pixel data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         train_series/ DIRECTORY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Structure:                                                                 │
│   ──────────                                                                 │
│   train_series/                                                              │
│   ├── 1.2.826.0.1.3680043.8.498.100048732.../    ← StudyInstanceUID        │
│   │   ├── 1.2.826.0.1.3680043.8.498.123431.../   ← SeriesInstanceUID       │
│   │   │   ├── slice001.dcm                             ← One 2D image      │
│   │   │   ├── slice002.dcm                             ← One 2D image      │
│   │   │   ├── slice003.dcm                             ← One 2D image      │
│   │   │   └── ...                                      ← ~30 slices total  │
│   │   ├── 1.2.826.0.1.3680043.8.498.138212.../       ← Another series     │
│   │   │   └── ...                                                          │
│   │   └── ...                                                              │
│   ├── 1.2.826.0.1.3680043.8.498.100264108.../        ← Another study      │
│   │   └── ...                                                              │
│   └── ...                                                                  │
│                                                                              │
│   Each .dcm file is one 2D grayscale image (e.g., 256×256 pixels)            │
│   A series is a stack of ~20-45 slices forming a 3D volume                   │
│   Total size: ~570 GB compressed                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 `test.csv` and `test_series.csv` — The Test Data

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TEST DATA STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   test.csv (3 example rows, will be ~1,300 at scoring)                       │
│   ──────────                                                                 │
│   StudyInstanceUID                                                           │
│   ──────────────────                                                         │
│   1.2.826.0.1.3680043.8.498.100470350575...                                 │
│   1.2.826.0.1.3680043.8.498.100628617831...                                 │
│   1.2.826.0.1.3680043.8.498.100675147070...                                 │
│                                                                              │
│   ⚠️  NO Report COLUMN! You cannot use text at inference time.               │
│                                                                              │
│   test_series.csv (15 example rows)                                          │
│   ────────────────────                                                       │
│   Same schema as train_series.csv:                                           │
│   StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane,                     │
│   Fluid_Sensitive, Fat_Suppression                                           │
│                                                                              │
│   sample_submission.csv                                                      │
│   ─────────────────────                                                      │
│   Shows the expected output format:                                          │
│   StudyInstanceUID, ACL, MCL, MedialMeniscus, ..., Fracture                  │
│   All values set to 0.5 (dummy baseline)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The 12 Knee Abnormalities

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE 12 KNEE ABNORMALITIES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   LIGAMENTS (2)                    MENISCI (2)                               │
│   ─────────────                    ─────────                                 │
│   ┌─────────────────────┐          ┌─────────────────────┐                  │
│   │ ACL                 │          │ Medial Meniscus     │                  │
│   │ Anterior Cruciate   │          │ Medial meniscus     │                  │
│   │ Ligament tear       │          │ tear                │                  │
│   │                     │          │                     │                  │
│   │ Gold prevalence:    │          │ Gold prevalence:    │                  │
│   │ 24/58 = 41%         │          │ 26/58 = 45%         │                  │
│   └─────────────────────┘          └─────────────────────┘                  │
│   ┌─────────────────────┐          ┌─────────────────────┐                  │
│   │ MCL                 │          │ Lateral Meniscus    │                  │
│   │ Medial Collateral   │          │ Lateral meniscus    │                  │
│   │ Ligament tear       │          │ tear                │                  │
│   │                     │          │                     │                  │
│   │ Gold prevalence:    │          │ Gold prevalence:    │                  │
│   │ 9/58 = 16%          │          │ 23/58 = 40%         │                  │
│   └─────────────────────┘          └─────────────────────┘                  │
│                                                                              │
│   OSTEOARTHRITIS (3)               OTHER FINDINGS (5)                        │
│   ────────────────                 ──────────────────                        │
│   ┌─────────────────────┐          ┌─────────────────────┐                  │
│   │ Medial OA           │          │ Effusion            │                  │
│   │ Medial compartment  │          │ Joint effusion /    │                  │
│   │ cartilage loss      │          │ excess fluid        │                  │
│   │                     │          │                     │                  │
│   │ Gold prevalence:    │          │ Gold prevalence:    │                  │
│   │ 15/58 = 26%         │          │ 35/58 = 60%         │                  │
│   └─────────────────────┘          └─────────────────────┘                  │
│   ┌─────────────────────┐          ┌─────────────────────┐                  │
│   │ Lateral OA          │          │ Synovitis           │                  │
│   │ Lateral compartment │          │ Synovial            │                  │
│   │ cartilage loss      │          │ inflammation        │                  │
│   │                     │          │                     │                  │
│   │ Gold prevalence:    │          │ Gold prevalence:    │                  │
│   │ 11/58 = 19%         │          │ 27/58 = 47%         │                  │
│   └─────────────────────┘          └─────────────────────┘                  │
│   ┌─────────────────────┐          ┌─────────────────────┐                  │
│   │ PF OA               │          │ Baker's Cyst        │                  │
│   │ Patellofemoral      │          │ Popliteal cyst      │                  │
│   │ cartilage loss      │          │ behind knee         │                  │
│   │                     │          │                     │                  │
│   │ Gold prevalence:    │          │ Gold prevalence:    │                  │
│   │ 21/58 = 36%         │          │ 12/58 = 21%         │                  │
│   └─────────────────────┘          └─────────────────────┘                  │
│                                    ┌─────────────────────┐                  │
│                                    │ Contusion           │                  │
│                                    │ Bone bruise /       │                  │
│                                    │ marrow edema        │                  │
│                                    │                     │                  │
│                                    │ Gold prevalence:    │                  │
│                                    │ 19/58 = 33%         │                  │
│                                    └─────────────────────┘                  │
│                                    ┌─────────────────────┐                  │
│                                    │ Fracture            │                  │
│                                    │ Acute cortical      │                  │
│                                    │ break               │                  │
│                                    │                     │                  │
│                                    │ Gold prevalence:    │                  │
│                                    │ 18/58 = 31%         │                  │
│                                    └─────────────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prevalence Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LABEL PREVALENCE (58 GOLD STUDIES)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Label                  Positives    Prevalence    Bar                      │
│   ─────────────────      ─────────    ──────────    ───                      │
│   Effusion               35/58        60%           ████████████████████▊    │
│   Medial Meniscus        26/58        45%           ███████████████▊         │
│   Synovitis              27/58        47%           ████████████████▏        │
│   ACL                    24/58        41%           ██████████████▏          │
│   Lateral Meniscus       23/58        40%           █████████████▊           │
│   PF OA                  21/58        36%           ████████████▍            │
│   Contusion              19/58        33%           ███████████▍             │
│   Fracture               18/58        31%           ██████████▊              │
│   Medial OA              15/58        26%           █████████                │
│   Baker's                12/58        21%           ███████▏                 │
│   Lateral OA             11/58        19%           ██████▌                  │
│   MCL                     9/58        16%           █████▍                   │
│                                                                              │
│   Average findings per study: 4.1 (range: 1-9)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Most Common Co-occurrences

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FINDINGS THAT OFTEN OCCUR TOGETHER                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Pair                                    Studies with both                 │
│   ────                                    ─────────────────                 │
│   Effusion + Synovitis                    22                                │
│   Medial Meniscus + Effusion              19                                │
│   ACL + Effusion                          18                                │
│   Lateral Meniscus + Effusion             16                                │
│   Medial Meniscus + Synovitis             15                                │
│   PF OA + Synovitis                       14                                │
│   PF OA + Effusion                        14                                │
│   Lateral Meniscus + Synovitis            13                                │
│   Effusion + Fracture                     13                                │
│   Effusion + Contusion                    13                                │
│   ACL + Medial Meniscus                   13                                │
│   ACL + Contusion                         13                                │
│                                                                              │
│   💡 Effusion is the "hub" — it co-occurs with almost everything            │
│   💡 This matters for label imputation strategies                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. MRI Image Structure

### What is an MRI Study?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ANATOMY OF A KNEE MRI STUDY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   A "study" is one knee, scanned in one session.                             │
│   It contains multiple "series" — different viewing angles and settings.     │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         STUDY (one knee)                             │   │
│   │                                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │ SERIES 1: Sagittal, Fluid-sensitive                         │   │   │
│   │   │ ~30 slices, each ~256×256 pixels                            │   │   │
│   │   │ Shows: ACL, menisci, patellofemoral cartilage               │   │   │
│   │   │                                                              │   │   │
│   │   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐              │   │   │
│   │   │  │slice│ │slice│ │slice│ │slice│ ... │slice│              │   │   │
│   │   │  │  1  │ │  2  │ │  3  │ │  4  │     │ 30  │              │   │   │
│   │   │  └─────┘ └─────┘ └─────┘ └─────┘     └─────┘              │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │ SERIES 2: Sagittal, Non-fluid (T1-weighted)                 │   │   │
│   │   │ ~30 slices                                                   │   │   │
│   │   │ Shows: Anatomy detail, cartilage                            │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │ SERIES 3: Coronal, Fluid-sensitive                          │   │   │
│   │   │ ~25 slices                                                   │   │   │
│   │   │ Shows: MCL, meniscal bodies, medial/lateral compartments    │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │ SERIES 4: Coronal, Non-fluid                                │   │   │
│   │   │ ~25 slices                                                   │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │ SERIES 5: Axial, Fluid-sensitive                            │   │   │
│   │   │ ~20 slices                                                   │   │   │
│   │   │ Shows: Patellofemoral joint, synovium, Baker's cysts        │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │   (Maybe) SERIES 6: Axial, Non-fluid                                │   │
│   │   Only present in 19% of studies — often dropped                    │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Three Anatomical Planes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE THREE VIEWING PLANES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        ┌─────────────────────┐                               │
│                        │      SAGITTAL       │                               │
│                        │     (Side view)     │                               │
│                        │                     │                               │
│                        │    ┌───────────┐    │                               │
│                        │    │   Femur   │    │                               │
│                        │    │ (thigh    │    │                               │
│                        │    │  bone)    │    │                               │
│                        │    └─────┬─────┘    │                               │
│                        │          │          │                               │
│                        │    ┌─────┴─────┐    │                               │
│                        │    │   Menisci │    │  ← Best for: ACL, menisci,    │
│                        │    └─────┬─────┘    │     patellofemoral cartilage  │
│                        │          │          │                               │
│                        │    ┌─────┴─────┐    │                               │
│                        │    │   Tibia   │    │                               │
│                        │    │  (shin    │    │                               │
│                        │    │   bone)   │    │                               │
│                        │    └───────────┘    │                               │
│                        └─────────────────────┘                               │
│                                                                              │
│   ┌─────────────────────┐              ┌─────────────────────┐               │
│   │      CORONAL        │              │       AXIAL         │               │
│   │    (Front view)     │              │   (Cross-section)   │               │
│   │                     │              │                     │               │
│   │  ┌─────┐   ┌─────┐  │              │    ┌───────────┐    │               │
│   │  │Med. │   │Lat. │  │              │    │  Patella  │    │               │
│   │  │condyle│ │condyle│ │              │    │ (kneecap) │    │               │
│   │  └──┬──┘   └──┬──┘  │              │    └─────┬─────┘    │  ← Best for:  │
│   │     │         │     │              │          │          │     PF joint,   │
│   │  ┌──┴─────────┴──┐  │              │    ┌─────┴─────┐    │     synovium,   │
│   │  │   Menisci     │  │              │    │  Femoral  │    │     Baker's     │
│   │  └──┬─────────┬──┘  │              │    │  condyles │    │     cysts       │
│   │     │         │     │              │    └─────┬─────┘    │               │
│   │  ┌──┴──┐   ┌──┴──┐  │              │    ┌─────┴─────┐    │               │
│   │  │Med. │   │Lat. │  │              │    │   Tibia   │    │               │
│   │  │plateau│ │plateau│ │              │    │  plateau  │    │               │
│   │  └─────┘   └─────┘  │              │    └───────────┘    │               │
│   └─────────────────────┘              └─────────────────────┘               │
│                                                                              │
│   ← Best for: MCL, meniscal bodies,                                          │
│              medial/lateral OA                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fluid-Sensitive vs Non-Fluid Sequences

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO CONTRAST TYPES PER PLANE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   FLUID-SENSITIVE (T2/PD with fat suppression)                               │
│   ────────────────────────────────────────────                               │
│   • Fluid, edema, and tears appear BRIGHT (white)                            │
│   • Fat signal is suppressed (dark)                                          │
│   • This is where MOST abnormalities are detected                            │
│   • Present in ~95% of studies for each plane                                │
│                                                                              │
│   Example: A meniscal tear appears as a bright line reaching the surface     │
│                                                                              │
│   NON-FLUID (T1-weighted)                                                    │
│   ────────────────────────                                                   │
│   • Better for anatomy detail                                                │
│   • Cartilage visualization                                                  │
│   • Present in ~77-97% of studies                                            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  PLANE        FLUID-SENSITIVE    NON-FLUID    BEST FOR               │   │
│   │  ─────        ───────────────    ─────────    ────────               │   │
│   │  Sagittal     ✓ 94%              ✓ 97%        ACL, menisci, PF OA    │   │
│   │  Coronal      ✓ 96%              ✓ 77%        MCL, meniscal bodies   │   │
│   │  Axial        ✓ 100%             ✗ 19%        PF joint, Baker's      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ⚠️  Fluid_Sensitive and Fat_Suppression are IDENTICAL columns!             │
│       There are only 2 contrast types, not 4.                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Physical Scale Variation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE PHYSICAL SCALE PROBLEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   All series cover ~150mm field of view, but at different resolutions:       │
│                                                                              │
│   Scanner A: 960×960 pixels at 0.1562 mm/pixel    → 150mm FOV               │
│   Scanner B: 896×896 pixels at 0.1674 mm/pixel    → 150mm FOV               │
│   Scanner C: 800×800 pixels at 0.1875 mm/pixel    → 150mm FOV               │
│   Scanner D: 640×640 pixels at 0.2344 mm/pixel    → 150mm FOV               │
│                                                                              │
│   PROBLEM: A 3mm meniscal tear occupies:                                     │
│   • Scanner A: 3 / 0.1562 = 19 pixels                                       │
│   • Scanner D: 3 / 0.2344 = 13 pixels                                       │
│                                                                              │
│   If you just resize to 224×224 without physical cropping:                   │
│   • Scanner A: 224 / (960/19) = 4.5 pixels for the tear                     │
│   • Scanner D: 224 / (640/13) = 4.5 pixels for the tear                     │
│   → Same size, but you've lost resolution information                        │
│                                                                              │
│   SOLUTION: Crop to constant physical extent (130mm) then resize:            │
│   • All studies → 130mm crop → 336 pixels → 0.387 mm/pixel                  │
│   • A 3mm tear = 7.7 pixels (consistent across scanners)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Core Challenge

### The Pipeline You Must Build

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE REQUIRED PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STAGE 1: TEXT → LABELS (Training only)                              │   │
│   │ ─────────────────────────────────────────────                       │   │
│   │                                                                      │   │
│   │  4,349 reports in 9 languages                                        │   │
│   │         │                                                            │   │
│   │         ▼                                                            │   │
│   │  ┌─────────────────┐                                                 │   │
│   │  │ Label Extractor │  ← Regex, LLM, or hybrid                       │   │
│   │  │ (NLP/LLM model) │                                                 │   │
│   │  └────────┬────────┘                                                 │   │
│   │           │                                                          │   │
│   │           ▼                                                          │   │
│   │  4,349 × 12 extracted labels (with confidence weights)               │   │
│   │                                                                      │   │
│   │  Combine with 58 gold labels (weighted higher)                       │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STAGE 2: IMAGES → MODEL (Training)                                  │   │
│   │ ─────────────────────────────────────                               │   │
│   │                                                                      │   │
│   │  4,407 studies × ~5 series × ~30 slices = ~660,000 DICOM images      │   │
│   │         │                                                            │   │
│   │         ▼                                                            │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │ Image Preprocessing                                          │    │   │
│   │  │ • Physical slice ordering (NOT filename)                     │    │   │
│   │  │ • Laterality normalization (left/right knee)                 │    │   │
│   │  │ • Physical crop (130mm) → resize (336px)                     │    │   │
│   │  │ • Intensity normalization (per-series percentiles)           │    │   │
│   │  │ • Slot assignment (plane × contrast)                         │    │   │
│   │  └───────────────────────────┬─────────────────────────────────┘    │   │
│   │                              │                                      │   │
│   │                              ▼                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │ Imaging Model (e.g., DINOv2 + attention head)                │    │   │
│   │  │ • 2.5D input: groups of 3 adjacent slices as 3 channels      │    │   │
│   │  │ • Per-diagnosis attention over slots                         │    │   │
│   │  │ • Trained with confidence-weighted BCE loss                  │    │   │
│   │  └───────────────────────────┬─────────────────────────────────┘    │   │
│   │                              │                                      │   │
│   └──────────────────────────────┼──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STAGE 3: MODEL → PREDICTIONS (Inference)                            │   │
│   │ ──────────────────────────────────────────────                      │   │
│   │                                                                      │   │
│   │  ~1,300 test studies (images only, NO reports)                       │   │
│   │         │                                                            │   │
│   │         ▼                                                            │   │
│   │  Same preprocessing + model → 12 probabilities per study             │   │
│   │                                                                      │   │
│   │  ⚠️  Text is NOT available at this stage!                            │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. The Evaluation Metric

### Macro-Averaged AUC

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOW YOU'RE SCORED                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Final Score = (1/12) × Σ AUC_i                                            │
│                   i=1 to 12                                                 │
│                                                                              │
│   For each of the 12 labels:                                                 │
│   • Rank all test studies by predicted probability                           │
│   • Compute AUC: probability that a random positive ranks above random       │
│     negative                                                                 │
│   • Average all 12 AUCs                                                      │
│                                                                              │
│   Example:                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Label              AUC      Contribution                            │   │
│   │ ─────              ───      ────────────                            │   │
│   │ ACL                0.85     0.85/12 = 0.0708                        │   │
│   │ MCL                0.78     0.78/12 = 0.0650                        │   │
│   │ Medial Meniscus    0.82     0.82/12 = 0.0683                        │   │
│   │ ...                                                                 │   │
│   │ Fracture           0.71     0.71/12 = 0.0592                        │   │
│   │ ─────────────────────────────────────────                           │   │
│   │ MACRO AUC          0.80     ← Your final score                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Three Critical Implications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   WHAT THE METRIC MEANS FOR YOUR STRATEGY                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. ONLY RANKING MATTERS                                                    │
│   ───────────────────────                                                    │
│   AUC is invariant under any strictly increasing transform.                  │
│   • Your predictions can be 0.51 or 0.99 for a positive — doesn't matter    │
│   • Only that positives score HIGHER than negatives                         │
│   • Calibration is worthless                                                │
│   • Thresholds are worthless                                                │
│   • For ensembling: use RANK averaging, not probability averaging           │
│                                                                              │
│   2. EVERY LABEL COSTS THE SAME                                              │
│   ─────────────────────────────                                              │
│   If you leave one label at chance (AUC = 0.5):                              │
│   • You lose (M - 0.5) / 12 of your final score                             │
│   • At M = 0.85, that's 0.029 — HUGE on a competitive leaderboard           │
│   • Rare labels (MCL, 16% prevalence) deserve MORE attention than common    │
│     ones (Effusion, 60% prevalence)                                         │
│                                                                              │
│   3. PREVALENCE DRIFT IS SURVIVABLE                                          │
│   ──────────────────────────────────────                                     │
│   AUC is invariant to positive rate in expectation.                          │
│   • The competition warns prevalence may differ across train/public/private │
│   • This is fine for AUC                                                    │
│   • But don't tune any thresholds — they won't transfer                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Why This Is Hard

### The Five Key Challenges

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WHY THIS COMPETITION IS HARD                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   CHALLENGE 1: LABEL NOISE                                                   │
│   ────────────────────────                                                   │
│   Reports and expert annotations disagree ~10-15% of the time.               │
│                                                                              │
│   Example:                                                                   │
│   Report says: "mild joint effusion"                                         │
│   Expert label: 0 (negative)                                                 │
│   Why? Annotators only marked SIGNIFICANT effusions as positive.             │
│                                                                              │
│   → Your extracted labels are imperfect. The imaging model can potentially   │
│     beat its own teacher by learning from images, not reports.               │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   CHALLENGE 2: MULTILINGUAL REPORTS                                          │
│   ────────────────────────────────────────                                   │
│   Reports come in 9 languages: English, Spanish, French, German, Dutch,      │
│   Bulgarian, Turkish, Greek, and others.                                     │
│                                                                              │
│   Example:                                                                   │
│   Spanish: "Rotura de menisco interno" → Medial Meniscus = 1                 │
│   German:  "Meniskusriss medial" → Medial Meniscus = 1                       │
│   French:  "Déchirure du ménisque médial" → Medial Meniscus = 1              │
│                                                                              │
│   → Your extractor must handle all languages.                                │
│   → Don't identify language first — test all cues simultaneously.            │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   CHALLENGE 3: "NOT ADDRESSED" AMBIGUITY                                     │
│   ──────────────────────────────────────────                                 │
│   25.4% of label cells are the report simply not mentioning a finding.       │
│                                                                              │
│   For some findings, silence means ABSENT:                                   │
│   • Baker's cyst: P(positive | silent) = 3%                                  │
│   • Silence IS the label                                                     │
│                                                                              │
│   For others, silence means UNINFORMATIVE:                                   │
│   • Synovitis: P(positive | silent) = 34%                                    │
│   • 84% of synovitis cells are silent — nobody looks for it                  │
│                                                                              │
│   → Treating all silence the same way hurts performance.                     │
│   → Selective imputation helps for some findings, hurts for others.          │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   CHALLENGE 4: DUPLICATE REPORTS                                             │
│   ──────────────────────────────────────                                     │
│   131 studies share verbatim reports. One report is shared by 37 studies.    │
│                                                                              │
│   If you split these across train/validation:                                │
│   • The same target vector appears in both splits                            │
│   • Target leakage → overfit validation → poor leaderboard                   │
│                                                                              │
│   → You MUST group identical reports before CV splitting.                    │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   CHALLENGE 5: REPORTS EXIST ONLY IN TRAINING                                │
│   ────────────────────────────────────────────────                           │
│   At test time, there is no report.                                          │
│                                                                              │
│   • You cannot use text at inference                                         │
│   • Text is only useful for creating training targets                        │
│   • No multimodal fusion at inference                                        │
│                                                                              │
│   → The pipeline shape is forced:                                            │
│     reports → targets → train imaging model → discard text → predict         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Silent Bugs That Destroy Performance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PITFALLS (SILENT BUGS)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   BUG 1: SORTING SLICES BY FILENAME                                          │
│   ──────────────────────────────────────                                     │
│   DICOM filenames are SOP Instance UIDs — assigned to be UNIQUE, not ORDERED │
│   • Spearman ρ ≈ 0.01 between filename rank and physical position            │
│   • "2.5D" input becomes 3 unrelated slices, not 3 adjacent ones             │
│   • "Middle of stack" is a random subset                                     │
│                                                                              │
│   FIX: Sort by physical position: k = ImagePositionPatient · slice_normal    │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   BUG 2: NOT NORMALIZING LATERALITY                                          │
│   ────────────────────────────────────────                                   │
│   5 of 12 labels distinguish medial from lateral.                            │
│   If half your studies are left knees and you do nothing:                    │
│   • Those 5 labels are learned as noise                                      │
│   • Nothing crashes — you lose ~0.10 AUC without a symptom                   │
│                                                                              │
│   FIX: Derive laterality from geometry (image center x-coordinate)           │
│   • Coronal/Axial: flip horizontally for left knees                          │
│   • Sagittal: reverse slice order for left knees                             │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   BUG 3: RESIZING WITHOUT PHYSICAL CROP                                      │
│   ──────────────────────────────────────────────                             │
│   PixelSpacing varies 3.4× across the corpus.                                │
│   Fixed-pixel resize hands the encoder images at different physical scales.  │
│   • A 1-3mm meniscal tear may be below the sampling limit                    │
│   • It's GONE before the first convolution                                   │
│                                                                              │
│   FIX: Crop to 130mm, resize to 336px → 0.387 mm/pixel (consistent)          │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   BUG 4: HORIZONTAL FLIP AUGMENTATION                                        │
│   ────────────────────────────────────────────                               │
│   After normalizing laterality, random horizontal flip destroys exactly      │
│   the information that 5 labels depend on.                                   │
│                                                                              │
│   FIX: Never flip horizontally. Use small rotation, scale, shift instead.    │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│   BUG 5: STACKING SLICES AS CHANNELS                                         │
│   ───────────────────────────────────────────                                │
│   Stacking N slices as channels and averaging the first conv's weights       │
│   destroys the pretrained input interface.                                   │
│                                                                              │
│   FIX: Feed groups of 3 adjacent slices as 3-channel input through the       │
│   untouched backbone. Aggregate feature vectors afterwards.                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. The Solution Pipeline

### What the Top Teams Do

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STATE-OF-THE-ART PIPELINE (0.90+ AUC)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 1. LABEL EXTRACTION                                                 │   │
│   │ ─────────────────                                                 │   │
│   │ • LLM reads reports in 9 languages                                 │   │
│   │ • Two-layer: LLM picks from closed vocabulary → deterministic      │   │
│   │   mapping to probabilities                                         │   │
│   │ • Graded targets (not binary): trace/unqualified/marked → 0.3/0.7/1 │   │
│   │ • Confidence weights per (study, finding) pair                     │   │
│   │ • Selective imputation for high-uncertainty findings               │   │
│   │ • Gold studies weighted 5× higher                                  │   │
│   │                                                                     │   │
│   │ Result: ~0.88 macro AUC vs gold-58                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 2. IMAGE PREPROCESSING                                              │   │
│   │ ──────────────────────                                              │   │
│   │ • Physical slice ordering (NOT filename)                             │   │
│   │ • Laterality from geometry (image center x-coordinate)              │   │
│   │ • Physical crop 130mm → resize 336px                                │   │
│   │ • Per-series 1st-99th percentile intensity normalization            │   │
│   │ • 5 slots: plane × contrast (drop axial non-fluid, only 19% fill)   │   │
│   │ • Sample from central 60-80% of each stack                          │   │
│   │ • Cache decoded volumes as uint8 (memory efficient)                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 3. MODEL ARCHITECTURE                                               │   │
│   │ ─────────────────────                                               │   │
│   │ • DINOv2 ViT-S/14 or ViT-B/14 backbone (partially fine-tuned)       │   │
│   │ • 2.5D input: groups of 3 adjacent slices as 3 channels             │   │
│   │ • Per-diagnosis attention head over slot embeddings                 │   │
│   │ • Masked softmax renormalizes over present slots                    │   │
│   │ • 5-fold CV with report grouping                                    │   │
│   │ • Rank-mean ensemble across folds                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 4. TRAINING                                                         │   │
│   │ ─────────                                                           │   │
│   │ • BCE loss with confidence weights                                  │   │
│   │ • Gold studies weighted 5×                                          │   │
│   │ • Weight EMA for epoch selection                                    │   │
│   │ • No vertical flip augmentation                                     │   │
│   │ • Small rotation, scale, shift only                                 │   │
│   │ • Pairwise ranking term (optional)                                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 5. ENSEMBLING & FRONTIER                                            │   │
│   │ ─────────────────────────────                                       │   │
│   │ • Multi-resolution: 224px + 336px models                            │   │
│   │ • Multi-architecture: DINOv2 + EfficientNet-B3                      │   │
│   │ • RadImageNet pretrained weights (20% rank vote)                    │   │
│   │ • Specialist models for weak labels (Synovitis, MCL)                │   │
│   │ • Target-wise rank blending                                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   PUBLIC LEADERBOARD SCORES:                                                 │
│   ──────────────────────────                                                 │
│   0.776  Frozen DINOv2 + head (no fine-tuning)                               │
│   0.866  Fine-tuned DINOv2 @224px, 5-fold                                    │
│   0.883  Fine-tuned DINOv2 @336px, 5-fold                                    │
│   0.891  20-member ensemble, lexicon + LLM labels                            │
│   0.899  Specialist ensemble, LLM soft labels                                │
│   0.906  + RadImageNet rank blend (current frontier)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TL;DR                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   WHAT: Detect 12 knee abnormalities from MRI scans                          │
│                                                                              │
│   DATA: 4,407 training studies (58 labeled, 4,349 with reports only)         │
│         ~1,300 test studies (images only, no reports)                        │
│                                                                              │
│   CHALLENGE: Read reports to create labels, train imaging model, predict     │
│                                                                              │
│   METRIC: Macro-averaged AUC across 12 labels                                │
│                                                                              │
│   KEY INSIGHTS:                                                              │
│   • Only ranking matters (not calibration)                                   │
│   • Every label costs the same (rare labels need more attention)             │
│   • Reports exist only in training (no text at inference)                    │
│   • Label quality is the binding constraint                                  │
│   • Physical preprocessing is critical (slice order, laterality, scale)      │
│   • 2.5D DINOv2 is the dominant approach                                     │
│                                                                              │
│   FRONTIER: 0.906 AUC (LLM labels + DINOv2 + RadImageNet ensemble)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
