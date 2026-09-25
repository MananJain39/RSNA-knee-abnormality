# RSNA Knee Abnormality Detection — Complete Project Plan

> **From Classical Deep Learning to Quantum Machine Learning: A Comprehensive Guide**

---

##  Table of Contents

1. [Project Overview](#project-overview)
2. [Understanding the Problem](#understanding-the-problem)
3. [Prerequisites & Learning Path](#prerequisites--learning-path)
4. [Project Architecture](#project-architecture)
5. [Classical ML Pipeline (Phases 0-5)](#classical-ml-pipeline)
6. [Quantum ML Pipeline (Phases 6-10)](#quantum-ml-pipeline)
7. [Dependencies & Setup](#dependencies--setup)
8. [Timeline & Milestones](#timeline--milestones)

---

##  Project Overview

### Implementation Status & Validation Gate (updated 2026-08-27)

The roadmap below describes the intended project, but the repository is currently
at the **classical-pipeline integration** stage, not at a score-bearing baseline:

- **Implemented and sample-checked:** DICOM loading, physical ordering/laterality,
  preprocessing, five-slot volume caching, acquisition clustering, grouped split
  creation, disk-backed dataset/collate, DINOv2 attention model, trainer,
  inference, submission writing, and synthetic end-to-end tests.
- **Available real sample:** three fully preprocessed studies in
  `data/processed/volumes_sample3`, all with five populated slots and matching
  external soft labels. The sample split is two train / one validation study.
- **Not yet demonstrated:** a forward/backward pass through the real DINOv2
  checkpoint on the real MRI volumes, any meaningful held-out AUC, a full-data
  training run, cross-validation, specialists, ensemble, or any quantum phase.

Do not use the three-study sample to estimate AUC or select architectures. Its
purpose is to catch integration defects before expensive Kaggle runs. The required
gate before any new model work is:

1. Install the explicit model extra: `pip install -e '.[classical]'`.
2. Run `python scripts/probe_sample_architecture.py` to validate real volumes,
   sample-label coverage, 2.5D grouping, five-slot collation, masks, and tensor
   shapes without loading a model.
3. On GPU with DINOv2 weights available, run
   `python scripts/probe_sample_architecture.py --with-model --device cuda`.
   It must produce finite `3 x 12` logits/loss, a non-zero attention-head gradient,
   and a changed head parameter after one optimizer step.
4. Only then preprocess a larger, fixed, label-stratified dev set (suggested
   100-200 studies, including gold-labelled studies) and run a frozen-backbone
   micro-overfit test plus an untouched dev-split evaluation. Promote an
   architecture only if it passes both contract and dev-set gates.

This creates a feedback loop: **real DICOM -> cached slot tensors -> model
forward/backward -> fixed dev metric**, instead of adding components based solely
on synthetic tests or a three-study AUC.

### What Are We Building?

This project solves the **RSNA 2026 Knee Abnormality Detection** challenge using a two-stage approach:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PROJECT GOAL                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT:                    OUTPUT:                                              │
│   ──────                    ───────                                              │
│   Knee MRI scans      →     12 probabilities (one per abnormality)              │
│   (4,407 training)          Each probability ∈ [0, 1]                           │
│   (~1,300 test)             Indicates confidence abnormality is PRESENT         │
│                                                                                  │
│   THE CHALLENGE:                                                                 │
│   ────────────────                                                                 │
│   • Only 58 of 4,407 training studies have expert labels (1.3%)                 │
│   • You must extract labels from radiology reports (9 languages!)               │
│   • Test data has NO reports — images only                                       │
│                                                                                  │
│   OUR APPROACH:                                                                  │
│   ─────────────                                                                  │
│   Stage 1: Classical Deep Learning (DINOv2) → ~0.90 AUC                         │
│   Stage 2: Quantum Machine Learning → Explore quantum advantage                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Why Two Stages?

**Stage 1 (Classical)**: Build a state-of-the-art model using proven techniques
- Establishes a performance baseline
- Provides features for quantum models
- Competitive in the Kaggle competition

**Stage 2 (Quantum)**: Explore cutting-edge quantum machine learning
- Research contribution: Where does quantum help?
- Hybrid models: Classical features + quantum classification
- Future-proofing: Quantum hardware is improving rapidly

---

##  Understanding the Problem

### The 12 Knee Abnormalities

Before diving into code, understand what you're detecting:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         THE 12 KNEE ABNORMALITIES                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   CATEGORY 1: LIGAMENTS (2)                                                      │
│   ───────────────────────                                                        │
│   ┌─────────────────────────────────────┐  ┌─────────────────────────────────┐  │
│   │ 1. ACL (Anterior Cruciate Ligament) │  │ 2. MCL (Medial Collateral)      │  │
│   │    • High-grade tear                │  │    • High-grade acute tear      │  │
│   │    • Prevalence: 41%                │  │    • Prevalence: 16% (rare!)    │  │
│   │    • Best seen: Sagittal plane      │  │    • Best seen: Coronal plane   │  │
│   └─────────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                                  │
│   CATEGORY 2: MENISCI (2)                                                        │
│   ───────────────────────                                                        │
│   ┌─────────────────────────────────────┐  ┌─────────────────────────────────┐  │
│   │ 3. Medial Meniscus                  │  │ 4. Lateral Meniscus             │  │
│   │    • Signal reaching surface        │  │    • Signal reaching surface    │  │
│   │    • Prevalence: 45%                │  │    • Prevalence: 40%            │  │
│   │    • Best seen: Sagittal + Coronal  │  │    • Best seen: Sagittal        │  │
│   └─────────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                                  │
│   CATEGORY 3: OSTEOARTHRITIS (3)                                                 │
│   ──────────────────────────────                                                 │
│   ┌─────────────────────────────────────┐  ┌─────────────────────────────────┐  │
│   │ 5. Medial OA                        │  │ 6. Lateral OA                   │  │
│   │    • Cartilage loss >50%            │  │    • Cartilage loss >50%        │  │
│   │    • Prevalence: 26%                │  │    • Prevalence: 19%            │  │
│   │    • Best seen: Coronal             │  │    • Best seen: Coronal         │  │
│   └─────────────────────────────────────┘  └─────────────────────────────────┘  │
│   ┌─────────────────────────────────────┐                                        │
│   │ 7. PF OA (Patellofemoral)           │                                        │
│   │    • Cartilage loss >50%            │                                        │
│   │    • Prevalence: 36%                │                                        │
│   │    • Best seen: Axial + Sagittal    │                                        │
│   └─────────────────────────────────────┘                                        │
│                                                                                  │
│   CATEGORY 4: OTHER FINDINGS (5)                                                 │
│   ──────────────────────────────                                                 │
│   ┌─────────────────────────────────────┐  ┌─────────────────────────────────┐  │
│   │ 8. Effusion                         │  │ 9. Synovitis                    │  │
│   │    • Moderate/large fluid           │  │    • Synovial inflammation      │  │
│   │    • Prevalence: 60% (common!)      │  │    • Prevalence: 47%            │  │
│   │    • Best seen: Sagittal fluid      │  │    • Best seen: Sagittal fluid  │  │
│   └─────────────────────────────────────┘  └─────────────────────────────────┘  │
│   ┌─────────────────────────────────────┐  ┌─────────────────────────────────┐  │
│   │ 10. Baker's Cyst                    │  │ 11. Contusion                   │  │
│   │    • Popliteal cyst                 │  │    • Bone marrow edema          │  │
│   │    • Prevalence: 21%                │  │    • Prevalence: 33%            │  │
│   │    • Best seen: Sagittal fluid      │  │    • Best seen: All fluid       │  │
│   └─────────────────────────────────────┘  └─────────────────────────────────┘  │
│   ┌─────────────────────────────────────┐                                        │
│   │ 12. Fracture                        │                                        │
│   │    • Acute cortical break           │                                        │
│   │    • Prevalence: 31%                │                                        │
│   │    • Best seen: All planes          │                                        │
│   └─────────────────────────────────────┘                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Evaluation Metric: Macro-Averaged AUC

**What is AUC?**
- AUC (Area Under ROC Curve) measures how well your model ranks positive vs negative cases
- AUC = 0.5 → random guessing
- AUC = 1.0 → perfect ranking
- **Only rank order matters** — calibration doesn't help

**What is Macro-Averaging?**
```
Final Score = (1/12) × Σ AUC_i  (average across all 12 labels)
```

**Critical Implications:**
1. **Every label matters equally** — A bad label costs you 1/12 of your score
2. **Rare labels deserve MORE attention** — MCL (16% prevalence) is just as important as Effusion (60%)
3. **Prevalence drift is survivable** — AUC is invariant to positive rate

---

##  Prerequisites & Learning Path

### Before You Start: Essential Knowledge

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PREREQUISITE CHECKLIST                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   PYTHON FUNDAMENTALS                                                            │
│   ───────────────────                                                            │
│   □ Comfortable with Python 3.10+                                                │
│   □ Understand OOP (classes, inheritance)                                        │
│   □ Can use type hints                                                           │
│   □ Familiar with virtual environments                                           │
│                                                                                  │
│   MACHINE LEARNING BASICS                                                        │
│   ────────────────────────                                                       │
│   □ Understand supervised learning (classification)                              │
│   □ Know what loss functions are (BCE, CrossEntropy)                             │
│   □ Understand train/val/test splits                                             │
│   □ Familiar with cross-validation                                               │
│   □ Know what overfitting/underfitting means                                     │
│                                                                                  │
│   DEEP LEARNING FUNDAMENTALS                                                     │
│   ──────────────────────────                                                     │
│   □ Understand neural networks (layers, activations)                             │
│   □ Know what CNNs are (convolutions, pooling)                                   │
│   □ Understand backpropagation and gradient descent                              │
│   □ Familiar with PyTorch or TensorFlow                                          │
│   □ Know what transfer learning is                                               │
│                                                                                  │
│   MEDICAL IMAGING (Helpful but not required)                                     │
│   ──────────────────────────────────────────                                     │
│   □ Basic understanding of MRI (what it measures)                                │
│   □ Know what DICOM format is                                                    │
│   □ Understand anatomical planes (sagittal, coronal, axial)                      │
│                                                                                  │
│   QUANTUM COMPUTING (For Phases 6-10)                                            │
│   ──────────────────────────────────────────                                     │
│   □ Understand qubits vs classical bits                                          │
│   □ Know basic quantum gates (X, Y, Z, H, CNOT)                                  │
│   □ Understand superposition and entanglement                                    │
│   □ Familiar with quantum circuits                                               │
│   □ Know what measurement means in quantum context                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Learning Resources by Phase

#### For Classical ML (Phases 0-5)

**Must-Know Concepts:**

1. **Vision Transformers (ViT)**
   - Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
   - Key idea: Split image into patches, treat as sequence
   - Resource: https://viso.ai/deep-learning/vision-transformer-vit/

2. **DINOv2 (Self-Supervised Learning)**
   - Paper: "DINOv2: Learning Robust Visual Features" (Oquab et al., 2023)
   - Key idea: Train model to agree with itself across augmented views
   - Why it matters: Features work well with linear probing
   - Resource: https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/

3. **2.5D Medical Image Analysis**
   - Concept: Process 3D volumes as sequences of 2D slices
   - Why not 3D? Memory constraints, pretrained weights are 2D
   - Our approach: Groups of 3 adjacent slices as 3-channel input

4. **Transfer Learning**
   - Concept: Use pretrained model, adapt to new domain
   - Our approach: DINOv2 (trained on natural images) → knee MRI
   - Key technique: Partial unfreezing (last 6 blocks trainable)

**Recommended Courses:**
- Fast.ai Practical Deep Learning (free): https://course.fast.ai/
- Stanford CS231n (CNNs): https://cs231n.stanford.edu/
- PyTorch Tutorials: https://pytorch.org/tutorials/

#### For Quantum ML (Phases 6-10)

**Must-Know Concepts:**

1. **Quantum Computing Basics**
   - Qubits: Superposition, measurement, probability amplitudes
   - Gates: Single-qubit (X, Y, Z, H, RY) and multi-qubit (CNOT, CZ)
   - Circuits: Sequence of gates applied to qubits
   - Resource: https://qiskit.org/learn

2. **Quantum Machine Learning**
   - Variational Quantum Circuits: Parameterized circuits trained like neural nets
   - Quantum Kernels: Compute similarity using quantum circuits
   - Hybrid Models: Classical + quantum components
   - Paper: "Quantum Machine Learning" (Schuld & Killoran, 2022)

3. **Data Encoding Strategies**
   - Amplitude Encoding: Encode N features in log₂(N) qubits
   - Angle Encoding: Encode each feature as rotation angle
   - ZZFeatureMap: Entangling feature maps for kernel methods
   - Resource: https://qiskit.org/documentation/machine-learning/tutorials/01_qnn_classification.html

4. **Quantum Frameworks**
   - Qiskit (IBM): Most popular, good documentation
   - PennyLane (Xanadu): Differentiable quantum computing
   - We'll use both for comparison

**Recommended Courses:**
- Qiskit Textbook (free): https://github.com/Qiskit/textbook
- PennyLane Demos: https://pennylane.ai/qml/demos
- IBM Quantum Learning: https://learning.quantum.ibm.com/

---

##  Project Architecture

### Directory Structure

```
rsna-knee/
│
├── PLAN.md                          # This file (project roadmap)
├── RESEARCH_SUMMARY.md              # Competition research & community findings
├── DATA_EXPLAINED.md                # Detailed data documentation
├── pyproject.toml                   # Python dependencies
├── Makefile                         # Build & run commands
│
├── configs/                         # Configuration files
│   ├── default.yaml                 # Default settings
│   ├── classical/                   # Classical model configs
│   │   ├── dinov2_baseline.yaml     # Phase 2 config
│   │   ├── dinov2_finetuned.yaml    # Phase 3 config
│   │   └── ensemble.yaml            # Phase 4 config
│   └── quantum/                     # Quantum model configs
│       ├── vqc_config.yaml          # Phase 7 config
│       ├── quantum_kernel.yaml      # Phase 8 config
│       └── hybrid_config.yaml       # Phase 9 config
│
├── src/                             # Source code
│   ├── __init__.py
│   ├── config.py                    # Configuration loader
│   │
│   ├── data/                        # Data handling (Phase 0-1)
│   │   ├── __init__.py
│   │   ├── loader.py                # Load studies & series
│   │   └── schema.py                # Data schemas & validation
│   │
│   ├── labels/                      # Label extraction (Phase 0)
│   │   ├── __init__.py
│   │   ├── extractor.py             # Base extractor interface
│   │   ├── regex_extractor.py       # Multilingual regex patterns
│   │   └── llm_extractor.py         # LLM-based extraction (GPT-4o)
│   │
│   ├── imaging/                     # Image processing (Phase 1)
│   │   ├── __init__.py
│   │   ├── dicom_loader.py          # DICOM reading & decoding
│   │   ├── preprocessing.py         # Crop, resize, normalize
│   │   ├── geometry.py              # Physical position calculations
│   │   └── slot_builder.py          # Organize series into slots
│   │
│   ├── classical/                   # Classical ML models (Phase 2-5)
│   │   ├── __init__.py
│   │   ├── dinov2_backbone.py       # DINOv2 wrapper
│   │   ├── attention_head.py        # Per-diagnosis attention
│   │   ├── dataset.py               # PyTorch Dataset class
│   │   ├── trainer.py               # Training loop
│   │   ├── ensemble.py              # Model ensembling
│   │   └── inference.py             # Prediction pipeline
│   │
│   ├── quantum/                     # Quantum ML models (Phase 6-10)
│   │   ├── __init__.py
│   │   │
│   │   ├── encoding/                # Quantum data encoding (Phase 6)
│   │   │   ├── __init__.py
│   │   │   ├── amplitude_encoding.py
│   │   │   ├── angle_encoding.py
│   │   │   └── feature_map.py
│   │   │
│   │   ├── circuits/                # Quantum circuits (Phase 7-8)
│   │   │   ├── __init__.py
│   │   │   ├── variational.py       # VQC circuits
│   │   │   ├── ansatz.py            # Parameterized layers
│   │   │   └── feature_maps.py      # ZZFeatureMap, etc.
│   │   │
│   │   ├── models/                  # Quantum models (Phase 7-9)
│   │   │   ├── __init__.py
│   │   │   ├── vqc_classifier.py    # Variational Quantum Classifier
│   │   │   ├── quantum_kernel.py    # Quantum kernel methods
│   │   │   ├── quantum_svm.py       # Quantum SVM
│   │   │   └── hybrid_model.py      # Classical-quantum hybrid
│   │   │
│   │   ├── training/                # Quantum training (Phase 7-9)
│   │   │   ├── __init__.py
│   │   │   ├── optimizer.py         # Quantum optimizers (SPSA, Adam)
│   │   │   ├── loss.py              # Quantum loss functions
│   │   │   └── trainer.py           # Quantum training loop
│   │   │
│   │   └── utils/                   # Quantum utilities
│   │       ├── __init__.py
│   │       ├── statevector.py       # State manipulation
│   │       └── metrics.py           # Quantum-specific metrics
│   │
│   ├── experiments/                 # Experiment tracking
│   │   ├── __init__.py
│   │   ├── tracker.py               # MLflow/W&B logging
│   │   ├── classical_runner.py      # Run classical experiments
│   │   └── quantum_runner.py        # Run quantum experiments
│   │
│   └── validation/                  # Validation & metrics
│       ├── __init__.py
│       ├── metrics.py               # AUC, accuracy, etc.
│       ├── gold.py                  # Gold-58 validation
│       └── audit.py                 # Sanity checks
│
├── scripts/                         # CLI scripts
│   ├── extract_labels.py            # Phase 0: Extract labels
│   ├── validate_labels.py           # Phase 0: Validate labels
│   ├── build_slots.py               # Phase 1: Build slots
│   ├── train_classical.py           # Phase 2-4: Train models
│   ├── train_quantum.py             # Phase 7-9: Train quantum
│   ├── inference.py                 # Generate submissions
│   └── benchmark_quantum.py         # Phase 10: Benchmark
│
├── notebooks/                       # Jupyter notebooks
│   ├── 00_eda.ipynb                 # Exploratory data analysis
│   ├── 01_label_extraction.ipynb    # Phase 0 walkthrough
│   ├── 02_dicom_preprocessing.ipynb # Phase 1 walkthrough
│   ├── 03_classical_training.ipynb  # Phase 2-4 walkthrough
│   ├── 04_quantum_fundamentals.ipynb # Phase 6 walkthrough
│   ├── 05_vqc_training.ipynb        # Phase 7 walkthrough
│   ├── 06_quantum_kernels.ipynb     # Phase 8 walkthrough
│   ├── 07_hybrid_models.ipynb       # Phase 9 walkthrough
│   └── 08_benchmarking.ipynb        # Phase 10 walkthrough
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_labels.py
│   ├── test_metrics.py
│   ├── test_quantum_encoding.py
│   └── test_quantum_circuits.py
│
├── experiments/                     # Experiment logs (created during training)
│   ├── classical/
│   └── quantum/
│
└── models/                          # Saved models (created during training)
    ├── classical/
    └── quantum/
```

---

##  Classical ML Pipeline

### Phase 0: Infrastructure & Label Extraction

**Duration**: Day 1  
**GPU Required**: No  
**Difficulty**:  Beginner  
**Prerequisites**: Python, basic NLP

#### What You'll Learn

- How to extract structured information from unstructured text
- Multilingual text processing (9 languages!)
- Working with LLMs (GPT-4o) for information extraction
- Handling uncertainty and "not addressed" cases

#### The Problem

You have 4,407 knee MRI studies but only **58 have expert labels** (1.3%). The other 4,349 have radiology reports in 9 different languages. You need to extract labels from these reports.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LABEL EXTRACTION PROBLEM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT:                                                                         │
│   ──────                                                                         │
│   Report (Spanish, 322 chars):                                                   │
│   "Técnica: RMN de la rodilla.                                                   │
│    Resultados: Rotura de menisco interno.                                        │
│    Artrosis femorotibial medial. Derrame."                                       │
│                                                                                  │
│   OUTPUT:                                                                        │
│   ───────                                                                        │
│   MedialMeniscus = 1 (rotura de menisco interno = medial meniscus tear)         │
│   MedialOA = 1 (artrosis femorotibial medial = medial osteoarthritis)           │
│   Effusion = 1 (derrame = effusion)                                             │
│   ACL = 0 (not mentioned → negative)                                            │
│   MCL = 0 (not mentioned → negative)                                            │
│   ... (other labels = 0)                                                         │
│                                                                                  │
│   THE CHALLENGE:                                                                 │
│   ────────────────                                                               │
│   • 9 languages: English, Spanish, French, German, Dutch, Bulgarian,            │
│                  Turkish, Greek, and others                                       │
│   • "Not addressed" ≠ negative (sometimes radiologist didn't look!)             │
│   • Graded findings: "small effusion" vs "large effusion"                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Two Approaches

**Approach 1: Regex/Lexicon (Fast, Deterministic)**

```python
# Example pattern for English
patterns = {
    'ACL': [
        r'ACL\s+tear',
        r'anterior\s+cruciate\s+ligament\s+tear',
        r'(?:complete|high[- ]grade)\s+ACL\s+(?:tear|rupture)',
    ],
    'Effusion': [
        r'(?:moderate|large)\s+(?:joint\s+)?effusion',
        r'(?:moderate|large)\s+(?:joint\s+)?fluid',
    ],
}

def extract_labels_regex(report: str) -> dict:
    labels = {}
    for finding, patterns in patterns.items():
        found = any(re.search(p, report, re.IGNORECASE) for p in patterns)
        labels[finding] = 1.0 if found else 0.0
    return labels
```

**Pros**: Fast, deterministic, no API costs  
**Cons**: Misses nuanced language, requires manual pattern creation for 9 languages  
**Expected AUC**: ~0.81 vs gold labels

**Approach 2: LLM-Based (Accurate, Expensive)**

```python
from openai import OpenAI

client = OpenAI()

def extract_labels_llm(report: str) -> dict:
    prompt = f"""
    Extract the presence of 12 knee abnormalities from this radiology report.
    For each finding, return:
    - 1.0 if definitely present
    - 0.0 if definitely absent
    - 0.5 if not addressed / uncertain
    
    Report: {report}
    
    Return JSON with keys: ACL, MCL, MedialMeniscus, LateralMeniscus,
    MedialOA, LateralOA, PFOA, Effusion, Synovitis, Bakers, Contusion, Fracture
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

**Pros**: Handles all languages, nuanced understanding, graded outputs  
**Cons**: API costs (~$50-100 for full dataset), slower  
**Expected AUC**: ~0.88 vs gold labels

#### Key Concepts

**1. Graded Targets (Not Binary)**

A report saying "small joint effusion" might be labeled negative by an expert who only marks significant effusions. Solution: emit probabilities, not binary labels.

```
Trace effusion     → 0.3
Small effusion     → 0.6
Moderate effusion  → 0.9
Large effusion     → 1.0
```

**2. "Not Addressed" is Not One Thing**

Critical insight from the community:

| Finding | When report is SILENT | When report SPEAKS | Interpretation |
|---------|----------------------|-------------------|----------------|
| Baker's | 3% positive | 44% positive | Silence = absent |
| Synovitis | 34% positive | 76% positive | Silence = uninformative |

**Rule**: Only impute "not addressed" for findings where silence is genuinely uninformative (high silence ratio).

**3. Confidence Weighting**

Reports that explicitly address a finding should have more influence than silent reports.

```python
# Loss function with confidence weights
loss = -confidence * (y * log(p) + (1 - y) * log(1 - p))
```

#### Implementation Steps

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 0 IMPLEMENTATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Step 1: Validate Data Schema                                                   │
│   ─────────────────────────────                                                  │
│   • Run scripts/validate_labels.py                                               │
│   • Check train.csv structure                                                    │
│   • Verify 58 gold studies have all 12 labels                                    │
│                                                                                  │
│   Step 2: Implement Multilingual Regex Extractor                                 │
│   ──────────────────────────────────────────────                                 │
│   • Create patterns for all 9 languages                                          │
│   • Handle Unicode edge cases:                                                   │
│     - Turkish: dotted/dotless i (İ/i, I/ı)                                       │
│     - Greek: micro sign (U+00B5) → real mu (U+03BC)                              │
│     - NFKD normalization for accents                                             │
│   • Test on 58 gold studies → target AUC ≥ 0.81                                  │
│                                                                                  │
│   Step 3: Implement LLM Extractor                                                │
│   ────────────────────────────                                                   │
│   • Use GPT-4o or local Qwen3-14B                                                │
│   • Prompt engineering: closed vocabulary → deterministic mapping                │
│   • Emit graded targets (probabilities, not binary)                              │
│   • Test on 58 gold studies → target AUC ≥ 0.87                                  │
│                                                                                  │
│   Step 4: Compute "Not Addressed" Rates                                          │
│   ────────────────────────────────────────                                       │
│   • For each finding, compute: P(positive | report silent)                       │
│   • Identify findings where silence is uninformative                             │
│                                                                                  │
│   Step 5: Selective Imputation                                                   │
│   ────────────────────────────                                                   │
│   • For Synovitis: fill from Effusion (P(syn|eff) = 0.63 vs P(syn|no eff) = 0.22)│
│   • For Baker's: DO NOT impute (silence = absent)                                │
│                                                                                  │
│   Step 6: Compute Confidence Weights                                             │
│   ──────────────────────────────────                                             │
│   • Per (study, finding) pair                                                    │
│   • Higher weight for explicit mentions                                          │
│   • Lower weight for "not addressed"                                             │
│                                                                                  │
│   Step 7: Publish Labels Dataset                                                 │
│   ────────────────────────────────                                               │
│   • Save as CSV: StudyInstanceUID, ACL, MCL, ..., confidence weights             │
│   • Upload to Kaggle as dataset                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Expected Output

```python
# labels.csv
StudyInstanceUID,ACL,MCL,MedialMeniscus,...,Fracture,ACL_conf,MCL_conf,...
1.2.826.0.1.368...,0.9,0.1,0.8,...,0.2,0.95,0.85,0.92,...
1.2.826.0.1.368...,0.1,0.0,0.0,...,0.1,0.90,0.80,0.88,...
```

#### Success Criteria

-  Macro AUC ≥ 0.87 vs 58 gold studies
-  All 9 languages handled correctly
-  Unicode edge cases resolved
-  Graded targets (not binary)
-  Confidence weights computed

#### Common Pitfalls

1. **Treating "not addressed" as negative** → Wrong! Sometimes radiologist didn't look
2. **Binary labels only** → Lose information from graded findings
3. **Ignoring language-specific patterns** → Turkish i, Greek mu will break your regex
4. **Not computing confidence weights** → Silent reports will dominate training

---

### Phase 1: DICOM Pipeline

**Duration**: Days 2-3  
**GPU Required**: No  
**Difficulty**:  Intermediate  
**Prerequisites**: Medical imaging basics, NumPy

#### What You'll Learn

- DICOM format (medical image standard)
- Physical coordinate systems in medical imaging
- Image preprocessing for deep learning
- Memory-efficient data loading

#### The Problem

You have ~570 GB of DICOM images across 24,371 series. You need to:
1. Load them correctly (handle different compression formats)
2. Sort slices by physical position (NOT filename!)
3. Normalize left/right knees
4. Crop to consistent physical scale
5. Organize into "slots" (plane × contrast)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DICOM PIPELINE CHALLENGES                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   CHALLENGE 1: Slice Ordering                                                    │
│   ───────────────────────────                                                    │
│    WRONG: Sort by filename (SOP Instance UID = random)                         │
│    RIGHT: Sort by physical position                                            │
│                                                                                  │
│   Physical position = ImagePositionPatient · slice_normal                        │
│   where slice_normal = cross(row_cosine, col_cosine)                             │
│                                                                                  │
│   CHALLENGE 2: Laterality (Left vs Right)                                        │
│   ────────────────────────────────────────────                                   │
│   5 of 12 labels distinguish medial from lateral                                 │
│   If you don't normalize L/R, those 5 labels become noise                        │
│                                                                                  │
│   ✅ Derive from geometry: sign of image center's x-coordinate                   │
│   ❌ NOT from corner (unreliable near origin)                                    │
│                                                                                  │
│   CHALLENGE 3: Physical Scale                                                    │
│   ───────────────────────────                                                    │
│   PixelSpacing varies 3.4× across corpus                                         │
│   A 1-3 mm meniscal tear is GONE if you resize without physical crop             │
│                                                                                  │
│   ✅ Crop to 130 mm, resize to 336 px                                            │
│   ❌ Resize without crop (loses small pathology)                                 │
│                                                                                  │
│   CHALLENGE 4: Intensity Normalization                                           │
│   ────────────────────────────────────                                           │
│   MRI has no absolute scale (same tissue, different number on different day)     │
│                                                                                  │
│   ✅ Per-series 1st-99th percentile normalization                                │
│   ❌ Per-slice min/max (one bright vessel compresses everything)                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Key Concepts

**1. DICOM Format**

DICOM (Digital Imaging and Communications in Medicine) is the standard for medical images.

```python
import pydicom

# Read DICOM file
ds = pydicom.dcmread("slice001.dcm")

# Important attributes:
ds.PixelData              # Raw pixel data (compressed or uncompressed)
ds.Rows, ds.Columns       # Image dimensions (e.g., 256×256)
ds.PixelSpacing           # Physical pixel size (e.g., [0.5, 0.5] mm)
ds.ImagePositionPatient   # 3D position in patient space [x, y, z]
ds.ImageOrientationPatient # Orientation [row_cosine(3), col_cosine(3)]
ds.TransferSyntaxUID      # Compression format (JPEG, JPEG2000, etc.)
```

**2. Physical Position Sorting**

```python
import numpy as np

def sort_by_physical_position(dicom_files):
    """Sort DICOM files by physical position along slice direction."""
    
    def get_physical_position(ds):
        # Slice normal = cross product of row and column direction cosines
        row_cosine = np.array(ds.ImageOrientationPatient[:3])
        col_cosine = np.array(ds.ImageOrientationPatient[3:])
        slice_normal = np.cross(row_cosine, col_cosine)
        
        # Project position onto slice normal
        position = np.array(ds.ImagePositionPatient)
        return np.dot(position, slice_normal)
    
    return sorted(dicom_files, key=lambda ds: get_physical_position(ds))
```

**3. Laterality Normalization**

```python
def normalize_laterality(ds, pixel_array):
    """Flip image if left knee to make all images right-knee orientation."""
    
    # Get image center in patient coordinates
    position = np.array(ds.ImagePositionPatient)
    row_cosine = np.array(ds.ImageOrientationPatient[:3])
    col_cosine = np.array(ds.ImageOrientationPatient[3:])
    
    row_spacing = ds.PixelSpacing[0]
    col_spacing = ds.PixelSpacing[1]
    
    center = (
        position
        + row_cosine * row_spacing * (ds.Rows / 2)
        + col_cosine * col_spacing * (ds.Columns / 2)
    )
    
    # Determine laterality from x-coordinate sign
    # (Dead zone: within 25mm of origin, use tag instead)
    if abs(center[0]) > 25:  # mm
        is_left = center[0] < 0
    else:
        is_left = ds.get("Laterality", "R") == "L"
    
    # Flip if left knee
    if is_left:
        pixel_array = np.fliplr(pixel_array)  # For coronal/axial
    
    return pixel_array, is_left
```

**4. Physical Crop & Resize**

```python
def physical_crop_and_resize(pixel_array, ds, target_size=336, physical_extent=130):
    """Crop to consistent physical size, then resize."""
    
    pixel_spacing = ds.PixelSpacing[0]  # mm per pixel
    current_extent = pixel_array.shape[0] * pixel_spacing  # mm
    
    # Crop to target physical extent
    target_pixels = int(physical_extent / pixel_spacing)
    
    if target_pixels < pixel_array.shape[0]:
        # Center crop
        start = (pixel_array.shape[0] - target_pixels) // 2
        end = start + target_pixels
        cropped = pixel_array[start:end, start:end]
    else:
        # Pad if image is smaller than target
        pad_size = (target_pixels - pixel_array.shape[0]) // 2
        cropped = np.pad(pixel_array, ((pad_size, pad_size), (pad_size, pad_size)))
    
    # Resize to target size
    from skimage.transform import resize
    resized = resize(cropped, (target_size, target_size), preserve_range=True)
    
    return resized.astype(np.uint8)
```

**5. Slot Organization**

Organize series into 5 slots based on plane × contrast:

```python
SLOT_DEFINITIONS = {
    'Sagittal_Fluid': {'plane': 'Sagittal', 'fluid': 1},
    'Sagittal_Structural': {'plane': 'Sagittal', 'fluid': 0},
    'Coronal_Fluid': {'plane': 'Coronal', 'fluid': 1},
    'Coronal_Structural': {'plane': 'Coronal', 'fluid': 0},
    'Axial_Fluid': {'plane': 'Axial', 'fluid': 1},
}

# Note: Fluid_Sensitive == Fat_Suppression (identical on all 24,371 rows)
# Axial_Structural exists for only 19% of studies → drop it
```

#### Implementation Steps

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1 IMPLEMENTATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Step 1: DICOM Loader                                                           │
│   ────────────────────                                                           │
│   • Handle all transfer syntaxes (Explicit VR, JPEG Lossless, JPEG 2000)         │
│   • Bundle pylibjpeg, python-gdcm for decoding                                   │
│   • Read with stop_before_pixels=True for headers                                │
│                                                                                  │
│   Step 2: Physical Position Sorting                                              │
│   ────────────────────────────────                                               │
│   • Compute slice normal from ImageOrientationPatient                            │
│   • Project ImagePositionPatient onto slice normal                               │
│   • Sort by projected position                                                   │
│   • Verify: Spearman ρ should be ~1.0 (not ~0.01 from filename sorting)          │
│                                                                                  │
│   Step 3: Laterality Normalization                                               │
│   ────────────────────────────                                                   │
│   • Compute image center in patient coordinates                                  │
│   • Determine laterality from x-coordinate sign                                  │
│   • Flip left knees to right orientation                                         │
│   • Per-plane: coronal/axial flip horizontally, sagittal reverse slice order     │
│                                                                                  │
│   Step 4: Physical Crop & Resize                                                 │
│   ────────────────────────────                                                   │
│   • Crop to 130 mm (smaller than smallest FOV)                                   │
│   • Resize to 336 px → 0.387 mm/pixel                                            │
│   • Verify crop actually fits (if L/s > image width, crop is no-op)              │
│                                                                                  │
│   Step 5: Intensity Normalization                                                │
│   ────────────────────────────                                                   │
│   • Per-series (not per-slice) 1st-99th percentile                               │
│   • Over whole sampled stack (preserves inter-slice contrast)                    │
│                                                                                  │
│   Step 6: Slot Assignment                                                        │
│   ────────────────────────                                                       │
│   • Assign each series to one of 5 slots                                         │
│   • Measure fill rates (should be 77-100%)                                       │
│                                                                                  │
│   Step 7: Slice Sampling                                                         │
│   ──────────────────────                                                         │
│   • Sample from central band (12-88% for sagittal, 20-80% for coronal/axial)     │
│   • Use np.linspace for even spacing                                             │
│                                                                                  │
│   Step 8: Volume Caching                                                         │
│   ────────────────────                                                           │
│   • Cache as uint8 (not float32) → 4× smaller                                    │
│   • Target: ~10 GB for full dataset                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Expected Output

```python
# Cached volume structure
cache = {
    'study_uid': {
        'Sagittal_Fluid': {
            'pixels': np.ndarray,  # shape: (12, 336, 336), dtype: uint8
            'series_uid': str,
            'num_slices': 12,
        },
        'Sagittal_Structural': {...},
        'Coronal_Fluid': {...},
        'Coronal_Structural': {...},
        'Axial_Fluid': {...},
    },
    ...
}
```

#### Success Criteria

- ✅ All 24,371 series processable
- ✅ No silent preprocessing bugs (verify on sample studies)
- ✅ Slice order matches physical position (ρ ≈ 1.0)
- ✅ Laterality normalization works (96.9% agreement with tag)
- ✅ Cache fits in ~10 GB RAM

#### Common Pitfalls

1. **Sorting by filename** → Spearman ρ ≈ 0.01 (random)
2. **Not normalizing laterality** → 5 labels become noise
3. **Resizing without physical crop** → Small pathology disappears
4. **Per-slice normalization** → Inter-slice contrast lost
5. **Horizontal flip augmentation** → Destroys medial/lateral information

---

### Phase 2: Baseline Model

**Duration**: Days 4-5  
**GPU Required**: Yes (1 GPU, ~4 hours)  
**Difficulty**: 🟡 Intermediate  
**Prerequisites**: PyTorch, transfer learning, CNNs/ViTs

#### What You'll Learn

- Vision Transformers (ViT) and DINOv2
- 2.5D medical image analysis
- Multi-label classification
- Cross-validation strategies

#### The Problem

Now that you have preprocessed images and extracted labels, you need to build a model that:
1. Takes MRI slices as input
2. Outputs 12 probabilities (one per abnormality)
3. Generalizes to unseen studies

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BASELINE MODEL ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT:                                                                         │
│   ──────                                                                         │
│   • 5 slots (Sagittal_Fluid, Sagittal_Structural, Coronal_Fluid, etc.)          │
│   • Each slot: 12 slices (or 8 for coronal/axial)                               │
│   • 2.5D input: groups of 3 adjacent slices as 3-channel RGB-like input         │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         DINOv2 BACKBONE (Frozen)                         │   │
│   │                                                                          │   │
│   │   Slice Group (3 slices) ──[DINOv2 ViT-S/14]──[CLS token + patches]     │   │
│   │          (3×336×336)              (frozen)          (768-dim)             │   │
│   │                                                                          │   │
│   │   Repeat for each slice group (12 groups per slot)                       │   │
│   │   Aggregate: mean pooling across groups → slot embedding (768-dim)       │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      PER-DIAGNOSIS ATTENTION HEAD                        │   │
│   │                                                                          │   │
│   │   For each slot s:                                                       │   │
│   │     h_s = φ(embedding_s) + e_s    # project + learned slot identity      │   │
│   │                                                                          │   │
│   │   For each diagnosis o (12 total):                                       │   │
│   │     α_{o,s} = softmax(<h_s, q_o> / √H, masked by presence)              │   │
│   │     c_o = Σ_s α_{o,s} · h_s         # attention-weighted context         │   │
│   │     logit_o = <c_o, w_o> + b_o      # final logit                        │   │
│   │                                                                          │   │
│   │   The masked softmax renormalizes over present slots                     │   │
│   │   (missing axial series → attention shifts to other sequences)           │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│                              12 Logits (one per label)                           │
│                              Apply sigmoid → 12 probabilities                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Key Concepts

**1. DINOv2 (Self-Supervised Vision Transformer)**

DINOv2 is a Vision Transformer trained to agree with itself across augmented views. Key properties:
- **Self-supervised**: No labels needed for pretraining
- **Patch-based**: Splits image into 14×14 patches, treats as sequence
- **CLS token**: Special token that aggregates global information
- **Works well with linear probing**: Features are high-quality

```python
import torch
from transformers import ViTModel

# Load DINOv2 (ViT-S/14 variant)
model = ViTModel.from_pretrained("facebook/dinov2-small")
model.eval()  # Freeze for baseline

# Forward pass
with torch.no_grad():
    outputs = model(pixel_values)  # shape: (batch, 1 + num_patches, 768)
    cls_token = outputs.last_hidden_state[:, 0, :]  # shape: (batch, 768)
    patch_tokens = outputs.last_hidden_state[:, 1:, :]  # shape: (batch, 576, 768)
```

**2. 2.5D Input**

Why not 3D?
- DINOv2 expects 3-channel input (RGB-like)
- Stacking N slices and averaging first conv weights destroys pretrained interface
- Memory: 3D volume at 336³ would be enormous

Solution: Process groups of 3 adjacent slices as 3-channel input

```python
def create_25d_input(volume, num_groups=12):
    """Convert 3D volume to 2.5D input (groups of 3 adjacent slices)."""
    # volume shape: (num_slices, height, width)
    
    # Sample slice indices evenly
    indices = np.linspace(0, volume.shape[0] - 3, num_groups, dtype=int)
    
    groups = []
    for idx in indices:
        # Group of 3 adjacent slices as 3 channels
        group = volume[idx:idx+3, :, :]  # shape: (3, H, W)
        groups.append(group)
    
    return np.stack(groups, axis=0)  # shape: (num_groups, 3, H, W)
```

**3. Per-Diagnosis Attention**

Different findings are visible on different sequences. Attention learns which slots matter for each diagnosis.

```python
class PerDiagnosisAttention(nn.Module):
    def __init__(self, num_slots=5, num_labels=12, hidden_dim=768):
        super().__init__()
        self.num_labels = num_labels
        
        # Slot projection
        self.slot_proj = nn.Linear(hidden_dim, hidden_dim)
        self.slot_embeddings = nn.Embedding(num_slots, hidden_dim)
        
        # Per-label query vectors
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_dim))
        
        # Per-label classification heads
        self.classifiers = nn.Linear(hidden_dim, 1)
    
    def forward(self, slot_embeddings, slot_mask):
        """
        slot_embeddings: (batch, num_slots, hidden_dim)
        slot_mask: (batch, num_slots) - 1 if slot present, 0 if missing
        """
        batch_size = slot_embeddings.shape[0]
        
        # Project slots + add learned slot identity
        h = self.slot_proj(slot_embeddings)  # (batch, num_slots, hidden)
        slot_ids = torch.arange(h.shape[1], device=h.device)
        h = h + self.slot_embeddings(slot_ids)  # (batch, num_slots, hidden)
        
        # Compute attention for each label
        logits = []
        for label_idx in range(self.num_labels):
            query = self.label_queries[label_idx]  # (hidden,)
            
            # Attention scores
            scores = torch.matmul(h, query) / np.sqrt(h.shape[-1])  # (batch, num_slots)
            
            # Mask missing slots and renormalize
            scores = scores.masked_fill(slot_mask == 0, float('-inf'))
            attention = F.softmax(scores, dim=1)  # (batch, num_slots)
            
            # Weighted sum
            context = torch.matmul(attention.unsqueeze(1), h).squeeze(1)  # (batch, hidden)
            
            # Classification
            logit = self.classifiers(context)  # (batch, 1)
            logits.append(logit)
        
        return torch.cat(logits, dim=1)  # (batch, num_labels)
```

**4. Cross-Validation with Report Grouping**

131 studies share identical reports. If you split them across folds, you get target leakage.

```python
from sklearn.model_selection import GroupKFold

# Group by identical report text
gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups=report_text):
    train_data = dataset[train_idx]
    val_data = dataset[val_idx]
    # Train and validate...
```

#### Implementation Steps

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2 IMPLEMENTATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Step 1: Load DINOv2 ViT-S/14                                                   │
│   ────────────────────────────                                                   │
│   • Download pretrained weights (Apache 2.0 license)                             │
│   • Freeze all parameters                                                        │
│   • Bundle as Kaggle dataset (no internet during submission)                     │
│                                                                                  │
│   Step 2: Create PyTorch Dataset                                                 │
│   ────────────────────────────                                                   │
│   • Load cached volumes from Phase 1                                             │
│   • Create 2.5D input (groups of 3 slices)                                       │
│   • Return: {slot_pixels, slot_mask, labels, confidence_weights}                 │
│                                                                                  │
│   Step 3: Implement Per-Diagnosis Attention Head                                 │
│   ────────────────────────────────────────────                                   │
│   • Slot projection + learned slot embeddings                                    │
│   • Per-label query vectors                                                      │
│   • Masked softmax attention                                                     │
│                                                                                  │
│   Step 4: Training Loop                                                          │
│   ──────────────────                                                             │
│   • Binary cross-entropy loss with confidence weights                            │
│   • Gold study weighting (5× for 58 expert-labeled studies)                      │
│   • Adam optimizer, lr=1e-3                                                      │
│   • 10 epochs per fold                                                           │
│                                                                                  │
│   Step 5: 5-Fold Cross-Validation                                                │
│   ────────────────────────────────                                               │
│   • Group by identical reports before splitting                                  │
│   • Train 5 models (one per fold)                                                │
│   • Collect out-of-fold predictions                                              │
│                                                                                  │
│   Step 6: Rank-Mean Ensemble                                                     │
│   ──────────────────────────                                                     │
│   • For each label, rank predictions from 5 folds                                │
│   • Average ranks → final prediction                                             │
│   • (Rank averaging, NOT probability averaging)                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Training Code Skeleton

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Model
backbone = load_dinov2()  # frozen
attention_head = PerDiagnosisAttention(num_slots=5, num_labels=12)

# Optimizer
optimizer = torch.optim.Adam(attention_head.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        slot_pixels = batch['slot_pixels']  # (batch, 5, 12, 3, 336, 336)
        slot_mask = batch['slot_mask']      # (batch, 5)
        labels = batch['labels']            # (batch, 12)
        weights = batch['weights']          # (batch, 12)
        
        # Extract features with frozen backbone
        with torch.no_grad():
            slot_features = []
            for slot_idx in range(5):
                # Process each slice group
                group_features = []
                for group_idx in range(12):
                    pixels = slot_pixels[:, slot_idx, group_idx]  # (batch, 3, 336, 336)
                    features = backbone(pixels)  # (batch, 768)
                    group_features.append(features)
                
                # Mean pooling across groups
                slot_feature = torch.stack(group_features, dim=1).mean(dim=1)
                slot_features.append(slot_feature)
            
            slot_features = torch.stack(slot_features, dim=1)  # (batch, 5, 768)
        
        # Forward pass through attention head
        logits = attention_head(slot_features, slot_mask)  # (batch, 12)
        
        # Loss with confidence weights
        probs = torch.sigmoid(logits)
        loss = -weights * (labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
        loss = loss.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| Gold-58 Macro AUC | 0.77-0.80 | vs 58 expert-labeled studies |
| Public LB | 0.77-0.80 | Kaggle leaderboard |
| Training time | ~4 hours | 5 folds × 10 epochs |

#### Success Criteria

- ✅ Training loop runs without errors
- ✅ Loss decreases over epochs
- ✅ Gold-58 AUC ≥ 0.77
- ✅ Public LB ≥ 0.77
- ✅ Out-of-fold predictions saved

#### Common Pitfalls

1. **Not freezing backbone properly** → Check `requires_grad=False` for all backbone params
2. **Stacking slices as channels incorrectly** → Use groups of 3, not all 12
3. **Probability averaging instead of rank averaging** → AUC cares about rank order
4. **Not grouping by report in CV** → Target leakage from duplicate reports

---

### Phase 3: Fine-tuning

**Duration**: Days 6-8  
**GPU Required**: Yes (1 GPU, ~6 hours)  
**Difficulty**: 🟡 Intermediate  
**Prerequisites**: Phase 2 complete, understanding of transfer learning

#### What You'll Learn

- Domain adaptation (natural images → medical images)
- Partial unfreezing strategies
- Weight exponential moving average (EMA)
- Data augmentation for medical images

#### The Problem

Your frozen DINOv2 baseline works, but it was trained on natural images, not knee MRIs. Fine-tuning adapts the backbone to your specific domain.

**Key insight from the community**: Domain adaptation does most of the work, not resolution.

| System | Gold-58 AUC | Public LB |
|--------|-------------|-----------|
| Frozen DINOv2 | 0.771 | 0.776 |
| Fine-tuned @224px | 0.824 | 0.866 |
| Fine-tuned @336px | — | 0.883 |

Fine-tuning buys you +0.09 AUC!

#### Key Concepts

**1. Partial Unfreezing**

Don't fine-tune the entire backbone (overfits quickly). Instead, unfreeze the last 6 blocks.

```python
# DINOv2 ViT-S/14 has 12 blocks
# Unfreeze last 6 blocks (10.7M trainable parameters)

for i, block in enumerate(model.blocks):
    if i >= 6:  # Last 6 blocks
        for param in block.parameters():
            param.requires_grad = True
    else:  # First 6 blocks stay frozen
        for param in block.parameters():
            param.requires_grad = False
```

**2. Weight EMA (Exponential Moving Average)**

Track an exponential moving average of model weights. Use EMA model for epoch selection (more stable than raw weights).

```python
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                      for name, param in model.named_parameters()}
    
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name]
                + (1 - self.decay) * param.data
            )
    
    def apply_shadow(self):
        # Replace model weights with EMA weights
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])
```

**3. Data Augmentation**

Medical images have fixed orientation (no horizontal flip!). Use:
- Small rotations (±10°)
- Small scale changes (0.9-1.1×)
- Small shifts (±10 pixels)

```python
from albumentations import Compose, Rotate, ShiftScaleRotate

augmentation = Compose([
    Rotate(limit=10, p=0.5),
    ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=0,  # Already handled above
        p=0.5
    ),
])
```

#### Implementation Steps

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3 IMPLEMENTATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Step 1: Partial Unfreezing                                                     │
│   ──────────────────────────                                                     │
│   • Unfreeze last 6 blocks of DINOv2 (10.7M trainable params)                    │
│   • Keep first 6 blocks frozen                                                   │
│                                                                                  │
│   Step 2: Lower Learning Rate for Backbone                                       │
│   ────────────────────────────────────────────                                   │
│   • Attention head: lr=1e-3                                                      │
│   • Backbone (unfrozen blocks): lr=1e-5 (100× smaller)                           │
│                                                                                  │
│   Step 3: Add Weight EMA                                                         │
│   ──────────────────────────                                                     │
│   • Track EMA of all trainable parameters                                        │
│   • Use EMA model for validation and epoch selection                             │
│                                                                                  │
│   Step 4: Add Data Augmentation                                                  │
│   ────────────────────────────                                                   │
│   • Small rotations (±10°)                                                       │
│   • Small scale/shift (±10%)                                                     │
│   • NO horizontal flip (destroys medial/lateral information)                     │
│                                                                                  │
│   Step 5: Train with EMA                                                         │
│   ────────────────────────                                                       │
│   • Train for 10 epochs per fold                                                 │
│   • Update EMA after each step                                                   │
│   • Validate with EMA model                                                      │
│                                                                                  │
│   Step 6: Select Best Epoch                                                      │
│   ────────────────────────────                                                   │
│   • Track validation AUC per epoch (using EMA model)                             │
│   • Select epoch with best average AUC across folds                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Expected Results

| Metric | Target | Improvement over Phase 2 |
|--------|--------|--------------------------|
| Gold-58 Macro AUC | 0.82-0.84 | +0.05-0.07 |
| Public LB | 0.86-0.88 | +0.08-0.10 |

#### Success Criteria

- ✅ Gold-58 AUC ≥ 0.82
- ✅ Public LB ≥ 0.86
- ✅ EMA model selected (not raw weights)
- ✅ No horizontal flip augmentation

---

### Phase 4: Frontier

**Duration**: Days 9-12  
**GPU Required**: Yes (2-4 GPUs, ~20 hours)  
**Difficulty**: 🔴 Advanced  
**Prerequisites**: Phases 2-3 complete, ensemble methods

#### What You'll Learn

- Multi-architecture ensembling
- Specialist models for weak labels
- Multi-resolution inference
- External pretrained weights (RadImageNet)

#### The Problem

You have a strong baseline (~0.86 AUC). Now push toward 0.90+ by:
1. Using larger backbones (ViT-B/14)
2. Training specialist models for weak labels
3. Ensembling diverse architectures
4. Using external pretrained weights

#### Key Concepts

**1. ViT-B/14 vs ViT-S/14**

| Variant | Parameters | Performance |
|---------|------------|-------------|
| ViT-S/14 | 21M (5M trainable) | 0.883 LB |
| ViT-B/14 | 86M (10.7M trainable) | 0.90+ LB (expected) |

Larger capacity, but slower and more prone to overfitting.

**2. Specialist Models**

Some labels are weak (MCL, Synovitis, Fracture). Train dedicated models for these.

```python
# For each weak label, train a specialist
specialists = {
    'MCL': train_specialist(label='MCL', backbone='ViT-B/14'),
    'Synovitis': train_specialist(label='Synovitis', backbone='ViT-B/14'),
    'Fracture': train_specialist(label='Fracture', backbone='EfficientNet-B3'),
}

# At inference, use specialist predictions for weak labels
final_predictions = general_model.predict(study)
for label, specialist in specialists.items():
    final_predictions[label] = specialist.predict(study)
```

**3. Multi-Resolution Inference**

Train at 224px and 336px, ensemble predictions.

```python
# Train two models
model_224 = train_model(resolution=224)
model_336 = train_model(resolution=336)

# At inference, average predictions
pred_224 = model_224.predict(study)
pred_336 = model_336.predict(study)
final_pred = (pred_224 + pred_336) / 2
```

**4. RadImageNet Pretrained Weights**

RadImageNet is pretrained on 1.3M medical images. Using it improved ensemble from 0.899 → 0.906.

```python
# Load RadImageNet ResNet-50
model = ResNet50(pretrained='radimagenet')

# Fine-tune on knee MRI
fine_tune(model, knee_data)

# Add to ensemble with 20% weight
ensemble = Ensemble([
    (dinov2_model, 0.8),
    (radimagenet_model, 0.2),
])
```

#### Implementation Steps

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 4 IMPLEMENTATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Step 1: ViT-B/14 Backbone                                                      │
│   ──────────────────────────                                                     │
│   • Replace ViT-S/14 with ViT-B/14                                               │
│   • Fine-tune last 6 blocks                                                      │
│   • Expected: +0.01-0.02 AUC                                                     │
│                                                                                  │
│   Step 2: RadImageNet Pretrained Weights                                         │
│   ────────────────────────────────────                                           │
│   • Download RadImageNet ResNet-50 weights                                       │
│   • Fine-tune on knee MRI                                                        │
│   • Add to ensemble with 20% weight                                              │
│                                                                                  │
│   Step 3: Specialist Models for Weak Labels                                      │
│   ────────────────────────────────────────────                                   │
│   • Identify weak labels (MCL, Synovitis, Fracture)                              │
│   • Train dedicated models for each                                              │
│   • Use specialist predictions at inference                                      │
│                                                                                  │
│   Step 4: Multi-Resolution Inference                                             │
│   ────────────────────────────────                                               │
│   • Train at 224px and 336px                                                     │
│   • Ensemble predictions (rank averaging)                                        │
│                                                                                  │
│   Step 5: EfficientNet-B3 Branch                                                 │
│   ────────────────────────────                                                   │
│   • Train EfficientNet-B3 (independent architecture)                             │
│   • Add to ensemble for diversity                                                │
│                                                                                  │
│   Step 6: Target-Wise Rank Blending                                              │
│   ────────────────────────────────                                               │
│   • For each label, select best model(s)                                         │
│   • Per-label ensemble weights                                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| Public LB | 0.90+ | Current frontier: 0.906 |

#### Success Criteria

- ✅ Public LB ≥ 0.90
- ✅ Ensemble of 5+ models
- ✅ Specialist models for weak labels
- ✅ Multi-resolution inference

---

### Phase 5: Hardening

**Duration**: Ongoing  
**GPU Required**: Minimal  
**Difficulty**: 🟡 Intermediate  
**Prerequisites**: Phases 2-4 complete

#### What You'll Learn

- Robustness testing
- Edge case handling
- Submission safety

#### Tasks

| # | Task | Priority |
|---|------|----------|
| 5.1 | Monitor per-label AUC (any at chance = -0.029) | P0 |
| 5.2 | Fix Fracture label derivation (97% constant) | P1 |
| 5.3 | Synovitis from Effusion imputation | P1 |
| 5.4 | Submission safety (valid file before real work) | P0 |
| 5.5 | Transfer syntax census on test | P1 |

---

## 🔬 Quantum ML Pipeline

### Phase 6: Quantum Foundations

**Duration**: Days 13-15  
**GPU Required**: No (CPU simulation)  
**Difficulty**: 🟡 Intermediate  
**Prerequisites**: Quantum computing basics, linear algebra

#### What You'll Learn

- Quantum data encoding strategies
- Quantum circuits and gates
- Statevector simulation
- Feature extraction from classical models

#### Prerequisites: Quantum Computing Basics

Before starting this phase, you MUST understand:

**1. Qubits vs Classical Bits**

```
Classical bit: 0 or 1
Qubit: α|0⟩ + β|1⟩ where |α|² + |β|² = 1

|0⟩ = [1, 0]ᵀ
|1⟩ = [0, 1]ᵀ

Measurement:
P(0) = |α|²
P(1) = |β|²
```

**2. Quantum Gates**

```
Single-qubit gates:
X (NOT): |0⟩ → |1⟩, |1⟩ → |0⟩
Y: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
Z: |0⟩ → |0⟩, |1⟩ → -|1⟩
H (Hadamard): |0⟩ → (|0⟩ + |1⟩)/√2

Rotation gates:
RX(θ), RY(θ), RZ(θ): Rotate around X/Y/Z axis by angle θ

Multi-qubit gates:
CNOT: |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩
CZ: |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |10⟩, |11⟩ → -|11⟩
```

**3. Quantum Circuits**

```
Example circuit:
|0⟩ ──[H]──[RY(θ)]──[CNOT]──[Measure]──
|0⟩ ─────────[X]──────•──────[Measure]──

Step-by-step:
1. Apply H to qubit 0 → superposition
2. Apply RY(θ) to qubit 0 → rotation
3. Apply X to qubit 1 → flip
4. Apply CNOT (control=0, target=1) → entanglement
5. Measure both qubits
```

**Recommended Resources:**
- Qiskit Textbook: https://qiskit.org/learn
- IBM Quantum Learning: https://learning.quantum.ibm.com/
- 3Blue1Brown Quantum Mechanics (YouTube)

#### The Problem

Quantum computers work with qubits, but your data is classical (DINOv2 features). You need to:
1. Extract features from your trained classical model
2. Encode classical features into quantum states
3. Validate that your quantum circuits work correctly

#### Key Concepts

**1. Feature Extraction**

Use your trained DINOv2 model to extract features.

```python
# Load trained model
model = load_trained_model()
model.eval()

# Extract features
with torch.no_grad():
    features = model.backbone(images)  # shape: (batch, 768)
    
# Save for quantum models
np.save('features.npy', features.cpu().numpy())
```

**2. Dimensionality Reduction**

Quantum circuits are expensive. Reduce 768-dim features to 8-16 dims.

```python
from sklearn.decomposition import PCA

# Reduce to 8 dimensions (3 qubits for amplitude encoding)
pca = PCA(n_components=8)
features_reduced = pca.fit_transform(features)

# Normalize to unit vector (required for amplitude encoding)
features_normalized = features_reduced / np.linalg.norm(features_reduced, axis=1, keepdims=True)
```

**3. Amplitude Encoding**

Encode N features into amplitudes of n = log₂(N) qubits.

```python
from qiskit import QuantumCircuit
import numpy as np

def amplitude_encoding(features):
    """
    Encode feature vector into quantum state amplitudes.
    features: normalized vector of length 2^n
    Returns: QuantumCircuit preparing the state
    """
    n_qubits = int(np.log2(len(features)))
    qc = QuantumCircuit(n_qubits)
    
    # Initialize state with amplitudes = features
    # |ψ⟩ = Σᵢ features[i] |i⟩
    qc.initialize(features, range(n_qubits))
    
    return qc

# Example: 8 features → 3 qubits
features = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1])
features = features / np.linalg.norm(features)  # normalize
qc = amplitude_encoding(features)
```

**Pros**: Exponentially compact (8 features in 3 qubits)  
**Cons**: Deep circuits, hard to train

**4. Angle Encoding**

Encode each feature as a rotation angle.

```python
def angle_encoding(features):
    """
    Encode each feature as RY rotation angle.
    features: vector of length n
    Returns: QuantumCircuit with n qubits
    """
    n_qubits = len(features)
    qc = QuantumCircuit(n_qubits)
    
    # Apply RY(feature[i]) to qubit i
    for i, feature in enumerate(features):
        qc.ry(feature, i)
    
    return qc

# Example: 8 features → 8 qubits
features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
qc = angle_encoding(features)
```

**Pros**: Simple, shallow circuits  
**Cons**: Linear scaling (N features → N qubits)

**5. ZZFeatureMap**

Entangling feature map that captures correlations.

```python
from qiskit.circuit.library import ZZFeatureMap

def zz_feature_map(features, reps=2):
    """
    ZZFeatureMap: Pauli-Z rotations + entangling ZZ gates.
    Captures feature correlations through entanglement.
    """
    n_qubits = len(features)
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    
    # Bind features to circuit parameters
    bound_circuit = feature_map.bind_parameters(features)
    
    return bound_circuit
```

#### Implementation Steps

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 6 IMPLEMENTATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Step 1: Feature Extraction                                                     │
│   ────────────────────────                                                       │
│   • Load trained DINOv2 model from Phase 3                                       │
│   • Extract CLS token features for all training studies                          │
│   • Save as numpy array: shape (4407, 768)                                       │
│                                                                                  │
│   Step 2: Dimensionality Reduction                                               │
│   ────────────────────────────                                                   │
│   • Apply PCA to reduce 768 → 8-16 dimensions                                    │
│   • Explained variance should be >80%                                            │
│   • Normalize features to unit vectors                                           │
│                                                                                  │
│   Step 3: Amplitude Encoding                                                     │
│   ────────────────────────                                                       │
│   • Implement amplitude encoding circuit                                         │
│   • Test on sample features                                                      │
│   • Verify statevector matches input features                                    │
│                                                                                  │
│   Step 4: Angle Encoding                                                         │
│   ──────────────────                                                             │
│   • Implement angle encoding circuit                                             │
│   • Test on sample features                                                      │
│   • Compare circuit depth vs amplitude encoding                                  │
│                                                                                  │
│   Step 5: ZZFeatureMap                                                           │
│   ────────────────                                                               │
│   • Implement ZZFeatureMap using Qiskit                                          │
│   • Test on sample features                                                      │
│   • Visualize circuit                                                            │
│                                                                                  │
│   Step 6: Circuit Simulation Validation                                          │
│   ────────────────────────────────                                               │
│   • Use statevector simulator to verify circuits                                 │
│   • Check that measurement probabilities match expectations                      │
│   • Test on 100 sample features                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Validation Code

```python
from qiskit_aer import AerSimulator

def validate_encoding(encoding_fn, features, n_qubits):
    """Validate that encoding produces correct quantum state."""
    
    # Create circuit
    qc = encoding_fn(features)
    qc.save_statevector()
    
    # Simulate
    simulator = AerSimulator()
    result = simulator.run(qc).result()
    statevector = result.get_statevector()
    
    # For amplitude encoding, statevector amplitudes should match features
    if encoding_fn.__name__ == 'amplitude_encoding':
        expected = np.zeros(2**n_qubits, dtype=complex)
        expected[:len(features)] = features
        fidelity = np.abs(np.dot(statevector.conj(), expected))**2
        print(f"Fidelity: {fidelity:.4f}")
        assert fidelity > 0.99, "Amplitude encoding failed"
    
    # For angle encoding, check measurement probabilities
    # (more complex, depends on specific implementation)
    
    print("✓ Encoding validated successfully")
```

#### Success Criteria

- ✅ Features extracted and reduced to 8-16 dimensions
- ✅ All encoding strategies produce valid quantum states
- ✅ Statevector simulation matches expected distributions
- ✅ Fidelity > 0.99 for amplitude encoding

---

### Phase 7: Variational Quantum Classifiers (VQC)

**Duration**: Days 16-20  
**GPU Required**: No (CPU simulation)  
**Difficulty**: 🔴 Advanced  
**Prerequisites**: Phase 6 complete, quantum optimization

#### What You'll Learn

- Variational quantum circuits
- Parameter-shift rule for gradients
- Quantum optimization (SPSA, Adam)
- Per-label quantum classification

#### Prerequisites: Variational Quantum Circuits

A Variational Quantum Circuit (VQC) is a parameterized quantum circuit trained like a neural network.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      VARIATIONAL QUANTUM CIRCUIT                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   |0⟩ ──[Feature Map]──[Ansatz Layer 1]──[Ansatz Layer 2]──[Measure Z]──       │
│   |0⟩ ──[Feature Map]──[Ansatz Layer 1]──[Ansatz Layer 2]──[Measure Z]──       │
│   |0⟩ ──[Feature Map]──[Ansatz Layer 1]──[Ansatz Layer 2]──[Measure Z]──       │
│    ⋮                                                                                │
│   |0⟩ ──[Feature Map]──[Ansatz Layer 1]──[Ansatz Layer 2]──[Measure Z]──       │
│                                                                                  │
│   Feature Map: Encodes input data (RY(xᵢ) or ZZFeatureMap)                      │
│   Ansatz: Trainable layers (RY(θ) + CNOT entangling)                            │
│   Measure: ⟨Z⟩ expectation value on output qubit                                │
│                                                                                  │
│   Output: P(y=1|x) = (⟨Z⟩ + 1) / 2                                              │
│                                                                                  │
│   Training: Minimize loss by adjusting θ using gradient descent                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Parameter-Shift Rule**

How to compute gradients on quantum hardware (or simulator):

```
∂⟨O⟩/∂θᵢ = [⟨O⟩(θᵢ + π/2) - ⟨O⟩(θᵢ - π/2)] / 2

Where:
- ⟨O⟩ is the expectation value (e.g., ⟨Z⟩)
- θᵢ is the parameter to differentiate
- Shift by ±π/2 and measure difference
```

This is the quantum analog of backpropagation!

#### Architecture

```python
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import VQC as QiskitVQC

class VariationalQuantumClassifier:
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit = self._build_circuit()
    
    def _build_circuit(self):
        """Build parameterized quantum circuit."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Feature map (angle encoding)
        for i in range(self.n_qubits):
            qc.ry(f'x_{i}', i)  # Parameterized by input
        
        # Ansatz layers
        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qc.ry(f'θ_{layer}_{i}', i)
            
            # Entangling layer (linear connectivity)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc
    
    def forward(self, x, params):
        """Forward pass: encode features, apply ansatz, measure."""
        # Bind input features and parameters
        param_dict = {f'x_{i}': x[i] for i in range(self.n_qubits)}
        param_dict.update(params)
        
        bound_circuit = self.circuit.bind_parameters(param_dict)
        
        # Simulate and measure ⟨Z⟩ on qubit 0
        # ... (simulation code)
        
        return expectation_value
```

#### Training Loop

```python
import numpy as np

def train_vqc(model, X_train, y_train, n_epochs=100, lr=0.01):
    """Train VQC using parameter-shift rule."""
    
    # Initialize parameters randomly
    params = {f'θ_{l}_{i}': np.random.uniform(0, 2*np.pi)
              for l in range(model.n_layers)
              for i in range(model.n_qubits)}
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for x, y in zip(X_train, y_train):
            # Forward pass
            pred = model.forward(x, params)
            
            # Loss (binary cross-entropy)
            loss = -y * np.log(pred) - (1 - y) * np.log(1 - pred)
            epoch_loss += loss
            
            # Compute gradients using parameter-shift rule
            gradients = {}
            for param_name in params:
                # Shift parameter by +π/2
                params_plus = params.copy()
                params_plus[param_name] += np.pi / 2
                pred_plus = model.forward(x, params_plus)
                
                # Shift parameter by -π/2
                params_minus = params.copy()
                params_minus[param_name] -= np.pi / 2
                pred_minus = model.forward(x, params_minus)
                
                # Parameter-shift gradient
                grad = (pred_plus - pred_minus) / 2
                gradients[param_name] = grad
            
            # Update parameters (gradient descent)
            for param_name in params:
                params[param_name] -= lr * gradients[param_name]
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(X_train):.4f}")
    
    return params
```

#### Implementation Steps

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 7 IMPLEMENTATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Step 1: Design Ansatz                                                          │
│   ────────────────────                                                           │
│   • Hardware-efficient ansatz (RY + CNOT layers)                                 │
│   • Start with 8 qubits, 2-3 layers                                              │
│   • Visualize circuit                                                            │
│                                                                                  │
│   Step 2: Implement VQC Forward Pass                                             │
│   ────────────────────────────────                                               │
│   • Encode features (angle encoding)                                             │
│   • Apply ansatz layers                                                          │
│   • Measure ⟨Z⟩ on output qubit                                                  │
│   • Convert to probability: P(y=1) = (⟨Z⟩ + 1) / 2                              │
│                                                                                  │
│   Step 3: Implement Parameter-Shift Gradient                                     │
│   ────────────────────────────────────                                           │
│   • For each parameter, shift by ±π/2                                            │
│   • Compute gradient: (f(θ+π/2) - f(θ-π/2)) / 2                                │
│   • Verify gradient numerically                                                  │
│                                                                                  │
│   Step 4: Training Loop                                                          │
│   ────────────────                                                               │
│   • Binary cross-entropy loss                                                    │
│   • Adam optimizer (or SPSA for hardware)                                        │
│   • 100 epochs per label                                                         │
│                                                                                  │
│   Step 5: Per-Label VQC Training                                                 │
│   ────────────────────────────                                                   │
│   • Train one VQC per label (12 total)                                           │
│   • Use reduced features from Phase 6                                            │
│   • Save trained parameters                                                      │
│                                                                                  │
│   Step 6: VQC Ensemble                                                           │
│   ────────────────────                                                           │
│   • Collect predictions from 12 VQCs                                             │
│   • Rank averaging across labels                                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| Per-label AUC | 0.65-0.75 | Quantum models are early-stage |
| Training time | 2-6 hours | CPU simulation |

#### Success Criteria

- ✅ VQC trains without errors
- ✅ Loss decreases over epochs
- ✅ Per-label AUC ≥ 0.65
- ✅ Gradient computation verified (parameter-shift)

---

### Phase 8: Quantum Kernel Methods

**Duration**: Days 21-24  
**GPU Required**: No (CPU simulation)  
**Difficulty**: 🔴 Advanced  
**Prerequisites**: Phase 6 complete, kernel methods

#### What You'll Learn

- Quantum kernel methods
- Swap test for fidelity estimation
- Quantum SVM
- Kernel alignment

#### Prerequisites: Quantum Kernels

A quantum kernel measures similarity between data points using quantum circuits.

```
K(xᵢ, xⱼ) = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²

Where:
- |ψ(x)⟩ = U(x)|0⟩ is the feature map applied to data x
- ⟨ψ(xᵢ)|ψ(xⱼ)⟩ is the inner product (fidelity)
- Computed via swap test or fidelity circuit
```

**Swap Test Circuit**

```
|0⟩ ──[H]───────────────[CSWAP]──────────[H]──[Measure]──
|ψ(xᵢ)⟩ ──────────────────│────────────────────
|ψ(xⱼ)⟩ ──────────────────│────────────────────

If ancilla measures |0⟩: states are identical
P(0) = (1 + |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²) / 2
```

#### Implementation

```python
from qiskit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel

def quantum_kernel_matrix(X, feature_map):
    """Compute quantum kernel matrix."""
    
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    K = kernel.evaluate(X)  # shape: (n_samples, n_samples)
    
    return K

# Train SVM on quantum kernel
from sklearn.svm import SVC

svm = SVC(kernel='precomputed', C=1.0)
svm.fit(K_train, y_train)
predictions = svm.predict(K_test)
```

#### Success Criteria

- ✅ Quantum kernel matrix computed
- ✅ QSVM trained and validated
- ✅ Per-label AUC ≥ 0.70

---

### Phase 9: Hybrid Classical-Quantum Models

**Duration**: Days 25-30  
**GPU Required**: Yes (for classical backbone)  
**Difficulty**: 🔴 Advanced  
**Prerequisites**: Phases 3, 6-8 complete

#### What You'll Learn

- Hybrid classical-quantum architectures
- Quantum-classical gradient flow
- End-to-end training

#### Architecture

```
MRI Slices ──[DINOv2 (frozen)]──[Classical MLP]──[Quantum Layer]──[12 logits]
```

#### Success Criteria

- ✅ Hybrid model trains end-to-end
- ✅ Gold-58 AUC ≥ 0.80

---

### Phase 10: Quantum Ensemble & Benchmarking

**Duration**: Days 31-35  
**GPU Required**: Yes  
**Difficulty**: 🟡 Intermediate  
**Prerequisites**: Phases 4, 7-9 complete

#### What You'll Learn

- Ensemble methods
- Quantum advantage analysis
- Benchmarking frameworks

#### Success Criteria

- ✅ Final macro AUC ≥ 0.88
- ✅ Quantum advantage documented per label

---

## 📦 Dependencies & Setup

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install classical ML dependencies
pip install -e ".[dev]"

# Install quantum ML dependencies
pip install -e ".[quantum]"
```

### pyproject.toml

```toml
[project]
name = "rsna-knee"
version = "0.1.0"
description = "RSNA Knee Abnormality Detection: Classical to Quantum ML"
requires-python = ">=3.10"

dependencies = [
    # Data handling
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # Machine learning
    "scikit-learn>=1.3.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "timm>=0.9.0",  # For DINOv2
    
    # Medical imaging
    "pydicom>=2.4.0",
    "pylibjpeg>=1.0.0",
    "python-gdcm>=3.0.0",
    
    # LLM for label extraction
    "openai>=1.0.0",
    
    # Utilities
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "albumentations>=1.3.0",
    "mlflow>=2.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "jupyter>=1.0.0",
]

quantum = [
    "qiskit>=1.0.0",
    "qiskit-machine-learning>=0.7.0",
    "qiskit-algorithms>=0.3.0",
    "qiskit-aer>=0.13.0",
    "pennylane>=0.35.0",
    "qulacs>=0.6.0",
]
```

---

## 📅 Timeline & Milestones

```
Week 1: Classical Foundation (Phases 0-2)
──────────────────────────────────────────
Day 1:    Phase 0 - Label extraction
Day 2-3:  Phase 1 - DICOM pipeline
Day 4-5:  Phase 2 - Baseline model

Milestone: Frozen DINOv2 baseline (~0.78 AUC)

Week 2: Classical Optimization (Phases 3-5)
────────────────────────────────────────────
Day 6-8:   Phase 3 - Fine-tuning
Day 9-12:  Phase 4 - Frontier models
Ongoing:   Phase 5 - Hardening

Milestone: Classical ensemble (~0.90 AUC)

Week 3: Quantum Foundations (Phases 6-7)
──────────────────────────────────────────
Day 13-15: Phase 6 - Quantum foundations
Day 16-20: Phase 7 - VQC training

Milestone: Per-label VQCs trained

Week 4: Quantum Advanced (Phases 8-9)
───────────────────────────────────────
Day 21-24: Phase 8 - Quantum kernels
Day 25-30: Phase 9 - Hybrid models

Milestone: Hybrid classical-quantum model

Week 5: Benchmarking (Phase 10)
────────────────────────────────
Day 31-35: Phase 10 - Ensemble & benchmarking

Milestone: Final results, quantum advantage analysis
```

---

## 🎓 Final Notes

### What Success Looks Like

**Classical Pipeline:**
- Competitive Kaggle submission (~0.90 AUC)
- Robust, well-tested codebase
- Clear documentation

**Quantum Pipeline:**
- Publication-quality comparison of classical vs quantum
- Analysis of where quantum models help (if anywhere)
- Open-source contribution to quantum ML for medical imaging

### Common Questions

**Q: Do I need a real quantum computer?**  
A: No! All phases use CPU simulation. Real hardware is optional for Phase 10.

**Q: What if quantum models underperform?**  
A: That's a valid research outcome! Document where quantum doesn't help.

**Q: How much will this cost?**  
A: Classical: ~$50-100 for GPT-4o label extraction. Quantum: $0 (simulation).

**Q: Can I skip the quantum phases?**  
A: Yes! Phases 0-5 are a complete project on their own.

### Resources

**Classical ML:**
- Fast.ai: https://course.fast.ai/
- PyTorch Tutorials: https://pytorch.org/tutorials/
- DINOv2 Paper: https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/

**Quantum ML:**
- Qiskit Textbook: https://qiskit.org/learn
- PennyLane Demos: https://pennylane.ai/qml/demos
- Quantum ML Review: https://arxiv.org/abs/2201.04048

**Competition:**
- RSNA Knee: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- Community Discussions: Check RESEARCH_SUMMARY.md

---

**Good luck! 🚀**

*Last updated: August 14, 2026*
