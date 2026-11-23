# RSNA Knee MRI Abnormality Detection

A research pipeline for predicting **12 knee abnormalities from multi-sequence MRI studies**. It combines report-derived soft labels, canonical DICOM preprocessing, 2.5D convolutional neural networks, report-grouped cross-validation, and model ensembling.

The current pipeline is the CNN rebuild in `scripts/preprocess.py`, `scripts/pack.py`, `scripts/train.py`, `scripts/evaluate.py`, and `scripts/infer.py`. The earlier DINOv2 implementation remains available for reference, but its data caches and checkpoints are not interchangeable with this pipeline.

## Problem overview

The RSNA Knee Abnormality Detection task presents a weak-label learning problem:

- **4,407 training studies**, of which only **58 have expert labels** (Gold-58).
- The remaining **4,349 studies** have free-text radiology reports in multiple languages.
- Test studies contain images but no reports.
- Each study receives independent predictions for ACL, MCL, medial meniscus, lateral meniscus, medial OA, lateral OA, patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, and fracture.
- The evaluation metric is **macro ROC-AUC**, giving each finding equal weight.

Gold-58 is excluded from model training and checkpoint selection. Cross-validation folds group identical report texts to reduce leakage.

## Results

| Model or label set | OOF macro AUC vs report-derived labels | Gold-58 macro AUC |
| --- | ---: | ---: |
| Sonnet/Haiku soft-label blend | — | 0.8762 |
| B1: ResNet34d, 16 slices, 4 folds | 0.8858 | 0.9052 |
| B2: EfficientNetV2-B0, 16 slices, 4 folds | 0.8823 | 0.9049 |
| B4: ResNet34d, 24 slices, 4 folds | 0.8847 | 0.9012 |
| **B1 + B2 + B4 rank-mean ensemble** | **0.8934** | **0.9076** |

The B1 late Kaggle submission scored **0.917 public / 0.905 private**.

These numbers answer different questions: OOF AUC compares held-out predictions with noisy report-derived targets, while Gold-58 compares against expert labels on only 58 studies. Neither is directly comparable to another team's leaderboard or development score.

Detailed reports are in [`exp/01_labels/results.md`](exp/01_labels/results.md), [`exp/B1_resnet34d_pool/results.md`](exp/B1_resnet34d_pool/results.md), and [`exp/ENS_B1B2B4/results.md`](exp/ENS_B1B2B4/results.md).

## Pipeline

```text
Radiology reports ──► Claude on Bedrock ──► soft labels ───────────────┐
                                                                      ├─► 2.5D CNN training
DICOM studies ──► canonical volumes ──► packed uint8 memmap ──────────┘
                                                        │
                                                        ▼
                                      report-grouped 4-fold validation
                                                        │
                                      ┌─────────────────┴─────────────────┐
                                      ▼                                   ▼
                              OOF/Gold evaluation             test-time ensemble
                                                                      │
                                                                      ▼
                                                              submission.csv
```

1. **Label extraction:** an LLM assigns a fixed finding state to each report. Sonnet and Haiku outputs are blended, and unmentioned synovitis is imputed from effusion. The curated result is `data/labels_v1.csv`; running the API pipeline is optional.
2. **Preprocessing:** DICOM slices are sorted geometrically, reoriented, normalized for laterality, resized, intensity-clipped, converted to `uint8`, and resampled to 32 slices per selected series.
3. **Packing:** five series slots per study are packed into an `(N, 5, 32, 256, 256)` memory-mapped array with missing-series masks.
4. **Training:** neighboring slices form three-channel inputs to a timm CNN. Masked max/mean pooling summarizes each series before a 12-output classification head.
5. **Evaluation and inference:** folds produce OOF and Gold-58 predictions. Evaluation rank-averages model families; submission inference averages checkpoint probabilities.

## Installation

**Requirements:** Python 3.10+, and a CUDA-capable GPU for the training CLI. Preprocessing and most tests run on CPU.

```bash
git clone https://github.com/MananJain39/RSNA-knee-abnormality.git
cd RSNA-knee-abnormality
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[rebuild,dev]"
```

Install a PyTorch build appropriate for your CUDA version if the default dependency resolution does not provide one. The `rebuild` extra contains the current CNN pipeline; the separate `classical` extra is only needed for the legacy DINOv2 code.

Verify the installation:

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite uses synthetic data and mocked cloud executables; it does not require AWS calls, Kaggle access, DICOM downloads, or model checkpoints.

## Data layout

The repository includes competition CSVs, curated labels, small development splits, and experiment metadata. Large DICOM archives, packed volumes, and checkpoints are intentionally excluded.

```text
data/
├── raw/
│   ├── train.csv
│   ├── train_series.csv
│   ├── test.csv
│   └── test_series.csv
├── processed/
│   ├── splits/                 # sample3 and sample200 development splits
│   └── volumes_sample200/      # lightweight preprocessing provenance
└── labels_v1.csv               # final report-derived soft labels + Gold-58
```

Supply external paths explicitly rather than relying on the GPU-server defaults under `/data`.

## Run the image pipeline

Set paths for your local or remote storage:

```bash
SERIES_ROOT=/path/to/train_series
VOLUME_DIR=/path/to/volumes
PACK_DIR=/path/to/pack
RUN_DIR=/path/to/runs/B1
```

### 1. Convert DICOM series to canonical volumes

```bash
.venv/bin/python scripts/preprocess.py \
  --series-root "$SERIES_ROOT" \
  --series-csv data/raw/train_series.csv \
  --out "$VOLUME_DIR" \
  --workers 8
```

To read directly from the competition archive, replace `--series-root` with `--zip /path/to/competition.zip`. Use `--limit 5` for a small preprocessing check.

### 2. Pack volumes for memory-mapped training

```bash
.venv/bin/python scripts/pack.py \
  --vol-dir "$VOLUME_DIR" \
  --labels data/labels_v1.csv \
  --out "$PACK_DIR"
```

This writes `vols.npy` and `index.csv`. For a synthetic packing check, use `--fake 5` instead of `--vol-dir`.

### 3. Train four folds

The following reproduces the B1 recipe sequentially. Folds can run in parallel when separate GPUs are available.

```bash
for FOLD in 0 1 2 3; do
  .venv/bin/python scripts/train.py \
    --pack "$PACK_DIR" \
    --labels data/labels_v1.csv \
    --train-csv data/raw/train.csv \
    --backbone resnet34d.ra2_in1k \
    --head pool \
    --fold "$FOLD" \
    --folds 4 \
    --epochs 10 \
    --bs 6 \
    --n-slices 16 \
    --out "$RUN_DIR"
done
```

Each fold writes `args.json`, epoch logs, checkpoints, OOF predictions, Gold-58 predictions, and study-ID files under `$RUN_DIR/fold*/`. Add `--smoke` for a one-epoch, reduced-data GPU check.

### 4. Evaluate one run or an ensemble

```bash
.venv/bin/python scripts/evaluate.py "$RUN_DIR" \
  --labels data/labels_v1.csv \
  --out "$RUN_DIR/results.md"
```

Pass multiple run directories to additionally evaluate their rank-mean ensemble:

```bash
.venv/bin/python scripts/evaluate.py /path/to/B1 /path/to/B2 /path/to/B4 \
  --labels data/labels_v1.csv \
  --out /path/to/ensemble-results.md
```

### 5. Create a submission

```bash
TEST_SERIES_ROOT=/path/to/test_series
MODEL_ROOT=/path/to/model-dataset/models

.venv/bin/python scripts/infer.py \
  --series-root "$TEST_SERIES_ROOT" \
  --series-csv data/raw/test_series.csv \
  --test-csv data/raw/test.csv \
  --ckpts "$MODEL_ROOT/*/fold*/best.pt" \
  --out submission.csv \
  --workers 4
```

Every `best.pt` must have its training-generated `args.json` beside it. Inference preserves the test CSV's study order, periodically rewrites the submission for crash safety, and assigns `0.5` to studies that cannot be processed.

For offline Kaggle execution, [`kaggle/kernel/infer_kernel.py`](kaggle/kernel/infer_kernel.py) discovers the competition input and attached model dataset, then invokes the same inference CLI.

## Repository layout

```text
.
├── data/                  # CSVs, labels, splits, and lightweight provenance
├── docs/                  # guides, integration notes, and research comparisons
├── exp/
│   ├── 01_labels/         # LLM outputs, bake-off, and label-build results
│   ├── B1_resnet34d_pool/ # ResNet34d baseline configuration and metrics
│   ├── B2_effv2b0_pool/   # EfficientNetV2-B0 configuration and metrics
│   ├── B4_resnet34d_24sl/ # 24-slice ablation
│   ├── ENS_B1B2B4/        # three-run ensemble evaluation
│   └── diagnostics/       # collapse investigation and laterality audit
├── kaggle/                # offline inference kernel and dataset metadata
├── output/pdf/            # generated model-comparison brief
├── scripts/               # current pipeline, utilities, and legacy entry points
├── src/                   # earlier DINOv2/classical implementation
└── tests/                 # unit, integration, and mocked operational tests
```

## Documentation

- [Learning journal](docs/LEARNING_JOURNAL.md) — concepts, decisions, experiments, and lessons.
- [Code guide](docs/CODE_GUIDE.md) — script-by-script CLI and implementation reference.
- [Integration notes](docs/MERGE_NOTES.md) — current versus legacy paths and setup.
- [Diagnostics](exp/diagnostics/README.md) — attention-head collapse, pooling fix, benchmarks, and laterality checks.
- [Team session outline](docs/TEAM_SESSION_OUTLINE.md) — a concise presentation version of the project.
- [Leading-solution comparison](docs/top-team-comparison-2026-10-30.md) — historical analysis of the earlier pipeline.
- [Model-comparison PDF](output/pdf/rsna_knee_model_comparison.pdf) — current baseline and gap analysis.

## Research notes

- LLM blend weights and synovitis imputation were selected using Gold-58, so its label-quality estimate is slightly optimistic.
- Gold-58 is small; per-label and macro AUC differences should be interpreted with substantial uncertainty.
- Evaluation rank-averages model families, while inference probability-averages checkpoints. This distinction is preserved intentionally.
- Cloud download and SSM helpers are configurable through environment variables; no account-specific instance or role identifiers are required in the scripts.
- This repository is for research and competition development, **not clinical diagnosis or patient care**.
