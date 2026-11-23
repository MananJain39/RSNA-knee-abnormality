# Latest project integration

Imported on November 1, 2026 from `/home/manan-jain/Downloads/knee_project/knee`.
The original Downloads copy is unchanged. The import manifest records SHA-256
hashes of all 97 source files: 92 were copied and five existing competition
CSVs were identical. No existing project files were overwritten by the import.
README was subsequently annotated to identify the current pipeline.

## Current versus earlier pipeline

| Area | Current CNN rebuild | Earlier DINOv2 pipeline |
| --- | --- | --- |
| Training | `scripts/train.py` | `scripts/train_classical.py`, `src/classical/` |
| Preprocessing | `scripts/preprocess.py`, `scripts/pack.py` | `scripts/preprocess_volumes.py`, `src/imaging/` |
| Labels | `data/labels_v1.csv`, `scripts/llm_labels.py`, `scripts/build_labels.py` | `data/processed/labels_final.csv`, `src/labels/` |
| Evaluation | `scripts/evaluate.py` | `scripts/evaluate_classical.py` |
| Inference | `scripts/infer.py`, `kaggle/` | `scripts/infer_classical.py`, `kaggle_kernels/` |
| Results | `exp/`, `docs/LEARNING_JOURNAL.md` | `experiments/`, historical docs |

The caches, model heads and checkpoints are not interchangeable. Older planning
documents and `docs/top-team-comparison-2026-10-30.md` concern the earlier pipeline.
The current model families are B1 ResNet34d, B2 EfficientNetV2-B0 and B4 ResNet34d
with more sampled slices, each trained in four folds.

## Setup and paths

Install the current pipeline dependencies with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[rebuild,dev]'
```

Existing GPU job defaults (`/data/pack`, `/data/code`, `/data/exp`) and cloud
references are retained for compatibility. Supply local paths explicitly when
running here, for example:

```bash
.venv/bin/python scripts/train.py --pack /path/to/pack \
  --labels data/labels_v1.csv --train-csv data/raw/train.csv \
  --backbone resnet34d.ra2_in1k --head pool --fold 0 \
  --out /path/to/run
```

Heavy DICOMs, packed volumes and model weights are not in the Downloads copy and
were not fetched. No remote jobs, uploads, extraction calls or training were run.
The Makefile Python path now follows the current checkout; existing targets
still address the earlier pipeline. The `rebuild` dependency group is additive.

## Unchanged research issue

Current evaluation rank-averages family predictions, whereas submission inference
probability-averages checkpoints. This known difference was preserved during
integration so that the imported results and prediction behavior remain traceable.
