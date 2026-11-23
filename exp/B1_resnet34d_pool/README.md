# B1 — ResNet34d 2.5D + pool head, 4-fold CV (first working image model)

## What and why
First image model that learned properly (after the A1 collapse, see `exp/diagnostics/`). Deliberately simple:
a small, fast pretrained CNN + the simplest pooling head, so we get a trustworthy baseline before adding complexity.

## Exact command (one process per fold, one GPU each, all 4 in parallel)
```
python scripts/train.py --backbone resnet34d.ra2_in1k --head pool --lr 3e-4 --head-lr 1e-3 \
    --fold {0,1,2,3} --folds 4 --epochs 10 --bs 6 --out /data/exp/B1_resnet34d_pool
```

## All parameters (from `fold*/args.json`)
| Parameter | Value | Meaning / why this value |
|---|---|---|
| `backbone` | `resnet34d.ra2_in1k` | ResNet-34 ("d" variant: better stem) pretrained on ImageNet-1k (timm "ra2" recipe). Fastest of the candidates we benchmarked (~2,660 img/s). |
| `head` | `pool` | Per-series masked max+mean over slices → concat 5 series → Dropout(0.2) → Linear(2,560 → 12). |
| `dim` | 256 | CNN features (512-d) projected to 256 + LayerNorm before pooling. |
| `folds` | 4 | 4-fold CV over 4,349 non-gold studies, grouped by identical report text. One GPU per fold. |
| `epochs` | 10 | Val AUC was still rising at epoch 8–9 (see `log_fold*.jsonl`); 10 fits the time budget (~22 min). |
| `bs` | 6 | Studies per step = 6 × 5 series × 16 slices = 480 CNN images/step (~11 GB GPU memory). |
| `n_slices` | 16 | Of 32 stored slices: evenly spaced, random offset in training (augmentation), centred at eval. |
| `crop` | 224 | Random affine crop from 256² in training; centre crop at eval. |
| `lr` | 3e-4 | AdamW learning rate for the pretrained CNN. |
| `head_lr` | 1e-3 | Higher LR for the new randomly initialised layers. |
| `wd` | 0.02 | AdamW weight decay (regularisation). |
| warm-up / schedule | 300 steps linear → cosine to 0 | Protects pretrained weights early; settles late. |
| precision | bfloat16 autocast, TF32 | ~2× faster, no accuracy loss on L40S. |
| grad clip | 2.0 | Stability. |
| drop_path | 0.1 | Stochastic depth in the backbone (regularisation). |
| slots | sag_fs, sag_t1, cor_fs, cor_t1, ax_fs | 5 series; missing ones masked. |
| seed | 0 | |

**Augmentation (all on GPU, `GpuBatcher`):** rotation ±12°, scale ±12.5%, shift ±8% (same transform for every slice of a
series); gamma ×exp(±0.25), contrast ±15%, brightness ±0.05 per series; 30% chance to hide one present series.
**No horizontal flips** (would swap medial/lateral).

**Loss:** binary cross-entropy vs soft labels (`data/labels_v1.csv`).
**Model selection:** epoch with best val-fold macro AUC vs LLM labels (binarised at 0.5). Gold-58 never used for
training or selection. Gold-58 is predicted by every fold model; the mean of the 4 = ensemble estimate.

## Sanity tests run before this
- Synthetic end-to-end smoke test (fake 64-study pack): train → validate → checkpoint OK.
- Overfit test (48 studies × 15 epochs): train AUC 0.985 ⇒ model can learn.
- Preprocessing parity: inference preprocessing reproduces training arrays bit-exactly (max diff 0).

## Results → `results.md`. Per-epoch logs → `log_fold{0..3}.jsonl`. Args → `args_fold{0..3}.json`.

Kaggle: weights uploaded as private dataset `mananjain39/rsna-knee-claude-models`; inference kernel
`mananjain39/rsna-knee-claude-infer` (`kaggle/kernel/infer_kernel.py` → `scripts/infer.py`).
