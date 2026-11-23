# B2 — EfficientNetV2-B0 + pool head (ensemble member)

**Why:** a different CNN family from ResNet makes different errors → better ensemble.
**Command:** `python scripts/train.py --backbone tf_efficientnetv2_b0.in1k --head pool --lr 3e-4 --head-lr 1e-3 --fold f --folds 4 --epochs 10 --bs 6 --out /data/exp/B2_effv2b0_pool`
All other parameters identical to B1 (see `../B1_resnet34d_pool/README.md` and `args_fold*.json`).

**Incident:** fold 1 crashed with CUDA out-of-memory on its first run (GPU 1 was shared with B4 fold 1 and the
laterality job). Re-run alone with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → completed.
Two jobs per GPU (B2 + B4) ran at ~5 min/epoch each instead of ~2–3 alone.

**Results:** OOF 0.882, gold-58 per fold 0.886 / 0.896 / 0.894 / 0.896 (mean 0.893), 4-fold ensemble **0.905**.
Better than ResNet on MCL (0.94 vs 0.91), Synovitis (0.79 vs 0.76), Contusion (0.95 vs 0.94); worse on PF OA. → `results.md`.
