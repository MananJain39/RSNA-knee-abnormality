# Code Guide — every file, what it does, its parameters

Run order: `eda.py` → `llm_labels.py` → `score_labels.py` / `build_labels.py` → (GPU box) `download.sh` →
`preprocess.py` → `pack.py` → `train.py` (× folds × models) → `evaluate.py` → `infer.py` (Kaggle) .
Helpers: `ssm.sh` (run commands on the GPU box), `laterality.py` (diagnostic).

---

## `scripts/eda.py` — table EDA
Reads `data/raw/{train,train_series}.csv` + previous label file. Prints gold positive rates, report lengths,
duplicates, rough language mix, series layout, slot fill rates. Output saved to `exp/eda_csv.txt`. No parameters.

## `scripts/llm_labels.py` — report → 12 finding states (Claude on Bedrock)
| Arg | Default | Meaning |
|---|---|---|
| `--model` | (required) | Bedrock model id, e.g. `us.anthropic.claude-sonnet-5-5` |
| `--effort` | `low` | reasoning effort (classification task → low is enough) |
| `--subset` | `gold` | `gold` (58), `nongold` (4,349) or `all` |
| `--limit` | 0 | first N reports only (smoke tests) |
| `--workers` | 16 | parallel API calls (we used 24) |
| `--out` | (required) | JSONL output; **resume-safe** (skips studies already done) |

Key internals: `SYSTEM` prompt = expert label definitions + rules; JSON-schema-constrained output
(`state` ∈ {negated, not_mentioned, below_threshold, equivocal, positive} + `evidence` per finding);
system prompt cached; retries with exponential back-off on 429/5xx.

## `scripts/score_labels.py` — label quality vs gold-58
Args: one or more JSONL files. Prints macro/per-label AUC, accuracy at "positive", and the
`P(gold positive | state)` table that justifies the state→value mapping. `DEFAULT_MAP` = 0.02/0.10/0.30/0.60/0.95.

## `scripts/build_labels.py` — final soft labels
Blends Sonnet (w=0.7) + Haiku (0.3), imputes Synovitis when silent: `0.15 + 0.5 × Effusion`. Prints the variant
comparison table, writes `data/labels_v1.csv` (12 columns + `is_gold`; gold rows keep expert 0/1).

## `scripts/download.sh` — competition zip → `/data/raw/comp.zip` (GPU box)
Reads Kaggle credentials from SSM Parameter Store `/rsna-knee/kaggle-cred` (bearer token or kaggle.json),
gets the signed URL, `aria2c -c -x16 -s32` (16 connections, resumable). `AWS_REGION`,
`KAGGLE_SSM_PARAMETER`, `KAGGLE_COMPETITION`, `OUT_DIR`, `OUT_FILE`, and `CONNECTIONS` override the defaults.
Run via `systemd-run` so it survives session disconnects.

## `scripts/preprocess.py` — DICOM → canonical uint8 volumes
| Arg | Default | Meaning |
|---|---|---|
| `--zip` | "" | read DICOMs directly from the competition zip (what we used) |
| `--series-root` | "" | alternative: folder of extracted `study/series/*.dcm` |
| `--series-csv` | (required) | `train_series.csv` / `test_series.csv` |
| `--out` | (required) | one `.npz` per study; metadata to `<out>_meta.jsonl` |
| `--workers` | all cores | process pool size (we used 46) |
| `--limit` | 0 | first N studies (smoke test) |
Constants: `SIZE=256`, `DEPTH=32`, `SLOTS` mapping (plane, fluid) → slot name.
Functions: `series_volume` (sort by position along normal, canonical in-plane orientation, metadata),
`laterality` (DICOM tag → series description → position fallback), `to_uint8` (pad square, area resize,
p0.5–p99.5 clip), `resample_depth` (nearest-index to 32). Left knees mirrored (sag: slice order; cor/ax: width).

## `scripts/pack.py` — all studies → one memory-mapped array
`/data/pack/vols.npy` uint8 (N, 5, 32, 256, 256) = 46 GB + `index.csv` (row, uid, mask per slot).
`--fake N` writes random volumes for smoke tests.

## `scripts/train.py` — model + training (one fold per process)
| Arg | Default | Meaning |
|---|---|---|
| `--backbone` | `tf_efficientnetv2_b0.in1k` | any timm model name |
| `--head` | `pool` | `pool` (max+mean per series, used in all good runs), `attn` (original, collapses), `attn2` |
| `--fold` / `--folds` | 0 / 4 | which fold to hold out / number of folds |
| `--epochs` | 10 | |
| `--bs` | 6 | studies per step |
| `--n-slices` | 16 | slices sampled per series (of 32) |
| `--crop` | 224 | training/eval crop size from 256 |
| `--lr` / `--head-lr` | 3e-4 / 1e-3 | AdamW LRs for backbone / new layers |
| `--wd` | 0.02 | weight decay |
| `--slots` | all 5 | subset of series to use |
| `--seed` | 0 | |
| `--max-train` / `--max-val` | 0 | subsample for quick diagnostics |
| `--eval-train` | off | also report AUC on 200 training studies (overfit test) |
| `--bench` | off | synthetic throughput benchmark |
| `--smoke` | off | 5 train / 5 val / 8 gold studies, 1 epoch |
Pieces: `KneeNet` (timm backbone → Linear+LayerNorm to 256 → head), `GpuBatcher` (2.5D slice stacking, GPU
augmentation, series dropout), `make_folds` (GroupKFold by report text), `macro_auc`, `predict`.
Outputs per fold: `args.json`, `log.jsonl` (per-epoch train/val loss, val AUC, gold AUC, per-label AUCs, `pred_std`),
`best.pt` (best val epoch), `last.pt`, `oof_val.npy`, `pred_gold.npy`, `val_uids.csv`, `gold_uids.csv`.

## `scripts/evaluate.py` — run/ensemble evaluation
`python evaluate.py RUN_DIR [RUN_DIR ...] --out results.md` → OOF AUC vs LLM labels, gold-58 per fold and fold
ensemble, per-label table; with several runs also the rank-mean ensemble.

## `scripts/infer.py` — test-time inference (Kaggle)
`--series-root --series-csv --test-csv --ckpts <glob...> --out --workers`. Same preprocessing code path as training
(checked bit-identical), averages all checkpoints, bf16 on Ampere+/fp16 on T4, writes submission every 25 studies,
0.5 fallback for failed studies.

## `kaggle/kernel/infer_kernel.py` + `kernel-metadata.json`
Offline Kaggle notebook (internet off, GPU on) that finds the competition data + our model dataset and calls
`infer.py`. Dataset: `mananjain39/rsna-knee-claude-models` (`scripts/` + `models/<run>/fold*/{best.pt,args.json}`).

## `scripts/laterality.py` — self-supervised left/right check (diagnostic)
Trains ResNet18 canonical-vs-mirrored on tagged studies; reports held-out accuracy by knee side and how many
untagged studies look mirrored; writes a corrected pack copy to `/data/pack_lat` (not used for training).

## `scripts/ssm.sh` — run a command on the GPU box
`SSM_INSTANCE_ID=i-... scripts/ssm.sh "<bash command>" [timeout_s]` via AWS SSM (no SSH/open ports).
`AWS_REGION`, `SSM_REMOTE_USER`, and `SSM_POLL_INTERVAL` are optional overrides. If credentials are expired,
the helper uses `ada` with the required `ADA_ACCOUNT` and `ADA_ROLE` settings (`ADA_PROVIDER` is optional).
Remote failures return a non-zero exit status.
