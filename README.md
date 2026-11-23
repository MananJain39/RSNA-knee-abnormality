# RSNA Knee MRI Abnormality Detection — rebuild

The latest project from `Downloads/knee_project/knee` is now integrated at this
repository root. Use `scripts/train.py`, `scripts/preprocess.py`,
`scripts/evaluate.py`, and `scripts/infer.py` for the current CNN pipeline.
The earlier DINOv2 pipeline remains in `src/` and the `*_classical.py` scripts
for reference and compatibility; its configuration and cached volumes are separate.
See [merge notes](docs/MERGE_NOTES.md) for setup and path details.

Predict 12 knee abnormalities from MRI. Only 58 studies have expert labels; the rest have free-text reports;
the test set has images only. Metric: macro ROC-AUC.

**Start here:** [`docs/LEARNING_JOURNAL.md`](docs/LEARNING_JOURNAL.md) — step-by-step explanation, results, lessons.
Code reference: [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md).

## Headline results
| Stage | Metric | Value |
|---|---|---|
| LLM labels (exp 01) | macro AUC vs expert labels (gold-58) | 0.876 |
| Image model B1 (ResNet34d, 4 folds) | gold-58, fold ensemble | **0.905** |
| | out-of-fold vs LLM labels (4,349 studies) | 0.886 |
| Ensemble B1+B2+B4 (12 models) | OOF / gold-58 | 0.893 / 0.908 |
| Kaggle late submission #1 (B1) | public / private LB | **0.917 / 0.905** |
| Kaggle late submission #2 (12-model ensemble) | public / private LB | scoring |

Leaderboard context (6,425 teams): median public 0.942, winner 0.966 → we are ~66th percentile. Gap analysis and
next steps: `docs/LEARNING_JOURNAL.md` §12.4b. Teaching outline: `docs/TEAM_SESSION_OUTLINE.md`.

## Layout
```
data/raw/            competition CSVs (from git history of the previous attempt)
data/labels_v1.csv   final soft labels
scripts/             all code (see docs/CODE_GUIDE.md)
exp/01_labels/       label extraction: raw LLM outputs, bake-off, README, results
exp/diagnostics/     A1 collapse + D/E debugging runs, backbone benchmark, laterality check
exp/B1_resnet34d_pool/  first working model: args, per-epoch logs, README, results
kaggle/              Kaggle inference kernel + dataset metadata
docs/                learning journal, code guide
```
Heavy data (265 GB zip, 46 GB packed volumes, checkpoints) lives only on the GPU box (`/data`) and S3,
never on the laptop.
